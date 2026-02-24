//! DPM++ 2M sampler for diffusion models.

use super::{
    DiffusionSchedule, NoiseSchedule, ProgressCallback,
    cfg::{predict_noise_cfg_batched, predict_noise_cfg_controlled},
};
use crate::{
    controlnet::ControlNetUnit,
    types::Backend,
    unet::{DenoisingUnet, Unet, UnetConditioning, UnetConfig},
};
use burn::prelude::*;

/// DPM++ 2M sampler (second-order multistep solver).
///
/// A more accurate sampler than Euler that uses the previous denoising
/// prediction to improve the current step. Produces better results with
/// fewer steps (typically 15-25 steps vs 50+ for Euler).
///
/// Based on DPM-Solver++ by Lu et al. (2022).
///
/// # Algorithm
///
/// The solver works in the "denoised" (predicted x0) space:
/// 1. First step uses first-order update
/// 2. Subsequent steps use second-order update with previous prediction
#[derive(Debug)]
pub struct DpmPp2mSampler {
    /// Noise scheduler for scaling.
    scheduler: NoiseSchedule,
}

impl DpmPp2mSampler {
    /// Create a new DPM++ 2M sampler with the given scheduler.
    pub fn new(scheduler: NoiseSchedule) -> Self {
        Self { scheduler }
    }

    /// Get a reference to the scheduler.
    pub fn scheduler(&self) -> &NoiseSchedule {
        &self.scheduler
    }

    /// Sample from the diffusion model using a pre-computed schedule.
    ///
    /// This is the optimized version that avoids per-step timestep lookups.
    ///
    /// # Arguments
    /// * `unet` - The UNet denoiser model
    /// * `latent` - Initial noisy latent `[B, 4, H, W]`
    /// * `positive` - Positive conditioning `[B, 77, context_dim]`
    /// * `negative` - Negative conditioning `[B, 77, context_dim]` (for CFG)
    /// * `schedule` - Pre-computed diffusion schedule with sigmas and timesteps
    /// * `cfg_scale` - Classifier-free guidance scale (typically 7.0-8.0)
    ///
    /// # Returns
    /// Denoised latent `[B, 4, H, W]`
    #[expect(
        clippy::too_many_arguments,
        reason = "progress callback adds one extra param"
    )]
    pub fn sample_with_schedule<U: DenoisingUnet>(
        &self,
        unet: &U,
        mut latent: Tensor<Backend, 4>,
        positive: &U::Cond,
        negative: &U::Cond,
        schedule: &DiffusionSchedule,
        cfg_scale: f32,
        progress: ProgressCallback<'_>,
    ) -> Tensor<Backend, 4> {
        let sigmas = &schedule.sigmas;
        let timesteps = &schedule.timesteps;

        // Pre-batch conditioning once before the loop (saves ~40 clones per run)
        let cond_batched = U::Cond::batch_cfg(positive, negative);

        // Pre-create timestep tensors (avoids one Tensor::full kernel per step)
        let device = latent.device();
        let [batch, _, _, _] = latent.shape().dims();
        let timestep_tensors: Vec<Tensor<Backend, 1>> = timesteps
            .iter()
            .map(|&t| Tensor::full([batch * 2], t, &device))
            .collect();

        // Track previous denoised prediction for second-order update
        let mut old_denoised: Option<Tensor<Backend, 4>> = None;

        for i in 0..sigmas.len() - 1 {
            if let Some(cb) = progress {
                if !cb(i + 1, sigmas.len() - 1) {
                    break;
                }
            }
            let sigma = sigmas[i];
            let sigma_next = sigmas[i + 1];

            // Skip if sigma is effectively 0
            if sigma < 1e-6 {
                continue;
            }

            let scale = self.scheduler.scale_model_input(sigma);

            let noise_pred = predict_noise_cfg_batched(
                unet,
                scale,
                &latent,
                &timestep_tensors[i],
                &cond_batched,
                cfg_scale,
            );

            // Convert noise prediction to denoised (x0) prediction
            // x0 = x - sigma * noise
            let denoised = latent.clone() - noise_pred * sigma;

            if sigma_next < 1e-6 {
                // Final step: just return denoised
                latent = denoised;
            } else {
                latent = dpm_pp_2m_step(&mut old_denoised, latent, denoised, sigmas, i, sigma_next);
            }
        }

        latent
    }

    /// Sample with ControlNet guidance using a pre-computed schedule.
    ///
    /// Like [`Self::sample_with_schedule`] but runs ControlNets at each step.
    #[expect(clippy::too_many_arguments, reason = "ControlNet needs extra params")]
    pub fn sample_with_schedule_controlled<C: UnetConfig, Cond: UnetConditioning>(
        &self,
        unet: &Unet<C>,
        controlnets: &[ControlNetUnit<'_, C>],
        mut latent: Tensor<Backend, 4>,
        positive: &Cond,
        negative: &Cond,
        schedule: &DiffusionSchedule,
        cfg_scale: f32,
        progress: ProgressCallback<'_>,
    ) -> Tensor<Backend, 4> {
        let sigmas = &schedule.sigmas;
        let timesteps = &schedule.timesteps;
        let num_steps = sigmas.len() - 1;

        let mut old_denoised: Option<Tensor<Backend, 4>> = None;

        for i in 0..num_steps {
            if let Some(cb) = progress {
                if !cb(i + 1, num_steps) {
                    break;
                }
            }
            let sigma = sigmas[i];
            let sigma_next = sigmas[i + 1];

            if sigma < 1e-6 {
                continue;
            }

            let timestep = timesteps[i];
            let scale = self.scheduler.scale_model_input(sigma);

            let noise_pred = predict_noise_cfg_controlled(
                unet,
                controlnets,
                scale,
                &latent,
                timestep,
                positive,
                negative,
                cfg_scale,
                i,
                num_steps,
            );

            // Convert noise prediction to denoised (x0) prediction
            let denoised = latent.clone() - noise_pred * sigma;

            if sigma_next < 1e-6 {
                latent = denoised;
            } else {
                latent = dpm_pp_2m_step(&mut old_denoised, latent, denoised, sigmas, i, sigma_next);
            }
        }

        latent
    }

    /// Sample from the diffusion model.
    ///
    /// # Arguments
    /// * `unet` - The UNet denoiser model
    /// * `latent` - Initial noisy latent `[B, 4, H, W]`
    /// * `positive` - Positive conditioning
    /// * `negative` - Negative conditioning (for CFG)
    /// * `sigmas` - Sigma schedule from high to low
    /// * `cfg_scale` - Classifier-free guidance scale (typically 7.0-8.0)
    ///
    /// # Returns
    /// Denoised latent `[B, 4, H, W]`
    pub fn sample<U: DenoisingUnet>(
        &self,
        unet: &U,
        mut latent: Tensor<Backend, 4>,
        positive: &U::Cond,
        negative: &U::Cond,
        sigmas: &[f32],
        cfg_scale: f32,
    ) -> Tensor<Backend, 4> {
        // Pre-batch conditioning once before the loop (saves ~40 clones per run)
        let cond_batched = U::Cond::batch_cfg(positive, negative);

        let device = latent.device();
        let [batch, _, _, _] = latent.shape().dims();

        // Track previous denoised prediction for second-order update
        let mut old_denoised: Option<Tensor<Backend, 4>> = None;

        for i in 0..sigmas.len() - 1 {
            let sigma = sigmas[i];
            let sigma_next = sigmas[i + 1];

            // Skip if sigma is effectively 0
            if sigma < 1e-6 {
                continue;
            }

            let timestep = self.scheduler.sigma_to_timestep(sigma);
            let scale = self.scheduler.scale_model_input(sigma);
            let timestep_tensor: Tensor<Backend, 1> = Tensor::full([batch * 2], timestep, &device);

            let noise_pred = predict_noise_cfg_batched(
                unet,
                scale,
                &latent,
                &timestep_tensor,
                &cond_batched,
                cfg_scale,
            );

            // Convert noise prediction to denoised (x0) prediction
            // x0 = x - sigma * noise
            let denoised = latent.clone() - noise_pred * sigma;

            if sigma_next < 1e-6 {
                // Final step: just return denoised
                latent = denoised;
            } else {
                latent = dpm_pp_2m_step(&mut old_denoised, latent, denoised, sigmas, i, sigma_next);
            }
        }

        latent
    }
}

/// Perform a single DPM++ 2M step (shared between `sample` and `sample_with_schedule`).
///
/// Applies first-order or second-order update depending on whether a previous
/// denoised prediction is available.
fn dpm_pp_2m_step(
    old_denoised: &mut Option<Tensor<Backend, 4>>,
    latent: Tensor<Backend, 4>,
    denoised: Tensor<Backend, 4>,
    sigmas: &[f32],
    i: usize,
    sigma_next: f32,
) -> Tensor<Backend, 4> {
    let sigma = sigmas[i];

    // Compute log-SNR timesteps
    let t = sigma_to_t(sigma);
    let t_next = sigma_to_t(sigma_next);
    let h = t_next - t;

    let result = if let Some(old_d) = old_denoised {
        // Second-order: use previous denoised prediction
        let t_prev = sigma_to_t(sigmas[i.saturating_sub(1).max(0)]);
        let h_last = t - t_prev;

        // Avoid division by zero
        let r = if h.abs() > 1e-6 { h_last / h } else { 1.0 };

        // Second-order correction
        // denoised_d = (1 + 1/(2r)) * denoised - 1/(2r) * old_denoised
        let coeff1 = 1.0 + 0.5 / r;
        let coeff2 = 0.5 / r;
        let denoised_d = denoised.clone() * coeff1 - old_d.clone() * coeff2;

        let ratio = sigma_next / sigma;
        let exp_m1 = (-h).exp() - 1.0;
        latent * ratio - denoised_d * exp_m1
    } else {
        // First step: first-order Euler in denoised space
        // x = (sigma_next / sigma) * x - expm1(-h) * denoised
        let ratio = sigma_next / sigma;
        let exp_m1 = (-h).exp() - 1.0; // expm1(-h)
        latent * ratio - denoised.clone() * exp_m1
    };

    *old_denoised = Some(denoised);
    result
}

/// Convert sigma to continuous-time "t" (log-SNR based).
///
/// t = -log(sigma)
fn sigma_to_t(sigma: f32) -> f32 {
    -sigma.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_to_t_conversion() {
        // Higher sigma (more noise) should give lower t
        let t1 = sigma_to_t(1.0);
        let t2 = sigma_to_t(10.0);

        assert!(t2 < t1, "Higher sigma should give lower t");
        assert!((t1 - 0.0).abs() < 1e-6, "sigma=1 should give t=0");
    }

    #[test]
    fn sampler_creation() {
        let scheduler = NoiseSchedule::linear(1000, 0.00085, 0.012);
        let _sampler = DpmPp2mSampler::new(scheduler);
    }

    #[test]
    fn schedule_for_steps() {
        let scheduler = NoiseSchedule::linear(1000, 0.00085, 0.012);
        let schedule = scheduler.schedule_for_steps(20);

        assert_eq!(
            schedule.sigmas.len(),
            21,
            "Should have num_steps + 1 sigmas"
        );
        assert_eq!(
            schedule.timesteps.len(),
            20,
            "Should have num_steps timesteps"
        );
    }
}
