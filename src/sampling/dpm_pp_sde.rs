//! DPM++ SDE sampler for diffusion models.

use super::{
    DiffusionSchedule, NoiseSchedule, ProgressCallback,
    cfg::{predict_noise_cfg_batched, predict_noise_cfg_controlled},
};
use crate::{
    controlnet::ControlNetUnit,
    types::Backend,
    unet::{DenoisingUnet, Unet, UnetConditioning, UnetConfig},
};
use burn::{prelude::*, tensor::Distribution};

/// DPM++ SDE sampler (stochastic second-order solver).
///
/// Like [`super::DpmPp2mSampler`] but adds stochastic noise at each step via
/// the SDE formulation. Produces more varied outputs while retaining the
/// quality benefits of second-order solving.
///
/// Uses simple Gaussian noise rather than Brownian trees, which is correct
/// for fixed-step schedules.
///
/// Based on DPM-Solver++ (SDE variant) by Lu et al. (2022).
///
/// # Algorithm
///
/// For each step from sigma to sigma_next:
/// 1. Predict noise via CFG, convert to denoised: `x0 = x - sigma * noise`
/// 2. If sigma_next == 0: return denoised
/// 3. Compute log-space step: `t = -ln(sigma)`, `h = -ln(sigma_next) - t`
/// 4. First-order or second-order drift (same as DPM++ 2M)
/// 5. Add SDE noise: `x += randn() * sigma_next * sqrt(1 - exp(-2h))`
#[derive(Debug)]
pub struct DpmPpSdeSampler {
    /// Noise scheduler for scaling.
    scheduler: NoiseSchedule,
}

impl DpmPpSdeSampler {
    /// Create a new DPM++ SDE sampler with the given scheduler.
    pub fn new(scheduler: NoiseSchedule) -> Self {
        Self { scheduler }
    }

    /// Get a reference to the scheduler.
    pub fn scheduler(&self) -> &NoiseSchedule {
        &self.scheduler
    }

    /// Sample from the diffusion model using a pre-computed schedule.
    ///
    /// # Arguments
    /// * `unet` - The UNet denoiser model
    /// * `latent` - Initial noisy latent `[B, 4, H, W]`
    /// * `positive` - Positive conditioning
    /// * `negative` - Negative conditioning (for CFG)
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

        let cond_batched = U::Cond::batch_cfg(positive, negative);
        let mut old_denoised: Option<Tensor<Backend, 4>> = None;

        // Pre-create timestep tensors (avoids one Tensor::full kernel per step)
        let device = latent.device();
        let [batch, _, _, _] = latent.shape().dims();
        let timestep_tensors: Vec<Tensor<Backend, 1>> = timesteps
            .iter()
            .map(|&t| Tensor::full([batch * 2], t, &device))
            .collect();

        for i in 0..sigmas.len() - 1 {
            if let Some(cb) = progress
                && !cb(i + 1, sigmas.len() - 1)
            {
                break;
            }
            let sigma = sigmas[i];
            let sigma_next = sigmas[i + 1];

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
            let denoised = latent.clone() - noise_pred * sigma;

            if sigma_next < 1e-6 {
                // Final step: return denoised directly
                latent = denoised;
            } else {
                // Log-space timesteps
                let t = sigma_to_t(sigma);
                let t_next = sigma_to_t(sigma_next);
                let h = t_next - t;

                // Deterministic drift (same as DPM++ 2M)
                if let Some(old_d) = &old_denoised {
                    // Second-order update
                    let t_prev = sigma_to_t(sigmas[i.saturating_sub(1).max(0)]);
                    let h_last = t - t_prev;
                    let r = if h.abs() > 1e-6 { h_last / h } else { 1.0 };

                    let coeff1 = 1.0 + 0.5 / r;
                    let coeff2 = 0.5 / r;
                    let denoised_d = denoised.clone() * coeff1 - old_d.clone() * coeff2;

                    let ratio = sigma_next / sigma;
                    let exp_m1 = (-h).exp() - 1.0;
                    latent = latent * ratio - denoised_d * exp_m1;
                } else {
                    // First-order update
                    let ratio = sigma_next / sigma;
                    let exp_m1 = (-h).exp() - 1.0;
                    latent = latent * ratio - denoised.clone() * exp_m1;
                }

                // SDE noise injection: x += randn() * sigma_next * sqrt(1 - exp(-2h))
                let noise_scale = sigma_next * (1.0 - (-2.0 * h).exp()).sqrt();
                if noise_scale > 1e-6 {
                    let device = latent.device();
                    let shape = latent.shape();
                    let noise: Tensor<Backend, 4> =
                        Tensor::random(shape.dims, Distribution::Normal(0.0, 1.0), &device);
                    latent = latent + noise * noise_scale;
                }

                old_denoised = Some(denoised);
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
            if let Some(cb) = progress
                && !cb(i + 1, num_steps)
            {
                break;
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
                // Log-space timesteps
                let t = sigma_to_t(sigma);
                let t_next = sigma_to_t(sigma_next);
                let h = t_next - t;

                // Deterministic drift (same as DPM++ 2M)
                if let Some(old_d) = &old_denoised {
                    let t_prev = sigma_to_t(sigmas[i.saturating_sub(1).max(0)]);
                    let h_last = t - t_prev;
                    let r = if h.abs() > 1e-6 { h_last / h } else { 1.0 };

                    let coeff1 = 1.0 + 0.5 / r;
                    let coeff2 = 0.5 / r;
                    let denoised_d = denoised.clone() * coeff1 - old_d.clone() * coeff2;

                    let ratio = sigma_next / sigma;
                    let exp_m1 = (-h).exp() - 1.0;
                    latent = latent * ratio - denoised_d * exp_m1;
                } else {
                    let ratio = sigma_next / sigma;
                    let exp_m1 = (-h).exp() - 1.0;
                    latent = latent * ratio - denoised.clone() * exp_m1;
                }

                // SDE noise injection
                let noise_scale = sigma_next * (1.0 - (-2.0 * h).exp()).sqrt();
                if noise_scale > 1e-6 {
                    let device = latent.device();
                    let shape = latent.shape();
                    let noise: Tensor<Backend, 4> =
                        Tensor::random(shape.dims, Distribution::Normal(0.0, 1.0), &device);
                    latent = latent + noise * noise_scale;
                }

                old_denoised = Some(denoised);
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
        let cond_batched = U::Cond::batch_cfg(positive, negative);
        let mut old_denoised: Option<Tensor<Backend, 4>> = None;

        let device = latent.device();
        let [batch, _, _, _] = latent.shape().dims();

        for i in 0..sigmas.len() - 1 {
            let sigma = sigmas[i];
            let sigma_next = sigmas[i + 1];

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

            let denoised = latent.clone() - noise_pred * sigma;

            if sigma_next < 1e-6 {
                latent = denoised;
            } else {
                let t = sigma_to_t(sigma);
                let t_next = sigma_to_t(sigma_next);
                let h = t_next - t;

                if let Some(old_d) = &old_denoised {
                    let t_prev = sigma_to_t(sigmas[i.saturating_sub(1).max(0)]);
                    let h_last = t - t_prev;
                    let r = if h.abs() > 1e-6 { h_last / h } else { 1.0 };

                    let coeff1 = 1.0 + 0.5 / r;
                    let coeff2 = 0.5 / r;
                    let denoised_d = denoised.clone() * coeff1 - old_d.clone() * coeff2;

                    let ratio = sigma_next / sigma;
                    let exp_m1 = (-h).exp() - 1.0;
                    latent = latent * ratio - denoised_d * exp_m1;
                } else {
                    let ratio = sigma_next / sigma;
                    let exp_m1 = (-h).exp() - 1.0;
                    latent = latent * ratio - denoised.clone() * exp_m1;
                }

                let noise_scale = sigma_next * (1.0 - (-2.0 * h).exp()).sqrt();
                if noise_scale > 1e-6 {
                    let device = latent.device();
                    let shape = latent.shape();
                    let noise: Tensor<Backend, 4> =
                        Tensor::random(shape.dims, Distribution::Normal(0.0, 1.0), &device);
                    latent = latent + noise * noise_scale;
                }

                old_denoised = Some(denoised);
            }
        }

        latent
    }
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
    fn sampler_creation() {
        let scheduler = NoiseSchedule::linear(1000, 0.00085, 0.012);
        let _sampler = DpmPpSdeSampler::new(scheduler);
    }
}
