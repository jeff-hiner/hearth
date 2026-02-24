//! Euler Ancestral sampler for diffusion models.

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

/// Euler Ancestral sampler (first-order stochastic solver).
///
/// Like [`super::EulerSampler`] but injects noise at each step, producing
/// more varied outputs. The ancestral decomposition splits sigma_next into
/// a deterministic part (sigma_down) and a stochastic part (sigma_up).
///
/// # Algorithm
///
/// For each step from sigma to sigma_next:
/// 1. Predict noise via CFG
/// 2. Compute ancestral decomposition:
///    - `sigma_up = min(sigma_next, sqrt(sigma_next² * (sigma² - sigma_next²) / sigma²))`
///    - `sigma_down = sqrt(sigma_next² - sigma_up²)`
/// 3. Euler step to sigma_down: `x += (sigma_down - sigma) * noise_pred`
/// 4. Add noise: `x += randn() * sigma_up` (skip on final step)
#[derive(Debug)]
pub struct EulerASampler {
    /// Noise scheduler for scaling.
    scheduler: NoiseSchedule,
}

impl EulerASampler {
    /// Create a new Euler Ancestral sampler with the given scheduler.
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

        // Pre-batch conditioning once before the loop
        let cond_batched = U::Cond::batch_cfg(positive, negative);

        // Pre-create timestep tensors (avoids one Tensor::full kernel per step)
        let device = latent.device();
        let [batch, _, _, _] = latent.shape().dims();
        let timestep_tensors: Vec<Tensor<Backend, 1>> = timesteps
            .iter()
            .map(|&t| Tensor::full([batch * 2], t, &device))
            .collect();

        for i in 0..sigmas.len() - 1 {
            if let Some(cb) = progress {
                if !cb(i + 1, sigmas.len() - 1) {
                    break;
                }
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

            // Ancestral decomposition
            let (sigma_up, sigma_down) = ancestral_step(sigma, sigma_next);

            // Euler step to sigma_down
            let delta = sigma_down - sigma;
            latent = latent + noise_pred * delta;

            // Add ancestral noise (skip on final step where sigma_up == 0)
            if sigma_up > 1e-6 {
                let device = latent.device();
                let shape = latent.shape();
                let noise: Tensor<Backend, 4> =
                    Tensor::random(shape.dims, Distribution::Normal(0.0, 1.0), &device);
                latent = latent + noise * sigma_up;
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

            // Ancestral decomposition
            let (sigma_up, sigma_down) = ancestral_step(sigma, sigma_next);

            // Euler step to sigma_down
            let delta = sigma_down - sigma;
            latent = latent + noise_pred * delta;

            // Add ancestral noise (skip on final step where sigma_up == 0)
            if sigma_up > 1e-6 {
                let device = latent.device();
                let shape = latent.shape();
                let noise: Tensor<Backend, 4> =
                    Tensor::random(shape.dims, Distribution::Normal(0.0, 1.0), &device);
                latent = latent + noise * sigma_up;
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

            // Ancestral decomposition
            let (sigma_up, sigma_down) = ancestral_step(sigma, sigma_next);

            // Euler step to sigma_down
            let delta = sigma_down - sigma;
            latent = latent + noise_pred * delta;

            // Add ancestral noise (skip on final step where sigma_up == 0)
            if sigma_up > 1e-6 {
                let device = latent.device();
                let shape = latent.shape();
                let noise: Tensor<Backend, 4> =
                    Tensor::random(shape.dims, Distribution::Normal(0.0, 1.0), &device);
                latent = latent + noise * sigma_up;
            }
        }

        latent
    }
}

/// Compute the ancestral step decomposition.
///
/// Returns `(sigma_up, sigma_down)` where:
/// - `sigma_up` = stochastic noise magnitude
/// - `sigma_down` = deterministic target sigma
///
/// The decomposition satisfies `sigma_down² + sigma_up² = sigma_next²`.
fn ancestral_step(sigma: f32, sigma_next: f32) -> (f32, f32) {
    // sigma_up = min(sigma_next, sqrt(sigma_next² * (sigma² - sigma_next²) / sigma²))
    let sigma_up = (sigma_next * sigma_next * (sigma * sigma - sigma_next * sigma_next)
        / (sigma * sigma))
        .sqrt()
        .min(sigma_next);

    // sigma_down = sqrt(sigma_next² - sigma_up²)
    let sigma_down = (sigma_next * sigma_next - sigma_up * sigma_up).sqrt();

    (sigma_up, sigma_down)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ancestral_step_decomposition() {
        let sigma = 10.0;
        let sigma_next = 8.0;
        let (sigma_up, sigma_down) = ancestral_step(sigma, sigma_next);

        // sigma_down² + sigma_up² should equal sigma_next²
        let reconstructed = (sigma_down * sigma_down + sigma_up * sigma_up).sqrt();
        assert!(
            (reconstructed - sigma_next).abs() < 1e-5,
            "Reconstruction error: {reconstructed} vs {sigma_next}"
        );
    }

    #[test]
    fn ancestral_step_final() {
        // When sigma_next is 0, both should be 0
        let (sigma_up, sigma_down) = ancestral_step(1.0, 0.0);
        assert!(sigma_up.abs() < 1e-6);
        assert!(sigma_down.abs() < 1e-6);
    }

    #[test]
    fn sampler_creation() {
        let scheduler = NoiseSchedule::linear(1000, 0.00085, 0.012);
        let _sampler = EulerASampler::new(scheduler);
    }
}
