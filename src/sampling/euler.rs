//! Euler sampler for diffusion models.

use super::{
    DiffusionSchedule, NoiseSchedule, ProgressCallback,
    cfg::{predict_noise_cfg_batched, predict_noise_cfg_controlled},
};
use crate::{
    controlnet::ControlNetUnit,
    types::Backend,
    unet::{DenoisingUnet, Unet, UnetConditioning, UnetConfig, UnetTimingStats},
};
use burn::prelude::*;
use std::{hint::black_box, time::Instant};

/// Force GPU synchronization by reading a single element back to CPU.
/// This ensures all queued GPU operations complete before returning.
fn sync_gpu<const D: usize>(tensor: &Tensor<Backend, D>) {
    let flat = tensor.clone().flatten::<1>(0, D - 1);
    let data = flat.slice(0..1).into_data();
    black_box(data);
}

/// Euler sampler (first-order ODE solver).
///
/// The simplest sampler: takes one derivative step per sigma transition.
/// Fast but may require more steps for quality.
///
/// # Algorithm
///
/// For each step from sigma_i to sigma_{i+1}:
/// 1. Predict noise: epsilon = UNet(x, sigma, conditioning)
/// 2. Compute "predicted x0": x0_pred = x - sigma * epsilon
/// 3. Compute derivative: d = (x - x0_pred) / sigma = epsilon
/// 4. Step: x = x + (sigma_{i+1} - sigma_i) * d
#[derive(Debug)]
pub struct EulerSampler {
    /// Noise scheduler for scaling.
    scheduler: NoiseSchedule,
}

impl EulerSampler {
    /// Create a new Euler sampler with the given scheduler.
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
    /// * `latent` - Initial noisy latent `[B, 4, H, W]` (should be pure noise scaled by sigma_max)
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

        // Pre-batch conditioning once before the loop (saves ~40 clones per run)
        let cond_batched = U::Cond::batch_cfg(positive, negative);

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

            // Euler step: x = x + (sigma_next - sigma) * noise_pred
            let delta_sigma = sigma_next - sigma;
            latent = latent + noise_pred * delta_sigma;
        }

        latent
    }

    /// Sample with ControlNet guidance using a pre-computed schedule.
    ///
    /// Like [`Self::sample_with_schedule`] but runs ControlNets at each step to
    /// steer generation toward the conditioning images.
    ///
    /// # Arguments
    /// * `unet` - The UNet denoiser model
    /// * `controlnets` - ControlNet units with hint images and settings
    /// * `latent` - Initial noisy latent `[B, 4, H, W]`
    /// * `positive` - Positive conditioning
    /// * `negative` - Negative conditioning (for CFG)
    /// * `schedule` - Pre-computed diffusion schedule with sigmas and timesteps
    /// * `cfg_scale` - Classifier-free guidance scale
    ///
    /// # Returns
    /// Denoised latent `[B, 4, H, W]`
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

            // Euler step: x = x + (sigma_next - sigma) * noise_pred
            let delta_sigma = sigma_next - sigma;
            latent = latent + noise_pred * delta_sigma;
        }

        latent
    }

    /// Sample from the diffusion model.
    ///
    /// # Arguments
    /// * `unet` - The UNet denoiser model
    /// * `latent` - Initial noisy latent `[B, 4, H, W]` (should be pure noise scaled by sigma_max)
    /// * `positive` - Positive conditioning
    /// * `negative` - Negative conditioning (for CFG)
    /// * `sigmas` - Sigma schedule from high to low (e.g., from `scheduler.sigmas_for_steps`)
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

            // Euler step: x = x + (sigma_next - sigma) * noise_pred
            let delta_sigma = sigma_next - sigma;
            latent = latent + noise_pred * delta_sigma;
        }

        latent
    }

    /// Sample with detailed timing instrumentation.
    ///
    /// Same as `sample_with_schedule` but prints per-step timing breakdown.
    /// GPU sync is forced at measurement points to get accurate timings.
    pub fn sample_with_schedule_timed<U: DenoisingUnet>(
        &self,
        unet: &U,
        mut latent: Tensor<Backend, 4>,
        positive: &U::Cond,
        negative: &U::Cond,
        schedule: &DiffusionSchedule,
        cfg_scale: f32,
    ) -> Tensor<Backend, 4> {
        let sigmas = &schedule.sigmas;
        let timesteps = &schedule.timesteps;

        // Accumulators for timing stats
        let mut total_scale_time = std::time::Duration::ZERO;
        let mut total_batch_time = std::time::Duration::ZERO;
        let mut total_cfg_time = std::time::Duration::ZERO;
        let mut total_step_time = std::time::Duration::ZERO;
        let mut unet_stats = UnetTimingStats::default();

        // Pre-batch conditioning once before the loop
        let cond_batched = U::Cond::batch_cfg(positive, negative);

        let num_steps = sigmas.len() - 1;
        println!("  [Timing] Starting {num_steps} steps with sync points");

        for i in 0..num_steps {
            let sigma = sigmas[i];
            let sigma_next = sigmas[i + 1];

            if sigma < 1e-6 {
                continue;
            }

            let timestep = timesteps[i];
            let device = latent.device();
            let [batch, _, _, _] = latent.shape().dims();

            // 1. Scale input
            let t0 = Instant::now();
            let scale = self.scheduler.scale_model_input(sigma);
            let latent_scaled = latent.clone() * scale;
            sync_gpu(&latent_scaled);
            total_scale_time += t0.elapsed();

            // 2. Batch latent for CFG
            let t0 = Instant::now();
            let latent_batched = Tensor::cat(vec![latent_scaled.clone(), latent_scaled], 0);
            let timestep_tensor: Tensor<Backend, 1> = Tensor::full([batch * 2], timestep, &device);
            sync_gpu(&latent_batched);
            total_batch_time += t0.elapsed();

            // 3. UNet forward pass (with detailed timing)
            let (noise_batched, step_unet_stats) =
                unet.forward_timed(latent_batched, timestep_tensor, &cond_batched);
            unet_stats.accumulate(&step_unet_stats);

            // 4. CFG combination
            let t0 = Instant::now();
            let [noise_pos, noise_neg]: [Tensor<Backend, 4>; 2] = noise_batched
                .chunk(2, 0)
                .try_into()
                .expect("chunk returned wrong count");
            let noise_pred = noise_neg * (1.0 - cfg_scale) + noise_pos * cfg_scale;
            sync_gpu(&noise_pred);
            total_cfg_time += t0.elapsed();

            // 5. Euler step
            let t0 = Instant::now();
            let delta_sigma = sigma_next - sigma;
            latent = latent + noise_pred * delta_sigma;
            sync_gpu(&latent);
            total_step_time += t0.elapsed();
        }

        let total_unet_time = unet_stats.total();
        let total_other = total_scale_time + total_batch_time + total_cfg_time + total_step_time;
        let total = total_unet_time + total_other;

        println!("  [Timing] Sampling breakdown for {num_steps} steps:");
        println!(
            "    Scale input:  {:>8.2?} ({:>5.1}%)",
            total_scale_time,
            100.0 * total_scale_time.as_secs_f64() / total.as_secs_f64()
        );
        println!(
            "    Batch latent: {:>8.2?} ({:>5.1}%)",
            total_batch_time,
            100.0 * total_batch_time.as_secs_f64() / total.as_secs_f64()
        );
        println!(
            "    UNet forward: {:>8.2?} ({:>5.1}%)",
            total_unet_time,
            100.0 * total_unet_time.as_secs_f64() / total.as_secs_f64()
        );
        println!(
            "    CFG combine:  {:>8.2?} ({:>5.1}%)",
            total_cfg_time,
            100.0 * total_cfg_time.as_secs_f64() / total.as_secs_f64()
        );
        println!(
            "    Euler step:   {:>8.2?} ({:>5.1}%)",
            total_step_time,
            100.0 * total_step_time.as_secs_f64() / total.as_secs_f64()
        );
        println!();
        println!("  [Timing] UNet internal breakdown (total across {num_steps} steps):");
        unet_stats.print();

        latent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        clip::{Sd15ClipTextEncoder, Sd15Conditioning},
        model_loader::SafeTensorsFile,
        unet::Sd15Unet2D,
        vae::Sd15VaeDecoder,
    };
    use burn::tensor::Distribution;
    use std::path::Path;

    #[test]
    fn sigma_to_timestep_conversion() {
        let scheduler = NoiseSchedule::linear(1000, 0.00085, 0.012);
        let sampler = EulerSampler::new(scheduler);

        // Check that conversion is approximately correct
        let t = sampler
            .scheduler
            .sigma_to_timestep(sampler.scheduler.timestep_to_sigma(500));
        assert!((t - 500.0).abs() < 10.0, "Should be close to 500, got {t}");

        let t = sampler
            .scheduler
            .sigma_to_timestep(sampler.scheduler.timestep_to_sigma(999));
        assert!((t - 999.0).abs() < 10.0, "Should be close to 999, got {t}");

        let t = sampler
            .scheduler
            .sigma_to_timestep(sampler.scheduler.timestep_to_sigma(0));
        assert!(t < 10.0, "Should be close to 0, got {t}");
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

        // Timesteps should correspond to sigmas
        for (i, &timestep) in schedule.timesteps.iter().enumerate() {
            let expected = scheduler.sigma_to_timestep(schedule.sigmas[i]);
            assert!(
                (timestep - expected).abs() < 1e-6,
                "Timestep {i} mismatch: {timestep} vs {expected}"
            );
        }
    }

    #[test]
    #[ignore = "slow test, run with --ignored"]
    fn full_sampling_pipeline() {
        const MODEL_PATH: &str = "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors";

        let path = Path::new(MODEL_PATH);
        if !path.exists() {
            println!("Model not found at {MODEL_PATH}, skipping");
            return;
        }

        let file = SafeTensorsFile::open(path).expect("Failed to open checkpoint");
        let tensors = file.tensors().expect("Failed to parse safetensors");
        let device = Default::default();

        println!("Loading models...");
        let scheduler = NoiseSchedule::load(&tensors, &device).expect("Failed to load scheduler");
        let unet = Sd15Unet2D::load(&tensors, &device).expect("Failed to load UNet");
        let _clip = Sd15ClipTextEncoder::load(&tensors, &device).expect("Failed to load CLIP");
        let _vae = Sd15VaeDecoder::load(&tensors, Some("first_stage_model"), &device)
            .expect("Failed to load VAE");

        println!("Models loaded successfully");

        // Create sampler
        let sampler = EulerSampler::new(scheduler);

        // Generate schedule for 4 steps (very fast test)
        let schedule = sampler.scheduler().schedule_for_steps(4);
        println!("Sigmas for 4 steps: {:?}", schedule.sigmas);

        // Create test inputs (random noise)
        let latent: Tensor<Backend, 4> =
            Tensor::random([1, 4, 64, 64], Distribution::Normal(0.0, 1.0), &device);
        let positive = Sd15Conditioning::new(Tensor::zeros([1, 77, 768], &device));
        let negative = Sd15Conditioning::new(Tensor::zeros([1, 77, 768], &device));

        println!("Running 4-step sampling (this may take a while)...");
        let result =
            sampler.sample_with_schedule(&unet, latent, &positive, &negative, &schedule, 7.5, None);

        let shape = result.shape().dims::<4>();
        println!("Output shape: {:?}", shape);
        assert_eq!(
            shape,
            [1, 4, 64, 64],
            "Expected output shape [1, 4, 64, 64]"
        );
    }
}
