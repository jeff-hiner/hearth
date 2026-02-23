//! Noise schedulers for diffusion models.

use crate::{
    model_loader::{LoadError, load_tensor_1d},
    types::Backend,
};
use burn::prelude::*;

/// Pre-computed diffusion schedule for sampling.
///
/// Contains both sigmas and timesteps to avoid per-step binary search.
#[derive(Debug, Clone)]
pub struct DiffusionSchedule {
    /// Noise levels from high to low, length: `num_steps + 1`.
    /// Final sigma is 0 (fully denoised).
    pub sigmas: Vec<f32>,
    /// UNet timesteps corresponding to each sigma, length: `num_steps`.
    /// Pre-computed via binary search to avoid per-step lookup.
    pub timesteps: Vec<f32>,
}

/// DDPM noise scheduler loaded from checkpoint.
///
/// Holds the pre-computed noise schedule parameters:
/// - `alphas_cumprod`: cumulative product of (1 - beta_t)
/// - Derived values: sigmas, sqrt_alphas, etc.
///
/// SD 1.5 uses 1000 discrete timesteps with linear beta schedule.
#[derive(Debug, Clone)]
pub struct NoiseSchedule {
    /// Cumulative product of alphas, shape `[num_timesteps]`.
    /// Starts near 1.0 (clean), ends near 0.0 (pure noise).
    alphas_cumprod: Vec<f32>,
    /// Total number of training timesteps (typically 1000).
    num_train_timesteps: usize,
}

impl NoiseSchedule {
    /// Load scheduler parameters from SD checkpoint.
    ///
    /// Expects key: `alphas_cumprod` with shape `[num_timesteps]`.
    pub fn load(
        tensors: &safetensors::SafeTensors<'_>,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let alphas_cumprod_tensor = load_tensor_1d(tensors, "alphas_cumprod", device)?;
        let num_train_timesteps = alphas_cumprod_tensor.shape().dims::<1>()[0];

        // Convert to Vec<f32> for CPU-side scheduling math
        // Use convert::<f32>() to handle both f32 and f16 backends
        let alphas_cumprod: Vec<f32> = alphas_cumprod_tensor
            .into_data()
            .convert::<f32>()
            .to_vec()
            .expect("alphas_cumprod conversion failed");

        Ok(Self {
            alphas_cumprod,
            num_train_timesteps,
        })
    }

    /// Create scheduler with scaled-linear beta schedule.
    ///
    /// This matches ldm/sgm's "linear" schedule: linearly interpolate the **square
    /// roots** of `beta_start` and `beta_end`, then square. This is the schedule used
    /// by both SD 1.5 and SDXL.
    ///
    /// # Arguments
    /// * `num_timesteps` - Number of diffusion steps (typically 1000)
    /// * `beta_start` - Starting beta value (typically 0.00085)
    /// * `beta_end` - Ending beta value (typically 0.012)
    pub fn linear(num_timesteps: usize, beta_start: f64, beta_end: f64) -> Self {
        let mut alphas_cumprod = Vec::with_capacity(num_timesteps);
        let mut cumprod = 1.0f64;

        let sqrt_start = beta_start.sqrt();
        let sqrt_end = beta_end.sqrt();

        for i in 0..num_timesteps {
            // ldm "linear" schedule: linspace(sqrt(start), sqrt(end), n) ** 2
            let sqrt_beta =
                sqrt_start + (sqrt_end - sqrt_start) * (i as f64 / (num_timesteps - 1) as f64);
            let beta = sqrt_beta * sqrt_beta;
            let alpha = 1.0 - beta;
            cumprod *= alpha;
            alphas_cumprod.push(cumprod as f32);
        }

        Self {
            alphas_cumprod,
            num_train_timesteps: num_timesteps,
        }
    }

    /// Get number of training timesteps.
    pub fn num_train_timesteps(&self) -> usize {
        self.num_train_timesteps
    }

    /// Convert timestep to sigma (noise level).
    ///
    /// sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)
    pub fn timestep_to_sigma(&self, timestep: usize) -> f32 {
        let alpha = self.alphas_cumprod[timestep];
        ((1.0 - alpha) / alpha).sqrt()
    }

    /// Generate sigmas for a given number of inference steps.
    ///
    /// Uses linear spacing in timestep space, then converts to sigmas.
    /// Returns `num_steps + 1` sigmas (includes final sigma=0).
    pub fn sigmas_for_steps(&self, num_steps: usize) -> Vec<f32> {
        let mut sigmas = Vec::with_capacity(num_steps + 1);

        // Linear spacing from max timestep down to 0
        for i in 0..num_steps {
            let t = (self.num_train_timesteps - 1) as f64 * (1.0 - i as f64 / num_steps as f64);
            let timestep = t.round() as usize;
            sigmas.push(self.timestep_to_sigma(timestep));
        }

        // Final sigma is 0 (fully denoised)
        sigmas.push(0.0);

        sigmas
    }

    /// Generate sigmas using Karras schedule (recommended for quality).
    ///
    /// Karras et al. found that spacing sigmas exponentially (in log space)
    /// produces better results than linear timestep spacing.
    ///
    /// Returns `num_steps + 1` sigmas (includes final sigma=0).
    pub fn sigmas_karras(
        &self,
        num_steps: usize,
        sigma_min: f32,
        sigma_max: f32,
        rho: f32,
    ) -> Vec<f32> {
        let mut sigmas = Vec::with_capacity(num_steps + 1);

        let ramp = |i: usize| i as f32 / (num_steps - 1) as f32;

        let min_inv_rho = sigma_min.powf(1.0 / rho);
        let max_inv_rho = sigma_max.powf(1.0 / rho);

        for i in 0..num_steps {
            let sigma = (max_inv_rho + ramp(i) * (min_inv_rho - max_inv_rho)).powf(rho);
            sigmas.push(sigma);
        }

        sigmas.push(0.0);
        sigmas
    }

    /// Get sigma range (min, max) from the schedule.
    ///
    /// Useful for computing Karras sigmas.
    pub fn sigma_range(&self) -> (f32, f32) {
        let sigma_max = self.timestep_to_sigma(self.num_train_timesteps - 1);
        let sigma_min = self.timestep_to_sigma(0);
        (sigma_min, sigma_max)
    }

    /// Scale noise to proper variance for a given sigma.
    ///
    /// Returns the scaling factor: 1 / sqrt(sigma^2 + 1)
    pub fn scale_model_input(&self, sigma: f32) -> f32 {
        1.0 / (sigma * sigma + 1.0).sqrt()
    }

    /// Generate a complete diffusion schedule for sampling.
    ///
    /// Returns sigmas and pre-computed timesteps to avoid per-step binary search.
    pub fn schedule_for_steps(&self, num_steps: usize) -> DiffusionSchedule {
        let sigmas = self.sigmas_for_steps(num_steps);
        let timesteps = sigmas[..sigmas.len() - 1]
            .iter()
            .map(|&sigma| self.sigma_to_timestep(sigma))
            .collect();

        DiffusionSchedule { sigmas, timesteps }
    }

    /// Generate a Karras diffusion schedule for sampling.
    ///
    /// Returns sigmas and pre-computed timesteps using Karras spacing.
    pub fn schedule_karras(
        &self,
        num_steps: usize,
        sigma_min: f32,
        sigma_max: f32,
        rho: f32,
    ) -> DiffusionSchedule {
        let sigmas = self.sigmas_karras(num_steps, sigma_min, sigma_max, rho);
        let timesteps = sigmas[..sigmas.len() - 1]
            .iter()
            .map(|&sigma| self.sigma_to_timestep(sigma))
            .collect();

        DiffusionSchedule { sigmas, timesteps }
    }

    /// Convert sigma to approximate timestep for UNet input.
    ///
    /// Uses binary search to find the closest timestep, then interpolates.
    pub fn sigma_to_timestep(&self, sigma: f32) -> f32 {
        let n = self.num_train_timesteps;

        // Handle edge cases
        let sigma_max = self.timestep_to_sigma(n - 1);
        let sigma_min = self.timestep_to_sigma(0);

        if sigma >= sigma_max {
            return (n - 1) as f32;
        }
        if sigma <= sigma_min {
            return 0.0;
        }

        // Binary search: find timestep where sigma is closest
        // sigma[t] is monotonically increasing with t
        let mut low = 0;
        let mut high = n - 1;

        while low < high {
            let mid = (low + high) / 2;
            let mid_sigma = self.timestep_to_sigma(mid);

            if mid_sigma < sigma {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        // Interpolate between adjacent timesteps
        let t_high = low;
        let t_low = low.saturating_sub(1);

        let sigma_low = self.timestep_to_sigma(t_low);
        let sigma_high = self.timestep_to_sigma(t_high);

        if (sigma_high - sigma_low).abs() < 1e-6 {
            return t_low as f32;
        }

        // Linear interpolation
        t_low as f32 + (sigma - sigma_low) / (sigma_high - sigma_low)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_loader::SafeTensorsFile;
    use std::path::Path;

    const MODEL_PATH: &str = "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors";

    #[test]
    fn load_scheduler_from_checkpoint() {
        let path = Path::new(MODEL_PATH);
        if !path.exists() {
            println!("Model not found at {MODEL_PATH}, skipping");
            return;
        }

        let file = SafeTensorsFile::open(path).expect("Failed to open checkpoint");
        let tensors = file.tensors().expect("Failed to parse safetensors");
        let device = Default::default();

        let scheduler = NoiseSchedule::load(&tensors, &device).expect("Failed to load scheduler");

        assert_eq!(scheduler.num_train_timesteps(), 1000);

        // Check sigma range
        let (sigma_min, sigma_max) = scheduler.sigma_range();
        println!("Sigma range: [{sigma_min}, {sigma_max}]");
        assert!(sigma_min < 0.1, "sigma_min should be small");
        assert!(sigma_max > 10.0, "sigma_max should be large");
    }

    #[test]
    fn linear_schedule() {
        let scheduler = NoiseSchedule::linear(1000, 0.00085, 0.012);

        assert_eq!(scheduler.num_train_timesteps(), 1000);

        // First alpha should be close to 1
        assert!(scheduler.alphas_cumprod[0] > 0.99);

        // Last alpha should be close to 0
        assert!(scheduler.alphas_cumprod[999] < 0.01);
    }

    #[test]
    fn sigmas_for_steps() {
        let scheduler = NoiseSchedule::linear(1000, 0.00085, 0.012);

        let sigmas = scheduler.sigmas_for_steps(20);
        assert_eq!(sigmas.len(), 21, "Should have num_steps + 1 sigmas");

        // Sigmas should decrease
        for i in 0..sigmas.len() - 1 {
            assert!(
                sigmas[i] >= sigmas[i + 1],
                "Sigmas should be non-increasing: {} >= {}",
                sigmas[i],
                sigmas[i + 1]
            );
        }

        // Final sigma should be 0
        assert_eq!(sigmas[20], 0.0);
    }
}
