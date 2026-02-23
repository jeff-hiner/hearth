//! Diffusion sampling algorithms.
//!
//! This module provides noise schedulers and samplers for denoising diffusion.
//!
//! # Schedulers
//!
//! Schedulers define the noise schedule - how noise is added and removed over timesteps.
//! The [`NoiseSchedule`] loads parameters from SD checkpoints.
//! [`SchedulerKind`] selects between Normal and Karras sigma spacing.
//!
//! # Samplers
//!
//! Samplers implement the actual denoising algorithm:
//! - [`EulerSampler`] - First-order Euler method (fast, simple, deterministic)
//! - [`EulerASampler`] - Euler Ancestral (first-order, stochastic)
//! - [`DpmPp2mSampler`] - DPM++ 2M (second-order, deterministic)
//! - [`DpmPpSdeSampler`] - DPM++ SDE (second-order, stochastic)
//!
//! [`SamplerKind`] selects between samplers from the CLI.
//!
//! # Example
//!
//! ```ignore
//! let scheduler = NoiseSchedule::load(&tensors, &device)?;
//! let schedule = SchedulerKind::Normal.schedule(&scheduler, 20);
//! let sampler = EulerSampler::new(scheduler);
//! let latent = sampler.sample_with_schedule(&unet, noise, &pos, &neg, &schedule, 7.5);
//! ```

mod cfg;
mod dpm_pp_2m;
mod dpm_pp_sde;
mod euler;
mod euler_a;
mod sampler_kind;
mod scheduler;
mod scheduler_kind;

pub use dpm_pp_2m::DpmPp2mSampler;
pub use dpm_pp_sde::DpmPpSdeSampler;
pub use euler::EulerSampler;
pub use euler_a::EulerASampler;
pub use sampler_kind::SamplerKind;
pub use scheduler::{DiffusionSchedule, NoiseSchedule};
pub use scheduler_kind::SchedulerKind;

/// Optional progress callback: `(current_step, total_steps)`.
pub type ProgressCallback<'a> = Option<&'a dyn Fn(usize, usize)>;
