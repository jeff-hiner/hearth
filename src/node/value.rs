//! Dynamic value type for node edges.
//!
//! [`NodeValue`] wraps all types that can flow between nodes in the execution
//! graph. Tensor types come from [`crate::types`]; model references are
//! lightweight [`Copy`] handles.

use super::handle::{ClipHandle, ControlNetHandle, ModelHandle, VaeHandle};
use crate::{
    sampling::{DiffusionSchedule, SamplerKind, SchedulerKind},
    types::{Conditioning, Image, Latent, Mask},
};

/// A dynamically typed value that flows along edges in the execution graph.
#[derive(Debug)]
pub enum NodeValue {
    /// Latent-space tensor `[B, 4, H, W]`.
    Latent(Latent),
    /// RGB image tensor `[B, H, W, 3]` in `[0, 1]`.
    Image(Image),
    /// Single-channel mask `[B, H, W]` in `[0, 1]`.
    Mask(Mask),
    /// Encoded text conditioning.
    Conditioning(Conditioning),
    /// Handle to a loaded UNet / diffusion model.
    Model(ModelHandle),
    /// Handle to a loaded VAE.
    Vae(VaeHandle),
    /// Handle to a loaded CLIP text encoder.
    Clip(ClipHandle),
    /// Handle to a loaded ControlNet.
    ControlNet(ControlNetHandle),
    /// Scalar float.
    Float(f32),
    /// Scalar integer.
    Int(i64),
    /// String value.
    String(String),
    /// Sampling algorithm selection.
    Sampler(SamplerKind),
    /// Sigma schedule selection.
    Scheduler(SchedulerKind),
    /// Pre-computed diffusion schedule (sigmas + timesteps).
    Schedule(DiffusionSchedule),
}

impl NodeValue {
    /// Human-readable type name for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Latent(_) => "LATENT",
            Self::Image(_) => "IMAGE",
            Self::Mask(_) => "MASK",
            Self::Conditioning(_) => "CONDITIONING",
            Self::Model(_) => "MODEL",
            Self::Vae(_) => "VAE",
            Self::Clip(_) => "CLIP",
            Self::ControlNet(_) => "CONTROL_NET",
            Self::Float(_) => "FLOAT",
            Self::Int(_) => "INT",
            Self::String(_) => "STRING",
            Self::Sampler(_) => "SAMPLER",
            Self::Scheduler(_) => "SCHEDULER",
            Self::Schedule(_) => "SCHEDULE",
        }
    }
}
