//! ControlNet integration for guided image generation.
//!
//! A ControlNet is a copy of the UNet encoder that takes an additional
//! conditioning image (hint) and produces residuals that steer the UNet's
//! generation toward the conditioning signal.
//!
//! # Architecture
//!
//! ```text
//!                      ┌─────────────┐
//!   hint image ──────►│ hint_encoder │
//!                      └──────┬──────┘
//!                             │ (added to first conv output)
//!                             ▼
//!   noisy latent ──►┌─ ControlNet Encoder ─┐
//!   timestep ──────►│  (same as UNet down  │──► zero_conv outputs ──► residuals
//!   conditioning ──►│   blocks + mid)      │
//!                   └──────────────────────┘
//! ```
//!
//! Multiple ControlNets can be applied simultaneously by summing their
//! weighted residuals before adding to the UNet.
//!
//! # Example
//!
//! ```ignore
//! let cn = Sd15ControlNet::load(&tensors, 3, &device)?;
//! let output = cn.forward(latent, &hint, timestep, &context, None);
//! let output = output.scale(0.8); // weight
//! // Pass to UNet via forward_with_emb(..., Some(&output))
//! ```

mod hint;
mod model;
mod output;

use crate::{types::Backend, unet::UnetConfig};
use burn::prelude::*;
pub use model::{ControlNet, Sd15ControlNet, SdxlControlNet};
pub use output::ControlNetOutput;

/// A ControlNet paired with its conditioning inputs and application settings.
///
/// Groups a borrowed ControlNet model with the hint image and parameters
/// that control how strongly and when it influences generation. The model
/// is borrowed because units only live for the duration of a sampling call.
pub struct ControlNetUnit<'a, C: UnetConfig> {
    /// The loaded ControlNet model (borrowed from the model manager).
    pub model: &'a ControlNet<C>,
    /// Pre-processed conditioning image `[1, hint_ch, H, W]` in [0, 1].
    pub hint: Tensor<Backend, 4>,
    /// Strength multiplier (0.0 = no effect, 1.0 = full effect).
    pub weight: f32,
    /// Normalized start of activation (0.0 = first step, 1.0 = last step).
    pub start: f32,
    /// Normalized end of activation (0.0 = first step, 1.0 = last step).
    pub end: f32,
}

impl<C: UnetConfig> std::fmt::Debug for ControlNetUnit<'_, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ControlNetUnit")
            .field("weight", &self.weight)
            .field("start", &self.start)
            .field("end", &self.end)
            .finish()
    }
}

impl<C: UnetConfig> ControlNetUnit<'_, C> {
    /// Check if this ControlNet is active at the given step.
    pub fn is_active(&self, step: usize, num_steps: usize) -> bool {
        let step_start = (self.start * num_steps as f32).floor() as usize;
        let step_end = (self.end * num_steps as f32).floor() as usize;
        step >= step_start && step < step_end
    }
}
