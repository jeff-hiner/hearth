//! UNet denoising model for Stable Diffusion.
//!
//! The UNet takes a noisy latent, timestep, and text conditioning,
//! and predicts the noise to be removed.
//!
//! # Architecture
//!
//! SD 1.5 UNet:
//! - Input: 4-channel latent `[B, 4, H, W]`
//! - Time embedding: sinusoidal -> MLP (320 -> 1280)
//! - Down blocks: 4 levels with ResNet + optional cross-attention
//! - Middle block: ResNet + cross-attention + ResNet
//! - Up blocks: 4 levels with ResNet + optional cross-attention + skip connections
//! - Output: 4-channel noise prediction `[B, 4, H, W]`
//!
//! Channel progression: 320 -> 640 -> 1280 -> 1280

mod attention;
pub(crate) mod config;
pub(crate) mod downsample;
pub(crate) mod embeddings;
pub(crate) mod mid_block;
mod model;
pub(crate) mod resnet;
pub(crate) mod transformer;

pub use config::{Sd15Unet, SdxlUnet, UnetConfig};
pub(crate) use model::InputBlock;
pub use model::{DenoisingUnet, Unet, UnetConditioning, UnetTimingStats};

/// SD 1.5 UNet type alias.
pub type Sd15Unet2D = Unet<Sd15Unet>;

/// SDXL UNet type alias.
pub type SdxlUnet2D = Unet<SdxlUnet>;
