//! VAE (Variational Autoencoder) for image generation.
//!
//! Converts between pixel images and latent space representations.

mod config;
mod decoder;
mod down_block;
mod encoder;
mod mid_block;
#[cfg(test)]
mod tests;
mod up_block;

pub use config::{Sd15Vae, SdxlVae, VaeConfig};
pub use decoder::VaeDecoder;
pub use encoder::VaeEncoder;

/// SD 1.5 VAE decoder with standard configuration.
pub type Sd15VaeDecoder = VaeDecoder<Sd15Vae, 32, 4, 3, 128, 256, 512, 512>;

/// SDXL VAE decoder with standard configuration.
pub type SdxlVaeDecoder = VaeDecoder<SdxlVae, 32, 4, 3, 128, 256, 512, 512>;

/// SD 1.5 VAE encoder with standard configuration.
pub type Sd15VaeEncoder = VaeEncoder<Sd15Vae, 32, 4, 3, 128, 256, 512, 512>;

/// SDXL VAE encoder with standard configuration.
pub type SdxlVaeEncoder = VaeEncoder<SdxlVae, 32, 4, 3, 128, 256, 512, 512>;
