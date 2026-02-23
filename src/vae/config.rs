//! VAE decoder configuration.

/// Configuration trait for VAE decoder variants.
///
/// Const generics encode the architecture parameters, while associated consts
/// hold values that can't be const generic (like floats).
///
/// Generic parameters: `<GROUPS, LATENT, OUT, CH0, CH1, CH2, CH3>`
pub trait VaeConfig<
    const GROUPS: usize,
    const LATENT: usize,
    const OUT: usize,
    const CH0: usize,
    const CH1: usize,
    const CH2: usize,
    const CH3: usize,
>
{
    /// Scaling factor for latent space.
    const SCALING_FACTOR: f32;
}

/// SD 1.5 VAE configuration.
#[derive(Debug, Clone, Copy)]
pub struct Sd15Vae;

impl VaeConfig<32, 4, 3, 128, 256, 512, 512> for Sd15Vae {
    const SCALING_FACTOR: f32 = 0.18215;
}

/// SDXL VAE configuration.
#[derive(Debug, Clone, Copy)]
pub struct SdxlVae;

impl VaeConfig<32, 4, 3, 128, 256, 512, 512> for SdxlVae {
    const SCALING_FACTOR: f32 = 0.13025;
}
