//! UNet configuration traits and implementations.

/// Configuration trait for UNet models.
///
/// Defines architectural parameters for different SD variants.
pub trait UnetConfig {
    /// Base channel count (first block).
    const BASE_CHANNELS: usize;
    /// Channel multipliers for each level.
    const CHANNEL_MULT: &'static [usize];
    /// Number of ResNet layers per block.
    const LAYERS_PER_BLOCK: usize;
    /// Number of transformer blocks at each level (0 = no attention).
    ///
    /// SD 1.5: `[1, 1, 1, 0]`, SDXL: `[0, 2, 10]`.
    const TRANSFORMER_DEPTH: &'static [usize];
    /// Number of transformer blocks in the mid block.
    const MID_BLOCK_DEPTH: usize;
    /// ADM (vector) conditioning input dimension. 0 means no vector conditioning.
    ///
    /// SD 1.5: 0, SDXL: 2816.
    const ADM_IN_CHANNELS: usize;
    /// Cross-attention context dimension (CLIP hidden size).
    const CONTEXT_DIM: usize;
    /// Group normalization groups.
    const NORM_GROUPS: usize;
    /// Group normalization epsilon.
    const NORM_EPS: f64;

    /// Returns `(num_heads, head_dim)` for attention at a given channel width.
    fn attention_params(out_ch: usize) -> (usize, usize);
}

/// SD 1.5 UNet configuration.
///
/// Architecture:
/// - Base channels: 320
/// - Channel multipliers: [1, 2, 4, 4] -> [320, 640, 1280, 1280]
/// - 2 ResNet layers per block
/// - Transformer depth: [1, 1, 1, 0] (single block at levels 0-2, none at 3)
/// - 8 attention heads
/// - Context dim: 768 (CLIP-ViT-L/14)
#[derive(Debug)]
pub struct Sd15Unet;

impl UnetConfig for Sd15Unet {
    const BASE_CHANNELS: usize = 320;
    const CHANNEL_MULT: &'static [usize] = &[1, 2, 4, 4];
    const LAYERS_PER_BLOCK: usize = 2;
    const TRANSFORMER_DEPTH: &'static [usize] = &[1, 1, 1, 0];
    const MID_BLOCK_DEPTH: usize = 1;
    const ADM_IN_CHANNELS: usize = 0;
    const CONTEXT_DIM: usize = 768;
    const NORM_GROUPS: usize = 32;
    const NORM_EPS: f64 = 1e-5;

    fn attention_params(out_ch: usize) -> (usize, usize) {
        (8, out_ch / 8)
    }
}

/// SDXL UNet configuration.
///
/// Architecture:
/// - Base channels: 320
/// - Channel multipliers: [1, 2, 4] -> [320, 640, 1280]
/// - 2 ResNet layers per block
/// - Transformer depth: [0, 2, 10] (no attention at level 0, 2 blocks at 1, 10 at 2)
/// - Fixed 64-dim heads (num_heads = out_ch / 64)
/// - Context dim: 2048 (concatenated CLIP-L 768 + OpenCLIP-G 1280)
/// - ADM conditioning: 2816 (pooled 1280 + 6 * 256 Fourier features)
#[derive(Debug)]
pub struct SdxlUnet;

impl UnetConfig for SdxlUnet {
    const BASE_CHANNELS: usize = 320;
    const CHANNEL_MULT: &'static [usize] = &[1, 2, 4];
    const LAYERS_PER_BLOCK: usize = 2;
    const TRANSFORMER_DEPTH: &'static [usize] = &[0, 2, 10];
    const MID_BLOCK_DEPTH: usize = 10;
    const ADM_IN_CHANNELS: usize = 2816;
    const CONTEXT_DIM: usize = 2048;
    const NORM_GROUPS: usize = 32;
    const NORM_EPS: f64 = 1e-5;

    fn attention_params(out_ch: usize) -> (usize, usize) {
        (out_ch / 64, 64)
    }
}
