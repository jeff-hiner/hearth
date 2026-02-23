//! Conditioning types for Stable Diffusion models.
//!
//! These are the typed conditioning inputs consumed by the UNet during denoising.
//! SD 1.5 uses a single CLIP context; SDXL uses dual-encoder hidden states plus
//! a pooled + Fourier-encoded vector.

use crate::{types::Backend, unet::UnetConditioning};
use burn::prelude::*;

// ---------------------------------------------------------------------------
// SD 1.5
// ---------------------------------------------------------------------------

/// SD 1.5 conditioning: just cross-attention context.
#[derive(Debug)]
pub struct Sd15Conditioning {
    /// CLIP text encoder output `[B, 77, 768]`.
    pub(crate) context: Tensor<Backend, 3>,
}

impl Sd15Conditioning {
    /// Create SD 1.5 conditioning from a CLIP encoder output.
    pub fn new(context: Tensor<Backend, 3>) -> Self {
        Self { context }
    }
}

impl UnetConditioning for Sd15Conditioning {
    fn batch_cfg(positive: &Self, negative: &Self) -> Self {
        Self {
            context: Tensor::cat(vec![positive.context.clone(), negative.context.clone()], 0),
        }
    }

    fn context(&self) -> &Tensor<Backend, 3> {
        &self.context
    }

    fn y(&self) -> Option<&Tensor<Backend, 2>> {
        None
    }
}

// ---------------------------------------------------------------------------
// SDXL
// ---------------------------------------------------------------------------

/// Fourier feature embedding dimension per scalar (256 sin + cos pairs = 256 total).
const FOURIER_DIM: usize = 256;

/// SDXL conditioning: cross-attention context + vector (ADM) conditioning.
#[derive(Debug)]
pub struct SdxlConditioning {
    /// Concatenated CLIP-L + OpenCLIP-G hidden states `[B, 77, 2048]`.
    pub(crate) context: Tensor<Backend, 3>,
    /// Pooled + Fourier features vector `[B, 2816]`.
    pub(crate) y: Tensor<Backend, 2>,
}

impl SdxlConditioning {
    /// Build SDXL conditioning from dual CLIP outputs and image metadata.
    ///
    /// # Arguments
    /// * `clip_l_hidden` - CLIP-L penultimate layer output `[B, 77, 768]`
    /// * `clip_g_hidden` - OpenCLIP-G hidden states `[B, 77, 1280]`
    /// * `pooled` - OpenCLIP-G pooled output `[B, 1280]`
    /// * `original_size` - Original image dimensions `(height, width)`
    /// * `crop_coords` - Top-left crop coordinates `(top, left)`
    /// * `target_size` - Target output dimensions `(height, width)`
    ///
    /// # Returns
    /// Complete SDXL conditioning with:
    /// - `context`: `[B, 77, 2048]` (concatenated CLIP-L + OpenCLIP-G hidden states)
    /// - `y`: `[B, 2816]` (pooled [1280] + 6 Fourier embeddings [6 * 256])
    pub fn new(
        clip_l_hidden: Tensor<Backend, 3>,
        clip_g_hidden: Tensor<Backend, 3>,
        pooled: Tensor<Backend, 2>,
        original_size: (f32, f32),
        crop_coords: (f32, f32),
        target_size: (f32, f32),
        device: &Device<Backend>,
    ) -> Self {
        // Concatenate hidden states along feature dimension: [B, 77, 768] + [B, 77, 1280] -> [B, 77, 2048]
        let context = Tensor::cat(vec![clip_l_hidden, clip_g_hidden], 2);

        // Build y vector: pooled [B, 1280] + 6 Fourier embeddings [B, 6*256]
        let [batch, _] = pooled.shape().dims();

        let scalars = [
            original_size.0,
            original_size.1,
            crop_coords.0,
            crop_coords.1,
            target_size.0,
            target_size.1,
        ];

        let mut fourier_parts = Vec::with_capacity(6);
        for &scalar in &scalars {
            let emb = fourier_embed(scalar, FOURIER_DIM, device);
            // Expand to batch: [FOURIER_DIM] -> [B, FOURIER_DIM]
            let emb = emb.unsqueeze::<2>().expand([batch, FOURIER_DIM]);
            fourier_parts.push(emb);
        }

        // Concatenate: pooled [B, 1280] + fourier [B, 6*256] -> [B, 2816]
        let mut y_parts = vec![pooled];
        y_parts.extend(fourier_parts);
        let y = Tensor::cat(y_parts, 1);

        Self { context, y }
    }
}

impl UnetConditioning for SdxlConditioning {
    fn batch_cfg(positive: &Self, negative: &Self) -> Self {
        Self {
            context: Tensor::cat(vec![positive.context.clone(), negative.context.clone()], 0),
            y: Tensor::cat(vec![positive.y.clone(), negative.y.clone()], 0),
        }
    }

    fn context(&self) -> &Tensor<Backend, 3> {
        &self.context
    }

    fn y(&self) -> Option<&Tensor<Backend, 2>> {
        Some(&self.y)
    }
}

/// Compute Fourier feature embedding for a single scalar value.
///
/// Uses sinusoidal encoding matching the `timestep_embedding` convention from sgm:
/// `[cos, sin]` ordering (cos-first), consistent with how SDXL's `label_emb` MLP was trained.
///
/// Returns a 1D tensor of shape `[dim]`.
fn fourier_embed(value: f32, dim: usize, device: &Device<Backend>) -> Tensor<Backend, 1> {
    let half = dim / 2;

    // Frequency exponents: exp(-ln(10000) * i / half)
    let exponent: Tensor<Backend, 1> = Tensor::arange(0..half as i64, device).float();
    let exponent = exponent * (-f64::ln(10000.0) / half as f64);
    let freqs = exponent.exp() * value;

    let sin = freqs.clone().sin();
    let cos = freqs.cos();

    Tensor::cat(vec![cos, sin], 0)
}
