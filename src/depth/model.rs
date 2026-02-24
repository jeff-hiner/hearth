//! Depth Anything V2 model combining ViT encoder and DPT decoder.
//!
//! Produces monocular relative depth maps from RGB images.

use super::{decoder::DptDecoder, encoder::VitEncoder};
use crate::{model_loader::LoadError, types::Backend};
use burn::{prelude::*, tensor::activation::relu};
use safetensors::SafeTensors;

/// Depth Anything V2 Small (ViT-S/14 encoder + DPT decoder).
///
/// Produces a relative inverse depth map where higher values = closer.
///
/// # Input
/// - Normalized RGB image `[B, 3, H, W]` (H, W divisible by 14)
/// - Normalization: ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
///
/// # Output
/// - Depth map `[B, 1, H_out, W_out]` with non-negative values
///   (H_out, W_out depend on DPT decoder upsampling)
#[derive(Debug)]
pub struct DepthAnythingV2 {
    /// ViT-S/14 encoder.
    encoder: VitEncoder,
    /// DPT depth head.
    decoder: DptDecoder,
}

impl DepthAnythingV2 {
    /// Load from a Depth Anything V2 safetensors file.
    ///
    /// Expected key prefixes:
    /// - `pretrained.*` — ViT encoder weights
    /// - `depth_head.*` — DPT decoder weights
    pub fn load(tensors: &SafeTensors<'_>, device: &Device<Backend>) -> Result<Self, LoadError> {
        let encoder = VitEncoder::load(tensors, "pretrained", device)?;
        let decoder = DptDecoder::load(tensors, "depth_head", device)?;

        Ok(Self { encoder, decoder })
    }

    /// Run depth estimation.
    ///
    /// # Arguments
    /// * `x` - Normalized RGB image `[B, 3, H, W]` (H, W divisible by 14)
    ///
    /// # Returns
    /// Depth map `[B, 1, H_out, W_out]` with non-negative relative depth values.
    pub fn forward(&self, x: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        let (features, patch_h, patch_w) = self.encoder.forward(x);
        let depth = self.decoder.forward(features, patch_h, patch_w);
        relu(depth)
    }
}
