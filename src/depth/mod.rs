//! Monocular depth estimation via Depth Anything V2.
//!
//! This module implements the Depth Anything V2 Small model (ViT-S/14 encoder
//! + DPT decoder) for relative monocular depth estimation.
//!
//! # Architecture
//!
//! - **Encoder**: DINOv2 ViT-S/14 — patches the input image with 14x14 stride,
//!   processes through 12 transformer blocks, extracts features from layers [2, 5, 8, 11].
//! - **Decoder**: DPT head — reassembles multi-scale features, fuses bottom-up
//!   via RefineNet blocks, outputs a single-channel depth map.
//!
//! # Usage
//!
//! ```ignore
//! let model = DepthAnythingV2::load(&tensors, &device)?;
//! let depth = model.forward(normalized_image); // [B, 1, H_out, W_out]
//! ```
//!
//! Input images should be resized to 518x518 (divisible by 14) and normalized
//! with ImageNet statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].

mod config;
mod decoder;
mod encoder;
mod model;

pub use model::DepthAnythingV2;

/// ImageNet normalization mean (RGB).
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// ImageNet normalization standard deviation (RGB).
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Default inference resolution for ViT-S/14 (518 = 37 * 14).
pub const DEFAULT_RESOLUTION: usize = 518;
