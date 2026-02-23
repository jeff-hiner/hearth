//! ComfyUI-compatible tensor types
//!
//! These types mirror ComfyUI's internal representations:
//! - LATENT: Latent space images [B, C, H, W] - typically [1, 4, H/8, W/8] for SD
//! - IMAGE: RGB images [B, H, W, C] float 0-1
//! - MASK: Single-channel masks [B, H, W] float 0-1
//! - CONDITIONING: Encoded prompts with metadata

use crate::node::handle::ControlNetHandle;
use burn::tensor::Tensor;
use std::collections::HashMap;

/// The compute backend type used throughout Hearth.
///
/// Selected via Cargo features:
/// - `gpu-vulkan-f16` (default): Uses Vulkan + SPIR-V with f16 precision
/// - `gpu-vulkan-bf16`: Uses Vulkan + SPIR-V with bf16 precision
#[cfg(feature = "gpu-vulkan-f16")]
pub type Backend = burn::backend::Wgpu<half::f16>;

#[cfg(feature = "gpu-vulkan-bf16")]
pub type Backend = burn::backend::Wgpu<half::bf16>;

/// Latent space image representation
///
/// Shape: [batch, channels, height, width]
/// - For SD 1.5/SDXL: channels = 4, spatial dims are 1/8 of pixel dims
#[derive(Debug, Clone)]
pub struct Latent {
    /// The underlying tensor data.
    pub samples: Tensor<Backend, 4>,
}

impl Latent {
    /// Wrap an existing tensor as a latent.
    pub fn new(samples: Tensor<Backend, 4>) -> Self {
        Self { samples }
    }

    /// Create an empty (zeroed) latent of the given dimensions.
    pub fn empty(batch: usize, channels: usize, height: usize, width: usize) -> Self {
        let device = Default::default();
        Self {
            samples: Tensor::zeros([batch, channels, height, width], &device),
        }
    }

    /// Create a latent filled with random noise (for txt2img).
    pub fn noise(batch: usize, channels: usize, height: usize, width: usize, seed: u64) -> Self {
        let device = Default::default();
        // TODO: Use seeded RNG for reproducibility
        let _ = seed;
        Self {
            samples: Tensor::random(
                [batch, channels, height, width],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
        }
    }

    /// Returns the shape as `[batch, channels, height, width]`.
    pub fn shape(&self) -> [usize; 4] {
        self.samples.shape().dims()
    }

    /// Get spatial dimensions (height, width) in latent space
    pub fn spatial_dims(&self) -> (usize, usize) {
        let dims = self.shape();
        (dims[2], dims[3])
    }
}

/// RGB image representation
///
/// Shape: [batch, height, width, channels]
/// Values are float 0.0-1.0
#[derive(Debug, Clone)]
pub struct Image {
    /// The underlying tensor data.
    pub data: Tensor<Backend, 4>,
}

impl Image {
    /// Wrap an existing tensor as an image.
    pub fn new(data: Tensor<Backend, 4>) -> Self {
        Self { data }
    }

    /// Returns the shape as `[batch, height, width, channels]`.
    pub fn shape(&self) -> [usize; 4] {
        self.data.shape().dims()
    }

    /// Get image dimensions (height, width)
    pub fn dims(&self) -> (usize, usize) {
        let shape = self.shape();
        (shape[1], shape[2])
    }

    /// Get number of images in batch
    pub fn batch_size(&self) -> usize {
        self.shape()[0]
    }

    /// Get number of channels (typically 3 for RGB)
    pub fn channels(&self) -> usize {
        self.shape()[3]
    }
}

/// Single-channel mask representation
///
/// Shape: [batch, height, width]
/// Values are float 0.0-1.0 (0 = masked/black, 1 = unmasked/white)
#[derive(Debug, Clone)]
pub struct Mask {
    /// The underlying tensor data.
    pub data: Tensor<Backend, 3>,
}

impl Mask {
    /// Wrap an existing tensor as a mask.
    pub fn new(data: Tensor<Backend, 3>) -> Self {
        Self { data }
    }

    /// Returns the shape as `[batch, height, width]`.
    pub fn shape(&self) -> [usize; 3] {
        self.data.shape().dims()
    }

    /// Create a mask of all ones (fully unmasked)
    pub fn ones(batch: usize, height: usize, width: usize) -> Self {
        let device = Default::default();
        Self {
            data: Tensor::ones([batch, height, width], &device),
        }
    }

    /// Create a mask of all zeros (fully masked)
    pub fn zeros(batch: usize, height: usize, width: usize) -> Self {
        let device = Default::default();
        Self {
            data: Tensor::zeros([batch, height, width], &device),
        }
    }

    /// Invert the mask (0 becomes 1, 1 becomes 0)
    pub fn invert(self) -> Self {
        let ones: Tensor<Backend, 3> = Tensor::ones_like(&self.data);
        Self {
            data: ones - self.data,
        }
    }
}

/// A ControlNet reference stored in conditioning metadata.
///
/// Carries the model handle plus the hint image and application settings
/// needed to build a [`ControlNetUnit`](crate::controlnet::ControlNetUnit)
/// at sampling time.
#[derive(Debug, Clone)]
pub struct ControlNetRef {
    /// Handle to the loaded ControlNet in the model manager.
    pub(crate) handle: ControlNetHandle,
    /// Pre-processed hint image `[1, 3, H, W]` in `[0, 1]`.
    pub(crate) hint: Tensor<Backend, 4>,
    /// Strength multiplier (0.0 = no effect, 1.0 = full effect).
    pub(crate) strength: f32,
    /// Normalized start of activation (0.0 = first step).
    pub(crate) start_percent: f32,
    /// Normalized end of activation (1.0 = last step).
    pub(crate) end_percent: f32,
}

/// Metadata associated with conditioning
pub type ConditioningMeta = HashMap<String, ConditioningValue>;

/// Values that can be stored in conditioning metadata.
#[derive(Debug, Clone)]
pub enum ConditioningValue {
    /// A floating-point value.
    Float(f32),
    /// An integer value.
    Int(i64),
    /// A string value.
    String(String),
    /// A 2D tensor value.
    Tensor(Tensor<Backend, 2>),
    /// A stack of ControlNet references to apply during sampling.
    ControlNetStack(Vec<ControlNetRef>),
}

/// A single conditioning entry (encoded prompt + metadata)
#[derive(Debug, Clone)]
pub struct ConditioningEntry {
    /// The encoded text embedding
    pub embedding: Tensor<Backend, 3>, // [batch, seq_len, hidden_dim]
    /// Additional metadata (pooled output, attention masks, etc.)
    pub meta: ConditioningMeta,
}

/// Conditioning representation (list of conditioning entries).
///
/// In ComfyUI, conditioning is a list of (tensor, dict) tuples.
/// Multiple entries allow for prompt weighting, area conditioning, etc.
#[derive(Debug, Clone)]
pub struct Conditioning {
    /// The list of conditioning entries.
    pub entries: Vec<ConditioningEntry>,
}

impl Conditioning {
    /// Create conditioning from a list of entries.
    pub fn new(entries: Vec<ConditioningEntry>) -> Self {
        Self { entries }
    }

    /// Create conditioning with a single embedding and no metadata.
    pub fn single(embedding: Tensor<Backend, 3>) -> Self {
        Self {
            entries: vec![ConditioningEntry {
                embedding,
                meta: HashMap::new(),
            }],
        }
    }

    /// Returns true if there are no conditioning entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of conditioning entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_creation() {
        let latent = Latent::empty(1, 4, 64, 64);
        assert_eq!(latent.shape(), [1, 4, 64, 64]);
        assert_eq!(latent.spatial_dims(), (64, 64));
    }

    #[test]
    fn latent_noise() {
        let latent = Latent::noise(1, 4, 64, 64, 42);
        assert_eq!(latent.shape(), [1, 4, 64, 64]);
    }

    #[test]
    fn image_creation() {
        let device = Default::default();
        let data: Tensor<Backend, 4> = Tensor::zeros([1, 512, 512, 3], &device);
        let image = Image::new(data);

        assert_eq!(image.dims(), (512, 512));
        assert_eq!(image.batch_size(), 1);
        assert_eq!(image.channels(), 3);
    }

    #[test]
    fn mask_operations() {
        let mask = Mask::ones(1, 64, 64);
        assert_eq!(mask.shape(), [1, 64, 64]);

        let inverted = mask.invert();
        // Use convert::<f32>() to handle both f32 and f16 backends
        let data: Vec<f32> = inverted.data.into_data().convert::<f32>().to_vec().unwrap();
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn conditioning_creation() {
        let device = Default::default();
        let embedding: Tensor<Backend, 3> = Tensor::zeros([1, 77, 768], &device);
        let cond = Conditioning::single(embedding);

        assert_eq!(cond.len(), 1);
        assert!(!cond.is_empty());
    }
}
