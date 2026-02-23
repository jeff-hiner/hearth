//! CLIP model configuration traits and implementations.

/// Configuration trait for CLIP text encoder models.
///
/// This trait defines the architectural parameters for different CLIP variants.
/// Const generics encode dimensions at compile time for type safety.
pub trait ClipConfig<
    const HIDDEN: usize,
    const HEADS: usize,
    const FF: usize,
    const LAYERS: usize,
    const VOCAB: usize,
    const SEQ_LEN: usize,
>
{
    /// Epsilon value for layer normalization.
    const LAYER_NORM_EPS: f64;
}

/// SD 1.5 CLIP-ViT-L/14 text encoder configuration.
///
/// Architecture:
/// - 12 transformer layers
/// - 768 hidden dimension
/// - 12 attention heads (64 dim per head)
/// - 3072 feed-forward dimension
/// - 49,408 vocab size
/// - 77 max sequence length
#[derive(Debug)]
pub struct Sd15Clip;

impl ClipConfig<768, 12, 3072, 12, 49408, 77> for Sd15Clip {
    const LAYER_NORM_EPS: f64 = 1e-5;
}
