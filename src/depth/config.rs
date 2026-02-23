//! Architecture constants for Depth Anything V2 models.
//!
//! Defines the DINOv2 ViT-S/14 encoder and DPT decoder configurations.

/// ViT-S/14 encoder configuration (DINOv2).
pub(crate) struct VitSmall;

impl VitSmall {
    /// Embedding dimension.
    pub(crate) const EMBED_DIM: usize = 384;
    /// Number of transformer blocks.
    pub(crate) const NUM_BLOCKS: usize = 12;
    /// Number of attention heads.
    pub(crate) const NUM_HEADS: usize = 6;
    /// MLP hidden dimension (4x embed_dim).
    pub(crate) const MLP_DIM: usize = 1536;
    /// Patch size in pixels.
    pub(crate) const PATCH_SIZE: usize = 14;
    /// Layer indices to extract intermediate features from (0-indexed).
    pub(crate) const INTERMEDIATE_LAYERS: [usize; 4] = [2, 5, 8, 11];
    /// LayerNorm epsilon.
    pub(crate) const LN_EPS: f64 = 1e-6;
}

/// DPT decoder configuration for ViT-S.
pub(crate) struct DptSmall;

impl DptSmall {
    /// Feature dimension used in the fusion/scratch layers.
    pub(crate) const FEATURES: usize = 64;
    /// Per-level output channel counts after reassembly projection.
    pub(crate) const OUT_CHANNELS: [usize; 4] = [48, 96, 192, 384];
}
