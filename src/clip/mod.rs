//! CLIP text encoder implementation.
//!
//! This module provides the CLIP text encoder used for encoding text prompts
//! into conditioning embeddings for Stable Diffusion.
//!
//! # Architecture
//!
//! The CLIP text encoder consists of:
//! - Token and position embeddings
//! - Transformer encoder with causal self-attention
//! - Final layer normalization
//!
//! # Usage
//!
//! ```ignore
//! use hearth::clip::{Sd15ClipTextEncoder, ClipTokenizer};
//!
//! // Load tokenizer
//! let tokenizer = ClipTokenizer::from_files(&vocab_path, &merges_path)?;
//!
//! // Load encoder from SD checkpoint
//! let encoder = Sd15ClipTextEncoder::load(&tensors, &device)?;
//!
//! // Encode text
//! let tokens = tokenizer.encode("a photo of a cat", &device);
//! let conditioning = encoder.forward(tokens);
//! ```

mod attention;
mod conditioning;
mod config;
mod embeddings;
mod encoder;
mod layer;
mod mlp;
mod open_clip;
mod tokenizer;
mod transformer;

pub use conditioning::{Sd15Conditioning, SdxlConditioning};
pub use config::{ClipConfig, Sd15Clip};
pub use encoder::ClipTextEncoder;
pub use open_clip::OpenClipTextEncoder;
pub use tokenizer::ClipTokenizer;

/// SD 1.5 CLIP text encoder type alias.
///
/// CLIP-ViT-L/14 configuration:
/// - 768 hidden dimension
/// - 12 attention heads
/// - 3072 feed-forward dimension
/// - 12 transformer layers
/// - 49,408 vocabulary size
/// - 77 max sequence length
/// - QuickGELU activation
pub type Sd15ClipTextEncoder = ClipTextEncoder<Sd15Clip, 768, 12, 3072, 12, 49408, 77, true>;

/// SDXL CLIP-L text encoder type alias.
///
/// Same architecture as SD 1.5 CLIP-L, but loaded with a different prefix
/// and used to extract penultimate layer output for SDXL.
pub type SdxlClipLTextEncoder = ClipTextEncoder<Sd15Clip, 768, 12, 3072, 12, 49408, 77, true>;
