//! Generic-erasing enums for model variants.
//!
//! `DenoisingUnet` has an associated type `Cond`, which prevents
//! `dyn DenoisingUnet`. These enums erase the generic parameter
//! so that the node system can store models uniformly.

use crate::{
    clip::{ClipTokenizer, OpenClipTextEncoder, Sd15ClipTextEncoder, SdxlClipLTextEncoder},
    controlnet::{Sd15ControlNet, SdxlControlNet},
    model_loader::LoadError,
    types::Backend,
    unet::{Sd15Unet2D, SdxlUnet2D},
    vae::{Sd15VaeDecoder, Sd15VaeEncoder, SdxlVaeDecoder, SdxlVaeEncoder},
};
use burn::tensor::Device;
use safetensors::SafeTensors;

/// Erased UNet variant.
#[derive(Debug)]
pub enum UnetVariant {
    /// Stable Diffusion 1.5 UNet.
    Sd15(Sd15Unet2D),
    /// Stable Diffusion XL UNet.
    Sdxl(SdxlUnet2D),
}

impl UnetVariant {
    /// Apply LoRA deltas to the UNet, dispatching to the correct variant.
    ///
    /// Returns the total number of deltas applied.
    pub(crate) fn apply_lora(
        &mut self,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        match self {
            Self::Sd15(unet) => unet.apply_lora("lora_unet", lora, strength, device),
            Self::Sdxl(unet) => unet.apply_lora("lora_unet", lora, strength, device),
        }
    }
}

/// Erased VAE variant holding both encoder and decoder.
#[derive(Debug)]
pub enum VaeVariant {
    /// Stable Diffusion 1.5 VAE.
    Sd15 {
        /// Encoder (image → latent).
        encoder: Sd15VaeEncoder,
        /// Decoder (latent → image).
        decoder: Sd15VaeDecoder,
    },
    /// Stable Diffusion XL VAE.
    Sdxl {
        /// Encoder (image → latent).
        encoder: SdxlVaeEncoder,
        /// Decoder (latent → image).
        decoder: SdxlVaeDecoder,
    },
}

/// Erased CLIP encoder variant.
///
/// SD 1.5 uses a single CLIP-L encoder. SDXL uses CLIP-L + OpenCLIP-G,
/// so the SDXL variant stores both encoders. Both variants are boxed to
/// keep enum size uniform (they carry ~2-3 KiB of weight pointers).
#[derive(Debug)]
pub enum ClipVariant {
    /// SD 1.5: single CLIP-ViT-L/14.
    Sd15 {
        /// The CLIP text encoder.
        encoder: Box<Sd15ClipTextEncoder>,
        /// Shared BPE tokenizer.
        tokenizer: ClipTokenizer,
    },
    /// SDXL: CLIP-L + OpenCLIP-G.
    Sdxl {
        /// CLIP-L text encoder (penultimate layer output).
        clip_l: Box<SdxlClipLTextEncoder>,
        /// OpenCLIP-G text encoder (hidden + pooled output).
        clip_g: Box<OpenClipTextEncoder>,
        /// Shared BPE tokenizer.
        tokenizer: ClipTokenizer,
    },
}

impl ClipVariant {
    /// Apply LoRA deltas to the CLIP encoder(s), dispatching to the correct variant.
    ///
    /// Returns the total number of deltas applied.
    pub(crate) fn apply_lora(
        &mut self,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        match self {
            Self::Sd15 { encoder, .. } => {
                encoder.apply_lora("text_model", "lora_te", lora, strength, device)
            }
            Self::Sdxl { clip_l, clip_g, .. } => {
                let mut count = 0;
                count += clip_l.apply_lora("text_model", "lora_te1", lora, strength, device)?;
                count += clip_g.apply_lora("lora_te2", lora, strength, device)?;
                Ok(count)
            }
        }
    }
}

/// Erased ControlNet variant.
#[derive(Debug)]
pub enum ControlNetVariant {
    /// SD 1.5 ControlNet.
    Sd15(Sd15ControlNet),
    /// SDXL ControlNet.
    Sdxl(SdxlControlNet),
}
