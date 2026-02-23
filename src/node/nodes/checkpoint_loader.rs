//! `CheckpointLoaderSimple` node — loads a Stable Diffusion checkpoint.

use crate::{
    clip::{ClipTokenizer, OpenClipTextEncoder, SdxlClipLTextEncoder},
    model_loader::{self, SafeTensorsFile},
    model_manager::compute_weight_bytes,
    node::{
        Node, ResolvedInput, SlotDef, ValueType,
        context::ExecutionContext,
        error::NodeError,
        value::NodeValue,
        variant::{ClipVariant, UnetVariant, VaeVariant},
    },
    unet::{Sd15Unet2D, SdxlUnet2D},
    vae::{Sd15VaeDecoder, Sd15VaeEncoder, SdxlVaeDecoder, SdxlVaeEncoder},
};
use std::path::PathBuf;

/// Loads a SD checkpoint and returns model/clip/vae handles.
///
/// ComfyUI equivalent: `CheckpointLoaderSimple`
///
/// Detects whether the checkpoint is SD 1.5 or SDXL from tensor names,
/// loads all components, registers them with the model manager, and
/// returns lightweight handles.
#[derive(Debug)]
pub struct CheckpointLoaderSimple {
    /// Checkpoint filename (relative to models/checkpoints/).
    ckpt_name: PathBuf,
}

impl CheckpointLoaderSimple {
    /// Create a new checkpoint loader node.
    pub fn new(ckpt_name: PathBuf) -> Self {
        Self { ckpt_name }
    }
}

impl Node for CheckpointLoaderSimple {
    fn type_name(&self) -> &'static str {
        "CheckpointLoaderSimple"
    }

    fn inputs(&self) -> &'static [SlotDef] {
        // ckpt_name is a widget value, not an edge.
        &[]
    }

    fn outputs(&self) -> &'static [SlotDef] {
        static OUTPUTS: [SlotDef; 3] = [
            SlotDef::required("MODEL", ValueType::Model),
            SlotDef::required("CLIP", ValueType::Clip),
            SlotDef::required("VAE", ValueType::Vae),
        ];
        &OUTPUTS
    }

    fn execute(
        &self,
        _inputs: &[ResolvedInput],
        ctx: &mut ExecutionContext,
    ) -> Result<Vec<NodeValue>, NodeError> {
        let ckpt_path = ctx.models_dir().join("checkpoints").join(&self.ckpt_name);
        tracing::info!(path = %ckpt_path.display(), "loading checkpoint");

        let file = SafeTensorsFile::open(&ckpt_path)?;
        let tensors = file.tensors()?;
        let device = ctx.device().clone();

        // Detect model type from tensor names
        let is_sdxl = tensors
            .names()
            .iter()
            .any(|n| n.starts_with(model_loader::prefix::SDXL_CLIP_L));

        if is_sdxl {
            self.load_sdxl(&tensors, &device, ctx, &ckpt_path)
        } else {
            self.load_sd15(&tensors, &device, ctx, &ckpt_path)
        }
    }
}

impl CheckpointLoaderSimple {
    fn load_sdxl(
        &self,
        tensors: &safetensors::SafeTensors<'_>,
        device: &burn::tensor::Device<crate::types::Backend>,
        ctx: &mut ExecutionContext,
        ckpt_path: &std::path::Path,
    ) -> Result<Vec<NodeValue>, NodeError> {
        tracing::info!("detected SDXL checkpoint");

        // Compute weight sizes from safetensors header (no data loaded)
        let unet_bytes = compute_weight_bytes(tensors, model_loader::prefix::UNET, true);
        let clip_l_bytes = compute_weight_bytes(tensors, model_loader::prefix::SDXL_CLIP_L, true);
        let clip_g_bytes = compute_weight_bytes(tensors, model_loader::prefix::SDXL_CLIP_G, true);

        // Load UNet
        let unet = SdxlUnet2D::load(tensors, device)?;
        let unet_handle =
            ctx.models
                .register_unet(UnetVariant::Sdxl(unet), unet_bytes, ckpt_path.to_path_buf());

        // Load CLIP-L + OpenCLIP-G
        let clip_l = SdxlClipLTextEncoder::load_with_prefix(
            tensors,
            model_loader::prefix::SDXL_CLIP_L,
            device,
        )?;
        let clip_g = OpenClipTextEncoder::load(tensors, model_loader::prefix::SDXL_CLIP_G, device)?;

        // Load tokenizer from standard path
        let vocab = ctx.models_dir().join("clip/vocab.json");
        let merges = ctx.models_dir().join("clip/merges.txt");
        let tokenizer =
            ClipTokenizer::from_files(&vocab, &merges).map_err(|e| NodeError::Execution {
                message: format!("tokenizer: {e}"),
            })?;

        let clip_handle = ctx.models.register_clip(
            ClipVariant::Sdxl {
                clip_l: Box::new(clip_l),
                clip_g: Box::new(clip_g),
                tokenizer,
            },
            clip_l_bytes + clip_g_bytes,
            ckpt_path.to_path_buf(),
        );

        // Load VAE. For SDXL, prefer the fp16-fix if available (both encoder
        // and decoder — the original SDXL VAE weights overflow in fp16).
        let vae_path = ctx.models_dir().join("vae/sdxl-vae-fp16-fix.safetensors");
        let (encoder, decoder, vae_bytes, vae_source) = if vae_path.exists() {
            tracing::info!("loading fp16-fix VAE");
            let vae_file = SafeTensorsFile::open(&vae_path)?;
            let vae_tensors = vae_file.tensors()?;
            let bytes = compute_weight_bytes(&vae_tensors, "", true);
            let encoder = SdxlVaeEncoder::load(&vae_tensors, None, device)?;
            let decoder = SdxlVaeDecoder::load(&vae_tensors, None, device)?;
            (encoder, decoder, bytes, vae_path)
        } else {
            tracing::info!("loading VAE from checkpoint");
            let bytes = compute_weight_bytes(tensors, model_loader::prefix::VAE, true);
            let encoder = SdxlVaeEncoder::load(tensors, Some("first_stage_model"), device)?;
            let decoder = SdxlVaeDecoder::load(tensors, Some("first_stage_model"), device)?;
            (encoder, decoder, bytes, ckpt_path.to_path_buf())
        };

        let vae_handle =
            ctx.models
                .register_vae(VaeVariant::Sdxl { encoder, decoder }, vae_bytes, vae_source);

        Ok(vec![
            NodeValue::Model(unet_handle),
            NodeValue::Clip(clip_handle),
            NodeValue::Vae(vae_handle),
        ])
    }

    fn load_sd15(
        &self,
        tensors: &safetensors::SafeTensors<'_>,
        device: &burn::tensor::Device<crate::types::Backend>,
        ctx: &mut ExecutionContext,
        ckpt_path: &std::path::Path,
    ) -> Result<Vec<NodeValue>, NodeError> {
        tracing::info!("detected SD 1.5 checkpoint");

        let unet_bytes = compute_weight_bytes(tensors, model_loader::prefix::UNET, true);
        let clip_bytes = compute_weight_bytes(tensors, model_loader::prefix::CLIP, true);
        let vae_bytes = compute_weight_bytes(tensors, model_loader::prefix::VAE, true);

        let unet = Sd15Unet2D::load(tensors, device)?;
        let unet_handle =
            ctx.models
                .register_unet(UnetVariant::Sd15(unet), unet_bytes, ckpt_path.to_path_buf());

        let clip = crate::clip::Sd15ClipTextEncoder::load(tensors, device)?;
        let vocab = ctx.models_dir().join("clip/vocab.json");
        let merges = ctx.models_dir().join("clip/merges.txt");
        let tokenizer =
            ClipTokenizer::from_files(&vocab, &merges).map_err(|e| NodeError::Execution {
                message: format!("tokenizer: {e}"),
            })?;

        let clip_handle = ctx.models.register_clip(
            ClipVariant::Sd15 {
                encoder: Box::new(clip),
                tokenizer,
            },
            clip_bytes,
            ckpt_path.to_path_buf(),
        );

        let encoder = Sd15VaeEncoder::load(tensors, Some("first_stage_model"), device)?;
        let decoder = Sd15VaeDecoder::load(tensors, Some("first_stage_model"), device)?;
        let vae_handle = ctx.models.register_vae(
            VaeVariant::Sd15 { encoder, decoder },
            vae_bytes,
            PathBuf::from(ckpt_path),
        );

        Ok(vec![
            NodeValue::Model(unet_handle),
            NodeValue::Clip(clip_handle),
            NodeValue::Vae(vae_handle),
        ])
    }
}
