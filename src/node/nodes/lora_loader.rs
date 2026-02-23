//! `LoraLoader` node — applies a LoRA to a model and CLIP encoder.

use crate::{
    lora::LoraFile,
    node::{
        Node, ResolvedInput, SlotDef, ValueType, context::ExecutionContext, error::NodeError,
        value::NodeValue,
    },
};
use std::path::PathBuf;

/// Applies a LoRA file to a UNet model and CLIP text encoder.
///
/// ComfyUI equivalent: `LoraLoader`
///
/// Takes ownership of the model and CLIP from the model manager, applies
/// the LoRA deltas in-place, re-registers them, and returns new handles.
#[derive(Debug)]
pub struct LoraLoader {
    /// LoRA filename (relative to models/loras/).
    lora_name: PathBuf,
    /// Strength applied to UNet weights.
    strength_model: f32,
    /// Strength applied to CLIP weights.
    strength_clip: f32,
}

impl LoraLoader {
    /// Create a new LoRA loader node.
    pub fn new(lora_name: PathBuf, strength_model: f32, strength_clip: f32) -> Self {
        Self {
            lora_name,
            strength_model,
            strength_clip,
        }
    }
}

impl Node for LoraLoader {
    fn type_name(&self) -> &'static str {
        "LoraLoader"
    }

    fn inputs(&self) -> &'static [SlotDef] {
        static INPUTS: [SlotDef; 2] = [
            SlotDef::required("MODEL", ValueType::Model),
            SlotDef::required("CLIP", ValueType::Clip),
        ];
        &INPUTS
    }

    fn outputs(&self) -> &'static [SlotDef] {
        static OUTPUTS: [SlotDef; 2] = [
            SlotDef::required("MODEL", ValueType::Model),
            SlotDef::required("CLIP", ValueType::Clip),
        ];
        &OUTPUTS
    }

    fn execute(
        &self,
        inputs: &[ResolvedInput],
        ctx: &mut ExecutionContext,
    ) -> Result<Vec<NodeValue>, NodeError> {
        // Extract handles from inputs
        let model_handle = match inputs[0].require(self.type_name())? {
            NodeValue::Model(h) => *h,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "MODEL",
                    expected: "MODEL",
                    got: other.type_name(),
                });
            }
        };
        let clip_handle = match inputs[1].require(self.type_name())? {
            NodeValue::Clip(h) => *h,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "CLIP",
                    expected: "CLIP",
                    got: other.type_name(),
                });
            }
        };

        // Open LoRA file
        let lora_path = ctx.models_dir().join("loras").join(&self.lora_name);
        tracing::info!(path = %lora_path.display(), "loading LoRA");
        let lora_file = LoraFile::open(&lora_path)?;
        let lora = lora_file.tensors()?;
        let device = ctx.device().clone();

        // Take UNet, apply LoRA, re-register
        let (mut unet, unet_bytes, unet_source) =
            ctx.models
                .take_unet(model_handle)
                .map_err(|e| NodeError::Execution {
                    message: format!("take UNet: {e}"),
                })?;

        let unet_deltas = unet
            .apply_lora(&lora, self.strength_model, &device)
            .map_err(NodeError::Load)?;
        tracing::info!(deltas = unet_deltas, "applied LoRA to UNet");

        let new_model_handle = ctx.models.register_unet(unet, unet_bytes, unet_source);

        // Take CLIP, apply LoRA, re-register
        let (mut clip, clip_bytes, clip_source) =
            ctx.models
                .take_clip(clip_handle)
                .map_err(|e| NodeError::Execution {
                    message: format!("take CLIP: {e}"),
                })?;

        let clip_deltas = clip
            .apply_lora(&lora, self.strength_clip, &device)
            .map_err(NodeError::Load)?;
        tracing::info!(deltas = clip_deltas, "applied LoRA to CLIP");

        let new_clip_handle = ctx.models.register_clip(clip, clip_bytes, clip_source);

        Ok(vec![
            NodeValue::Model(new_model_handle),
            NodeValue::Clip(new_clip_handle),
        ])
    }
}
