//! `CLIPTextEncode` node — encodes text into conditioning.

use crate::{
    clip::SdxlConditioning,
    node::{
        Node, ResolvedInput, SlotDef, ValueType, context::ExecutionContext, error::NodeError,
        value::NodeValue, variant::ClipVariant,
    },
    types::{Conditioning, ConditioningEntry, ConditioningMeta, ConditioningValue},
};
use std::collections::HashMap;

/// Encodes a text prompt into conditioning using a CLIP model.
///
/// ComfyUI equivalent: `CLIPTextEncode`
///
/// Takes a CLIP handle and produces conditioning. The actual encoding
/// dispatches on the CLIP variant (SD 1.5 vs SDXL).
#[derive(Debug)]
pub struct ClipTextEncode {
    /// The text prompt to encode.
    text: String,
}

impl ClipTextEncode {
    /// Create a new CLIP text encode node.
    pub fn new(text: String) -> Self {
        Self { text }
    }
}

impl Node for ClipTextEncode {
    fn type_name(&self) -> &'static str {
        "CLIPTextEncode"
    }

    fn inputs(&self) -> &'static [SlotDef] {
        static INPUTS: [SlotDef; 1] = [SlotDef::required("clip", ValueType::Clip)];
        &INPUTS
    }

    fn outputs(&self) -> &'static [SlotDef] {
        static OUTPUTS: [SlotDef; 1] = [SlotDef::required("CONDITIONING", ValueType::Conditioning)];
        &OUTPUTS
    }

    fn execute(
        &self,
        inputs: &[ResolvedInput],
        ctx: &mut ExecutionContext,
    ) -> Result<Vec<NodeValue>, NodeError> {
        let clip_handle = match inputs[0].require("CLIPTextEncode")? {
            NodeValue::Clip(h) => *h,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "clip",
                    expected: "CLIP",
                    got: other.type_name(),
                });
            }
        };

        let device = ctx.device().clone();

        let clip = ctx
            .models
            .borrow_clip(clip_handle)
            .map_err(|e| NodeError::Execution {
                message: format!("failed to borrow CLIP: {e}"),
            })?;

        let conditioning = match clip {
            ClipVariant::Sd15 {
                encoder, tokenizer, ..
            } => {
                let tokens =
                    tokenizer
                        .encode(&self.text, &device)
                        .map_err(|e| NodeError::Execution {
                            message: format!("tokenization failed: {e}"),
                        })?;
                let hidden = encoder.forward(tokens);
                Conditioning::single(hidden)
            }
            ClipVariant::Sdxl {
                clip_l,
                clip_g,
                tokenizer,
            } => {
                // CLIP-L: penultimate layer (layer 10 of 12)
                let tokens_l =
                    tokenizer
                        .encode(&self.text, &device)
                        .map_err(|e| NodeError::Execution {
                            message: format!("tokenization failed: {e}"),
                        })?;
                let clip_l_hidden = clip_l.forward_hidden_layer(tokens_l, Some(10));

                // OpenCLIP-G
                let tokens_g = tokenizer
                    .encode_open_clip(&self.text, &device)
                    .map_err(|e| NodeError::Execution {
                        message: format!("tokenization failed: {e}"),
                    })?;
                let (clip_g_hidden, pooled) = clip_g.forward(tokens_g);

                // Build SDXL conditioning with default metadata
                let sdxl_cond = SdxlConditioning::new(
                    clip_l_hidden,
                    clip_g_hidden,
                    pooled,
                    (1024.0, 1024.0),
                    (0.0, 0.0),
                    (1024.0, 1024.0),
                    &device,
                );

                // Wrap SdxlConditioning into generic Conditioning.
                // Store the y vector in metadata so KSampler can reconstruct.
                wrap_sdxl_conditioning(sdxl_cond)
            }
        };

        Ok(vec![NodeValue::Conditioning(conditioning)])
    }
}

/// Wrap an [`SdxlConditioning`] into the generic [`Conditioning`] type.
///
/// The `y` (pooled) vector is stored in metadata under the key `"y"` so that
/// KSampler can reconstruct the `SdxlConditioning` from a `Conditioning`.
fn wrap_sdxl_conditioning(cond: SdxlConditioning) -> Conditioning {
    let mut meta: ConditioningMeta = HashMap::new();
    meta.insert("y".to_string(), ConditioningValue::Tensor(cond.y));

    Conditioning::new(vec![ConditioningEntry {
        embedding: cond.context,
        meta,
    }])
}
