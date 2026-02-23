//! `VAEDecode` node — decodes a latent tensor to an image.

use crate::{
    node::{
        Node, ResolvedInput, SlotDef, ValueType, context::ExecutionContext, error::NodeError,
        value::NodeValue, variant::VaeVariant,
    },
    types::Image,
};

/// Decodes latent-space samples into pixel images using a VAE.
///
/// ComfyUI equivalent: `VAEDecode`
///
/// Uses tiled decoding to avoid OOM on attention matrices.
#[derive(Debug, Default)]
pub struct VaeDecode;

impl Node for VaeDecode {
    fn type_name(&self) -> &'static str {
        "VAEDecode"
    }

    fn inputs(&self) -> &'static [SlotDef] {
        static INPUTS: [SlotDef; 2] = [
            SlotDef::required("samples", ValueType::Latent),
            SlotDef::required("vae", ValueType::Vae),
        ];
        &INPUTS
    }

    fn outputs(&self) -> &'static [SlotDef] {
        static OUTPUTS: [SlotDef; 1] = [SlotDef::required("IMAGE", ValueType::Image)];
        &OUTPUTS
    }

    fn execute(
        &self,
        inputs: &[ResolvedInput],
        ctx: &mut ExecutionContext,
    ) -> Result<Vec<NodeValue>, NodeError> {
        let latent = match inputs[0].require("VAEDecode")? {
            NodeValue::Latent(l) => l,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "samples",
                    expected: "LATENT",
                    got: other.type_name(),
                });
            }
        };

        let vae_handle = match inputs[1].require("VAEDecode")? {
            NodeValue::Vae(h) => *h,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "vae",
                    expected: "VAE",
                    got: other.type_name(),
                });
            }
        };

        let vae = ctx
            .models
            .borrow_vae(vae_handle)
            .map_err(|e| NodeError::Execution {
                message: format!("failed to borrow VAE: {e}"),
            })?;

        // Use tiled decoding: 64 latent-pixel tiles with 8-pixel overlap
        // to keep attention matrices manageable.
        let decoded = match vae {
            VaeVariant::Sd15 { decoder, .. } => {
                decoder.forward_tiled(latent.samples.clone(), 64, 8)
            }
            VaeVariant::Sdxl { decoder, .. } => {
                decoder.forward_tiled(latent.samples.clone(), 64, 8)
            }
        };

        Ok(vec![NodeValue::Image(Image::new(decoded))])
    }
}
