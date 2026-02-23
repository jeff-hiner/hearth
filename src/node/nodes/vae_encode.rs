//! `VAEEncode` node — encodes a pixel image to latent space.

use crate::{
    node::{
        Node, ResolvedInput, SlotDef, ValueType, context::ExecutionContext, error::NodeError,
        value::NodeValue, variant::VaeVariant,
    },
    types::Latent,
};

/// Encodes pixel images into latent-space samples using a VAE.
///
/// ComfyUI equivalent: `VAEEncode`
///
/// Uses tiled encoding to avoid OOM on attention matrices.
#[derive(Debug, Default)]
pub struct VaeEncode;

impl Node for VaeEncode {
    fn type_name(&self) -> &'static str {
        "VAEEncode"
    }

    fn inputs(&self) -> &'static [SlotDef] {
        static INPUTS: [SlotDef; 2] = [
            SlotDef::required("pixels", ValueType::Image),
            SlotDef::required("vae", ValueType::Vae),
        ];
        &INPUTS
    }

    fn outputs(&self) -> &'static [SlotDef] {
        static OUTPUTS: [SlotDef; 1] = [SlotDef::required("LATENT", ValueType::Latent)];
        &OUTPUTS
    }

    fn execute(
        &self,
        inputs: &[ResolvedInput],
        ctx: &mut ExecutionContext,
    ) -> Result<Vec<NodeValue>, NodeError> {
        let image = match inputs[0].require("VAEEncode")? {
            NodeValue::Image(i) => i,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "pixels",
                    expected: "IMAGE",
                    got: other.type_name(),
                });
            }
        };

        let vae_handle = match inputs[1].require("VAEEncode")? {
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

        // Image comes in as [B, H, W, 3] (BHWC) — permute to [B, 3, H, W] (BCHW)
        let pixel_data = image.data.clone().permute([0, 3, 1, 2]);

        // Use tiled encoding: 64 latent-pixel tiles with 8-pixel overlap
        let latent = match vae {
            VaeVariant::Sd15 { encoder, .. } => encoder.forward_tiled(pixel_data, 64, 8),
            VaeVariant::Sdxl { encoder, .. } => encoder.forward_tiled(pixel_data, 64, 8),
        };

        Ok(vec![NodeValue::Latent(Latent::new(latent))])
    }
}
