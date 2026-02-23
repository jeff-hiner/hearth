//! `ControlNetLoader` node — loads a ControlNet model from a safetensors file.

use crate::{
    controlnet::{Sd15ControlNet, SdxlControlNet},
    model_loader::SafeTensorsFile,
    model_manager::compute_weight_bytes,
    node::{
        Node, ResolvedInput, SlotDef, ValueType, context::ExecutionContext, error::NodeError,
        value::NodeValue, variant::ControlNetVariant,
    },
};
use std::path::PathBuf;

/// Loads a ControlNet model and returns a handle.
///
/// ComfyUI equivalent: `ControlNetLoader`
///
/// Detects whether the model is SD 1.5 or SDXL from tensor names,
/// loads it, registers with the model manager, and returns a handle.
#[derive(Debug)]
pub(crate) struct ControlNetLoader {
    /// ControlNet filename (relative to models/controlnet/).
    control_net_name: PathBuf,
}

impl ControlNetLoader {
    /// Create a new ControlNet loader node.
    pub(crate) fn new(control_net_name: PathBuf) -> Self {
        Self { control_net_name }
    }
}

impl Node for ControlNetLoader {
    fn type_name(&self) -> &'static str {
        "ControlNetLoader"
    }

    fn inputs(&self) -> &'static [SlotDef] {
        &[]
    }

    fn outputs(&self) -> &'static [SlotDef] {
        static OUTPUTS: [SlotDef; 1] = [SlotDef::required("CONTROL_NET", ValueType::ControlNet)];
        &OUTPUTS
    }

    fn execute(
        &self,
        _inputs: &[ResolvedInput],
        ctx: &mut ExecutionContext,
    ) -> Result<Vec<NodeValue>, NodeError> {
        let cn_path = ctx
            .models_dir()
            .join("controlnet")
            .join(&self.control_net_name);
        tracing::info!(path = %cn_path.display(), "loading ControlNet");

        let file = SafeTensorsFile::open(&cn_path)?;
        let tensors = file.tensors()?;
        let device = ctx.device().clone();

        // Detect SDXL vs SD 1.5 by checking for label_emb (SDXL-only).
        let is_sdxl = tensors
            .names()
            .iter()
            .any(|n| n.starts_with("control_model.label_emb."));

        // Detect hint channels from the first conv of the hint encoder.
        let hint_channels = tensors
            .tensor("control_model.input_hint_block.0.weight")
            .map(|t| t.shape()[1])
            .unwrap_or(3);

        let weight_bytes = compute_weight_bytes(&tensors, "control_model", true);

        let variant = if is_sdxl {
            tracing::info!("detected SDXL ControlNet");
            let model = SdxlControlNet::load(&tensors, hint_channels, &device)?;
            ControlNetVariant::Sdxl(model)
        } else {
            tracing::info!("detected SD 1.5 ControlNet");
            let model = Sd15ControlNet::load(&tensors, hint_channels, &device)?;
            ControlNetVariant::Sd15(model)
        };

        let handle = ctx
            .models
            .register_controlnet(variant, weight_bytes, cn_path.to_path_buf());

        Ok(vec![NodeValue::ControlNet(handle)])
    }
}
