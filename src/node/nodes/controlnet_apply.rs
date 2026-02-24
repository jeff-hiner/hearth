//! `ControlNetApply` node — attaches a ControlNet to conditioning metadata.

use crate::{
    node::{
        Node, ResolvedInput, SlotDef, ValueType, context::ExecutionContext, error::NodeError,
        value::NodeValue,
    },
    types::{ConditioningValue, ControlNetRef},
};

/// Attaches a ControlNet + hint image to conditioning.
///
/// ComfyUI equivalent: `ControlNetApply`
///
/// Clones the input conditioning and appends a [`ControlNetRef`] to the
/// `"control_net"` metadata key (creating the key if absent). Multiple
/// `ControlNetApply` nodes can be chained to stack ControlNets.
#[derive(Debug)]
pub(crate) struct ControlNetApply {
    /// Strength multiplier (0.0 = no effect, 1.0 = full effect).
    strength: f32,
    /// Step fraction at which ControlNet activates (0.0 = first step).
    start_percent: f32,
    /// Step fraction at which ControlNet deactivates (1.0 = last step).
    end_percent: f32,
}

impl ControlNetApply {
    /// Create a new ControlNetApply node with guidance range.
    pub(crate) fn new(strength: f32, start_percent: f32, end_percent: f32) -> Self {
        Self {
            strength,
            start_percent,
            end_percent,
        }
    }
}

impl Node for ControlNetApply {
    fn type_name(&self) -> &'static str {
        "ControlNetApply"
    }

    fn inputs(&self) -> &'static [SlotDef] {
        static INPUTS: [SlotDef; 3] = [
            SlotDef::required("conditioning", ValueType::Conditioning),
            SlotDef::required("control_net", ValueType::ControlNet),
            SlotDef::required("image", ValueType::Image),
        ];
        &INPUTS
    }

    fn outputs(&self) -> &'static [SlotDef] {
        static OUTPUTS: [SlotDef; 1] = [SlotDef::required("CONDITIONING", ValueType::Conditioning)];
        &OUTPUTS
    }

    fn execute(
        &self,
        inputs: &[ResolvedInput],
        _ctx: &mut ExecutionContext,
    ) -> Result<Vec<NodeValue>, NodeError> {
        let conditioning = match inputs[0].require("ControlNetApply")? {
            NodeValue::Conditioning(c) => c,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "conditioning",
                    expected: "CONDITIONING",
                    got: other.type_name(),
                });
            }
        };

        let cn_handle = match inputs[1].require("ControlNetApply")? {
            NodeValue::ControlNet(h) => *h,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "control_net",
                    expected: "CONTROL_NET",
                    got: other.type_name(),
                });
            }
        };

        let image = match inputs[2].require("ControlNetApply")? {
            NodeValue::Image(img) => img,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "image",
                    expected: "IMAGE",
                    got: other.type_name(),
                });
            }
        };

        // Convert IMAGE [B, H, W, 3] to hint [B, 3, H, W] via permute
        let hint = image.data.clone().permute([0, 3, 1, 2]);

        // Clone conditioning and append ControlNet ref
        let mut cond = conditioning.clone();
        let cn_ref = ControlNetRef {
            handle: cn_handle,
            hint,
            strength: self.strength,
            start_percent: self.start_percent,
            end_percent: self.end_percent,
        };

        for entry in &mut cond.entries {
            let stack = entry
                .meta
                .entry("control_net".to_string())
                .or_insert_with(|| ConditioningValue::ControlNetStack(Vec::new()));

            match stack {
                ConditioningValue::ControlNetStack(v) => v.push(cn_ref.clone()),
                _ => {
                    return Err(NodeError::Execution {
                        message: "conditioning 'control_net' key has unexpected type".to_string(),
                    });
                }
            }
        }

        Ok(vec![NodeValue::Conditioning(cond)])
    }
}
