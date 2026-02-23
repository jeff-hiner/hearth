//! `EmptyLatentImage` node — creates a zeroed latent tensor.

use crate::{
    node::{
        Node, ResolvedInput, SlotDef, ValueType, context::ExecutionContext, error::NodeError,
        value::NodeValue,
    },
    types::Latent,
};
use burn::tensor::{Distribution, Tensor};

/// Creates an empty (noise-filled) latent image.
///
/// ComfyUI equivalent: `EmptyLatentImage`
#[derive(Debug)]
pub struct EmptyLatentImage {
    /// Image width in pixels.
    width: usize,
    /// Image height in pixels.
    height: usize,
    /// Batch size.
    batch_size: usize,
}

impl EmptyLatentImage {
    /// Create a new `EmptyLatentImage` node.
    pub fn new(width: usize, height: usize, batch_size: usize) -> Self {
        Self {
            width,
            height,
            batch_size,
        }
    }
}

impl Node for EmptyLatentImage {
    fn type_name(&self) -> &'static str {
        "EmptyLatentImage"
    }

    fn inputs(&self) -> &'static [SlotDef] {
        // Width, height, batch_size are widget values, not edges.
        &[]
    }

    fn outputs(&self) -> &'static [SlotDef] {
        static OUTPUTS: [SlotDef; 1] = [SlotDef::required("LATENT", ValueType::Latent)];
        &OUTPUTS
    }

    fn execute(
        &self,
        _inputs: &[ResolvedInput],
        ctx: &mut ExecutionContext,
    ) -> Result<Vec<NodeValue>, NodeError> {
        // Latent space is 1/8 of pixel space, 4 channels
        let latent_h = self.height / 8;
        let latent_w = self.width / 8;

        let samples = Tensor::random(
            [self.batch_size, 4, latent_h, latent_w],
            Distribution::Normal(0.0, 1.0),
            ctx.device(),
        );

        Ok(vec![NodeValue::Latent(Latent::new(samples))])
    }
}
