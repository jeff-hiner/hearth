//! `SaveImage` node — saves a decoded image to disk as PNG.

use crate::{
    node::{
        Node, ResolvedInput, SlotDef, ValueType, context::ExecutionContext, error::NodeError,
        value::NodeValue,
    },
    types::Backend,
};
use burn::tensor::Tensor;

/// Saves an image tensor to disk.
///
/// ComfyUI equivalent: `SaveImage`
#[derive(Debug)]
pub struct SaveImage {
    /// Filename prefix (images are numbered sequentially).
    filename_prefix: String,
}

impl SaveImage {
    /// Create a new `SaveImage` node.
    pub fn new(filename_prefix: String) -> Self {
        Self { filename_prefix }
    }
}

impl Node for SaveImage {
    fn type_name(&self) -> &'static str {
        "SaveImage"
    }

    fn inputs(&self) -> &'static [SlotDef] {
        static INPUTS: [SlotDef; 1] = [SlotDef::required("images", ValueType::Image)];
        &INPUTS
    }

    fn outputs(&self) -> &'static [SlotDef] {
        // SaveImage has no outputs (terminal node).
        &[]
    }

    fn execute(
        &self,
        inputs: &[ResolvedInput],
        ctx: &mut ExecutionContext,
    ) -> Result<Vec<NodeValue>, NodeError> {
        let image = match inputs[0].require("SaveImage")? {
            NodeValue::Image(img) => img,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "images",
                    expected: "IMAGE",
                    got: other.type_name(),
                });
            }
        };

        // VAE output is [B, 3, H, W] in [-1, 1]. Convert to PNG.
        let tensor: &Tensor<Backend, 4> = &image.data;
        let [batch, channels, height, width] = tensor.shape().dims();

        for b in 0..batch {
            let single = tensor
                .clone()
                .slice([b..b + 1, 0..channels, 0..height, 0..width]);
            let data: Vec<f32> = single.into_data().convert::<f32>().to_vec().unwrap();

            let hw = height * width;
            let mut rgb = vec![0u8; hw * 3];

            for (i, pixel) in rgb.chunks_exact_mut(3).enumerate() {
                pixel[0] = to_u8(data[i]);
                pixel[1] = to_u8(data[hw + i]);
                pixel[2] = to_u8(data[2 * hw + i]);
            }

            let path = ctx
                .output_dir()
                .join(format!("{}_{:05}.png", self.filename_prefix, b));
            let img =
                image::RgbImage::from_raw(width as u32, height as u32, rgb).ok_or_else(|| {
                    NodeError::Execution {
                        message: "failed to create image buffer".to_string(),
                    }
                })?;
            img.save(&path).map_err(|e| NodeError::Execution {
                message: format!("failed to save {}: {e}", path.display()),
            })?;

            tracing::info!(path = %path.display(), "saved image");
        }

        Ok(vec![])
    }
}

/// Clamp to [-1, 1] and scale to [0, 255].
fn to_u8(val: f32) -> u8 {
    ((val.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8
}
