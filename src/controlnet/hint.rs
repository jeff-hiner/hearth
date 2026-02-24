//! ControlNet hint (conditioning image) encoder.
//!
//! A small convolutional stack that converts a pixel-space conditioning image
//! (e.g., a depth map) into latent-space dimensions matching the UNet's conv_in output.
//!
//! Architecture:
//! ```text
//! Conv2d(hint_ch, 16, 3, pad=1) → SiLU
//! Conv2d(16, 16, 3, pad=1)      → SiLU
//! Conv2d(16, 32, 3, stride=2)   → SiLU
//! Conv2d(32, 32, 3, pad=1)      → SiLU
//! Conv2d(32, 96, 3, stride=2)   → SiLU
//! Conv2d(96, 96, 3, pad=1)      → SiLU
//! Conv2d(96, 256, 3, stride=2)  → SiLU
//! Conv2d(256, base_channels, 3, pad=1)
//! ```
//!
//! Loaded from `control_model.input_hint_block.{0,2,4,6,8,10,12,14}` (even indices
//! are convolutions; odd indices are SiLU activations with no weights).

use crate::{
    model_loader::{LoadError, load_tensor_1d, load_tensor_4d},
    types::Backend,
};
use burn::{
    module::Param,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::silu,
};
use safetensors::SafeTensors;

/// Hint encoder for ControlNet conditioning images.
///
/// Converts a pixel-space image `[B, hint_ch, H, W]` into a tensor with the
/// same spatial dimensions and channel count as the UNet's conv_in output.
#[derive(Debug)]
pub(crate) struct HintEncoder {
    /// The 8 convolution layers (SiLU activations are inline, not stored).
    convs: [Conv2d<Backend>; 8],
}

impl HintEncoder {
    /// Load from safetensors.
    ///
    /// Weights are at `{prefix}.{0,2,4,6,8,10,12,14}.{weight,bias}`.
    pub(crate) fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        hint_channels: usize,
        base_channels: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        // Conv indices in the checkpoint: 0, 2, 4, 6, 8, 10, 12, 14
        // (odd indices are SiLU layers with no weights)
        let conv_specs: [(usize, usize, usize, usize); 8] = [
            // (in_ch, out_ch, stride, checkpoint_idx)
            (hint_channels, 16, 1, 0),
            (16, 16, 1, 2),
            (16, 32, 2, 4),
            (32, 32, 1, 6),
            (32, 96, 2, 8),
            (96, 96, 1, 10),
            (96, 256, 2, 12),
            (256, base_channels, 1, 14),
        ];

        let convs: Vec<Conv2d<Backend>> = conv_specs
            .iter()
            .map(|&(in_ch, out_ch, stride, idx)| {
                load_conv2d(
                    tensors,
                    &format!("{prefix}.{idx}"),
                    3,
                    in_ch,
                    out_ch,
                    stride,
                    device,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let convs: [Conv2d<Backend>; 8] = convs
            .try_into()
            .expect("should have exactly 8 convolutions");

        Ok(Self { convs })
    }

    /// Forward pass: conditioning image → latent-space features.
    ///
    /// # Arguments
    /// * `hint` - Conditioning image `[B, hint_ch, H, W]` in [0, 1]
    ///
    /// # Returns
    /// Features `[B, base_channels, H/8, W/8]` matching UNet conv_in output.
    pub(crate) fn forward(&self, hint: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        let mut x = hint;
        // First 7 convs have SiLU activation; last conv has no activation
        for (i, conv) in self.convs.iter().enumerate() {
            x = conv.forward(x);
            if i < 7 {
                x = silu(x);
            }
        }
        x
    }
}

/// Load a Conv2d layer with stride and padding.
fn load_conv2d(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    device: &Device<Backend>,
) -> Result<Conv2d<Backend>, LoadError> {
    let weight = load_tensor_4d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let padding = kernel_size / 2;
    let config = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(padding, padding));

    let mut conv = config.init(device);
    conv.weight = Param::from_tensor(weight);
    conv.bias = Some(Param::from_tensor(bias));

    Ok(conv)
}
