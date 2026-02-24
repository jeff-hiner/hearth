//! Down block for VAE encoder.

use crate::{
    layers::ResnetBlock2D,
    model_loader::{LoadError, load_tensor_1d, load_tensor_4d},
    types::Backend,
};
use burn::{
    module::Param,
    nn::conv::{Conv2d, Conv2dConfig},
    prelude::*,
};
use safetensors::SafeTensors;

/// Number of ResNet layers per encoder down block (LAYERS_PER_BLOCK = 2).
///
/// Note: the decoder uses LAYERS_PER_BLOCK + 1 = 3 resnets per block,
/// but the encoder uses exactly LAYERS_PER_BLOCK = 2.
const RESNETS_PER_BLOCK: usize = 2;

/// Down encoder block with multiple ResNet layers and optional downsampling.
///
/// Mirror of [`UpDecoderBlock2D`](super::up_block::UpDecoderBlock2D), but with
/// 2 resnets per block (vs 3 in the decoder).
#[derive(Debug)]
pub(super) struct DownEncoderBlock2D<const GROUPS: usize> {
    /// ResNet blocks for feature extraction.
    resnets: [ResnetBlock2D; RESNETS_PER_BLOCK],
    /// Optional stride-2 convolution for spatial downsampling.
    ///
    /// Uses asymmetric padding `(0, 1, 0, 1)` — right and bottom only —
    /// matching the original LDM VAE encoder. The padding is applied
    /// manually in `forward`, so the conv itself has `padding=0`.
    downsample: Option<Conv2d<Backend>>,
}

impl<const GROUPS: usize> DownEncoderBlock2D<GROUPS> {
    /// Load a DownEncoderBlock2D from safetensors weights.
    ///
    /// VAE encoder checkpoint keys:
    /// - `{prefix}.block.{0,1}` for ResNet blocks
    /// - `{prefix}.downsample.conv` for the stride-2 convolution
    pub(super) fn load<const IN_CHANNELS: usize, const OUT_CHANNELS: usize>(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        add_downsample: bool,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let resnet0 = ResnetBlock2D::load::<GROUPS, IN_CHANNELS, OUT_CHANNELS>(
            tensors,
            &format!("{prefix}.block.0"),
            device,
        )?;

        let resnet1 = ResnetBlock2D::load::<GROUPS, OUT_CHANNELS, OUT_CHANNELS>(
            tensors,
            &format!("{prefix}.block.1"),
            device,
        )?;

        let downsample = if add_downsample {
            // VAE encoder uses key "{prefix}.downsample.conv" (not ".op" like UNet)
            let ds_prefix = format!("{prefix}.downsample.conv");
            let weight = load_tensor_4d(tensors, &format!("{ds_prefix}.weight"), device)?;
            let bias = load_tensor_1d(tensors, &format!("{ds_prefix}.bias"), device)?;

            // No conv padding — asymmetric pad applied manually in forward()
            let config =
                Conv2dConfig::new([OUT_CHANNELS, OUT_CHANNELS], [3, 3]).with_stride([2, 2]);

            let mut conv = config.init(device);
            conv.weight = Param::from_tensor(weight);
            conv.bias = Some(Param::from_tensor(bias));

            Some(conv)
        } else {
            None
        };

        Ok(Self {
            resnets: [resnet0, resnet1],
            downsample,
        })
    }

    /// Forward pass through the down block.
    pub(super) fn forward(&self, x: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        let mut hidden = x;

        for resnet in &self.resnets {
            hidden = resnet.forward(hidden);
        }

        if let Some(downsample) = &self.downsample {
            // Asymmetric padding (0, 1, 0, 1): pad right and bottom with zeros.
            // This matches the original LDM VAE encoder which does
            // `F.pad(x, (0,1,0,1))` before a stride-2 conv with padding=0.
            hidden = asymmetric_pad(hidden);
            hidden = downsample.forward(hidden);
        }

        hidden
    }
}

/// Pad a `[B, C, H, W]` tensor with zeros on the right and bottom edges.
///
/// Equivalent to PyTorch's `F.pad(x, (0, 1, 0, 1), mode='constant', value=0)`.
fn asymmetric_pad(x: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
    let [b, c, h, w] = x.shape().dims();
    let device = x.device();
    let mut padded: Tensor<Backend, 4> = Tensor::zeros([b, c, h + 1, w + 1], &device);
    padded = padded.slice_assign([0..b, 0..c, 0..h, 0..w], x);
    padded
}
