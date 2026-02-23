//! Upsampling layer implementation.

use super::resnet::load_conv2d;
use crate::{model_loader::LoadError, types::Backend};
use burn::{
    nn::conv::Conv2d,
    prelude::*,
    tensor::{
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

/// 2D upsampling layer using nearest-neighbor interpolation followed by convolution.
#[derive(Debug)]
pub(crate) struct Upsample2D {
    conv: Conv2d<Backend>,
}

impl Upsample2D {
    /// Load an Upsample2D from safetensors weights.
    pub(crate) fn load<const CHANNELS: usize>(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let conv =
            load_conv2d::<3, CHANNELS, CHANNELS>(tensors, &format!("{prefix}.conv"), device)?;
        Ok(Self { conv })
    }

    /// Forward pass: 2x nearest upsample then conv.
    pub(crate) fn forward(&self, x: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        let [_batch, _channels, height, width] = x.shape().dims();

        // Nearest-neighbor 2x upsample
        let upsampled = interpolate(
            x,
            [height * 2, width * 2],
            InterpolateOptions::new(InterpolateMode::Nearest),
        );

        self.conv.forward(upsampled)
    }
}
