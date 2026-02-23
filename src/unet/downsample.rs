//! Downsampling layers for UNet.

use crate::{
    lora::{apply_lora_conv2d, lora_key_base},
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
};
use safetensors::SafeTensors;

/// 2D downsampling via strided convolution.
///
/// Reduces spatial dimensions by half using a stride-2 convolution.
#[derive(Debug)]
pub(crate) struct Downsample2D {
    /// Strided convolution for downsampling.
    conv: Conv2d<Backend>,
}

impl Downsample2D {
    /// Load from safetensors weights.
    ///
    /// SD checkpoint key: `{prefix}.op`
    pub(crate) fn load(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        channels: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let weight = load_tensor_4d(tensors, &format!("{prefix}.op.weight"), device)?;
        let bias = load_tensor_1d(tensors, &format!("{prefix}.op.bias"), device)?;

        // Stride-2 convolution with padding=1
        let config = Conv2dConfig::new([channels, channels], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1));

        let mut conv = config.init(device);
        conv.weight = Param::from_tensor(weight);
        conv.bias = Some(Param::from_tensor(bias));

        Ok(Self { conv })
    }

    /// Forward pass: halves spatial dimensions.
    pub(crate) fn forward(&self, x: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        self.conv.forward(x)
    }

    /// Apply LoRA delta to the downsample convolution.
    ///
    /// Target: `op` (the strided conv)
    pub(crate) fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let key = lora_key_base(lora_prefix, &format!("{prefix}.op"));
        if apply_lora_conv2d(&mut self.conv, &key, lora, strength, device)? {
            Ok(1)
        } else {
            Ok(0)
        }
    }
}
