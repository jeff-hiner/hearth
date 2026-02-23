//! ResNet block implementation.

use crate::{
    model_loader::{LoadError, load_tensor_1d, load_tensor_4d},
    types::Backend,
};
use burn::{
    module::Param,
    nn::{
        GroupNorm, GroupNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::silu,
};

/// A 2D ResNet block with group normalization.
///
/// Architecture:
/// ```text
/// x → GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv → (+skip) → out
/// ```
#[derive(Debug)]
pub(crate) struct ResnetBlock2D {
    norm1: GroupNorm<Backend>,
    conv1: Conv2d<Backend>,
    norm2: GroupNorm<Backend>,
    conv2: Conv2d<Backend>,
    /// Optional skip connection conv when in_channels != out_channels.
    skip_conv: Option<Conv2d<Backend>>,
}

impl ResnetBlock2D {
    /// Load a ResnetBlock2D from safetensors weights.
    pub(crate) fn load<const GROUPS: usize, const IN_CHANNELS: usize, const OUT_CHANNELS: usize>(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let norm1 =
            load_group_norm::<GROUPS, IN_CHANNELS>(tensors, &format!("{prefix}.norm1"), device)?;
        let conv1 = load_conv2d::<3, IN_CHANNELS, OUT_CHANNELS>(
            tensors,
            &format!("{prefix}.conv1"),
            device,
        )?;
        let norm2 =
            load_group_norm::<GROUPS, OUT_CHANNELS>(tensors, &format!("{prefix}.norm2"), device)?;
        let conv2 = load_conv2d::<3, OUT_CHANNELS, OUT_CHANNELS>(
            tensors,
            &format!("{prefix}.conv2"),
            device,
        )?;

        let skip_conv = if IN_CHANNELS != OUT_CHANNELS {
            Some(load_conv2d::<1, IN_CHANNELS, OUT_CHANNELS>(
                tensors,
                &format!("{prefix}.nin_shortcut"),
                device,
            )?)
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            skip_conv,
        })
    }

    /// Forward pass through the resnet block.
    pub(crate) fn forward(&self, x: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        let hidden = self.norm1.forward(x.clone());
        let hidden = silu(hidden);
        let hidden = self.conv1.forward(hidden);

        let hidden = self.norm2.forward(hidden);
        let hidden = silu(hidden);
        let hidden = self.conv2.forward(hidden);

        let skip = match &self.skip_conv {
            Some(conv) => conv.forward(x),
            None => x,
        };

        hidden + skip
    }
}

/// Load a GroupNorm layer from safetensors.
pub(crate) fn load_group_norm<const GROUPS: usize, const NUM_CHANNELS: usize>(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    device: &Device<Backend>,
) -> Result<GroupNorm<Backend>, LoadError> {
    let weight = load_tensor_1d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let config = GroupNormConfig::new(GROUPS, NUM_CHANNELS);
    let mut norm = config.init(device);

    norm.gamma = Some(Param::from_tensor(weight));
    norm.beta = Some(Param::from_tensor(bias));

    Ok(norm)
}

/// Load a Conv2d layer from safetensors.
///
/// `KERNEL` is the kernel size (typically 1 or 3).
pub(crate) fn load_conv2d<
    const KERNEL: usize,
    const IN_CHANNELS: usize,
    const OUT_CHANNELS: usize,
>(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    device: &Device<Backend>,
) -> Result<Conv2d<Backend>, LoadError> {
    let weight = load_tensor_4d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let config = Conv2dConfig::new([IN_CHANNELS, OUT_CHANNELS], [KERNEL, KERNEL])
        .with_padding(PaddingConfig2d::Explicit(KERNEL / 2, KERNEL / 2));

    let mut conv = config.init(device);
    conv.weight = Param::from_tensor(weight);
    conv.bias = Some(Param::from_tensor(bias));

    Ok(conv)
}
