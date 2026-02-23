//! Mid block for VAE decoder.

use crate::{
    layers::{AttentionBlock, ResnetBlock2D},
    model_loader::LoadError,
    types::Backend,
};
use burn::prelude::*;

/// Mid block containing ResNet + Attention + ResNet.
#[derive(Debug)]
pub(super) struct UNetMidBlock2D<const GROUPS: usize, const CHANNELS: usize> {
    resnet1: ResnetBlock2D,
    attn: AttentionBlock,
    resnet2: ResnetBlock2D,
}

impl<const GROUPS: usize, const CHANNELS: usize> UNetMidBlock2D<GROUPS, CHANNELS> {
    /// Load a UNetMidBlock2D from safetensors weights.
    pub(super) fn load(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let resnet1 = ResnetBlock2D::load::<GROUPS, CHANNELS, CHANNELS>(
            tensors,
            &format!("{prefix}.block_1"),
            device,
        )?;

        // VAE uses single-head attention
        let attn = AttentionBlock::load::<GROUPS, CHANNELS>(
            tensors,
            &format!("{prefix}.attn_1"),
            1, // single head
            device,
        )?;

        let resnet2 = ResnetBlock2D::load::<GROUPS, CHANNELS, CHANNELS>(
            tensors,
            &format!("{prefix}.block_2"),
            device,
        )?;

        Ok(Self {
            resnet1,
            attn,
            resnet2,
        })
    }

    /// Forward pass through the mid block.
    pub(super) fn forward(&self, x: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        let hidden = self.resnet1.forward(x);
        let hidden = self.attn.forward(hidden);
        self.resnet2.forward(hidden)
    }
}
