//! Up block for VAE decoder.

use crate::{
    layers::{ResnetBlock2D, Upsample2D},
    model_loader::LoadError,
    types::Backend,
};
use burn::prelude::*;
use safetensors::SafeTensors;

/// Number of ResNet layers per up block (LAYERS_PER_BLOCK + 1 = 3).
const RESNETS_PER_BLOCK: usize = 3;

/// Up decoder block with multiple ResNet layers and optional upsampling.
#[derive(Debug)]
pub(super) struct UpDecoderBlock2D<const GROUPS: usize> {
    resnets: [ResnetBlock2D; RESNETS_PER_BLOCK],
    upsample: Option<Upsample2D>,
}

impl<const GROUPS: usize> UpDecoderBlock2D<GROUPS> {
    /// Load an UpDecoderBlock2D from safetensors weights.
    pub(super) fn load<const IN_CHANNELS: usize, const OUT_CHANNELS: usize>(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        add_upsample: bool,
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

        let resnet2 = ResnetBlock2D::load::<GROUPS, OUT_CHANNELS, OUT_CHANNELS>(
            tensors,
            &format!("{prefix}.block.2"),
            device,
        )?;

        let upsample = if add_upsample {
            Some(Upsample2D::load::<OUT_CHANNELS>(
                tensors,
                &format!("{prefix}.upsample"),
                device,
            )?)
        } else {
            None
        };

        Ok(Self {
            resnets: [resnet0, resnet1, resnet2],
            upsample,
        })
    }

    /// Forward pass through the up block.
    pub(super) fn forward(&self, x: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        let mut hidden = x;

        for resnet in &self.resnets {
            hidden = resnet.forward(hidden);
        }

        if let Some(upsample) = &self.upsample {
            hidden = upsample.forward(hidden);
        }

        hidden
    }
}
