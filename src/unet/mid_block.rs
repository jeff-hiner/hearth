//! UNet middle block with cross-attention.

use super::{resnet::ResnetBlock2D, transformer::SpatialTransformer};
use crate::{model_loader::LoadError, types::Backend};
use burn::prelude::*;
use safetensors::SafeTensors;

/// UNet middle block: ResNet -> Attention -> ResNet.
///
/// SD checkpoint structure:
/// - `{prefix}.0` - First ResNet block
/// - `{prefix}.1` - SpatialTransformer (attention)
/// - `{prefix}.2` - Second ResNet block
#[derive(Debug)]
pub(crate) struct UNetMidBlock {
    /// First ResNet block.
    pub(crate) resnet1: ResnetBlock2D,
    /// Cross-attention transformer.
    pub(crate) attn: SpatialTransformer,
    /// Second ResNet block.
    pub(crate) resnet2: ResnetBlock2D,
}

impl UNetMidBlock {
    /// Load from safetensors weights.
    #[expect(
        clippy::too_many_arguments,
        reason = "loading config needs all parameters"
    )]
    pub(crate) fn load(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        channels: usize,
        time_embed_dim: usize,
        context_dim: usize,
        num_heads: usize,
        head_dim: usize,
        depth: usize,
        groups: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let resnet1 = ResnetBlock2D::load(
            tensors,
            &format!("{prefix}.0"),
            channels,
            channels,
            time_embed_dim,
            groups,
            device,
        )?;

        let attn = SpatialTransformer::load(
            tensors,
            &format!("{prefix}.1"),
            channels,
            num_heads,
            head_dim,
            context_dim,
            depth,
            groups,
            device,
        )?;

        let resnet2 = ResnetBlock2D::load(
            tensors,
            &format!("{prefix}.2"),
            channels,
            channels,
            time_embed_dim,
            groups,
            device,
        )?;

        Ok(Self {
            resnet1,
            attn,
            resnet2,
        })
    }

    /// Forward pass.
    pub(crate) fn forward(
        &self,
        x: Tensor<Backend, 4>,
        time_emb: Option<&Tensor<Backend, 2>>,
        context: Option<&Tensor<Backend, 3>>,
    ) -> Tensor<Backend, 4> {
        let x = self.resnet1.forward(x, time_emb);
        let x = self.attn.forward(x, context);
        self.resnet2.forward(x, time_emb)
    }

    /// Apply LoRA deltas to the mid block.
    ///
    /// Delegates to resnet1 (`.0`), attn (`.1`), resnet2 (`.2`).
    pub(crate) fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;
        count +=
            self.resnet1
                .apply_lora(&format!("{prefix}.0"), lora_prefix, lora, strength, device)?;
        count +=
            self.attn
                .apply_lora(&format!("{prefix}.1"), lora_prefix, lora, strength, device)?;
        count +=
            self.resnet2
                .apply_lora(&format!("{prefix}.2"), lora_prefix, lora, strength, device)?;
        Ok(count)
    }
}
