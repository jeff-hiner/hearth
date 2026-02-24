//! Self-attention block implementation.

use super::resnet::{load_conv2d, load_group_norm};
use crate::{model_loader::LoadError, types::Backend};
use burn::{
    nn::{GroupNorm, conv::Conv2d},
    prelude::*,
};
use safetensors::SafeTensors;

/// Self-attention block for VAE.
///
/// Architecture:
/// ```text
/// x → GroupNorm → Q,K,V projections → Attention → Output projection → (+x) → out
/// ```
#[derive(Debug)]
pub(crate) struct AttentionBlock {
    norm: GroupNorm<Backend>,
    q: Conv2d<Backend>,
    k: Conv2d<Backend>,
    v: Conv2d<Backend>,
    proj_out: Conv2d<Backend>,
    num_heads: usize,
}

impl AttentionBlock {
    /// Load an AttentionBlock from safetensors weights.
    pub(crate) fn load<const GROUPS: usize, const CHANNELS: usize>(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        num_heads: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let norm = load_group_norm::<GROUPS, CHANNELS>(tensors, &format!("{prefix}.norm"), device)?;
        let q = load_conv2d::<1, CHANNELS, CHANNELS>(tensors, &format!("{prefix}.q"), device)?;
        let k = load_conv2d::<1, CHANNELS, CHANNELS>(tensors, &format!("{prefix}.k"), device)?;
        let v = load_conv2d::<1, CHANNELS, CHANNELS>(tensors, &format!("{prefix}.v"), device)?;
        let proj_out =
            load_conv2d::<1, CHANNELS, CHANNELS>(tensors, &format!("{prefix}.proj_out"), device)?;

        Ok(Self {
            norm,
            q,
            k,
            v,
            proj_out,
            num_heads,
        })
    }

    /// Forward pass through the attention block.
    pub(crate) fn forward(&self, x: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        let [batch, channels, height, width] = x.shape().dims();
        let hidden = self.norm.forward(x.clone());

        let q = self.q.forward(hidden.clone());
        let k = self.k.forward(hidden.clone());
        let v = self.v.forward(hidden);

        // Reshape for attention: [B, C, H, W] -> [B, heads, H*W, head_dim]
        let head_dim = channels / self.num_heads;
        let seq_len = height * width;

        let q = reshape_for_attention(q, self.num_heads, head_dim, seq_len);
        let k = reshape_for_attention(k, self.num_heads, head_dim, seq_len);
        let v = reshape_for_attention(v, self.num_heads, head_dim, seq_len);

        // Scaled dot-product attention (uses flash attention on supported backends)
        let attn_out = burn::tensor::module::attention(q, k, v, None);

        // Reshape back: [B, heads, H*W, head_dim] -> [B, C, H, W]
        let attn_out = attn_out
            .swap_dims(1, 2) // [B, H*W, heads, head_dim]
            .reshape([batch, seq_len, channels])
            .swap_dims(1, 2) // [B, C, H*W]
            .reshape([batch, channels, height, width]);

        let out = self.proj_out.forward(attn_out);

        x + out
    }
}

/// Reshape tensor for multi-head attention.
fn reshape_for_attention(
    x: Tensor<Backend, 4>,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Tensor<Backend, 4> {
    let [batch, _channels, _h, _w] = x.shape().dims();
    x.reshape([batch, num_heads, head_dim, seq_len])
        .swap_dims(2, 3) // [B, heads, seq_len, head_dim]
}
