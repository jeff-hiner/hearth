//! Single CLIP transformer layer.

use super::{attention::ClipAttention, mlp::ClipMlp};
use crate::{layers::load_layer_norm, model_loader::LoadError, types::Backend};
use burn::{nn::LayerNorm, prelude::*};
use safetensors::SafeTensors;

/// A single CLIP transformer layer.
///
/// Architecture (pre-norm):
/// ```text
/// x -> LayerNorm -> Attention -> (+x) -> LayerNorm -> MLP -> (+residual) -> out
/// ```
#[derive(Debug)]
pub(crate) struct ClipEncoderLayer<
    const HIDDEN: usize,
    const HEADS: usize,
    const FF: usize,
    const LAYER_NORM_EPS: u64,
    const QUICK_GELU: bool,
> {
    /// Pre-attention layer normalization.
    layer_norm1: LayerNorm<Backend>,
    /// Self-attention block.
    self_attn: ClipAttention<HIDDEN, HEADS>,
    /// Pre-MLP layer normalization.
    layer_norm2: LayerNorm<Backend>,
    /// Feed-forward MLP.
    mlp: ClipMlp<HIDDEN, FF, QUICK_GELU>,
}

impl<
    const HIDDEN: usize,
    const HEADS: usize,
    const FF: usize,
    const LAYER_NORM_EPS: u64,
    const QUICK_GELU: bool,
> ClipEncoderLayer<HIDDEN, HEADS, FF, LAYER_NORM_EPS, QUICK_GELU>
{
    /// Load a transformer layer from safetensors.
    ///
    /// Expects tensors at:
    /// - `{prefix}.layer_norm1.{weight,bias}`
    /// - `{prefix}.self_attn.{q,k,v,out}_proj.{weight,bias}`
    /// - `{prefix}.layer_norm2.{weight,bias}`
    /// - `{prefix}.mlp.fc{1,2}.{weight,bias}`
    pub(crate) fn load(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let eps = f64::from_bits(LAYER_NORM_EPS);

        let layer_norm1 =
            load_layer_norm::<HIDDEN>(tensors, &format!("{prefix}.layer_norm1"), eps, device)?;
        let self_attn = ClipAttention::load(tensors, &format!("{prefix}.self_attn"), device)?;
        let layer_norm2 =
            load_layer_norm::<HIDDEN>(tensors, &format!("{prefix}.layer_norm2"), eps, device)?;
        let mlp = ClipMlp::load(tensors, &format!("{prefix}.mlp"), device)?;

        Ok(Self {
            layer_norm1,
            self_attn,
            layer_norm2,
            mlp,
        })
    }

    /// Forward pass through the layer.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor of shape `[batch, seq_len, hidden]`
    ///
    /// # Returns
    /// Output tensor of shape `[batch, seq_len, hidden]`
    pub(crate) fn forward(&self, hidden_states: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
        // Self-attention with residual
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm1.forward(hidden_states);
        let hidden_states = self.self_attn.forward(hidden_states);
        let hidden_states = residual + hidden_states;

        // MLP with residual
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm2.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);

        residual + hidden_states
    }

    /// Apply LoRA deltas to this encoder layer.
    ///
    /// Delegates to self_attn and mlp.
    pub(crate) fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;
        count += self.self_attn.apply_lora(
            &format!("{prefix}.self_attn"),
            lora_prefix,
            lora,
            strength,
            device,
        )?;
        count += self.mlp.apply_lora(
            &format!("{prefix}.mlp"),
            lora_prefix,
            lora,
            strength,
            device,
        )?;
        Ok(count)
    }
}
