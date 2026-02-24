//! CLIP multi-head self-attention with causal masking.

use crate::{
    layers::load_linear,
    lora::{apply_lora_linear, lora_key_base},
    model_loader::LoadError,
    types::Backend,
};
use burn::{nn::Linear, prelude::*};
use safetensors::SafeTensors;

/// Multi-head self-attention layer for CLIP.
///
/// Uses causal masking to prevent attending to future tokens.
#[derive(Debug)]
pub(crate) struct ClipAttention<const HIDDEN: usize, const HEADS: usize> {
    /// Query projection.
    q_proj: Linear<Backend>,
    /// Key projection.
    k_proj: Linear<Backend>,
    /// Value projection.
    v_proj: Linear<Backend>,
    /// Output projection.
    out_proj: Linear<Backend>,
}

impl<const HIDDEN: usize, const HEADS: usize> ClipAttention<HIDDEN, HEADS> {
    /// Dimension per attention head.
    const HEAD_DIM: usize = HIDDEN / HEADS;

    /// Load attention weights from safetensors.
    ///
    /// Expects tensors at:
    /// - `{prefix}.q_proj.{weight,bias}`
    /// - `{prefix}.k_proj.{weight,bias}`
    /// - `{prefix}.v_proj.{weight,bias}`
    /// - `{prefix}.out_proj.{weight,bias}`
    pub(crate) fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let q_proj = load_linear::<HIDDEN, HIDDEN>(tensors, &format!("{prefix}.q_proj"), device)?;
        let k_proj = load_linear::<HIDDEN, HIDDEN>(tensors, &format!("{prefix}.k_proj"), device)?;
        let v_proj = load_linear::<HIDDEN, HIDDEN>(tensors, &format!("{prefix}.v_proj"), device)?;
        let out_proj =
            load_linear::<HIDDEN, HIDDEN>(tensors, &format!("{prefix}.out_proj"), device)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    /// Forward pass with causal self-attention.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor of shape `[batch, seq_len, hidden]`
    ///
    /// # Returns
    /// Output tensor of shape `[batch, seq_len, hidden]`
    pub(crate) fn forward(&self, hidden_states: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
        let [batch, seq_len, _hidden] = hidden_states.shape().dims();
        let device = hidden_states.device();

        // Project to Q, K, V
        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        // Reshape for multi-head attention: [batch, seq_len, hidden] -> [batch, heads, seq_len, head_dim]
        let q = self.reshape_for_attention(q, batch, seq_len);
        let k = self.reshape_for_attention(k, batch, seq_len);
        let v = self.reshape_for_attention(v, batch, seq_len);

        // Scaled dot-product attention
        let scale = (Self::HEAD_DIM as f32).sqrt();
        let attn_weights: Tensor<Backend, 4> = q.matmul(k.transpose()) / scale;

        // Apply causal mask: mask out future positions with -inf
        let causal_mask = create_causal_mask(seq_len, &device);
        let attn_weights = attn_weights
            + causal_mask
                .unsqueeze::<4>()
                .expand([batch, HEADS, seq_len, seq_len]);

        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v);

        // Reshape back: [batch, heads, seq_len, head_dim] -> [batch, seq_len, hidden]
        let attn_output = attn_output
            .swap_dims(1, 2) // [batch, seq_len, heads, head_dim]
            .reshape([batch, seq_len, HIDDEN]);

        self.out_proj.forward(attn_output)
    }

    /// Apply LoRA deltas to attention projections.
    ///
    /// Targets: `q_proj`, `k_proj`, `v_proj`, `out_proj`
    pub(crate) fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;

        for (field, suffix) in [
            (&mut self.q_proj, "q_proj"),
            (&mut self.k_proj, "k_proj"),
            (&mut self.v_proj, "v_proj"),
            (&mut self.out_proj, "out_proj"),
        ] {
            let key = lora_key_base(lora_prefix, &format!("{prefix}.{suffix}"));
            if apply_lora_linear(field, &key, lora, strength, device)? {
                count += 1;
            }
        }

        Ok(count)
    }

    /// Reshape tensor for multi-head attention.
    fn reshape_for_attention(
        &self,
        x: Tensor<Backend, 3>,
        batch: usize,
        seq_len: usize,
    ) -> Tensor<Backend, 4> {
        x.reshape([batch, seq_len, HEADS, Self::HEAD_DIM])
            .swap_dims(1, 2) // [batch, heads, seq_len, head_dim]
    }
}

/// Create a causal attention mask.
///
/// Returns a mask where positions that should be masked (future tokens)
/// contain negative infinity, and valid positions contain 0.
fn create_causal_mask(seq_len: usize, device: &Device<Backend>) -> Tensor<Backend, 2> {
    // Create upper triangular mask (1s above diagonal, 0s on and below)
    let ones: Tensor<Backend, 2> = Tensor::ones([seq_len, seq_len], device);
    let mask = ones.triu(1);

    // Convert to attention mask: 0 -> 0.0, 1 -> -inf
    let neg_inf: Tensor<Backend, 2> = Tensor::full([seq_len, seq_len], f32::NEG_INFINITY, device);
    let zeros: Tensor<Backend, 2> = Tensor::zeros([seq_len, seq_len], device);

    // Where mask is 1 (future positions), use -inf; otherwise 0
    zeros.mask_where(mask.bool(), neg_inf)
}
