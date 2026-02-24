//! Attention mechanisms for UNet cross-attention.

use crate::{
    lora::{apply_lora_linear, lora_key_base},
    model_loader::{LoadError, load_tensor_1d, load_tensor_2d},
    types::Backend,
};
use burn::{
    module::Param,
    nn::{Linear, LinearConfig},
    prelude::*,
};
use safetensors::SafeTensors;

/// Cross-attention layer.
///
/// Computes attention between queries (from image features) and
/// keys/values (from text conditioning or self).
#[derive(Debug)]
pub(crate) struct CrossAttention {
    /// Query projection.
    to_q: Linear<Backend>,
    /// Key projection.
    to_k: Linear<Backend>,
    /// Value projection.
    to_v: Linear<Backend>,
    /// Output projection.
    to_out: Linear<Backend>,
    /// Number of attention heads.
    num_heads: usize,
    /// Dimension per head.
    head_dim: usize,
}

impl CrossAttention {
    /// Load from safetensors weights.
    ///
    /// SD checkpoint keys:
    /// - `{prefix}.to_q.weight`
    /// - `{prefix}.to_k.weight`
    /// - `{prefix}.to_v.weight`
    /// - `{prefix}.to_out.0.weight`, `{prefix}.to_out.0.bias`
    pub(crate) fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        query_dim: usize,
        context_dim: usize,
        num_heads: usize,
        head_dim: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let inner_dim = num_heads * head_dim;

        // Q, K, V projections (note: SD stores these without bias)
        let to_q = load_linear_no_bias(
            tensors,
            &format!("{prefix}.to_q"),
            query_dim,
            inner_dim,
            device,
        )?;
        let to_k = load_linear_no_bias(
            tensors,
            &format!("{prefix}.to_k"),
            context_dim,
            inner_dim,
            device,
        )?;
        let to_v = load_linear_no_bias(
            tensors,
            &format!("{prefix}.to_v"),
            context_dim,
            inner_dim,
            device,
        )?;

        // Output projection (has bias)
        let to_out = load_linear_with_bias(
            tensors,
            &format!("{prefix}.to_out.0"),
            inner_dim,
            query_dim,
            device,
        )?;

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Query input `[B, seq_len, query_dim]`
    /// * `context` - Key/value context `[B, context_len, context_dim]` (None = self-attention)
    pub(crate) fn forward(
        &self,
        x: Tensor<Backend, 3>,
        context: Option<&Tensor<Backend, 3>>,
    ) -> Tensor<Backend, 3> {
        let [batch, seq_len, _] = x.shape().dims();
        let context_owned;
        let context = match context {
            Some(ctx) => ctx,
            None => {
                context_owned = x.clone();
                &context_owned
            }
        };
        let [_, context_len, _] = context.shape().dims();

        // Project to Q, K, V
        let q = self.to_q.forward(x);
        let k = self.to_k.forward(context.clone());
        let v = self.to_v.forward(context.clone());

        // Reshape for multi-head attention: [B, seq, heads*head_dim] -> [B, heads, seq, head_dim]
        let q = self.reshape_heads(q, batch, seq_len);
        let k = self.reshape_heads(k, batch, context_len);
        let v = self.reshape_heads(v, batch, context_len);

        // Scaled dot-product attention (uses flash attention on supported backends)
        let attn_out = burn::tensor::module::attention(q, k, v, None);

        // Reshape back: [B, heads, seq, head_dim] -> [B, seq, heads*head_dim]
        let attn_out = attn_out
            .swap_dims(1, 2) // [B, seq, heads, head_dim]
            .reshape([batch, seq_len, self.num_heads * self.head_dim]);

        self.to_out.forward(attn_out)
    }

    /// Apply LoRA deltas to attention projections.
    ///
    /// Targets: `to_q`, `to_k`, `to_v`, `to_out.0`
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
            (&mut self.to_q, "to_q"),
            (&mut self.to_k, "to_k"),
            (&mut self.to_v, "to_v"),
            (&mut self.to_out, "to_out.0"),
        ] {
            let key = lora_key_base(lora_prefix, &format!("{prefix}.{suffix}"));
            if apply_lora_linear(field, &key, lora, strength, device)? {
                count += 1;
            }
        }

        Ok(count)
    }

    /// Reshape for multi-head attention.
    fn reshape_heads(
        &self,
        x: Tensor<Backend, 3>,
        batch: usize,
        seq_len: usize,
    ) -> Tensor<Backend, 4> {
        x.reshape([batch, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2) // [B, heads, seq, head_dim]
    }
}

/// Feed-forward network in transformer block.
#[derive(Debug)]
pub(crate) struct FeedForward {
    /// First linear: dim -> inner_dim (typically 4x)
    linear1: Linear<Backend>,
    /// Second linear: inner_dim -> dim
    linear2: Linear<Backend>,
}

impl FeedForward {
    /// Load from safetensors weights.
    ///
    /// SD uses GEGLU activation, so the first projection is 2x wider.
    /// Keys: `{prefix}.net.0.proj`, `{prefix}.net.2`
    pub(crate) fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        dim: usize,
        mult: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let inner_dim = dim * mult;
        // GEGLU: projects to 2x inner_dim, then splits for gating
        let linear1 = load_linear_with_bias(
            tensors,
            &format!("{prefix}.net.0.proj"),
            dim,
            inner_dim * 2,
            device,
        )?;
        let linear2 =
            load_linear_with_bias(tensors, &format!("{prefix}.net.2"), inner_dim, dim, device)?;

        Ok(Self { linear1, linear2 })
    }

    /// Forward with GEGLU activation.
    pub(crate) fn forward(&self, x: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
        let hidden = self.linear1.forward(x);

        // GEGLU: split in half along last dim, apply gelu to gate, multiply
        let [x, gate]: [Tensor<Backend, 3>; 2] = hidden
            .chunk(2, 2)
            .try_into()
            .expect("chunk returned wrong count");
        let hidden = x * burn::tensor::activation::gelu(gate);

        self.linear2.forward(hidden)
    }

    /// Apply LoRA deltas to feed-forward projections.
    ///
    /// Targets: `net.0.proj` (GEGLU gate+proj), `net.2` (output linear)
    pub(crate) fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;

        let key = lora_key_base(lora_prefix, &format!("{prefix}.net.0.proj"));
        if apply_lora_linear(&mut self.linear1, &key, lora, strength, device)? {
            count += 1;
        }

        let key = lora_key_base(lora_prefix, &format!("{prefix}.net.2"));
        if apply_lora_linear(&mut self.linear2, &key, lora, strength, device)? {
            count += 1;
        }

        Ok(count)
    }
}

/// Load linear layer without bias.
fn load_linear_no_bias(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    device: &Device<Backend>,
) -> Result<Linear<Backend>, LoadError> {
    let weight = load_tensor_2d(tensors, &format!("{prefix}.weight"), device)?;
    let weight = weight.transpose();

    let config = LinearConfig::new(in_features, out_features).with_bias(false);
    let mut linear = config.init(device);
    linear.weight = Param::from_tensor(weight);

    Ok(linear)
}

/// Load linear layer with bias.
fn load_linear_with_bias(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    device: &Device<Backend>,
) -> Result<Linear<Backend>, LoadError> {
    let weight = load_tensor_2d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;
    let weight = weight.transpose();

    let config = LinearConfig::new(in_features, out_features);
    let mut linear = config.init(device);
    linear.weight = Param::from_tensor(weight);
    linear.bias = Some(Param::from_tensor(bias));

    Ok(linear)
}
