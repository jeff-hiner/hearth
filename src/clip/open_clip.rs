//! OpenCLIP-G text encoder for SDXL.
//!
//! OpenCLIP-G (ViT-bigG-14) has a fundamentally different key naming scheme
//! and fused QKV projections compared to the HuggingFace CLIP-L encoder.
//! This module provides a standalone implementation.
//!
//! # Architecture
//!
//! - 32 transformer layers
//! - 1280 hidden dimension
//! - 20 attention heads (64 dim per head)
//! - 5120 feed-forward dimension
//! - Fused QKV: single `in_proj_weight` [3*hidden, hidden]
//! - Standard GELU activation (not QuickGELU)
//! - Text projection matrix for pooled output

use crate::{
    lora::{apply_lora_linear, lora_key_base},
    model_loader::{LoadError, load_tensor_1d, load_tensor_2d},
    types::Backend,
};
use burn::{module::Param, nn::Linear, prelude::*};
use safetensors::SafeTensors;

/// Number of transformer layers in OpenCLIP-G.
const NUM_LAYERS: usize = 32;
/// Hidden dimension of OpenCLIP-G.
const HIDDEN: usize = 1280;
/// Number of attention heads.
const HEADS: usize = 20;
/// Feed-forward dimension.
const FF_DIM: usize = 5120;
/// Dimension per attention head.
const HEAD_DIM: usize = HIDDEN / HEADS;
/// Vocabulary size.
const VOCAB_SIZE: usize = 49408;

/// OpenCLIP-G text encoder.
///
/// Produces both sequence hidden states and a pooled output projected through
/// `text_projection`.
#[derive(Debug)]
pub struct OpenClipTextEncoder {
    /// Token embedding lookup table.
    token_embedding: burn::nn::Embedding<Backend>,
    /// Learned positional embeddings `[SEQ_LEN, HIDDEN]`.
    positional_embedding: Tensor<Backend, 2>,
    /// Transformer blocks.
    resblocks: Vec<OpenClipResBlock>,
    /// Final layer normalization.
    ln_final: burn::nn::LayerNorm<Backend>,
    /// Text projection matrix `[HIDDEN, HIDDEN]`.
    text_projection: Tensor<Backend, 2>,
}

impl OpenClipTextEncoder {
    /// Load from safetensors weights.
    ///
    /// SDXL checkpoint prefix: `conditioner.embedders.1.model`
    pub fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        // Token embedding
        let token_weight =
            load_tensor_2d(tensors, &format!("{prefix}.token_embedding.weight"), device)?;
        let config = burn::nn::EmbeddingConfig::new(VOCAB_SIZE, HIDDEN);
        let mut token_embedding = config.init(device);
        token_embedding.weight = Param::from_tensor(token_weight);

        // Positional embedding (no .weight suffix in OpenCLIP)
        let positional_embedding =
            load_tensor_2d(tensors, &format!("{prefix}.positional_embedding"), device)?;

        // Transformer blocks
        let mut resblocks = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            let block = OpenClipResBlock::load(
                tensors,
                &format!("{prefix}.transformer.resblocks.{i}"),
                device,
            )?;
            resblocks.push(block);
        }

        // Final layer norm
        let ln_final_weight =
            load_tensor_1d(tensors, &format!("{prefix}.ln_final.weight"), device)?;
        let ln_final_bias = load_tensor_1d(tensors, &format!("{prefix}.ln_final.bias"), device)?;
        let ln_config = burn::nn::LayerNormConfig::new(HIDDEN);
        let mut ln_final = ln_config.init(device);
        ln_final.gamma = Param::from_tensor(ln_final_weight);
        ln_final.beta = Some(Param::from_tensor(ln_final_bias));

        // Text projection matrix
        let text_projection =
            load_tensor_2d(tensors, &format!("{prefix}.text_projection"), device)?;

        Ok(Self {
            token_embedding,
            positional_embedding,
            resblocks,
            ln_final,
            text_projection,
        })
    }

    /// Forward pass producing penultimate hidden states and pooled output.
    ///
    /// SDXL expects the penultimate transformer layer output (layer 30 of 32) as
    /// cross-attention conditioning, matching the convention used by CLIP-L. The pooled
    /// output is derived from the final layer + `ln_final` + `text_projection`.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs `[B, 77]`
    ///
    /// # Returns
    /// `(penultimate_hidden_states, pooled_output)`:
    /// - `penultimate_hidden_states`: `[B, 77, 1280]` - layer 30 output (penultimate)
    /// - `pooled_output`: `[B, 1280]` - EOS token from final layer, projected through `text_projection`
    pub fn forward(
        &self,
        input_ids: Tensor<Backend, 2, Int>,
    ) -> (Tensor<Backend, 3>, Tensor<Backend, 2>) {
        let [batch, seq_len] = input_ids.shape().dims();

        // Token + positional embeddings
        let token_embeds = self.token_embedding.forward(input_ids.clone());
        let pos_embeds = self
            .positional_embedding
            .clone()
            .slice([0..seq_len, 0..HIDDEN])
            .unsqueeze::<3>()
            .expand([batch, seq_len, HIDDEN]);
        let mut hidden = token_embeds + pos_embeds;

        // Transformer blocks — capture penultimate layer output for cross-attention
        let mut penultimate = None;
        for (i, block) in self.resblocks.iter().enumerate() {
            hidden = block.forward(hidden);
            if i == NUM_LAYERS - 2 {
                penultimate = Some(hidden.clone());
            }
        }

        // Final layer norm (applied to final layer output for pooling)
        let final_hidden = self.ln_final.forward(hidden);

        // Pooled output: take hidden state at EOS position
        // EOS is the last non-padding token; for CLIP it's always the highest-id token position
        let eos_indices = input_ids.argmax(1); // [B, 1]
        let pooled = gather_at_indices(&final_hidden, &eos_indices);

        // Project through text_projection: [B, HIDDEN] @ [HIDDEN, HIDDEN] -> [B, HIDDEN]
        let pooled = pooled.matmul(self.text_projection.clone());

        (
            penultimate.expect("OpenCLIP-G must have >= 2 layers"),
            pooled,
        )
    }

    /// Apply LoRA deltas to this OpenCLIP-G text encoder.
    ///
    /// LoRA prefix is typically `"lora_te2"`.
    /// Model path root starts at `transformer.resblocks.{i}`.
    ///
    /// Fused `in_proj` LoRA is uncommon and not supported — a warning is
    /// logged if encountered.
    ///
    /// Returns the total number of deltas applied.
    pub fn apply_lora(
        &mut self,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;
        for (i, block) in self.resblocks.iter_mut().enumerate() {
            count += block.apply_lora(
                &format!("transformer.resblocks.{i}"),
                lora_prefix,
                lora,
                strength,
                device,
            )?;
        }
        Ok(count)
    }
}

/// Gather hidden states at specific sequence positions.
///
/// Takes `hidden [B, S, H]` and `indices [B, 1]`, returns `[B, H]`.
fn gather_at_indices(
    hidden: &Tensor<Backend, 3>,
    indices: &Tensor<Backend, 2, Int>,
) -> Tensor<Backend, 2> {
    let [batch, _seq, dim] = hidden.shape().dims();

    // Expand indices to [B, 1, H] for gathering
    let indices_expanded = indices
        .clone()
        .unsqueeze_dim::<3>(2)
        .expand([batch, 1, dim]);

    // Gather along seq dimension and squeeze
    hidden
        .clone()
        .gather(1, indices_expanded)
        .reshape([batch, dim])
}

/// A single OpenCLIP transformer block.
#[derive(Debug)]
struct OpenClipResBlock {
    /// Pre-attention layer norm.
    ln_1: burn::nn::LayerNorm<Backend>,
    /// Fused QKV attention.
    attn: OpenClipAttention,
    /// Pre-MLP layer norm.
    ln_2: burn::nn::LayerNorm<Backend>,
    /// Feed-forward MLP.
    mlp: OpenClipMlp,
}

impl OpenClipResBlock {
    fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let ln_1 = load_openclip_layer_norm(tensors, &format!("{prefix}.ln_1"), device)?;
        let attn = OpenClipAttention::load(tensors, &format!("{prefix}.attn"), device)?;
        let ln_2 = load_openclip_layer_norm(tensors, &format!("{prefix}.ln_2"), device)?;
        let mlp = OpenClipMlp::load(tensors, &format!("{prefix}.mlp"), device)?;

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    fn forward(&self, x: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
        // Self-attention with residual
        let residual = x.clone();
        let x = self.ln_1.forward(x);
        let x = self.attn.forward(x);
        let x = residual + x;

        // MLP with residual
        let residual = x.clone();
        let x = self.ln_2.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }

    fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;
        count += self.attn.apply_lora(
            &format!("{prefix}.attn"),
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

/// OpenCLIP attention with fused QKV projection.
///
/// Uses a single `in_proj_weight` `[3*HIDDEN, HIDDEN]` and `in_proj_bias` `[3*HIDDEN]`
/// instead of separate Q, K, V matrices.
#[derive(Debug)]
struct OpenClipAttention {
    /// Fused QKV weight `[3*HIDDEN, HIDDEN]` (stored transposed for Burn).
    in_proj: Linear<Backend>,
    /// Output projection.
    out_proj: Linear<Backend>,
}

impl OpenClipAttention {
    fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        // Fused QKV: [3*HIDDEN, HIDDEN]
        let in_proj_weight = load_tensor_2d(tensors, &format!("{prefix}.in_proj_weight"), device)?;
        let in_proj_bias = load_tensor_1d(tensors, &format!("{prefix}.in_proj_bias"), device)?;
        let in_proj_weight = in_proj_weight.transpose();

        let in_config = burn::nn::LinearConfig::new(HIDDEN, 3 * HIDDEN);
        let mut in_proj = in_config.init(device);
        in_proj.weight = Param::from_tensor(in_proj_weight);
        in_proj.bias = Some(Param::from_tensor(in_proj_bias));

        // Output projection
        let out_weight = load_tensor_2d(tensors, &format!("{prefix}.out_proj.weight"), device)?;
        let out_bias = load_tensor_1d(tensors, &format!("{prefix}.out_proj.bias"), device)?;
        let out_weight = out_weight.transpose();

        let out_config = burn::nn::LinearConfig::new(HIDDEN, HIDDEN);
        let mut out_proj = out_config.init(device);
        out_proj.weight = Param::from_tensor(out_weight);
        out_proj.bias = Some(Param::from_tensor(out_bias));

        Ok(Self { in_proj, out_proj })
    }

    fn forward(&self, x: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
        let [batch, seq_len, _] = x.shape().dims();
        let device = x.device();

        // Fused QKV projection: [B, S, H] -> [B, S, 3*H]
        let qkv = self.in_proj.forward(x);

        // Split into Q, K, V: each [B, S, H]
        let [q, k, v]: [Tensor<Backend, 3>; 3] = qkv
            .chunk(3, 2)
            .try_into()
            .expect("chunk returned wrong count");

        // Reshape for multi-head attention: [B, S, H] -> [B, HEADS, S, HEAD_DIM]
        let q = q.reshape([batch, seq_len, HEADS, HEAD_DIM]).swap_dims(1, 2);
        let k = k.reshape([batch, seq_len, HEADS, HEAD_DIM]).swap_dims(1, 2);
        let v = v.reshape([batch, seq_len, HEADS, HEAD_DIM]).swap_dims(1, 2);

        // Scaled dot-product attention with causal mask
        let scale = (HEAD_DIM as f32).sqrt();
        let attn_weights: Tensor<Backend, 4> = q.matmul(k.transpose()) / scale;

        // Causal mask
        let causal_mask = create_causal_mask(seq_len, &device);
        let attn_weights = attn_weights
            + causal_mask
                .unsqueeze::<4>()
                .expand([batch, HEADS, seq_len, seq_len]);

        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v);

        // Reshape back: [B, HEADS, S, HEAD_DIM] -> [B, S, H]
        let attn_output = attn_output
            .swap_dims(1, 2)
            .reshape([batch, seq_len, HIDDEN]);

        self.out_proj.forward(attn_output)
    }

    fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;

        // Check for fused in_proj LoRA — uncommon, warn and skip
        let in_proj_key = lora_key_base(lora_prefix, &format!("{prefix}.in_proj"));
        let in_proj_down = format!("{in_proj_key}.lora_down.weight");
        if lora.tensor(&in_proj_down).is_ok() {
            tracing::warn!(
                key = %in_proj_key,
                "fused in_proj LoRA not supported for OpenCLIP, skipping"
            );
        }

        let key = lora_key_base(lora_prefix, &format!("{prefix}.out_proj"));
        if apply_lora_linear(&mut self.out_proj, &key, lora, strength, device)? {
            count += 1;
        }

        Ok(count)
    }
}

/// OpenCLIP MLP with standard GELU activation.
///
/// Key naming: `c_fc` (up-projection) and `c_proj` (down-projection).
#[derive(Debug)]
struct OpenClipMlp {
    /// Up-projection: HIDDEN -> FF_DIM.
    c_fc: Linear<Backend>,
    /// Down-projection: FF_DIM -> HIDDEN.
    c_proj: Linear<Backend>,
}

impl OpenClipMlp {
    fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let c_fc =
            load_openclip_linear(tensors, &format!("{prefix}.c_fc"), HIDDEN, FF_DIM, device)?;
        let c_proj =
            load_openclip_linear(tensors, &format!("{prefix}.c_proj"), FF_DIM, HIDDEN, device)?;

        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, x: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
        let x = self.c_fc.forward(x);
        let x = burn::tensor::activation::gelu(x);
        self.c_proj.forward(x)
    }

    fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;

        let key = lora_key_base(lora_prefix, &format!("{prefix}.c_fc"));
        if apply_lora_linear(&mut self.c_fc, &key, lora, strength, device)? {
            count += 1;
        }

        let key = lora_key_base(lora_prefix, &format!("{prefix}.c_proj"));
        if apply_lora_linear(&mut self.c_proj, &key, lora, strength, device)? {
            count += 1;
        }

        Ok(count)
    }
}

/// Create a causal attention mask.
fn create_causal_mask(seq_len: usize, device: &Device<Backend>) -> Tensor<Backend, 2> {
    let ones: Tensor<Backend, 2> = Tensor::ones([seq_len, seq_len], device);
    let mask = ones.triu(1);
    let neg_inf: Tensor<Backend, 2> = Tensor::full([seq_len, seq_len], f32::NEG_INFINITY, device);
    let zeros: Tensor<Backend, 2> = Tensor::zeros([seq_len, seq_len], device);
    zeros.mask_where(mask.bool(), neg_inf)
}

/// Load a LayerNorm for OpenCLIP.
fn load_openclip_layer_norm(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    device: &Device<Backend>,
) -> Result<burn::nn::LayerNorm<Backend>, LoadError> {
    let weight = load_tensor_1d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let config = burn::nn::LayerNormConfig::new(HIDDEN);
    let mut norm = config.init(device);
    norm.gamma = Param::from_tensor(weight);
    norm.beta = Some(Param::from_tensor(bias));

    Ok(norm)
}

/// Load a linear layer for OpenCLIP.
fn load_openclip_linear(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    device: &Device<Backend>,
) -> Result<Linear<Backend>, LoadError> {
    let weight = load_tensor_2d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;
    let weight = weight.transpose();

    let config = burn::nn::LinearConfig::new(in_features, out_features);
    let mut linear = config.init(device);
    linear.weight = Param::from_tensor(weight);
    linear.bias = Some(Param::from_tensor(bias));

    Ok(linear)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_loader::SafeTensorsFile;
    use std::path::Path;

    #[test]
    fn load_openclip_from_checkpoint() {
        let path = Path::new("models/checkpoints/sd_xl_base_1.0.safetensors");
        if !path.exists() {
            println!("SDXL model not found, skipping");
            return;
        }

        let file = SafeTensorsFile::open(path).expect("Failed to open checkpoint");
        let tensors = file.tensors().expect("Failed to parse safetensors");
        let device = Default::default();

        println!("Loading OpenCLIP-G...");
        let encoder = OpenClipTextEncoder::load(&tensors, "conditioner.embedders.1.model", &device)
            .expect("Failed to load OpenCLIP-G");

        // Dummy tokens
        let mut token_ids = vec![49407i32; 77];
        token_ids[0] = 49406;
        token_ids[1] = 320;
        token_ids[2] = 2368;
        token_ids[3] = 49407;

        let input_ids: Tensor<Backend, 1, Int> = Tensor::from_ints(token_ids.as_slice(), &device);
        let input_ids = input_ids.unsqueeze::<2>();

        let (hidden, pooled) = encoder.forward(input_ids);
        println!(
            "OpenCLIP-G hidden shape: {:?}, pooled shape: {:?}",
            hidden.shape().dims::<3>(),
            pooled.shape().dims::<2>()
        );
        assert_eq!(hidden.shape().dims(), [1, 77, 1280]);
        assert_eq!(pooled.shape().dims(), [1, 1280]);
    }
}
