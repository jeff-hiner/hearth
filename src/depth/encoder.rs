//! DINOv2 ViT-S/14 encoder for Depth Anything V2.
//!
//! Implements a Vision Transformer (ViT) encoder with:
//! - Patch embedding via 14x14 stride convolution
//! - CLS token + learned positional embeddings
//! - 12 transformer blocks with multi-head self-attention and GELU MLP
//! - Intermediate feature extraction from layers [2, 5, 8, 11]

use super::config::VitSmall as C;
use crate::{
    model_loader::{LoadError, load_tensor_1d, load_tensor_2d, load_tensor_4d},
    types::Backend,
};
use burn::{
    module::Param,
    nn::{
        LayerNorm, Linear, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::gelu,
};

/// DINOv2 ViT-S/14 encoder.
///
/// Extracts multi-scale features from an input image by running it through
/// a patch embedding followed by 12 transformer blocks. Features are extracted
/// from intermediate layers [2, 5, 8, 11] for the DPT decoder.
#[derive(Debug)]
pub(crate) struct VitEncoder {
    /// Patch embedding: Conv2d(3, 384, kernel=14, stride=14).
    patch_embed: Conv2d<Backend>,
    /// Learnable CLS token `[1, 1, 384]`.
    cls_token: Tensor<Backend, 3>,
    /// Positional embeddings `[1, 1+num_patches, 384]`.
    pos_embed: Tensor<Backend, 3>,
    /// Transformer blocks.
    blocks: Vec<VitBlock>,
    /// Final layer norm (applied after all blocks).
    norm: LayerNorm<Backend>,
}

impl VitEncoder {
    /// Load the encoder from a safetensors file.
    ///
    /// Expects weights under the `pretrained.` prefix.
    pub(crate) fn load(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        // Patch embedding: Conv2d(3, 384, 14, stride=14) with no padding
        let patch_embed = load_patch_embed(tensors, &format!("{prefix}.patch_embed.proj"), device)?;

        // CLS token [1, 1, 384] and positional embeddings [1, N+1, 384]
        let cls_token = load_tensor_3d(tensors, &format!("{prefix}.cls_token"), device)?;
        let pos_embed = load_tensor_3d(tensors, &format!("{prefix}.pos_embed"), device)?;

        // 12 transformer blocks
        let mut blocks = Vec::with_capacity(C::NUM_BLOCKS);
        for i in 0..C::NUM_BLOCKS {
            let block = VitBlock::load(tensors, &format!("{prefix}.blocks.{i}"), device)?;
            blocks.push(block);
        }

        // Final layer norm
        let norm = load_layer_norm(tensors, &format!("{prefix}.norm"), device)?;

        Ok(Self {
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
        })
    }

    /// Forward pass returning intermediate features from specified layers.
    ///
    /// # Arguments
    /// * `x` - Input image `[B, 3, H, W]` normalized (H, W must be divisible by 14)
    ///
    /// # Returns
    /// Four feature tensors from layers [2, 5, 8, 11], each `[B, num_patches, 384]`.
    /// Also returns `(patch_h, patch_w)` for spatial reshaping.
    pub(crate) fn forward(&self, x: Tensor<Backend, 4>) -> ([Tensor<Backend, 3>; 4], usize, usize) {
        let [batch, _, h, w] = x.shape().dims();
        let patch_h = h / C::PATCH_SIZE;
        let patch_w = w / C::PATCH_SIZE;

        // Patch embedding: [B, 3, H, W] -> [B, 384, patch_h, patch_w]
        let x = self.patch_embed.forward(x);
        // Flatten spatial dims: [B, 384, patch_h, patch_w] -> [B, 384, N] -> [B, N, 384]
        let [b, c, ph, pw] = x.shape().dims();
        let num_patches = ph * pw;
        let x = x.reshape([b, c, num_patches]).swap_dims(1, 2);

        // Prepend CLS token: [B, N, 384] -> [B, 1+N, 384]
        let cls = self.cls_token.clone().repeat_dim(0, batch);
        let x = Tensor::cat(vec![cls, x], 1);

        // Add positional embeddings
        let x = x + self.pos_embed.clone();

        // Run through transformer blocks, collecting intermediate outputs
        let mut features = Vec::with_capacity(4);
        let mut x = x;
        for (idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(x);
            if C::INTERMEDIATE_LAYERS.contains(&idx) {
                features.push(x.clone());
            }
        }

        // Apply final norm to the last extracted features only
        // (DPT uses unnormed intermediates for earlier layers, normed for the last)
        let last_idx = features.len() - 1;
        features[last_idx] = self.norm.forward(features[last_idx].clone());

        let features: [Tensor<Backend, 3>; 4] = features
            .try_into()
            .expect("should have exactly 4 intermediate features");

        (features, patch_h, patch_w)
    }
}

/// A single ViT transformer block.
///
/// Architecture: `x + ls1 * attn(norm1(x))`, then `x + ls2 * mlp(norm2(x))`.
/// DINOv2 uses LayerScale (per-channel scaling of residuals).
#[derive(Debug)]
struct VitBlock {
    /// Pre-attention layer norm.
    norm1: LayerNorm<Backend>,
    /// Multi-head self-attention.
    attn: VitAttention,
    /// LayerScale for attention residual `[384]`.
    ls1: Tensor<Backend, 1>,
    /// Pre-MLP layer norm.
    norm2: LayerNorm<Backend>,
    /// Two-layer MLP with GELU activation.
    mlp: VitMlp,
    /// LayerScale for MLP residual `[384]`.
    ls2: Tensor<Backend, 1>,
}

impl VitBlock {
    fn load(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let norm1 = load_layer_norm(tensors, &format!("{prefix}.norm1"), device)?;
        let attn = VitAttention::load(tensors, &format!("{prefix}.attn"), device)?;
        let ls1 = load_tensor_1d(tensors, &format!("{prefix}.ls1.gamma"), device)?;
        let norm2 = load_layer_norm(tensors, &format!("{prefix}.norm2"), device)?;
        let mlp = VitMlp::load(tensors, &format!("{prefix}.mlp"), device)?;
        let ls2 = load_tensor_1d(tensors, &format!("{prefix}.ls2.gamma"), device)?;

        Ok(Self {
            norm1,
            attn,
            ls1,
            norm2,
            mlp,
            ls2,
        })
    }

    /// Forward: pre-norm attention + pre-norm MLP with LayerScale residuals.
    fn forward(&self, x: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
        // LayerScale: [D] -> [1, 1, D] for broadcasting with [B, N, D]
        let ls1: Tensor<Backend, 3> = self.ls1.clone().unsqueeze::<2>().unsqueeze::<3>();
        let ls2: Tensor<Backend, 3> = self.ls2.clone().unsqueeze::<2>().unsqueeze::<3>();
        let x = x.clone() + self.attn.forward(self.norm1.forward(x)) * ls1;
        x.clone() + self.mlp.forward(self.norm2.forward(x)) * ls2
    }
}

/// Multi-head self-attention with fused QKV projection.
#[derive(Debug)]
struct VitAttention {
    /// Fused QKV projection: Linear(384, 384*3).
    qkv: Linear<Backend>,
    /// Output projection: Linear(384, 384).
    proj: Linear<Backend>,
    /// Number of attention heads.
    num_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Scaling factor (1/sqrt(head_dim)).
    scale: f32,
}

impl VitAttention {
    fn load(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let qkv = load_linear_dyn(
            tensors,
            &format!("{prefix}.qkv"),
            C::EMBED_DIM,
            C::EMBED_DIM * 3,
            device,
        )?;
        let proj = load_linear_dyn(
            tensors,
            &format!("{prefix}.proj"),
            C::EMBED_DIM,
            C::EMBED_DIM,
            device,
        )?;

        let head_dim = C::EMBED_DIM / C::NUM_HEADS;
        let scale = 1.0 / (head_dim as f32).sqrt();

        Ok(Self {
            qkv,
            proj,
            num_heads: C::NUM_HEADS,
            head_dim,
            scale,
        })
    }

    /// Multi-head self-attention forward pass.
    ///
    /// Input: `[B, N, D]` -> Output: `[B, N, D]`
    fn forward(&self, x: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
        let [b, n, _d] = x.shape().dims();

        // Fused QKV: [B, N, D] -> [B, N, 3*D]
        let qkv = self.qkv.forward(x);

        // Reshape to [B, N, 3, num_heads, head_dim] then permute
        let qkv = qkv.reshape([b, n, 3, self.num_heads, self.head_dim]);
        // -> [3, B, num_heads, N, head_dim]
        let qkv = qkv.permute([2, 0, 3, 1, 4]);

        // Split into Q, K, V: each [B, num_heads, N, head_dim]
        let q = qkv
            .clone()
            .slice(0..1_usize)
            .reshape([b, self.num_heads, n, self.head_dim]);
        let k = qkv
            .clone()
            .slice(1..2_usize)
            .reshape([b, self.num_heads, n, self.head_dim]);
        let v = qkv
            .slice(2..3_usize)
            .reshape([b, self.num_heads, n, self.head_dim]);

        // Scaled dot-product attention
        // attn = softmax(Q @ K^T / sqrt(d_k))
        let attn = q.matmul(k.transpose()) * self.scale;
        let attn = burn::tensor::activation::softmax(attn, 3);

        // attn @ V: [B, num_heads, N, head_dim]
        let out = attn.matmul(v);

        // Reshape back: [B, num_heads, N, head_dim] -> [B, N, D]
        let out = out
            .swap_dims(1, 2)
            .reshape([b, n, self.num_heads * self.head_dim]);

        // Output projection
        self.proj.forward(out)
    }
}

/// Two-layer MLP: Linear(D, 4D) -> GELU -> Linear(4D, D).
#[derive(Debug)]
struct VitMlp {
    /// First linear: D -> 4D.
    fc1: Linear<Backend>,
    /// Second linear: 4D -> D.
    fc2: Linear<Backend>,
}

impl VitMlp {
    fn load(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let fc1 = load_linear_dyn(
            tensors,
            &format!("{prefix}.fc1"),
            C::EMBED_DIM,
            C::MLP_DIM,
            device,
        )?;
        let fc2 = load_linear_dyn(
            tensors,
            &format!("{prefix}.fc2"),
            C::MLP_DIM,
            C::EMBED_DIM,
            device,
        )?;

        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
        let x = self.fc1.forward(x);
        let x = gelu(x);
        self.fc2.forward(x)
    }
}

// ---------------------------------------------------------------------------
// Loading helpers
// ---------------------------------------------------------------------------

/// Load a 3D tensor (e.g., cls_token, pos_embed).
fn load_tensor_3d(
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
    device: &Device<Backend>,
) -> Result<Tensor<Backend, 3>, LoadError> {
    // Load as flat 1D, then reshape to the original shape
    let view = tensors
        .tensor(name)
        .map_err(|_| LoadError::TensorNotFound(name.to_string()))?;

    let shape = view.shape().to_vec();
    if shape.len() != 3 {
        return Err(LoadError::ShapeMismatch {
            expected: vec![0, 0, 0],
            got: shape,
        });
    }

    // Load as 1D and reshape
    let flat = load_tensor_1d(tensors, name, device)?;
    Ok(flat.reshape([shape[0], shape[1], shape[2]]))
}

/// Load a Conv2d with kernel=14, stride=14, padding=0 (patch embedding).
fn load_patch_embed(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    device: &Device<Backend>,
) -> Result<Conv2d<Backend>, LoadError> {
    let weight = load_tensor_4d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let config = Conv2dConfig::new([3, C::EMBED_DIM], [C::PATCH_SIZE, C::PATCH_SIZE])
        .with_stride([C::PATCH_SIZE, C::PATCH_SIZE])
        .with_padding(PaddingConfig2d::Explicit(0, 0));

    let mut conv = config.init(device);
    conv.weight = Param::from_tensor(weight);
    conv.bias = Some(Param::from_tensor(bias));

    Ok(conv)
}

/// Load a LayerNorm with embed_dim channels.
fn load_layer_norm(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    device: &Device<Backend>,
) -> Result<LayerNorm<Backend>, LoadError> {
    let weight = load_tensor_1d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let config = burn::nn::LayerNormConfig::new(C::EMBED_DIM).with_epsilon(C::LN_EPS);
    let mut norm = config.init(device);

    norm.gamma = Param::from_tensor(weight);
    norm.beta = Some(Param::from_tensor(bias));

    Ok(norm)
}

/// Load a Linear layer with dynamic dimensions.
fn load_linear_dyn(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    device: &Device<Backend>,
) -> Result<Linear<Backend>, LoadError> {
    let weight = load_tensor_2d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    // Transpose: PyTorch [OUT, IN] -> Burn [IN, OUT]
    let weight = weight.transpose();

    let config = burn::nn::LinearConfig::new(in_features, out_features);
    let mut linear = config.init(device);

    linear.weight = Param::from_tensor(weight);
    linear.bias = Some(Param::from_tensor(bias));

    Ok(linear)
}
