//! Spatial transformer blocks for UNet cross-attention.

use super::attention::{CrossAttention, FeedForward};
use crate::{
    lora::{apply_lora_conv2d, apply_lora_linear, lora_key_base},
    model_loader::{LoadError, load_tensor_1d, load_tensor_2d, load_tensor_4d},
    types::Backend,
};
use burn::{
    module::Param,
    nn::{GroupNorm, LayerNorm, Linear, conv::Conv2dConfig},
    prelude::*,
};
use safetensors::SafeTensors;

/// Basic transformer block with self-attention, cross-attention, and feed-forward.
#[derive(Debug)]
pub(crate) struct BasicTransformerBlock {
    /// Layer norm before self-attention.
    norm1: LayerNorm<Backend>,
    /// Self-attention.
    attn1: CrossAttention,
    /// Layer norm before cross-attention.
    norm2: LayerNorm<Backend>,
    /// Cross-attention (attends to context).
    attn2: CrossAttention,
    /// Layer norm before feed-forward.
    norm3: LayerNorm<Backend>,
    /// Feed-forward network.
    ff: FeedForward,
}

impl BasicTransformerBlock {
    /// Load from safetensors weights.
    pub(crate) fn load(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        dim: usize,
        context_dim: usize,
        num_heads: usize,
        head_dim: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let norm1 = load_layer_norm(tensors, &format!("{prefix}.norm1"), dim, device)?;
        let attn1 = CrossAttention::load(
            tensors,
            &format!("{prefix}.attn1"),
            dim,
            dim, // self-attention: context_dim = query_dim
            num_heads,
            head_dim,
            device,
        )?;

        let norm2 = load_layer_norm(tensors, &format!("{prefix}.norm2"), dim, device)?;
        let attn2 = CrossAttention::load(
            tensors,
            &format!("{prefix}.attn2"),
            dim,
            context_dim,
            num_heads,
            head_dim,
            device,
        )?;

        let norm3 = load_layer_norm(tensors, &format!("{prefix}.norm3"), dim, device)?;
        let ff = FeedForward::load(tensors, &format!("{prefix}.ff"), dim, 4, device)?;

        Ok(Self {
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ff,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input `[B, seq_len, dim]`
    /// * `context` - Cross-attention context `[B, context_len, context_dim]`
    pub(crate) fn forward(
        &self,
        x: Tensor<Backend, 3>,
        context: Option<&Tensor<Backend, 3>>,
    ) -> Tensor<Backend, 3> {
        // Self-attention with residual
        let residual = x.clone();
        let x = residual + self.attn1.forward(self.norm1.forward(x), None);

        // Cross-attention with residual
        let residual = x.clone();
        let x = residual + self.attn2.forward(self.norm2.forward(x), context);

        // Feed-forward with residual
        let residual = x.clone();
        residual + self.ff.forward(self.norm3.forward(x))
    }

    /// Apply LoRA deltas to this transformer block.
    ///
    /// Delegates to attn1, attn2, and ff with appropriate sub-prefixes.
    pub(crate) fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;
        count += self.attn1.apply_lora(
            &format!("{prefix}.attn1"),
            lora_prefix,
            lora,
            strength,
            device,
        )?;
        count += self.attn2.apply_lora(
            &format!("{prefix}.attn2"),
            lora_prefix,
            lora,
            strength,
            device,
        )?;
        count +=
            self.ff
                .apply_lora(&format!("{prefix}.ff"), lora_prefix, lora, strength, device)?;
        Ok(count)
    }
}

/// Spatial transformer that applies attention to spatial feature maps.
///
/// Converts 4D image features to sequence format, applies transformer blocks,
/// then converts back to spatial format.
#[derive(Debug)]
pub(crate) struct SpatialTransformer {
    /// Input normalization.
    norm: GroupNorm<Backend>,
    /// Input projection (conv or linear).
    proj_in: SpatialProjection,
    /// Transformer blocks.
    transformer_blocks: Vec<BasicTransformerBlock>,
    /// Output projection.
    proj_out: SpatialProjection,
}

/// Projection layer for spatial transformer input/output.
///
/// SD 1.5 uses 1x1 convolutions (4D weights), SDXL uses linear layers (2D weights)
/// for deeper levels. Detected at load time by tensor dimensionality.
#[derive(Debug)]
enum SpatialProjection {
    /// 1x1 convolution projection.
    Conv(burn::nn::conv::Conv2d<Backend>),
    /// Linear projection (SDXL style).
    Linear(Linear<Backend>),
}

impl SpatialTransformer {
    /// Load from safetensors weights.
    ///
    /// SD checkpoint structure:
    /// - `{prefix}.norm` - GroupNorm
    /// - `{prefix}.proj_in` - Conv2d 1x1 or Linear
    /// - `{prefix}.transformer_blocks.{i}` - BasicTransformerBlock
    /// - `{prefix}.proj_out` - Conv2d 1x1 or Linear
    #[expect(
        clippy::too_many_arguments,
        reason = "loading config needs all parameters"
    )]
    pub(crate) fn load(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        in_channels: usize,
        num_heads: usize,
        head_dim: usize,
        context_dim: usize,
        depth: usize,
        groups: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let inner_dim = num_heads * head_dim;

        let norm = load_group_norm(
            tensors,
            &format!("{prefix}.norm"),
            groups,
            in_channels,
            device,
        )?;

        // Detect projection type by tensor shape dimensionality
        let proj_in = load_spatial_projection(
            tensors,
            &format!("{prefix}.proj_in"),
            in_channels,
            inner_dim,
            device,
        )?;
        let proj_out = load_spatial_projection(
            tensors,
            &format!("{prefix}.proj_out"),
            inner_dim,
            in_channels,
            device,
        )?;

        let mut transformer_blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let block = BasicTransformerBlock::load(
                tensors,
                &format!("{prefix}.transformer_blocks.{i}"),
                inner_dim,
                context_dim,
                num_heads,
                head_dim,
                device,
            )?;
            transformer_blocks.push(block);
        }

        Ok(Self {
            norm,
            proj_in,
            transformer_blocks,
            proj_out,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Input features `[B, C, H, W]`
    /// * `context` - Cross-attention context `[B, seq_len, context_dim]`
    pub(crate) fn forward(
        &self,
        x: Tensor<Backend, 4>,
        context: Option<&Tensor<Backend, 3>>,
    ) -> Tensor<Backend, 4> {
        let [batch, _channels, height, width] = x.shape().dims();
        let residual = x.clone();

        // Normalize
        let x = self.norm.forward(x);

        // Project in + reshape to sequence
        let (x, inner_dim) = match &self.proj_in {
            SpatialProjection::Conv(conv) => {
                // Conv path: project first, then reshape
                let x = conv.forward(x);
                let [_, inner_dim, _, _] = x.shape().dims();
                let x = x
                    .swap_dims(1, 2) // [B, H, C, W]
                    .swap_dims(2, 3) // [B, H, W, C]
                    .reshape([batch, height * width, inner_dim]);
                (x, inner_dim)
            }
            SpatialProjection::Linear(linear) => {
                // Linear path: reshape first, then project
                let [_, inner_dim, _, _] = x.shape().dims();
                let x = x
                    .swap_dims(1, 2) // [B, H, C, W]
                    .swap_dims(2, 3) // [B, H, W, C]
                    .reshape([batch, height * width, inner_dim]);
                let x = linear.forward(x);
                let actual_dim = x.shape().dims::<3>()[2];
                (x, actual_dim)
            }
        };

        // Apply transformer blocks
        let mut x = x;
        for block in &self.transformer_blocks {
            x = block.forward(x, context);
        }

        // Project out + reshape back to spatial
        let x = match &self.proj_out {
            SpatialProjection::Conv(conv) => {
                // Conv path: reshape back first, then project
                let x = x
                    .reshape([batch, height, width, inner_dim])
                    .swap_dims(2, 3) // [B, H, C, W]
                    .swap_dims(1, 2); // [B, C, H, W]
                conv.forward(x)
            }
            SpatialProjection::Linear(linear) => {
                // Linear path: project first, then reshape back
                let x = linear.forward(x);
                let out_dim = x.shape().dims::<3>()[2];
                x.reshape([batch, height, width, out_dim])
                    .swap_dims(2, 3) // [B, H, C, W]
                    .swap_dims(1, 2) // [B, C, H, W]
            }
        };

        // Residual connection
        x + residual
    }

    /// Apply LoRA deltas to this spatial transformer.
    ///
    /// Targets: proj_in, proj_out (Conv1x1 or Linear), and each transformer_block.
    pub(crate) fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;

        // proj_in may be Conv or Linear
        match &mut self.proj_in {
            SpatialProjection::Conv(conv) => {
                let key = lora_key_base(lora_prefix, &format!("{prefix}.proj_in"));
                if apply_lora_conv2d(conv, &key, lora, strength, device)? {
                    count += 1;
                }
            }
            SpatialProjection::Linear(linear) => {
                let key = lora_key_base(lora_prefix, &format!("{prefix}.proj_in"));
                if apply_lora_linear(linear, &key, lora, strength, device)? {
                    count += 1;
                }
            }
        }

        // proj_out may be Conv or Linear
        match &mut self.proj_out {
            SpatialProjection::Conv(conv) => {
                let key = lora_key_base(lora_prefix, &format!("{prefix}.proj_out"));
                if apply_lora_conv2d(conv, &key, lora, strength, device)? {
                    count += 1;
                }
            }
            SpatialProjection::Linear(linear) => {
                let key = lora_key_base(lora_prefix, &format!("{prefix}.proj_out"));
                if apply_lora_linear(linear, &key, lora, strength, device)? {
                    count += 1;
                }
            }
        }

        // Transformer blocks
        for (i, block) in self.transformer_blocks.iter_mut().enumerate() {
            count += block.apply_lora(
                &format!("{prefix}.transformer_blocks.{i}"),
                lora_prefix,
                lora,
                strength,
                device,
            )?;
        }

        Ok(count)
    }
}

/// Load a spatial projection, auto-detecting Conv vs Linear from tensor shape.
fn load_spatial_projection(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    in_dim: usize,
    out_dim: usize,
    device: &Device<Backend>,
) -> Result<SpatialProjection, LoadError> {
    let weight_name = format!("{prefix}.weight");
    let view = tensors
        .tensor(&weight_name)
        .map_err(|_| LoadError::TensorNotFound(weight_name.clone()))?;

    if view.shape().len() == 4 {
        // 4D weight -> Conv2d 1x1
        let conv = load_conv1x1(tensors, prefix, in_dim, out_dim, device)?;
        Ok(SpatialProjection::Conv(conv))
    } else {
        // 2D weight -> Linear
        let linear = load_linear_with_bias(tensors, prefix, in_dim, out_dim, device)?;
        Ok(SpatialProjection::Linear(linear))
    }
}

/// Load LayerNorm.
fn load_layer_norm(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    dim: usize,
    device: &Device<Backend>,
) -> Result<LayerNorm<Backend>, LoadError> {
    let weight = load_tensor_1d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let config = burn::nn::LayerNormConfig::new(dim);
    let mut norm = config.init(device);
    norm.gamma = Param::from_tensor(weight);
    norm.beta = Some(Param::from_tensor(bias));

    Ok(norm)
}

/// Load GroupNorm.
fn load_group_norm(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    groups: usize,
    channels: usize,
    device: &Device<Backend>,
) -> Result<GroupNorm<Backend>, LoadError> {
    let weight = load_tensor_1d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let config = burn::nn::GroupNormConfig::new(groups, channels);
    let mut norm = config.init(device);
    norm.gamma = Some(Param::from_tensor(weight));
    norm.beta = Some(Param::from_tensor(bias));

    Ok(norm)
}

/// Load 1x1 Conv2d.
fn load_conv1x1(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    device: &Device<Backend>,
) -> Result<burn::nn::conv::Conv2d<Backend>, LoadError> {
    let weight = load_tensor_4d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let config = Conv2dConfig::new([in_channels, out_channels], [1, 1]);
    let mut conv = config.init(device);
    conv.weight = Param::from_tensor(weight);
    conv.bias = Some(Param::from_tensor(bias));

    Ok(conv)
}

/// Load linear layer with bias (for spatial projection).
fn load_linear_with_bias(
    tensors: &safetensors::SafeTensors<'_>,
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
