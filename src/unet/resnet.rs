//! ResNet blocks with time embedding conditioning for UNet.

use crate::{
    lora::{apply_lora_conv2d, apply_lora_linear, lora_key_base},
    model_loader::{LoadError, load_tensor_1d, load_tensor_2d, load_tensor_4d},
    types::Backend,
};
use burn::{
    module::Param,
    nn::{
        GroupNorm, GroupNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::silu,
};
use safetensors::SafeTensors;

/// ResNet block with time embedding conditioning.
///
/// Architecture:
/// ```text
/// x ─────────────────────────────────────────────┐
/// │                                              │ (skip connection)
/// ├─> norm1 -> silu -> conv1 -> (+time_emb) ─┐   │
/// │                                          │   │
/// └──────────────────────────────────────────┘   │
///                    │                           │
///                    v                           │
///              norm2 -> silu -> conv2 ──────────(+)─> out
/// ```
#[derive(Debug)]
pub(crate) struct ResnetBlock2D {
    /// First group norm.
    norm1: GroupNorm<Backend>,
    /// First convolution (in -> out channels).
    conv1: Conv2d<Backend>,
    /// Time embedding projection (optional).
    time_emb_proj: Option<burn::nn::Linear<Backend>>,
    /// Second group norm.
    norm2: GroupNorm<Backend>,
    /// Second convolution (out -> out channels).
    conv2: Conv2d<Backend>,
    /// Skip connection conv when in_channels != out_channels.
    skip_conv: Option<Conv2d<Backend>>,
}

impl ResnetBlock2D {
    /// Load a ResNet block from safetensors.
    ///
    /// SD checkpoint key structure:
    /// - `{prefix}.in_layers.0` - norm1 (GroupNorm)
    /// - `{prefix}.in_layers.2` - conv1
    /// - `{prefix}.emb_layers.1` - time_emb_proj (Linear)
    /// - `{prefix}.out_layers.0` - norm2 (GroupNorm)
    /// - `{prefix}.out_layers.3` - conv2
    /// - `{prefix}.skip_connection` - skip conv (optional)
    pub(crate) fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        time_embed_dim: usize,
        groups: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let norm1 = load_group_norm_dyn(
            tensors,
            &format!("{prefix}.in_layers.0"),
            groups,
            in_channels,
            device,
        )?;
        let conv1 = load_conv2d_dyn(
            tensors,
            &format!("{prefix}.in_layers.2"),
            3,
            in_channels,
            out_channels,
            device,
        )?;

        // Time embedding projection
        let time_emb_proj = if time_embed_dim > 0 {
            Some(load_linear_dyn(
                tensors,
                &format!("{prefix}.emb_layers.1"),
                time_embed_dim,
                out_channels,
                device,
            )?)
        } else {
            None
        };

        let norm2 = load_group_norm_dyn(
            tensors,
            &format!("{prefix}.out_layers.0"),
            groups,
            out_channels,
            device,
        )?;
        let conv2 = load_conv2d_dyn(
            tensors,
            &format!("{prefix}.out_layers.3"),
            3,
            out_channels,
            out_channels,
            device,
        )?;

        // Skip connection only when channel dimensions differ
        let skip_conv = if in_channels != out_channels {
            Some(load_conv2d_dyn(
                tensors,
                &format!("{prefix}.skip_connection"),
                1,
                in_channels,
                out_channels,
                device,
            )?)
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            time_emb_proj,
            norm2,
            conv2,
            skip_conv,
        })
    }

    /// Forward pass with optional time embedding.
    ///
    /// # Arguments
    /// * `x` - Input tensor `[B, C, H, W]`
    /// * `time_emb` - Optional reference to time embedding `[B, time_embed_dim]`
    pub(crate) fn forward(
        &self,
        x: Tensor<Backend, 4>,
        time_emb: Option<&Tensor<Backend, 2>>,
    ) -> Tensor<Backend, 4> {
        // First half: norm -> silu -> conv
        let hidden = self.norm1.forward(x.clone());
        let hidden = silu(hidden);
        let hidden = self.conv1.forward(hidden);

        // Add time embedding if provided
        let hidden = match (&self.time_emb_proj, time_emb) {
            (Some(proj), Some(emb)) => {
                let emb = silu(emb.clone());
                let emb = proj.forward(emb);
                // Reshape [B, C] -> [B, C, 1, 1] for broadcasting
                let [b, c] = emb.shape().dims();
                let emb = emb.reshape([b, c, 1, 1]);
                hidden + emb
            }
            _ => hidden,
        };

        // Second half: norm -> silu -> conv
        let hidden = self.norm2.forward(hidden);
        let hidden = silu(hidden);
        let hidden = self.conv2.forward(hidden);

        // Skip connection
        let skip = match &self.skip_conv {
            Some(conv) => conv.forward(x),
            None => x,
        };

        hidden + skip
    }

    /// Apply LoRA deltas to this ResNet block.
    ///
    /// Targets: `in_layers.2` (conv1), `out_layers.3` (conv2),
    /// `emb_layers.1` (time_emb_proj), `skip_connection`
    pub(crate) fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;

        let key = lora_key_base(lora_prefix, &format!("{prefix}.in_layers.2"));
        if apply_lora_conv2d(&mut self.conv1, &key, lora, strength, device)? {
            count += 1;
        }

        let key = lora_key_base(lora_prefix, &format!("{prefix}.out_layers.3"));
        if apply_lora_conv2d(&mut self.conv2, &key, lora, strength, device)? {
            count += 1;
        }

        if let Some(ref mut proj) = self.time_emb_proj {
            let key = lora_key_base(lora_prefix, &format!("{prefix}.emb_layers.1"));
            if apply_lora_linear(proj, &key, lora, strength, device)? {
                count += 1;
            }
        }

        if let Some(ref mut conv) = self.skip_conv {
            let key = lora_key_base(lora_prefix, &format!("{prefix}.skip_connection"));
            if apply_lora_conv2d(conv, &key, lora, strength, device)? {
                count += 1;
            }
        }

        Ok(count)
    }
}

/// Load GroupNorm with dynamic dimensions.
fn load_group_norm_dyn(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    groups: usize,
    channels: usize,
    device: &Device<Backend>,
) -> Result<GroupNorm<Backend>, LoadError> {
    let weight = load_tensor_1d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let config = GroupNormConfig::new(groups, channels);
    let mut norm = config.init(device);

    norm.gamma = Some(Param::from_tensor(weight));
    norm.beta = Some(Param::from_tensor(bias));

    Ok(norm)
}

/// Load Conv2d with dynamic dimensions.
fn load_conv2d_dyn(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
    device: &Device<Backend>,
) -> Result<Conv2d<Backend>, LoadError> {
    let weight = load_tensor_4d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let padding = kernel_size / 2;
    let config = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
        .with_padding(PaddingConfig2d::Explicit(padding, padding));

    let mut conv = config.init(device);
    conv.weight = Param::from_tensor(weight);
    conv.bias = Some(Param::from_tensor(bias));

    Ok(conv)
}

/// Load Linear with dynamic dimensions.
fn load_linear_dyn(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    in_features: usize,
    out_features: usize,
    device: &Device<Backend>,
) -> Result<burn::nn::Linear<Backend>, LoadError> {
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
