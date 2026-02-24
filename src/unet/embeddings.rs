//! Time step embeddings for UNet conditioning.

use crate::{
    model_loader::{LoadError, load_tensor_1d, load_tensor_2d},
    types::Backend,
};
use burn::{module::Param, nn::Linear, prelude::*};
use safetensors::SafeTensors;

/// Sinusoidal timestep embeddings.
///
/// Converts scalar timesteps to high-dimensional embeddings using
/// sinusoidal position encoding (similar to Transformer positional encoding).
#[derive(Debug)]
pub(crate) struct Timesteps {
    /// Number of embedding channels (half sin, half cos).
    num_channels: usize,
    /// Whether to put cos before sin.
    flip_sin_to_cos: bool,
    /// Frequency downscale shift.
    downscale_freq_shift: f64,
}

impl Timesteps {
    /// Create new timestep embeddings.
    pub(crate) fn new(
        num_channels: usize,
        flip_sin_to_cos: bool,
        downscale_freq_shift: f64,
    ) -> Self {
        Self {
            num_channels,
            flip_sin_to_cos,
            downscale_freq_shift,
        }
    }

    /// Compute sinusoidal embeddings for timesteps.
    ///
    /// # Arguments
    /// * `timesteps` - Tensor of shape `[batch]` containing timestep values
    ///
    /// # Returns
    /// Embeddings of shape `[batch, num_channels]`
    pub(crate) fn forward(&self, timesteps: Tensor<Backend, 1>) -> Tensor<Backend, 2> {
        let device = timesteps.device();
        let half_dim = self.num_channels / 2;

        // Compute frequency exponents: exp(-ln(10000) * i / (half_dim - shift))
        let exponent: Tensor<Backend, 1> = Tensor::arange(0..half_dim as i64, &device).float();
        let exponent =
            exponent * (-f64::ln(10000.0) / (half_dim as f64 - self.downscale_freq_shift));
        let exponent = exponent.exp();

        // Outer product: timesteps [B, 1] * exponent [1, half_dim] -> [B, half_dim]
        let emb = timesteps.unsqueeze_dim(1).matmul(exponent.unsqueeze_dim(0));

        // Compute sin and cos
        let sin = emb.clone().sin();
        let cos = emb.cos();

        // Concatenate based on flip setting
        if self.flip_sin_to_cos {
            Tensor::cat(vec![cos, sin], 1)
        } else {
            Tensor::cat(vec![sin, cos], 1)
        }
    }
}

/// MLP that processes timestep embeddings.
///
/// Two linear layers with SiLU activation in between.
#[derive(Debug)]
pub(crate) struct TimestepEmbedding {
    /// First linear: channels -> time_embed_dim.
    pub(crate) linear_1: Linear<Backend>,
    /// Second linear: time_embed_dim -> time_embed_dim.
    pub(crate) linear_2: Linear<Backend>,
}

impl TimestepEmbedding {
    /// Load from safetensors weights.
    ///
    /// SD checkpoint keys:
    /// - `{prefix}.0.weight`, `{prefix}.0.bias` (linear_1)
    /// - `{prefix}.2.weight`, `{prefix}.2.bias` (linear_2)
    pub(crate) fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        // SD uses sequential numbering: 0 = linear, 1 = silu, 2 = linear
        let linear_1 = load_linear_dyn(
            tensors,
            &format!("{prefix}.0"),
            in_channels,
            out_channels,
            device,
        )?;
        let linear_2 = load_linear_dyn(
            tensors,
            &format!("{prefix}.2"),
            out_channels,
            out_channels,
            device,
        )?;

        Ok(Self { linear_1, linear_2 })
    }

    /// Forward pass: linear -> silu -> linear.
    pub(crate) fn forward(&self, x: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
        let x = self.linear_1.forward(x);
        let x = burn::tensor::activation::silu(x);
        self.linear_2.forward(x)
    }
}

/// Load a linear layer with dynamic dimensions (not const generic).
fn load_linear_dyn(
    tensors: &SafeTensors<'_>,
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
