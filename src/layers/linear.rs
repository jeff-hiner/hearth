//! Linear layer loading utilities.

use crate::{
    model_loader::{LoadError, load_tensor_1d, load_tensor_2d},
    types::Backend,
};
use burn::{module::Param, nn::Linear, prelude::*};

/// Load a Linear layer from safetensors.
///
/// PyTorch stores linear weights as `[out_features, in_features]`, but Burn expects
/// `[in_features, out_features]`, so we transpose on load.
pub(crate) fn load_linear<const IN: usize, const OUT: usize>(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    device: &Device<Backend>,
) -> Result<Linear<Backend>, LoadError> {
    let weight = load_tensor_2d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    // Transpose: PyTorch [OUT, IN] -> Burn [IN, OUT]
    let weight = weight.transpose();

    let config = burn::nn::LinearConfig::new(IN, OUT);
    let mut linear = config.init(device);

    linear.weight = Param::from_tensor(weight);
    linear.bias = Some(Param::from_tensor(bias));

    Ok(linear)
}
