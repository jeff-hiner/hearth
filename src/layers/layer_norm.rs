//! LayerNorm loading utilities.

use crate::{
    model_loader::{LoadError, load_tensor_1d},
    types::Backend,
};
use burn::{module::Param, nn::LayerNorm, prelude::*};
use safetensors::SafeTensors;

/// Load a LayerNorm layer from safetensors.
pub(crate) fn load_layer_norm<const DIM: usize>(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    eps: f64,
    device: &Device<Backend>,
) -> Result<LayerNorm<Backend>, LoadError> {
    let weight = load_tensor_1d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let config = burn::nn::LayerNormConfig::new(DIM).with_epsilon(eps);
    let mut norm = config.init(device);

    norm.gamma = Param::from_tensor(weight);
    norm.beta = Some(Param::from_tensor(bias));

    Ok(norm)
}
