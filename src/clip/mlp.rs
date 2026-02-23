//! CLIP feed-forward MLP with configurable activation.

use crate::{
    layers::load_linear,
    lora::{apply_lora_linear, lora_key_base},
    model_loader::LoadError,
    types::Backend,
};
use burn::{nn::Linear, prelude::*};
use safetensors::SafeTensors;

/// Feed-forward MLP for CLIP transformer layers.
///
/// When `QUICK_GELU` is true, uses QuickGELU activation: `x * sigmoid(1.702 * x)`.
/// When false, uses standard GELU.
#[derive(Debug)]
pub(crate) struct ClipMlp<const HIDDEN: usize, const FF: usize, const QUICK_GELU: bool> {
    /// First linear layer: hidden -> ff.
    fc1: Linear<Backend>,
    /// Second linear layer: ff -> hidden.
    fc2: Linear<Backend>,
}

impl<const HIDDEN: usize, const FF: usize, const QUICK_GELU: bool> ClipMlp<HIDDEN, FF, QUICK_GELU> {
    /// Load MLP weights from safetensors.
    ///
    /// Expects tensors at:
    /// - `{prefix}.fc1.{weight,bias}`
    /// - `{prefix}.fc2.{weight,bias}`
    pub(crate) fn load(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let fc1 = load_linear::<HIDDEN, FF>(tensors, &format!("{prefix}.fc1"), device)?;
        let fc2 = load_linear::<FF, HIDDEN>(tensors, &format!("{prefix}.fc2"), device)?;

        Ok(Self { fc1, fc2 })
    }

    /// Forward pass through MLP.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor of shape `[batch, seq_len, hidden]`
    ///
    /// # Returns
    /// Output tensor of shape `[batch, seq_len, hidden]`
    pub(crate) fn forward(&self, hidden_states: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
        let hidden = self.fc1.forward(hidden_states);
        let hidden = if QUICK_GELU {
            quick_gelu(hidden)
        } else {
            burn::tensor::activation::gelu(hidden)
        };
        self.fc2.forward(hidden)
    }

    /// Apply LoRA deltas to MLP projections.
    ///
    /// Targets: `fc1`, `fc2`
    pub(crate) fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;

        let key = lora_key_base(lora_prefix, &format!("{prefix}.fc1"));
        if apply_lora_linear(&mut self.fc1, &key, lora, strength, device)? {
            count += 1;
        }

        let key = lora_key_base(lora_prefix, &format!("{prefix}.fc2"));
        if apply_lora_linear(&mut self.fc2, &key, lora, strength, device)? {
            count += 1;
        }

        Ok(count)
    }
}

/// QuickGELU activation function.
///
/// Approximation of GELU used by OpenAI CLIP: `x * sigmoid(1.702 * x)`
fn quick_gelu(x: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
    x.clone() * burn::tensor::activation::sigmoid(x * 1.702)
}
