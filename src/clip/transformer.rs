//! CLIP transformer encoder (stack of layers).

use super::layer::ClipEncoderLayer;
use crate::{model_loader::LoadError, types::Backend};
use burn::prelude::*;
use safetensors::SafeTensors;

/// CLIP transformer encoder consisting of stacked transformer layers.
#[derive(Debug)]
pub(crate) struct ClipEncoder<
    const HIDDEN: usize,
    const HEADS: usize,
    const FF: usize,
    const LAYERS: usize,
    const LAYER_NORM_EPS: u64,
    const QUICK_GELU: bool,
> {
    /// The transformer layers.
    layers: Vec<ClipEncoderLayer<HIDDEN, HEADS, FF, LAYER_NORM_EPS, QUICK_GELU>>,
}

impl<
    const HIDDEN: usize,
    const HEADS: usize,
    const FF: usize,
    const LAYERS: usize,
    const LAYER_NORM_EPS: u64,
    const QUICK_GELU: bool,
> ClipEncoder<HIDDEN, HEADS, FF, LAYERS, LAYER_NORM_EPS, QUICK_GELU>
{
    /// Load the transformer encoder from safetensors.
    ///
    /// Expects tensors at `{prefix}.layers.{0..LAYERS-1}.*`
    pub(crate) fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let mut layers = Vec::with_capacity(LAYERS);

        for i in 0..LAYERS {
            let layer_prefix = format!("{prefix}.layers.{i}");
            let layer = ClipEncoderLayer::load(tensors, &layer_prefix, device)?;
            layers.push(layer);
        }

        Ok(Self { layers })
    }

    /// Forward pass through all transformer layers.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor of shape `[batch, seq_len, hidden]`
    ///
    /// # Returns
    /// Output tensor of shape `[batch, seq_len, hidden]`
    pub(crate) fn forward(&self, mut hidden_states: Tensor<Backend, 3>) -> Tensor<Backend, 3> {
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states);
        }
        hidden_states
    }

    /// Apply LoRA deltas to all encoder layers.
    ///
    /// Iterates layers with prefix `{prefix}.layers.{i}`.
    pub(crate) fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            count += layer.apply_lora(
                &format!("{prefix}.layers.{i}"),
                lora_prefix,
                lora,
                strength,
                device,
            )?;
        }
        Ok(count)
    }

    /// Forward pass through transformer layers up to (and including) `layer_idx`.
    ///
    /// Used by SDXL to extract penultimate layer output from CLIP-L.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor of shape `[batch, seq_len, hidden]`
    /// * `layer_idx` - Last layer index to run (inclusive, 0-based)
    ///
    /// # Returns
    /// Output tensor of shape `[batch, seq_len, hidden]`
    pub(crate) fn forward_to_layer(
        &self,
        mut hidden_states: Tensor<Backend, 3>,
        layer_idx: usize,
    ) -> Tensor<Backend, 3> {
        for layer in self.layers.iter().take(layer_idx + 1) {
            hidden_states = layer.forward(hidden_states);
        }
        hidden_states
    }
}
