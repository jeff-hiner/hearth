//! Complete CLIP text encoder.

use super::{config::ClipConfig, embeddings::ClipTextEmbeddings, transformer::ClipEncoder};
use crate::{layers::load_layer_norm, model_loader::LoadError, types::Backend};
use burn::{nn::LayerNorm, prelude::*};
use safetensors::SafeTensors;
use std::marker::PhantomData;

/// Complete CLIP text encoder model.
///
/// Combines embeddings, transformer, and final layer normalization.
#[derive(Debug)]
pub struct ClipTextEncoder<
    C: ClipConfig<HIDDEN, HEADS, FF, LAYERS, VOCAB, SEQ_LEN>,
    const HIDDEN: usize,
    const HEADS: usize,
    const FF: usize,
    const LAYERS: usize,
    const VOCAB: usize,
    const SEQ_LEN: usize,
    const QUICK_GELU: bool,
> {
    /// Token and position embeddings.
    embeddings: ClipTextEmbeddings<VOCAB, SEQ_LEN, HIDDEN>,
    /// Transformer encoder layers.
    encoder: ClipEncoder<HIDDEN, HEADS, FF, LAYERS, { f64::to_bits(1e-5) }, QUICK_GELU>,
    /// Final layer normalization.
    final_layer_norm: LayerNorm<Backend>,
    /// Phantom data to track the config type.
    _config: PhantomData<C>,
}

impl<
    C: ClipConfig<HIDDEN, HEADS, FF, LAYERS, VOCAB, SEQ_LEN>,
    const HIDDEN: usize,
    const HEADS: usize,
    const FF: usize,
    const LAYERS: usize,
    const VOCAB: usize,
    const SEQ_LEN: usize,
    const QUICK_GELU: bool,
> ClipTextEncoder<C, HIDDEN, HEADS, FF, LAYERS, VOCAB, SEQ_LEN, QUICK_GELU>
{
    /// Load the CLIP text encoder from an SD checkpoint.
    ///
    /// Uses the standard SD 1.5 prefix: `cond_stage_model.transformer.text_model`
    ///
    /// # Arguments
    /// * `tensors` - Parsed safetensors file
    /// * `device` - Device to load tensors onto
    ///
    /// # Returns
    /// Loaded encoder or error
    pub fn load(tensors: &SafeTensors<'_>, device: &Device<Backend>) -> Result<Self, LoadError> {
        Self::load_with_prefix(tensors, "cond_stage_model.transformer.text_model", device)
    }

    /// Load the CLIP text encoder with a custom prefix.
    ///
    /// # Arguments
    /// * `tensors` - Parsed safetensors file
    /// * `prefix` - Weight key prefix
    /// * `device` - Device to load tensors onto
    ///
    /// # Returns
    /// Loaded encoder or error
    pub fn load_with_prefix(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let embeddings =
            ClipTextEmbeddings::load(tensors, &format!("{prefix}.embeddings"), device)?;
        let encoder = ClipEncoder::load(tensors, &format!("{prefix}.encoder"), device)?;
        let final_layer_norm = load_layer_norm::<HIDDEN>(
            tensors,
            &format!("{prefix}.final_layer_norm"),
            C::LAYER_NORM_EPS,
            device,
        )?;

        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            _config: PhantomData,
        })
    }

    /// Encode token IDs to conditioning embeddings.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape `[batch, seq_len]`
    ///
    /// # Returns
    /// Conditioning tensor of shape `[batch, seq_len, hidden]`
    pub fn forward(&self, input_ids: Tensor<Backend, 2, Int>) -> Tensor<Backend, 3> {
        let hidden_states = self.embeddings.forward(input_ids);
        let hidden_states = self.encoder.forward(hidden_states);
        self.final_layer_norm.forward(hidden_states)
    }

    /// Encode token IDs, returning output after a specific layer.
    ///
    /// When `layer_idx` is `Some(n)`, returns the hidden state after layer `n`
    /// **without** applying `final_layer_norm`. This is used by SDXL's CLIP-L
    /// to extract penultimate-layer output (e.g., `Some(10)` for 12-layer model).
    ///
    /// When `layer_idx` is `None`, equivalent to [`forward`](Self::forward).
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs `[B, seq_len]`
    /// * `layer_idx` - Stop after this layer (inclusive), skipping final norm
    pub fn forward_hidden_layer(
        &self,
        input_ids: Tensor<Backend, 2, Int>,
        layer_idx: Option<usize>,
    ) -> Tensor<Backend, 3> {
        let hidden_states = self.embeddings.forward(input_ids);
        match layer_idx {
            Some(idx) => self.encoder.forward_to_layer(hidden_states, idx),
            None => {
                let hidden_states = self.encoder.forward(hidden_states);
                self.final_layer_norm.forward(hidden_states)
            }
        }
    }

    /// Apply LoRA deltas to this CLIP text encoder.
    ///
    /// The `model_path_root` is the checkpoint-relative path root (e.g.
    /// `"text_model"` for SD 1.5). LoRA keys are formed by stripping the
    /// checkpoint prefix and replacing `.` with `_`.
    ///
    /// Returns the total number of deltas applied.
    pub fn apply_lora(
        &mut self,
        model_path_root: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        self.encoder.apply_lora(
            &format!("{model_path_root}.encoder"),
            lora_prefix,
            lora,
            strength,
            device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::Sd15ClipTextEncoder;
    use crate::{model_loader::SafeTensorsFile, types::Backend};
    use burn::prelude::*;
    use std::path::Path;

    const MODEL_PATH: &str = "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors";

    #[test]
    fn load_clip_from_checkpoint() {
        let path = Path::new(MODEL_PATH);
        if !path.exists() {
            println!("Model not found at {MODEL_PATH}, skipping");
            return;
        }

        let file = SafeTensorsFile::open(path).expect("Failed to open checkpoint");
        let tensors = file.tensors().expect("Failed to parse safetensors");
        let device = Default::default();

        let encoder =
            Sd15ClipTextEncoder::load(&tensors, &device).expect("Failed to load CLIP encoder");

        // Create dummy token IDs: [BOS, some tokens, EOS, padding...]
        // BOS = 49406, EOS = 49407
        let mut token_ids = vec![49407i32; 77]; // Fill with EOS (padding)
        token_ids[0] = 49406; // BOS
        token_ids[1] = 320; // "a"
        token_ids[2] = 1125; // "photo"
        token_ids[3] = 539; // "of"
        token_ids[4] = 320; // "a"
        token_ids[5] = 2368; // "cat"
        token_ids[6] = 49407; // EOS

        let input_ids: Tensor<Backend, 1, Int> = Tensor::from_ints(token_ids.as_slice(), &device);
        let input_ids = input_ids.unsqueeze::<2>(); // [1, 77]

        let output = encoder.forward(input_ids);
        let shape = output.shape().dims();

        println!("CLIP output shape: {:?}", shape);
        assert_eq!(shape, [1, 77, 768], "Expected output shape [1, 77, 768]");
    }
}
