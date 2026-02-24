//! CLIP text embeddings layer.
//!
//! Combines token embeddings with learned positional embeddings.

use crate::{
    model_loader::{LoadError, load_tensor_2d},
    types::Backend,
};
use burn::{module::Param, nn::Embedding, prelude::*};
use safetensors::SafeTensors;

/// CLIP text embeddings combining token and position embeddings.
///
/// The forward pass looks up token embeddings and adds learned position embeddings.
#[derive(Debug)]
pub(crate) struct ClipTextEmbeddings<const VOCAB: usize, const SEQ_LEN: usize, const HIDDEN: usize>
{
    /// Token embedding lookup table.
    token_embedding: Embedding<Backend>,
    /// Learned positional embeddings.
    position_embedding: Tensor<Backend, 2>,
}

impl<const VOCAB: usize, const SEQ_LEN: usize, const HIDDEN: usize>
    ClipTextEmbeddings<VOCAB, SEQ_LEN, HIDDEN>
{
    /// Load embeddings from safetensors weights.
    ///
    /// Expects tensors at:
    /// - `{prefix}.token_embedding.weight` shape `[VOCAB, HIDDEN]`
    /// - `{prefix}.position_embedding.weight` shape `[SEQ_LEN, HIDDEN]`
    pub(crate) fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let token_weight =
            load_tensor_2d(tensors, &format!("{prefix}.token_embedding.weight"), device)?;
        let position_embedding = load_tensor_2d(
            tensors,
            &format!("{prefix}.position_embedding.weight"),
            device,
        )?;

        let config = burn::nn::EmbeddingConfig::new(VOCAB, HIDDEN);
        let mut token_embedding = config.init(device);
        token_embedding.weight = Param::from_tensor(token_weight);

        Ok(Self {
            token_embedding,
            position_embedding,
        })
    }

    /// Forward pass: token lookup + positional embedding.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs tensor of shape `[batch, seq_len]`
    ///
    /// # Returns
    /// Combined embeddings of shape `[batch, seq_len, hidden]`
    pub(crate) fn forward(&self, input_ids: Tensor<Backend, 2, Int>) -> Tensor<Backend, 3> {
        let [batch, seq_len] = input_ids.shape().dims();

        // Token embeddings: [batch, seq_len, hidden]
        let token_embeds = self.token_embedding.forward(input_ids);

        // Position embeddings: [seq_len, hidden] -> [1, seq_len, hidden] -> broadcast
        let pos_embeds = self
            .position_embedding
            .clone()
            .slice([0..seq_len, 0..HIDDEN])
            .unsqueeze::<3>()
            .expand([batch, seq_len, HIDDEN]);

        token_embeds + pos_embeds
    }
}
