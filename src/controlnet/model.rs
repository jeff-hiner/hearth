//! ControlNet model: a copy of the UNet encoder that produces residuals.
//!
//! A ControlNet mirrors the UNet's encoder (down blocks + mid block) and adds:
//! - A hint encoder that processes the conditioning image
//! - Zero convolutions on each skip connection and mid block output
//!
//! The residuals are then added to the UNet's skip connections and mid block
//! to steer generation toward the conditioning signal.

use super::{hint::HintEncoder, output::ControlNetOutput};
use crate::{
    model_loader::{LoadError, load_tensor_1d, load_tensor_4d},
    types::Backend,
    unet::{
        InputBlock, UnetConfig,
        config::{Sd15Unet, SdxlUnet},
        downsample::Downsample2D,
        embeddings::{TimestepEmbedding, Timesteps},
        mid_block::UNetMidBlock,
        resnet::ResnetBlock2D,
        transformer::SpatialTransformer,
    },
};
use burn::{
    module::Param,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};
use safetensors::SafeTensors;
use std::marker::PhantomData;

/// ControlNet model generic over SD variant.
///
/// Mirrors the UNet encoder path and produces residuals that influence the
/// UNet's generation process. The hint encoder converts a conditioning image
/// (e.g., a depth map) into features that are added to the first conv output.
#[derive(Debug)]
pub struct ControlNet<C: UnetConfig> {
    /// Input convolution (4 -> base_channels).
    conv_in: Conv2d<Backend>,
    /// Sinusoidal timestep encoding.
    time_proj: Timesteps,
    /// Timestep MLP.
    time_embedding: TimestepEmbedding,
    /// Optional vector conditioning embedding (SDXL `label_emb`).
    label_emb: Option<TimestepEmbedding>,
    /// Down blocks (encoder path, same structure as UNet).
    input_blocks: Vec<InputBlock>,
    /// Middle block.
    mid_block: UNetMidBlock,
    /// Hint encoder: conditioning image → base_channels features.
    hint_encoder: HintEncoder,
    /// Zero convolutions, one per skip connection output.
    zero_convs: Vec<Conv2d<Backend>>,
    /// Zero convolution for mid block output.
    mid_block_out: Conv2d<Backend>,
    /// Config marker.
    _config: PhantomData<C>,
}

impl<C: UnetConfig> ControlNet<C> {
    /// Load a ControlNet from a safetensors file.
    ///
    /// Expects weights under the `control_model.` prefix.
    ///
    /// # Arguments
    /// * `tensors` - Parsed safetensors data
    /// * `hint_channels` - Number of channels in the conditioning image (typically 3)
    /// * `device` - Compute device
    pub fn load(
        tensors: &SafeTensors<'_>,
        hint_channels: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        Self::load_with_prefix(tensors, "control_model", hint_channels, device)
    }

    /// Load with a custom prefix.
    pub fn load_with_prefix(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        hint_channels: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let base = C::BASE_CHANNELS;
        let time_embed_dim = base * 4;

        // Input conv: 4 -> base_channels
        let conv_in = load_conv2d(
            tensors,
            &format!("{prefix}.input_blocks.0.0"),
            3,
            4,
            base,
            device,
        )?;

        // Time embeddings
        let time_proj = Timesteps::new(base, true, 0.0);
        let time_embedding = TimestepEmbedding::load(
            tensors,
            &format!("{prefix}.time_embed"),
            base,
            time_embed_dim,
            device,
        )?;

        // Optional vector conditioning (label_emb) for SDXL
        let label_emb = if C::ADM_IN_CHANNELS > 0 {
            Some(TimestepEmbedding::load(
                tensors,
                &format!("{prefix}.label_emb.0"),
                C::ADM_IN_CHANNELS,
                time_embed_dim,
                device,
            )?)
        } else {
            None
        };

        // Load input blocks (encoder path, same structure as UNet)
        let input_blocks = Self::load_input_blocks(tensors, prefix, time_embed_dim, device)?;

        // Middle block
        let mid_channels = base * C::CHANNEL_MULT[C::CHANNEL_MULT.len() - 1];
        let (mid_heads, mid_head_dim) = C::attention_params(mid_channels);
        let mid_block = UNetMidBlock::load(
            tensors,
            &format!("{prefix}.middle_block"),
            mid_channels,
            time_embed_dim,
            C::CONTEXT_DIM,
            mid_heads,
            mid_head_dim,
            C::MID_BLOCK_DEPTH,
            C::NORM_GROUPS,
            device,
        )?;

        // Hint encoder
        let hint_encoder = HintEncoder::load(
            tensors,
            &format!("{prefix}.input_hint_block"),
            hint_channels,
            base,
            device,
        )?;

        // Count total zero convolutions needed
        let num_zero_convs = Self::count_skip_outputs();

        // Load zero convolutions
        let mut zero_convs = Vec::with_capacity(num_zero_convs);
        for i in 0..num_zero_convs {
            let ch = Self::skip_channel_at(i, base);
            let zc = load_conv2d(
                tensors,
                &format!("{prefix}.zero_convs.{i}.0"),
                1,
                ch,
                ch,
                device,
            )?;
            zero_convs.push(zc);
        }

        // Mid block zero conv
        let mid_block_out = load_conv2d(
            tensors,
            &format!("{prefix}.middle_block_out.0"),
            1,
            mid_channels,
            mid_channels,
            device,
        )?;

        Ok(Self {
            conv_in,
            time_proj,
            time_embedding,
            label_emb,
            input_blocks,
            mid_block,
            hint_encoder,
            zero_convs,
            mid_block_out,
            _config: PhantomData,
        })
    }

    /// Run the ControlNet forward pass.
    ///
    /// # Arguments
    /// * `x` - Noisy latent `[B, 4, H, W]`
    /// * `hint` - Conditioning image `[B, hint_ch, H_full, W_full]` in [0, 1]
    /// * `timestep` - Diffusion timestep
    /// * `context` - CLIP conditioning
    /// * `y` - Optional vector conditioning (SDXL only)
    ///
    /// # Returns
    /// Residuals for UNet skip connections and mid block.
    pub fn forward(
        &self,
        x: Tensor<Backend, 4>,
        hint: &Tensor<Backend, 4>,
        timestep: Tensor<Backend, 1>,
        context: &Tensor<Backend, 3>,
        y: Option<&Tensor<Backend, 2>>,
    ) -> ControlNetOutput {
        // Compute time embedding
        let emb = self.compute_emb(timestep, y);

        // Encode hint image
        let hint_features = self.hint_encoder.forward(hint.clone());

        // Input conv + add hint
        let mut x = self.conv_in.forward(x) + hint_features;

        // Collect skip outputs and apply zero convolutions
        let mut down_residuals = Vec::new();
        let mut zc_idx = 0;

        // First skip: conv_in output (+ hint)
        down_residuals.push(self.zero_convs[zc_idx].forward(x.clone()));
        zc_idx += 1;

        // Down path
        for block in &self.input_blocks {
            for (i, resnet) in block.resnets.iter().enumerate() {
                x = resnet.forward(x, Some(&emb));
                if i < block.attns.len() {
                    x = block.attns[i].forward(x, Some(context));
                }
                down_residuals.push(self.zero_convs[zc_idx].forward(x.clone()));
                zc_idx += 1;
            }
            if let Some(ref ds) = block.downsample {
                x = ds.forward(x);
                down_residuals.push(self.zero_convs[zc_idx].forward(x.clone()));
                zc_idx += 1;
            }
        }

        // Middle block
        x = self.mid_block.forward(x, Some(&emb), Some(context));
        let mid_residual = self.mid_block_out.forward(x);

        ControlNetOutput {
            down_residuals,
            mid_residual,
        }
    }

    /// Compute the combined embedding from timestep (and optional vector conditioning).
    fn compute_emb(
        &self,
        timestep: Tensor<Backend, 1>,
        y: Option<&Tensor<Backend, 2>>,
    ) -> Tensor<Backend, 2> {
        let t_emb = self.time_proj.forward(timestep);
        let emb: Tensor<Backend, 2> = self.time_embedding.forward(t_emb);

        match (&self.label_emb, y) {
            (Some(label_emb), Some(y)) => {
                let y_emb: Tensor<Backend, 2> = label_emb.forward(y.clone());
                emb + y_emb
            }
            _ => emb,
        }
    }

    /// Load input (down) blocks from checkpoint (same structure as UNet encoder).
    fn load_input_blocks(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        time_embed_dim: usize,
        device: &Device<Backend>,
    ) -> Result<Vec<InputBlock>, LoadError> {
        let base = C::BASE_CHANNELS;
        let mult = C::CHANNEL_MULT;
        let layers = C::LAYERS_PER_BLOCK;

        let mut blocks = Vec::new();
        let mut block_idx = 1usize; // input_blocks.0 is conv_in

        for (level, &m) in mult.iter().enumerate() {
            let out_ch = base * m;
            let in_ch = if level == 0 {
                base
            } else {
                base * mult[level - 1]
            };
            let depth = C::TRANSFORMER_DEPTH[level];
            let has_attn = depth > 0;
            let is_last = level == mult.len() - 1;

            let mut resnets = Vec::new();
            let mut attns = Vec::new();

            for layer in 0..layers {
                let res_in_ch = if layer == 0 { in_ch } else { out_ch };

                let resnet = ResnetBlock2D::load(
                    tensors,
                    &format!("{prefix}.input_blocks.{block_idx}.0"),
                    res_in_ch,
                    out_ch,
                    time_embed_dim,
                    C::NORM_GROUPS,
                    device,
                )?;
                resnets.push(resnet);

                if has_attn {
                    let (num_heads, head_dim) = C::attention_params(out_ch);
                    let attn = SpatialTransformer::load(
                        tensors,
                        &format!("{prefix}.input_blocks.{block_idx}.1"),
                        out_ch,
                        num_heads,
                        head_dim,
                        C::CONTEXT_DIM,
                        depth,
                        C::NORM_GROUPS,
                        device,
                    )?;
                    attns.push(attn);
                }

                block_idx += 1;
            }

            // Downsample (except for last level)
            let downsample = if !is_last {
                let ds = Downsample2D::load(
                    tensors,
                    &format!("{prefix}.input_blocks.{block_idx}.0"),
                    out_ch,
                    device,
                )?;
                block_idx += 1;
                Some(ds)
            } else {
                None
            };

            blocks.push(InputBlock {
                resnets,
                attns,
                downsample,
            });
        }

        Ok(blocks)
    }

    /// Count the total number of skip connection outputs.
    fn count_skip_outputs() -> usize {
        let mult = C::CHANNEL_MULT;
        let layers = C::LAYERS_PER_BLOCK;
        let num_levels = mult.len();
        // 1 (conv_in) + layers_per_block * num_levels + (num_levels - 1) downsamples
        1 + layers * num_levels + (num_levels - 1)
    }

    /// Get the channel count for the skip connection at a given index.
    fn skip_channel_at(idx: usize, base: usize) -> usize {
        if idx == 0 {
            return base; // conv_in output
        }

        let mult = C::CHANNEL_MULT;
        let layers = C::LAYERS_PER_BLOCK;
        let mut current = 1; // skip idx 0 is conv_in

        for (level, &m) in mult.iter().enumerate() {
            let ch = base * m;
            let is_last = level == mult.len() - 1;

            for _ in 0..layers {
                if current == idx {
                    return ch;
                }
                current += 1;
            }

            if !is_last {
                if current == idx {
                    return ch;
                }
                current += 1;
            }
        }

        base // fallback (shouldn't happen)
    }
}

/// SD 1.5 ControlNet type alias.
pub type Sd15ControlNet = ControlNet<Sd15Unet>;

/// SDXL ControlNet type alias.
pub type SdxlControlNet = ControlNet<SdxlUnet>;

/// Load Conv2d with dynamic dimensions (1x1 kernel, stride 1).
fn load_conv2d(
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
