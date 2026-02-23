//! Main UNet model for Stable Diffusion.

use super::{
    config::UnetConfig,
    downsample::Downsample2D,
    embeddings::{TimestepEmbedding, Timesteps},
    mid_block::UNetMidBlock,
    resnet::ResnetBlock2D,
    transformer::SpatialTransformer,
};
use crate::{
    clip::{Sd15Conditioning, SdxlConditioning},
    lora::{
        LoraFormat, apply_lora_conv2d, apply_lora_linear, detect_unet_lora_format, lora_key_base,
    },
    model_loader::{LoadError, load_tensor_1d, load_tensor_4d},
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
use std::{
    hint::black_box,
    marker::PhantomData,
    time::{Duration, Instant},
};

/// Concatenate two tensors along the channel dimension (dim 1) using in-place
/// slice assignment instead of `Tensor::cat`, avoiding the concat kernel dispatch.
fn cat_along_dim1(a: Tensor<Backend, 4>, b: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
    let [batch, ca, h, w] = a.shape().dims();
    let [_, cb, _, _] = b.shape().dims();
    let buf = Tensor::empty([batch, ca + cb, h, w], &a.device());
    let buf = buf.slice_assign([0..batch, 0..ca, 0..h, 0..w], a);
    buf.slice_assign([0..batch, ca..(ca + cb), 0..h, 0..w], b)
}

/// Force GPU synchronization by reading a single element back to CPU.
fn sync_gpu<const D: usize>(tensor: &Tensor<Backend, D>) {
    let flat = tensor.clone().flatten::<1>(0, D - 1);
    let data = flat.slice(0..1).into_data();
    black_box(data);
}

// ---------------------------------------------------------------------------
// Conditioning types
// ---------------------------------------------------------------------------

/// Trait for UNet conditioning data that can be batched for CFG.
pub trait UnetConditioning: std::fmt::Debug {
    /// Concatenate positive and negative conditioning along batch dim for CFG.
    fn batch_cfg(positive: &Self, negative: &Self) -> Self;

    /// Cross-attention context tensor `[B, seq_len, context_dim]`.
    fn context(&self) -> &Tensor<Backend, 3>;

    /// Optional vector conditioning (SDXL ADM embedding) `[B, adm_dim]`.
    fn y(&self) -> Option<&Tensor<Backend, 2>>;
}

// ---------------------------------------------------------------------------
// DenoisingUnet trait
// ---------------------------------------------------------------------------

/// Trait for denoising UNet models, abstracting over SD variants.
pub trait DenoisingUnet: std::fmt::Debug {
    /// Conditioning data type for this UNet variant.
    type Cond: UnetConditioning;

    /// Run a single denoising step.
    fn forward(
        &self,
        x: Tensor<Backend, 4>,
        timestep: Tensor<Backend, 1>,
        cond: &Self::Cond,
    ) -> Tensor<Backend, 4>;

    /// Run a single denoising step with timing instrumentation.
    fn forward_timed(
        &self,
        x: Tensor<Backend, 4>,
        timestep: Tensor<Backend, 1>,
        cond: &Self::Cond,
    ) -> (Tensor<Backend, 4>, UnetTimingStats);
}

// ---------------------------------------------------------------------------
// UNet blocks
// ---------------------------------------------------------------------------

/// Input block: ResNet blocks + optional attention + optional downsample.
#[derive(Debug)]
pub(crate) struct InputBlock {
    /// ResNet blocks in this level.
    pub(crate) resnets: Vec<ResnetBlock2D>,
    /// Optional attention (SpatialTransformer).
    pub(crate) attns: Vec<SpatialTransformer>,
    /// Optional downsampler.
    pub(crate) downsample: Option<Downsample2D>,
}

/// Output block: ResNet blocks + optional attention + optional upsample.
#[derive(Debug)]
struct OutputBlock {
    /// ResNet blocks in this level.
    resnets: Vec<ResnetBlock2D>,
    /// Optional attention (SpatialTransformer).
    attns: Vec<SpatialTransformer>,
    /// Optional upsampler.
    upsample: Option<Upsample2D>,
}

/// 2D upsampling via interpolation + convolution.
#[derive(Debug)]
struct Upsample2D {
    /// Convolution after upsampling.
    conv: Conv2d<Backend>,
}

impl Upsample2D {
    fn load(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        channels: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let weight = load_tensor_4d(tensors, &format!("{prefix}.conv.weight"), device)?;
        let bias = load_tensor_1d(tensors, &format!("{prefix}.conv.bias"), device)?;

        let config = Conv2dConfig::new([channels, channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1));

        let mut conv = config.init(device);
        conv.weight = Param::from_tensor(weight);
        conv.bias = Some(Param::from_tensor(bias));

        Ok(Self { conv })
    }

    fn forward(&self, x: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        // Nearest neighbor 2x upsample then conv
        let [b, c, h, w] = x.shape().dims();
        let x = x
            .reshape([b, c, h, 1, w, 1])
            .repeat_dim(3, 2)
            .repeat_dim(5, 2)
            .reshape([b, c, h * 2, w * 2]);
        self.conv.forward(x)
    }

    /// Apply LoRA delta to the upsample convolution.
    fn apply_lora(
        &mut self,
        prefix: &str,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let key = lora_key_base(lora_prefix, &format!("{prefix}.conv"));
        if apply_lora_conv2d(&mut self.conv, &key, lora, strength, device)? {
            Ok(1)
        } else {
            Ok(0)
        }
    }
}

// ---------------------------------------------------------------------------
// UNet model
// ---------------------------------------------------------------------------

/// UNet denoising model.
#[derive(Debug)]
pub struct Unet<C: UnetConfig> {
    /// Input convolution (4 -> base_channels).
    conv_in: Conv2d<Backend>,
    /// Sinusoidal timestep encoding.
    time_proj: Timesteps,
    /// Timestep MLP.
    time_embedding: TimestepEmbedding,
    /// Optional vector conditioning embedding (SDXL `label_emb`).
    label_emb: Option<TimestepEmbedding>,
    /// Down blocks (encoder path).
    input_blocks: Vec<InputBlock>,
    /// Middle block.
    mid_block: UNetMidBlock,
    /// Up blocks (decoder path).
    output_blocks: Vec<OutputBlock>,
    /// Output normalization.
    out_norm: GroupNorm<Backend>,
    /// Output convolution (base_channels -> 4).
    out_conv: Conv2d<Backend>,
    /// Config marker.
    _config: PhantomData<C>,
}

impl<C: UnetConfig> Unet<C> {
    /// Load UNet from SD checkpoint.
    ///
    /// Uses prefix `model.diffusion_model`.
    pub fn load(
        tensors: &safetensors::SafeTensors<'_>,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        Self::load_with_prefix(tensors, "model.diffusion_model", device)
    }

    /// Load UNet with custom prefix.
    pub fn load_with_prefix(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
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

        // Load input blocks (down path)
        // SD structure: input_blocks.1-11 contain the actual down blocks
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

        // Load output blocks (up path)
        let output_blocks = Self::load_output_blocks(tensors, prefix, time_embed_dim, device)?;

        // Output layers
        let out_norm = load_group_norm(
            tensors,
            &format!("{prefix}.out.0"),
            C::NORM_GROUPS,
            base,
            device,
        )?;
        let out_conv = load_conv2d(tensors, &format!("{prefix}.out.2"), 3, base, 4, device)?;

        Ok(Self {
            conv_in,
            time_proj,
            time_embedding,
            label_emb,
            input_blocks,
            mid_block,
            output_blocks,
            out_norm,
            out_conv,
            _config: PhantomData,
        })
    }

    /// Load input (down) blocks from checkpoint.
    fn load_input_blocks(
        tensors: &safetensors::SafeTensors<'_>,
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
            // SD structure: downsample blocks have a single .0 sub-module containing the op
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

    /// Load output (up) blocks from checkpoint.
    fn load_output_blocks(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: &str,
        time_embed_dim: usize,
        device: &Device<Backend>,
    ) -> Result<Vec<OutputBlock>, LoadError> {
        let base = C::BASE_CHANNELS;
        let mult = C::CHANNEL_MULT;
        let layers = C::LAYERS_PER_BLOCK;

        let mut blocks = Vec::new();
        let mut block_idx = 0usize;

        // Output blocks go in reverse order of input blocks
        for (level, &m) in mult.iter().enumerate().rev() {
            let out_ch = base * m;
            let depth = C::TRANSFORMER_DEPTH[level];
            let has_attn = depth > 0;
            let is_last = level == 0;

            // Skip channels from encoder
            let _skip_ch = if level == mult.len() - 1 {
                out_ch
            } else {
                base * mult[level + 1]
            };

            let mut resnets = Vec::new();
            let mut attns = Vec::new();

            // Output blocks have layers_per_block + 1 resnets
            for layer in 0..(layers + 1) {
                // Input channels: previous out + skip connection
                let prev_ch = if layer == 0 {
                    if level == mult.len() - 1 {
                        out_ch
                    } else {
                        base * mult[level + 1]
                    }
                } else {
                    out_ch
                };
                let layer_skip_ch = if layer == layers {
                    if level == 0 {
                        base
                    } else {
                        base * mult[level - 1]
                    }
                } else {
                    out_ch
                };
                let res_in_ch = prev_ch + layer_skip_ch;

                let resnet = ResnetBlock2D::load(
                    tensors,
                    &format!("{prefix}.output_blocks.{block_idx}.0"),
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
                        &format!("{prefix}.output_blocks.{block_idx}.1"),
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

            // Upsample (except for last level = level 0)
            // The upsample is part of the last resnet block at each level
            let upsample = if !is_last {
                // Upsample is at output_blocks.{prev_idx}.{1 or 2} depending on attention
                let up_idx = block_idx - 1;
                let up_sub = if has_attn { 2 } else { 1 };
                let up = Upsample2D::load(
                    tensors,
                    &format!("{prefix}.output_blocks.{up_idx}.{up_sub}"),
                    out_ch,
                    device,
                )?;
                Some(up)
            } else {
                None
            };

            blocks.push(OutputBlock {
                resnets,
                attns,
                upsample,
            });
        }

        Ok(blocks)
    }

    /// Shared forward pass logic (used by both SD 1.5 and SDXL trait impls).
    ///
    /// # Arguments
    /// * `x` - Noisy latent `[B, 4, H, W]`
    /// * `emb` - Combined time + optional vector embedding `[B, time_embed_dim]`
    /// * `context` - CLIP conditioning `[B, 77, context_dim]`
    /// * `controlnet_residuals` - Optional ControlNet residuals to add to skip
    ///   connections and mid block output
    pub(crate) fn forward_with_emb(
        &self,
        x: Tensor<Backend, 4>,
        emb: Tensor<Backend, 2>,
        context: &Tensor<Backend, 3>,
        controlnet_residuals: Option<&crate::controlnet::ControlNetOutput>,
    ) -> Tensor<Backend, 4> {
        // Input conv
        let mut x = self.conv_in.forward(x);

        // Down path with skip connections
        let mut skips = vec![x.clone()];
        for block in &self.input_blocks {
            for (i, resnet) in block.resnets.iter().enumerate() {
                x = resnet.forward(x, Some(&emb));
                if i < block.attns.len() {
                    x = block.attns[i].forward(x, Some(context));
                }
                skips.push(x.clone());
            }
            if let Some(ref ds) = block.downsample {
                x = ds.forward(x);
                skips.push(x.clone());
            }
        }

        // Middle block
        x = self.mid_block.forward(x, Some(&emb), Some(context));

        // Apply ControlNet mid-block residual
        if let Some(r) = controlnet_residuals {
            x = x + r.mid_residual.clone();
        }

        // Up path with skip connections
        for block in &self.output_blocks {
            for (i, resnet) in block.resnets.iter().enumerate() {
                // Concatenate skip connection (with optional ControlNet residual)
                let mut skip = skips.pop().expect("skip connection mismatch");
                if let Some(r) = controlnet_residuals {
                    // skips.len() after pop gives the index of the popped element
                    let idx = skips.len();
                    if idx < r.down_residuals.len() {
                        skip = skip + r.down_residuals[idx].clone();
                    }
                }
                x = cat_along_dim1(x, skip);
                x = resnet.forward(x, Some(&emb));
                if i < block.attns.len() {
                    x = block.attns[i].forward(x, Some(context));
                }
            }
            if let Some(ref up) = block.upsample {
                x = up.forward(x);
            }
        }

        // Output
        let x = self.out_norm.forward(x);
        let x = silu(x);
        self.out_conv.forward(x)
    }

    /// Compute the combined embedding from timestep (and optional vector conditioning).
    pub(crate) fn compute_emb(
        &self,
        timestep: Tensor<Backend, 1>,
        y: Option<&Tensor<Backend, 2>>,
    ) -> Tensor<Backend, 2> {
        let t_emb = self.time_proj.forward(timestep);
        let emb = self.time_embedding.forward(t_emb);

        match (&self.label_emb, y) {
            (Some(label_emb), Some(y)) => emb + label_emb.forward(y.clone()),
            _ => emb,
        }
    }

    /// Apply LoRA deltas to all layers in this UNet.
    ///
    /// Iterates the stored block structure and reconstructs checkpoint-relative
    /// paths (e.g. `input_blocks.3.1`) to map LoRA keys to the correct layers.
    ///
    /// Returns the total number of deltas applied.
    pub fn apply_lora(
        &mut self,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let format = detect_unet_lora_format(lora, lora_prefix);
        let Some(format) = format else {
            tracing::warn!("LoRA contains no recognized UNet keys for prefix {lora_prefix:?}");
            return Ok(0);
        };
        tracing::debug!(?format, "detected UNet LoRA key format");

        match format {
            LoraFormat::Ldm => self.apply_lora_ldm(lora_prefix, lora, strength, device),
            LoraFormat::Diffusers => self.apply_lora_diffusers(lora_prefix, lora, strength, device),
        }
    }

    /// Apply LoRA using CompVis/ldm key format (`input_blocks`, `middle_block`, `output_blocks`).
    fn apply_lora_ldm(
        &mut self,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;

        // conv_in (input_blocks.0.0)
        let key = lora_key_base(lora_prefix, "input_blocks.0.0");
        if apply_lora_conv2d(&mut self.conv_in, &key, lora, strength, device)? {
            count += 1;
        }

        // time_embed.0 and time_embed.2
        let key = lora_key_base(lora_prefix, "time_embed.0");
        if apply_lora_linear(
            &mut self.time_embedding.linear_1,
            &key,
            lora,
            strength,
            device,
        )? {
            count += 1;
        }
        let key = lora_key_base(lora_prefix, "time_embed.2");
        if apply_lora_linear(
            &mut self.time_embedding.linear_2,
            &key,
            lora,
            strength,
            device,
        )? {
            count += 1;
        }

        // label_emb.0.0 and label_emb.0.2 (SDXL only)
        if let Some(ref mut label_emb) = self.label_emb {
            let key = lora_key_base(lora_prefix, "label_emb.0.0");
            if apply_lora_linear(&mut label_emb.linear_1, &key, lora, strength, device)? {
                count += 1;
            }
            let key = lora_key_base(lora_prefix, "label_emb.0.2");
            if apply_lora_linear(&mut label_emb.linear_2, &key, lora, strength, device)? {
                count += 1;
            }
        }

        // Input blocks — reconstruct block_idx from structure
        let mut block_idx = 1usize;
        for input_block in &mut self.input_blocks {
            for (ri, resnet) in input_block.resnets.iter_mut().enumerate() {
                count += resnet.apply_lora(
                    &format!("input_blocks.{block_idx}.0"),
                    lora_prefix,
                    lora,
                    strength,
                    device,
                )?;
                if ri < input_block.attns.len() {
                    count += input_block.attns[ri].apply_lora(
                        &format!("input_blocks.{block_idx}.1"),
                        lora_prefix,
                        lora,
                        strength,
                        device,
                    )?;
                }
                block_idx += 1;
            }
            if let Some(ref mut ds) = input_block.downsample {
                count += ds.apply_lora(
                    &format!("input_blocks.{block_idx}.0"),
                    lora_prefix,
                    lora,
                    strength,
                    device,
                )?;
                block_idx += 1;
            }
        }

        // Middle block
        count += self
            .mid_block
            .apply_lora("middle_block", lora_prefix, lora, strength, device)?;

        // Output blocks — reconstruct block_idx from structure
        let mut block_idx = 0usize;
        for output_block in &mut self.output_blocks {
            for (ri, resnet) in output_block.resnets.iter_mut().enumerate() {
                count += resnet.apply_lora(
                    &format!("output_blocks.{block_idx}.0"),
                    lora_prefix,
                    lora,
                    strength,
                    device,
                )?;
                if ri < output_block.attns.len() {
                    count += output_block.attns[ri].apply_lora(
                        &format!("output_blocks.{block_idx}.1"),
                        lora_prefix,
                        lora,
                        strength,
                        device,
                    )?;
                }
                block_idx += 1;
            }
            // Upsample is at the last block entry of the level
            if let Some(ref mut up) = output_block.upsample {
                let up_idx = block_idx - 1;
                let up_sub = if output_block.attns.is_empty() { 1 } else { 2 };
                count += up.apply_lora(
                    &format!("output_blocks.{up_idx}.{up_sub}"),
                    lora_prefix,
                    lora,
                    strength,
                    device,
                )?;
            }
        }

        // out.2 (output conv)
        let key = lora_key_base(lora_prefix, "out.2");
        if apply_lora_conv2d(&mut self.out_conv, &key, lora, strength, device)? {
            count += 1;
        }

        Ok(count)
    }

    /// Apply LoRA using HuggingFace diffusers key format (`down_blocks`, `mid_block`, `up_blocks`).
    ///
    /// Diffusers uses `down_blocks.{level}.attentions.{idx}` instead of flat `input_blocks.N.1`,
    /// and `down_blocks.{level}.resnets.{idx}` instead of `input_blocks.N.0`. The sub-module
    /// internal keys (`.transformer_blocks.0.attn1.to_q`, `.proj_in`, etc.) are the same.
    fn apply_lora_diffusers(
        &mut self,
        lora_prefix: &str,
        lora: &SafeTensors<'_>,
        strength: f32,
        device: &Device<Backend>,
    ) -> Result<usize, LoadError> {
        let mut count = 0;

        // conv_in
        let key = lora_key_base(lora_prefix, "conv_in");
        if apply_lora_conv2d(&mut self.conv_in, &key, lora, strength, device)? {
            count += 1;
        }

        // time_embedding.linear_1, time_embedding.linear_2
        let key = lora_key_base(lora_prefix, "time_embedding.linear_1");
        if apply_lora_linear(
            &mut self.time_embedding.linear_1,
            &key,
            lora,
            strength,
            device,
        )? {
            count += 1;
        }
        let key = lora_key_base(lora_prefix, "time_embedding.linear_2");
        if apply_lora_linear(
            &mut self.time_embedding.linear_2,
            &key,
            lora,
            strength,
            device,
        )? {
            count += 1;
        }

        // add_embedding.linear_1, add_embedding.linear_2 (SDXL only)
        if let Some(ref mut label_emb) = self.label_emb {
            let key = lora_key_base(lora_prefix, "add_embedding.linear_1");
            if apply_lora_linear(&mut label_emb.linear_1, &key, lora, strength, device)? {
                count += 1;
            }
            let key = lora_key_base(lora_prefix, "add_embedding.linear_2");
            if apply_lora_linear(&mut label_emb.linear_2, &key, lora, strength, device)? {
                count += 1;
            }
        }

        // Down blocks: down_blocks.{level}.resnets.{idx} / attentions.{idx} / downsamplers.0
        for (level, input_block) in self.input_blocks.iter_mut().enumerate() {
            for (ri, resnet) in input_block.resnets.iter_mut().enumerate() {
                count += resnet.apply_lora(
                    &format!("down_blocks.{level}.resnets.{ri}"),
                    lora_prefix,
                    lora,
                    strength,
                    device,
                )?;
                if ri < input_block.attns.len() {
                    count += input_block.attns[ri].apply_lora(
                        &format!("down_blocks.{level}.attentions.{ri}"),
                        lora_prefix,
                        lora,
                        strength,
                        device,
                    )?;
                }
            }
            if let Some(ref mut ds) = input_block.downsample {
                count += ds.apply_lora(
                    &format!("down_blocks.{level}.downsamplers.0"),
                    lora_prefix,
                    lora,
                    strength,
                    device,
                )?;
            }
        }

        // Mid block: mid_block.resnets.{0,1} and mid_block.attentions.0
        count += self.mid_block.resnet1.apply_lora(
            "mid_block.resnets.0",
            lora_prefix,
            lora,
            strength,
            device,
        )?;
        count += self.mid_block.attn.apply_lora(
            "mid_block.attentions.0",
            lora_prefix,
            lora,
            strength,
            device,
        )?;
        count += self.mid_block.resnet2.apply_lora(
            "mid_block.resnets.1",
            lora_prefix,
            lora,
            strength,
            device,
        )?;

        // Up blocks: up_blocks.{level}.resnets.{idx} / attentions.{idx} / upsamplers.0
        for (level, output_block) in self.output_blocks.iter_mut().enumerate() {
            for (ri, resnet) in output_block.resnets.iter_mut().enumerate() {
                count += resnet.apply_lora(
                    &format!("up_blocks.{level}.resnets.{ri}"),
                    lora_prefix,
                    lora,
                    strength,
                    device,
                )?;
                if ri < output_block.attns.len() {
                    count += output_block.attns[ri].apply_lora(
                        &format!("up_blocks.{level}.attentions.{ri}"),
                        lora_prefix,
                        lora,
                        strength,
                        device,
                    )?;
                }
            }
            if let Some(ref mut up) = output_block.upsample {
                count += up.apply_lora(
                    &format!("up_blocks.{level}.upsamplers.0"),
                    lora_prefix,
                    lora,
                    strength,
                    device,
                )?;
            }
        }

        // conv_out
        let key = lora_key_base(lora_prefix, "conv_out");
        if apply_lora_conv2d(&mut self.out_conv, &key, lora, strength, device)? {
            count += 1;
        }

        Ok(count)
    }

    /// Shared timed forward pass logic.
    fn forward_with_emb_timed(
        &self,
        x: Tensor<Backend, 4>,
        emb: Tensor<Backend, 2>,
        context: &Tensor<Backend, 3>,
    ) -> (Tensor<Backend, 4>, UnetTimingStats) {
        let mut stats = UnetTimingStats::default();

        // Input conv
        let t0 = Instant::now();
        let mut x = self.conv_in.forward(x);
        sync_gpu(&x);
        stats.conv_in = t0.elapsed();

        // Down path with skip connections
        let mut skips = vec![x.clone()];
        for block in &self.input_blocks {
            for (i, resnet) in block.resnets.iter().enumerate() {
                let t0 = Instant::now();
                x = resnet.forward(x, Some(&emb));
                sync_gpu(&x);
                stats.down_resnet += t0.elapsed();

                if i < block.attns.len() {
                    let t0 = Instant::now();
                    x = block.attns[i].forward(x, Some(context));
                    sync_gpu(&x);
                    stats.down_attn += t0.elapsed();
                }
                skips.push(x.clone());
            }
            if let Some(ref ds) = block.downsample {
                let t0 = Instant::now();
                x = ds.forward(x);
                sync_gpu(&x);
                stats.down_sample += t0.elapsed();
                skips.push(x.clone());
            }
        }

        // Middle block
        let t0 = Instant::now();
        x = self.mid_block.forward(x, Some(&emb), Some(context));
        sync_gpu(&x);
        stats.mid_block = t0.elapsed();

        // Up path with skip connections
        for block in &self.output_blocks {
            for (i, resnet) in block.resnets.iter().enumerate() {
                let skip = skips.pop().expect("skip connection mismatch");

                let t0 = Instant::now();
                x = cat_along_dim1(x, skip);
                x = resnet.forward(x, Some(&emb));
                sync_gpu(&x);
                stats.up_resnet += t0.elapsed();

                if i < block.attns.len() {
                    let t0 = Instant::now();
                    x = block.attns[i].forward(x, Some(context));
                    sync_gpu(&x);
                    stats.up_attn += t0.elapsed();
                }
            }
            if let Some(ref up) = block.upsample {
                let t0 = Instant::now();
                x = up.forward(x);
                sync_gpu(&x);
                stats.up_sample += t0.elapsed();
            }
        }

        // Output
        let t0 = Instant::now();
        let x = self.out_norm.forward(x);
        let x = silu(x);
        let x = self.out_conv.forward(x);
        sync_gpu(&x);
        stats.conv_out = t0.elapsed();

        (x, stats)
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `x` - Noisy latent `[B, 4, H, W]`
    /// * `timestep` - Diffusion timestep tensor `[batch]` (pre-computed on GPU)
    /// * `context` - CLIP conditioning `[B, 77, context_dim]`
    ///
    /// # Returns
    /// Predicted noise `[B, 4, H, W]`
    pub fn forward(
        &self,
        x: Tensor<Backend, 4>,
        timestep: Tensor<Backend, 1>,
        context: Tensor<Backend, 3>,
    ) -> Tensor<Backend, 4> {
        let emb = self.compute_emb(timestep, None);
        self.forward_with_emb(x, emb, &context, None)
    }

    /// Forward pass with timing instrumentation.
    ///
    /// Returns (output, timing_stats) where timing_stats breaks down time spent
    /// in different components.
    pub fn forward_timed(
        &self,
        x: Tensor<Backend, 4>,
        timestep: Tensor<Backend, 1>,
        context: Tensor<Backend, 3>,
    ) -> (Tensor<Backend, 4>, UnetTimingStats) {
        let t0 = Instant::now();
        let emb = self.compute_emb(timestep, None);
        sync_gpu(&emb);
        let time_embed_dur = t0.elapsed();

        let (result, mut stats) = self.forward_with_emb_timed(x, emb, &context);
        stats.time_embed = time_embed_dur;
        (result, stats)
    }
}

// ---------------------------------------------------------------------------
// DenoisingUnet impl for SD 1.5
// ---------------------------------------------------------------------------

impl DenoisingUnet for Unet<super::config::Sd15Unet> {
    type Cond = Sd15Conditioning;

    fn forward(
        &self,
        x: Tensor<Backend, 4>,
        timestep: Tensor<Backend, 1>,
        cond: &Sd15Conditioning,
    ) -> Tensor<Backend, 4> {
        let emb = self.compute_emb(timestep, cond.y());
        self.forward_with_emb(x, emb, cond.context(), None)
    }

    fn forward_timed(
        &self,
        x: Tensor<Backend, 4>,
        timestep: Tensor<Backend, 1>,
        cond: &Sd15Conditioning,
    ) -> (Tensor<Backend, 4>, UnetTimingStats) {
        let t0 = Instant::now();
        let emb = self.compute_emb(timestep, cond.y());
        sync_gpu(&emb);
        let time_embed_dur = t0.elapsed();

        let (result, mut stats) = self.forward_with_emb_timed(x, emb, cond.context());
        stats.time_embed = time_embed_dur;
        (result, stats)
    }
}

// ---------------------------------------------------------------------------
// DenoisingUnet impl for SDXL
// ---------------------------------------------------------------------------

impl DenoisingUnet for Unet<super::config::SdxlUnet> {
    type Cond = SdxlConditioning;

    fn forward(
        &self,
        x: Tensor<Backend, 4>,
        timestep: Tensor<Backend, 1>,
        cond: &SdxlConditioning,
    ) -> Tensor<Backend, 4> {
        let emb = self.compute_emb(timestep, cond.y());
        self.forward_with_emb(x, emb, cond.context(), None)
    }

    fn forward_timed(
        &self,
        x: Tensor<Backend, 4>,
        timestep: Tensor<Backend, 1>,
        cond: &SdxlConditioning,
    ) -> (Tensor<Backend, 4>, UnetTimingStats) {
        let t0 = Instant::now();
        let emb = self.compute_emb(timestep, cond.y());
        sync_gpu(&emb);
        let time_embed_dur = t0.elapsed();

        let (result, mut stats) = self.forward_with_emb_timed(x, emb, cond.context());
        stats.time_embed = time_embed_dur;
        (result, stats)
    }
}

// ---------------------------------------------------------------------------
// Timing stats
// ---------------------------------------------------------------------------

/// Timing breakdown for UNet forward pass.
#[derive(Debug, Default)]
pub struct UnetTimingStats {
    /// Time embedding computation.
    pub time_embed: Duration,
    /// Input convolution.
    pub conv_in: Duration,
    /// Down path ResNet blocks.
    pub down_resnet: Duration,
    /// Down path attention layers.
    pub down_attn: Duration,
    /// Downsampling operations.
    pub down_sample: Duration,
    /// Middle block (ResNet + attention).
    pub mid_block: Duration,
    /// Up path ResNet blocks.
    pub up_resnet: Duration,
    /// Up path attention layers.
    pub up_attn: Duration,
    /// Upsampling operations.
    pub up_sample: Duration,
    /// Output norm + conv.
    pub conv_out: Duration,
}

impl UnetTimingStats {
    /// Total time across all components.
    pub fn total(&self) -> Duration {
        self.time_embed
            + self.conv_in
            + self.down_resnet
            + self.down_attn
            + self.down_sample
            + self.mid_block
            + self.up_resnet
            + self.up_attn
            + self.up_sample
            + self.conv_out
    }

    /// Print breakdown with percentages.
    pub fn print(&self) {
        let total = self.total().as_secs_f64();
        let pct = |d: Duration| 100.0 * d.as_secs_f64() / total;

        println!(
            "    Time embed:   {:>8.2?} ({:>5.1}%)",
            self.time_embed,
            pct(self.time_embed)
        );
        println!(
            "    Conv in:      {:>8.2?} ({:>5.1}%)",
            self.conv_in,
            pct(self.conv_in)
        );
        println!(
            "    Down ResNet:  {:>8.2?} ({:>5.1}%)",
            self.down_resnet,
            pct(self.down_resnet)
        );
        println!(
            "    Down Attn:    {:>8.2?} ({:>5.1}%)",
            self.down_attn,
            pct(self.down_attn)
        );
        println!(
            "    Down Sample:  {:>8.2?} ({:>5.1}%)",
            self.down_sample,
            pct(self.down_sample)
        );
        println!(
            "    Mid Block:    {:>8.2?} ({:>5.1}%)",
            self.mid_block,
            pct(self.mid_block)
        );
        println!(
            "    Up ResNet:    {:>8.2?} ({:>5.1}%)",
            self.up_resnet,
            pct(self.up_resnet)
        );
        println!(
            "    Up Attn:      {:>8.2?} ({:>5.1}%)",
            self.up_attn,
            pct(self.up_attn)
        );
        println!(
            "    Up Sample:    {:>8.2?} ({:>5.1}%)",
            self.up_sample,
            pct(self.up_sample)
        );
        println!(
            "    Conv out:     {:>8.2?} ({:>5.1}%)",
            self.conv_out,
            pct(self.conv_out)
        );
        println!("    TOTAL:        {:>8.2?}", self.total());
    }

    /// Accumulate stats from another run.
    pub fn accumulate(&mut self, other: &UnetTimingStats) {
        self.time_embed += other.time_embed;
        self.conv_in += other.conv_in;
        self.down_resnet += other.down_resnet;
        self.down_attn += other.down_attn;
        self.down_sample += other.down_sample;
        self.mid_block += other.mid_block;
        self.up_resnet += other.up_resnet;
        self.up_attn += other.up_attn;
        self.up_sample += other.up_sample;
        self.conv_out += other.conv_out;
    }
}

// ---------------------------------------------------------------------------
// Loading helpers
// ---------------------------------------------------------------------------

/// Load Conv2d with dynamic dimensions.
fn load_conv2d(
    tensors: &safetensors::SafeTensors<'_>,
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

/// Load GroupNorm with dynamic dimensions.
fn load_group_norm(
    tensors: &safetensors::SafeTensors<'_>,
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

#[cfg(test)]
mod tests {
    use super::super::Sd15Unet2D;
    use crate::{model_loader::SafeTensorsFile, types::Backend};
    use burn::prelude::*;
    use std::path::Path;

    const MODEL_PATH: &str = "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors";

    #[test]
    fn load_unet_from_checkpoint() {
        let path = Path::new(MODEL_PATH);
        if !path.exists() {
            println!("Model not found at {MODEL_PATH}, skipping");
            return;
        }

        let file = SafeTensorsFile::open(path).expect("Failed to open checkpoint");
        let tensors = file.tensors().expect("Failed to parse safetensors");
        let device = Default::default();

        println!("Loading UNet...");
        let unet = Sd15Unet2D::load(&tensors, &device).expect("Failed to load UNet");

        // Create dummy inputs
        let latent: Tensor<Backend, 4> = Tensor::zeros([1, 4, 64, 64], &device);
        let timestep: Tensor<Backend, 1> = Tensor::full([1], 500.0f32, &device);
        let context: Tensor<Backend, 3> = Tensor::zeros([1, 77, 768], &device);

        println!("Running forward pass...");
        let output = unet.forward(latent, timestep, context);
        let shape = output.shape().dims();

        println!("UNet output shape: {:?}", shape);
        assert_eq!(
            shape,
            [1, 4, 64, 64],
            "Expected output shape [1, 4, 64, 64]"
        );
    }

    #[test]
    fn load_sdxl_unet_from_checkpoint() {
        use super::super::SdxlUnet2D;

        let path = Path::new("models/checkpoints/sd_xl_base_1.0.safetensors");
        if !path.exists() {
            println!("SDXL model not found, skipping");
            return;
        }

        let file = SafeTensorsFile::open(path).expect("Failed to open checkpoint");
        let tensors = file.tensors().expect("Failed to parse safetensors");
        let device = Default::default();

        println!("Loading SDXL UNet...");
        let unet = SdxlUnet2D::load(&tensors, &device).expect("Failed to load SDXL UNet");

        // Create dummy inputs (SDXL uses 128x128 latents for 1024x1024 output)
        let latent: Tensor<Backend, 4> = Tensor::zeros([1, 4, 128, 128], &device);
        let timestep: Tensor<Backend, 1> = Tensor::full([1], 500.0f32, &device);
        let context: Tensor<Backend, 3> = Tensor::zeros([1, 77, 2048], &device);

        println!("Running SDXL forward pass...");
        // Use the raw forward (without label_emb) for a quick smoke test
        let output = unet.forward(latent, timestep, context);
        let shape = output.shape().dims();

        println!("SDXL UNet output shape: {:?}", shape);
        assert_eq!(
            shape,
            [1, 4, 128, 128],
            "Expected output shape [1, 4, 128, 128]"
        );
    }
}
