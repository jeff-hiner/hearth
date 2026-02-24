//! VAE decoder implementation.

use super::{config::VaeConfig, mid_block::UNetMidBlock2D, up_block::UpDecoderBlock2D};
use crate::{
    layers::{load_conv2d, load_group_norm},
    model_loader::LoadError,
    types::Backend,
};
use burn::{
    nn::{GroupNorm, conv::Conv2d},
    prelude::{Backend as _, *},
    tensor::activation::silu,
};
use safetensors::SafeTensors;
use std::marker::PhantomData;
#[cfg(test)]
use std::time::Instant;
use tracing::info;

/// VAE decoder that converts latents to images.
///
/// Generic over configuration type which provides architecture constants
/// for compile-time optimization.
#[derive(Debug)]
pub struct VaeDecoder<
    C: VaeConfig<GROUPS, LATENT, OUT, CH0, CH1, CH2, CH3>,
    const GROUPS: usize,
    const LATENT: usize,
    const OUT: usize,
    const CH0: usize,
    const CH1: usize,
    const CH2: usize,
    const CH3: usize,
> {
    post_quant_conv: Conv2d<Backend>,
    conv_in: Conv2d<Backend>,
    mid_block: UNetMidBlock2D<GROUPS, CH3>,
    up_blocks: [UpDecoderBlock2D<GROUPS>; 4],
    conv_norm_out: GroupNorm<Backend>,
    conv_out: Conv2d<Backend>,
    _config: PhantomData<C>,
}

impl<
    C: VaeConfig<GROUPS, LATENT, OUT, CH0, CH1, CH2, CH3>,
    const GROUPS: usize,
    const LATENT: usize,
    const OUT: usize,
    const CH0: usize,
    const CH1: usize,
    const CH2: usize,
    const CH3: usize,
> VaeDecoder<C, GROUPS, LATENT, OUT, CH0, CH1, CH2, CH3>
{
    /// Load a VAE decoder from safetensors weights.
    ///
    /// The prefix should be the VAE base path (e.g., `Some("first_stage_model")`
    /// for a full SD checkpoint, or `None` for a standalone VAE file).
    /// This function loads both `post_quant_conv` and the decoder sub-components.
    pub fn load(
        tensors: &SafeTensors<'_>,
        prefix: Option<&str>,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        // Build "prefix." or "" for key construction
        let p = match prefix {
            Some(s) => format!("{s}."),
            None => String::new(),
        };

        // post_quant_conv: applied to latents before decoder
        let post_quant_conv =
            load_conv2d::<1, LATENT, LATENT>(tensors, &format!("{p}post_quant_conv"), device)?;

        // conv_in: latent_channels -> last_block_out_channels
        let conv_in =
            load_conv2d::<3, LATENT, CH3>(tensors, &format!("{p}decoder.conv_in"), device)?;

        // Mid block
        let mid_block = UNetMidBlock2D::load(tensors, &format!("{p}decoder.mid"), device)?;

        // Up blocks (reverse order: 512->512->256->128)
        // Channels: [CH0, CH1, CH2, CH3] = [128, 256, 512, 512]

        // Block 0: CH3->CH3 with upsample
        let up_block_0 = UpDecoderBlock2D::<GROUPS>::load::<CH3, CH3>(
            tensors,
            &format!("{p}decoder.up.3"),
            true,
            device,
        )?;

        // Block 1: CH3->CH2 with upsample
        let up_block_1 = UpDecoderBlock2D::<GROUPS>::load::<CH3, CH2>(
            tensors,
            &format!("{p}decoder.up.2"),
            true,
            device,
        )?;

        // Block 2: CH2->CH1 with upsample
        let up_block_2 = UpDecoderBlock2D::<GROUPS>::load::<CH2, CH1>(
            tensors,
            &format!("{p}decoder.up.1"),
            true,
            device,
        )?;

        // Block 3: CH1->CH0 no upsample
        let up_block_3 = UpDecoderBlock2D::<GROUPS>::load::<CH1, CH0>(
            tensors,
            &format!("{p}decoder.up.0"),
            false,
            device,
        )?;

        let up_blocks = [up_block_0, up_block_1, up_block_2, up_block_3];

        // Output normalization and conv
        let conv_norm_out =
            load_group_norm::<GROUPS, CH0>(tensors, &format!("{p}decoder.norm_out"), device)?;

        let conv_out =
            load_conv2d::<3, CH0, OUT>(tensors, &format!("{p}decoder.conv_out"), device)?;

        Ok(Self {
            post_quant_conv,
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            _config: PhantomData,
        })
    }

    /// Decode latents to an image.
    ///
    /// Input shape: `[batch, 4, height, width]` (latent space)
    /// Output shape: `[batch, 3, height*8, width*8]` (pixel space)
    pub fn forward(&self, latents: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        // Scale latents
        let hidden = latents / C::SCALING_FACTOR;
        self.decode_single(hidden)
    }

    /// Decode latents using overlapping tiles to stay within GPU memory limits.
    ///
    /// Splits the latent into overlapping tiles, decodes each independently,
    /// then blends in pixel space with a cosine window. Required for SDXL on
    /// GPUs where the full VAE attention matrix exceeds buffer size limits.
    ///
    /// - `tile_size`: tile side length in latent pixels (e.g. 64 → 512px tiles)
    /// - `overlap`: overlap per side in latent pixels (e.g. 8 → 64px overlap)
    pub fn forward_tiled(
        &self,
        latents: Tensor<Backend, 4>,
        tile_size: usize,
        overlap: usize,
    ) -> Tensor<Backend, 4> {
        let [batch, _channels, lat_h, lat_w] = latents.shape().dims();
        let device = latents.device();

        // Scale latents (cheap at 4 channels)
        let latents = latents / C::SCALING_FACTOR;

        // If the latent fits in a single tile, skip tiling entirely
        if lat_h <= tile_size && lat_w <= tile_size {
            return self.decode_single(latents);
        }

        let stride = tile_size - 2 * overlap;
        let pixel_scale = 8;
        let out_h = lat_h * pixel_scale;
        let out_w = lat_w * pixel_scale;

        // Accumulation buffers in pixel space
        let mut output: Tensor<Backend, 4> = Tensor::zeros([batch, OUT, out_h, out_w], &device);
        let mut weights: Tensor<Backend, 4> = Tensor::zeros([1, 1, out_h, out_w], &device);

        // Compute tile grid positions
        let tile_positions: Vec<(usize, usize)> = {
            let mut positions = Vec::new();
            let mut y = 0;
            while y < lat_h {
                let mut x = 0;
                while x < lat_w {
                    positions.push((y, x));
                    if x + tile_size >= lat_w {
                        break;
                    }
                    x += stride;
                }
                if y + tile_size >= lat_h {
                    break;
                }
                y += stride;
            }
            positions
        };

        let total_tiles = tile_positions.len();
        info!(
            total_tiles,
            tile_size, overlap, stride, "Starting tiled VAE decode"
        );

        for (i, &(ty, tx)) in tile_positions.iter().enumerate() {
            // Clamp tile to latent bounds
            let tile_h = tile_size.min(lat_h - ty);
            let tile_w = tile_size.min(lat_w - tx);

            // Slice latent tile
            let tile =
                latents
                    .clone()
                    .slice([0..batch, 0..LATENT, ty..(ty + tile_h), tx..(tx + tile_w)]);

            // Decode through the full VAE pipeline
            let decoded = self.decode_single(tile);

            // Pixel-space coordinates
            let py = ty * pixel_scale;
            let px = tx * pixel_scale;
            let ph = tile_h * pixel_scale;
            let pw = tile_w * pixel_scale;

            // Build cosine blend window for this tile
            let overlap_px = overlap * pixel_scale;
            let window = cosine_window_2d(
                ph,
                pw,
                overlap_px,
                TileEdges {
                    at_top: ty == 0,
                    at_bottom: ty + tile_h >= lat_h,
                    at_left: tx == 0,
                    at_right: tx + tile_w >= lat_w,
                },
                &device,
            );

            // Multiply decoded tile by window
            let weighted = decoded * window.clone();

            // Pad tile into full-size buffer and accumulate via addition.
            // Using slice_assign on a fresh zeros tensor avoids read-modify-write
            // on `output`, which breaks under Burn's lazy fusion backend.
            let padded: Tensor<Backend, 4> = Tensor::zeros([batch, OUT, out_h, out_w], &device);
            let padded =
                padded.slice_assign([0..batch, 0..OUT, py..(py + ph), px..(px + pw)], weighted);
            output = output + padded;

            let padded_w: Tensor<Backend, 4> = Tensor::zeros([1, 1, out_h, out_w], &device);
            let padded_w =
                padded_w.slice_assign([0..1, 0..1, py..(py + ph), px..(px + pw)], window);
            weights = weights + padded_w;

            // Force the fusion backend to materialize output before cleanup,
            // otherwise memory_cleanup can reclaim buffers still in the lazy graph.
            let _ = Backend::sync(&device);
            Backend::memory_cleanup(&device);
            info!(tile = i + 1, total_tiles, "Decoded tile");
        }

        // Normalize by accumulated weights
        output / weights
    }

    /// Run the decoder pipeline on a pre-scaled latent tensor.
    ///
    /// This is the core decode path shared by [`forward`] and [`forward_tiled`].
    /// The input must already be divided by the scaling factor.
    fn decode_single(&self, hidden: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        // Post-quantization conv (1x1, applied before main decoder)
        let hidden = self.post_quant_conv.forward(hidden);

        // Initial conv
        let hidden = self.conv_in.forward(hidden);

        // Mid block
        let hidden = self.mid_block.forward(hidden);

        // Up blocks
        let mut hidden = hidden;
        for up_block in &self.up_blocks {
            hidden = up_block.forward(hidden);
        }

        // Output
        let hidden = self.conv_norm_out.forward(hidden);
        let hidden = silu(hidden);
        self.conv_out.forward(hidden)
    }

    /// Decode latents to an image with detailed timing for each step.
    ///
    /// This variant prints timing information for profiling purposes.
    /// Shows both dispatch time (async op submission) and sync time (GPU completion).
    #[cfg(test)]
    pub(crate) fn forward_timed(&self, latents: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        let total_start = Instant::now();

        let hidden = timed_step("scale_latents", || latents / C::SCALING_FACTOR);
        let hidden = timed_step("post_quant_conv", || self.post_quant_conv.forward(hidden));
        let hidden = timed_step("conv_in", || self.conv_in.forward(hidden));
        let hidden = timed_step("mid_block", || self.mid_block.forward(hidden));

        let mut hidden = hidden;
        for (i, up_block) in self.up_blocks.iter().enumerate() {
            hidden = timed_step(&format!("up_block[{i}]"), || up_block.forward(hidden));
        }

        let hidden = timed_step("conv_norm_out", || self.conv_norm_out.forward(hidden));
        let hidden = timed_step("silu", || silu(hidden));
        let output = timed_step("conv_out", || self.conv_out.forward(hidden));

        println!("  TOTAL forward: {:?}", total_start.elapsed());

        output
    }
}

/// Tile edge flags for [`cosine_window_2d`].
struct TileEdges {
    /// Tile is at the top edge of the image.
    at_top: bool,
    /// Tile is at the bottom edge of the image.
    at_bottom: bool,
    /// Tile is at the left edge of the image.
    at_left: bool,
    /// Tile is at the right edge of the image.
    at_right: bool,
}

/// Build a 2D cosine blend window for tile-based decoding.
///
/// Creates a `[1, 1, height, width]` tensor where interior pixels are 1.0 and
/// edges that overlap with adjacent tiles ramp from 0→1 via a cosine curve.
/// Edges touching the image boundary are kept at 1.0.
fn cosine_window_2d(
    height: usize,
    width: usize,
    overlap: usize,
    edges: TileEdges,
    device: &Device<Backend>,
) -> Tensor<Backend, 4> {
    let ramp_1d = |size: usize, ramp_start: bool, ramp_end: bool| -> Vec<f32> {
        let mut values = vec![1.0_f32; size];
        if ramp_start && overlap > 0 {
            let n = overlap.min(size);
            for (i, val) in values.iter_mut().take(n).enumerate() {
                // Cosine ramp: 0 at i=0, 1 at i=overlap
                let t = (i as f32 + 0.5) / overlap as f32;
                *val = 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
            }
        }
        if ramp_end && overlap > 0 {
            let n = overlap.min(size);
            for i in 0..n {
                let idx = size - 1 - i;
                let t = (i as f32 + 0.5) / overlap as f32;
                values[idx] = 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
            }
        }
        values
    };

    let row_weights = ramp_1d(width, !edges.at_left, !edges.at_right);
    let col_weights = ramp_1d(height, !edges.at_top, !edges.at_bottom);

    // Outer product → [height, width], then reshape to [1, 1, H, W]
    let row_tensor: Tensor<Backend, 1> = Tensor::from_floats(row_weights.as_slice(), device);
    let col_tensor: Tensor<Backend, 1> = Tensor::from_floats(col_weights.as_slice(), device);

    // col [H, 1] * row [1, W] → [H, W]
    let window_2d = col_tensor.reshape([height, 1]) * row_tensor.reshape([1, width]);
    window_2d.reshape([1, 1, height, width])
}

/// Force GPU synchronization by reading tensor data.
///
/// This ensures all pending GPU operations complete before returning.
/// Reads a single element to minimize data transfer overhead.
#[cfg(test)]
fn sync_tensor<const D: usize>(tensor: &Tensor<Backend, D>) {
    let flat = tensor.clone().flatten::<1>(0, D - 1);
    let _ = flat.slice(0..1).into_data();
}

/// Run `op`, sync the result to GPU, print dispatch/sync timing, and return the result.
#[cfg(test)]
fn timed_step(name: &str, op: impl FnOnce() -> Tensor<Backend, 4>) -> Tensor<Backend, 4> {
    let start = Instant::now();
    let result = op();
    let dispatch = start.elapsed();
    let sync_start = Instant::now();
    sync_tensor(&result);
    println!(
        "  {name}: dispatch={dispatch:?} sync={:?}",
        sync_start.elapsed()
    );
    result
}
