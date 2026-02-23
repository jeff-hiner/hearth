//! VAE encoder implementation.

use super::{config::VaeConfig, down_block::DownEncoderBlock2D, mid_block::UNetMidBlock2D};
use crate::{
    layers::{load_conv2d, load_group_norm},
    model_loader::LoadError,
    types::Backend,
};
use burn::{
    nn::{GroupNorm, conv::Conv2d},
    prelude::*,
    tensor::activation::silu,
};
use std::marker::PhantomData;
use tracing::info;

/// VAE encoder that converts images to latents.
///
/// Architecturally the mirror of [`VaeDecoder`](super::decoder::VaeDecoder):
/// downsampling instead of upsampling, channel widths in forward order.
#[derive(Debug)]
pub struct VaeEncoder<
    C: VaeConfig<GROUPS, LATENT, OUT, CH0, CH1, CH2, CH3>,
    const GROUPS: usize,
    const LATENT: usize,
    const OUT: usize,
    const CH0: usize,
    const CH1: usize,
    const CH2: usize,
    const CH3: usize,
> {
    /// Initial convolution: image channels → first block channels.
    conv_in: Conv2d<Backend>,
    /// Downsampling blocks (4 blocks, increasing channel width).
    down_blocks: [DownEncoderBlock2D<GROUPS>; 4],
    /// Mid block: ResNet → Attention → ResNet at final channel width.
    mid_block: UNetMidBlock2D<GROUPS, CH3>,
    /// Output group normalization.
    conv_norm_out: GroupNorm<Backend>,
    /// Final convolution: last block channels → latent channels * 2 (mean + logvar).
    conv_out: Conv2d<Backend>,
    /// Quantization convolution (1x1): 2*latent → 2*latent.
    quant_conv: Conv2d<Backend>,
    _config: PhantomData<C>,
}

// Double-latent channel count: encoder outputs mean + logvar, so 2 * LATENT.
// For SD 1.5/SDXL with LATENT=4, this is 8.
const DOUBLE_LATENT: usize = 8;

impl<
    C: VaeConfig<GROUPS, LATENT, OUT, CH0, CH1, CH2, CH3>,
    const GROUPS: usize,
    const LATENT: usize,
    const OUT: usize,
    const CH0: usize,
    const CH1: usize,
    const CH2: usize,
    const CH3: usize,
> VaeEncoder<C, GROUPS, LATENT, OUT, CH0, CH1, CH2, CH3>
{
    /// Load a VAE encoder from safetensors weights.
    ///
    /// The prefix should be the VAE base path (e.g., `Some("first_stage_model")`
    /// for a full SD checkpoint, or `None` for a standalone VAE file).
    /// This function loads both `quant_conv` and the encoder sub-components.
    pub fn load(
        tensors: &safetensors::SafeTensors<'_>,
        prefix: Option<&str>,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        // Build "prefix." or "" for key construction
        let p = match prefix {
            Some(s) => format!("{s}."),
            None => String::new(),
        };

        // quant_conv: applied to encoder output before sampling
        let quant_conv = load_conv2d::<1, DOUBLE_LATENT, DOUBLE_LATENT>(
            tensors,
            &format!("{p}quant_conv"),
            device,
        )?;

        // conv_in: image channels (3) -> first block channels
        let conv_in = load_conv2d::<3, OUT, CH0>(tensors, &format!("{p}encoder.conv_in"), device)?;

        // Down blocks (forward order: 128→128, 128→256, 256→512, 512→512)
        // Block 0: CH0→CH0, + downsample ÷2
        let down_block_0 = DownEncoderBlock2D::<GROUPS>::load::<CH0, CH0>(
            tensors,
            &format!("{p}encoder.down.0"),
            true,
            device,
        )?;

        // Block 1: CH0→CH1, + downsample ÷2
        let down_block_1 = DownEncoderBlock2D::<GROUPS>::load::<CH0, CH1>(
            tensors,
            &format!("{p}encoder.down.1"),
            true,
            device,
        )?;

        // Block 2: CH1→CH2, + downsample ÷2
        let down_block_2 = DownEncoderBlock2D::<GROUPS>::load::<CH1, CH2>(
            tensors,
            &format!("{p}encoder.down.2"),
            true,
            device,
        )?;

        // Block 3: CH2→CH3, no downsample (last block)
        let down_block_3 = DownEncoderBlock2D::<GROUPS>::load::<CH2, CH3>(
            tensors,
            &format!("{p}encoder.down.3"),
            false,
            device,
        )?;

        let down_blocks = [down_block_0, down_block_1, down_block_2, down_block_3];

        // Mid block
        let mid_block = UNetMidBlock2D::load(tensors, &format!("{p}encoder.mid"), device)?;

        // Output normalization and conv
        let conv_norm_out =
            load_group_norm::<GROUPS, CH3>(tensors, &format!("{p}encoder.norm_out"), device)?;

        let conv_out =
            load_conv2d::<3, CH3, DOUBLE_LATENT>(tensors, &format!("{p}encoder.conv_out"), device)?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
            quant_conv,
            _config: PhantomData,
        })
    }

    /// Encode an image to latent space.
    ///
    /// Input: `[batch, 3, height, width]` in `[0, 1]` (pixel space, BCHW)
    /// Output: `[batch, 4, height/8, width/8]` (latent space)
    pub fn forward(&self, image: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        // Map [0, 1] → [-1, 1] (VAE encoder expects this range)
        let image = image * 2.0 - 1.0;
        let hidden = self.encode_single(image);
        self.sample_and_scale(hidden)
    }

    /// Encode an image using overlapping tiles to stay within GPU memory limits.
    ///
    /// Mirrors the tiled decode approach in [`VaeDecoder::forward_tiled`].
    /// Operates in pixel space: splits the image into overlapping tiles,
    /// encodes each independently, then blends in latent space.
    ///
    /// - `tile_size`: tile side length in latent pixels (e.g. 64 → 512px tiles)
    /// - `overlap`: overlap per side in latent pixels (e.g. 8 → 64px overlap)
    pub fn forward_tiled(
        &self,
        image: Tensor<Backend, 4>,
        tile_size: usize,
        overlap: usize,
    ) -> Tensor<Backend, 4> {
        // Map [0, 1] → [-1, 1] once before tiling
        let image = image * 2.0 - 1.0;

        let [batch, channels, img_h, img_w] = image.shape().dims();
        let device = image.device();
        let pixel_scale: usize = 8;

        // Convert tile params from latent to pixel space
        let tile_px = tile_size * pixel_scale;
        let overlap_px = overlap * pixel_scale;

        // If the image fits in a single tile, skip tiling (already scaled)
        if img_h <= tile_px && img_w <= tile_px {
            let hidden = self.encode_single(image);
            return self.sample_and_scale(hidden);
        }

        let stride_px = tile_px - 2 * overlap_px;
        let lat_h = img_h / pixel_scale;
        let lat_w = img_w / pixel_scale;

        // Accumulation buffers in latent space
        let mut output: Tensor<Backend, 4> = Tensor::zeros([batch, LATENT, lat_h, lat_w], &device);
        let mut weights: Tensor<Backend, 4> = Tensor::zeros([1, 1, lat_h, lat_w], &device);

        // Compute tile grid positions (in pixel space)
        let tile_positions: Vec<(usize, usize)> = {
            let mut positions = Vec::new();
            let mut y = 0;
            while y < img_h {
                let mut x = 0;
                while x < img_w {
                    positions.push((y, x));
                    if x + tile_px >= img_w {
                        break;
                    }
                    x += stride_px;
                }
                if y + tile_px >= img_h {
                    break;
                }
                y += stride_px;
            }
            positions
        };

        let total_tiles = tile_positions.len();
        info!(total_tiles, tile_size, overlap, "Starting tiled VAE encode");

        for (i, &(py, px)) in tile_positions.iter().enumerate() {
            // Clamp tile to image bounds
            let tile_h = tile_px.min(img_h - py);
            let tile_w = tile_px.min(img_w - px);

            // Slice pixel tile
            let tile =
                image
                    .clone()
                    .slice([0..batch, 0..channels, py..(py + tile_h), px..(px + tile_w)]);

            // Encode through the full VAE pipeline
            let encoded = self.encode_single(tile);
            let encoded = self.sample_and_scale(encoded);

            // Latent-space coordinates
            let ly = py / pixel_scale;
            let lx = px / pixel_scale;
            let lh = tile_h / pixel_scale;
            let lw = tile_w / pixel_scale;

            // Build cosine blend window in latent space
            let window = cosine_window_2d(
                lh,
                lw,
                overlap,
                TileEdges {
                    at_top: py == 0,
                    at_bottom: py + tile_h >= img_h,
                    at_left: px == 0,
                    at_right: px + tile_w >= img_w,
                },
                &device,
            );

            // Multiply encoded tile by window
            let weighted = encoded * window.clone();

            // Pad tile into full-size buffer and accumulate
            let padded: Tensor<Backend, 4> = Tensor::zeros([batch, LATENT, lat_h, lat_w], &device);
            let padded = padded.slice_assign(
                [0..batch, 0..LATENT, ly..(ly + lh), lx..(lx + lw)],
                weighted,
            );
            output = output + padded;

            let padded_w: Tensor<Backend, 4> = Tensor::zeros([1, 1, lat_h, lat_w], &device);
            let padded_w =
                padded_w.slice_assign([0..1, 0..1, ly..(ly + lh), lx..(lx + lw)], window);
            weights = weights + padded_w;

            // Force materialization before cleanup
            let _ = <Backend as burn::tensor::backend::Backend>::sync(&device);
            <Backend as burn::tensor::backend::Backend>::memory_cleanup(&device);
            info!(tile = i + 1, total_tiles, "Encoded tile");
        }

        // Normalize by accumulated weights
        output / weights
    }

    /// Run the encoder pipeline on a pixel-space tensor.
    ///
    /// Returns the raw encoder output (mean + logvar, 8 channels)
    /// before sampling and scaling.
    fn encode_single(&self, image: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        // Initial conv
        let hidden = self.conv_in.forward(image);

        // Down blocks
        let mut hidden = hidden;
        for down_block in &self.down_blocks {
            hidden = down_block.forward(hidden);
        }

        // Mid block
        let hidden = self.mid_block.forward(hidden);

        // Output
        let hidden = self.conv_norm_out.forward(hidden);
        let hidden = silu(hidden);
        let hidden = self.conv_out.forward(hidden);

        // Quantization conv
        self.quant_conv.forward(hidden)
    }

    /// Take the mean (first LATENT channels) and multiply by scaling factor.
    fn sample_and_scale(&self, hidden: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        let [batch, _channels, h, w] = hidden.shape().dims();
        // Slice first LATENT channels (mean), discard logvar
        let mean = hidden.slice([0..batch, 0..LATENT, 0..h, 0..w]);
        mean * C::SCALING_FACTOR
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

/// Build a 2D cosine blend window for tile-based encoding.
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

    let row_tensor: Tensor<Backend, 1> = Tensor::from_floats(row_weights.as_slice(), device);
    let col_tensor: Tensor<Backend, 1> = Tensor::from_floats(col_weights.as_slice(), device);

    let window_2d = col_tensor.reshape([height, 1]) * row_tensor.reshape([1, width]);
    window_2d.reshape([1, 1, height, width])
}
