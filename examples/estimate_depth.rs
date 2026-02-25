//! Monocular depth estimation using Depth Anything V2 Small.
//!
//! Generates a depth map from an input image using the DINOv2 ViT-S/14 encoder
//! and DPT decoder.
//!
//! Usage:
//!   cargo run --release --example estimate_depth -- input.jpg -o depth.png
//!
//! Requires:
//!   - models/depth/depth_anything_v2_vits.safetensors

use burn::{
    prelude::*,
    tensor::{Device, TensorData},
};
use clap::Parser;
use hearth::{
    depth::{DEFAULT_RESOLUTION, DepthAnythingV2, IMAGENET_MEAN, IMAGENET_STD},
    model_loader::SafeTensorsFile,
    startup,
    types::Backend,
};
use image::imageops::FilterType;
use std::{
    path::{Path, PathBuf},
    process,
    time::Instant,
};
use tracing_subscriber::EnvFilter;

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error(transparent)]
    Load(#[from] hearth::model_loader::LoadError),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Data(#[from] burn::tensor::DataError),
    #[error("failed to create image buffer")]
    ImageBuffer,
}

/// Default path to the Depth Anything V2 Small model.
const MODEL_PATH: &str = "models/depth/depth_anything_v2_vits_fp16.safetensors";

/// Estimate depth from an input image using Depth Anything V2 Small.
#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Input image path (any format supported by the image crate).
    input: PathBuf,

    /// Output depth map path (16-bit grayscale PNG).
    #[arg(short, long, default_value = "depth.png")]
    output: PathBuf,

    /// Inference resolution (must be divisible by 14).
    #[arg(long, default_value_t = DEFAULT_RESOLUTION)]
    resolution: usize,

    /// Path to the Depth Anything V2 model file.
    #[arg(long, default_value = MODEL_PATH)]
    model: PathBuf,
}

fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    if args.resolution % 14 != 0 {
        eprintln!(
            "Error: resolution must be divisible by 14 (got {})",
            args.resolution
        );
        process::exit(1);
    }

    println!("Hearth Depth Estimator");
    println!("Input: {}", args.input.display());
    println!("Output: {}", args.output.display());
    println!("Resolution: {}x{}", args.resolution, args.resolution);
    println!();

    let file_check = startup::check_model_files(&[&args.model, &args.input]);
    if !file_check.all_found() {
        eprintln!("{}", file_check.format_error());
        process::exit(1);
    }

    let device: Device<Backend> = Default::default();

    // Load input image
    println!("Loading input image...");
    let start = Instant::now();
    let (input_tensor, orig_w, orig_h) =
        load_and_preprocess(&args.input, args.resolution, &device)?;
    println!("  Image loaded and preprocessed in {:?}", start.elapsed());

    // Load model
    println!("Loading Depth Anything V2 Small...");
    let start = Instant::now();
    let file = SafeTensorsFile::open(&args.model)?;
    let tensors = file.tensors()?;
    let model = DepthAnythingV2::load(&tensors, &device)?;
    println!("  Model loaded in {:?}", start.elapsed());

    // Run inference
    println!("Estimating depth...");
    let start = Instant::now();
    let depth = model.forward(input_tensor);
    println!("  Inference completed in {:?}", start.elapsed());

    // Save output
    println!("Saving depth map...");
    save_depth_map(depth, orig_w, orig_h, &args.output)?;
    println!("Done! Saved to {}", args.output.display());

    Ok(())
}

/// Load an image, resize to the target resolution, and normalize.
///
/// Returns `(tensor [1, 3, res, res], original_width, original_height)`.
fn load_and_preprocess(
    path: &Path,
    resolution: usize,
    device: &Device<Backend>,
) -> Result<(Tensor<Backend, 4>, u32, u32), Error> {
    let img = image::open(path)?;
    let (orig_w, orig_h) = (img.width(), img.height());

    // Resize to target resolution
    let img = img.resize_exact(resolution as u32, resolution as u32, FilterType::Lanczos3);
    let rgb = img.to_rgb8();

    // Build tensor in channel-first order [1, 3, H, W]
    let res = resolution;
    let hw = res * res;
    let mut chw = vec![0.0f32; 3 * hw];
    for (i, p) in rgb.pixels().enumerate() {
        chw[i] = p[0] as f32 / 255.0;
        chw[hw + i] = p[1] as f32 / 255.0;
        chw[2 * hw + i] = p[2] as f32 / 255.0;
    }
    // Convert f32 data to the backend's native float type (f16) before sending to GPU
    let td = TensorData::new(chw, [1, 3, res, res])
        .convert::<<Backend as burn::tensor::backend::Backend>::FloatElem>();
    let tensor: Tensor<Backend, 4> = Tensor::from_data(td, device);

    // ImageNet normalization: (x - mean) / std
    let mean_td = TensorData::new(IMAGENET_MEAN.to_vec(), [1, 3, 1, 1])
        .convert::<<Backend as burn::tensor::backend::Backend>::FloatElem>();
    let mean: Tensor<Backend, 4> = Tensor::from_data(mean_td, device);
    let std_td = TensorData::new(IMAGENET_STD.to_vec(), [1, 3, 1, 1])
        .convert::<<Backend as burn::tensor::backend::Backend>::FloatElem>();
    let std: Tensor<Backend, 4> = Tensor::from_data(std_td, device);
    let tensor = (tensor - mean) / std;

    Ok((tensor, orig_w, orig_h))
}

/// Save a depth tensor as a 16-bit grayscale PNG.
///
/// The depth values are normalized to [0, 65535] using min-max normalization.
fn save_depth_map(
    depth: Tensor<Backend, 4>,
    orig_w: u32,
    orig_h: u32,
    path: &Path,
) -> Result<(), Error> {
    // depth is [1, 1, H_out, W_out], convert to f32
    let [_b, _c, h, w] = depth.shape().dims();
    let data: Vec<f32> = depth.into_data().convert::<f32>().to_vec()?;

    // Min-max normalize to [0, 65535]
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-6);
    println!("  Depth range: [{min:.2}, {max:.2}], output: {w}x{h}");

    let pixels_u16: Vec<u16> = data
        .iter()
        .map(|&v| ((v - min) / range * 65535.0) as u16)
        .collect();

    let depth_img: image::ImageBuffer<image::Luma<u16>, Vec<u16>> =
        image::ImageBuffer::from_raw(w as u32, h as u32, pixels_u16).ok_or(Error::ImageBuffer)?;

    // Resize to original image dimensions
    let resized = image::imageops::resize(&depth_img, orig_w, orig_h, FilterType::Lanczos3);

    resized.save(path)?;

    Ok(())
}
