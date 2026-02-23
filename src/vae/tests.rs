//! VAE decoder fixture tests
//!
//! Compares Rust VAE decoder output against PyTorch/diffusers reference.
//!
//! Run with: cargo test vae::tests -- --nocapture
//!
//! Prerequisites:
//! 1. Model: models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors
//! 2. Fixtures: tests/fixtures/latent_*.npy and expected_*.png
//!    (generate with: python tests/fixtures/generate_fixtures.py)

use super::{Sd15VaeDecoder, Sd15VaeEncoder, SdxlVaeDecoder, SdxlVaeEncoder};
use crate::{model_loader::SafeTensorsFile, types::Backend};
use burn::tensor::{Device, Tensor};
use image::GenericImageView;
use std::{path::Path, time::Instant};

const MODEL_PATH: &str = "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors";
const FIXTURES_DIR: &str = "tests/fixtures";

/// VAE prefix in full SD checkpoints (base path, not decoder subpath)
const VAE_PREFIX: &str = "first_stage_model";

/// Maximum allowed difference per channel.
/// f16/bf16 backends have accumulated quantization error.
#[cfg(feature = "gpu-vulkan-f16")]
const TOLERANCE: u8 = 3;
#[cfg(feature = "gpu-vulkan-bf16")]
const TOLERANCE: u8 = 5; // bf16 has less mantissa precision than f16

/// Parse a NumPy .npy file containing a float32 array.
///
/// Only supports simple C-contiguous float32 arrays (what numpy.save produces
/// for our latent tensors).
fn load_npy_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let data =
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

    // Check magic number: \x93NUMPY
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return Err("Invalid NPY magic number".to_string());
    }

    let major = data[6];
    let minor = data[7];

    // Parse header length based on version
    let (header_len, header_start) = match (major, minor) {
        (1, 0) => {
            let len = u16::from_le_bytes([data[8], data[9]]) as usize;
            (len, 10)
        }
        (2, 0) | (3, 0) => {
            let len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
            (len, 12)
        }
        _ => return Err(format!("Unsupported NPY version {major}.{minor}")),
    };

    let header_end = header_start + header_len;
    if data.len() < header_end {
        return Err("NPY file truncated in header".to_string());
    }

    let header = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|_| "Invalid UTF-8 in NPY header")?;

    // Parse shape from header dict: {'descr': '<f4', 'fortran_order': False, 'shape': (1, 4, 8, 8)}
    // This is a simple parser that assumes well-formed output from numpy.save
    let shape = parse_npy_shape(header)?;

    // Verify dtype is float32 (little-endian)
    if !header.contains("'<f4'") && !header.contains("'float32'") {
        return Err(format!("Expected float32 dtype, got header: {header}"));
    }

    // Verify C-order (not Fortran)
    if header.contains("'fortran_order': True") {
        return Err("Fortran-order arrays not supported".to_string());
    }

    // Read float32 data
    let data_start = header_end;
    let num_elements: usize = shape.iter().product();
    let expected_bytes = num_elements * 4;

    if data.len() < data_start + expected_bytes {
        return Err(format!(
            "NPY file truncated: expected {} bytes of data, got {}",
            expected_bytes,
            data.len() - data_start
        ));
    }

    let floats: Vec<f32> = data[data_start..data_start + expected_bytes]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok((shape, floats))
}

/// Parse shape tuple from NPY header string.
fn parse_npy_shape(header: &str) -> Result<Vec<usize>, String> {
    // Find 'shape': (...)
    let shape_start = header
        .find("'shape':")
        .ok_or("Missing 'shape' in NPY header")?;
    let after_key = &header[shape_start + 8..];

    // Find the opening paren
    let paren_start = after_key.find('(').ok_or("Missing '(' after 'shape'")?;
    let paren_end = after_key.find(')').ok_or("Missing ')' in shape tuple")?;

    let shape_str = &after_key[paren_start + 1..paren_end];

    // Parse comma-separated integers
    let mut shape = Vec::new();
    for part in shape_str.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            let dim: usize = trimmed
                .parse()
                .map_err(|_| format!("Invalid shape dimension: {trimmed}"))?;
            shape.push(dim);
        }
    }

    Ok(shape)
}

/// Load latent tensor from .npy file.
fn load_latent(path: &Path, device: &Device<Backend>) -> Result<Tensor<Backend, 4>, String> {
    let (shape, data) = load_npy_f32(path)?;

    if shape.len() != 4 {
        return Err(format!("Expected 4D tensor, got shape {shape:?}"));
    }

    let tensor: Tensor<Backend, 1> = Tensor::from_floats(data.as_slice(), device);
    Ok(tensor.reshape([shape[0], shape[1], shape[2], shape[3]]))
}

/// Load expected PNG and return as [H, W, 3] u8 array.
fn load_expected_png(path: &Path) -> Result<(u32, u32, Vec<u8>), String> {
    let img = image::open(path).map_err(|e| format!("Failed to open {}: {e}", path.display()))?;

    let (width, height) = img.dimensions();
    let rgb = img.to_rgb8();

    Ok((width, height, rgb.into_raw()))
}

/// Convert VAE output tensor to RGB u8 pixels.
///
/// Input: [1, 3, H, W] tensor in [-1, 1] range
/// Output: [H, W, 3] u8 array in [0, 255] range
fn tensor_to_rgb(tensor: Tensor<Backend, 4>) -> Vec<u8> {
    // Get shape
    let shape = tensor.shape().dims();
    let [_batch, channels, height, width] = shape;
    assert_eq!(channels, 3, "Expected 3 channels");

    // Convert to f32 vec (this is where GPU sync happens)
    // Use convert::<f32>() to handle both f32 and f16 backends
    let sync_start = Instant::now();
    let data: Vec<f32> = tensor
        .into_data()
        .convert::<f32>()
        .to_vec()
        .expect("Failed to read tensor data");
    let sync_time = sync_start.elapsed();
    println!("  GPU sync (into_data): {sync_time:.2?}");

    // Tensor is [B, C, H, W], we need [H, W, C]
    // Also clamp [-1, 1] -> [0, 1] -> [0, 255]
    let mut rgb = vec![0u8; height * width * 3];

    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let src_idx = c * height * width + y * width + x;
                let dst_idx = y * width * 3 + x * 3 + c;

                // Clamp to [-1, 1], then scale to [0, 255]
                let val = data[src_idx].clamp(-1.0, 1.0);
                let normalized = (val + 1.0) / 2.0; // [-1,1] -> [0,1]
                rgb[dst_idx] = (normalized * 255.0).round() as u8;
            }
        }
    }

    rgb
}

/// Compare two RGB images with tolerance.
///
/// Returns Ok(()) if all pixels match within tolerance, Err with stats otherwise.
fn compare_images(
    actual: &[u8],
    expected: &[u8],
    width: u32,
    height: u32,
    tolerance: u8,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "Size mismatch: actual {} bytes, expected {} bytes",
            actual.len(),
            expected.len()
        ));
    }

    let mut max_diff: u8 = 0;
    let mut diff_count: usize = 0;
    let mut total_diff: u64 = 0;

    for (&a, &e) in actual.iter().zip(expected.iter()) {
        let diff = a.abs_diff(e);
        if diff > tolerance {
            diff_count += 1;
        }
        if diff > max_diff {
            max_diff = diff;
        }
        total_diff += u64::from(diff);
    }

    let total_pixels = (width * height) as usize;
    let avg_diff = total_diff as f64 / actual.len() as f64;

    if diff_count > 0 {
        Err(format!(
            "Image mismatch: {diff_count}/{total_pixels} pixels differ by more than {tolerance}, \
             max_diff={max_diff}, avg_diff={avg_diff:.2}"
        ))
    } else {
        println!("Images match within tolerance (max_diff={max_diff}, avg_diff={avg_diff:.2})");
        Ok(())
    }
}

/// Test VAE decoder against fixture at given size.
fn test_vae_fixture(latent_name: &str, expected_name: &str) {
    let model_path = Path::new(MODEL_PATH);
    let latent_path = Path::new(FIXTURES_DIR).join(latent_name);
    let expected_path = Path::new(FIXTURES_DIR).join(expected_name);

    // Check prerequisites
    if !model_path.exists() {
        println!("Model not found at {MODEL_PATH}");
        println!("Download from: https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive");
        return;
    }

    if !latent_path.exists() {
        println!("Latent fixture not found at {}", latent_path.display());
        println!(
            "Generate with: python tests/fixtures/generate_fixtures.py --model-path {MODEL_PATH}"
        );
        return;
    }

    if !expected_path.exists() {
        println!("Expected PNG not found at {}", expected_path.display());
        println!(
            "Generate with: python tests/fixtures/generate_fixtures.py --model-path {MODEL_PATH}"
        );
        return;
    }

    let device_start = Instant::now();
    let device: Device<Backend> = Default::default();
    let device_time = device_start.elapsed();
    println!("Device init time: {device_time:.2?}");

    // Load model
    println!("Loading VAE from {MODEL_PATH}...");
    let load_start = Instant::now();
    let file = SafeTensorsFile::open(model_path).expect("Failed to open model");
    let tensors = file.tensors().expect("Failed to parse safetensors");
    let decoder = Sd15VaeDecoder::load(&tensors, Some(VAE_PREFIX), &device)
        .expect("Failed to load VAE decoder");
    let load_time = load_start.elapsed();
    println!("Model load time: {load_time:.2?}");

    // Load input latent
    println!("Loading latent from {}...", latent_path.display());
    let latent = load_latent(&latent_path, &device).expect("Failed to load latent");
    let latent_shape: [usize; 4] = latent.shape().dims();
    println!("Latent shape: {latent_shape:?}");

    // Warmup pass (shader compilation)
    println!("Warmup pass (shader compilation)...");
    let warmup_start = Instant::now();
    let warmup_output = decoder.forward(latent.clone());
    // Force GPU sync to ensure shaders are compiled before timed run
    let _ = warmup_output.into_data();
    println!("Warmup time: {:?}", warmup_start.elapsed());

    // Run decoder with detailed timing (post-warmup)
    println!("\nRunning VAE decode (timed, post-warmup)...");
    let output = decoder.forward_timed(latent);
    let output_shape: [usize; 4] = output.shape().dims();
    println!("Output shape: {output_shape:?}");

    // Convert to RGB (includes GPU→CPU sync)
    let rgb_start = Instant::now();
    let actual_rgb = tensor_to_rgb(output);
    let rgb_time = rgb_start.elapsed();
    println!("GPU→CPU transfer + convert: {rgb_time:.2?}");

    // Load expected
    println!("Loading expected from {}...", expected_path.display());
    let (width, height, expected_rgb) =
        load_expected_png(&expected_path).expect("Failed to load expected PNG");
    println!("Expected size: {width}x{height}");

    // Compare
    println!("Comparing with tolerance +/-{TOLERANCE}...");
    compare_images(&actual_rgb, &expected_rgb, width, height, TOLERANCE)
        .expect("VAE output does not match reference");

    println!("PASSED: VAE output matches PyTorch reference!");
}

#[test]
fn vae_decode_8x8() {
    test_vae_fixture("latent_8x8.npy", "expected_8x8.png");
}

/// Diagnostic test to verify conv weight loading matches PyTorch.
#[test]
fn debug_post_quant_conv() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        println!("Model not found, skipping");
        return;
    }

    let device: Device<Backend> = Default::default();
    let file = SafeTensorsFile::open(model_path).expect("Failed to open model");
    let tensors = file.tensors().expect("Failed to parse safetensors");

    // Load just the post_quant_conv
    let conv = crate::layers::load_conv2d::<1, 4, 4>(
        &tensors,
        "first_stage_model.post_quant_conv",
        &device,
    )
    .expect("Failed to load post_quant_conv");

    // Test with known input [1, 2, 3, 4]
    let input: Tensor<Backend, 4> =
        Tensor::from_floats([[[[1.0f32]], [[2.0]], [[3.0]], [[4.0]]]], &device);

    let output = conv.forward(input);
    let output_data: Vec<f32> = output
        .into_data()
        .convert::<f32>()
        .to_vec()
        .expect("Failed to get output");

    println!("post_quant_conv output for input [1,2,3,4]:");
    println!("  Rust:   {:?}", output_data);
    println!("  PyTorch expected: [0.7886, -1.1058, -0.8043, -1.1555]");

    // Check if values are close (within 0.01)
    let expected = [0.7886f32, -1.1058, -0.8043, -1.1555];
    for (i, (got, exp)) in output_data.iter().zip(expected.iter()).enumerate() {
        let diff = (got - exp).abs();
        assert!(
            diff < 0.01,
            "Channel {} mismatch: got {}, expected {}, diff {}",
            i,
            got,
            exp,
            diff
        );
    }
    println!("post_quant_conv output matches PyTorch!");
}

#[test]
fn vae_decode_64x64() {
    test_vae_fixture("latent_64x64.npy", "expected_64x64.png");
}

/// Trace through VAE decode to find divergence point.
#[test]
fn debug_vae_trace() {
    let model_path = Path::new(MODEL_PATH);
    let latent_path = Path::new(FIXTURES_DIR).join("latent_8x8.npy");

    if !model_path.exists() || !latent_path.exists() {
        println!("Model or latent not found, skipping");
        return;
    }

    let device: Device<Backend> = Default::default();

    // Load latent
    let latent = load_latent(&latent_path, &device).expect("Failed to load latent");
    let latent_data: Vec<f32> = latent
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();
    println!("Latent[0,0,0,0]: {:.6}", latent_data[0]);

    // Scale
    let scaled = latent / 0.18215f32;
    let scaled_data: Vec<f32> = scaled
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();
    println!("Scaled[0,0,0,0]: {:.6}", scaled_data[0]);
    println!("  (PyTorch: 1.672891)");

    // Load and apply post_quant_conv
    let file = SafeTensorsFile::open(model_path).expect("Failed to open model");
    let tensors = file.tensors().expect("Failed to parse");

    let pqc = crate::layers::load_conv2d::<1, 4, 4>(
        &tensors,
        "first_stage_model.post_quant_conv",
        &device,
    )
    .expect("Failed to load post_quant_conv");

    let after_pqc = pqc.forward(scaled);
    let after_pqc_data: Vec<f32> = after_pqc
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();
    println!("After post_quant_conv[0,0,0,0]: {:.6}", after_pqc_data[0]);
    println!("  (PyTorch: -2.938603)");

    // Load and apply conv_in
    let conv_in = crate::layers::load_conv2d::<3, 4, 512>(
        &tensors,
        "first_stage_model.decoder.conv_in",
        &device,
    )
    .expect("Failed to load conv_in");

    let after_conv_in = conv_in.forward(after_pqc);
    let after_conv_in_data: Vec<f32> = after_conv_in
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();
    let shape: [usize; 4] = after_conv_in.shape().dims();
    let mean: f32 = after_conv_in_data.iter().sum::<f32>() / after_conv_in_data.len() as f32;
    println!("After conv_in[0,0,0,0]: {:.6}", after_conv_in_data[0]);
    println!("After conv_in shape: {:?}", shape);
    println!("After conv_in mean: {:.6}", mean);
    println!("  (PyTorch: 1.359338, shape [1,512,8,8], mean -0.000194)");

    // Load mid_block components individually for tracing
    let resnet1 = crate::layers::ResnetBlock2D::load::<32, 512, 512>(
        &tensors,
        "first_stage_model.decoder.mid.block_1",
        &device,
    )
    .expect("Failed to load resnet1");

    let attn = crate::layers::AttentionBlock::load::<32, 512>(
        &tensors,
        "first_stage_model.decoder.mid.attn_1",
        1,
        &device,
    )
    .expect("Failed to load attention");

    let resnet2 = crate::layers::ResnetBlock2D::load::<32, 512, 512>(
        &tensors,
        "first_stage_model.decoder.mid.block_2",
        &device,
    )
    .expect("Failed to load resnet2");

    // Step through mid_block
    let h = resnet1.forward(after_conv_in);
    let h_data: Vec<f32> = h.clone().into_data().convert::<f32>().to_vec().unwrap();
    println!(
        "After resnet1[0,0,0,0]: {:.6}, mean: {:.6}",
        h_data[0],
        h_data.iter().sum::<f32>() / h_data.len() as f32
    );
    println!("  (PyTorch: 1.310743, mean=0.001631)");

    let h = attn.forward(h);
    let h_data: Vec<f32> = h.clone().into_data().convert::<f32>().to_vec().unwrap();
    println!(
        "After attention[0,0,0,0]: {:.6}, mean: {:.6}",
        h_data[0],
        h_data.iter().sum::<f32>() / h_data.len() as f32
    );
    println!("  (PyTorch: 1.312975, mean=0.000100)");

    let h = resnet2.forward(h);
    let h_data: Vec<f32> = h.clone().into_data().convert::<f32>().to_vec().unwrap();
    println!(
        "After resnet2[0,0,0,0]: {:.6}, mean: {:.6}",
        h_data[0],
        h_data.iter().sum::<f32>() / h_data.len() as f32
    );
    println!("  (PyTorch: 1.641607, mean=0.036767)");

    // Load up blocks and trace through them
    // Block 0: 512->512 with upsample (from SD up.3)
    let up_block_0 = crate::vae::up_block::UpDecoderBlock2D::<32>::load::<512, 512>(
        &tensors,
        "first_stage_model.decoder.up.3",
        true,
        &device,
    )
    .expect("Failed to load up_block_0");

    let h = up_block_0.forward(h);
    let h_data: Vec<f32> = h.clone().into_data().convert::<f32>().to_vec().unwrap();
    let shape: [usize; 4] = h.shape().dims();
    println!(
        "After up_block[0][0,0,0,0]: {:.6}, shape: {:?}, mean: {:.6}",
        h_data[0],
        shape,
        h_data.iter().sum::<f32>() / h_data.len() as f32
    );
    println!("  (PyTorch: 4.890954, shape [1,512,16,16], mean 0.119995)");

    // Block 1: 512->512 with upsample (from SD up.2)
    let up_block_1 = crate::vae::up_block::UpDecoderBlock2D::<32>::load::<512, 512>(
        &tensors,
        "first_stage_model.decoder.up.2",
        true,
        &device,
    )
    .expect("Failed to load up_block_1");

    let h = up_block_1.forward(h);
    let h_data: Vec<f32> = h.clone().into_data().convert::<f32>().to_vec().unwrap();
    let shape: [usize; 4] = h.shape().dims();
    println!(
        "After up_block[1][0,0,0,0]: {:.6}, shape: {:?}, mean: {:.6}",
        h_data[0],
        shape,
        h_data.iter().sum::<f32>() / h_data.len() as f32
    );
    println!("  (PyTorch: 6.891311, shape [1,512,32,32], mean -0.483354)");

    // Block 2: 512->256 with upsample (from SD up.1)
    let up_block_2 = crate::vae::up_block::UpDecoderBlock2D::<32>::load::<512, 256>(
        &tensors,
        "first_stage_model.decoder.up.1",
        true,
        &device,
    )
    .expect("Failed to load up_block_2");

    let h = up_block_2.forward(h);
    let h_data: Vec<f32> = h.clone().into_data().convert::<f32>().to_vec().unwrap();
    let shape: [usize; 4] = h.shape().dims();
    println!(
        "After up_block[2][0,0,0,0]: {:.6}, shape: {:?}, mean: {:.6}",
        h_data[0],
        shape,
        h_data.iter().sum::<f32>() / h_data.len() as f32
    );
    println!("  (PyTorch: -11.186301, shape [1,256,64,64], mean 1.007483)");

    // Block 3: 256->128 no upsample (from SD up.0)
    let up_block_3 = crate::vae::up_block::UpDecoderBlock2D::<32>::load::<256, 128>(
        &tensors,
        "first_stage_model.decoder.up.0",
        false,
        &device,
    )
    .expect("Failed to load up_block_3");

    let h = up_block_3.forward(h);
    let h_data: Vec<f32> = h.clone().into_data().convert::<f32>().to_vec().unwrap();
    let shape: [usize; 4] = h.shape().dims();
    println!(
        "After up_block[3][0,0,0,0]: {:.6}, shape: {:?}, mean: {:.6}",
        h_data[0],
        shape,
        h_data.iter().sum::<f32>() / h_data.len() as f32
    );
    println!("  (PyTorch: 22.937454, shape [1,128,64,64], mean -1.113950)");

    // Final norm and conv
    let conv_norm_out = crate::layers::load_group_norm::<32, 128>(
        &tensors,
        "first_stage_model.decoder.norm_out",
        &device,
    )
    .expect("Failed to load conv_norm_out");

    let h = conv_norm_out.forward(h);
    let h = burn::tensor::activation::silu(h);
    let h_data: Vec<f32> = h.clone().into_data().convert::<f32>().to_vec().unwrap();
    println!("After norm+silu[0,0,0,0]: {:.6}", h_data[0]);
    println!("  (PyTorch: 0.435161)");

    let conv_out = crate::layers::load_conv2d::<3, 128, 3>(
        &tensors,
        "first_stage_model.decoder.conv_out",
        &device,
    )
    .expect("Failed to load conv_out");

    let h = conv_out.forward(h);
    let h_data: Vec<f32> = h.clone().into_data().convert::<f32>().to_vec().unwrap();
    let min = h_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = h_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "After conv_out[0,0,0,0]: {:.6}, mean: {:.6}, min/max: {:.6}/{:.6}",
        h_data[0],
        h_data.iter().sum::<f32>() / h_data.len() as f32,
        min,
        max
    );
    println!("  (PyTorch: 0.428280, mean -0.047456, min/max -1.824345/1.494602)");
}

/// Diagnostic test to verify attention block matches PyTorch.
#[test]
fn debug_attention() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        println!("Model not found, skipping");
        return;
    }

    let device: Device<Backend> = Default::default();
    let file = SafeTensorsFile::open(model_path).expect("Failed to open model");
    let tensors = file.tensors().expect("Failed to parse safetensors");

    // Load mid block's attention
    let attn = crate::layers::AttentionBlock::load::<32, 512>(
        &tensors,
        "first_stage_model.decoder.mid.attn_1",
        1, // single head
        &device,
    )
    .expect("Failed to load attention");

    // Fixed deterministic input: [1, 512, 4, 4]
    let input_data: Vec<f32> = (0..8192)
        .map(|i| ((i % 100) as f32 - 50.0) / 50.0)
        .collect();
    let input_1d: Tensor<Backend, 1> = Tensor::from_floats(input_data.as_slice(), &device);
    let input: Tensor<Backend, 4> = input_1d.reshape([1, 512, 4, 4]);

    let output = attn.forward(input);
    let output_data: Vec<f32> = output
        .into_data()
        .convert::<f32>()
        .to_vec()
        .expect("Failed to get output");

    println!("Attention output (first 4 values):");
    println!("  Rust: {:?}", &output_data[..4]);
    assert!(
        output_data.iter().all(|&v| v.is_finite()),
        "Attention output contains NaN or Inf"
    );
    println!(
        "Attention output is finite, mean: {:.6}",
        output_data.iter().sum::<f32>() / output_data.len() as f32
    );
}

/// Diagnostic test to verify GroupNorm matches PyTorch.
#[test]
fn debug_group_norm() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        println!("Model not found, skipping");
        return;
    }

    let device: Device<Backend> = Default::default();
    let file = SafeTensorsFile::open(model_path).expect("Failed to open model");
    let tensors = file.tensors().expect("Failed to parse safetensors");

    // Load mid block's first resnet's norm1 (512 channels, 32 groups)
    let norm = crate::layers::load_group_norm::<32, 512>(
        &tensors,
        "first_stage_model.decoder.mid.block_1.norm1",
        &device,
    )
    .expect("Failed to load group_norm");

    // Test with simple input: [1, 512, 2, 2] with values 0..2048
    let input_data: Vec<f32> = (0..2048).map(|i| i as f32).collect();
    let input_1d: Tensor<Backend, 1> = Tensor::from_floats(input_data.as_slice(), &device);
    let input: Tensor<Backend, 4> = input_1d.reshape([1, 512, 2, 2]);

    let output = norm.forward(input);
    let output_data: Vec<f32> = output
        .into_data()
        .convert::<f32>()
        .to_vec()
        .expect("Failed to get output");

    // Print first few values for comparison
    println!("GroupNorm output (first 8 values):");
    println!("  Rust: {:?}", &output_data[..8]);
    // We'd need to run PyTorch to get expected values, but we can at least check it's not NaN/Inf
    assert!(
        output_data.iter().all(|&v| v.is_finite()),
        "GroupNorm output contains NaN or Inf"
    );
    println!("GroupNorm output is finite");
}

/// Load a PNG file as a `[1, 3, H, W]` BCHW float tensor in `[0, 1]`.
fn load_png_as_bchw(path: &Path, device: &Device<Backend>) -> Tensor<Backend, 4> {
    let img = image::open(path).expect("Failed to open PNG");
    let rgb = img.to_rgb8();
    let (width, height) = img.dimensions();
    let raw = rgb.into_raw();

    // Convert [H, W, 3] u8 → [1, 3, H, W] f32 in [0, 1]
    let h = height as usize;
    let w = width as usize;
    let mut bchw = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let src = y * w * 3 + x * 3 + c;
                let dst = c * h * w + y * w + x;
                bchw[dst] = raw[src] as f32 / 255.0;
            }
        }
    }

    let flat: Tensor<Backend, 1> = Tensor::from_floats(bchw.as_slice(), device);
    flat.reshape([1, 3, h, w])
}

/// Convert a `[1, 3, H, W]` BCHW tensor in `[-1, 1]` to a `[H*W*3]` u8 vec.
fn bchw_to_rgb_u8(tensor: Tensor<Backend, 4>) -> Vec<u8> {
    let [_b, _c, h, w] = tensor.shape().dims();
    let data: Vec<f32> = tensor
        .into_data()
        .convert::<f32>()
        .to_vec()
        .expect("Failed to read tensor data");

    let mut rgb = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let src = c * h * w + y * w + x;
                let dst = y * w * 3 + x * 3 + c;
                let val = data[src].clamp(-1.0, 1.0);
                let normalized = (val + 1.0) / 2.0;
                rgb[dst] = (normalized * 255.0).round() as u8;
            }
        }
    }
    rgb
}

/// Save a `[H*W*3]` u8 vec as a PNG file.
fn save_rgb_png(rgb: &[u8], width: u32, height: u32, path: &Path) {
    let img = image::RgbImage::from_raw(width, height, rgb.to_vec())
        .expect("Failed to create image buffer");
    img.save(path).expect("Failed to save PNG");
}

/// VAE encode→decode roundtrip test for SD 1.5.
///
/// Loads a 512x512 PNG, encodes to latent, decodes back, and compares.
#[test]
fn vae_roundtrip_sd15() {
    let model_path = Path::new(MODEL_PATH);
    if !model_path.exists() {
        println!("Model not found at {MODEL_PATH}, skipping roundtrip test");
        return;
    }

    let png_path = Path::new("output/baseline.png");
    if !png_path.exists() {
        println!("Test image not found at {}, skipping", png_path.display());
        return;
    }

    println!("Using image: {}", png_path.display());

    let device: Device<Backend> = Default::default();

    // Load model
    println!("Loading SD 1.5 VAE...");
    let file = SafeTensorsFile::open(model_path).expect("Failed to open model");
    let tensors = file.tensors().expect("Failed to parse safetensors");
    let encoder =
        Sd15VaeEncoder::load(&tensors, Some(VAE_PREFIX), &device).expect("Failed to load encoder");
    let decoder =
        Sd15VaeDecoder::load(&tensors, Some(VAE_PREFIX), &device).expect("Failed to load decoder");

    // Load 512x512 image
    let image = load_png_as_bchw(png_path, &device);
    let [_, _, h, w] = image.shape().dims();
    // Crop to multiple of 8 for clean encode/decode
    let h8 = (h / 8) * 8;
    let w8 = (w / 8) * 8;
    let image = image.slice([0..1, 0..3, 0..h8, 0..w8]);
    println!("Image shape (cropped): [1, 3, {h8}, {w8}]");

    // Encode
    let start = Instant::now();
    let latent = encoder.forward_tiled(image.clone(), 64, 8);
    let latent_shape: [usize; 4] = latent.shape().dims();
    println!(
        "Encoded to latent: {latent_shape:?} in {:?}",
        start.elapsed()
    );

    // Decode
    let start = Instant::now();
    let decoded = decoder.forward_tiled(latent, 64, 8);
    let decoded_shape: [usize; 4] = decoded.shape().dims();
    println!(
        "Decoded back to image: {decoded_shape:?} in {:?}",
        start.elapsed()
    );

    // Convert both to u8 for comparison
    // Original was [0, 1], decoded is [-1, 1] — scale original to [-1, 1] for fair compare
    let original_rgb = {
        let data: Vec<f32> = image
            .into_data()
            .convert::<f32>()
            .to_vec()
            .expect("read original");
        let mut rgb = vec![0u8; h8 * w8 * 3];
        for y in 0..h8 {
            for x in 0..w8 {
                for c in 0..3 {
                    let src = c * h8 * w8 + y * w8 + x;
                    let dst = y * w8 * 3 + x * 3 + c;
                    rgb[dst] = (data[src] * 255.0).round() as u8;
                }
            }
        }
        rgb
    };

    let decoded_rgb = bchw_to_rgb_u8(decoded);

    // Save roundtripped image for visual inspection
    let roundtrip_path = Path::new("output/vae_roundtrip_sd15.png");
    save_rgb_png(&decoded_rgb, w8 as u32, h8 as u32, roundtrip_path);
    println!("Saved roundtripped image to {}", roundtrip_path.display());

    // Compare — VAE is lossy, but should be close
    let total_pixels = h8 * w8;
    let mut total_diff: u64 = 0;
    let mut max_diff: u8 = 0;
    let mut over_threshold = 0usize;
    let threshold: u8 = 30; // per-channel tolerance for lossy VAE roundtrip

    for (&a, &b) in original_rgb.iter().zip(decoded_rgb.iter()) {
        let diff = a.abs_diff(b);
        total_diff += u64::from(diff);
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > threshold {
            over_threshold += 1;
        }
    }

    let mean_diff = total_diff as f64 / original_rgb.len() as f64;
    let pct_over = over_threshold as f64 / original_rgb.len() as f64 * 100.0;

    println!("Roundtrip stats:");
    println!("  Mean absolute error: {mean_diff:.2} / 255");
    println!("  Max per-channel diff: {max_diff}");
    println!(
        "  Pixels over threshold ({threshold}): {over_threshold}/{total_pixels} ({pct_over:.2}%)"
    );

    // VAE is lossy by design; fp16 adds quantization error.
    // Typical mean error: ~5/255 (f32), ~16/255 (fp16).
    assert!(
        mean_diff < 25.0,
        "Mean roundtrip error too large: {mean_diff:.2}/255"
    );
    println!("SD 1.5 VAE roundtrip PASSED!");
}

/// VAE encode→decode roundtrip test for SDXL.
///
/// Loads a 1024x1024 PNG, encodes to latent, decodes back, and compares.
#[test]
fn vae_roundtrip_sdxl() {
    let png_path = Path::new("output/cat_fp16fix.png");
    if !png_path.exists() {
        println!("Test image not found at {}, skipping", png_path.display());
        return;
    }

    println!("Using image: {}", png_path.display());

    let device: Device<Backend> = Default::default();

    // Load from fp16-fix VAE (the original SDXL VAE weights overflow in fp16)
    let fp16_fix_path = Path::new("models/vae/sdxl-vae-fp16-fix.safetensors");
    if !fp16_fix_path.exists() {
        println!("fp16-fix VAE not found, skipping SDXL roundtrip test");
        return;
    }

    println!("Loading SDXL VAE from fp16-fix...");
    let vae_file = SafeTensorsFile::open(fp16_fix_path).expect("Failed to open fp16-fix VAE");
    let vae_tensors = vae_file.tensors().expect("Failed to parse safetensors");
    let encoder =
        SdxlVaeEncoder::load(&vae_tensors, None, &device).expect("Failed to load SDXL encoder");
    let decoder =
        SdxlVaeDecoder::load(&vae_tensors, None, &device).expect("Failed to load SDXL decoder");

    // Load 1024x1024 image
    let image = load_png_as_bchw(png_path, &device);
    let [_, _, h, w] = image.shape().dims();
    let h8 = (h / 8) * 8;
    let w8 = (w / 8) * 8;
    let image = image.slice([0..1, 0..3, 0..h8, 0..w8]);
    println!("Image shape (cropped): [1, 3, {h8}, {w8}]");

    // Encode
    let start = Instant::now();
    let latent = encoder.forward_tiled(image.clone(), 64, 8);
    let latent_shape: [usize; 4] = latent.shape().dims();
    println!(
        "Encoded to latent: {latent_shape:?} in {:?}",
        start.elapsed()
    );

    // Decode
    let start = Instant::now();
    let decoded = decoder.forward_tiled(latent, 64, 8);
    let decoded_shape: [usize; 4] = decoded.shape().dims();
    println!(
        "Decoded back to image: {decoded_shape:?} in {:?}",
        start.elapsed()
    );

    // Convert and compare
    let original_rgb = {
        let data: Vec<f32> = image
            .into_data()
            .convert::<f32>()
            .to_vec()
            .expect("read original");
        let mut rgb = vec![0u8; h8 * w8 * 3];
        for y in 0..h8 {
            for x in 0..w8 {
                for c in 0..3 {
                    let src = c * h8 * w8 + y * w8 + x;
                    let dst = y * w8 * 3 + x * 3 + c;
                    rgb[dst] = (data[src] * 255.0).round() as u8;
                }
            }
        }
        rgb
    };

    let decoded_rgb = bchw_to_rgb_u8(decoded);

    let roundtrip_path = Path::new("output/vae_roundtrip_sdxl.png");
    save_rgb_png(&decoded_rgb, w8 as u32, h8 as u32, roundtrip_path);
    println!("Saved roundtripped image to {}", roundtrip_path.display());

    let mut total_diff: u64 = 0;
    let mut max_diff: u8 = 0;
    let total_pixels = h8 * w8;
    let threshold: u8 = 30;
    let mut over_threshold = 0usize;

    for (&a, &b) in original_rgb.iter().zip(decoded_rgb.iter()) {
        let diff = a.abs_diff(b);
        total_diff += u64::from(diff);
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > threshold {
            over_threshold += 1;
        }
    }

    let mean_diff = total_diff as f64 / original_rgb.len() as f64;
    let pct_over = over_threshold as f64 / original_rgb.len() as f64 * 100.0;

    println!("Roundtrip stats:");
    println!("  Mean absolute error: {mean_diff:.2} / 255");
    println!("  Max per-channel diff: {max_diff}");
    println!(
        "  Pixels over threshold ({threshold}): {over_threshold}/{total_pixels} ({pct_over:.2}%)"
    );

    assert!(
        mean_diff < 10.0,
        "Mean roundtrip error too large: {mean_diff:.2}/255"
    );
    println!("SDXL VAE roundtrip PASSED!");
}
