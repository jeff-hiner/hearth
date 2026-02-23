//! UNet step comparison test: INT8 CMMA vs f16 attention.
//!
//! Compares a single UNet forward pass output using:
//! - Default: f16 CMMA (FlashAttention)
//! - INT8_CMMA=1: INT8 CMMA (SageAttention)
//!
//! Run with:
//!   cargo test --test unet_step_comparison --features gpu-vulkan-f16 -- --nocapture

#![cfg(feature = "gpu-vulkan-f16")]

use burn::tensor::{Distribution, Tensor};
use hearth::{model_loader::SafeTensorsFile, types::Backend, unet::Sd15Unet2D};
use std::path::Path;

/// Path to the SD 1.5 checkpoint file.
const CHECKPOINT_PATH: &str = "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors";

/// Compare single UNet step between INT8 (SageAttention) and f16 (FlashAttention).
///
/// This test exercises the UNet with identical inputs under both attention modes
/// and compares the output tensors to verify INT8 attention produces acceptable results.
#[test]
fn test_unet_step_int8_vs_f16() {
    let checkpoint_path = Path::new(CHECKPOINT_PATH);
    if !checkpoint_path.exists() {
        println!("Model not found at {CHECKPOINT_PATH}, skipping");
        return;
    }

    let device = Default::default();

    // Set deterministic seed for reproducibility
    let seed = 42u64;
    <Backend as burn::tensor::backend::Backend>::seed(&device, seed);

    println!("\n=== UNet Step Comparison: INT8 vs f16 ===\n");

    // Load checkpoint and UNet
    println!("Loading checkpoint...");
    let file = SafeTensorsFile::open(checkpoint_path).expect("Failed to open checkpoint");
    let tensors = file.tensors().expect("Failed to parse safetensors");

    println!("Loading UNet...");
    let unet = Sd15Unet2D::load(&tensors, &device).expect("Failed to load UNet");

    // Create deterministic test inputs
    // Latent: [1, 4, 64, 64] - standard 512x512 latent size
    // Reset seed before each tensor creation for reproducibility
    <Backend as burn::tensor::backend::Backend>::seed(&device, seed);
    let latent: Tensor<Backend, 4> =
        Tensor::random([1, 4, 64, 64], Distribution::Normal(0.0, 1.0), &device);

    // Timestep: a mid-range value (500 out of 1000)
    let timestep: Tensor<Backend, 1> = Tensor::full([1], 500.0f32, &device);

    // Context: [1, 77, 768] - CLIP text embedding dimensions
    <Backend as burn::tensor::backend::Backend>::seed(&device, seed + 1);
    let context: Tensor<Backend, 3> =
        Tensor::random([1, 77, 768], Distribution::Normal(0.0, 1.0), &device);

    println!("Test inputs:");
    println!("  Latent shape: {:?}", latent.shape().dims::<4>());
    println!("  Timestep: 500.0");
    println!("  Context shape: {:?}", context.shape().dims::<3>());
    println!();

    // Run with f16 attention (FlashAttention) — the default
    println!("Running UNet with f16 attention (default)...");
    // SAFETY: We're in a single-threaded test, no other code accesses this env var concurrently.
    unsafe { std::env::remove_var("INT8_CMMA") };

    let f16_output = unet.forward(latent.clone(), timestep.clone(), context.clone());
    let f16_data: Vec<f32> = f16_output
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();

    println!("  f16 output shape: {:?}", f16_output.shape().dims::<4>());
    println!(
        "  f16 output range: [{:.4}, {:.4}]",
        f16_data.iter().cloned().fold(f32::INFINITY, f32::min),
        f16_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "  f16 output first 8: {:?}",
        &f16_data[..8.min(f16_data.len())]
    );
    println!();

    // Run with INT8 attention (SageAttention — requires INT8_CMMA=1)
    println!("Running UNet with INT8 attention (INT8_CMMA=1)...");
    // SAFETY: We're in a single-threaded test, no other code accesses this env var concurrently.
    unsafe { std::env::set_var("INT8_CMMA", "1") };

    let int8_output = unet.forward(latent, timestep, context);
    let int8_data: Vec<f32> = int8_output
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();

    println!("  INT8 output shape: {:?}", int8_output.shape().dims::<4>());
    println!(
        "  INT8 output range: [{:.4}, {:.4}]",
        int8_data.iter().cloned().fold(f32::INFINITY, f32::min),
        int8_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "  INT8 output first 8: {:?}",
        &int8_data[..8.min(int8_data.len())]
    );
    println!();

    // Compare outputs
    println!("=== Comparison ===");

    let (max_diff, max_diff_idx, avg_diff, rel_error) = compute_diff_stats(&f16_data, &int8_data);

    println!(
        "Max absolute diff: {:.6} at index {}",
        max_diff, max_diff_idx
    );
    println!("  f16[{}] = {:.6}", max_diff_idx, f16_data[max_diff_idx]);
    println!("  INT8[{}] = {:.6}", max_diff_idx, int8_data[max_diff_idx]);
    println!("Average absolute diff: {:.6}", avg_diff);
    println!("Relative error: {:.2}%", rel_error * 100.0);
    println!("Total elements: {}", f16_data.len());

    // Show histogram of differences
    print_diff_histogram(&f16_data, &int8_data);

    // Check for NaN/Inf
    let f16_has_nan = f16_data.iter().any(|x| x.is_nan());
    let f16_has_inf = f16_data.iter().any(|x| x.is_infinite());
    let int8_has_nan = int8_data.iter().any(|x| x.is_nan());
    let int8_has_inf = int8_data.iter().any(|x| x.is_infinite());

    assert!(!f16_has_nan, "f16 output contains NaN");
    assert!(!f16_has_inf, "f16 output contains Inf");
    assert!(!int8_has_nan, "INT8 output contains NaN");
    assert!(!int8_has_inf, "INT8 output contains Inf");

    // Assert relative error is within tolerance
    // Plan specifies <15% for single UNet step
    let tolerance = 0.15;
    assert!(
        rel_error < tolerance,
        "Relative error {:.2}% exceeds tolerance {:.0}%",
        rel_error * 100.0,
        tolerance * 100.0
    );

    println!("\nUNet step comparison PASSED!");
}

/// Compute statistics about the differences between two vectors.
fn compute_diff_stats(a: &[f32], b: &[f32]) -> (f32, usize, f32, f32) {
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0usize;
    let mut sum_diff = 0.0f32;
    let mut sum_sq_a = 0.0f32;
    let mut sum_sq_diff = 0.0f32;

    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
        sum_diff += diff;
        sum_sq_a += va * va;
        sum_sq_diff += (va - vb) * (va - vb);
    }

    let avg_diff = sum_diff / a.len() as f32;
    let rel_error = (sum_sq_diff / sum_sq_a).sqrt();

    (max_diff, max_diff_idx, avg_diff, rel_error)
}

/// Print a histogram of absolute differences.
fn print_diff_histogram(a: &[f32], b: &[f32]) {
    let thresholds = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0];
    let mut counts = vec![0usize; thresholds.len() + 1];

    for (&va, &vb) in a.iter().zip(b.iter()) {
        let diff = (va - vb).abs();
        let mut found = false;
        for (i, &t) in thresholds.iter().enumerate() {
            if diff < t {
                counts[i] += 1;
                found = true;
                break;
            }
        }
        if !found {
            counts[thresholds.len()] += 1;
        }
    }

    println!("\nDifference distribution:");
    let total = a.len();
    for (i, &t) in thresholds.iter().enumerate() {
        let pct = 100.0 * counts[i] as f64 / total as f64;
        let label = if i == 0 {
            format!("< {:.3}", t)
        } else {
            format!("{:.3} - {:.3}", thresholds[i - 1], t)
        };
        println!("  {:12}: {:>8} ({:>5.1}%)", label, counts[i], pct);
    }
    let pct = 100.0 * counts[thresholds.len()] as f64 / total as f64;
    println!(
        "  {:12}: {:>8} ({:>5.1}%)",
        format!(">= {:.3}", thresholds[thresholds.len() - 1]),
        counts[thresholds.len()],
        pct
    );
}

/// Test with multiple timesteps to check error accumulation pattern.
#[test]
fn test_unet_multiple_timesteps() {
    let checkpoint_path = Path::new(CHECKPOINT_PATH);
    if !checkpoint_path.exists() {
        println!("Model not found at {CHECKPOINT_PATH}, skipping");
        return;
    }

    let device = Default::default();
    let seed = 42u64;

    println!("\n=== UNet Multiple Timesteps: INT8 vs f16 ===\n");

    // Load checkpoint and UNet
    let file = SafeTensorsFile::open(checkpoint_path).expect("Failed to open checkpoint");
    let tensors = file.tensors().expect("Failed to parse safetensors");
    let unet = Sd15Unet2D::load(&tensors, &device).expect("Failed to load UNet");

    // Test at different timesteps
    let timesteps = [999.0f32, 750.0, 500.0, 250.0, 50.0];

    println!(
        "{:>10} {:>12} {:>12} {:>12}",
        "Timestep", "Max Diff", "Avg Diff", "Rel Error"
    );
    println!("{:-<50}", "");

    for &t in &timesteps {
        // Create inputs with same seed
        <Backend as burn::tensor::backend::Backend>::seed(&device, seed);
        let latent: Tensor<Backend, 4> =
            Tensor::random([1, 4, 64, 64], Distribution::Normal(0.0, 1.0), &device);
        let timestep: Tensor<Backend, 1> = Tensor::full([1], t, &device);
        <Backend as burn::tensor::backend::Backend>::seed(&device, seed + 1);
        let context: Tensor<Backend, 3> =
            Tensor::random([1, 77, 768], Distribution::Normal(0.0, 1.0), &device);

        // f16 attention (default)
        // SAFETY: We're in a single-threaded test, no other code accesses this env var concurrently.
        unsafe { std::env::remove_var("INT8_CMMA") };
        let f16_out = unet.forward(latent.clone(), timestep.clone(), context.clone());
        let f16_data: Vec<f32> = f16_out.into_data().convert::<f32>().to_vec().unwrap();

        // INT8 attention
        // SAFETY: We're in a single-threaded test, no other code accesses this env var concurrently.
        unsafe { std::env::set_var("INT8_CMMA", "1") };
        let int8_out = unet.forward(latent, timestep, context);
        let int8_data: Vec<f32> = int8_out.into_data().convert::<f32>().to_vec().unwrap();

        let (max_diff, _, avg_diff, rel_error) = compute_diff_stats(&f16_data, &int8_data);

        println!(
            "{:>10.0} {:>12.6} {:>12.6} {:>11.2}%",
            t,
            max_diff,
            avg_diff,
            rel_error * 100.0
        );

        // Assert within tolerance
        assert!(
            rel_error < 0.20,
            "Timestep {} relative error {:.2}% exceeds 20%",
            t,
            rel_error * 100.0
        );
    }

    println!("\nMultiple timesteps test PASSED!");
}
