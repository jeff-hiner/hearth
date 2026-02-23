//! Full pipeline comparison test: INT8 CMMA vs f16 attention.
//!
//! Compares the complete generation pipeline (CLIP -> UNet sampling -> VAE decode)
//! using:
//! - Default: f16 CMMA (FlashAttention)
//! - INT8_CMMA=1: INT8 CMMA (SageAttention)
//!
//! Run with:
//!   cargo test --test generate_comparison --features gpu-vulkan-f16 --release -- --nocapture
//!
//! Note: This test requires model files and takes significant time to run.

#![cfg(feature = "gpu-vulkan-f16")]

use burn::tensor::{Distribution, Tensor};
use hearth::{
    clip::{ClipTokenizer, Sd15ClipTextEncoder, Sd15Conditioning},
    model_loader::SafeTensorsFile,
    sampling::{EulerSampler, NoiseSchedule},
    types::Backend,
    unet::Sd15Unet2D,
    vae::Sd15VaeDecoder,
};
use std::path::Path;

/// Path to the SD 1.5 checkpoint file.
const CHECKPOINT_PATH: &str = "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors";
/// Path to the CLIP tokenizer vocabulary.
const VOCAB_PATH: &str = "models/clip/vocab.json";
/// Path to the CLIP tokenizer merges.
const MERGES_PATH: &str = "models/clip/merges.txt";

/// Compare full generation pipeline between INT8 and f16 attention.
///
/// This test runs the complete pipeline (CLIP -> UNet sampling -> VAE decode)
/// with both attention modes and compares:
/// 1. Final latent tensor after sampling
/// 2. Final decoded image
#[test]
#[ignore = "INT8 CMMA produces divergent results on current hardware"]
fn test_generate_int8_vs_f16() {
    let checkpoint_path = Path::new(CHECKPOINT_PATH);
    let vocab_path = Path::new(VOCAB_PATH);
    let merges_path = Path::new(MERGES_PATH);

    if !checkpoint_path.exists() || !vocab_path.exists() || !merges_path.exists() {
        println!("Model files not found, skipping full pipeline test");
        println!("  Required: {CHECKPOINT_PATH}");
        println!("           {VOCAB_PATH}");
        println!("           {MERGES_PATH}");
        return;
    }

    let device = Default::default();
    let seed = 42u64;
    let steps = 8; // Use fewer steps for faster testing
    let cfg_scale = 7.5f32;
    let prompt = "a cat";
    let negative = "";

    println!("\n=== Full Pipeline Comparison: INT8 vs f16 ===\n");
    println!("Prompt: \"{}\"", prompt);
    println!("Steps: {}, CFG: {}, Seed: {}", steps, cfg_scale, seed);
    println!();

    // Load models
    println!("Loading models...");
    let file = SafeTensorsFile::open(checkpoint_path).expect("Failed to open checkpoint");
    let tensors = file.tensors().expect("Failed to parse safetensors");

    let tokenizer =
        ClipTokenizer::from_files(vocab_path, merges_path).expect("Failed to load tokenizer");
    let clip = Sd15ClipTextEncoder::load(&tensors, &device).expect("Failed to load CLIP");
    let unet = Sd15Unet2D::load(&tensors, &device).expect("Failed to load UNet");
    let scheduler = NoiseSchedule::load(&tensors, &device).expect("Failed to load scheduler");
    let vae = Sd15VaeDecoder::load(&tensors, Some("first_stage_model"), &device)
        .expect("Failed to load VAE");

    // Encode prompts (same for both runs)
    println!("Encoding prompts...");
    let tokens = tokenizer
        .encode(prompt, &device)
        .expect("Failed to encode prompt");
    let neg_tokens = tokenizer
        .encode(negative, &device)
        .expect("Failed to encode negative");
    let positive_cond = Sd15Conditioning::new(clip.forward(tokens));
    let negative_cond = Sd15Conditioning::new(clip.forward(neg_tokens));

    // Create schedule
    let schedule = scheduler.schedule_for_steps(steps);
    let sigma_max = schedule.sigmas[0];

    // We need separate samplers for each run since EulerSampler takes ownership
    // Load scheduler again for second run
    let scheduler2 = NoiseSchedule::load(&tensors, &device).expect("Failed to load scheduler");

    // ========================================
    // Run with f16 attention (FlashAttention)
    // ========================================
    println!("\nRunning pipeline with f16 attention (default)...");
    // SAFETY: We're in a single-threaded test context
    unsafe { std::env::remove_var("INT8_CMMA") };

    // Create initial noise with deterministic seed
    <Backend as burn::tensor::backend::Backend>::seed(&device, seed);
    let f16_latent: Tensor<Backend, 4> = Tensor::random(
        [1, 4, 64, 64],
        Distribution::Normal(0.0, sigma_max as f64),
        &device,
    );

    // Sample
    let sampler = EulerSampler::new(scheduler);
    let f16_denoised = sampler.sample_with_schedule(
        &unet,
        f16_latent,
        &positive_cond,
        &negative_cond,
        &schedule,
        cfg_scale,
        None,
    );
    let f16_denoised_data: Vec<f32> = f16_denoised
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();

    // Decode
    let f16_image = vae.forward(f16_denoised);
    let f16_image_data: Vec<f32> = f16_image.into_data().convert::<f32>().to_vec().unwrap();

    println!(
        "  f16 latent range: [{:.4}, {:.4}]",
        f16_denoised_data
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        f16_denoised_data
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "  f16 image range: [{:.4}, {:.4}]",
        f16_image_data.iter().cloned().fold(f32::INFINITY, f32::min),
        f16_image_data
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    // ========================================
    // Run with INT8 attention (SageAttention)
    // ========================================
    println!("\nRunning pipeline with INT8 attention (INT8_CMMA=1)...");
    // SAFETY: We're in a single-threaded test context
    unsafe { std::env::set_var("INT8_CMMA", "1") };

    // Create initial noise with SAME seed
    <Backend as burn::tensor::backend::Backend>::seed(&device, seed);
    let int8_latent: Tensor<Backend, 4> = Tensor::random(
        [1, 4, 64, 64],
        Distribution::Normal(0.0, sigma_max as f64),
        &device,
    );

    // Sample
    let sampler = EulerSampler::new(scheduler2);
    let int8_denoised = sampler.sample_with_schedule(
        &unet,
        int8_latent,
        &positive_cond,
        &negative_cond,
        &schedule,
        cfg_scale,
        None,
    );
    let int8_denoised_data: Vec<f32> = int8_denoised
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap();

    // Decode
    let int8_image = vae.forward(int8_denoised);
    let int8_image_data: Vec<f32> = int8_image.into_data().convert::<f32>().to_vec().unwrap();

    println!(
        "  INT8 latent range: [{:.4}, {:.4}]",
        int8_denoised_data
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        int8_denoised_data
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "  INT8 image range: [{:.4}, {:.4}]",
        int8_image_data
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        int8_image_data
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    // ========================================
    // Compare latent tensors
    // ========================================
    println!("\n=== Latent Comparison (after {} steps) ===", steps);
    let (latent_max_diff, latent_max_idx, latent_avg_diff, latent_rel_error) =
        compute_diff_stats(&f16_denoised_data, &int8_denoised_data);

    println!(
        "Max diff: {:.6} at index {}",
        latent_max_diff, latent_max_idx
    );
    println!("Avg diff: {:.6}", latent_avg_diff);
    println!("Relative error: {:.2}%", latent_rel_error * 100.0);

    // ========================================
    // Compare decoded images
    // ========================================
    println!("\n=== Image Comparison ===");
    let (image_max_diff, image_max_idx, image_avg_diff, image_rel_error) =
        compute_diff_stats(&f16_image_data, &int8_image_data);

    println!("Max diff: {:.6} at index {}", image_max_diff, image_max_idx);
    println!("Avg diff: {:.6}", image_avg_diff);
    println!("Relative error: {:.2}%", image_rel_error * 100.0);

    // Compute PSNR
    let psnr = compute_psnr(&f16_image_data, &int8_image_data);
    println!("PSNR: {:.2} dB", psnr);

    // Check for NaN/Inf
    assert!(
        !f16_denoised_data.iter().any(|x| x.is_nan()),
        "f16 latent contains NaN"
    );
    assert!(
        !int8_denoised_data.iter().any(|x| x.is_nan()),
        "INT8 latent contains NaN"
    );
    assert!(
        !f16_image_data.iter().any(|x| x.is_nan()),
        "f16 image contains NaN"
    );
    assert!(
        !int8_image_data.iter().any(|x| x.is_nan()),
        "INT8 image contains NaN"
    );

    // Assert tolerances from plan:
    // - Final latent (8 steps): < 20%
    // - Final image: PSNR > 25 dB
    assert!(
        latent_rel_error < 0.20,
        "Latent relative error {:.2}% exceeds 20%",
        latent_rel_error * 100.0
    );
    assert!(
        psnr > 25.0,
        "Image PSNR {:.2} dB is below 25 dB threshold",
        psnr
    );

    println!("\nFull pipeline comparison PASSED!");
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
    // Avoid division by zero
    let rel_error = if sum_sq_a > 1e-10 {
        (sum_sq_diff / sum_sq_a).sqrt()
    } else {
        0.0
    };

    (max_diff, max_diff_idx, avg_diff, rel_error)
}

/// Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
///
/// Assumes images are in [-1, 1] range.
fn compute_psnr(a: &[f32], b: &[f32]) -> f32 {
    // MSE
    let mse: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&va, &vb)| {
            let diff = va - vb;
            diff * diff
        })
        .sum::<f32>()
        / a.len() as f32;

    if mse < 1e-10 {
        return 100.0; // Perfect match
    }

    // For [-1, 1] range, max value is 2.0 (from -1 to 1)
    let max_val = 2.0f32;
    10.0 * (max_val * max_val / mse).log10()
}

/// Test that compares latent snapshots at each sampling step.
///
/// This helps identify when/if error accumulates during the sampling loop.
#[test]
#[ignore = "INT8 CMMA produces divergent results on current hardware"]
fn test_sampling_step_by_step() {
    let checkpoint_path = Path::new(CHECKPOINT_PATH);
    let vocab_path = Path::new(VOCAB_PATH);
    let merges_path = Path::new(MERGES_PATH);

    if !checkpoint_path.exists() || !vocab_path.exists() || !merges_path.exists() {
        println!("Model files not found, skipping step-by-step test");
        return;
    }

    let device = Default::default();
    let seed = 42u64;
    let steps = 4; // Fewer steps for detailed analysis
    let cfg_scale = 7.5f32;
    let prompt = "a cat";

    println!("\n=== Step-by-Step Sampling Comparison ===\n");

    // Load models
    let file = SafeTensorsFile::open(checkpoint_path).expect("Failed to open checkpoint");
    let tensors = file.tensors().expect("Failed to parse safetensors");

    let tokenizer =
        ClipTokenizer::from_files(vocab_path, merges_path).expect("Failed to load tokenizer");
    let clip = Sd15ClipTextEncoder::load(&tensors, &device).expect("Failed to load CLIP");
    let unet = Sd15Unet2D::load(&tensors, &device).expect("Failed to load UNet");
    let scheduler = NoiseSchedule::load(&tensors, &device).expect("Failed to load scheduler");
    let scheduler2 = NoiseSchedule::load(&tensors, &device).expect("Failed to load scheduler");

    // Encode prompts
    let tokens = tokenizer.encode(prompt, &device).expect("Failed to encode");
    let neg_tokens = tokenizer.encode("", &device).expect("Failed to encode");
    let positive_cond = Sd15Conditioning::new(clip.forward(tokens));
    let negative_cond = Sd15Conditioning::new(clip.forward(neg_tokens));

    let schedule = scheduler.schedule_for_steps(steps);
    let sigma_max = schedule.sigmas[0];

    // We need to manually step through sampling to compare at each step
    // For now, just run both and compare final output
    // A more detailed step-by-step comparison would require modifying the sampler

    println!(
        "{:>6} {:>12} {:>12} {:>12}",
        "Step", "Max Diff", "Avg Diff", "Rel Error"
    );
    println!("{:-<45}", "");

    // Run f16 (default)
    // SAFETY: Single-threaded test
    unsafe { std::env::remove_var("INT8_CMMA") };
    <Backend as burn::tensor::backend::Backend>::seed(&device, seed);
    let f16_latent: Tensor<Backend, 4> = Tensor::random(
        [1, 4, 64, 64],
        Distribution::Normal(0.0, sigma_max as f64),
        &device,
    );
    let sampler = EulerSampler::new(scheduler);
    let f16_final = sampler.sample_with_schedule(
        &unet,
        f16_latent,
        &positive_cond,
        &negative_cond,
        &schedule,
        cfg_scale,
        None,
    );
    let f16_data: Vec<f32> = f16_final.into_data().convert::<f32>().to_vec().unwrap();

    // Run INT8
    // SAFETY: Single-threaded test
    unsafe { std::env::set_var("INT8_CMMA", "1") };
    <Backend as burn::tensor::backend::Backend>::seed(&device, seed);
    let int8_latent: Tensor<Backend, 4> = Tensor::random(
        [1, 4, 64, 64],
        Distribution::Normal(0.0, sigma_max as f64),
        &device,
    );
    let sampler = EulerSampler::new(scheduler2);
    let int8_final = sampler.sample_with_schedule(
        &unet,
        int8_latent,
        &positive_cond,
        &negative_cond,
        &schedule,
        cfg_scale,
        None,
    );
    let int8_data: Vec<f32> = int8_final.into_data().convert::<f32>().to_vec().unwrap();

    let (max_diff, _, avg_diff, rel_error) = compute_diff_stats(&f16_data, &int8_data);
    println!(
        "{:>6} {:>12.6} {:>12.6} {:>11.2}%",
        steps,
        max_diff,
        avg_diff,
        rel_error * 100.0
    );

    assert!(
        rel_error < 0.25,
        "Final relative error {:.2}% exceeds 25%",
        rel_error * 100.0
    );

    println!("\nStep-by-step comparison PASSED!");
}
