//! Minimal image generation demo.
//!
//! Generates a 512x512 image from a text prompt using Stable Diffusion 1.5.
//!
//! Usage:
//!   cargo run --release --example generate -- "a cat sitting on a couch"
//!   cargo run --release --example generate -- "a cat" -o cat.png --steps 30
//!
//! Requires:
//!   - models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors
//!   - models/clip/vocab.json
//!   - models/clip/merges.txt

use burn::tensor::{Distribution, Tensor};
use clap::Parser;
use hearth::{
    clip::{ClipTokenizer, Sd15ClipTextEncoder, Sd15Conditioning},
    controlnet::{ControlNetUnit, Sd15ControlNet},
    lora::LoraFile,
    model_loader::SafeTensorsFile,
    sampling::{
        DpmPp2mSampler, DpmPpSdeSampler, EulerASampler, EulerSampler, NoiseSchedule, SamplerKind,
        SchedulerKind,
    },
    startup,
    types::Backend,
    unet::Sd15Unet2D,
    vae::Sd15VaeDecoder,
};
use std::{cell::Cell, path::Path, process, time::Instant};
use tracing_subscriber::EnvFilter;

/// Path to the SD 1.5 checkpoint file.
const CHECKPOINT_PATH: &str = "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors";
/// Path to the CLIP tokenizer vocabulary.
const VOCAB_PATH: &str = "models/clip/vocab.json";
/// Path to the CLIP tokenizer merges.
const MERGES_PATH: &str = "models/clip/merges.txt";

/// Generate images from text prompts using Stable Diffusion 1.5.
#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Text prompt describing the image to generate.
    prompt: String,

    /// Negative prompt (things to avoid in the image).
    #[arg(short, long, default_value = "")]
    negative: String,

    /// Number of sampling steps.
    #[arg(short, long, default_value_t = 20)]
    steps: usize,

    /// Classifier-free guidance scale.
    #[arg(short, long, default_value_t = 7.5)]
    cfg: f32,

    /// Output file path.
    #[arg(short, long, default_value = "output.png")]
    output: String,

    /// Random seed for reproducibility.
    #[arg(long)]
    seed: Option<u64>,

    /// Sampling algorithm.
    #[arg(long, default_value = "euler")]
    sampler: SamplerKind,

    /// Sigma schedule (normal = linear timestep spacing, karras = exponential).
    #[arg(long, default_value = "normal")]
    scheduler: SchedulerKind,

    /// Enable per-layer timing (adds GPU sync points, slower).
    /// Only supported with euler sampler; ignored otherwise.
    #[arg(long)]
    profile: bool,

    /// ControlNet model path (repeatable; starts a new ControlNet group).
    #[arg(long = "cn-model")]
    cn_model: Vec<String>,

    /// Conditioning image for the corresponding ControlNet.
    #[arg(long = "cn-image")]
    cn_image: Vec<String>,

    /// Weight for the corresponding ControlNet (default: 1.0).
    #[arg(long = "cn-weight")]
    cn_weight: Vec<f32>,

    /// Normalized start of ControlNet activation, 0.0=first step (default: 0.0).
    #[arg(long = "cn-start")]
    cn_start: Vec<f32>,

    /// Normalized end of ControlNet activation, 1.0=last step (default: 1.0).
    #[arg(long = "cn-end")]
    cn_end: Vec<f32>,

    /// LoRA file path (repeatable).
    #[arg(long = "lora")]
    lora: Vec<String>,

    /// LoRA strength for UNet (default: 1.0, one per --lora).
    #[arg(long = "lora-strength")]
    lora_strength: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber (reads RUST_LOG env var)
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    println!("Hearth SD 1.5 Generator");

    // Check required model files exist before doing expensive GPU work
    let checkpoint_path = Path::new(CHECKPOINT_PATH);
    let vocab_path = Path::new(VOCAB_PATH);
    let merges_path = Path::new(MERGES_PATH);

    let file_check = startup::check_model_files(&[checkpoint_path, vocab_path, merges_path]);
    if !file_check.all_found() {
        eprintln!("{}", file_check.format_error());
        process::exit(1);
    }

    // Validate ControlNet args: each --cn-model must have a matching --cn-image
    if args.cn_model.len() != args.cn_image.len() {
        eprintln!(
            "Error: {} --cn-model but {} --cn-image; each ControlNet needs both",
            args.cn_model.len(),
            args.cn_image.len()
        );
        process::exit(1);
    }

    let attention_backend = if std::env::var("INT8_CMMA").is_ok_and(|v| v == "1") {
        "SageAttention (INT8 CMMA)"
    } else {
        "FlashAttention (f16 CMMA)"
    };
    println!("Attention: {attention_backend}");
    println!("Prompt: {}", args.prompt);
    println!("Negative: {}", args.negative);
    println!(
        "Steps: {}, CFG: {}, Sampler: {}, Scheduler: {}",
        args.steps, args.cfg, args.sampler, args.scheduler
    );
    println!();

    let device = Default::default();

    // Set random seed if provided
    if let Some(seed) = args.seed {
        <Backend as burn::tensor::backend::Backend>::seed(&device, seed);
        println!("Seed: {}", seed);
    }

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = ClipTokenizer::from_files(vocab_path, merges_path)?;

    // Load checkpoint
    println!("Loading checkpoint...");
    let start = Instant::now();
    let file = SafeTensorsFile::open(checkpoint_path)?;
    let tensors = file.tensors()?;
    println!("  Checkpoint opened in {:?}", start.elapsed());

    // Load LoRA files (if any)
    let lora_files: Vec<_> = args
        .lora
        .iter()
        .map(LoraFile::open)
        .collect::<Result<_, _>>()?;

    // Load models
    // Load CLIP, encode prompts, then drop CLIP to free GPU memory
    println!("Loading CLIP...");
    let start = Instant::now();
    let mut clip = Sd15ClipTextEncoder::load(&tensors, &device)?;
    println!("  CLIP loaded in {:?}", start.elapsed());

    // Apply LoRA to CLIP
    for (i, lora_file) in lora_files.iter().enumerate() {
        let strength = args.lora_strength.get(i).copied().unwrap_or(1.0);
        let lora_tensors = lora_file.tensors()?;
        let start = Instant::now();
        let n = clip.apply_lora("text_model", "lora_te", &lora_tensors, strength, &device)?;
        println!(
            "  LoRA {i}: applied {n} deltas to CLIP (strength {strength}) in {:?}",
            start.elapsed()
        );
    }

    println!("Encoding prompts...");
    let start = Instant::now();
    let tokens = tokenizer.encode(&args.prompt, &device)?;
    let neg_tokens = tokenizer.encode(&args.negative, &device)?;

    let positive_cond = Sd15Conditioning::new(clip.forward(tokens));
    let negative_cond = Sd15Conditioning::new(clip.forward(neg_tokens));
    println!("  Prompts encoded in {:?}", start.elapsed());
    drop(clip); // Free CLIP memory before loading UNet

    println!("Loading UNet...");
    let start = Instant::now();
    let mut unet = Sd15Unet2D::load(&tensors, &device)?;
    println!("  UNet loaded in {:?}", start.elapsed());

    // Apply LoRA to UNet
    for (i, lora_file) in lora_files.iter().enumerate() {
        let strength = args.lora_strength.get(i).copied().unwrap_or(1.0);
        let lora_tensors = lora_file.tensors()?;
        let start = Instant::now();
        let n = unet.apply_lora("lora_unet", &lora_tensors, strength, &device)?;
        println!(
            "  LoRA {i}: applied {n} deltas to UNet (strength {strength}) in {:?}",
            start.elapsed()
        );
    }

    // Load ControlNets (if any)
    let loaded_controlnets = load_controlnets(&args, &device)?;
    let controlnet_units = build_controlnet_units(&loaded_controlnets);
    let has_controlnets = !controlnet_units.is_empty();

    println!("Loading scheduler...");
    let scheduler = NoiseSchedule::load(&tensors, &device)?;

    // Generate schedule with pre-computed timesteps
    let schedule = args.scheduler.schedule(&scheduler, args.steps);

    // Create noise scaled by sigma_max
    println!("Creating initial noise...");
    let sigma_max = schedule.sigmas[0];
    let latent: Tensor<Backend, 4> = Tensor::random(
        [1, 4, 64, 64],
        Distribution::Normal(0.0, sigma_max as f64),
        &device,
    );
    println!(
        "Sigmas: {} steps, range [{:.3}, {:.3}]",
        schedule.sigmas.len() - 1,
        schedule.sigmas[0],
        schedule.sigmas[schedule.sigmas.len() - 2]
    );
    println!();

    // Sample using optimized batched inference
    println!("Sampling ({} steps)...", args.steps);
    let start = Instant::now();
    let last_report = Cell::new(Instant::now());
    let progress_cb = |current: usize, total: usize| {
        let now = Instant::now();
        if current == total || now.duration_since(last_report.get()).as_secs() >= 1 {
            last_report.set(now);
            eprintln!(
                "  Step {current}/{total} ({:.1}%)",
                current as f64 / total as f64 * 100.0
            );
        }
    };
    let denoised = if has_controlnets {
        // ControlNet path: uses predict_noise_cfg_controlled (unbatched CFG)
        match args.sampler {
            SamplerKind::Euler => {
                let sampler = EulerSampler::new(scheduler);
                sampler.sample_with_schedule_controlled(
                    &unet,
                    &controlnet_units,
                    latent,
                    &positive_cond,
                    &negative_cond,
                    &schedule,
                    args.cfg,
                    Some(&progress_cb),
                )
            }
            SamplerKind::EulerA => {
                let sampler = EulerASampler::new(scheduler);
                sampler.sample_with_schedule_controlled(
                    &unet,
                    &controlnet_units,
                    latent,
                    &positive_cond,
                    &negative_cond,
                    &schedule,
                    args.cfg,
                    Some(&progress_cb),
                )
            }
            SamplerKind::DpmPp2m => {
                let sampler = DpmPp2mSampler::new(scheduler);
                sampler.sample_with_schedule_controlled(
                    &unet,
                    &controlnet_units,
                    latent,
                    &positive_cond,
                    &negative_cond,
                    &schedule,
                    args.cfg,
                    Some(&progress_cb),
                )
            }
            SamplerKind::DpmPpSde => {
                let sampler = DpmPpSdeSampler::new(scheduler);
                sampler.sample_with_schedule_controlled(
                    &unet,
                    &controlnet_units,
                    latent,
                    &positive_cond,
                    &negative_cond,
                    &schedule,
                    args.cfg,
                    Some(&progress_cb),
                )
            }
        }
    } else {
        // Standard path: batched CFG without ControlNets
        match args.sampler {
            SamplerKind::Euler => {
                let sampler = EulerSampler::new(scheduler);
                if args.profile {
                    sampler.sample_with_schedule_timed(
                        &unet,
                        latent,
                        &positive_cond,
                        &negative_cond,
                        &schedule,
                        args.cfg,
                    )
                } else {
                    sampler.sample_with_schedule(
                        &unet,
                        latent,
                        &positive_cond,
                        &negative_cond,
                        &schedule,
                        args.cfg,
                        Some(&progress_cb),
                    )
                }
            }
            SamplerKind::EulerA => {
                let sampler = EulerASampler::new(scheduler);
                sampler.sample_with_schedule(
                    &unet,
                    latent,
                    &positive_cond,
                    &negative_cond,
                    &schedule,
                    args.cfg,
                    Some(&progress_cb),
                )
            }
            SamplerKind::DpmPp2m => {
                let sampler = DpmPp2mSampler::new(scheduler);
                sampler.sample_with_schedule(
                    &unet,
                    latent,
                    &positive_cond,
                    &negative_cond,
                    &schedule,
                    args.cfg,
                    Some(&progress_cb),
                )
            }
            SamplerKind::DpmPpSde => {
                let sampler = DpmPpSdeSampler::new(scheduler);
                sampler.sample_with_schedule(
                    &unet,
                    latent,
                    &positive_cond,
                    &negative_cond,
                    &schedule,
                    args.cfg,
                    Some(&progress_cb),
                )
            }
        }
    };
    println!("  Sampling completed in {:?}", start.elapsed());
    drop(unet); // Free UNet memory before loading VAE
    drop(controlnet_units);
    drop(loaded_controlnets);

    // Load VAE only when needed (after sampling completes)
    println!("Loading VAE...");
    let start = Instant::now();
    let vae = Sd15VaeDecoder::load(&tensors, Some("first_stage_model"), &device)?;
    println!("  VAE loaded in {:?}", start.elapsed());

    // Decode
    println!("Decoding latent to image...");
    let start = Instant::now();
    let image = vae.forward(denoised);
    println!("  Decoding completed in {:?}", start.elapsed());

    // Save image
    println!("Saving to {}...", args.output);
    save_image(image, &args.output)?;
    println!("Done!");

    Ok(())
}

/// Loaded ControlNet data (model + hint + settings), ready to be converted
/// into borrowed [`ControlNetUnit`]s for sampling.
struct LoadedControlNet {
    /// The loaded ControlNet model.
    model: Sd15ControlNet,
    /// Pre-processed hint image `[1, 3, H, W]`.
    hint: Tensor<Backend, 4>,
    /// Strength multiplier.
    weight: f32,
    /// Normalized activation start.
    start: f32,
    /// Normalized activation end.
    end: f32,
}

/// Load ControlNet models and hint images from CLI args.
///
/// Returns an empty Vec if no ControlNets were specified.
fn load_controlnets(
    args: &Args,
    device: &burn::tensor::Device<Backend>,
) -> Result<Vec<LoadedControlNet>, Box<dyn std::error::Error>> {
    let mut loaded = Vec::new();

    for i in 0..args.cn_model.len() {
        let model_path = &args.cn_model[i];
        let image_path = &args.cn_image[i];
        let weight = args.cn_weight.get(i).copied().unwrap_or(1.0);
        let start = args.cn_start.get(i).copied().unwrap_or(0.0);
        let end = args.cn_end.get(i).copied().unwrap_or(1.0);

        println!("Loading ControlNet {i}: {model_path}");
        println!("  hint: {image_path}, weight: {weight}, range: [{start}, {end}]");

        let cn_start = Instant::now();
        let cn_file = SafeTensorsFile::open(Path::new(model_path))?;
        let cn_tensors = cn_file.tensors()?;
        let model = Sd15ControlNet::load(&cn_tensors, 3, device)?;
        println!("  ControlNet loaded in {:?}", cn_start.elapsed());

        let hint = load_hint_image(image_path, 512, 512, device)?;

        loaded.push(LoadedControlNet {
            model,
            hint,
            weight,
            start,
            end,
        });
    }

    Ok(loaded)
}

/// Build borrowed [`ControlNetUnit`]s from loaded ControlNet data.
fn build_controlnet_units<'a>(
    loaded: &'a [LoadedControlNet],
) -> Vec<ControlNetUnit<'a, hearth::unet::Sd15Unet>> {
    loaded
        .iter()
        .map(|cn| ControlNetUnit {
            model: &cn.model,
            hint: cn.hint.clone(),
            weight: cn.weight,
            start: cn.start,
            end: cn.end,
        })
        .collect()
}

/// Load a conditioning image as a `[1, 3, H, W]` tensor in [0, 1].
fn load_hint_image(
    path: &str,
    target_w: u32,
    target_h: u32,
    device: &burn::tensor::Device<Backend>,
) -> Result<Tensor<Backend, 4>, Box<dyn std::error::Error>> {
    let img = image::open(path)?.resize_exact(target_w, target_h, image::imageops::Lanczos3);
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);

    // Convert HWC u8 → CHW f32 in [0, 1]
    let raw = rgb.as_raw();
    let mut data = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            data[y * w + x] = raw[idx] as f32 / 255.0;
            data[h * w + y * w + x] = raw[idx + 1] as f32 / 255.0;
            data[2 * h * w + y * w + x] = raw[idx + 2] as f32 / 255.0;
        }
    }

    let td = burn::tensor::TensorData::new(data, [1, 3, h, w])
        .convert::<<Backend as burn::tensor::backend::Backend>::FloatElem>();
    Ok(Tensor::from_data(td, device))
}

/// Save a VAE output tensor as a PNG image.
///
/// Input: [1, 3, H, W] tensor in [-1, 1] range
/// Output: RGB PNG file
fn save_image(tensor: Tensor<Backend, 4>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let [batch, channels, height, width] = tensor.shape().dims();
    assert_eq!(batch, 1, "Only single image supported");
    assert_eq!(channels, 3, "Expected 3 channels (RGB)");

    // Convert to f32 vec
    let data: Vec<f32> = tensor.into_data().convert::<f32>().to_vec()?;

    // Convert [B, C, H, W] in [-1, 1] to [H, W, C] in [0, 255]
    // Iterate sequentially through memory for better cache locality
    let hw = height * width;
    let mut rgb = vec![0u8; hw * 3];

    #[inline]
    fn to_u8(val: f32) -> u8 {
        // Clamp to [-1, 1], scale to [0, 255]
        ((val.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8
    }

    for (i, pixel) in rgb.chunks_exact_mut(3).enumerate() {
        pixel[0] = to_u8(data[i]);
        pixel[1] = to_u8(data[hw + i]);
        pixel[2] = to_u8(data[2 * hw + i]);
    }

    // Create and save image
    let img = image::RgbImage::from_raw(width as u32, height as u32, rgb)
        .ok_or("Failed to create image buffer")?;
    img.save(path)?;

    Ok(())
}
