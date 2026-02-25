//! Minimal image generation demo for SDXL.
//!
//! Generates a 1024x1024 image from a text prompt using Stable Diffusion XL.
//!
//! Usage:
//!   cargo run --release --example generate_xl -- "a cat sitting on a couch"
//!   cargo run --release --example generate_xl -- "a cat" -o cat.png --steps 30
//!
//! Requires:
//!   - models/checkpoints/sd_xl_base_1.0.safetensors
//!   - models/clip/vocab.json
//!   - models/clip/merges.txt

use burn::tensor::{Device, Distribution, Tensor, TensorData, backend::Backend as _};
use clap::Parser;
use hearth::{
    clip::{ClipTokenizer, OpenClipTextEncoder, SdxlClipLTextEncoder, SdxlConditioning},
    controlnet::{ControlNetUnit, SdxlControlNet},
    model_loader::{self, SafeTensorsFile},
    sampling::{
        DpmPp2mSampler, DpmPpSdeSampler, EulerASampler, EulerSampler, NoiseSchedule, SamplerKind,
        SchedulerKind,
    },
    startup,
    types::Backend,
    unet::{SdxlUnet, SdxlUnet2D},
    vae::SdxlVaeDecoder,
};
use std::{
    cell::Cell,
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
    Tokenizer(#[from] hearth::clip::TokenizerError),
    #[error(transparent)]
    Image(#[from] image::ImageError),
    #[error(transparent)]
    Data(#[from] burn::tensor::DataError),
    #[error("failed to create image buffer")]
    ImageBuffer,
}

/// Path to the SDXL checkpoint file.
const CHECKPOINT_PATH: &str = "models/checkpoints/sd_xl_base_1.0.safetensors";
/// Path to the fp16-safe SDXL VAE (stock SDXL VAE overflows in fp16).
const VAE_PATH: &str = "models/vae/sdxl-vae-fp16-fix.safetensors";
/// Path to the CLIP tokenizer vocabulary.
const VOCAB_PATH: &str = "models/clip/vocab.json";
/// Path to the CLIP tokenizer merges.
const MERGES_PATH: &str = "models/clip/merges.txt";

/// Generate images from text prompts using Stable Diffusion XL.
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
    #[arg(short, long, default_value = "output_xl.png")]
    output: PathBuf,

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
    cn_model: Vec<PathBuf>,

    /// Conditioning image for the corresponding ControlNet.
    #[arg(long = "cn-image")]
    cn_image: Vec<PathBuf>,

    /// Weight for the corresponding ControlNet (default: 1.0).
    #[arg(long = "cn-weight")]
    cn_weight: Vec<f32>,

    /// Normalized start of ControlNet activation, 0.0=first step (default: 0.0).
    #[arg(long = "cn-start")]
    cn_start: Vec<f32>,

    /// Normalized end of ControlNet activation, 1.0=last step (default: 1.0).
    #[arg(long = "cn-end")]
    cn_end: Vec<f32>,
}

fn main() -> Result<(), Error> {
    // Initialize tracing subscriber (reads RUST_LOG env var)
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    println!("Hearth SDXL Generator");

    // Check required model files exist before doing expensive GPU work
    let checkpoint_path = Path::new(CHECKPOINT_PATH);
    let vae_path = Path::new(VAE_PATH);
    let vocab_path = Path::new(VOCAB_PATH);
    let merges_path = Path::new(MERGES_PATH);

    let file_check =
        startup::check_model_files(&[checkpoint_path, vae_path, vocab_path, merges_path]);
    if !file_check.all_found() {
        eprintln!("{}", file_check.format_error());
        process::exit(1);
    }

    // Validate ControlNet args
    if args.cn_model.len() != args.cn_image.len() {
        eprintln!(
            "Error: {} --cn-model but {} --cn-image; each ControlNet needs both",
            args.cn_model.len(),
            args.cn_image.len()
        );
        process::exit(1);
    }

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
        Backend::seed(&device, seed);
        println!("Seed: {seed}");
    }

    // Load tokenizer (shared between both CLIP encoders)
    println!("Loading tokenizer...");
    let tokenizer = ClipTokenizer::from_files(vocab_path, merges_path)?;

    // Load checkpoint
    println!("Loading checkpoint...");
    let start = Instant::now();
    let file = SafeTensorsFile::open(checkpoint_path)?;
    let tensors = file.tensors()?;
    println!("  Checkpoint opened in {:?}", start.elapsed());

    // Load both CLIP encoders, encode prompts, then drop to free memory
    println!("Loading CLIP-L...");
    let start = Instant::now();
    let clip_l = SdxlClipLTextEncoder::load_with_prefix(
        &tensors,
        model_loader::prefix::SDXL_CLIP_L,
        &device,
    )?;
    println!("  CLIP-L loaded in {:?}", start.elapsed());

    println!("Loading OpenCLIP-G...");
    let start = Instant::now();
    let clip_g = OpenClipTextEncoder::load(&tensors, model_loader::prefix::SDXL_CLIP_G, &device)?;
    println!("  OpenCLIP-G loaded in {:?}", start.elapsed());

    // Encode positive prompt
    println!("Encoding prompts...");
    let start = Instant::now();
    let pos_tokens_l = tokenizer.encode(&args.prompt, &device)?;
    let neg_tokens_l = tokenizer.encode(&args.negative, &device)?;
    // OpenCLIP-G uses 0 for padding instead of EOS
    let pos_tokens_g = tokenizer.encode_open_clip(&args.prompt, &device)?;
    let neg_tokens_g = tokenizer.encode_open_clip(&args.negative, &device)?;

    // CLIP-L: extract penultimate layer (layer 10 of 12, i.e. index 10)
    let pos_clip_l_hidden = clip_l.forward_hidden_layer(pos_tokens_l, Some(10));
    let neg_clip_l_hidden = clip_l.forward_hidden_layer(neg_tokens_l, Some(10));

    // OpenCLIP-G: get hidden states and pooled output
    let (pos_clip_g_hidden, pos_pooled) = clip_g.forward(pos_tokens_g);
    let (neg_clip_g_hidden, neg_pooled) = clip_g.forward(neg_tokens_g);

    // Build SDXL conditioning with default metadata (1024x1024, no crop)
    let positive_cond = SdxlConditioning::new(
        pos_clip_l_hidden,
        pos_clip_g_hidden,
        pos_pooled,
        (1024.0, 1024.0), // original_size
        (0.0, 0.0),       // crop_coords
        (1024.0, 1024.0), // target_size
        &device,
    );
    let negative_cond = SdxlConditioning::new(
        neg_clip_l_hidden,
        neg_clip_g_hidden,
        neg_pooled,
        (1024.0, 1024.0),
        (0.0, 0.0),
        (1024.0, 1024.0),
        &device,
    );
    println!("  Prompts encoded in {:?}", start.elapsed());

    // Free CLIP memory before loading UNet
    drop(clip_l);
    drop(clip_g);

    println!("Loading UNet...");
    let start = Instant::now();
    let unet = SdxlUnet2D::load(&tensors, &device)?;
    println!("  UNet loaded in {:?}", start.elapsed());

    // Load ControlNets (if any)
    let loaded_controlnets = load_controlnets(&args, &device)?;
    let controlnet_units = build_controlnet_units(&loaded_controlnets);
    let has_controlnets = !controlnet_units.is_empty();

    // SDXL checkpoints don't include pre-computed alphas_cumprod, so we
    // construct the standard linear beta schedule directly. These are the
    // same DDPM parameters that SD 1.5 and SDXL both use.
    println!("Creating scheduler...");
    let scheduler = NoiseSchedule::linear(1000, 0.00085, 0.012);

    // Generate schedule with pre-computed timesteps
    let schedule = args.scheduler.schedule(&scheduler, args.steps);

    // Create noise scaled by sigma_max
    // SDXL generates 1024x1024 images -> 128x128 latent (8x downscale)
    println!("Creating initial noise...");
    let sigma_max = schedule.sigmas[0];
    let latent: Tensor<Backend, 4> = Tensor::random(
        [1, 4, 128, 128],
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
    let progress_cb = |current: usize, total: usize| -> bool {
        let now = Instant::now();
        if current == total || now.duration_since(last_report.get()).as_secs() >= 1 {
            last_report.set(now);
            eprintln!(
                "  Step {current}/{total} ({:.1}%)",
                current as f64 / total as f64 * 100.0
            );
        }
        true // always continue (no cancellation in CLI)
    };
    let denoised = if has_controlnets {
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

    // Free all sampling-related GPU memory before loading VAE.
    // SDXL is large enough that we need to reclaim UNet/conditioning buffers.
    drop(unet);
    drop(controlnet_units);
    drop(loaded_controlnets);
    drop(positive_cond);
    drop(negative_cond);
    let _ = Backend::sync(&device);
    Backend::memory_cleanup(&device);
    println!("  GPU memory cleaned up");

    // Load fp16-safe VAE (stock SDXL VAE weights overflow in fp16).
    println!("Loading VAE (fp16-fix)...");
    let start = Instant::now();
    let vae_file = SafeTensorsFile::open(vae_path)?;
    let vae_tensors = vae_file.tensors()?;
    let vae = SdxlVaeDecoder::load(&vae_tensors, None, &device)?;
    println!("  VAE loaded in {:?}", start.elapsed());

    // Decode with tiled VAE to avoid OOM on the attention matrix.
    // 64 latent-pixel tiles (512px) with 8-pixel overlap keeps the mid-block
    // attention at [1, 4096, 4096] = 67 MB, well within GPU buffer limits.
    println!("Decoding latent to image (tiled)...");
    let start = Instant::now();
    let image = vae.forward_tiled(denoised, 64, 8);
    println!("  Decoding completed in {:?}", start.elapsed());

    // Save image
    println!("Saving to {}...", args.output.display());
    save_image(image, &args.output)?;
    println!("Done!");

    Ok(())
}

/// Loaded ControlNet data (model + hint + settings), ready to be converted
/// into borrowed [`ControlNetUnit`]s for sampling.
struct LoadedControlNet {
    /// The loaded ControlNet model.
    model: SdxlControlNet,
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
fn load_controlnets(args: &Args, device: &Device<Backend>) -> Result<Vec<LoadedControlNet>, Error> {
    let mut loaded = Vec::new();

    for i in 0..args.cn_model.len() {
        let model_path = &args.cn_model[i];
        let image_path = &args.cn_image[i];
        let weight = args.cn_weight.get(i).copied().unwrap_or(1.0);
        let start = args.cn_start.get(i).copied().unwrap_or(0.0);
        let end = args.cn_end.get(i).copied().unwrap_or(1.0);

        println!("Loading ControlNet {i}: {}", model_path.display());
        println!(
            "  hint: {}, weight: {weight}, range: [{start}, {end}]",
            image_path.display()
        );

        let cn_start = Instant::now();
        let cn_file = SafeTensorsFile::open(model_path)?;
        let cn_tensors = cn_file.tensors()?;
        let model = SdxlControlNet::load(&cn_tensors, 3, device)?;
        println!("  ControlNet loaded in {:?}", cn_start.elapsed());

        let hint = load_hint_image(image_path, 1024, 1024, device)?;

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
fn build_controlnet_units(loaded: &[LoadedControlNet]) -> Vec<ControlNetUnit<'_, SdxlUnet>> {
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
    path: &Path,
    target_w: u32,
    target_h: u32,
    device: &Device<Backend>,
) -> Result<Tensor<Backend, 4>, Error> {
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

    let td = TensorData::new(data, [1, 3, h, w])
        .convert::<<Backend as burn::tensor::backend::Backend>::FloatElem>();
    Ok(Tensor::from_data(td, device))
}

/// Save a VAE output tensor as a PNG image.
///
/// Input: [1, 3, H, W] tensor in [-1, 1] range
/// Output: RGB PNG file
fn save_image(tensor: Tensor<Backend, 4>, path: &Path) -> Result<(), Error> {
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
    let img =
        image::RgbImage::from_raw(width as u32, height as u32, rgb).ok_or(Error::ImageBuffer)?;
    img.save(path)?;

    Ok(())
}
