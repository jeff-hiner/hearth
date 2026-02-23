//! `POST /sdapi/v1/txt2img` — text-to-image generation.

use super::types::{Txt2ImgInfo, Txt2ImgRequest, Txt2ImgResponse};
use crate::{
    node::{
        CheckpointLoaderSimple, ClipTextEncode, EmptyLatentImage, KSampler, SaveImage, VaeDecode,
        executor::{ExecutionGraph, Executor},
        value::NodeValue,
    },
    sampling::{SamplerKind, SchedulerKind},
    server::{AppState, ProgressInfo, error::ApiError},
    types::Backend,
};
use axum::{Json, extract::State};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use burn::tensor::Tensor;
use image::{ImageEncoder, codecs::png::PngEncoder};
use std::{sync::Arc, time::SystemTime};

/// Handle txt2img requests by building and executing a node graph.
pub(super) async fn txt2img(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Txt2ImgRequest>,
) -> Result<Json<Txt2ImgResponse>, ApiError> {
    tracing::info!(
        prompt = %req.prompt,
        width = req.width,
        height = req.height,
        steps = req.steps,
        "txt2img request"
    );

    // Determine checkpoint
    let ckpt_name = req
        .override_settings
        .as_ref()
        .and_then(|s| s.sd_model_checkpoint.clone())
        .or_else(|| {
            let opts = state.options.blocking_lock();
            if opts.sd_model_checkpoint.is_empty() {
                None
            } else {
                Some(opts.sd_model_checkpoint.clone())
            }
        })
        .ok_or_else(|| {
            ApiError::BadRequest(
                "no checkpoint specified: set sd_model_checkpoint in options or override_settings"
                    .to_string(),
            )
        })?;

    // Parse sampler/scheduler
    let sampler_kind: SamplerKind = req
        .sampler_name
        .parse()
        .map_err(|e: String| ApiError::BadRequest(e))?;
    let scheduler_kind: SchedulerKind = req
        .scheduler
        .parse()
        .map_err(|e: String| ApiError::BadRequest(e))?;

    // Resolve seed
    let seed = if req.seed < 0 {
        rand_seed()
    } else {
        req.seed as u64
    };

    // Build the execution graph:
    // CheckpointLoader → CLIPTextEncode(+) → CLIPTextEncode(-) → EmptyLatent → KSampler → VAEDecode → SaveImage
    let mut graph = ExecutionGraph::new();

    let loader_id = graph.add_node(Box::new(CheckpointLoaderSimple::new(
        ckpt_name.clone().into(),
    )));
    let clip_pos_id = graph.add_node(Box::new(ClipTextEncode::new(req.prompt.clone())));
    let clip_neg_id = graph.add_node(Box::new(ClipTextEncode::new(req.negative_prompt.clone())));
    let empty_latent_id = graph.add_node(Box::new(EmptyLatentImage::new(
        req.width as usize,
        req.height as usize,
        req.batch_size as usize,
    )));
    let ksampler_id = graph.add_node(Box::new(KSampler::new(
        seed,
        req.steps as usize,
        req.cfg_scale,
        sampler_kind,
        scheduler_kind,
        req.denoising_strength,
    )));
    let vae_decode_id = graph.add_node(Box::new(VaeDecode));
    let save_image_id = graph.add_node(Box::new(SaveImage::new("hearth".to_string())));

    // Wire edges:
    // CheckpointLoader outputs: 0=MODEL, 1=CLIP, 2=VAE
    // CLIPTextEncode input: 0=clip
    graph.add_edge(loader_id, 1, clip_pos_id, 0); // CLIP → positive encoder
    graph.add_edge(loader_id, 1, clip_neg_id, 0); // CLIP → negative encoder

    // KSampler inputs: 0=model, 1=positive, 2=negative, 3=latent_image
    graph.add_edge(loader_id, 0, ksampler_id, 0); // MODEL → KSampler
    graph.add_edge(clip_pos_id, 0, ksampler_id, 1); // pos conditioning → KSampler
    graph.add_edge(clip_neg_id, 0, ksampler_id, 2); // neg conditioning → KSampler
    graph.add_edge(empty_latent_id, 0, ksampler_id, 3); // latent → KSampler

    // VAEDecode inputs: 0=samples, 1=vae
    graph.add_edge(ksampler_id, 0, vae_decode_id, 0); // denoised latent → VAEDecode
    graph.add_edge(loader_id, 2, vae_decode_id, 1); // VAE → VAEDecode

    // SaveImage input: 0=images
    graph.add_edge(vae_decode_id, 0, save_image_id, 0);

    // Signal that generation is starting
    let _ = state.progress_tx.send(ProgressInfo {
        progress: 0.0,
        eta_relative: 0.0,
        current_step: 0,
        total_steps: req.steps as usize,
        active: true,
    });

    // Execute the graph (holds the mutex for the duration)
    let outputs = {
        let mut ctx = state.ctx.lock().await;
        let progress_tx = state.progress_tx.clone();
        ctx.set_progress(Box::new(move |current, total| {
            let _ = progress_tx.send(ProgressInfo {
                progress: current as f32 / total as f32,
                eta_relative: 0.0,
                current_step: current,
                total_steps: total,
                active: true,
            });
        }));
        Executor::run(&graph, &mut ctx)?
    };

    // Signal completion
    let _ = state.progress_tx.send(ProgressInfo::default());

    // Extract the decoded image from VAEDecode output
    let image = outputs
        .get(&(vae_decode_id, 0))
        .ok_or_else(|| ApiError::Internal("VAEDecode produced no output".to_string()))?;

    let images_b64 = match image {
        NodeValue::Image(img) => encode_images_base64(img)?,
        other => {
            return Err(ApiError::Internal(format!(
                "expected IMAGE, got {}",
                other.type_name()
            )));
        }
    };

    let info = Txt2ImgInfo {
        prompt: req.prompt.clone(),
        negative_prompt: req.negative_prompt.clone(),
        seed,
        steps: req.steps,
        cfg_scale: req.cfg_scale,
        sampler_name: req.sampler_name.clone(),
        scheduler: req.scheduler.clone(),
        width: req.width,
        height: req.height,
        sd_model_checkpoint: ckpt_name,
    };

    Ok(Json(Txt2ImgResponse {
        images: images_b64,
        info: serde_json::to_string(&info).expect("Txt2ImgInfo is always serializable"),
        parameters: info,
    }))
}

/// Encode image tensor as base64 PNG strings.
fn encode_images_base64(image: &crate::types::Image) -> Result<Vec<String>, ApiError> {
    let tensor: &Tensor<Backend, 4> = &image.data;
    let [batch, channels, height, width] = tensor.shape().dims();

    let mut result = Vec::with_capacity(batch);

    for b in 0..batch {
        let single = tensor
            .clone()
            .slice([b..b + 1, 0..channels, 0..height, 0..width]);
        let data: Vec<f32> = single.into_data().convert::<f32>().to_vec().unwrap();

        let hw = height * width;
        let mut rgb = vec![0u8; hw * 3];

        for (i, pixel) in rgb.chunks_exact_mut(3).enumerate() {
            pixel[0] = to_u8(data[i]);
            pixel[1] = to_u8(data[hw + i]);
            pixel[2] = to_u8(data[2 * hw + i]);
        }

        let mut png_buf = Vec::new();
        let encoder = PngEncoder::new(&mut png_buf);
        encoder
            .write_image(
                &rgb,
                width as u32,
                height as u32,
                image::ColorType::Rgb8.into(),
            )
            .map_err(|e| ApiError::Internal(format!("PNG encode failed: {e}")))?;

        result.push(BASE64.encode(&png_buf));
    }

    Ok(result)
}

/// Clamp to [-1, 1] and scale to [0, 255].
fn to_u8(val: f32) -> u8 {
    ((val.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8
}

/// Generate a random seed.
fn rand_seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42)
}
