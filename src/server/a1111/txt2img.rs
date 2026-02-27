//! `POST /sdapi/v1/txt2img` — text-to-image generation.

use super::{
    types::{Txt2ImgInfo, Txt2ImgRequest, Txt2ImgResponse},
    util::{apply_controlnet_units, encode_images_base64, rand_seed},
};
use crate::{
    node::{
        CheckpointLoaderSimple, ClipTextEncode, EmptyLatentImage, KSampler, SaveImage, VaeDecode,
        executor::{ExecutionGraph, Executor},
        value::NodeValue,
    },
    sampling::{SamplerKind, SchedulerKind},
    server::{AppState, ProgressInfo, error::ApiError},
};
use axum::{Json, extract::State};
use std::sync::Arc;

/// Handle txt2img requests by building and executing a node graph.
pub(super) async fn txt2img(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Txt2ImgRequest>,
) -> Result<Json<Txt2ImgResponse>, ApiError> {
    let cn_unit_count = req
        .alwayson_scripts
        .as_ref()
        .and_then(|s| s.controlnet.as_ref())
        .map_or(0, |cn| cn.args.iter().filter(|u| u.enabled).count());
    tracing::info!(
        prompt = %req.prompt,
        width = req.width,
        height = req.height,
        steps = req.steps,
        sampler = %req.sampler_name,
        scheduler = %req.scheduler,
        controlnet_units = cn_unit_count,
        has_alwayson_scripts = req.alwayson_scripts.is_some(),
        "txt2img request"
    );

    if req.tiling {
        return Err(ApiError::BadRequest(
            "Tiling mode is not yet supported by Hearth. Set tiling to false.".to_string(),
        ));
    }

    // Determine checkpoint
    let ckpt_name = match req
        .override_settings
        .as_ref()
        .and_then(|s| s.sd_model_checkpoint.clone())
    {
        Some(name) => name,
        None => {
            let opts = state.options.lock().await;
            if opts.sd_model_checkpoint.is_empty() {
                return Err(ApiError::BadRequest(
                    "no checkpoint specified: set sd_model_checkpoint in options or override_settings"
                        .to_string(),
                ));
            }
            opts.sd_model_checkpoint.clone()
        }
    };

    // Parse sampler/scheduler
    let sampler_kind: SamplerKind = req
        .sampler_name
        .parse()
        .map_err(|e: strum::ParseError| ApiError::BadRequest(e.to_string()))?;
    let scheduler_kind: SchedulerKind = req
        .scheduler
        .parse()
        .map_err(|e: strum::ParseError| ApiError::BadRequest(e.to_string()))?;

    // Resolve seed
    let seed = if req.seed < 0 {
        rand_seed()
    } else {
        req.seed as u64
    };

    let n_iter = req.n_iter.max(1) as usize;
    let mut all_images_b64 = Vec::new();

    for iter_i in 0..n_iter {
        let iter_seed = seed + iter_i as u64;

        // Build the execution graph:
        // CheckpointLoader → CLIPTextEncode(+/-) → KSampler → VAEDecode → SaveImage
        let mut graph = ExecutionGraph::new();

        let loader_id = graph.add_node(Box::new(CheckpointLoaderSimple::new(
            ckpt_name.clone().into(),
        )));
        let clip_pos_id = graph.add_node(Box::new(ClipTextEncode::new(req.prompt.clone())));
        let clip_neg_id =
            graph.add_node(Box::new(ClipTextEncode::new(req.negative_prompt.clone())));
        let empty_latent_id = graph.add_node(Box::new(EmptyLatentImage::new(
            req.width as usize,
            req.height as usize,
            req.batch_size as usize,
        )));
        let ksampler_id = graph.add_node(Box::new(KSampler::new(
            iter_seed,
            req.steps as usize,
            req.cfg_scale,
            sampler_kind,
            scheduler_kind,
            1.0, // txt2img always fully denoises from pure noise
        )));
        let vae_decode_id = graph.add_node(Box::new(VaeDecode));
        let save_image_id = graph.add_node(Box::new(SaveImage::new("hearth".to_string())));

        // Wire edges
        graph.add_edge(loader_id, 1, clip_pos_id, 0);
        graph.add_edge(loader_id, 1, clip_neg_id, 0);

        let (pos_id, pos_out) =
            apply_controlnet_units(&mut graph, &req.alwayson_scripts, clip_pos_id, 0)?;

        graph.add_edge(loader_id, 0, ksampler_id, 0);
        graph.add_edge(pos_id, pos_out, ksampler_id, 1);
        graph.add_edge(clip_neg_id, 0, ksampler_id, 2);
        graph.add_edge(empty_latent_id, 0, ksampler_id, 3);
        graph.add_edge(ksampler_id, 0, vae_decode_id, 0);
        graph.add_edge(loader_id, 2, vae_decode_id, 1);
        graph.add_edge(vae_decode_id, 0, save_image_id, 0);

        // Signal progress
        let _ = state.progress_tx.send(ProgressInfo {
            progress: 0.0,
            eta_relative: 0.0,
            current_step: 0,
            total_steps: req.steps as usize,
            active: true,
        });

        // Execute
        state.reset_cancel();
        let outputs = {
            let mut ctx = state.ctx.lock().await;
            ctx.set_cancel(state.cancel.clone());
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

        let image = outputs
            .get(&(vae_decode_id, 0))
            .ok_or_else(|| ApiError::Internal("VAEDecode produced no output".to_string()))?;

        match image {
            NodeValue::Image(img) => {
                all_images_b64.extend(encode_images_base64(img)?);
            }
            other => {
                return Err(ApiError::Internal(format!(
                    "expected IMAGE, got {}",
                    other.type_name()
                )));
            }
        }
    }

    // Signal completion
    let _ = state.progress_tx.send(ProgressInfo::default());

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
        images: all_images_b64,
        info: serde_json::to_string(&info).expect("Txt2ImgInfo is always serializable"),
        parameters: info,
    }))
}
