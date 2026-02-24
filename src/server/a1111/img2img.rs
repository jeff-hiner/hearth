//! `POST /sdapi/v1/img2img` — image-to-image generation.

use super::{
    types::{Img2ImgRequest, Txt2ImgInfo, Txt2ImgResponse},
    util::{
        apply_controlnet_units, decode_base64_image, decode_base64_mask, encode_images_base64,
        rand_seed,
    },
};
use crate::{
    node::{
        CheckpointLoaderSimple, ClipTextEncode, KSampler, SaveImage, VaeDecode, VaeEncode,
        executor::{ExecutionGraph, Executor},
        value::NodeValue,
    },
    sampling::{SamplerKind, SchedulerKind},
    server::{AppState, ProgressInfo, error::ApiError},
    types::Backend,
};
use axum::{Json, extract::State};
use burn::tensor::Tensor;
use std::sync::Arc;

/// Handle img2img requests by building and executing a node graph.
pub(super) async fn img2img(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Img2ImgRequest>,
) -> Result<Json<Txt2ImgResponse>, ApiError> {
    tracing::info!(
        prompt = %req.prompt,
        width = req.width,
        height = req.height,
        steps = req.steps,
        denoise = req.denoising_strength,
        has_mask = req.mask.is_some(),
        "img2img request"
    );

    if req.tiling {
        return Err(ApiError::BadRequest(
            "Tiling mode is not yet supported by Hearth. Set tiling to false.".to_string(),
        ));
    }

    let init_b64 = req.init_images.first().ok_or_else(|| {
        ApiError::BadRequest("init_images must contain at least one image".to_string())
    })?;

    // Decode init image → [1, H, W, 3] in [0, 1]
    let init_image = decode_base64_image(init_b64)?;

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

    // Prepare mask once (if present)
    let prepared_mask = if let Some(ref mask_b64) = req.mask {
        Some(decode_and_prepare_mask(
            mask_b64,
            &init_image,
            req.inpainting_mask_invert,
            req.mask_blur,
        )?)
    } else {
        None
    };

    let n_iter = req.n_iter.max(1) as usize;
    let mut all_images_b64 = Vec::new();

    for iter_i in 0..n_iter {
        let iter_seed = seed + iter_i as u64;

        // Build execution graph
        let mut graph = ExecutionGraph::new();

        let loader_id = graph.add_node(Box::new(CheckpointLoaderSimple::new(
            ckpt_name.clone().into(),
        )));
        let clip_pos_id = graph.add_node(Box::new(ClipTextEncode::new(req.prompt.clone())));
        let clip_neg_id =
            graph.add_node(Box::new(ClipTextEncode::new(req.negative_prompt.clone())));
        let vae_encode_id = graph.add_node(Box::new(VaeEncode));
        let ksampler_id = graph.add_node(Box::new(KSampler::new(
            iter_seed,
            req.steps as usize,
            req.cfg_scale,
            sampler_kind,
            scheduler_kind,
            req.denoising_strength,
        )));
        let vae_decode_id = graph.add_node(Box::new(VaeDecode));
        let save_image_id = graph.add_node(Box::new(SaveImage::new("hearth_img2img".to_string())));

        graph.set_constant(vae_encode_id, 0, NodeValue::Image(init_image.clone()));

        // Wire edges
        graph.add_edge(loader_id, 1, clip_pos_id, 0);
        graph.add_edge(loader_id, 1, clip_neg_id, 0);
        graph.add_edge(loader_id, 2, vae_encode_id, 1);

        let (pos_id, pos_out) =
            apply_controlnet_units(&mut graph, &req.alwayson_scripts, clip_pos_id, 0)?;

        graph.add_edge(loader_id, 0, ksampler_id, 0);
        graph.add_edge(pos_id, pos_out, ksampler_id, 1);
        graph.add_edge(clip_neg_id, 0, ksampler_id, 2);
        graph.add_edge(vae_encode_id, 0, ksampler_id, 3);
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

        let output_image = outputs
            .get(&(vae_decode_id, 0))
            .ok_or_else(|| ApiError::Internal("VAEDecode produced no output".to_string()))?;

        let generated = match output_image {
            NodeValue::Image(img) => img,
            other => {
                return Err(ApiError::Internal(format!(
                    "expected IMAGE, got {}",
                    other.type_name()
                )));
            }
        };

        // Composite with mask if present
        let final_image = if let Some(ref mask) = prepared_mask {
            composite_with_mask(&init_image, generated, mask)?
        } else {
            generated.clone()
        };

        all_images_b64.extend(encode_images_base64(&final_image)?);
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

/// Decode a base64 mask, optionally invert it, and apply Gaussian blur.
///
/// Returns a mask tensor `[1, H, W]` in `[0, 1]` where 1 = inpaint region.
fn decode_and_prepare_mask(
    mask_b64: &str,
    init_image: &crate::types::Image,
    invert: u32,
    blur_radius: u32,
) -> Result<Tensor<Backend, 3>, ApiError> {
    let mut mask = decode_base64_mask(mask_b64)?;

    // Resize mask to match init image dimensions if needed
    let [_, img_h, img_w, _] = init_image.shape();
    let [_, mask_h, mask_w] = mask.shape().dims();
    if mask_h != img_h || mask_w != img_w {
        // Simple nearest-neighbor resize: just re-decode at the right size
        // For now, return an error if dimensions don't match
        return Err(ApiError::BadRequest(format!(
            "mask dimensions ({mask_w}x{mask_h}) must match init image ({img_w}x{img_h})"
        )));
    }

    // Invert mask if requested
    if invert == 1 {
        let ones: Tensor<Backend, 3> = Tensor::ones_like(&mask);
        mask = ones - mask;
    }

    // Apply box blur as an approximation of Gaussian blur
    if blur_radius > 0 {
        mask = box_blur_mask(mask, blur_radius);
    }

    Ok(mask)
}

/// Simple box blur for a `[1, H, W]` mask tensor.
///
/// Approximates Gaussian blur by averaging over a `(2*radius+1)` square kernel.
/// Applied via two separable 1D passes (horizontal then vertical) for efficiency.
fn box_blur_mask(mask: Tensor<Backend, 3>, radius: u32) -> Tensor<Backend, 3> {
    if radius == 0 {
        return mask;
    }

    let [_, h, w] = mask.shape().dims();
    let r = radius as i64;
    let kernel_size = (2 * r + 1) as f32;

    // Horizontal pass: average over columns using shifted copies (GPU-friendly)
    let mut h_acc: Tensor<Backend, 3> = Tensor::zeros_like(&mask);
    for dx in -r..=r {
        h_acc = h_acc + shift_horizontal(&mask, dx, w);
    }
    let blurred = h_acc / kernel_size;

    // Vertical pass: average over rows
    let mut v_acc: Tensor<Backend, 3> = Tensor::zeros_like(&blurred);
    for dy in -r..=r {
        v_acc = v_acc + shift_vertical(&blurred, dy, h);
    }
    v_acc / kernel_size
}

/// Shift a `[B, H, W]` tensor horizontally by `dx` pixels, clamping at edges.
fn shift_horizontal(t: &Tensor<Backend, 3>, dx: i64, w: usize) -> Tensor<Backend, 3> {
    if dx == 0 {
        return t.clone();
    }

    let [batch, h, _] = t.shape().dims();

    if dx > 0 {
        // Shift right: take [0..W-dx] and pad left with edge values
        let dx_u = dx as usize;
        let src = t.clone().slice([0..batch, 0..h, 0..w - dx_u]);
        let edge = t.clone().slice([0..batch, 0..h, 0..1]);
        let pad = edge.repeat_dim(2, dx_u);
        Tensor::cat(vec![pad, src], 2)
    } else {
        // Shift left: take [|dx|..W] and pad right with edge values
        let dx_u = (-dx) as usize;
        let src = t.clone().slice([0..batch, 0..h, dx_u..w]);
        let edge = t.clone().slice([0..batch, 0..h, w - 1..w]);
        let pad = edge.repeat_dim(2, dx_u);
        Tensor::cat(vec![src, pad], 2)
    }
}

/// Shift a `[B, H, W]` tensor vertically by `dy` pixels, clamping at edges.
fn shift_vertical(t: &Tensor<Backend, 3>, dy: i64, h: usize) -> Tensor<Backend, 3> {
    if dy == 0 {
        return t.clone();
    }

    let [batch, _, w] = t.shape().dims();

    if dy > 0 {
        let dy_u = dy as usize;
        let src = t.clone().slice([0..batch, 0..h - dy_u, 0..w]);
        let edge = t.clone().slice([0..batch, 0..1, 0..w]);
        let pad = edge.repeat_dim(1, dy_u);
        Tensor::cat(vec![pad, src], 1)
    } else {
        let dy_u = (-dy) as usize;
        let src = t.clone().slice([0..batch, dy_u..h, 0..w]);
        let edge = t.clone().slice([0..batch, h - 1..h, 0..w]);
        let pad = edge.repeat_dim(1, dy_u);
        Tensor::cat(vec![src, pad], 1)
    }
}

/// Composite generated image with init image using a mask.
///
/// `output = init * (1 - mask) + generated * mask`
///
/// Mask is `[1, H, W]` in `[0, 1]`, images are `[1, H, W, 3]`.
fn composite_with_mask(
    init: &crate::types::Image,
    generated: &crate::types::Image,
    mask: &Tensor<Backend, 3>,
) -> Result<crate::types::Image, ApiError> {
    // Expand mask from [1, H, W] to [1, H, W, 1] for broadcasting over channels
    let mask_4d: Tensor<Backend, 4> = mask.clone().unsqueeze_dim(3);

    let ones: Tensor<Backend, 4> = Tensor::ones_like(&mask_4d);
    let inv_mask = ones - mask_4d.clone();

    let composited = init.data.clone() * inv_mask + generated.data.clone() * mask_4d;
    Ok(crate::types::Image::new(composited))
}
