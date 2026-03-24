//! Shared utilities for A1111 API handlers.

use super::types::AlwaysOnScripts;
use crate::{
    node::{
        error::NodeId,
        executor::ExecutionGraph,
        nodes::{ControlNetApply, ControlNetLoader},
        value::NodeValue,
    },
    server::error::ApiError,
    types::{Backend, Image},
};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use burn::tensor::{Device, Tensor, TensorData};
use image::{ImageEncoder, codecs::png::PngEncoder};
use std::time::SystemTime;

/// Decode a base64-encoded PNG/JPEG image into raw [`TensorData`].
///
/// Accepts optional `data:image/...;base64,` prefix (stripped automatically).
/// Returns `[1, H, W, 3]` f32 data in `[0.0, 1.0]`. Callers must supply a
/// device to construct the final tensor.
pub(super) fn decode_base64_image(b64: &str) -> Result<TensorData, ApiError> {
    // Strip optional data URI prefix
    let raw_b64 = if let Some(idx) = b64.find(";base64,") {
        &b64[idx + 8..]
    } else {
        b64
    };

    let bytes = BASE64
        .decode(raw_b64)
        .map_err(|e| ApiError::BadRequest(format!("invalid base64 image: {e}")))?;

    let dyn_img = image::load_from_memory(&bytes)
        .map_err(|e| ApiError::BadRequest(format!("failed to decode image: {e}")))?;

    let rgb = dyn_img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let raw = rgb.as_raw();

    // Convert HWC u8 [0,255] → HWC f32 [0,1]
    let mut data = vec![0.0f32; h * w * 3];
    for (i, chunk) in raw.chunks_exact(3).enumerate() {
        data[i * 3] = chunk[0] as f32 / 255.0;
        data[i * 3 + 1] = chunk[1] as f32 / 255.0;
        data[i * 3 + 2] = chunk[2] as f32 / 255.0;
    }

    Ok(TensorData::new(data, [1, h, w, 3])
        .convert::<<Backend as burn::tensor::backend::Backend>::FloatElem>())
}

/// Decode a base64-encoded image into raw grayscale [`TensorData`].
///
/// Returns `[1, H, W]` f32 data in `[0.0, 1.0]` where 1.0 = white = inpaint
/// region. Callers must supply a device to construct the final tensor.
pub(super) fn decode_base64_mask(b64: &str) -> Result<TensorData, ApiError> {
    let raw_b64 = if let Some(idx) = b64.find(";base64,") {
        &b64[idx + 8..]
    } else {
        b64
    };

    let bytes = BASE64
        .decode(raw_b64)
        .map_err(|e| ApiError::BadRequest(format!("invalid base64 mask: {e}")))?;

    let dyn_img = image::load_from_memory(&bytes)
        .map_err(|e| ApiError::BadRequest(format!("failed to decode mask image: {e}")))?;

    let gray = dyn_img.to_luma8();
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    let raw = gray.as_raw();

    let data: Vec<f32> = raw.iter().map(|&v| v as f32 / 255.0).collect();

    Ok(TensorData::new(data, [1, h, w])
        .convert::<<Backend as burn::tensor::backend::Backend>::FloatElem>())
}

/// Encode an image tensor as base64 PNG strings (one per batch element).
pub(super) fn encode_images_base64(image: &Image) -> Result<Vec<String>, ApiError> {
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

/// Generate a random seed from system time.
pub(super) fn rand_seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42)
}

/// Wire ControlNet units from `alwayson_scripts` into an execution graph.
///
/// For each enabled unit, adds a `ControlNetLoader` and `ControlNetApply` node,
/// chaining onto the positive conditioning. Returns the final conditioning node ID
/// (which may be the original `pos_cond_id` if no ControlNets are enabled).
///
/// Rejects units with `module != "none"` since Hearth doesn't support preprocessors.
pub(super) fn apply_controlnet_units(
    device: &Device<Backend>,
    graph: &mut ExecutionGraph,
    scripts: &Option<AlwaysOnScripts>,
    pos_cond_id: NodeId,
    pos_cond_output: usize,
) -> Result<(NodeId, usize), ApiError> {
    let scripts = match scripts {
        Some(s) => s,
        None => return Ok((pos_cond_id, pos_cond_output)),
    };

    for key in scripts.unknown.keys() {
        tracing::warn!(
            extension = %key,
            "unrecognized alwayson_scripts extension (ignored)"
        );
    }

    let units = match &scripts.controlnet {
        Some(cn_args) => &cn_args.args,
        None => return Ok((pos_cond_id, pos_cond_output)),
    };

    let mut current_cond_id = pos_cond_id;
    let mut current_cond_output = pos_cond_output;

    for (i, unit) in units.iter().enumerate() {
        if !unit.enabled {
            continue;
        }

        if unit.module != "none" {
            return Err(ApiError::BadRequest(format!(
                "ControlNet preprocessor '{}' is not supported. \
                 Use module='none' with a pre-processed image.",
                unit.module
            )));
        }

        if unit.model.is_empty() {
            return Err(ApiError::BadRequest(format!(
                "ControlNet unit {i} is enabled but has no model specified"
            )));
        }

        let image_b64 = unit.image.as_deref().ok_or_else(|| {
            ApiError::BadRequest(format!("ControlNet unit {i} is enabled but has no image"))
        })?;

        let control_image = Image::new(Tensor::from_data(decode_base64_image(image_b64)?, device));

        let loader_id = graph.add_node(Box::new(ControlNetLoader::new(unit.model.clone().into())));
        let apply_id = graph.add_node(Box::new(ControlNetApply::new(
            unit.weight,
            unit.guidance_start,
            unit.guidance_end,
        )));

        // Set the decoded image as a constant on ControlNetApply's image input (slot 2)
        graph.set_constant(apply_id, 2, NodeValue::Image(control_image));

        // ControlNetApply inputs: 0=conditioning, 1=control_net, 2=image
        graph.add_edge(current_cond_id, current_cond_output, apply_id, 0);
        graph.add_edge(loader_id, 0, apply_id, 1);

        current_cond_id = apply_id;
        current_cond_output = 0;

        tracing::info!(
            unit = i,
            model = %unit.model,
            weight = unit.weight,
            guidance_start = unit.guidance_start,
            guidance_end = unit.guidance_end,
            "ControlNet unit added"
        );
    }

    Ok((current_cond_id, current_cond_output))
}
