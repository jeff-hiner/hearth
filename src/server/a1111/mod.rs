//! A1111-compatible REST API.
//!
//! Provides the subset of the Automatic1111 WebUI API needed by
//! StableProjectorz and similar clients.

mod controlnet;
mod img2img;
mod interrupt;
mod models;
mod options;
mod progress;
mod samplers;
mod schedulers;
mod sysinfo;
mod txt2img;
mod types;
mod unload;
mod upscalers;
mod util;
mod vae;

use crate::server::AppState;
use axum::{Router, http::StatusCode, routing};
use std::sync::Arc;
pub use types::{
    AlwaysOnScripts, ControlNetArgs, ControlNetUnit, OverrideSettings, SdModel, Txt2ImgInfo,
    Txt2ImgRequest, Txt2ImgResponse,
};

/// Build the A1111 API router.
pub(super) fn router() -> Router<Arc<AppState>> {
    Router::new()
        // Startup probe
        .route("/internal/ping", routing::get(|| async { StatusCode::OK }))
        // Generation
        .route("/sdapi/v1/txt2img", routing::post(txt2img::txt2img))
        .route("/sdapi/v1/img2img", routing::post(img2img::img2img))
        // Model/sampler/scheduler info
        .route("/sdapi/v1/sd-models", routing::get(models::sd_models))
        .route("/sdapi/v1/samplers", routing::get(samplers::samplers))
        .route("/sdapi/v1/schedulers", routing::get(schedulers::schedulers))
        // Options
        .route(
            "/sdapi/v1/options",
            routing::get(options::get_options).post(options::set_options),
        )
        // Progress / control
        .route("/sdapi/v1/progress", routing::get(progress::progress))
        .route("/sdapi/v1/interrupt", routing::post(interrupt::interrupt))
        // ControlNet extension
        .route(
            "/controlnet/model_list",
            routing::get(controlnet::model_list),
        )
        .route(
            "/controlnet/module_list",
            routing::get(controlnet::module_list),
        )
        .route(
            "/controlnet/control_types",
            routing::get(controlnet::control_types),
        )
        .route("/controlnet/settings", routing::get(controlnet::settings))
        .route("/controlnet/detect", routing::post(controlnet::detect))
        // Upscaling
        .route("/sdapi/v1/upscalers", routing::get(upscalers::upscalers))
        .route(
            "/sdapi/v1/extra-batch-images",
            routing::post(upscalers::extra_batch_images),
        )
        // Misc
        .route("/internal/sysinfo", routing::get(sysinfo::sysinfo))
        .route(
            "/sdapi/v1/unload-checkpoint",
            routing::post(unload::unload_checkpoint),
        )
        .route("/sdapi/v1/sd-vae", routing::get(vae::sd_vae))
}
