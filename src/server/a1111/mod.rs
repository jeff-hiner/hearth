//! A1111-compatible REST API.
//!
//! Provides the subset of the Automatic1111 WebUI API needed by
//! StableProjectorz and similar clients.

mod models;
mod options;
mod progress;
mod txt2img;
pub(crate) mod types;

use crate::server::AppState;
use axum::{Router, routing};
use std::sync::Arc;

/// Build the A1111 API router.
pub(super) fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/sdapi/v1/txt2img", routing::post(txt2img::txt2img))
        .route("/sdapi/v1/sd-models", routing::get(models::sd_models))
        .route(
            "/sdapi/v1/options",
            routing::get(options::get_options).post(options::set_options),
        )
        .route("/sdapi/v1/progress", routing::get(progress::progress))
}
