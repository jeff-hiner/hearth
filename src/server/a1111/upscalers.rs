//! Upscaler-related endpoints for the A1111 API.

use axum::{Json, http::StatusCode, response::IntoResponse};
use serde::Serialize;

/// `GET /sdapi/v1/upscalers` — list available upscalers.
///
/// Returns only "None" since Hearth does not yet support upscaling.
pub(super) async fn upscalers() -> Json<Vec<UpscalerItem>> {
    Json(vec![UpscalerItem {
        name: "None".to_string(),
        model_name: None,
        model_path: None,
        model_url: None,
        scale: 1,
    }])
}

/// `POST /sdapi/v1/extra-batch-images` — upscale a batch of images.
///
/// Not implemented — returns 501.
pub(super) async fn extra_batch_images() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": "not_implemented",
            "detail": "Upscaling is not yet supported by Hearth"
        })),
    )
}

#[derive(Serialize)]
pub(super) struct UpscalerItem {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_url: Option<String>,
    scale: u32,
}
