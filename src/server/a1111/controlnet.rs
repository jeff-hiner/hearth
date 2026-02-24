//! ControlNet extension endpoints for the A1111 API.
//!
//! Provides the subset of the ControlNet extension API that StableProjectorz
//! queries during initialization and usage.

use crate::server::AppState;
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use serde::Serialize;
use std::sync::Arc;

/// `GET /controlnet/model_list` — list available ControlNet models.
pub(super) async fn model_list(State(state): State<Arc<AppState>>) -> Json<ControlNetModelList> {
    let cn_dir = state.models_dir.join("controlnet");
    let mut models = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&cn_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    models.push(name.to_string());
                }
            }
        }
    }

    models.sort();
    Json(ControlNetModelList { model_list: models })
}

/// `GET /controlnet/module_list` — list available preprocessor modules.
pub(super) async fn module_list() -> Json<ControlNetModuleList> {
    Json(ControlNetModuleList {
        module_list: vec!["none".to_string()],
    })
}

/// `GET /controlnet/control_types` — list control type categories.
pub(super) async fn control_types() -> Json<ControlNetControlTypes> {
    Json(ControlNetControlTypes {
        control_types: serde_json::json!({}),
    })
}

/// `GET /controlnet/settings` — ControlNet extension settings.
pub(super) async fn settings() -> Json<ControlNetSettings> {
    Json(ControlNetSettings {
        control_net_unit_count: 3,
    })
}

/// `POST /controlnet/detect` — run a preprocessor on an image.
///
/// Not implemented — Hearth does not bundle preprocessors.
pub(super) async fn detect() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": "not_implemented",
            "detail": "ControlNet preprocessing is not yet supported by Hearth. \
                       Send pre-processed images directly."
        })),
    )
}

#[derive(Serialize)]
pub(super) struct ControlNetModelList {
    model_list: Vec<String>,
}

#[derive(Serialize)]
pub(super) struct ControlNetModuleList {
    module_list: Vec<String>,
}

#[derive(Serialize)]
pub(super) struct ControlNetControlTypes {
    control_types: serde_json::Value,
}

#[derive(Serialize)]
pub(super) struct ControlNetSettings {
    control_net_unit_count: u32,
}
