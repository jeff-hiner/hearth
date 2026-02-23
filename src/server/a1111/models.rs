//! `GET /sdapi/v1/sd-models` — list available checkpoints.

use super::types::SdModel;
use crate::server::AppState;
use axum::{Json, extract::State};
use std::sync::Arc;

/// List all `.safetensors` files in `models/checkpoints/`.
pub(super) async fn sd_models(State(state): State<Arc<AppState>>) -> Json<Vec<SdModel>> {
    let ckpt_dir = state.models_dir.join("checkpoints");

    let mut models = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&ckpt_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                let filename = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let model_name = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                models.push(SdModel {
                    title: model_name.clone(),
                    model_name,
                    filename,
                });
            }
        }
    }

    models.sort_by(|a, b| a.title.cmp(&b.title));
    Json(models)
}
