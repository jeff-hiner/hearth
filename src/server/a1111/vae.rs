//! `GET /sdapi/v1/sd-vae` — list available VAE models.

use crate::server::AppState;
use axum::{Json, extract::State};
use serde::Serialize;
use std::sync::Arc;

/// List `.safetensors` files in `models/vae/`.
pub(super) async fn sd_vae(State(state): State<Arc<AppState>>) -> Json<Vec<VaeItem>> {
    let vae_dir = state.models_dir.join("vae");
    let mut items = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&vae_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors")
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
            {
                items.push(VaeItem {
                    model_name: name.to_string(),
                    filename: path.to_string_lossy().into_owned(),
                });
            }
        }
    }

    items.sort_by(|a, b| a.model_name.cmp(&b.model_name));
    Json(items)
}

#[derive(Serialize)]
pub(super) struct VaeItem {
    model_name: String,
    filename: String,
}
