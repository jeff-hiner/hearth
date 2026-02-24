//! `POST /sdapi/v1/unload-checkpoint` — unload all loaded models.

use crate::server::AppState;
use axum::{extract::State, http::StatusCode};
use std::sync::Arc;

/// Unload all models from VRAM.
pub(super) async fn unload_checkpoint(State(state): State<Arc<AppState>>) -> StatusCode {
    let mut ctx = state.ctx.lock().await;
    ctx.models.unload_all();
    tracing::info!("checkpoint unloaded via API");
    StatusCode::OK
}
