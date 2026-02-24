//! `GET/POST /sdapi/v1/options` — server options.

use super::types::SetOptionsRequest;
use crate::server::{AppState, ServerOptions};
use axum::{Json, extract::State};
use std::sync::Arc;

/// Get current server options.
pub(super) async fn get_options(State(state): State<Arc<AppState>>) -> Json<ServerOptions> {
    let opts = state.options.lock().await;
    Json(opts.clone())
}

/// Partially update server options.
///
/// Only fields present (non-null) in the request body are applied;
/// omitted fields keep their current values.
pub(super) async fn set_options(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SetOptionsRequest>,
) -> Json<ServerOptions> {
    let mut opts = state.options.lock().await;
    if let Some(ckpt) = req.sd_model_checkpoint {
        opts.sd_model_checkpoint = ckpt;
    }
    if let Some(vae) = req.sd_vae {
        opts.sd_vae = vae;
    }
    Json(opts.clone())
}
