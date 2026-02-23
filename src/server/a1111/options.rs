//! `GET/POST /sdapi/v1/options` — server options.

use crate::server::{AppState, ServerOptions};
use axum::{Json, extract::State};
use std::sync::Arc;

/// Get current server options.
pub(super) async fn get_options(State(state): State<Arc<AppState>>) -> Json<ServerOptions> {
    let opts = state.options.lock().await;
    Json(opts.clone())
}

/// Update server options.
pub(super) async fn set_options(
    State(state): State<Arc<AppState>>,
    Json(new_opts): Json<ServerOptions>,
) -> Json<ServerOptions> {
    let mut opts = state.options.lock().await;
    *opts = new_opts;
    Json(opts.clone())
}
