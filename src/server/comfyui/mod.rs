//! ComfyUI-compatible API.
//!
//! Provides the subset of the ComfyUI API needed by the ComfyUI frontend:
//! workflow submission, WebSocket progress, queue status, and history.

mod history;
mod prompt;
mod queue;
mod types;
mod workflow;
mod ws;

use crate::server::AppState;
use axum::{Router, routing};
use std::sync::Arc;

/// Build the ComfyUI API router.
pub(super) fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/prompt", routing::post(prompt::submit_prompt))
        .route("/queue", routing::get(queue::get_queue))
        .route("/history/{prompt_id}", routing::get(history::get_history))
        .route("/ws", routing::get(ws::ws_handler))
}
