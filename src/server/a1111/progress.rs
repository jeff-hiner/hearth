//! `GET /sdapi/v1/progress` — generation progress.

use crate::server::{AppState, ProgressInfo};
use axum::{Json, extract::State};
use std::sync::Arc;

/// Get current generation progress.
pub(super) async fn progress(State(state): State<Arc<AppState>>) -> Json<ProgressInfo> {
    let info = state.progress_rx.borrow().clone();
    Json(info)
}
