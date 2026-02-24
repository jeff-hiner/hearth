//! `GET /sdapi/v1/progress` — generation progress.

use super::types::A1111ProgressResponse;
use crate::server::AppState;
use axum::{Json, extract::State};
use std::sync::Arc;

/// Get current generation progress in A1111's nested format.
pub(super) async fn progress(State(state): State<Arc<AppState>>) -> Json<A1111ProgressResponse> {
    let info = state.progress_rx.borrow().clone();
    Json(A1111ProgressResponse::from_progress(&info))
}
