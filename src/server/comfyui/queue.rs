//! `GET /queue` — queue status.

use super::types::QueueStatus;
use crate::server::AppState;
use axum::{Json, extract::State};
use std::sync::Arc;

/// Get queue status.
///
/// Currently always returns empty queues since execution is synchronous.
pub(super) async fn get_queue(State(_state): State<Arc<AppState>>) -> Json<QueueStatus> {
    Json(QueueStatus {
        queue_pending: vec![],
        queue_running: vec![],
    })
}
