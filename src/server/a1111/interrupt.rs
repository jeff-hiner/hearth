//! `POST /sdapi/v1/interrupt` — cancel the current generation.

use crate::server::AppState;
use axum::{extract::State, http::StatusCode};
use std::sync::{Arc, atomic::Ordering};

/// Signal cancellation of the current generation job.
///
/// The sampling loop checks the flag each step and breaks early.
pub(super) async fn interrupt(State(state): State<Arc<AppState>>) -> StatusCode {
    state.cancel.store(true, Ordering::Relaxed);
    tracing::info!("interrupt requested");
    StatusCode::OK
}
