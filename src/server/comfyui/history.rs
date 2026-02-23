//! `GET /history/{prompt_id}` — prompt execution history.

use super::types::HistoryEntry;
use crate::server::AppState;
use axum::{
    Json,
    extract::{Path, State},
};
use std::{collections::HashMap, sync::Arc};

/// Get execution history for a prompt.
///
/// Currently returns an empty object. History tracking is deferred.
pub(super) async fn get_history(
    State(_state): State<Arc<AppState>>,
    Path(_prompt_id): Path<String>,
) -> Json<HashMap<String, HistoryEntry>> {
    // TODO: Track execution history per prompt_id
    Json(HashMap::new())
}
