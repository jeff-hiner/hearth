//! `POST /prompt` — submit a ComfyUI workflow for execution.

use super::{
    types::{PromptRequest, PromptResponse},
    workflow,
};
use crate::{
    node::executor::Executor,
    server::{AppState, ProgressInfo, error::ApiError},
};
use axum::{Json, extract::State};
use std::sync::Arc;

/// Submit a ComfyUI workflow for execution.
pub(super) async fn submit_prompt(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PromptRequest>,
) -> Result<Json<PromptResponse>, ApiError> {
    let prompt_id = uuid::Uuid::new_v4().to_string();

    tracing::info!(
        prompt_id = %prompt_id,
        num_nodes = req.prompt.len(),
        "received ComfyUI prompt"
    );

    // Parse the workflow into an execution graph
    let graph = workflow::parse_workflow(&req.prompt)?;

    // Execute (holds the mutex for the duration)
    {
        let mut ctx = state.ctx.lock().await;
        let progress_tx = state.progress_tx.clone();
        ctx.set_progress(Box::new(move |current, total| {
            let _ = progress_tx.send(ProgressInfo {
                progress: current as f32 / total as f32,
                eta_relative: 0.0,
                current_step: current,
                total_steps: total,
                active: true,
            });
        }));
        Executor::run(&graph, &mut ctx)?;
    }

    Ok(Json(PromptResponse { prompt_id }))
}
