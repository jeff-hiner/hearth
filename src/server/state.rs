//! Shared application state for the server.

use crate::node::context::ExecutionContext;
use std::{path::PathBuf, sync::Arc};
use tokio::sync::{Mutex, watch};

/// Shared application state, wrapped in `Arc` at the router level.
pub struct AppState {
    /// Execution context (owns model manager + device). Mutex serializes GPU access.
    pub ctx: Mutex<ExecutionContext>,
    /// Base directory for model files.
    pub models_dir: PathBuf,
    /// Base directory for output images.
    pub output_dir: PathBuf,
    /// Server-level options (current model, VAE, etc.).
    pub options: Mutex<ServerOptions>,
    /// Progress sender — updated during generation.
    pub progress_tx: watch::Sender<ProgressInfo>,
    /// Progress receiver — read by progress endpoint.
    pub progress_rx: watch::Receiver<ProgressInfo>,
}

impl AppState {
    /// Create a new `AppState` wrapped in `Arc`.
    pub fn new(models_dir: PathBuf, output_dir: PathBuf) -> Arc<Self> {
        let device = Default::default();
        let (progress_tx, progress_rx) = watch::channel(ProgressInfo::default());
        let ctx = ExecutionContext::new(device, models_dir.clone(), output_dir.clone());

        Arc::new(Self {
            ctx: Mutex::new(ctx),
            models_dir,
            output_dir,
            options: Mutex::new(ServerOptions::default()),
            progress_tx,
            progress_rx,
        })
    }
}

/// Server options (mirrors A1111's `/sdapi/v1/options`).
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ServerOptions {
    /// Currently loaded checkpoint filename.
    pub sd_model_checkpoint: String,
}

/// Progress information for the current generation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProgressInfo {
    /// Progress fraction `[0.0, 1.0]`.
    pub progress: f32,
    /// ETA in seconds (0.0 if unknown).
    pub eta_relative: f32,
    /// Current step.
    pub current_step: usize,
    /// Total steps.
    pub total_steps: usize,
    /// Whether a job is currently running.
    pub active: bool,
}

impl Default for ProgressInfo {
    fn default() -> Self {
        Self {
            progress: 0.0,
            eta_relative: 0.0,
            current_step: 0,
            total_steps: 0,
            active: false,
        }
    }
}
