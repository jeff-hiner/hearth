//! Shared application state for the server.

use crate::{node::context::ExecutionContext, types::Backend};
use burn::{
    backend::wgpu::{RuntimeOptions, WgpuDevice, graphics::AutoGraphicsApi, init_setup},
    tensor::Device,
};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::{Mutex, watch};

/// Shared application state, wrapped in `Arc` at the router level.
pub struct AppState {
    /// Execution context (owns model manager + device). Mutex serializes GPU access.
    pub ctx: Mutex<ExecutionContext>,
    /// The GPU device used for inference. Exposed so request handlers can create tensors
    /// without calling `Default::default()` (which skips our `init_setup` options).
    pub device: Device<Backend>,
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
    /// Cancellation flag — set by `/sdapi/v1/interrupt`, checked each sampling step.
    pub cancel: Arc<AtomicBool>,
}

impl AppState {
    /// Create a new `AppState` wrapped in `Arc`.
    pub fn new(models_dir: PathBuf, output_dir: PathBuf) -> Arc<Self> {
        // Pre-register the wgpu runtime with an explicit tasks_max so that the
        // subsequent Default::default() device construction reuses this server.
        init_setup::<AutoGraphicsApi>(
            &WgpuDevice::DefaultDevice,
            RuntimeOptions { tasks_max: 512, ..Default::default() },
        );
        let device: Device<Backend> = Default::default();
        let (progress_tx, progress_rx) = watch::channel(ProgressInfo::default());
        let ctx = ExecutionContext::new(device.clone(), models_dir.clone(), output_dir.clone());

        Arc::new(Self {
            ctx: Mutex::new(ctx),
            device,
            models_dir,
            output_dir,
            options: Mutex::new(ServerOptions::default()),
            progress_tx,
            progress_rx,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Returns `true` if cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Reset the cancellation flag (call before starting a new job).
    pub fn reset_cancel(&self) {
        self.cancel.store(false, Ordering::Relaxed);
    }
}

/// Server options (mirrors A1111's `/sdapi/v1/options`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ServerOptions {
    /// Currently loaded checkpoint filename.
    pub sd_model_checkpoint: String,
    /// VAE selection ("Automatic" or a specific filename).
    #[serde(default = "default_vae")]
    pub sd_vae: String,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            sd_model_checkpoint: String::new(),
            sd_vae: default_vae(),
        }
    }
}

fn default_vae() -> String {
    "Automatic".to_string()
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
