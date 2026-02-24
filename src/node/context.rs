//! Execution context for node evaluation.
//!
//! [`ExecutionContext`] provides nodes with access to the model manager,
//! device, and progress reporting during graph execution.

use crate::{model_manager::ModelManager, types::Backend};
use burn::tensor::Device;
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

/// Progress callback signature: `(current_step, total_steps)`.
pub(crate) type ProgressFn = Box<dyn Fn(usize, usize) + Send + Sync>;

/// Context passed to each node during execution.
///
/// Provides access to shared resources without requiring nodes to hold
/// references to the model manager or device themselves.
pub struct ExecutionContext {
    /// Model lifecycle manager — owns all loaded models.
    pub models: ModelManager,
    /// Base directory for output files.
    output_dir: PathBuf,
    /// Optional progress callback for long-running operations.
    pub(crate) progress: Option<ProgressFn>,
    /// Cancellation flag — checked each sampling step for early exit.
    pub(crate) cancel: Arc<AtomicBool>,
}

impl std::fmt::Debug for ExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("models", &self.models)
            .field("output_dir", &self.output_dir)
            .finish()
    }
}

impl ExecutionContext {
    /// Create a new execution context.
    pub fn new(device: Device<Backend>, models_dir: PathBuf, output_dir: PathBuf) -> Self {
        Self {
            models: ModelManager::new(models_dir, device),
            output_dir,
            progress: None,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set the progress callback.
    pub fn set_progress(&mut self, f: ProgressFn) {
        self.progress = Some(f);
    }

    /// Set the cancellation flag (shared with the server's interrupt handler).
    pub fn set_cancel(&mut self, flag: Arc<AtomicBool>) {
        self.cancel = flag;
    }

    /// Returns `true` if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Report progress (no-op if no callback is set).
    pub fn report_progress(&self, current: usize, total: usize) {
        if let Some(f) = &self.progress {
            f(current, total);
        }
    }

    /// Get the Burn device.
    pub fn device(&self) -> &Device<Backend> {
        self.models.device()
    }

    /// Get the models directory.
    pub fn models_dir(&self) -> &Path {
        self.models.models_dir()
    }

    /// Get the output directory.
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }
}
