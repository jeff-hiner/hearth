//! Error types for model management operations.

use crate::model_loader::LoadError;
use std::fmt;

/// Error type for model manager operations.
#[derive(Debug)]
pub enum ModelError {
    /// Model loading failed.
    Load(LoadError),
    /// Model handle not found.
    NotFound {
        /// The handle ID that was not found.
        id: u64,
    },
    /// Model is still borrowed and cannot be evicted.
    StillBorrowed {
        /// The handle ID that is still borrowed.
        id: u64,
    },
    /// Not enough VRAM to load the requested model.
    OutOfMemory {
        /// How many bytes were needed.
        needed: u64,
        /// How many bytes were available after eviction.
        available: u64,
    },
    /// An I/O error occurred.
    Io(std::io::Error),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(e) => write!(f, "model load error: {e}"),
            Self::NotFound { id } => write!(f, "model handle {id} not found"),
            Self::StillBorrowed { id } => write!(f, "model {id} is still borrowed"),
            Self::OutOfMemory { needed, available } => {
                write!(
                    f,
                    "out of VRAM: need {} MiB, only {} MiB available",
                    needed / (1024 * 1024),
                    available / (1024 * 1024)
                )
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<LoadError> for ModelError {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}

impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
