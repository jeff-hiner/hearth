//! Error types for node execution.

use crate::model_loader::LoadError;
use std::fmt;

/// Identifier for a node in the execution graph.
pub(crate) type NodeId = u64;

/// Error type for node execution.
#[derive(Debug)]
pub enum NodeError {
    /// A required input slot was not connected.
    MissingInput {
        /// Name of the input slot.
        slot: &'static str,
        /// Node that expected the input.
        node_type: &'static str,
    },
    /// Input value type didn't match expected type.
    TypeMismatch {
        /// Name of the input slot.
        slot: &'static str,
        /// Expected type description.
        expected: &'static str,
        /// Actual type description.
        got: &'static str,
    },
    /// Model handle was not found in the model manager.
    ModelNotFound {
        /// Description of the model that was expected.
        description: String,
    },
    /// Model loading failed.
    Load(LoadError),
    /// VRAM budget exceeded.
    OutOfMemory {
        /// How many bytes were needed.
        needed: u64,
        /// How many bytes were available.
        available: u64,
    },
    /// Graph has a cycle and cannot be executed.
    CycleDetected,
    /// A node referenced a non-existent node ID.
    UnknownNode {
        /// The ID that was not found.
        id: NodeId,
    },
    /// An output slot index was out of bounds.
    SlotOutOfBounds {
        /// The node whose output was referenced.
        node_id: NodeId,
        /// The slot index that was requested.
        slot: usize,
        /// How many output slots the node actually has.
        available: usize,
    },
    /// Generic execution error with context.
    Execution {
        /// What went wrong.
        message: String,
    },
}

impl fmt::Display for NodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingInput { slot, node_type } => {
                write!(f, "{node_type}: missing required input '{slot}'")
            }
            Self::TypeMismatch {
                slot,
                expected,
                got,
            } => {
                write!(f, "slot '{slot}': expected {expected}, got {got}")
            }
            Self::ModelNotFound { description } => {
                write!(f, "model not found: {description}")
            }
            Self::Load(e) => write!(f, "model load error: {e}"),
            Self::OutOfMemory { needed, available } => {
                write!(
                    f,
                    "out of VRAM: need {} MiB, only {} MiB available",
                    needed / (1024 * 1024),
                    available / (1024 * 1024)
                )
            }
            Self::CycleDetected => write!(f, "execution graph contains a cycle"),
            Self::UnknownNode { id } => write!(f, "unknown node ID: {id}"),
            Self::SlotOutOfBounds {
                node_id,
                slot,
                available,
            } => {
                write!(
                    f,
                    "node {node_id}: output slot {slot} out of bounds (has {available})"
                )
            }
            Self::Execution { message } => write!(f, "execution error: {message}"),
        }
    }
}

impl std::error::Error for NodeError {}

impl From<LoadError> for NodeError {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}
