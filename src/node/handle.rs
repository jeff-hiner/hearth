//! Lightweight model handle IDs for the node system.
//!
//! Handles are `Copy` u64 identifiers that reference models owned by the
//! [`ModelManager`](crate::model_manager::ModelManager). They avoid putting
//! multi-GB model structs behind `Arc` or into `NodeValue` directly.

/// Handle to a loaded UNet model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelHandle(pub(crate) u64);

/// Handle to a loaded VAE model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VaeHandle(pub(crate) u64);

/// Handle to a loaded CLIP text encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClipHandle(pub(crate) u64);

/// Handle to a loaded ControlNet model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ControlNetHandle(pub(crate) u64);
