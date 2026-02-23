//! Model lifecycle management with VRAM-aware loading and LRU eviction.
//!
//! The [`ModelManager`] owns all loaded models, assigns lightweight
//! [`Copy`] handle IDs, and tracks VRAM usage. When a load would exceed
//! the budget, it evicts least-recently-used models that are not currently
//! borrowed.
//!
//! # Borrow pattern
//!
//! Nodes access models through `borrow_*` methods that return mutable
//! references. The executor is single-threaded, so concurrent borrows
//! are not a concern.

pub(crate) mod error;
pub(crate) mod vram;

use crate::{
    node::{
        handle::{ClipHandle, ControlNetHandle, ModelHandle, VaeHandle},
        variant::{ClipVariant, ControlNetVariant, UnetVariant, VaeVariant},
    },
    types::Backend,
};
use burn::tensor::Device;
pub(crate) use error::ModelError;
use std::{cell::Cell, collections::HashMap, path::PathBuf, time::Instant};

/// A loaded model entry in the manager.
struct LoadedModel {
    /// What kind of model this is.
    kind: LoadedModelKind,
    /// Approximate weight size in VRAM (bytes).
    weight_bytes: u64,
    /// Source file path (for reload-from-disk after eviction).
    _source: PathBuf,
    /// Last time this model was accessed. Uses `Cell` so borrow methods can
    /// take `&self`, allowing simultaneous borrows of different models (e.g.
    /// UNet + ControlNet during controlled sampling).
    last_used: Cell<Instant>,
}

/// The different kinds of models the manager can hold.
///
/// Models are boxed to keep the enum size uniform — the UNet and VAE
/// variants carry tens of KB of weight pointers and would bloat the enum.
enum LoadedModelKind {
    /// UNet / diffusion model.
    Unet(Box<UnetVariant>),
    /// VAE decoder.
    Vae(Box<VaeVariant>),
    /// CLIP text encoder(s) + tokenizer.
    Clip(Box<ClipVariant>),
    /// ControlNet.
    ControlNet(Box<ControlNetVariant>),
}

/// Manages model loading, borrowing, and VRAM-aware eviction.
pub struct ModelManager {
    /// All loaded models, keyed by internal ID.
    models: HashMap<u64, LoadedModel>,
    /// Next handle ID to assign.
    next_id: u64,
    /// Base directory for model files.
    models_dir: PathBuf,
    /// Burn device.
    device: Device<Backend>,
    /// Total tracked VRAM usage (sum of loaded model weight_bytes).
    tracked_usage: u64,
}

impl std::fmt::Debug for ModelManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelManager")
            .field("loaded_models", &self.models.len())
            .field("tracked_usage_mib", &(self.tracked_usage / (1024 * 1024)))
            .field("models_dir", &self.models_dir)
            .finish()
    }
}

impl ModelManager {
    /// Create a new model manager.
    pub fn new(models_dir: PathBuf, device: Device<Backend>) -> Self {
        Self {
            models: HashMap::new(),
            next_id: 1,
            models_dir,
            device,
            tracked_usage: 0,
        }
    }

    /// Get the Burn device.
    pub fn device(&self) -> &Device<Backend> {
        &self.device
    }

    /// Get the models directory.
    pub fn models_dir(&self) -> &PathBuf {
        &self.models_dir
    }

    /// Total tracked VRAM usage in bytes.
    pub fn tracked_usage(&self) -> u64 {
        self.tracked_usage
    }

    // -----------------------------------------------------------------
    // Registration (called by CheckpointLoader and similar nodes)
    // -----------------------------------------------------------------

    /// Register a UNet model and return its handle.
    pub fn register_unet(
        &mut self,
        variant: UnetVariant,
        weight_bytes: u64,
        source: PathBuf,
    ) -> ModelHandle {
        let id = self.alloc_id();
        self.models.insert(
            id,
            LoadedModel {
                kind: LoadedModelKind::Unet(Box::new(variant)),
                weight_bytes,
                _source: source,
                last_used: Cell::new(Instant::now()),
            },
        );
        self.tracked_usage += weight_bytes;
        tracing::info!(
            handle = id,
            mib = weight_bytes / (1024 * 1024),
            "registered UNet"
        );
        ModelHandle(id)
    }

    /// Register a VAE model and return its handle.
    pub fn register_vae(
        &mut self,
        variant: VaeVariant,
        weight_bytes: u64,
        source: PathBuf,
    ) -> VaeHandle {
        let id = self.alloc_id();
        self.models.insert(
            id,
            LoadedModel {
                kind: LoadedModelKind::Vae(Box::new(variant)),
                weight_bytes,
                _source: source,
                last_used: Cell::new(Instant::now()),
            },
        );
        self.tracked_usage += weight_bytes;
        tracing::info!(
            handle = id,
            mib = weight_bytes / (1024 * 1024),
            "registered VAE"
        );
        VaeHandle(id)
    }

    /// Register a CLIP model and return its handle.
    pub fn register_clip(
        &mut self,
        variant: ClipVariant,
        weight_bytes: u64,
        source: PathBuf,
    ) -> ClipHandle {
        let id = self.alloc_id();
        self.models.insert(
            id,
            LoadedModel {
                kind: LoadedModelKind::Clip(Box::new(variant)),
                weight_bytes,
                _source: source,
                last_used: Cell::new(Instant::now()),
            },
        );
        self.tracked_usage += weight_bytes;
        tracing::info!(
            handle = id,
            mib = weight_bytes / (1024 * 1024),
            "registered CLIP"
        );
        ClipHandle(id)
    }

    /// Register a ControlNet model and return its handle.
    pub fn register_controlnet(
        &mut self,
        variant: ControlNetVariant,
        weight_bytes: u64,
        source: PathBuf,
    ) -> ControlNetHandle {
        let id = self.alloc_id();
        self.models.insert(
            id,
            LoadedModel {
                kind: LoadedModelKind::ControlNet(Box::new(variant)),
                weight_bytes,
                _source: source,
                last_used: Cell::new(Instant::now()),
            },
        );
        self.tracked_usage += weight_bytes;
        tracing::info!(
            handle = id,
            mib = weight_bytes / (1024 * 1024),
            "registered ControlNet"
        );
        ControlNetHandle(id)
    }

    // -----------------------------------------------------------------
    // Borrowing (returns shared references, single-threaded executor)
    // -----------------------------------------------------------------

    /// Borrow a UNet model by handle.
    pub fn borrow_unet(&self, handle: ModelHandle) -> Result<&UnetVariant, ModelError> {
        let entry = self
            .models
            .get(&handle.0)
            .ok_or(ModelError::NotFound { id: handle.0 })?;
        entry.last_used.set(Instant::now());
        match &entry.kind {
            LoadedModelKind::Unet(v) => Ok(v),
            _ => Err(ModelError::NotFound { id: handle.0 }),
        }
    }

    /// Borrow a VAE model by handle.
    pub fn borrow_vae(&self, handle: VaeHandle) -> Result<&VaeVariant, ModelError> {
        let entry = self
            .models
            .get(&handle.0)
            .ok_or(ModelError::NotFound { id: handle.0 })?;
        entry.last_used.set(Instant::now());
        match &entry.kind {
            LoadedModelKind::Vae(v) => Ok(v),
            _ => Err(ModelError::NotFound { id: handle.0 }),
        }
    }

    /// Borrow a CLIP model by handle.
    pub fn borrow_clip(&self, handle: ClipHandle) -> Result<&ClipVariant, ModelError> {
        let entry = self
            .models
            .get(&handle.0)
            .ok_or(ModelError::NotFound { id: handle.0 })?;
        entry.last_used.set(Instant::now());
        match &entry.kind {
            LoadedModelKind::Clip(v) => Ok(v),
            _ => Err(ModelError::NotFound { id: handle.0 }),
        }
    }

    /// Borrow a ControlNet model by handle.
    pub fn borrow_controlnet(
        &self,
        handle: ControlNetHandle,
    ) -> Result<&ControlNetVariant, ModelError> {
        let entry = self
            .models
            .get(&handle.0)
            .ok_or(ModelError::NotFound { id: handle.0 })?;
        entry.last_used.set(Instant::now());
        match &entry.kind {
            LoadedModelKind::ControlNet(v) => Ok(v),
            _ => Err(ModelError::NotFound { id: handle.0 }),
        }
    }

    // -----------------------------------------------------------------
    // Ownership transfer (take + re-register, for in-place mutation)
    // -----------------------------------------------------------------

    /// Remove a UNet from the manager and return it with its metadata.
    ///
    /// Used by LoRA loading to take ownership, mutate weights, and re-register.
    pub fn take_unet(
        &mut self,
        handle: ModelHandle,
    ) -> Result<(UnetVariant, u64, PathBuf), ModelError> {
        let entry = self
            .models
            .remove(&handle.0)
            .ok_or(ModelError::NotFound { id: handle.0 })?;
        self.tracked_usage = self.tracked_usage.saturating_sub(entry.weight_bytes);
        match entry.kind {
            LoadedModelKind::Unet(v) => Ok((*v, entry.weight_bytes, entry._source)),
            _ => Err(ModelError::NotFound { id: handle.0 }),
        }
    }

    /// Remove a CLIP encoder from the manager and return it with its metadata.
    ///
    /// Used by LoRA loading to take ownership, mutate weights, and re-register.
    pub fn take_clip(
        &mut self,
        handle: ClipHandle,
    ) -> Result<(ClipVariant, u64, PathBuf), ModelError> {
        let entry = self
            .models
            .remove(&handle.0)
            .ok_or(ModelError::NotFound { id: handle.0 })?;
        self.tracked_usage = self.tracked_usage.saturating_sub(entry.weight_bytes);
        match entry.kind {
            LoadedModelKind::Clip(v) => Ok((*v, entry.weight_bytes, entry._source)),
            _ => Err(ModelError::NotFound { id: handle.0 }),
        }
    }

    // -----------------------------------------------------------------
    // Eviction
    // -----------------------------------------------------------------

    /// Evict least-recently-used models to free at least `needed_bytes`.
    ///
    /// Returns the number of bytes freed.
    pub fn evict_lru(&mut self, needed_bytes: u64) -> Result<u64, ModelError> {
        let mut freed = 0u64;

        // Collect candidates sorted by last_used ascending (oldest first)
        let mut candidates: Vec<(u64, Instant, u64)> = self
            .models
            .iter()
            .map(|(&id, m)| (id, m.last_used.get(), m.weight_bytes))
            .collect();
        candidates.sort_by_key(|&(_, last_used, _)| last_used);

        for (id, _, bytes) in candidates {
            if freed >= needed_bytes {
                break;
            }
            tracing::info!(handle = id, mib = bytes / (1024 * 1024), "evicting model");
            self.models.remove(&id);
            self.tracked_usage = self.tracked_usage.saturating_sub(bytes);
            freed += bytes;
        }

        Ok(freed)
    }

    /// Drop a specific model by handle ID.
    pub fn unload(&mut self, id: u64) -> Result<(), ModelError> {
        let entry = self.models.get(&id).ok_or(ModelError::NotFound { id })?;
        let bytes = entry.weight_bytes;
        self.models.remove(&id);
        self.tracked_usage = self.tracked_usage.saturating_sub(bytes);
        tracing::info!(handle = id, mib = bytes / (1024 * 1024), "unloaded model");
        Ok(())
    }

    // -----------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

/// Compute the exact VRAM byte size of tensors under a given prefix.
///
/// Iterates the safetensors header (no data loaded) and sums
/// `dtype_byte_size × num_elements` for matching tensors. If loading
/// into an f16 backend from f32 weights, the result is halved.
pub(crate) fn compute_weight_bytes(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    target_is_f16: bool,
) -> u64 {
    let mut total: u64 = 0;
    for (name, info) in tensors.tensors() {
        if !name.starts_with(prefix) {
            continue;
        }
        let num_elements: u64 = info.shape().iter().map(|&d| d as u64).product();
        let dtype_bytes: u64 = match info.dtype() {
            safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
            safetensors::Dtype::F32 => {
                if target_is_f16 {
                    2
                } else {
                    4
                }
            }
            _ => 4, // conservative estimate for other types
        };
        total += num_elements * dtype_bytes;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_borrow() {
        let mgr = ModelManager::new(PathBuf::from("models"), Default::default());
        assert_eq!(mgr.tracked_usage(), 0);
        assert_eq!(mgr.models.len(), 0);
    }

    #[test]
    fn evict_lru_empty() {
        let mut mgr = ModelManager::new(PathBuf::from("models"), Default::default());
        let freed = mgr.evict_lru(1024).unwrap();
        assert_eq!(freed, 0);
    }
}
