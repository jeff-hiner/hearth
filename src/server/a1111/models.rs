//! `GET /sdapi/v1/sd-models` — list available checkpoints.

use super::types::SdModel;
use crate::server::AppState;
use axum::{Json, extract::State};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read},
    path::Path,
    sync::Arc,
};

/// List all `.safetensors` files in `models/checkpoints/`.
pub(super) async fn sd_models(State(state): State<Arc<AppState>>) -> Json<Vec<SdModel>> {
    let ckpt_dir = state.models_dir.join("checkpoints");

    let mut models = Vec::new();

    if let Ok(entries) = fs::read_dir(&ckpt_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "safetensors") {
                let filename = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let model_name = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();

                let sha256 = file_sha256_cached(&path);
                let hash = sha256.get(..10).unwrap_or(&sha256).to_string();
                let title = format!("{model_name} [{hash}]");

                models.push(SdModel {
                    title,
                    model_name,
                    filename,
                    hash,
                    sha256,
                    config: None,
                });
            }
        }
    }

    models.sort_by(|a, b| a.title.cmp(&b.title));
    Json(models)
}

/// Compute the SHA-256 of a file, caching the result in a `.sha256` sidecar.
///
/// On cache miss, reads the entire file in 8 KiB chunks and writes the hex
/// digest to `<path>.sha256`. On cache hit, reads the sidecar directly.
fn file_sha256_cached(path: &Path) -> String {
    let sidecar = path.with_extension("sha256");

    // Try reading cached hash.
    if let Ok(cached) = fs::read_to_string(&sidecar) {
        let trimmed = cached.trim();
        if trimmed.len() == 64 && trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
            return trimmed.to_string();
        }
    }

    // Compute hash.
    let hex = match compute_sha256(path) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!(?path, %e, "failed to hash model file");
            return String::new();
        }
    };

    // Best-effort cache write — don't fail the request if this doesn't work.
    if let Err(e) = fs::write(&sidecar, &hex) {
        tracing::debug!(?sidecar, %e, "could not write sha256 sidecar");
    }

    hex
}

fn compute_sha256(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
