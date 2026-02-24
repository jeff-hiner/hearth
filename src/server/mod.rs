//! HTTP server with A1111 and ComfyUI-compatible APIs.
//!
//! The server wraps the node/DAG executor behind HTTP endpoints.
//! GPU access is serialized through a `Mutex<ExecutionContext>` — one
//! job runs at a time.
//!
//! # Endpoints
//!
//! **A1111 API** (`/sdapi/v1/…`):
//! - `GET  /internal/ping` — connection probe
//! - `POST /sdapi/v1/txt2img` — text-to-image generation
//! - `POST /sdapi/v1/img2img` — image-to-image generation
//! - `GET  /sdapi/v1/sd-models` — list available checkpoints
//! - `GET  /sdapi/v1/samplers` — list sampling algorithms
//! - `GET  /sdapi/v1/schedulers` — list sigma schedules
//! - `GET  /sdapi/v1/options` — get server options
//! - `POST /sdapi/v1/options` — set server options (partial update)
//! - `GET  /sdapi/v1/progress` — current generation progress
//! - `POST /sdapi/v1/interrupt` — cancel the active generation
//! - `GET  /sdapi/v1/upscalers` — list available upscalers
//! - `POST /sdapi/v1/extra-batch-images` — upscale images (not implemented)
//! - `GET  /sdapi/v1/sd-vae` — list available VAE models
//! - `POST /sdapi/v1/unload-checkpoint` — unload all models from VRAM
//! - `GET  /internal/sysinfo` — platform and version info
//!
//! **ControlNet extension** (`/controlnet/…`):
//! - `GET  /controlnet/model_list` — list available ControlNet models
//! - `GET  /controlnet/module_list` — list preprocessor modules
//! - `GET  /controlnet/control_types` — control type categories
//! - `GET  /controlnet/settings` — extension settings
//! - `POST /controlnet/detect` — run preprocessor (not implemented)
//!
//! **ComfyUI API** (`/…`):
//! - `POST /prompt` — submit a workflow for execution
//! - `GET  /ws` — WebSocket for progress events
//! - `GET  /queue` — queue status
//! - `GET  /history/{id}` — prompt execution history

pub mod a1111;
mod comfyui;
mod error;
mod state;
use axum::Router;
pub use state::{AppState, ProgressInfo, ServerOptions};
use std::{net::SocketAddr, path::PathBuf, sync::Arc};

/// Build a router serving only the ComfyUI API.
pub fn build_comfyui_router(state: Arc<AppState>) -> Router {
    let cors = tower_http::cors::CorsLayer::permissive();

    Router::new()
        .merge(comfyui::router())
        .layer(cors)
        .with_state(state)
}

/// Build a router serving only the A1111 API.
pub fn build_a1111_router(state: Arc<AppState>) -> Router {
    let cors = tower_http::cors::CorsLayer::permissive();

    Router::new()
        .merge(a1111::router())
        .layer(cors)
        .with_state(state)
}

/// Build the combined router serving both A1111 and ComfyUI APIs.
///
/// Useful for tests that run both APIs on a single ephemeral port.
pub fn build_router(state: Arc<AppState>) -> Router {
    let cors = tower_http::cors::CorsLayer::permissive();

    Router::new()
        .merge(a1111::router())
        .merge(comfyui::router())
        .layer(cors)
        .with_state(state)
}

/// Start the server, binding the ComfyUI and A1111 APIs on separate addresses.
///
/// Blocks until either listener fails or is shut down.
pub async fn run(
    comfyui_addr: SocketAddr,
    a1111_addr: SocketAddr,
    models_dir: PathBuf,
    output_dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = AppState::new(models_dir, output_dir);

    let comfyui_app = build_comfyui_router(Arc::clone(&state));
    let a1111_app = build_a1111_router(state);

    tracing::info!(%comfyui_addr, "starting ComfyUI API server");
    let comfyui_listener = tokio::net::TcpListener::bind(comfyui_addr).await?;

    tracing::info!(%a1111_addr, "starting A1111 API server");
    let a1111_listener = tokio::net::TcpListener::bind(a1111_addr).await?;

    tokio::select! {
        result = axum::serve(comfyui_listener, comfyui_app) => result?,
        result = axum::serve(a1111_listener, a1111_app) => result?,
    }

    Ok(())
}
