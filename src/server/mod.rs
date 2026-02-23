//! HTTP server with A1111 and ComfyUI-compatible APIs.
//!
//! The server wraps the node/DAG executor behind HTTP endpoints.
//! GPU access is serialized through a `Mutex<ExecutionContext>` — one
//! job runs at a time.
//!
//! # Endpoints
//!
//! **A1111 API** (`/sdapi/v1/…`):
//! - `POST /sdapi/v1/txt2img` — text-to-image generation
//! - `GET  /sdapi/v1/sd-models` — list available checkpoints
//! - `GET  /sdapi/v1/options` — get server options
//! - `POST /sdapi/v1/options` — set server options
//! - `GET  /sdapi/v1/progress` — current generation progress
//!
//! **ComfyUI API** (`/…`):
//! - `POST /prompt` — submit a workflow for execution
//! - `GET  /ws` — WebSocket for progress events
//! - `GET  /queue` — queue status
//! - `GET  /history/{id}` — prompt execution history

mod a1111;
mod comfyui;
mod error;
mod state;

pub use a1111::types::{OverrideSettings, SdModel, Txt2ImgRequest, Txt2ImgResponse};
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
