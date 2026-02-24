//! Hearth — Diffusion Model Inference Server

use std::{net::SocketAddr, path::PathBuf};

#[tokio::main]
async fn main() -> std::io::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let comfyui_addr: SocketAddr = "127.0.0.1:8188".parse().unwrap();
    let a1111_addr: SocketAddr = "127.0.0.1:7860".parse().unwrap();
    let models_dir = PathBuf::from("models");
    let output_dir = PathBuf::from("output");

    // Ensure output directory exists
    std::fs::create_dir_all(&output_dir)?;

    tracing::info!("Hearth - Diffusion Model Inference Server");
    tracing::info!(models_dir = %models_dir.display(), "models directory");
    tracing::info!(output_dir = %output_dir.display(), "output directory");

    hearth::server::run(comfyui_addr, a1111_addr, models_dir, output_dir).await
}
