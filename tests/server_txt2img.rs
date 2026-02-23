//! Integration test: spin up the server and run a txt2img job.
//!
//! Requires model files on disk and a GPU. Run with:
//!   cargo test --test server_txt2img --features gpu-vulkan-f16 --release -- --ignored --nocapture
//!
//! Uses the SDXL checkpoint by default (override with `HEARTH_TEST_CHECKPOINT` env var).

#![cfg(feature = "gpu-vulkan-f16")]

use base64::{Engine, engine::general_purpose::STANDARD};
use hearth::server::{AppState, OverrideSettings, SdModel, Txt2ImgRequest, Txt2ImgResponse};
use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};
use tokio::{net::TcpListener, task::JoinHandle};

/// Default SDXL checkpoint filename.
const DEFAULT_CHECKPOINT: &str = "sd_xl_base_1.0.safetensors";

/// Spin up the server on an ephemeral port and return its base URL.
async fn start_server(state: Arc<AppState>) -> (String, JoinHandle<()>) {
    let app = hearth::server::build_router(state);

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr: SocketAddr = listener.local_addr().expect("local addr");
    let base_url = format!("http://{addr}");

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.expect("server crashed");
    });

    base_url_ready(&base_url).await;
    (base_url, handle)
}

/// Wait until the server is accepting connections.
async fn base_url_ready(base_url: &str) {
    let client = reqwest::Client::new();
    for _ in 0..50 {
        if client
            .get(format!("{base_url}/sdapi/v1/progress"))
            .send()
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server did not become ready within 2.5s");
}

/// Full integration test: POST txt2img and verify we get back a base64 PNG.
#[tokio::test]
#[ignore = "requires SDXL checkpoint and GPU"]
async fn txt2img_returns_base64_png() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init()
        .ok();

    let ckpt =
        std::env::var("HEARTH_TEST_CHECKPOINT").unwrap_or_else(|_| DEFAULT_CHECKPOINT.to_string());

    // Verify checkpoint exists before spinning up the server
    let ckpt_path = PathBuf::from("models/checkpoints").join(&ckpt);
    assert!(
        ckpt_path.exists(),
        "checkpoint not found: {}",
        ckpt_path.display()
    );

    let models_dir = PathBuf::from("models");
    let output_dir = PathBuf::from("output/test_server");
    tokio::fs::create_dir_all(&output_dir)
        .await
        .expect("create output dir");

    let state = AppState::new(models_dir, output_dir);
    let (base_url, server_handle) = start_server(state).await;

    let client = reqwest::Client::new();

    // Verify non-generation endpoints work first
    let resp = client
        .get(format!("{base_url}/sdapi/v1/sd-models"))
        .send()
        .await
        .expect("sd-models request failed");
    assert_eq!(resp.status(), 200);
    let models: Vec<SdModel> = resp.json().await.expect("parse sd-models");
    tracing::info!(count = models.len(), "found checkpoints");
    assert!(
        !models.is_empty(),
        "no checkpoints found in models/checkpoints/"
    );

    let resp = client
        .get(format!("{base_url}/sdapi/v1/progress"))
        .send()
        .await
        .expect("progress request failed");
    assert_eq!(resp.status(), 200);

    // Pick resolution based on model: SDXL is tuned for 1024x1024, SD 1.5 for 512x512
    let is_sdxl = ckpt.contains("xl");
    let (width, height) = if is_sdxl { (1024, 1024) } else { (512, 512) };

    // POST txt2img — 4 steps for speed
    let body = Txt2ImgRequest {
        prompt: "a photograph of a cat".to_string(),
        negative_prompt: "blurry, bad quality".to_string(),
        width,
        height,
        steps: 4,
        cfg_scale: 7.0,
        sampler_name: "euler".to_string(),
        scheduler: "normal".to_string(),
        seed: 42,
        denoising_strength: 1.0,
        batch_size: 1,
        override_settings: Some(OverrideSettings {
            sd_model_checkpoint: Some(ckpt),
        }),
    };

    tracing::info!("sending txt2img request (this will take a while)...");
    let resp = client
        .post(format!("{base_url}/sdapi/v1/txt2img"))
        .json(&body)
        .send()
        .await
        .expect("txt2img request failed");

    assert_eq!(
        resp.status(),
        200,
        "txt2img returned non-200: {}",
        resp.text().await.unwrap_or_default()
    );

    let result: Txt2ImgResponse = resp.json().await.expect("parse txt2img response");

    // Verify response structure
    assert_eq!(result.images.len(), 1, "expected 1 image");

    let b64 = &result.images[0];
    assert!(!b64.is_empty(), "base64 image is empty");

    // Decode base64 and verify it's a valid PNG
    let png_bytes = STANDARD.decode(b64).expect("invalid base64");
    assert!(
        png_bytes.starts_with(&[0x89, b'P', b'N', b'G']),
        "decoded bytes are not a PNG (header: {:?})",
        &png_bytes[..4.min(png_bytes.len())]
    );

    // Verify info field
    assert!(!result.info.is_empty(), "response missing 'info' string");

    tracing::info!(
        png_size = png_bytes.len(),
        "txt2img succeeded — received valid PNG"
    );

    // Clean up
    server_handle.abort();
}
