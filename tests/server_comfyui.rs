//! Integration test: spin up the server and exercise the ComfyUI API endpoints.
//!
//! Requires model files on disk and a GPU. Run with:
//!   cargo test --test server_comfyui --features gpu-vulkan-f16 --release -- --ignored --nocapture

#![cfg(feature = "gpu-vulkan-f16")]

use hearth::server::AppState;
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

/// Wait until the server is accepting connections (polls the ComfyUI `/queue` endpoint).
async fn base_url_ready(base_url: &str) {
    let client = reqwest::Client::new();
    for _ in 0..50 {
        if client.get(format!("{base_url}/queue")).send().await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server did not become ready within 2.5s");
}

/// Full integration test: POST /prompt with a txt2img workflow and verify a PNG is saved to disk.
#[tokio::test]
#[ignore = "requires SDXL checkpoint and GPU"]
async fn comfyui_txt2img_basic() {
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
    let output_dir = PathBuf::from("output/test_comfyui");
    tokio::fs::create_dir_all(&output_dir)
        .await
        .expect("create output dir");

    let state = AppState::new(models_dir, output_dir);
    let (base_url, server_handle) = start_server(state).await;

    let client = reqwest::Client::new();

    // Load the fixture workflow
    let fixture = std::fs::read_to_string("tests/fixtures/comfyui/input/txt2img_basic.json")
        .expect("read fixture JSON");
    let body: serde_json::Value = serde_json::from_str(&fixture).expect("parse fixture JSON");

    tracing::info!("sending ComfyUI /prompt request (this will take a while)...");
    let resp = client
        .post(format!("{base_url}/prompt"))
        .json(&body)
        .send()
        .await
        .expect("/prompt request failed");

    assert_eq!(
        resp.status(),
        200,
        "/prompt returned non-200: {}",
        resp.text().await.unwrap_or_default()
    );

    let result: serde_json::Value = resp.json().await.expect("parse /prompt response");
    let prompt_id = result["prompt_id"]
        .as_str()
        .expect("response missing prompt_id");
    assert!(!prompt_id.is_empty(), "prompt_id is empty");
    tracing::info!(prompt_id, "prompt accepted");

    // Verify the output file was written
    let output_path = PathBuf::from("output/test_comfyui/comfyui_test_00000.png");
    assert!(
        output_path.exists(),
        "output image not found: {}",
        output_path.display()
    );

    // Verify PNG magic bytes
    let png_bytes = std::fs::read(&output_path).expect("read output PNG");
    assert!(
        png_bytes.starts_with(&[0x89, b'P', b'N', b'G']),
        "output file is not a valid PNG (header: {:?})",
        &png_bytes[..4.min(png_bytes.len())]
    );

    tracing::info!(
        png_size = png_bytes.len(),
        "comfyui txt2img succeeded — valid PNG on disk"
    );

    server_handle.abort();
}

/// Verify the stub queue and history endpoints return valid JSON.
#[tokio::test]
#[ignore = "requires gpu-vulkan-f16 feature for AppState"]
async fn comfyui_queue_and_history() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init()
        .ok();

    let models_dir = PathBuf::from("models");
    let output_dir = PathBuf::from("output/test_comfyui_queue");
    tokio::fs::create_dir_all(&output_dir)
        .await
        .expect("create output dir");

    let state = AppState::new(models_dir, output_dir);
    let (base_url, server_handle) = start_server(state).await;

    let client = reqwest::Client::new();

    // GET /queue — should return queue_pending and queue_running
    let resp = client
        .get(format!("{base_url}/queue"))
        .send()
        .await
        .expect("/queue request failed");
    assert_eq!(resp.status(), 200);
    let queue: serde_json::Value = resp.json().await.expect("parse /queue response");
    assert!(
        queue.get("queue_pending").is_some(),
        "response missing queue_pending"
    );
    assert!(
        queue.get("queue_running").is_some(),
        "response missing queue_running"
    );

    // GET /history/nonexistent — should return empty object
    let resp = client
        .get(format!("{base_url}/history/nonexistent"))
        .send()
        .await
        .expect("/history request failed");
    assert_eq!(resp.status(), 200);
    let history: serde_json::Value = resp.json().await.expect("parse /history response");
    assert!(
        history.as_object().is_some(),
        "history response is not a JSON object"
    );

    tracing::info!("queue and history endpoints OK");

    server_handle.abort();
}
