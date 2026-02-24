//! Integration test: spin up the server and run a txt2img job.
//!
//! Requires model files on disk and a GPU. Run with:
//!   cargo test --test server_txt2img --features gpu-vulkan-f16 --release -- --ignored --nocapture
//!
//! Uses the SDXL checkpoint by default (override with `HEARTH_TEST_CHECKPOINT` env var).

#![cfg(feature = "gpu-vulkan-f16")]

use base64::{Engine, engine::general_purpose::STANDARD};
use hearth::server::{
    AppState,
    a1111::{OverrideSettings, SdModel, Txt2ImgRequest, Txt2ImgResponse},
};
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

    // --- /internal/ping ---
    let resp = client
        .get(format!("{base_url}/internal/ping"))
        .send()
        .await
        .expect("ping request failed");
    assert_eq!(resp.status(), 200, "ping should return 200");

    // --- /sdapi/v1/sd-models (verify hash fields) ---
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
    for model in &models {
        assert_eq!(
            model.sha256.len(),
            64,
            "sha256 should be 64 hex chars: {}",
            model.model_name
        );
        assert!(
            model.sha256.chars().all(|c| c.is_ascii_hexdigit()),
            "sha256 should be hex: {}",
            model.sha256
        );
        assert_eq!(
            model.hash,
            &model.sha256[..10],
            "hash should be first 10 chars of sha256"
        );
        assert!(
            model.title.contains(&model.hash),
            "title '{}' should contain hash '{}'",
            model.title,
            model.hash
        );
    }

    // --- /sdapi/v1/samplers ---
    let resp = client
        .get(format!("{base_url}/sdapi/v1/samplers"))
        .send()
        .await
        .expect("samplers request failed");
    assert_eq!(resp.status(), 200);
    let samplers: Vec<serde_json::Value> = resp.json().await.expect("parse samplers");
    assert!(samplers.len() >= 4, "expected at least 4 samplers");
    let names: Vec<&str> = samplers.iter().filter_map(|s| s["name"].as_str()).collect();
    assert!(names.contains(&"Euler"), "missing Euler sampler");
    assert!(names.contains(&"Euler a"), "missing Euler a sampler");
    assert!(names.contains(&"DPM++ 2M"), "missing DPM++ 2M sampler");
    assert!(names.contains(&"DPM++ SDE"), "missing DPM++ SDE sampler");
    // Verify structure has aliases and options
    for sampler in &samplers {
        assert!(
            sampler["aliases"].is_array(),
            "sampler should have aliases array"
        );
        assert!(
            sampler["options"].is_object(),
            "sampler should have options object"
        );
    }

    // --- /sdapi/v1/schedulers ---
    let resp = client
        .get(format!("{base_url}/sdapi/v1/schedulers"))
        .send()
        .await
        .expect("schedulers request failed");
    assert_eq!(resp.status(), 200);
    let schedulers: Vec<serde_json::Value> = resp.json().await.expect("parse schedulers");
    assert!(schedulers.len() >= 2, "expected at least 2 schedulers");
    let sched_names: Vec<&str> = schedulers
        .iter()
        .filter_map(|s| s["name"].as_str())
        .collect();
    assert!(sched_names.contains(&"normal"), "missing normal scheduler");
    assert!(sched_names.contains(&"karras"), "missing karras scheduler");
    for sched in &schedulers {
        assert!(sched["label"].is_string(), "scheduler should have label");
        assert!(
            sched["aliases"].is_array(),
            "scheduler should have aliases array"
        );
    }

    // --- /sdapi/v1/progress (verify A1111 nested format) ---
    let resp = client
        .get(format!("{base_url}/sdapi/v1/progress"))
        .send()
        .await
        .expect("progress request failed");
    assert_eq!(resp.status(), 200);
    let progress: serde_json::Value = resp.json().await.expect("parse progress");
    assert!(
        progress["progress"].is_f64(),
        "progress should have progress float"
    );
    assert!(
        progress["eta_relative"].is_f64(),
        "progress should have eta_relative float"
    );
    assert!(
        progress["state"].is_object(),
        "progress should have nested state object"
    );
    let state = &progress["state"];
    assert!(
        state["sampling_step"].is_u64(),
        "state should have sampling_step"
    );
    assert!(
        state["sampling_steps"].is_u64(),
        "state should have sampling_steps"
    );
    assert!(
        state["skipped"].is_boolean(),
        "state should have skipped bool"
    );
    assert!(
        state["interrupted"].is_boolean(),
        "state should have interrupted bool"
    );

    // --- /sdapi/v1/options (verify sd_vae field and partial update) ---
    let resp = client
        .get(format!("{base_url}/sdapi/v1/options"))
        .send()
        .await
        .expect("get options failed");
    assert_eq!(resp.status(), 200);
    let opts: serde_json::Value = resp.json().await.expect("parse options");
    assert_eq!(
        opts["sd_vae"].as_str(),
        Some("Automatic"),
        "default sd_vae should be Automatic"
    );

    // Partial update: set only sd_vae, checkpoint should be unchanged
    let prev_ckpt = opts["sd_model_checkpoint"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let resp = client
        .post(format!("{base_url}/sdapi/v1/options"))
        .json(&serde_json::json!({ "sd_vae": "test_vae.safetensors" }))
        .send()
        .await
        .expect("set options failed");
    assert_eq!(resp.status(), 200);
    let opts: serde_json::Value = resp.json().await.expect("parse set options response");
    assert_eq!(
        opts["sd_vae"].as_str(),
        Some("test_vae.safetensors"),
        "sd_vae should be updated"
    );
    assert_eq!(
        opts["sd_model_checkpoint"].as_str().unwrap_or(""),
        prev_ckpt,
        "sd_model_checkpoint should be unchanged after partial update"
    );

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
        n_iter: 1,
        tiling: false,
        override_settings: Some(OverrideSettings {
            sd_model_checkpoint: Some(ckpt),
        }),
        alwayson_scripts: None,
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
