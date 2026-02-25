//! Integration tests for A1111 API endpoints that don't require a GPU or checkpoint.
//!
//! These spin up the server with a temporary models directory and verify
//! response shapes for all informational / stub endpoints.
//!
//! Run with:
//!   cargo test --test server_endpoints

use hearth::server::AppState;
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::{net::TcpListener, task::JoinHandle};

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

/// Spin up a server with a temporary models directory (no real models needed).
async fn setup() -> (String, JoinHandle<()>, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let models_dir = tmp.path().join("models");
    let output_dir = tmp.path().join("output");

    // Create subdirectories the endpoints scan
    std::fs::create_dir_all(models_dir.join("checkpoints")).expect("create checkpoints dir");
    std::fs::create_dir_all(models_dir.join("controlnet")).expect("create controlnet dir");
    std::fs::create_dir_all(models_dir.join("vae")).expect("create vae dir");
    std::fs::create_dir_all(&output_dir).expect("create output dir");

    let state = AppState::new(models_dir, output_dir);
    let (base_url, handle) = start_server(state).await;
    (base_url, handle, tmp)
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

#[tokio::test]
async fn ping() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/internal/ping"))
        .send()
        .await
        .expect("ping request failed");
    assert_eq!(resp.status(), 200);

    handle.abort();
}

#[tokio::test]
async fn sysinfo() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/internal/sysinfo"))
        .send()
        .await
        .expect("sysinfo request failed");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("parse sysinfo");
    assert_eq!(body["Platform"].as_str(), Some("Hearth"));
    assert!(body["Version"].is_string(), "Version should be a string");

    handle.abort();
}

#[tokio::test]
async fn samplers() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/sdapi/v1/samplers"))
        .send()
        .await
        .expect("samplers request failed");
    assert_eq!(resp.status(), 200);
    let samplers: Vec<serde_json::Value> = resp.json().await.expect("parse samplers");
    assert!(samplers.len() >= 4, "expected at least 4 samplers");

    let names: Vec<&str> = samplers.iter().filter_map(|s| s["name"].as_str()).collect();
    assert!(names.contains(&"Euler"), "missing Euler");
    assert!(names.contains(&"Euler a"), "missing Euler a");
    assert!(names.contains(&"DPM++ 2M"), "missing DPM++ 2M");
    assert!(names.contains(&"DPM++ SDE"), "missing DPM++ SDE");

    for sampler in &samplers {
        assert!(sampler["aliases"].is_array(), "should have aliases array");
        assert!(sampler["options"].is_object(), "should have options object");
    }

    handle.abort();
}

#[tokio::test]
async fn schedulers() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/sdapi/v1/schedulers"))
        .send()
        .await
        .expect("schedulers request failed");
    assert_eq!(resp.status(), 200);
    let schedulers: Vec<serde_json::Value> = resp.json().await.expect("parse schedulers");
    assert!(schedulers.len() >= 2, "expected at least 2 schedulers");

    let names: Vec<&str> = schedulers
        .iter()
        .filter_map(|s| s["name"].as_str())
        .collect();
    assert!(names.contains(&"normal"), "missing normal");
    assert!(names.contains(&"karras"), "missing karras");

    for sched in &schedulers {
        assert!(sched["label"].is_string(), "should have label");
        assert!(sched["aliases"].is_array(), "should have aliases array");
    }

    handle.abort();
}

#[tokio::test]
async fn progress() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/sdapi/v1/progress"))
        .send()
        .await
        .expect("progress request failed");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("parse progress");

    assert!(body["progress"].is_f64(), "should have progress float");
    assert!(body["eta_relative"].is_f64(), "should have eta_relative");
    assert!(body["state"].is_object(), "should have nested state object");

    let state = &body["state"];
    assert!(state["sampling_step"].is_u64(), "should have sampling_step");
    assert!(
        state["sampling_steps"].is_u64(),
        "should have sampling_steps"
    );
    assert!(state["skipped"].is_boolean(), "should have skipped bool");
    assert!(
        state["interrupted"].is_boolean(),
        "should have interrupted bool"
    );

    handle.abort();
}

#[tokio::test]
async fn options_get_and_partial_update() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    // GET default options
    let resp = client
        .get(format!("{base_url}/sdapi/v1/options"))
        .send()
        .await
        .expect("get options failed");
    assert_eq!(resp.status(), 200);
    let opts: serde_json::Value = resp.json().await.expect("parse options");
    assert_eq!(opts["sd_vae"].as_str(), Some("Automatic"));

    // Partial update: change sd_vae only
    let prev_ckpt = opts["sd_model_checkpoint"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let resp = client
        .post(format!("{base_url}/sdapi/v1/options"))
        .json(&serde_json::json!({ "sd_vae": "test.safetensors" }))
        .send()
        .await
        .expect("set options failed");
    assert_eq!(resp.status(), 200);
    let opts: serde_json::Value = resp.json().await.expect("parse set options");
    assert_eq!(opts["sd_vae"].as_str(), Some("test.safetensors"));
    assert_eq!(
        opts["sd_model_checkpoint"].as_str().unwrap_or(""),
        prev_ckpt,
        "checkpoint should be unchanged after partial update"
    );

    handle.abort();
}

#[tokio::test]
async fn upscalers() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/sdapi/v1/upscalers"))
        .send()
        .await
        .expect("upscalers request failed");
    assert_eq!(resp.status(), 200);
    let upscalers: Vec<serde_json::Value> = resp.json().await.expect("parse upscalers");
    assert!(!upscalers.is_empty(), "should have at least one upscaler");
    assert_eq!(upscalers[0]["name"].as_str(), Some("None"));
    assert_eq!(upscalers[0]["scale"].as_u64(), Some(1));

    handle.abort();
}

#[tokio::test]
async fn extra_batch_images_returns_501() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base_url}/sdapi/v1/extra-batch-images"))
        .json(&serde_json::json!({}))
        .send()
        .await
        .expect("extra-batch-images request failed");
    assert_eq!(resp.status(), 501);
    let body: serde_json::Value = resp.json().await.expect("parse body");
    assert_eq!(body["error"].as_str(), Some("not_implemented"));

    handle.abort();
}

#[tokio::test]
async fn sd_vae_empty_dir() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/sdapi/v1/sd-vae"))
        .send()
        .await
        .expect("sd-vae request failed");
    assert_eq!(resp.status(), 200);
    let vaes: Vec<serde_json::Value> = resp.json().await.expect("parse sd-vae");
    assert!(vaes.is_empty(), "empty vae dir should return empty list");

    handle.abort();
}

#[tokio::test]
async fn sd_models_empty_dir() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/sdapi/v1/sd-models"))
        .send()
        .await
        .expect("sd-models request failed");
    assert_eq!(resp.status(), 200);
    let models: Vec<serde_json::Value> = resp.json().await.expect("parse sd-models");
    assert!(
        models.is_empty(),
        "empty checkpoints dir should return empty list"
    );

    handle.abort();
}

#[tokio::test]
async fn interrupt() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base_url}/sdapi/v1/interrupt"))
        .send()
        .await
        .expect("interrupt request failed");
    assert_eq!(resp.status(), 200);

    handle.abort();
}

#[tokio::test]
async fn unload_checkpoint() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base_url}/sdapi/v1/unload-checkpoint"))
        .send()
        .await
        .expect("unload-checkpoint request failed");
    assert_eq!(resp.status(), 200);

    handle.abort();
}

#[tokio::test]
async fn controlnet_model_list() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/controlnet/model_list"))
        .send()
        .await
        .expect("model_list request failed");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("parse model_list");
    assert!(body["model_list"].is_array());
    assert!(
        body["model_list"].as_array().unwrap().is_empty(),
        "empty controlnet dir should return empty list"
    );

    handle.abort();
}

#[tokio::test]
async fn controlnet_module_list() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/controlnet/module_list"))
        .send()
        .await
        .expect("module_list request failed");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("parse module_list");
    let modules = body["module_list"].as_array().expect("should be array");
    assert!(modules.iter().any(|m| m.as_str() == Some("none")));

    handle.abort();
}

#[tokio::test]
async fn controlnet_control_types() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/controlnet/control_types"))
        .send()
        .await
        .expect("control_types request failed");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("parse control_types");
    assert!(body["control_types"].is_object());

    handle.abort();
}

#[tokio::test]
async fn controlnet_settings() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base_url}/controlnet/settings"))
        .send()
        .await
        .expect("settings request failed");
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.expect("parse settings");
    assert!(body["control_net_unit_count"].as_u64().unwrap_or(0) >= 1);

    handle.abort();
}

#[tokio::test]
async fn controlnet_detect_returns_501() {
    let (base_url, handle, _tmp) = setup().await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{base_url}/controlnet/detect"))
        .json(&serde_json::json!({}))
        .send()
        .await
        .expect("detect request failed");
    assert_eq!(resp.status(), 501);
    let body: serde_json::Value = resp.json().await.expect("parse detect");
    assert_eq!(body["error"].as_str(), Some("not_implemented"));

    handle.abort();
}
