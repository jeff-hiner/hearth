//! `GET /internal/sysinfo` — basic system/version info.

use axum::Json;
use serde::Serialize;

/// Return minimal system info (platform name and version).
pub(super) async fn sysinfo() -> Json<SysInfo> {
    Json(SysInfo {
        platform: "Hearth".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[derive(Serialize)]
pub(super) struct SysInfo {
    #[serde(rename = "Platform")]
    platform: String,
    #[serde(rename = "Version")]
    version: String,
}
