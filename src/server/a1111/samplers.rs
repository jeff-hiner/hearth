//! `GET /sdapi/v1/samplers` — list available sampling algorithms.

use super::types::{SamplerItem, SamplerOptions};
use axum::Json;

/// Return the hardcoded list of supported samplers.
pub(super) async fn samplers() -> Json<Vec<SamplerItem>> {
    Json(vec![
        SamplerItem {
            name: "Euler".to_string(),
            aliases: vec!["euler".to_string()],
            options: SamplerOptions {
                second_order: Some(false),
                ..Default::default()
            },
        },
        SamplerItem {
            name: "Euler a".to_string(),
            aliases: vec!["euler_a".to_string()],
            options: SamplerOptions {
                brownian_noise: Some(true),
                ..Default::default()
            },
        },
        SamplerItem {
            name: "DPM++ 2M".to_string(),
            aliases: vec!["dpm++_2m".to_string()],
            options: SamplerOptions {
                second_order: Some(true),
                ..Default::default()
            },
        },
        SamplerItem {
            name: "DPM++ SDE".to_string(),
            aliases: vec!["dpm++_sde".to_string()],
            options: SamplerOptions {
                second_order: Some(true),
                brownian_noise: Some(true),
                ..Default::default()
            },
        },
    ])
}
