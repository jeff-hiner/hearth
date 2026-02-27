//! `GET /sdapi/v1/samplers` — list available sampling algorithms.

use super::types::{SamplerItem, SamplerOptions};
use crate::sampling::SamplerKind;
use axum::Json;
use strum::VariantArray;

/// Return the list of supported samplers, derived from [`SamplerKind`].
pub(super) async fn samplers() -> Json<Vec<SamplerItem>> {
    Json(
        SamplerKind::VARIANTS
            .iter()
            .map(|k| SamplerItem {
                name: k.to_string(),
                aliases: vec![k.to_string().to_lowercase().replace(' ', "_")],
                options: sampler_options(k),
            })
            .collect(),
    )
}

/// Per-variant sampler options matching A1111 format.
fn sampler_options(kind: &SamplerKind) -> SamplerOptions {
    match kind {
        SamplerKind::Euler => SamplerOptions {
            second_order: Some(false),
            ..Default::default()
        },
        SamplerKind::EulerA => SamplerOptions {
            brownian_noise: Some(true),
            ..Default::default()
        },
        SamplerKind::DpmPp2m => SamplerOptions {
            second_order: Some(true),
            ..Default::default()
        },
        SamplerKind::DpmPpSde => SamplerOptions {
            second_order: Some(true),
            brownian_noise: Some(true),
            ..Default::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn samplers_endpoint_snapshot() {
        let Json(items) = samplers().await;
        let json = serde_json::to_value(&items).expect("serialize");
        let expected = serde_json::json!([
            {"name":"Euler","aliases":["euler"],"options":{"second_order":false}},
            {"name":"Euler a","aliases":["euler_a"],"options":{"brownian_noise":true}},
            {"name":"DPM++ 2M","aliases":["dpm++_2m"],"options":{"second_order":true}},
            {"name":"DPM++ SDE","aliases":["dpm++_sde"],"options":{"second_order":true,"brownian_noise":true}}
        ]);
        assert_eq!(json, expected);
    }
}
