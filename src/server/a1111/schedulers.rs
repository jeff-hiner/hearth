//! `GET /sdapi/v1/schedulers` — list available sigma schedules.

use super::types::SchedulerItem;
use crate::sampling::SchedulerKind;
use axum::Json;
use strum::VariantArray;

/// Return the list of supported schedulers, derived from [`SchedulerKind`].
pub(super) async fn schedulers() -> Json<Vec<SchedulerItem>> {
    Json(
        SchedulerKind::VARIANTS
            .iter()
            .map(|k| SchedulerItem {
                name: k.to_string().to_lowercase(),
                label: k.to_string(),
                aliases: vec![],
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn schedulers_endpoint_snapshot() {
        let Json(items) = schedulers().await;
        let json = serde_json::to_value(&items).expect("serialize");
        let expected = serde_json::json!([
            {"name":"normal","label":"Normal","aliases":[]},
            {"name":"karras","label":"Karras","aliases":[]}
        ]);
        assert_eq!(json, expected);
    }
}
