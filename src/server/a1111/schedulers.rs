//! `GET /sdapi/v1/schedulers` — list available sigma schedules.

use super::types::SchedulerItem;
use axum::Json;

/// Return the hardcoded list of supported schedulers.
pub(super) async fn schedulers() -> Json<Vec<SchedulerItem>> {
    Json(vec![
        SchedulerItem {
            name: "normal".to_string(),
            label: "Normal".to_string(),
            aliases: vec![],
        },
        SchedulerItem {
            name: "karras".to_string(),
            label: "Karras".to_string(),
            aliases: vec![],
        },
    ])
}
