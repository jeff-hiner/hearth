//! `GET /ws` — WebSocket endpoint for ComfyUI progress events.

use super::types::{WsExecInfo, WsMessage, WsProgressData, WsQueueRemaining, WsStatusData};
use crate::server::AppState;
use axum::{
    extract::{
        State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    response::Response,
};
use std::sync::Arc;

/// Upgrade to WebSocket for progress events.
pub(super) async fn ws_handler(
    State(state): State<Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle an active WebSocket connection.
///
/// Sends progress updates as JSON messages matching the ComfyUI protocol.
async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    tracing::info!("WebSocket client connected");

    // Send initial status
    let status = WsMessage {
        msg_type: "status".to_string(),
        data: WsStatusData {
            status: WsExecInfo {
                exec_info: WsQueueRemaining { queue_remaining: 0 },
            },
        },
    };
    let status_json = serde_json::to_string(&status).expect("WsMessage is always serializable");
    if socket
        .send(Message::Text(status_json.into()))
        .await
        .is_err()
    {
        return;
    }

    // Watch for progress updates
    let mut rx = state.progress_rx.clone();
    loop {
        if rx.changed().await.is_err() {
            break;
        }
        let info = rx.borrow_and_update().clone();
        let msg = WsMessage {
            msg_type: "progress".to_string(),
            data: WsProgressData {
                value: info.current_step,
                max: info.total_steps,
            },
        };
        let msg_json = serde_json::to_string(&msg).expect("WsMessage is always serializable");
        if socket.send(Message::Text(msg_json.into())).await.is_err() {
            break;
        }
    }

    tracing::info!("WebSocket client disconnected");
}
