//! API error types mapped to HTTP responses.

use crate::node::error::NodeError;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

/// API error that maps to an HTTP response.
#[derive(Debug)]
pub(super) enum ApiError {
    /// A node execution error.
    NodeError(NodeError),
    /// A generic internal error.
    Internal(String),
    /// Bad request (invalid parameters).
    BadRequest(String),
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeError(e) => write!(f, "node error: {e}"),
            Self::Internal(msg) => write!(f, "internal error: {msg}"),
            Self::BadRequest(msg) => write!(f, "bad request: {msg}"),
        }
    }
}

impl std::error::Error for ApiError {}

/// JSON body returned by error responses.
#[derive(Debug, Serialize)]
struct ApiErrorBody {
    /// Short error description.
    error: String,
    /// Detailed error message.
    detail: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            Self::NodeError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            Self::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            Self::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
        };

        tracing::error!(%status, %message, "API error");

        let body = ApiErrorBody {
            error: message.clone(),
            detail: message,
        };

        (status, axum::Json(body)).into_response()
    }
}

impl From<NodeError> for ApiError {
    fn from(e: NodeError) -> Self {
        Self::NodeError(e)
    }
}
