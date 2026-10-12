//! Tool errors and their codes (§6.10, §10.3).

use rmcp::handler::server::tool::IntoCallToolResult;
use rmcp::model::{CallToolResponse, CallToolResult};
use serde_json::json;
use tracing::error;
use uuid::Uuid;

use crate::embed::EmbedError;
use crate::paper_id::InvalidPaperId;
use crate::qdrant::StoreError;

/// A tool failure. Each variant maps to one error code.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// The argument is not a well-formed paper id.
    #[error("not a paper id: {0:?}")]
    InvalidPaperId(String),
    /// A well-formed id with no record.
    #[error("paper {0} is not indexed")]
    PaperNotIndexed(String),
    /// The paper has no converted markdown.
    #[error("paper {0} has no converted markdown")]
    MarkdownNotConverted(String),
    /// A missing or malformed argument.
    #[error("{0}")]
    InvalidArgument(String),
    /// An unexpected failure. Its details are logged, never returned.
    #[error("internal error")]
    Internal(#[source] Box<dyn std::error::Error + Send + Sync>),
}

impl ToolError {
    /// The error code returned to the caller.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            ToolError::InvalidPaperId(_) => "invalid_paper_id",
            ToolError::PaperNotIndexed(_) => "paper_not_indexed",
            ToolError::MarkdownNotConverted(_) => "markdown_not_converted",
            ToolError::InvalidArgument(_) => "invalid_argument",
            ToolError::Internal(_) => "internal_error",
        }
    }
}

impl From<InvalidPaperId> for ToolError {
    fn from(error: InvalidPaperId) -> Self {
        ToolError::InvalidPaperId(error.0)
    }
}

impl From<StoreError> for ToolError {
    fn from(error: StoreError) -> Self {
        ToolError::Internal(Box::new(error))
    }
}

impl From<EmbedError> for ToolError {
    fn from(error: EmbedError) -> Self {
        ToolError::Internal(Box::new(error))
    }
}

fn chain(error: &dyn std::error::Error) -> String {
    let mut out = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        out.push_str(": ");
        out.push_str(&cause.to_string());
        source = cause.source();
    }
    out
}

impl IntoCallToolResult for ToolError {
    fn into_call_tool_result(self) -> Result<CallToolResponse, rmcp::ErrorData> {
        let body = if let ToolError::Internal(cause) = &self {
            let request_id = Uuid::new_v4().to_string();
            error!(
                request_id = request_id.as_str(),
                error = chain(cause.as_ref()),
                "internal error"
            );
            json!({
                "code": self.code(),
                "message": format!("internal error; request id {request_id}"),
                "request_id": request_id,
            })
        } else {
            json!({ "code": self.code(), "message": self.to_string() })
        };
        Ok(CallToolResult::structured_error(body).into())
    }
}
