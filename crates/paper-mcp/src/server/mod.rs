// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! The MCP server: tools served over authenticated Streamable HTTP.
//!
//! [`build_app`] mounts rmcp's Streamable HTTP service at [`MCP_PATH`]
//! behind the bearer-token middleware of the `auth` module, which guards
//! every route. [`PaperServer`] is the MCP handler, exposing tools only,
//! and [`config`] holds the `serve` settings and their startup validation.

use std::sync::Arc;

use axum::Router;
use axum::middleware::from_fn_with_state;
use rmcp::tool_router;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::error::Error;

mod auth;
pub mod config;

pub use config::{BearerToken, ServeConfig, ServeOptions};

/// Path at which the MCP endpoint is served.
pub const MCP_PATH: &str = "/mcp";

/// The MCP handler, exposing the paper tools and nothing else.
#[derive(Clone, Debug, Default)]
pub struct PaperServer {
    _private: (),
}

impl PaperServer {
    /// Create a server handler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

// The read and search tools land in this block; until then the router is
// legitimately empty. The macro expansion names a bare `Result`, so this
// module does not import the crate's `Result` alias.
#[tool_router(server_handler, allow_empty)]
impl PaperServer {}

/// Build the HTTP app: the MCP endpoint at [`MCP_PATH`], with every route
/// behind the bearer-token check.
///
/// Cancelling `shutdown` ends every MCP session.
pub fn build_app(config: &ServeConfig, server: PaperServer, shutdown: CancellationToken) -> Router {
    let mcp_config = StreamableHttpServerConfig::default()
        .with_allowed_hosts(config.allowed_hosts.iter().cloned())
        .with_cancellation_token(shutdown);
    let mcp = StreamableHttpService::new(
        move || Ok(server.clone()),
        Arc::new(LocalSessionManager::default()),
        mcp_config,
    );
    Router::new()
        .nest_service(MCP_PATH, mcp)
        .layer(from_fn_with_state(
            auth::ExpectedToken::new(&config.token),
            auth::require_bearer,
        ))
}

/// Serve `server` on the configured address until `shutdown` is cancelled.
///
/// # Errors
///
/// Returns [`Error::Bind`] when the address cannot be bound and
/// [`Error::Serve`] when the server fails while running.
pub async fn serve(
    config: &ServeConfig,
    server: PaperServer,
    shutdown: CancellationToken,
) -> crate::error::Result<()> {
    tracing::info!(?config, "starting MCP server");
    let listener = TcpListener::bind(config.bind)
        .await
        .map_err(|source| Error::Bind {
            addr: config.bind,
            source,
        })?;
    tracing::info!(addr = %config.bind, path = MCP_PATH, "listening");
    let app = build_app(config, server, shutdown.clone());
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown.cancelled_owned())
        .await
        .map_err(Error::Serve)
}

/// Cancel `shutdown` on Ctrl-C, for graceful shutdown.
pub async fn ctrl_c(shutdown: CancellationToken) {
    if let Err(e) = tokio::signal::ctrl_c().await {
        tracing::error!(error = %e, "cannot listen for Ctrl-C");
        return;
    }
    tracing::info!("shutting down");
    shutdown.cancel();
}
