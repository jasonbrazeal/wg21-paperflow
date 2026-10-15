// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! The MCP server: tools served over authenticated Streamable HTTP.
//!
//! [`build_app`] mounts rmcp's Streamable HTTP service at [`MCP_PATH`]
//! behind the bearer-token middleware of the `auth` module, which guards
//! every route. [`PaperServer`] is the MCP handler, exposing tools only,
//! [`tools`] defines those tools, and [`config`] holds the `serve` settings
//! and their startup validation.

use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use axum::Router;
use axum::middleware::from_fn_with_state;
use paperstore::StorageBackend;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::error::Error;

mod auth;
pub mod config;
pub mod tools;

pub use config::{BearerToken, ServeConfig, ServeOptions};

/// Path at which the MCP endpoint is served.
pub const MCP_PATH: &str = "/mcp";

/// The paperstore backend shared by every session's handler.
type SharedStore = Arc<Mutex<Box<dyn StorageBackend + Send>>>;

/// The MCP handler, exposing the paper tools and nothing else.
///
/// Clones share one paperstore backend, which serves one tool call at a time.
#[derive(Clone)]
pub struct PaperServer {
    store: SharedStore,
}

impl fmt::Debug for PaperServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PaperServer").finish_non_exhaustive()
    }
}

impl PaperServer {
    /// Create a handler whose tools read `store`.
    #[must_use]
    pub fn new(store: impl StorageBackend + Send + 'static) -> Self {
        Self {
            store: Arc::new(Mutex::new(Box::new(store))),
        }
    }

    /// Run `op` against the store on a blocking thread, turning its error
    /// into the message of an MCP tool error.
    async fn read<T, F>(&self, op: F) -> std::result::Result<T, String>
    where
        T: Send + 'static,
        F: FnOnce(&dyn StorageBackend) -> crate::error::Result<T> + Send + 'static,
    {
        let store = Arc::clone(&self.store);
        let joined = tokio::task::spawn_blocking(move || {
            let guard = store.lock().unwrap_or_else(PoisonError::into_inner);
            op(guard.as_ref())
        })
        .await;
        match joined {
            Ok(result) => result.map_err(|e| e.to_string()),
            Err(e) => {
                tracing::error!(error = %e, "tool task failed");
                Err("internal error: the tool task failed".to_owned())
            }
        }
    }
}

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
