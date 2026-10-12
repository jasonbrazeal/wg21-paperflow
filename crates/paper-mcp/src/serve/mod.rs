//! The serve command (§9): MCP over stateless Streamable HTTP, behind
//! bearer-token auth, answering every call from Qdrant.

pub mod auth;
pub mod error;
pub mod tools;

use std::sync::Arc;

use axum::Router;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use tokio::net::TcpListener;
use tokio::signal::unix::{SignalKind, signal};
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::config::Config;
use crate::embed::Embedder;
use crate::qdrant::Store;

pub use auth::Keys;
pub use error::ToolError;
pub use tools::PaperServer;

/// A failure that stops the server.
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    /// Listening or serving failed.
    #[error("server I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Builds the HTTP router: the MCP endpoint at `path`, behind auth.
pub fn router(
    config: &Config,
    server: PaperServer,
    keys: Arc<Keys>,
    shutdown: CancellationToken,
) -> Router {
    let mut http = StreamableHttpServerConfig::default();
    http.legacy_session_mode = false;
    http.json_response = true;
    http.allowed_hosts.clone_from(&config.serve.allowed_hosts);
    http.cancellation_token = shutdown;
    let service: StreamableHttpService<PaperServer, LocalSessionManager> =
        StreamableHttpService::new(move || Ok(server.clone()), Arc::default(), http);
    Router::new()
        .nest_service(&config.serve.path, service)
        .layer(axum::middleware::from_fn_with_state(
            keys,
            auth::require_bearer,
        ))
}

/// Reloads the keys on every SIGHUP until shutdown.
fn spawn_reloader(keys: Arc<Keys>, shutdown: CancellationToken) -> Result<(), ServeError> {
    let mut hangups = signal(SignalKind::hangup())?;
    tokio::spawn(async move {
        loop {
            tokio::select! {
                () = shutdown.cancelled() => break,
                received = hangups.recv() => {
                    if received.is_none() {
                        break;
                    }
                    keys.reload();
                }
            }
        }
    });
    Ok(())
}

/// Serves until SIGTERM or SIGINT, then lets in-flight requests finish.
///
/// # Errors
///
/// Returns [`ServeError`] when the listener cannot be bound or serving
/// fails.
pub async fn run(
    config: &Config,
    store: Store,
    embedder: Embedder,
    keys: Arc<Keys>,
    listener: TcpListener,
) -> Result<(), ServeError> {
    let shutdown = CancellationToken::new();
    spawn_reloader(Arc::clone(&keys), shutdown.clone())?;
    let server = PaperServer::new(store, embedder, config.search);
    // The MCP service gets a child token so its open streams close only
    // after the HTTP server has stopped accepting connections.
    let app = router(config, server, keys, shutdown.child_token());
    let mut terminate = signal(SignalKind::terminate())?;
    let mut interrupt = signal(SignalKind::interrupt())?;
    let stop = shutdown.clone();
    info!(addr = %listener.local_addr()?, path = config.serve.path.as_str(), "serving");
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            tokio::select! {
                _ = terminate.recv() => {}
                _ = interrupt.recv() => {}
                () = stop.cancelled() => {}
            }
            info!("shutting down");
        })
        .await?;
    shutdown.cancel();
    Ok(())
}
