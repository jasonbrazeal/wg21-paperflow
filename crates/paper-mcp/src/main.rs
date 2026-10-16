//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! The `paper-mcp` command: build the paper index and serve it over MCP.
//!
//! `paper-mcp index` brings `$WG21_DATA_DIR/paperindex/` up to date with
//! paperstore. `paper-mcp serve` exposes the index as MCP tools over
//! Streamable HTTP at `/mcp`, behind `cpp-mcp`-compatible bearer keys.
//! Diagnostics go to stderr; `RUST_LOG` overrides the default log filter.

use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, bail};
use clap::{Parser, Subcommand};
use paper_mcp::auth::{BearerKeyLayer, KeyStore, resolve_keys_file};
use paper_mcp::embed::{
    Embedder, FastEmbedder, HashEmbedder, TEST_EMBEDDER_DIM, TEST_EMBEDDER_ENV, TEST_EMBEDDER_HASH,
    TEST_EMBEDDER_MODEL_ID,
};
use paper_mcp::index::{Indexer, PaperIndex, default_index_dir};
use paper_mcp::tools::PaperTools;
use paperstore_sqlite::SqliteBackend;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use tokio::net::TcpListener;
#[cfg(unix)]
use tokio::signal::unix::{SignalKind, signal};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

/// Log filter used when `RUST_LOG` is unset.
const DEFAULT_LOG_FILTER: &str = "warn,paper_mcp=info";

/// Path the MCP service is mounted at.
const MCP_PATH: &str = "/mcp";

/// Default `serve` bind address.
const DEFAULT_HOST: &str = "127.0.0.1";

/// Default `serve` port, one above `cpp-mcp`'s documented 8001.
const DEFAULT_PORT: u16 = 8002;

#[derive(Debug, Parser)]
#[command(
    name = "paper-mcp",
    version,
    about = "Search and read the WG21 paper corpus over MCP"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Bring the paper index under `$WG21_DATA_DIR` up to date.
    Index,
    /// Serve the paper index as MCP tools over Streamable HTTP at `/mcp`.
    Serve {
        /// Address to bind.
        #[arg(long, default_value = DEFAULT_HOST)]
        host: String,
        /// Port to bind; 0 picks a free port.
        #[arg(long, default_value_t = DEFAULT_PORT)]
        port: u16,
        /// Bearer keys file, one key per line. Defaults to
        /// `$CPP_MCP_KEYS_FILE`, then `/etc/cpp-mcp/keys`. Re-read on
        /// SIGHUP.
        #[arg(long)]
        keys_file: Option<PathBuf>,
        /// Accept unauthenticated requests. Host-header validation stays on,
        /// limiting clients to loopback names.
        #[arg(long)]
        no_auth: bool,
    },
}

fn main() -> anyhow::Result<()> {
    init_logging();
    let cli = Cli::parse();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to start the async runtime")?;
    match cli.command {
        Command::Index => runtime.block_on(index()),
        Command::Serve {
            host,
            port,
            keys_file,
            no_auth,
        } => runtime.block_on(serve(&host, port, keys_file, no_auth)),
    }
}

fn init_logging() {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_FILTER));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .init();
}

/// Select the embedder: the model, or the hash embedder when the hidden
/// test variable asks for it.
fn embedder() -> anyhow::Result<Arc<dyn Embedder>> {
    if std::env::var(TEST_EMBEDDER_ENV).is_ok_and(|v| v == TEST_EMBEDDER_HASH) {
        tracing::warn!("{TEST_EMBEDDER_ENV}={TEST_EMBEDDER_HASH}: using the test hash embedder");
        return Ok(Arc::new(HashEmbedder::new(
            TEST_EMBEDDER_MODEL_ID,
            TEST_EMBEDDER_DIM,
        )));
    }
    Ok(Arc::new(
        FastEmbedder::new(None).context("failed to load the embedding model")?,
    ))
}

async fn index() -> anyhow::Result<()> {
    let backend = SqliteBackend::from_env().context("failed to open paperstore")?;
    let dir = default_index_dir()?;
    let index = PaperIndex::open(&dir).await?;
    let embedder = embedder()?;
    tracing::info!("indexing into {}", dir.display());
    let report = Indexer::new(&backend, embedder.as_ref(), &index)
        .run()
        .await?;
    println!("{report}");
    if !report.is_success() {
        bail!("{} papers failed to index", report.failures.len());
    }
    Ok(())
}

async fn serve(
    host: &str,
    port: u16,
    keys_file: Option<PathBuf>,
    no_auth: bool,
) -> anyhow::Result<()> {
    let backend = SqliteBackend::from_env().context("failed to open paperstore")?;
    let index = PaperIndex::open(&default_index_dir()?).await?;
    let embedder = embedder()?;
    index.check_manifest(embedder.as_ref()).await?;

    let keys = if no_auth {
        tracing::warn!("--no-auth: serving without bearer key authentication");
        None
    } else {
        let path = resolve_keys_file(keys_file.as_deref());
        let keys = KeyStore::open(&path)?;
        if keys.is_empty() {
            bail!("no keys in {}; add a key or pass --no-auth", path.display());
        }
        tracing::info!("loaded {} keys from {}", keys.len(), path.display());
        Some(keys)
    };

    let shutdown = CancellationToken::new();
    // Host-header validation guards unauthenticated loopback servers against
    // DNS rebinding. With bearer keys it is redundant and would reject the
    // public names a reverse proxy forwards.
    let mut config =
        StreamableHttpServerConfig::default().with_cancellation_token(shutdown.child_token());
    if keys.is_some() {
        config = config.disable_allowed_hosts();
    }
    let tools = PaperTools::new(backend, index, embedder);
    let service = StreamableHttpService::new(
        move || Ok(tools.clone()),
        Arc::new(LocalSessionManager::default()),
        config,
    );
    let mut router = axum::Router::new().nest_service(MCP_PATH, service);
    if let Some(keys) = &keys {
        router = router.layer(BearerKeyLayer::new(keys.clone()));
        spawn_key_reloader(keys.clone())?;
    }

    let listener = TcpListener::bind((host, port))
        .await
        .with_context(|| format!("failed to bind {host}:{port}"))?;
    let addr = listener.local_addr()?;
    tracing::info!("listening on http://{addr}{MCP_PATH}");
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            shutdown.cancel();
        })
        .await
        .context("server failed")
}

/// Re-read the keys file on every SIGHUP.
#[cfg(unix)]
fn spawn_key_reloader(keys: KeyStore) -> anyhow::Result<()> {
    let mut hangups = signal(SignalKind::hangup()).context("failed to install SIGHUP handler")?;
    tokio::spawn(async move {
        while hangups.recv().await.is_some() {
            match keys.reload() {
                Ok(count) => {
                    tracing::info!("reloaded {count} keys from {}", keys.path().display());
                }
                Err(e) => tracing::error!("key reload failed, keeping previous keys: {e}"),
            }
        }
    });
    Ok(())
}

#[cfg(not(unix))]
fn spawn_key_reloader(_keys: KeyStore) -> anyhow::Result<()> {
    tracing::warn!("SIGHUP key reload is unavailable on this platform");
    Ok(())
}

/// Resolve on Ctrl-C or, on Unix, SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!("failed to listen for Ctrl-C: {e}");
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                term.recv().await;
            }
            Err(e) => {
                tracing::error!("failed to listen for SIGTERM: {e}");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
    tracing::info!("shutting down");
}
