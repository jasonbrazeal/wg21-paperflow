//! The `paper-mcp` command: `ingest` and `serve`.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use paper_mcp::config::Config;
use paper_mcp::embed::Embedder;
use paper_mcp::ingest::{Ingest, parse_targets};
use paper_mcp::qdrant::Store;
use paper_mcp::serve::{self, Keys};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(version, about = "MCP server for WG21 papers, backed by Qdrant")]
struct Cli {
    /// TOML configuration file. Defaults to $PAPER_MCP_CONFIG when set.
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Copy papers from the paperstore into Qdrant.
    Ingest {
        /// `all`, one or more four-digit years, or one or more paper ids.
        #[arg(required = true)]
        targets: Vec<String>,
        /// Chunk and embed every selected paper even when its chunks are current.
        #[arg(long)]
        force: bool,
    },
    /// Serve the MCP tools.
    Serve,
}

fn embedder(config: &Config) -> Result<Embedder> {
    let cache = config.model_cache_dir()?;
    info!(cache = %cache.display(), "loading embedding model");
    Ok(Embedder::start(
        cache,
        config.model.max_length,
        config.model.batch_tokens,
        config.model.queue_depth,
    )?)
}

async fn ingest(config: &Config, targets: &[String], force: bool) -> Result<bool> {
    let targets = parse_targets(targets)?;
    let data_dir = config.data_dir()?.to_path_buf();
    let store = Store::connect(config)?;
    store.health().await.context("cannot reach Qdrant")?;
    let embedder = embedder(config)?;
    let report = Ingest::new(config, &store, &embedder, force)
        .run(&data_dir, &targets)
        .await?;
    info!(
        indexed = report.indexed,
        skipped = report.skipped,
        failed = report.failed,
        no_markdown = report.no_markdown.len(),
        "ingest finished"
    );
    if !report.no_markdown.is_empty() {
        info!(papers = ?report.no_markdown, "papers without markdown");
    }
    Ok(report.failed == 0)
}

async fn serve(config: &Config) -> Result<()> {
    let keys = Keys::load(config.keys_file()?)?;
    let addr = config.bind_addr()?;
    let store = Store::connect(config)?;
    store.health().await.context("cannot reach Qdrant")?;
    let embedder = embedder(config)?;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("cannot listen on {addr}"))?;
    serve::run(config, store, embedder, keys, listener).await?;
    Ok(())
}

async fn run(cli: Cli) -> Result<bool> {
    let config = Config::load(cli.config.as_deref())?;
    match cli.command {
        Command::Ingest { targets, force } => ingest(&config, &targets, force).await,
        Command::Serve => serve(&config).await.map(|()| true),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,rmcp=warn")),
        )
        .with_writer(std::io::stderr)
        .init();
    match run(Cli::parse()).await {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            error!("{e:#}");
            ExitCode::FAILURE
        }
    }
}
