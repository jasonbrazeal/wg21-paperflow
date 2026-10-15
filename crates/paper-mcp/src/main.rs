// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! The `paper-mcp` command: index WG21 papers and serve search over MCP.

use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use clap::{Parser, Subcommand};
use paper_mcp::server::config::{DEFAULT_BIND, default_allowed_hosts};
use paper_mcp::{
    DEFAULT_CURSOR_CAPACITY, DEFAULT_CURSOR_TTL, DEFAULT_PAGE_SIZE, DEFAULT_RERANK_DEPTH,
    DEFAULT_TOP_K, Embedder as _, IndexLock, IndexOptions, IndexReport, LanceStore, OnnxEmbedder,
    OnnxReranker, PaperServer, Reranker as _, ServeConfig, ServeOptions, TOOL_NAME,
};
use paperstore::{StorageBackend as _, default_workspace_dir};
use paperstore_sqlite::SqliteBackend;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

/// Name of the paperstore database inside the workspace directory.
const PAPERSTORE_DB: &str = "paperstore.db";

/// Index WG21 papers held in paperstore and serve search over MCP.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Index papers: one or more paper ids, a four-digit year, or `all`.
    Index {
        /// Paper ids, years, or `all`.
        #[arg(required = true, value_name = "SELECTOR")]
        selectors: Vec<String>,
        /// Reindex papers whose index records are already current.
        #[arg(long)]
        force: bool,
        /// Papers chunked and embedded at once; index writes stay serialized.
        #[arg(long, default_value = "1", value_name = "N")]
        workers: NonZeroUsize,
        /// Model directory; defaults to `$PAPER_MCP_MODEL_DIR`.
        #[arg(long, value_name = "DIR")]
        model_dir: Option<PathBuf>,
    },
    /// Serve search over MCP Streamable HTTP, behind a bearer token.
    Serve(ServeArgs),
}

#[derive(Debug, clap::Args)]
struct ServeArgs {
    /// Address to listen on.
    #[arg(long, default_value_t = DEFAULT_BIND, value_name = "ADDR")]
    bind: SocketAddr,
    /// File holding the bearer token; defaults to `$PAPER_MCP_TOKEN`.
    #[arg(long, value_name = "FILE")]
    token_file: Option<PathBuf>,
    /// Model directory; defaults to `$PAPER_MCP_MODEL_DIR`.
    #[arg(long, value_name = "DIR")]
    model_dir: Option<PathBuf>,
    /// Accepted `Host` header value; repeat for several. Replaces the
    /// loopback defaults, so a server behind a proxy names its public host.
    #[arg(long = "allowed-host", value_name = "HOST", default_values_t = default_allowed_hosts())]
    allowed_hosts: Vec<String>,
    /// Default number of nearest chunks a vector or hybrid search returns.
    #[arg(long, default_value_t = non_zero(DEFAULT_TOP_K), value_name = "N")]
    top_k: NonZeroUsize,
    /// Number of leading results the reranker reorders.
    #[arg(long, default_value_t = non_zero(DEFAULT_RERANK_DEPTH), value_name = "N")]
    rerank_depth: NonZeroUsize,
    /// Results per page.
    #[arg(long, default_value_t = DEFAULT_PAGE_SIZE, value_name = "N")]
    page_size: NonZeroUsize,
    /// Idle seconds after which a stored result list expires.
    #[arg(long, default_value_t = DEFAULT_CURSOR_TTL.as_secs(), value_name = "SECS")]
    cursor_ttl_secs: u64,
    /// Maximum number of stored result lists.
    #[arg(long, default_value_t = DEFAULT_CURSOR_CAPACITY, value_name = "N")]
    cursor_capacity: NonZeroUsize,
}

impl From<ServeArgs> for ServeOptions {
    fn from(args: ServeArgs) -> Self {
        Self {
            bind: args.bind,
            token_file: args.token_file,
            model_dir: args.model_dir,
            allowed_hosts: args.allowed_hosts,
            top_k: args.top_k,
            rerank_depth: args.rerank_depth,
            page_size: args.page_size,
            cursor_ttl: Duration::from_secs(args.cursor_ttl_secs),
            cursor_capacity: args.cursor_capacity,
        }
    }
}

/// Convert a default to [`NonZeroUsize`], mapping zero to one.
fn non_zero(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN)
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Command::Index {
            selectors,
            force,
            workers,
            model_dir,
        } => index(&selectors, force, workers, model_dir),
        Command::Serve(args) => serve(args.into()),
    }
}

fn serve(options: ServeOptions) -> Result<ExitCode> {
    let config = ServeConfig::load(options)?;
    let runtime = tokio::runtime::Runtime::new().context("cannot start the async runtime")?;
    runtime.block_on(async {
        let shutdown = CancellationToken::new();
        tokio::spawn(paper_mcp::server::ctrl_c(shutdown.clone()));
        paper_mcp::serve(&config, PaperServer::new(), shutdown).await
    })?;
    Ok(ExitCode::SUCCESS)
}

fn index(
    selectors: &[String],
    force: bool,
    workers: NonZeroUsize,
    model_dir: Option<PathBuf>,
) -> Result<ExitCode> {
    let parsed = paper_mcp::parse_selectors(selectors)?;
    let workspace = default_workspace_dir()?;
    let db = workspace.join(PAPERSTORE_DB);
    if !db.is_file() {
        bail!(
            "no paperstore database at {}; check WG21_DATA_DIR",
            db.display()
        );
    }
    let model_dir = paper_mcp::resolve_model_dir(model_dir.as_deref())?;
    let embedder = OnnxEmbedder::load(&model_dir)?;
    let reranker = OnnxReranker::load(&model_dir)?;

    let backend = SqliteBackend::new(&workspace)?;
    let papers = paper_mcp::resolve(&backend, &parsed)?;
    let tool_dir = backend.tool_dir(TOOL_NAME);
    let _lock = IndexLock::acquire(&tool_dir)?;
    let options = IndexOptions {
        force,
        workers,
        ..IndexOptions::new(&embedder, reranker.id())
    };
    tracing::info!(count = papers.len(), "selected papers");

    let runtime = tokio::runtime::Runtime::new().context("cannot start the async runtime")?;
    let report = runtime.block_on(async {
        let store = LanceStore::open(&tool_dir, embedder.dim()).await?;
        Ok::<_, paper_mcp::Error>(
            paper_mcp::index_papers(&backend, &store, &embedder, &papers, &options).await,
        )
    })?;

    for line in report.summary_lines() {
        eprintln!("{line}");
    }
    Ok(exit_code(&report))
}

/// Return the process exit code for an index run: failure when any paper or
/// the final optimization failed.
fn exit_code(report: &IndexReport) -> ExitCode {
    if report.has_failures() {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use paper_mcp::{Outcome, PaperReport};
    use paperstore::PaperNum;

    use super::*;

    fn report(outcome: Outcome, optimize_error: Option<&str>) -> IndexReport {
        let paper = PaperNum::parse("P2300R10").unwrap_or_else(|e| panic!("parse: {e}"));
        IndexReport {
            papers: vec![PaperReport { paper, outcome }],
            optimize_error: optimize_error.map(str::to_owned),
        }
    }

    #[test]
    fn exit_code_fails_on_any_failure() {
        assert_eq!(
            exit_code(&report(Outcome::Indexed { chunks: 2 }, None)),
            ExitCode::SUCCESS
        );
        assert_eq!(
            exit_code(&report(Outcome::Skipped, None)),
            ExitCode::SUCCESS
        );
        assert_eq!(
            exit_code(&report(Outcome::Failed("boom".to_owned()), None)),
            ExitCode::FAILURE
        );
        assert_eq!(
            exit_code(&report(Outcome::Skipped, Some("disk full"))),
            ExitCode::FAILURE
        );
    }
}
