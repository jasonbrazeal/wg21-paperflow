// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! The `paper-mcp` command: index WG21 papers and serve search over MCP.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context as _, Result, bail};
use clap::{Parser, Subcommand};
use paper_mcp::{
    Embedder as _, IndexLock, IndexOptions, IndexReport, LanceStore, OnnxEmbedder, OnnxReranker,
    Reranker as _, TOOL_NAME,
};
use paperstore::{StorageBackend as _, default_workspace_dir};
use paperstore_sqlite::SqliteBackend;
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
    /// Serve search over MCP Streamable HTTP.
    Serve,
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
        Command::Serve => bail!("serve is not implemented yet"),
    }
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
