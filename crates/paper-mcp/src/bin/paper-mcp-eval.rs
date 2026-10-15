// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Measure paper-mcp retrieval quality against the golden query set.
//!
//! Loads and validates the golden set, checks its labels against the
//! paperstore corpus, optionally builds the index first, then runs every
//! query in each retrieval configuration and prints recall, latency and
//! indexing throughput.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use clap::Parser;
use paper_mcp::{
    DEFAULT_RECALL_K, DEFAULT_RERANK_DEPTH, DEFAULT_TOP_K, Embedder as _, EvalReport, GoldenSet,
    IndexLock, IndexOptions, IndexStore, LanceStore, LanceStoreOptions, OnnxEmbedder, OnnxReranker,
    Reranker as _, SearchParams, TOOL_NAME, Throughput,
};
use paperstore::{StorageBackend as _, default_workspace_dir};
use paperstore_sqlite::SqliteBackend;
use tracing_subscriber::EnvFilter;

/// Name of the paperstore database inside the workspace directory.
const PAPERSTORE_DB: &str = "paperstore.db";

/// The golden set shipped with the crate.
const DEFAULT_GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/eval/golden.jsonl");

/// Evaluate paper-mcp retrieval against a hand-labeled golden query set.
#[derive(Debug, Parser)]
#[command(version)]
struct Args {
    /// Golden set, JSON Lines.
    #[arg(long, default_value = DEFAULT_GOLDEN, value_name = "FILE")]
    golden: PathBuf,
    /// Model directory; defaults to `$PAPER_MCP_MODEL_DIR`.
    #[arg(long, value_name = "DIR")]
    model_dir: Option<PathBuf>,
    /// Directory holding the index; defaults to the workspace's paper-mcp
    /// tool directory, the one `paper-mcp index` writes.
    #[arg(long, value_name = "DIR")]
    index_dir: Option<PathBuf>,
    /// Build the index for these selectors (paper ids, years, or `all`)
    /// before evaluating, reindexing current papers, and report its
    /// throughput.
    #[arg(long, num_args = 1.., value_name = "SELECTOR")]
    build: Vec<String>,
    /// Papers chunked and embedded at once during `--build`.
    #[arg(long, default_value = "1", value_name = "N")]
    workers: NonZeroUsize,
    /// Nearest chunks the vector side retrieves.
    #[arg(long, default_value_t = DEFAULT_TOP_K, value_name = "N")]
    top_k: usize,
    /// Leading results the reranker reorders in the reranking configuration.
    #[arg(long, default_value_t = DEFAULT_RERANK_DEPTH, value_name = "N")]
    rerank_depth: usize,
    /// Result-list depth at which recall is measured.
    #[arg(long, default_value_t = DEFAULT_RECALL_K, value_name = "K")]
    k: usize,
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    match run(Args::parse()) {
        Ok(report) => {
            for line in report.summary_lines() {
                println!("{line}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<EvalReport> {
    let text = std::fs::read_to_string(&args.golden)
        .with_context(|| format!("cannot read {}", args.golden.display()))?;
    let golden = GoldenSet::parse(&text)?;

    let workspace = default_workspace_dir()?;
    let db = workspace.join(PAPERSTORE_DB);
    if !db.is_file() {
        bail!(
            "no paperstore database at {}; check WG21_DATA_DIR",
            db.display()
        );
    }
    let backend = SqliteBackend::new(&workspace)?;
    let problems = golden.check_corpus(&backend);
    if !problems.is_empty() {
        bail!(
            "the golden set does not match the corpus:\n  {}",
            problems.join("\n  ")
        );
    }

    let model_dir = paper_mcp::resolve_model_dir(args.model_dir.as_deref())?;
    let embedder = Arc::new(OnnxEmbedder::load(&model_dir)?);
    let reranker = Arc::new(OnnxReranker::load(&model_dir)?);
    let index_dir = args
        .index_dir
        .unwrap_or_else(|| backend.tool_dir(TOOL_NAME));
    let params = SearchParams {
        top_k: args.top_k,
        rerank_depth: args.rerank_depth,
    };

    let runtime = tokio::runtime::Runtime::new().context("cannot start the async runtime")?;
    runtime.block_on(async {
        if args.build.is_empty() {
            let store = LanceStore::open_read_only(
                &index_dir,
                embedder.dim(),
                LanceStoreOptions::default(),
            )
            .await?;
            return evaluate(&store, &embedder, &reranker, params, &golden, args.k).await;
        }
        let selectors = paper_mcp::parse_selectors(&args.build)?;
        let papers = paper_mcp::resolve(&backend, &selectors)?;
        let _lock = IndexLock::acquire(&index_dir)?;
        let options = IndexOptions {
            force: true,
            workers: args.workers,
            ..IndexOptions::new(embedder.as_ref(), reranker.id())
        };
        let store = LanceStore::open(&index_dir, embedder.dim()).await?;
        let started = Instant::now();
        let index_report =
            paper_mcp::index_papers(&backend, &store, embedder.as_ref(), &papers, &options).await;
        let elapsed = started.elapsed();
        if index_report.has_failures() {
            bail!(
                "the index run failed:\n  {}",
                index_report.summary_lines().join("\n  ")
            );
        }
        let mut report = evaluate(&store, &embedder, &reranker, params, &golden, args.k).await?;
        report.throughput = Some(Throughput::from_report(&index_report, elapsed));
        Ok(report)
    })
}

async fn evaluate<S: IndexStore>(
    store: &S,
    embedder: &Arc<OnnxEmbedder>,
    reranker: &Arc<OnnxReranker>,
    params: SearchParams,
    golden: &GoldenSet,
    k: usize,
) -> Result<EvalReport> {
    Ok(paper_mcp::evaluate(store, embedder, reranker, params, golden, k).await?)
}
