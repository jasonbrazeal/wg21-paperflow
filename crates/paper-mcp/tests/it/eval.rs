// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! `evaluate` over the indexed fixture workspace.

use std::sync::Arc;

use paper_mcp::model::fake::{HashEmbedder, IdentityReranker};
use paper_mcp::{
    Category, EvalConfig, GoldenQuery, GoldenSet, LabeledRange, SearchParams, evaluate,
    recall_at_k, search_chunks,
};

use crate::ingest::{ALL_PAPERS, Workspace, num, nums, options, run};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Small enough that hits past the head do not count.
const K: usize = 3;

/// Lines covering a whole fixture paper.
const WHOLE: (usize, usize) = (1, 10_000);

fn golden_query(
    id: &str,
    category: Category,
    text: &str,
    paper_id: Option<&str>,
    labeled: &str,
) -> GoldenQuery {
    GoldenQuery {
        id: id.to_owned(),
        category,
        query: text.to_owned(),
        all_revisions: false,
        paper_id: paper_id.map(num),
        relevant: vec![LabeledRange {
            paper_id: num(labeled),
            start: WHOLE.0,
            end: WHOLE.1,
        }],
    }
}

fn golden() -> GoldenSet {
    GoldenSet {
        queries: vec![
            golden_query(
                "latest",
                Category::Prose,
                "sender scheduler",
                None,
                "P2300R10",
            ),
            golden_query(
                "older",
                Category::Identifier,
                "sender scheduler",
                None,
                "P2300R4",
            ),
            golden_query(
                "versioned",
                Category::Revision,
                "sender scheduler",
                Some("P2300R4"),
                "P2300R4",
            ),
            golden_query(
                "placeholder",
                Category::WordingDiff,
                "placeholder with no name",
                None,
                "P2169R4",
            ),
        ],
    }
}

#[tokio::test]
async fn evaluate_runs_every_query_in_every_config() -> TestResult {
    let ws = Workspace::new()?;
    let store = ws.store().await?;
    run(&ws.backend, &store, &nums(&ALL_PAPERS), &options()).await?;
    let embedder = Arc::new(HashEmbedder::default());
    let reranker = Arc::new(IdentityReranker);
    let params = SearchParams::default();
    let golden = golden();

    let report = evaluate(&store, &embedder, &reranker, params, &golden, K).await?;

    assert_eq!(report.k, K);
    assert_eq!(report.throughput, None);
    let configs: Vec<EvalConfig> = report.configs.iter().map(|c| c.config).collect();
    assert_eq!(configs, EvalConfig::ALL);
    for result in &report.configs {
        let config = result.config;
        assert_eq!(result.queries.len(), golden.queries.len(), "{config}");
        for (got, query) in result.queries.iter().zip(&golden.queries) {
            assert_eq!(got.id, query.id, "{config}");
            assert_eq!(got.category, query.category, "{config}");
            let hits = search_chunks(
                &store,
                &embedder,
                &reranker,
                &config.params(params),
                &query.request(config.mode()),
            )
            .await?;
            let expected = recall_at_k(&hits, &query.relevant, K);
            assert!(
                (got.recall - expected).abs() < f64::EPSILON,
                "{config} {}: {} != {expected}",
                query.id,
                got.recall
            );
        }
    }

    let keyword = report.config(EvalConfig::Bm25).ok_or("no bm25 result")?;
    let recall = |id: &str| {
        keyword
            .queries
            .iter()
            .find(|q| q.id == id)
            .map(|q| q.recall)
    };
    assert_eq!(recall("latest"), Some(1.0));
    assert_eq!(recall("older"), Some(0.0));
    assert_eq!(recall("versioned"), Some(1.0));
    assert_eq!(recall("placeholder"), Some(1.0));
    Ok(())
}

/// A query labeling one line range of one paper.
fn ranged(
    id: &str,
    category: Category,
    text: &str,
    labeled: &str,
    (start, end): (usize, usize),
) -> GoldenQuery {
    GoldenQuery {
        relevant: vec![LabeledRange {
            paper_id: num(labeled),
            start,
            end,
        }],
        ..golden_query(id, category, text, None, labeled)
    }
}

/// Papers of the regression subset: two real papers and two revisions of a stub.
const SUBSET: [&str; 4] = ["P2169R4", "P2300R10", "P2300R4", "P2322R4"];

fn subset_golden() -> GoldenSet {
    GoldenSet {
        queries: vec![
            ranged(
                "motivation",
                Category::Prose,
                "why give variables named underscore no name",
                "P2169R4",
                (66, 80),
            ),
            ranged(
                "mock",
                Category::Prose,
                "impact on Google Mock",
                "P2169R4",
                (221, 237),
            ),
            ranged(
                "fold-right",
                Category::Identifier,
                "fold_right",
                "P2322R4",
                (218, 245),
            ),
            ranged(
                "short-circuit",
                Category::Prose,
                "short-circuiting folds",
                "P2322R4",
                (269, 405),
            ),
            ranged(
                "fold-wording",
                Category::WordingDiff,
                "ranges::fold_left_with_iter",
                "P2322R4",
                (525, 868),
            ),
            GoldenQuery {
                paper_id: Some(num("P2300R4")),
                ..ranged(
                    "versioned",
                    Category::Revision,
                    "sender scheduler",
                    "P2300R4",
                    (12, 29),
                )
            },
        ],
    }
}

/// Recall@`K` of every subset query, per configuration in report order.
type Snapshot = [(EvalConfig, [f64; 6]); 4];

const SUBSET_SNAPSHOT: Snapshot = [
    (EvalConfig::Dense, [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
    (EvalConfig::Bm25, [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
    (EvalConfig::Hybrid, [0.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
    (EvalConfig::HybridRerank, [0.0, 1.0, 1.0, 1.0, 0.0, 1.0]),
];

/// Guards the harness: indexing the subset with the pinned chunk budget and
/// keyword tokenizer and evaluating it must reproduce the recorded recall.
#[tokio::test]
async fn harness_reproduces_recorded_recall_on_fixture_subset() -> TestResult {
    let ws = Workspace::new()?;
    let store = ws.store().await?;
    run(&ws.backend, &store, &nums(&SUBSET), &options()).await?;
    let embedder = Arc::new(HashEmbedder::default());
    let reranker = Arc::new(IdentityReranker);
    let golden = subset_golden();
    let problems = golden.check_corpus(&ws.backend);
    assert!(problems.is_empty(), "{problems:?}");

    let report = evaluate(
        &store,
        &embedder,
        &reranker,
        SearchParams::default(),
        &golden,
        K,
    )
    .await?;

    let got: Vec<(EvalConfig, Vec<f64>)> = report
        .configs
        .iter()
        .map(|c| (c.config, c.queries.iter().map(|q| q.recall).collect()))
        .collect();
    let expected: Vec<(EvalConfig, Vec<f64>)> = SUBSET_SNAPSHOT
        .iter()
        .map(|(c, r)| (*c, r.to_vec()))
        .collect();
    assert_eq!(got, expected);
    Ok(())
}

/// Golden set shipped with the crate.
const GOLDEN: &str = include_str!("../../eval/golden.jsonl");

/// Checks the criterion recorded in `eval/RESULTS.md` against the full-corpus
/// index in `$WG21_DATA_DIR` built with the models in `$PAPER_MCP_MODEL_DIR`:
/// the served configuration beats dense-only and BM25-only recall.
#[tokio::test]
#[ignore = "needs PAPER_MCP_MODEL_DIR and a full-corpus index in WG21_DATA_DIR"]
async fn full_corpus_served_recall_beats_dense_and_bm25() -> TestResult {
    let workspace = paperstore::default_workspace_dir()?;
    let backend = paperstore_sqlite::SqliteBackend::new(&workspace)?;
    let model_dir = paper_mcp::resolve_model_dir(None)?;
    let embedder = Arc::new(paper_mcp::OnnxEmbedder::load(&model_dir)?);
    let reranker = Arc::new(paper_mcp::OnnxReranker::load(&model_dir)?);
    let store = paper_mcp::LanceStore::open_read_only(
        &paperstore::StorageBackend::tool_dir(&backend, paper_mcp::TOOL_NAME),
        paper_mcp::Embedder::dim(embedder.as_ref()),
        paper_mcp::LanceStoreOptions::default(),
    )
    .await?;
    let golden = GoldenSet::parse(GOLDEN)?;
    let problems = golden.check_corpus(&backend);
    assert!(problems.is_empty(), "{problems:?}");

    let report = evaluate(
        &store,
        &embedder,
        &reranker,
        SearchParams::default(),
        &golden,
        paper_mcp::DEFAULT_RECALL_K,
    )
    .await?;

    let recall = |config| {
        report
            .config(config)
            .map(paper_mcp::ConfigResult::recall)
            .ok_or("missing configuration")
    };
    let served = recall(EvalConfig::HybridRerank)?;
    assert!(served > recall(EvalConfig::Dense)?, "{report:?}");
    assert!(served > recall(EvalConfig::Bm25)?, "{report:?}");
    Ok(())
}
