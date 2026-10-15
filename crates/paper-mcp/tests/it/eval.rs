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
