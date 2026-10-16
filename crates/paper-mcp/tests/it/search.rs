//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Search tests: keyword, vector, and hybrid queries over the indexed
//! fixture corpus.

use std::collections::BTreeSet;
use std::path::Path;

use paper_mcp::embed::{Embedder, HashEmbedder};
use paper_mcp::filter::{FilterInput, SearchFilters};
use paper_mcp::index::{INDEX_DIR_NAME, Indexer, PaperIndex};
use paper_mcp::search::{
    MAX_SEARCH_LIMIT, SearchError, SearchHit, SearchRequest, SearchType, search,
};
use paperstore::StorageBackend;

use super::{METADATA_ONLY_ID, TEXT_BEARING_IDS, num, open_from_env, with_fixture};

const MODEL_ID: &str = "test/hash";
const DIM: usize = 16;

/// Matches every chunk through its contextual prefix labels, and ranks the
/// P2000R0 chunks highest through its title.
const QUERY: &str = "Paper Title hazard pointer cells";

/// Latest revisions with text in the fixture.
const LATEST_TEXT_IDS: [&str; 4] = ["N5001", "P1000R2", "P2000R0", "SD-4"];

fn block_on<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| panic!("build runtime: {e}"))
        .block_on(future)
}

/// Index the fixture workspace and open the result.
async fn indexed(workspace: &Path, embedder: &dyn Embedder) -> PaperIndex {
    let backend = open_from_env();
    let index = PaperIndex::open(&workspace.join(INDEX_DIR_NAME))
        .await
        .unwrap_or_else(|e| panic!("open index: {e}"));
    let report = Indexer::new(&backend, embedder, &index)
        .run()
        .await
        .unwrap_or_else(|e| panic!("index run: {e}"));
    assert!(report.is_success(), "{report:?}");
    index
}

fn filters(input: &FilterInput) -> SearchFilters {
    SearchFilters::from_input(input).unwrap_or_else(|e| panic!("filters: {e}"))
}

fn paper_filter(id: &str) -> SearchFilters {
    filters(&FilterInput {
        paper_ids: Some(vec![id.to_owned()]),
        ..FilterInput::default()
    })
}

fn scope(scope: &str) -> SearchFilters {
    filters(&FilterInput {
        revision_scope: Some(scope.to_owned()),
        ..FilterInput::default()
    })
}

fn request(search_type: SearchType, filters: SearchFilters, limit: usize) -> SearchRequest {
    SearchRequest {
        limit,
        ..SearchRequest::new(QUERY, search_type, filters)
    }
}

async fn run(index: &PaperIndex, embedder: &dyn Embedder, req: &SearchRequest) -> Vec<SearchHit> {
    search(index, embedder, req)
        .await
        .unwrap_or_else(|e| panic!("{:?} search: {e}", req.search_type))
}

fn paper_ids(hits: &[SearchHit]) -> BTreeSet<&str> {
    hits.iter().map(|h| h.paper_id.as_str()).collect()
}

async fn chunk_count(index: &PaperIndex, id: &str) -> usize {
    index
        .chunks_table()
        .await
        .unwrap_or_else(|e| panic!("open chunks: {e}"))
        .count_rows(Some(format!("paper_id = '{id}' AND has_text = true")))
        .await
        .unwrap_or_else(|e| panic!("count {id}: {e}"))
}

#[test]
fn paper_id_filter_returns_only_that_paper() {
    with_fixture(|workspace| {
        block_on(async {
            let embedder = HashEmbedder::new(MODEL_ID, DIM);
            let index = indexed(workspace, &embedder).await;
            for search_type in SearchType::ALL {
                for id in ["P1000R2", "P1000R0", "N5001"] {
                    let req = request(search_type, paper_filter(id), MAX_SEARCH_LIMIT);
                    let hits = run(&index, &embedder, &req).await;
                    assert_eq!(
                        paper_ids(&hits),
                        BTreeSet::from([id]),
                        "{search_type:?} {id}"
                    );
                    assert_eq!(
                        hits.len(),
                        chunk_count(&index, id).await,
                        "{search_type:?} {id} returns every chunk"
                    );
                }
            }
        });
    });
}

#[test]
fn no_paper_filter_returns_only_latest_revisions_by_default() {
    with_fixture(|workspace| {
        block_on(async {
            let embedder = HashEmbedder::new(MODEL_ID, DIM);
            let index = indexed(workspace, &embedder).await;
            for search_type in SearchType::ALL {
                for filters in [SearchFilters::default(), scope("latest")] {
                    let req = request(search_type, filters, MAX_SEARCH_LIMIT);
                    let hits = run(&index, &embedder, &req).await;
                    assert_eq!(
                        paper_ids(&hits),
                        BTreeSet::from(LATEST_TEXT_IDS),
                        "{search_type:?}"
                    );
                }
            }
        });
    });
}

#[test]
fn all_scope_includes_older_revisions() {
    with_fixture(|workspace| {
        block_on(async {
            let embedder = HashEmbedder::new(MODEL_ID, DIM);
            let index = indexed(workspace, &embedder).await;
            for search_type in SearchType::ALL {
                let req = request(search_type, scope("all"), MAX_SEARCH_LIMIT);
                let hits = run(&index, &embedder, &req).await;
                assert_eq!(
                    paper_ids(&hits),
                    BTreeSet::from(TEXT_BEARING_IDS),
                    "{search_type:?}"
                );
            }
        });
    });
}

#[test]
fn metadata_only_papers_never_appear() {
    with_fixture(|workspace| {
        block_on(async {
            let embedder = HashEmbedder::new(MODEL_ID, DIM);
            let index = indexed(workspace, &embedder).await;
            for search_type in SearchType::ALL {
                let mut req = request(
                    search_type,
                    paper_filter(METADATA_ONLY_ID),
                    MAX_SEARCH_LIMIT,
                );
                req.query = "Contracts for coroutines".to_owned();
                let hits = run(&index, &embedder, &req).await;
                assert!(hits.is_empty(), "{search_type:?} {hits:?}");

                for filters in [SearchFilters::default(), scope("all")] {
                    let mut req = request(search_type, filters, MAX_SEARCH_LIMIT);
                    req.query = "Contracts for coroutines".to_owned();
                    let hits = run(&index, &embedder, &req).await;
                    assert!(
                        !paper_ids(&hits).contains(METADATA_ONLY_ID),
                        "{search_type:?}"
                    );
                }
            }
        });
    });
}

#[test]
fn filtered_search_fills_the_limit_from_matching_chunks() {
    with_fixture(|workspace| {
        block_on(async {
            let embedder = HashEmbedder::new(MODEL_ID, DIM);
            let index = indexed(workspace, &embedder).await;
            let n = chunk_count(&index, "P1000R0").await;
            assert!(n > 1, "P1000R0 has several chunks");
            for search_type in SearchType::ALL {
                for limit in [1, n - 1, n] {
                    let req = request(search_type, paper_filter("P1000R0"), limit);
                    let hits = run(&index, &embedder, &req).await;
                    assert_eq!(hits.len(), limit, "{search_type:?} limit {limit}");
                }
                let filters = filters(&FilterInput {
                    revision_scope: Some("all".to_owned()),
                    year_to: Some(2024),
                    ..FilterInput::default()
                });
                let req = request(search_type, filters, n);
                let hits = run(&index, &embedder, &req).await;
                assert_eq!(hits.len(), n, "{search_type:?} year filter");
                assert!(
                    !paper_ids(&hits).contains("P2000R0"),
                    "{search_type:?} year filter excludes 2025 papers"
                );
            }
        });
    });
}

#[test]
fn hits_carry_url_metadata_and_citable_line_ranges() {
    with_fixture(|workspace| {
        block_on(async {
            let embedder = HashEmbedder::new(MODEL_ID, DIM);
            let index = indexed(workspace, &embedder).await;
            let backend = open_from_env();
            for search_type in SearchType::ALL {
                let req = request(search_type, scope("all"), MAX_SEARCH_LIMIT);
                let hits = run(&index, &embedder, &req).await;
                assert!(!hits.is_empty(), "{search_type:?}");
                for hit in &hits {
                    assert_eq!(hit.url, format!("https://wg21.link/{}", hit.paper_id));
                    let row = backend
                        .meta(&num(&hit.paper_id))
                        .unwrap_or_else(|e| panic!("meta {}: {e}", hit.paper_id));
                    assert_eq!(hit.title, row.title, "{}", hit.paper_id);
                    assert_eq!(hit.authors, row.authors, "{}", hit.paper_id);
                    let md = backend
                        .paper_md(&num(&hit.paper_id))
                        .unwrap_or_else(|e| panic!("paper_md {}: {e}", hit.paper_id));
                    let lines: Vec<&str> = md.lines().collect();
                    let start = usize::try_from(hit.start_line).unwrap_or(usize::MAX);
                    let end = usize::try_from(hit.end_line).unwrap_or(usize::MAX);
                    assert_eq!(hit.text, lines[start..end].join("\n"), "{}", hit.paper_id);
                    assert!(hit.score.is_finite(), "{search_type:?} score");
                }
                let r2 = hits
                    .iter()
                    .find(|h| h.paper_id == "P1000R2")
                    .unwrap_or_else(|| panic!("{search_type:?} no P1000R2 hit"));
                assert_eq!(r2.url, "https://wg21.link/P1000R2");
                assert_eq!(r2.revision, Some(2));
                let n = hits
                    .iter()
                    .find(|h| h.paper_id == "N5001")
                    .unwrap_or_else(|| panic!("{search_type:?} no N5001 hit"));
                assert_eq!(n.revision, None);
                assert_eq!(n.url, "https://wg21.link/N5001");
            }
        });
    });
}

#[test]
fn results_are_ordered_by_score_then_paper_then_line_and_repeatable() {
    with_fixture(|workspace| {
        block_on(async {
            let embedder = HashEmbedder::new(MODEL_ID, DIM);
            let index = indexed(workspace, &embedder).await;
            for search_type in SearchType::ALL {
                let req = request(search_type, scope("all"), MAX_SEARCH_LIMIT);
                let hits = run(&index, &embedder, &req).await;
                assert!(
                    hits.is_sorted_by(|a, b| {
                        b.score.total_cmp(&a.score).then_with(|| {
                            (a.paper_id.as_str(), a.start_line)
                                .cmp(&(b.paper_id.as_str(), b.start_line))
                        }) != std::cmp::Ordering::Greater
                    }),
                    "{search_type:?} order"
                );
                assert_eq!(run(&index, &embedder, &req).await, hits, "{search_type:?}");
            }
        });
    });
}

#[test]
fn keyword_ranks_matching_chunks_first() {
    with_fixture(|workspace| {
        block_on(async {
            let embedder = HashEmbedder::new(MODEL_ID, DIM);
            let index = indexed(workspace, &embedder).await;
            let mut req = request(SearchType::Keyword, scope("all"), 1);
            req.query = "try_push_back".to_owned();
            let hits = run(&index, &embedder, &req).await;
            assert_eq!(hits.len(), 1);
            assert!(hits[0].text.contains("try_push_back"), "{hits:?}");
        });
    });
}

#[test]
fn invalid_requests_are_rejected() {
    with_fixture(|workspace| {
        block_on(async {
            let embedder = HashEmbedder::new(MODEL_ID, DIM);
            let index = indexed(workspace, &embedder).await;
            for search_type in SearchType::ALL {
                for limit in [0, MAX_SEARCH_LIMIT + 1] {
                    let req = request(search_type, SearchFilters::default(), limit);
                    assert!(
                        matches!(
                            search(&index, &embedder, &req).await,
                            Err(SearchError::Limit(l)) if l == limit
                        ),
                        "{search_type:?} limit {limit}"
                    );
                }
                let mut req = request(search_type, SearchFilters::default(), 1);
                req.query = "  \n ".to_owned();
                assert!(
                    matches!(
                        search(&index, &embedder, &req).await,
                        Err(SearchError::EmptyQuery)
                    ),
                    "{search_type:?} empty query"
                );
            }
        });
    });
}
