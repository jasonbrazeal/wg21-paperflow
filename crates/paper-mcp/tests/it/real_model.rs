//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Opt-in end-to-end test with the real embedding model on real papers.
//!
//! Runs only when [`REAL_MODEL_ENV`] is `1`. It downloads
//! `nomic-embed-text-v1.5` (about 0.5 GB) into a cache under Cargo's target
//! directory, indexes P1048R0 and P1048R1 from the real `$WG21_DATA_DIR`
//! into a temporary index, and checks each search type.

use std::path::{Path, PathBuf};

use paper_mcp::embed::FastEmbedder;
use paper_mcp::filter::{FilterInput, SearchFilters};
use paper_mcp::index::{Indexer, PaperIndex};
use paper_mcp::search::{SearchRequest, SearchType, search};
use paperstore::{PaperNum, PaperRow, StorageBackend, default_workspace_dir};
use paperstore_sqlite::SqliteBackend;
use tempfile::TempDir;

/// Environment variable that enables this test when set to `1`.
const REAL_MODEL_ENV: &str = "PAPER_MCP_REAL_MODEL";

/// Revisionless ID of the only paper family indexed.
const PAPER_FAMILY: &str = "P1048";

/// Every P1048 revision expected in the real paperstore.
const REVISION_COUNT: usize = 2;

/// The latest P1048 revision.
const LATEST_ID: &str = "P1048R1";

const QUERY: &str = "type trait to detect scoped enumerations";

const HIT_LIMIT: usize = 10;

/// Read-only view of a backend listing only one paper family's revisions.
struct FamilyBackend<'a> {
    inner: &'a SqliteBackend,
    family: &'a str,
}

impl FamilyBackend<'_> {
    fn in_family(&self, paper_id: &str) -> bool {
        paper_id
            .strip_prefix(self.family)
            .and_then(|rest| rest.strip_prefix('R'))
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    }
}

impl StorageBackend for FamilyBackend<'_> {
    fn meta(&self, num: &PaperNum) -> paperstore::Result<PaperRow> {
        self.inner.meta(num)
    }

    fn paper_md(&self, num: &PaperNum) -> paperstore::Result<String> {
        self.inner.paper_md(num)
    }

    fn list_papers(&self) -> paperstore::Result<Vec<PaperRow>> {
        let mut rows = self.inner.list_papers()?;
        rows.retain(|r| self.in_family(&r.paper_num));
        Ok(rows)
    }
}

fn model_cache_dir() -> PathBuf {
    Path::new(env!("CARGO_TARGET_TMPDIR")).join("fastembed")
}

/// Resolves the real `$WG21_DATA_DIR` under temp-env's lock.
///
/// Sibling tests in this binary run in parallel and point the variable at
/// fixture workspaces through temp-env; reading it outside the lock could
/// observe one of those.
fn real_workspace_dir() -> PathBuf {
    let no_overrides: [(&str, Option<&str>); 0] = [];
    temp_env::with_vars(no_overrides, default_workspace_dir)
        .unwrap_or_else(|e| panic!("resolve $WG21_DATA_DIR: {e}"))
}

#[test]
fn real_model_finds_latest_p1048_for_each_search_type() {
    if std::env::var(REAL_MODEL_ENV).as_deref() != Ok("1") {
        eprintln!("skipped: set {REAL_MODEL_ENV}=1 to run");
        return;
    }
    let workspace = real_workspace_dir();
    let sqlite = SqliteBackend::new(&workspace)
        .unwrap_or_else(|e| panic!("open {}: {e}", workspace.display()));
    let backend = FamilyBackend {
        inner: &sqlite,
        family: PAPER_FAMILY,
    };
    let rows = backend
        .list_papers()
        .unwrap_or_else(|e| panic!("list_papers: {e}"));
    assert_eq!(rows.len(), REVISION_COUNT, "P1048 revisions in paperstore");

    let embedder =
        FastEmbedder::new(Some(model_cache_dir())).unwrap_or_else(|e| panic!("load model: {e}"));
    let index_dir = TempDir::new().unwrap_or_else(|e| panic!("create temp dir: {e}"));

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| panic!("build runtime: {e}"))
        .block_on(async {
            let index = PaperIndex::open(index_dir.path())
                .await
                .unwrap_or_else(|e| panic!("open index: {e}"));
            let report = Indexer::new(&backend, &embedder, &index)
                .run()
                .await
                .unwrap_or_else(|e| panic!("index run: {e}"));
            assert!(report.is_success(), "{report:?}");
            assert_eq!(report.embedded.len(), REVISION_COUNT, "{report:?}");

            let filters = SearchFilters::from_input(&FilterInput {
                revision_scope: Some("latest".to_owned()),
                ..FilterInput::default()
            })
            .unwrap_or_else(|e| panic!("filters: {e}"));
            for search_type in SearchType::ALL {
                let req = SearchRequest {
                    limit: HIT_LIMIT,
                    ..SearchRequest::new(QUERY, search_type, filters.clone())
                };
                let hits = search(&index, &embedder, &req)
                    .await
                    .unwrap_or_else(|e| panic!("{search_type:?} search: {e}"));
                assert!(!hits.is_empty(), "{search_type:?} returned no hits");
                for hit in &hits {
                    assert_eq!(hit.paper_id, LATEST_ID, "{search_type:?}");
                }
                assert!(
                    hits.iter()
                        .any(|h| h.text.to_lowercase().contains("scoped")),
                    "{search_type:?} hits never mention scoped enumerations"
                );
            }
        });
}
