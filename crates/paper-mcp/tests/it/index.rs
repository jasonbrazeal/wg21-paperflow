//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Indexer tests: full and incremental indexing of the fixture corpus.

use std::collections::BTreeMap;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::TryStreamExt;
use lancedb::arrow::arrow_array::RecordBatch;
use lancedb::arrow::arrow_array::cast::AsArray;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use paper_mcp::chunk::{CHUNKER_VERSION, PrefixMeta, chunk_markdown};
use paper_mcp::embed::{EmbedError, Embedder, HashEmbedder};
use paper_mcp::index::{
    INDEX_DIR_NAME, IndexError, IndexReport, Indexer, ManifestField, PaperIndex, REBUILD_COMMAND,
    default_index_dir,
};
use paperstore::StorageBackend;
use rusqlite::Connection;

use super::{
    FIXTURE_PAPER_IDS, METADATA_ONLY_ID, TEXT_BEARING_IDS, num, open_from_env, with_fixture,
};

const MODEL_ID: &str = "test/hash";
const DIM: usize = 16;

/// Wraps [`HashEmbedder`], counting `embed_documents` calls.
///
/// Panics on call number `panic_on_call` (1-based) to simulate a crash, and
/// fails documents of `fail_paper` with an error.
struct CountingEmbedder {
    inner: HashEmbedder,
    calls: AtomicUsize,
    panic_on_call: Option<usize>,
    fail_paper: Option<String>,
    embedded_papers: Mutex<Vec<String>>,
}

impl CountingEmbedder {
    fn new(model_id: &str, dim: usize) -> Self {
        Self {
            inner: HashEmbedder::new(model_id, dim),
            calls: AtomicUsize::new(0),
            panic_on_call: None,
            fail_paper: None,
            embedded_papers: Mutex::new(Vec::new()),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn embedded_papers(&self) -> Vec<String> {
        self.embedded_papers
            .lock()
            .unwrap_or_else(|e| panic!("lock: {e}"))
            .clone()
    }
}

/// Paper ID named on the first line of a contextual prefix.
fn prefix_paper_id(text: &str) -> String {
    text.lines()
        .next()
        .and_then(|l| l.strip_prefix("Paper: "))
        .unwrap_or_else(|| panic!("no paper line in {text:?}"))
        .to_owned()
}

impl Embedder for CountingEmbedder {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        assert!(
            self.panic_on_call != Some(call),
            "simulated crash on embed call {call}"
        );
        let paper = texts
            .first()
            .map(|t| prefix_paper_id(t))
            .unwrap_or_default();
        if self.fail_paper.as_deref() == Some(paper.as_str()) {
            return Err(EmbedError::CountMismatch {
                expected: texts.len(),
                actual: 0,
            });
        }
        self.embedded_papers
            .lock()
            .unwrap_or_else(|e| panic!("lock: {e}"))
            .push(paper);
        self.inner.embed_documents(texts)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.inner.embed_query(text)
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| panic!("build runtime: {e}"))
        .block_on(future)
}

fn index_path(workspace: &Path) -> PathBuf {
    workspace.join(INDEX_DIR_NAME)
}

async fn open_index(workspace: &Path) -> PaperIndex {
    PaperIndex::open(&index_path(workspace))
        .await
        .unwrap_or_else(|e| panic!("open index: {e}"))
}

async fn run_index(workspace: &Path, embedder: &dyn Embedder, chunker_version: u32) -> IndexReport {
    let backend = open_from_env();
    let index = open_index(workspace).await;
    Indexer::new(&backend, embedder, &index)
        .with_chunker_version(chunker_version)
        .run()
        .await
        .unwrap_or_else(|e| panic!("index run: {e}"))
}

async fn query_chunks(
    index: &PaperIndex,
    filter: Option<&str>,
    columns: &[&str],
) -> Vec<RecordBatch> {
    let table = index
        .chunks_table()
        .await
        .unwrap_or_else(|e| panic!("open chunks: {e}"));
    let mut query = table.query().select(Select::columns(columns));
    if let Some(filter) = filter {
        query = query.only_if(filter);
    }
    query
        .execute()
        .await
        .unwrap_or_else(|e| panic!("query chunks: {e}"))
        .try_collect()
        .await
        .unwrap_or_else(|e| panic!("collect chunks: {e}"))
}

fn strings(batches: &[RecordBatch], column: &str) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            b.column_by_name(column)
                .unwrap_or_else(|| panic!("no column {column}"))
                .as_string::<i32>()
                .iter()
                .map(|v| v.unwrap_or_default().to_owned())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn bools(batches: &[RecordBatch], column: &str) -> Vec<bool> {
    batches
        .iter()
        .flat_map(|b| {
            b.column_by_name(column)
                .unwrap_or_else(|| panic!("no column {column}"))
                .as_boolean()
                .iter()
                .map(|v| v.unwrap_or_else(|| panic!("null {column}")))
                .collect::<Vec<_>>()
        })
        .collect()
}

async fn count(index: &PaperIndex, filter: &str) -> usize {
    index
        .chunks_table()
        .await
        .unwrap_or_else(|e| panic!("open chunks: {e}"))
        .count_rows(Some(filter.to_owned()))
        .await
        .unwrap_or_else(|e| panic!("count {filter}: {e}"))
}

async fn paper_count(index: &PaperIndex) -> usize {
    index
        .papers_table()
        .await
        .unwrap_or_else(|e| panic!("open papers: {e}"))
        .count_rows(None)
        .await
        .unwrap_or_else(|e| panic!("count papers: {e}"))
}

/// Chunk count per text-bearing paper, from chunking its Markdown directly.
fn expected_chunks() -> BTreeMap<String, usize> {
    let backend = open_from_env();
    TEXT_BEARING_IDS
        .iter()
        .map(|id| {
            let row = backend
                .meta(&num(id))
                .unwrap_or_else(|e| panic!("meta {id}: {e}"));
            let md = backend
                .paper_md(&num(id))
                .unwrap_or_else(|e| panic!("paper_md {id}: {e}"));
            let meta = PrefixMeta {
                paper_id: (*id).to_owned(),
                title: row.title,
                authors: row.authors,
            };
            ((*id).to_owned(), chunk_markdown(&md, &meta).len())
        })
        .collect()
}

fn markdown_file(workspace: &Path, id: &str) -> PathBuf {
    workspace.join(format!("paperstore/{}.md", id.to_lowercase()))
}

fn text_ids() -> Vec<String> {
    TEXT_BEARING_IDS.iter().map(|id| (*id).to_owned()).collect()
}

#[test]
fn full_index_produces_expected_chunk_and_paper_counts() {
    with_fixture(|workspace| {
        block_on(async {
            let embedder = CountingEmbedder::new(MODEL_ID, DIM);
            let report = run_index(workspace, &embedder, CHUNKER_VERSION).await;
            assert!(report.is_success(), "{report:?}");
            assert_eq!(report.embedded, text_ids());
            assert_eq!(report.metadata_only, vec![METADATA_ONLY_ID.to_owned()]);
            assert_eq!(embedder.calls(), TEXT_BEARING_IDS.len());

            let index = open_index(workspace).await;
            let expected = expected_chunks();
            for (id, n) in &expected {
                assert!(*n > 0, "{id} has chunks");
                assert_eq!(
                    count(&index, &format!("paper_id = '{id}' AND has_text = true")).await,
                    *n,
                    "{id} chunks"
                );
            }
            let total: usize = expected.values().sum();
            assert_eq!(count(&index, "has_text = true").await, total);
            assert_eq!(count(&index, "true").await, total + 1);
            assert_eq!(paper_count(&index).await, FIXTURE_PAPER_IDS.len());

            let batches = query_chunks(&index, None, &["chunk_id"]).await;
            let mut ids = strings(&batches, "chunk_id");
            let n = ids.len();
            ids.sort();
            ids.dedup();
            assert_eq!(ids.len(), n, "chunk ids are unique");
        });
    });
}

#[test]
fn chunk_rows_carry_paper_metadata_and_latest_flags() {
    with_fixture(|workspace| {
        block_on(async {
            let embedder = HashEmbedder::new(MODEL_ID, DIM);
            run_index(workspace, &embedder, CHUNKER_VERSION).await;
            let index = open_index(workspace).await;

            for (id, latest) in [
                ("P1000R0", false),
                ("P1000R1", false),
                ("P1000R2", true),
                ("N5001", true),
                ("SD-4", true),
                ("P2000R0", true),
                (METADATA_ONLY_ID, true),
            ] {
                let batches =
                    query_chunks(&index, Some(&format!("paper_id = '{id}'")), &["is_latest"]).await;
                let flags = bools(&batches, "is_latest");
                assert!(!flags.is_empty(), "{id} rows");
                assert!(flags.iter().all(|f| *f == latest), "{id} is_latest");
            }

            let batches = query_chunks(
                &index,
                Some("paper_id = 'P1000R1'"),
                &[
                    "revisionless_id",
                    "mailing_date",
                    "target_group",
                    "intent",
                    "authors",
                ],
            )
            .await;
            assert!(
                strings(&batches, "revisionless_id")
                    .iter()
                    .all(|v| v == "P1000")
            );
            assert!(
                strings(&batches, "mailing_date")
                    .iter()
                    .all(|v| v == "2024-05")
            );
            assert!(
                strings(&batches, "target_group")
                    .iter()
                    .all(|v| v == "LEWG")
            );
            assert!(strings(&batches, "intent").iter().all(|v| v == "ask"));
            assert!(
                strings(&batches, "authors")
                    .iter()
                    .all(|v| v.contains("Ada Lovelace") && v.contains("Charles Babbage"))
            );
            assert_eq!(
                count(&index, "paper_id = 'P1000R1' AND revision = 1").await,
                batches.iter().map(RecordBatch::num_rows).sum::<usize>()
            );
            assert_eq!(
                count(&index, "paper_id = 'P1000R2' AND year = 2025").await,
                count(&index, "paper_id = 'P1000R2'").await
            );
            assert_eq!(
                count(&index, "paper_id = 'N5001' AND revision IS NULL").await,
                count(&index, "paper_id = 'N5001'").await
            );
        });
    });
}

#[test]
fn metadata_only_paper_has_no_searchable_chunks() {
    with_fixture(|workspace| {
        block_on(async {
            let embedder = CountingEmbedder::new(MODEL_ID, DIM);
            run_index(workspace, &embedder, CHUNKER_VERSION).await;
            assert!(
                !embedder
                    .embedded_papers()
                    .contains(&METADATA_ONLY_ID.to_owned())
            );

            let index = open_index(workspace).await;
            let id = METADATA_ONLY_ID;
            assert_eq!(count(&index, &format!("paper_id = '{id}'")).await, 1);
            assert_eq!(
                count(&index, &format!("paper_id = '{id}' AND has_text = false")).await,
                1
            );
            assert_eq!(
                count(&index, &format!("paper_id = '{id}' AND has_text = true")).await,
                0
            );
            let batches = query_chunks(
                &index,
                Some(&format!("paper_id = '{id}'")),
                &["text", "indexed_text"],
            )
            .await;
            assert_eq!(strings(&batches, "text"), vec![String::new()]);
            assert_eq!(strings(&batches, "indexed_text"), vec![String::new()]);
            assert_eq!(count(&index, "has_text = false").await, 1);
        });
    });
}

#[test]
fn reindex_after_touching_one_markdown_file_reembeds_only_that_paper() {
    with_fixture(|workspace| {
        block_on(async {
            run_index(
                workspace,
                &CountingEmbedder::new(MODEL_ID, DIM),
                CHUNKER_VERSION,
            )
            .await;

            let unchanged = CountingEmbedder::new(MODEL_ID, DIM);
            let report = run_index(workspace, &unchanged, CHUNKER_VERSION).await;
            assert_eq!(unchanged.calls(), 0, "no change, no embedding");
            assert!(report.embedded.is_empty());
            assert_eq!(report.unchanged, FIXTURE_PAPER_IDS.len());

            let path = markdown_file(workspace, "P2000R0");
            let original = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read: {e}"));
            fs::write(&path, &original).unwrap_or_else(|e| panic!("touch: {e}"));
            let touched = CountingEmbedder::new(MODEL_ID, DIM);
            run_index(workspace, &touched, CHUNKER_VERSION).await;
            assert_eq!(touched.calls(), 0, "identical bytes are not re-embedded");

            fs::write(
                &path,
                format!("{original}\n## Appendix\n\nA new closing paragraph.\n"),
            )
            .unwrap_or_else(|e| panic!("edit: {e}"));
            let edited = CountingEmbedder::new(MODEL_ID, DIM);
            let report = run_index(workspace, &edited, CHUNKER_VERSION).await;
            assert_eq!(edited.calls(), 1);
            assert_eq!(edited.embedded_papers(), vec!["P2000R0".to_owned()]);
            assert_eq!(report.embedded, vec!["P2000R0".to_owned()]);

            let index = open_index(workspace).await;
            let batches = query_chunks(&index, Some("paper_id = 'P2000R0'"), &["text"]).await;
            let texts = strings(&batches, "text");
            assert_eq!(
                texts
                    .iter()
                    .filter(|t| t.contains("A new closing paragraph."))
                    .count(),
                1
            );
            let expected: usize = expected_chunks().values().sum();
            assert_eq!(count(&index, "has_text = true").await, expected);
        });
    });
}

#[test]
fn changing_chunker_version_or_model_id_reembeds_all() {
    with_fixture(|workspace| {
        block_on(async {
            run_index(
                workspace,
                &CountingEmbedder::new(MODEL_ID, DIM),
                CHUNKER_VERSION,
            )
            .await;
            let expected: usize = expected_chunks().values().sum();

            let rechunked = CountingEmbedder::new(MODEL_ID, DIM);
            let report = run_index(workspace, &rechunked, CHUNKER_VERSION + 1).await;
            assert_eq!(rechunked.calls(), TEXT_BEARING_IDS.len());
            assert_eq!(report.embedded, text_ids());

            let new_model = CountingEmbedder::new("test/other", DIM);
            run_index(workspace, &new_model, CHUNKER_VERSION + 1).await;
            assert_eq!(new_model.calls(), TEXT_BEARING_IDS.len());

            let new_dim = CountingEmbedder::new("test/other", DIM * 2);
            run_index(workspace, &new_dim, CHUNKER_VERSION + 1).await;
            assert_eq!(new_dim.calls(), TEXT_BEARING_IDS.len());

            let index = open_index(workspace).await;
            assert_eq!(count(&index, "has_text = true").await, expected);
            assert_eq!(paper_count(&index).await, FIXTURE_PAPER_IDS.len());
            assert!(
                index
                    .check_manifest_with(&new_dim, CHUNKER_VERSION + 1)
                    .await
                    .is_ok()
            );
        });
    });
}

#[test]
fn interrupted_run_resumes() {
    with_fixture(|workspace| {
        let crashing = CountingEmbedder {
            panic_on_call: Some(3),
            ..CountingEmbedder::new(MODEL_ID, DIM)
        };
        let crashed = catch_unwind(AssertUnwindSafe(|| {
            block_on(run_index(workspace, &crashing, CHUNKER_VERSION))
        }));
        assert!(crashed.is_err(), "the run should crash");
        let completed = crashing.embedded_papers();
        assert_eq!(completed, vec!["N5001".to_owned(), "P1000R0".to_owned()]);

        block_on(async {
            let index = open_index(workspace).await;
            assert!(matches!(
                index.check_manifest(&crashing).await,
                Err(IndexError::ManifestMissing)
            ));

            let resumed = CountingEmbedder::new(MODEL_ID, DIM);
            let report = run_index(workspace, &resumed, CHUNKER_VERSION).await;
            assert!(report.is_success(), "{report:?}");
            assert_eq!(resumed.calls(), TEXT_BEARING_IDS.len() - completed.len());
            assert!(
                completed
                    .iter()
                    .all(|id| !resumed.embedded_papers().contains(id))
            );

            let expected = expected_chunks();
            for (id, n) in &expected {
                assert_eq!(
                    count(&index, &format!("paper_id = '{id}'")).await,
                    *n,
                    "{id}"
                );
            }
            assert_eq!(paper_count(&index).await, FIXTURE_PAPER_IDS.len());
            assert!(index.check_manifest(&resumed).await.is_ok());
        });
    });
}

async fn delete_manifest_row(index: &PaperIndex, id: &str) {
    index
        .papers_table()
        .await
        .unwrap_or_else(|e| panic!("open papers: {e}"))
        .delete(&format!("paper_id = '{id}'"))
        .await
        .unwrap_or_else(|e| panic!("delete manifest row {id}: {e}"));
}

fn delete_paperstore_row(workspace: &Path, id: &str) {
    let conn = Connection::open(workspace.join("paperstore.db"))
        .unwrap_or_else(|e| panic!("open db: {e}"));
    conn.execute("DELETE FROM papers WHERE paper_id = ?1", [id])
        .unwrap_or_else(|e| panic!("delete row: {e}"));
}

#[test]
fn chunks_without_a_manifest_row_are_rewritten() {
    with_fixture(|workspace| {
        block_on(async {
            run_index(
                workspace,
                &HashEmbedder::new(MODEL_ID, DIM),
                CHUNKER_VERSION,
            )
            .await;
            let index = open_index(workspace).await;
            // The state a crash between a paper's chunk write and its
            // manifest write leaves behind.
            delete_manifest_row(&index, "P2000R0").await;

            let resumed = CountingEmbedder::new(MODEL_ID, DIM);
            let report = run_index(workspace, &resumed, CHUNKER_VERSION).await;
            assert!(report.is_success(), "{report:?}");
            assert_eq!(resumed.embedded_papers(), vec!["P2000R0".to_owned()]);
            assert_eq!(report.embedded, vec!["P2000R0".to_owned()]);

            let expected = expected_chunks();
            assert_eq!(
                count(&index, "paper_id = 'P2000R0'").await,
                expected["P2000R0"],
                "old chunk rows are replaced, not duplicated"
            );
            let total: usize = expected.values().sum();
            assert_eq!(count(&index, "has_text = true").await, total);
            assert_eq!(paper_count(&index).await, FIXTURE_PAPER_IDS.len());
        });
    });
}

#[test]
fn orphan_chunks_of_a_removed_paper_are_deleted() {
    with_fixture(|workspace| {
        block_on(async {
            run_index(
                workspace,
                &HashEmbedder::new(MODEL_ID, DIM),
                CHUNKER_VERSION,
            )
            .await;
            let index = open_index(workspace).await;
            delete_manifest_row(&index, "P2000R0").await;
            delete_paperstore_row(workspace, "P2000R0");

            let report = run_index(
                workspace,
                &HashEmbedder::new(MODEL_ID, DIM),
                CHUNKER_VERSION,
            )
            .await;
            assert_eq!(report.removed, vec!["P2000R0".to_owned()]);
            assert_eq!(count(&index, "paper_id = 'P2000R0'").await, 0);
            assert_eq!(paper_count(&index).await, FIXTURE_PAPER_IDS.len() - 1);
        });
    });
}

#[test]
fn metadata_change_reembeds_that_paper() {
    with_fixture(|workspace| {
        block_on(async {
            run_index(
                workspace,
                &HashEmbedder::new(MODEL_ID, DIM),
                CHUNKER_VERSION,
            )
            .await;
            let conn = Connection::open(workspace.join("paperstore.db"))
                .unwrap_or_else(|e| panic!("open db: {e}"));
            conn.execute(
                "UPDATE papers SET title = 'Renamed Paper', target_group = 'SG1' \
                 WHERE paper_id = 'P2000R0'",
                [],
            )
            .unwrap_or_else(|e| panic!("update row: {e}"));
            drop(conn);

            let embedder = CountingEmbedder::new(MODEL_ID, DIM);
            let report = run_index(workspace, &embedder, CHUNKER_VERSION).await;
            assert_eq!(embedder.embedded_papers(), vec!["P2000R0".to_owned()]);
            assert_eq!(report.embedded, vec!["P2000R0".to_owned()]);
            assert_eq!(report.unchanged, FIXTURE_PAPER_IDS.len() - 1);

            let index = open_index(workspace).await;
            let batches = query_chunks(
                &index,
                Some("paper_id = 'P2000R0'"),
                &["target_group", "indexed_text"],
            )
            .await;
            assert!(!batches.is_empty());
            assert!(strings(&batches, "target_group").iter().all(|v| v == "SG1"));
            assert!(
                strings(&batches, "indexed_text")
                    .iter()
                    .all(|v| v.contains("Renamed Paper"))
            );
        });
    });
}

#[test]
fn per_paper_failure_is_skipped_and_reported() {
    with_fixture(|workspace| {
        block_on(async {
            run_index(
                workspace,
                &HashEmbedder::new(MODEL_ID, DIM),
                CHUNKER_VERSION,
            )
            .await;
            let failing = CountingEmbedder {
                fail_paper: Some("P2000R0".to_owned()),
                ..CountingEmbedder::new("test/other", DIM)
            };
            let report = run_index(workspace, &failing, CHUNKER_VERSION).await;
            assert!(!report.is_success());
            assert_eq!(report.failures.len(), 1);
            assert_eq!(report.failures[0].paper_id, "P2000R0");
            assert!(!report.failures[0].error.is_empty());
            assert_eq!(report.embedded.len(), TEXT_BEARING_IDS.len() - 1);

            let index = open_index(workspace).await;
            assert_eq!(
                count(&index, "paper_id = 'P2000R0'").await,
                0,
                "stale chunks from the old model are purged"
            );
            assert_eq!(paper_count(&index).await, FIXTURE_PAPER_IDS.len() - 1);
            let expected = expected_chunks();
            assert_eq!(
                count(&index, "paper_id = 'P1000R2'").await,
                expected["P1000R2"]
            );

            let retry = CountingEmbedder::new("test/other", DIM);
            let report = run_index(workspace, &retry, CHUNKER_VERSION).await;
            assert!(report.is_success());
            assert_eq!(retry.embedded_papers(), vec!["P2000R0".to_owned()]);
        });
    });
}

#[test]
fn removed_paper_is_dropped_and_latest_recomputed() {
    with_fixture(|workspace| {
        block_on(async {
            run_index(
                workspace,
                &HashEmbedder::new(MODEL_ID, DIM),
                CHUNKER_VERSION,
            )
            .await;
            let conn = Connection::open(workspace.join("paperstore.db"))
                .unwrap_or_else(|e| panic!("open db: {e}"));
            conn.execute("DELETE FROM papers WHERE paper_id = 'P1000R2'", [])
                .unwrap_or_else(|e| panic!("delete row: {e}"));
            drop(conn);

            let embedder = CountingEmbedder::new(MODEL_ID, DIM);
            let report = run_index(workspace, &embedder, CHUNKER_VERSION).await;
            assert_eq!(embedder.calls(), 0);
            assert_eq!(report.removed, vec!["P1000R2".to_owned()]);

            let index = open_index(workspace).await;
            assert_eq!(count(&index, "paper_id = 'P1000R2'").await, 0);
            assert_eq!(paper_count(&index).await, FIXTURE_PAPER_IDS.len() - 1);
            let r1 = count(&index, "paper_id = 'P1000R1'").await;
            assert!(r1 > 0);
            assert_eq!(
                count(&index, "paper_id = 'P1000R1' AND is_latest = true").await,
                r1
            );
            assert_eq!(
                count(&index, "paper_id = 'P1000R0' AND is_latest = true").await,
                0
            );
        });
    });
}

#[test]
fn check_manifest_rejects_model_dim_and_chunker_mismatches() {
    with_fixture(|workspace| {
        block_on(async {
            let index = open_index(workspace).await;
            let embedder = HashEmbedder::new(MODEL_ID, DIM);
            let missing = index.check_manifest(&embedder).await;
            match missing {
                Err(e @ IndexError::ManifestMissing) => {
                    assert!(e.to_string().contains(REBUILD_COMMAND), "{e}");
                }
                other => panic!("expected ManifestMissing, got {other:?}"),
            }

            run_index(workspace, &embedder, CHUNKER_VERSION + 1).await;
            assert!(
                index
                    .check_manifest_with(&embedder, CHUNKER_VERSION + 1)
                    .await
                    .is_ok()
            );

            let cases: [(&dyn Embedder, ManifestField); 3] = [
                (
                    &HashEmbedder::new("test/other", DIM),
                    ManifestField::ModelId,
                ),
                (&HashEmbedder::new(MODEL_ID, DIM * 2), ManifestField::Dim),
                (&embedder, ManifestField::ChunkerVersion),
            ];
            for (candidate, field) in cases {
                match index.check_manifest(candidate).await {
                    Err(e @ IndexError::ManifestMismatch { field: got, .. }) => {
                        assert_eq!(got, field);
                        assert!(e.to_string().contains(REBUILD_COMMAND), "{e}");
                    }
                    other => panic!("expected {field:?} mismatch, got {other:?}"),
                }
            }
        });
    });
}

#[test]
fn indexing_builds_full_text_and_scalar_indexes() {
    with_fixture(|workspace| {
        block_on(async {
            run_index(
                workspace,
                &HashEmbedder::new(MODEL_ID, DIM),
                CHUNKER_VERSION,
            )
            .await;
            let index = open_index(workspace).await;
            let indices = index
                .chunks_table()
                .await
                .unwrap_or_else(|e| panic!("open chunks: {e}"))
                .list_indices()
                .await
                .unwrap_or_else(|e| panic!("list indices: {e}"));
            let indexed: Vec<(String, String)> = indices
                .iter()
                .map(|i| (i.columns.join(","), i.index_type.to_string()))
                .collect();
            assert!(
                indexed.contains(&("indexed_text".to_owned(), "FTS".to_owned())),
                "{indexed:?}"
            );
            for column in [
                "paper_id",
                "revisionless_id",
                "is_latest",
                "has_text",
                "year",
                "mailing_date",
                "target_group",
                "intent",
                "chunk_kind",
            ] {
                assert!(
                    indexed.iter().any(|(c, _)| c == column),
                    "{column} index in {indexed:?}"
                );
            }
        });
    });
}

fn snapshot(workspace: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
        let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
        for entry in entries {
            let path = entry.unwrap_or_else(|e| panic!("entry: {e}")).path();
            if path == root.join(INDEX_DIR_NAME) {
                continue;
            }
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let bytes =
                    fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or_else(|e| panic!("strip prefix: {e}"))
                    .to_path_buf();
                out.insert(rel, bytes);
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(workspace, workspace, &mut out);
    out
}

#[test]
fn paperstore_files_are_unmodified_after_indexing() {
    with_fixture(|workspace| {
        let backend = open_from_env();
        let before = snapshot(workspace);
        assert!(before.contains_key(Path::new("paperstore.db")));
        block_on(async {
            let dir = default_index_dir().unwrap_or_else(|e| panic!("default dir: {e}"));
            assert_eq!(dir, index_path(workspace));
            let index = PaperIndex::open(&dir)
                .await
                .unwrap_or_else(|e| panic!("open index: {e}"));
            let report = Indexer::new(&backend, &HashEmbedder::new(MODEL_ID, DIM), &index)
                .run()
                .await
                .unwrap_or_else(|e| panic!("index run: {e}"));
            assert!(report.is_success());
        });
        drop(backend);
        assert_eq!(snapshot(workspace), before);
        assert!(index_path(workspace).is_dir());
        let top: Vec<String> = fs::read_dir(workspace)
            .unwrap_or_else(|e| panic!("read workspace: {e}"))
            .map(|e| {
                e.unwrap_or_else(|e| panic!("entry: {e}"))
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert!(
            top.iter()
                .all(|n| ["paperstore", "paperstore.db", INDEX_DIR_NAME].contains(&n.as_str())),
            "{top:?}"
        );
    });
}
