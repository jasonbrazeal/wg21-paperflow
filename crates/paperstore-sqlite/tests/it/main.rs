//! Integration tests for [`SqliteBackend`] against temp-dir workspaces.

use std::fs;
use std::path::Path;

use paperstore::{Error, PaperNum, StorageBackend, WORKSPACE_ENV_VAR};
use paperstore_sqlite::SqliteBackend;
use rusqlite::Connection;
use tempfile::TempDir;

const FULL_INSERT: &str = "\
INSERT INTO papers (\
    paper_id, year, title, authors, target_group, intent, url, \
    document_date, mailing_date, disposition, previous_version, \
    source_file, markdown_path, dissect_path, advocatus_path, agora_path, \
    assay_path, citations_extracted_at, line_count, status, error, updated_at\
) VALUES (\
    'P4003R2', '2026', 'Concepts for C++26', '[\"Alice Liddell\",\"Bob Carroll\"]', \
    'LEWG', 'proposal', 'https://wg21.link/p4003r2', \
    '2026-02-14', '2026-02', 'none', 'P4003R1', \
    'sources/p4003r2.pdf', 'papers/p4003r2.md', 'dissect/p4003r2.json', \
    'adv/p4003r2.json', 'agora/p4003r2.json', 'assay/p4003r2.md', \
    '2026-03-01T00:00:00+00:00', 1234, 5, '', '2026-03-02T00:00:00+00:00'\
)";

/// The pre-migration `papers` table: all 21 schema columns, no `assay_path`,
/// as an old Python-created store would have it.
const PRE_MIGRATION_TABLE: &str = "
CREATE TABLE papers (
    paper_id         TEXT PRIMARY KEY,
    year             TEXT DEFAULT '',
    title            TEXT DEFAULT '',
    authors          TEXT DEFAULT '',
    target_group     TEXT DEFAULT '',
    intent           TEXT DEFAULT '',
    url              TEXT DEFAULT '',
    document_date    TEXT DEFAULT '',
    mailing_date     TEXT DEFAULT '',
    disposition      TEXT DEFAULT '',
    previous_version TEXT DEFAULT '',
    source_file      TEXT DEFAULT '',
    markdown_path    TEXT DEFAULT '',
    dissect_path     TEXT DEFAULT '',
    advocatus_path   TEXT DEFAULT '',
    agora_path             TEXT DEFAULT '',
    citations_extracted_at TEXT DEFAULT '',
    line_count             INTEGER DEFAULT 0,
    status                 INTEGER NOT NULL DEFAULT 0,
    error                  TEXT DEFAULT '',
    updated_at             TEXT DEFAULT ''
);
";

/// A `papers` table as migrated by an older Python paperstore: `assay_path`
/// already added, `citations_extracted_at` not yet present.
const PRE_CITATIONS_TABLE: &str = "
CREATE TABLE papers (
    paper_id         TEXT PRIMARY KEY,
    year             TEXT DEFAULT '',
    title            TEXT DEFAULT '',
    authors          TEXT DEFAULT '',
    target_group     TEXT DEFAULT '',
    intent           TEXT DEFAULT '',
    url              TEXT DEFAULT '',
    document_date    TEXT DEFAULT '',
    mailing_date     TEXT DEFAULT '',
    disposition      TEXT DEFAULT '',
    previous_version TEXT DEFAULT '',
    source_file      TEXT DEFAULT '',
    markdown_path    TEXT DEFAULT '',
    dissect_path     TEXT DEFAULT '',
    advocatus_path   TEXT DEFAULT '',
    agora_path       TEXT DEFAULT '',
    assay_path       TEXT DEFAULT '',
    line_count       INTEGER DEFAULT 0,
    status           INTEGER NOT NULL DEFAULT 0,
    error            TEXT DEFAULT '',
    updated_at       TEXT DEFAULT ''
);
";

struct Workspace {
    dir: TempDir,
}

impl Workspace {
    fn new() -> Self {
        let dir = TempDir::new().unwrap_or_else(|e| panic!("create temp dir: {e}"));
        Self { dir }
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }

    fn open_backend(&self) -> SqliteBackend {
        SqliteBackend::new(self.path()).unwrap_or_else(|e| panic!("open backend: {e}"))
    }

    fn raw_connection(&self) -> Connection {
        Connection::open(self.path().join("paperstore.db"))
            .unwrap_or_else(|e| panic!("open raw connection: {e}"))
    }

    fn insert_row(&self, paper_id: &str, markdown_path: &str) {
        self.raw_connection()
            .execute(
                "INSERT INTO papers (paper_id, markdown_path) VALUES (?1, ?2)",
                rusqlite::params![paper_id, markdown_path],
            )
            .unwrap_or_else(|e| panic!("insert row: {e}"));
    }
}

fn paper_num() -> PaperNum {
    PaperNum::parse("P4003R2").unwrap_or_else(|e| panic!("parse: {e}"))
}

#[test]
fn meta_maps_all_fields_and_decodes_authors() {
    let ws = Workspace::new();
    let backend = ws.open_backend();
    ws.raw_connection()
        .execute(FULL_INSERT, [])
        .unwrap_or_else(|e| panic!("insert full row: {e}"));

    let row = backend
        .meta(&paper_num())
        .unwrap_or_else(|e| panic!("meta: {e}"));

    assert_eq!(row.paper_num, "P4003R2");
    assert_eq!(row.year, "2026");
    assert_eq!(row.title, "Concepts for C++26");
    assert_eq!(row.authors, ["Alice Liddell", "Bob Carroll"]);
    assert_eq!(row.target_group, "LEWG");
    assert_eq!(row.intent, "proposal");
    assert_eq!(row.url, "https://wg21.link/p4003r2");
    assert_eq!(row.document_date, "2026-02-14");
    assert_eq!(row.mailing_date, "2026-02");
    assert_eq!(row.disposition, "none");
    assert_eq!(row.previous_version, "P4003R1");
    assert_eq!(row.source_file, "sources/p4003r2.pdf");
    assert_eq!(row.markdown_path, "papers/p4003r2.md");
    assert_eq!(row.dissect_path, "dissect/p4003r2.json");
    assert_eq!(row.agora_path, "agora/p4003r2.json");
    assert_eq!(row.assay_path, "assay/p4003r2.md");
    assert_eq!(row.citations_extracted_at, "2026-03-01T00:00:00+00:00");
    assert_eq!(row.error, "");
    assert_eq!(row.line_count, 1234);
    assert_eq!(row.status, 5);
}

#[test]
fn meta_missing_row_gives_missing_paper() {
    let ws = Workspace::new();
    let backend = ws.open_backend();

    let result = backend.meta(&paper_num());

    // The variant is `#[non_exhaustive]`, so downstream code cannot name it
    // in a pattern; assert the Display contract instead.
    let err = match result {
        Err(err) => err,
        Ok(row) => panic!("expected an error, got {row:?}"),
    };
    assert_eq!(err.to_string(), "no metadata for paper P4003R2");
}

#[test]
fn paper_md_returns_file_contents() {
    let ws = Workspace::new();
    let backend = ws.open_backend();
    let md = ws.path().join("p4003r2.md");
    fs::write(&md, "# Hello\n").unwrap_or_else(|e| panic!("write md: {e}"));
    ws.insert_row("P4003R2", &md.to_string_lossy());

    let text = backend
        .paper_md(&paper_num())
        .unwrap_or_else(|e| panic!("paper_md: {e}"));

    assert_eq!(text, "# Hello\n");
}

#[test]
fn paper_md_missing_row_gives_missing_markdown() {
    let ws = Workspace::new();
    let backend = ws.open_backend();

    let result = backend.paper_md(&paper_num());

    // The variant is `#[non_exhaustive]`, so downstream code cannot name it
    // in a pattern; assert the Display contract instead.
    let err = match result {
        Err(err) => err,
        Ok(text) => panic!("expected an error, got {text:?}"),
    };
    assert_eq!(err.to_string(), "no converted markdown for paper P4003R2");
}

#[test]
fn paper_md_empty_path_gives_missing_markdown() {
    let ws = Workspace::new();
    let backend = ws.open_backend();
    ws.insert_row("P4003R2", "");

    let result = backend.paper_md(&paper_num());

    // The variant is `#[non_exhaustive]`, so downstream code cannot name it
    // in a pattern; assert the Display contract instead.
    let err = match result {
        Err(err) => err,
        Ok(text) => panic!("expected an error, got {text:?}"),
    };
    assert_eq!(err.to_string(), "no converted markdown for paper P4003R2");
}

#[test]
fn paper_md_deleted_file_gives_missing_markdown() {
    let ws = Workspace::new();
    let backend = ws.open_backend();
    let md = ws.path().join("p4003r2.md");
    fs::write(&md, "# Hello\n").unwrap_or_else(|e| panic!("write md: {e}"));
    ws.insert_row("P4003R2", &md.to_string_lossy());
    fs::remove_file(&md).unwrap_or_else(|e| panic!("remove md: {e}"));

    let result = backend.paper_md(&paper_num());

    // The variant is `#[non_exhaustive]`, so downstream code cannot name it
    // in a pattern; assert the Display contract instead.
    let err = match result {
        Err(err) => err,
        Ok(text) => panic!("expected an error, got {text:?}"),
    };
    assert_eq!(err.to_string(), "no converted markdown for paper P4003R2");
}

#[test]
fn from_env_unset_gives_workspace_env_unset() {
    temp_env::with_var_unset(WORKSPACE_ENV_VAR, || {
        let result = SqliteBackend::from_env();
        assert!(
            matches!(result, Err(Error::WorkspaceEnvUnset)),
            "unexpected result: {result:?}"
        );
    });
}

#[test]
fn opening_existing_database_twice_is_schema_idempotent() {
    let ws = Workspace::new();
    let first = ws.open_backend();
    ws.insert_row("P4003R2", "");
    drop(first);

    let second = ws.open_backend();

    let row = second
        .meta(&paper_num())
        .unwrap_or_else(|e| panic!("meta: {e}"));
    assert_eq!(row.paper_num, "P4003R2");
}

#[test]
fn existing_table_without_assay_path_is_migrated() {
    let ws = Workspace::new();
    ws.raw_connection()
        .execute_batch(PRE_MIGRATION_TABLE)
        .unwrap_or_else(|e| panic!("create old table: {e}"));

    let backend = ws.open_backend();
    ws.insert_row("P4003R2", "");

    let row = backend
        .meta(&paper_num())
        .unwrap_or_else(|e| panic!("meta: {e}"));
    assert_eq!(row.assay_path, "");
}

#[test]
fn existing_table_without_citations_extracted_at_is_migrated() {
    let ws = Workspace::new();
    ws.raw_connection()
        .execute_batch(PRE_CITATIONS_TABLE)
        .unwrap_or_else(|e| panic!("create old table: {e}"));

    let backend = ws.open_backend();
    ws.insert_row("P4003R2", "");

    let row = backend
        .meta(&paper_num())
        .unwrap_or_else(|e| panic!("meta: {e}"));
    assert_eq!(row.citations_extracted_at, "");
}

fn listed_ids(backend: &SqliteBackend) -> Vec<String> {
    backend
        .list_papers()
        .unwrap_or_else(|e| panic!("list_papers: {e}"))
        .into_iter()
        .map(|row| row.paper_num)
        .collect()
}

#[test]
fn list_papers_empty_store_gives_empty_list() {
    let ws = Workspace::new();
    let backend = ws.open_backend();

    assert!(listed_ids(&backend).is_empty());
}

#[test]
fn list_papers_mixed_store_returns_every_row_sorted_by_paper_id() {
    let ws = Workspace::new();
    let backend = ws.open_backend();
    for id in ["SD-5", "P4003R2", "N5001", "SD-1", "P2583R3", "N4950"] {
        ws.insert_row(id, "");
    }

    assert_eq!(
        listed_ids(&backend),
        ["N4950", "N5001", "P2583R3", "P4003R2", "SD-1", "SD-5"]
    );
}

#[test]
fn list_papers_maps_all_fields_like_meta() {
    let ws = Workspace::new();
    let backend = ws.open_backend();
    ws.raw_connection()
        .execute(FULL_INSERT, [])
        .unwrap_or_else(|e| panic!("insert full row: {e}"));

    let rows = backend
        .list_papers()
        .unwrap_or_else(|e| panic!("list_papers: {e}"));
    let meta = backend
        .meta(&paper_num())
        .unwrap_or_else(|e| panic!("meta: {e}"));

    assert_eq!(rows.len(), 1);
    assert_eq!(format!("{:?}", rows[0]), format!("{meta:?}"));
}

#[test]
fn list_papers_includes_failed_and_unconverted_rows() {
    let ws = Workspace::new();
    let backend = ws.open_backend();
    ws.insert_row("P4003R2", "");
    ws.raw_connection()
        .execute(
            "INSERT INTO papers (paper_id, status, error) VALUES ('P2583R3', -2, 'boom')",
            [],
        )
        .unwrap_or_else(|e| panic!("insert failed row: {e}"));

    let rows = backend
        .list_papers()
        .unwrap_or_else(|e| panic!("list_papers: {e}"));

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].paper_num, "P2583R3");
    assert_eq!(rows[0].status, -2);
    assert_eq!(rows[0].error, "boom");
    assert_eq!(rows[1].paper_num, "P4003R2");
    assert_eq!(rows[1].markdown_path, "");
}

#[test]
fn sqlite_backend_is_send_and_static() {
    // `rusqlite::Connection` is `Send` but not `Sync`; pin the auto traits
    // that hold so losing one is a compile error, not a silent break.
    fn assert_bounds<T: Send + 'static>() {}
    assert_bounds::<SqliteBackend>();
}
