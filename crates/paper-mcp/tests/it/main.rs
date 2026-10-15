//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Integration tests for `paper-mcp` against the fixture corpus.
//!
//! The fixture under `tests/fixtures/` is a Python-created paperstore whose
//! artifact paths are stored relative to the workspace. [`with_fixture`]
//! copies it into a temp dir, rewrites those paths to absolute ones (as the
//! Python backend records them), and points `$WG21_DATA_DIR` at the copy.

use std::fs;
use std::path::{Path, PathBuf};

use paperstore::{Error, PaperNum, PaperRow, StorageBackend, WORKSPACE_ENV_VAR};
use paperstore_sqlite::SqliteBackend;
use rusqlite::Connection;
use tempfile::TempDir;

/// Every paper in the fixture, in `paper_id` order.
const FIXTURE_PAPER_IDS: [&str; 7] = [
    "N5001", "P1000R0", "P1000R1", "P1000R2", "P2000R0", "P3000R0", "SD-4",
];

/// The fixture papers that have converted Markdown.
const TEXT_BEARING_IDS: [&str; 6] = ["N5001", "P1000R0", "P1000R1", "P1000R2", "P2000R0", "SD-4"];

/// The metadata-only fixture paper: a failed download, no source, no text.
const METADATA_ONLY_ID: &str = "P3000R0";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap_or_else(|e| panic!("create {}: {e}", to.display()));
    let entries = fs::read_dir(from).unwrap_or_else(|e| panic!("read {}: {e}", from.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| panic!("read entry: {e}"));
        let target = to.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target)
                .unwrap_or_else(|e| panic!("copy {}: {e}", entry.path().display()));
        }
    }
}

/// Copies the fixture corpus into a fresh temp workspace and runs `f` with
/// `$WG21_DATA_DIR` set to it.
///
/// Relative `source_file` and `markdown_path` values are rewritten to
/// absolute paths under the copy.
fn with_fixture<R>(f: impl FnOnce(&Path) -> R) -> R {
    let dir = TempDir::new().unwrap_or_else(|e| panic!("create temp dir: {e}"));
    let workspace = dir.path();
    let fixture = fixture_dir();
    fs::copy(
        fixture.join("paperstore.db"),
        workspace.join("paperstore.db"),
    )
    .unwrap_or_else(|e| panic!("copy fixture db: {e}"));
    copy_dir(&fixture.join("paperstore"), &workspace.join("paperstore"));

    let prefix = format!("{}/", workspace.display());
    let conn = Connection::open(workspace.join("paperstore.db"))
        .unwrap_or_else(|e| panic!("open fixture db: {e}"));
    for column in ["source_file", "markdown_path"] {
        conn.execute(
            &format!("UPDATE papers SET {column} = ?1 || {column} WHERE {column} != ''"),
            [&prefix],
        )
        .unwrap_or_else(|e| panic!("rewrite {column}: {e}"));
    }
    drop(conn);

    temp_env::with_var(WORKSPACE_ENV_VAR, Some(workspace), || f(workspace))
}

fn open_from_env() -> SqliteBackend {
    SqliteBackend::from_env().unwrap_or_else(|e| panic!("open backend from env: {e}"))
}

fn num(id: &str) -> PaperNum {
    PaperNum::parse(id).unwrap_or_else(|e| panic!("parse {id}: {e}"))
}

fn row<'a>(rows: &'a [PaperRow], id: &str) -> &'a PaperRow {
    rows.iter()
        .find(|r| r.paper_num == id)
        .unwrap_or_else(|| panic!("no row for {id}"))
}

fn file_name(path: &str) -> &str {
    Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_else(|| panic!("no file name in {path:?}"))
}

#[test]
fn fixture_opens_through_sqlite_backend() {
    with_fixture(|_| {
        let backend = open_from_env();
        let meta = backend
            .meta(&num("P1000R2"))
            .unwrap_or_else(|e| panic!("meta: {e}"));
        assert_eq!(meta.paper_num, "P1000R2");
        assert!(!meta.title.is_empty());
    });
}

#[test]
fn list_papers_returns_expected_rows() {
    with_fixture(|workspace| {
        let rows = open_from_env()
            .list_papers()
            .unwrap_or_else(|e| panic!("list_papers: {e}"));

        let ids: Vec<&str> = rows.iter().map(|r| r.paper_num.as_str()).collect();
        assert_eq!(ids, FIXTURE_PAPER_IDS);

        assert_eq!(row(&rows, "P1000R0").previous_version, "");
        assert_eq!(row(&rows, "P1000R1").previous_version, "p1000r0");
        assert_eq!(row(&rows, "P1000R2").previous_version, "p1000r1");

        assert_eq!(file_name(&row(&rows, "P2000R0").source_file), "p2000r0.pdf");
        for id in ["N5001", "P1000R0", "P1000R1", "P1000R2", "SD-4"] {
            assert_eq!(
                file_name(&row(&rows, id).source_file),
                format!("{}.html", id.to_lowercase()),
            );
        }

        for id in TEXT_BEARING_IDS {
            let r = row(&rows, id);
            assert_eq!(
                Path::new(&r.markdown_path),
                workspace.join(format!("paperstore/{}.md", id.to_lowercase())),
            );
            assert!(r.line_count > 0, "{id} line_count");
            assert!(r.status >= 0, "{id} status");
            assert!(!r.authors.is_empty(), "{id} authors");
            assert!(!r.mailing_date.is_empty(), "{id} mailing_date");
        }

        let failed = row(&rows, METADATA_ONLY_ID);
        assert!(failed.status < 0);
        assert!(!failed.error.is_empty());
        assert_eq!(failed.source_file, "");
        assert_eq!(failed.markdown_path, "");
        assert!(!failed.title.is_empty());
    });
}

#[test]
fn paper_md_reads_each_text_bearing_paper() {
    with_fixture(|_| {
        let backend = open_from_env();
        for id in TEXT_BEARING_IDS {
            let md = backend
                .paper_md(&num(id))
                .unwrap_or_else(|e| panic!("paper_md {id}: {e}"));
            assert!(md.starts_with("---\ntitle: "), "{id} front matter");
            assert!(md.contains(&format!("\ndocument: {id}\n")), "{id} document");
            let line_count = backend
                .meta(&num(id))
                .unwrap_or_else(|e| panic!("meta {id}: {e}"))
                .line_count;
            let md_lines = u64::try_from(md.split('\n').count())
                .unwrap_or_else(|e| panic!("line count {id}: {e}"));
            assert_eq!(md_lines, line_count, "{id} line_count");
        }
        assert!(matches!(
            backend.paper_md(&num(METADATA_ONLY_ID)),
            Err(Error::MissingMarkdown { .. })
        ));
    });
}

#[test]
fn fixture_markdown_covers_every_chunk_kind() {
    with_fixture(|_| {
        let backend = open_from_env();
        for id in TEXT_BEARING_IDS {
            let md = backend
                .paper_md(&num(id))
                .unwrap_or_else(|e| panic!("paper_md {id}: {e}"));
            assert!(md.contains("\n## "), "{id} heading");
            assert!(md.contains("\n```cpp\n"), "{id} fenced code block");
            assert!(md.contains("\n| --- |"), "{id} table");
            assert!(md.contains("\n## Wording\n"), "{id} wording section");
            assert!(
                md.contains("\n## Revision History\n"),
                "{id} revision history"
            );
        }
    });
}
