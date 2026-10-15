//! Schema DDL and migrations for the `papers` table.
//!
//! The DDL is copied column-for-column from the `_SCHEMA` string in the
//! Python `paperstore.sqlite_backend`, and [`ensure_schema`] ports the
//! `_migrate`-style guards that add `assay_path` and
//! `citations_extracted_at` to older databases.

use rusqlite::Connection;

/// The `papers` table, exactly as in the Python `_SCHEMA` string.
///
/// `assay_path` is absent on purpose: Python-created databases received it
/// through a migration, so [`ensure_schema`] adds it the same way.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS papers (
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
    dissect_path     TEXT DEFAULT '',  -- deprecated, kept for migration safety
    advocatus_path   TEXT DEFAULT '',  -- deprecated, kept for migration safety
    agora_path             TEXT DEFAULT '',
    citations_extracted_at TEXT DEFAULT '',
    line_count             INTEGER DEFAULT 0,
    status                 INTEGER NOT NULL DEFAULT 0,
    error                  TEXT DEFAULT '',
    updated_at             TEXT DEFAULT ''
);
";

/// Creates the `papers` table if needed, then adds `assay_path` and
/// `citations_extracted_at` when `pragma_table_info` shows them missing
/// (the `_migrate` guards).
pub(crate) fn ensure_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(SCHEMA)?;
    let mut stmt = conn.prepare("PRAGMA table_info(papers)")?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    if !columns.iter().any(|column| column == "assay_path") {
        conn.execute_batch("ALTER TABLE papers ADD COLUMN assay_path TEXT DEFAULT ''")?;
    }
    if !columns
        .iter()
        .any(|column| column == "citations_extracted_at")
    {
        conn.execute_batch("ALTER TABLE papers ADD COLUMN citations_extracted_at TEXT DEFAULT ''")?;
    }
    Ok(())
}
