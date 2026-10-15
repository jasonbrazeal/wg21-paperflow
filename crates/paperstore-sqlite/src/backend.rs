//! The SQLite-backed [`StorageBackend`] implementation.

use std::path::Path;

use paperstore::{
    Error, PaperNum, PaperRow, Result, StorageBackend, decode_authors, default_workspace_dir,
};
use rusqlite::{Connection, OptionalExtension};

use crate::schema;

const SELECT_PAPER_COLUMNS: &str = "SELECT paper_id, year, title, authors, target_group, intent, url, \
     document_date, mailing_date, disposition, previous_version, \
     source_file, markdown_path, dissect_path, agora_path, assay_path, \
     citations_extracted_at, line_count, status, error \
     FROM papers";

/// SQLite-backed [`StorageBackend`] over `<workspace_dir>/paperstore.db`.
#[derive(Debug)]
pub struct SqliteBackend {
    conn: Connection,
}

impl SqliteBackend {
    /// Opens the database under `workspace_dir`, creating the file if
    /// missing, sets `PRAGMA busy_timeout = 5000` (matching the Python
    /// backend), and ensures the schema.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Backend`] when SQLite cannot open or migrate the
    /// database.
    pub fn new(workspace_dir: &Path) -> Result<Self> {
        let conn = Connection::open(workspace_dir.join("paperstore.db")).map_err(backend_err)?;
        conn.execute_batch("PRAGMA busy_timeout = 5000")
            .map_err(backend_err)?;
        schema::ensure_schema(&conn).map_err(backend_err)?;
        Ok(Self { conn })
    }

    /// Opens the database under `$WG21_DATA_DIR`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WorkspaceEnvUnset`] when the variable is unset or
    /// empty, and [`Error::Backend`] when the database cannot be opened.
    pub fn from_env() -> Result<Self> {
        Self::new(&default_workspace_dir()?)
    }
}

impl StorageBackend for SqliteBackend {
    /// Returns the metadata row for `num`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingPaper`] when no row exists for `num`, and
    /// [`Error::Backend`] when the query fails.
    fn meta(&self, num: &PaperNum) -> Result<PaperRow> {
        let mut stmt = self
            .conn
            .prepare(&format!("{SELECT_PAPER_COLUMNS} WHERE paper_id = ?"))
            .map_err(backend_err)?;
        stmt.query_row([num.as_str()], row_to_paper)
            .optional()
            .map_err(backend_err)?
            .ok_or_else(|| Error::missing_paper(num.clone()))
    }

    /// Returns the converted markdown for `num`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingMarkdown`] when the paper has no converted
    /// markdown or the file is absent on disk, [`Error::Backend`] when the
    /// query fails, and [`Error::Io`] when the file cannot be read.
    fn paper_md(&self, num: &PaperNum) -> Result<String> {
        // Port of `get_paper_md` (sqlite_backend.py lines 1160-1176). The
        // step-1 `Error::MissingMarkdown` variant carries only the paper
        // number, so the missing-on-disk case cannot name the path the way
        // the Python message does; the path is available via `meta`.
        let path: Option<String> = self
            .conn
            .query_row(
                "SELECT markdown_path FROM papers WHERE paper_id = ?",
                [num.as_str()],
                |row| row.get(0),
            )
            .optional()
            .map_err(backend_err)?;
        let path = path
            .filter(|p| !p.is_empty())
            .ok_or_else(|| Error::missing_markdown(num.clone()))?;
        if !Path::new(&path).exists() {
            return Err(Error::missing_markdown(num.clone()));
        }
        Ok(std::fs::read_to_string(&path)?)
    }

    /// Returns every metadata row, ordered by `paper_id`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Backend`] when the query fails.
    fn list_papers(&self) -> Result<Vec<PaperRow>> {
        let mut stmt = self
            .conn
            .prepare(&format!("{SELECT_PAPER_COLUMNS} ORDER BY paper_id"))
            .map_err(backend_err)?;
        stmt.query_map([], row_to_paper)
            .map_err(backend_err)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(backend_err)
    }
}

fn row_to_paper(row: &rusqlite::Row<'_>) -> rusqlite::Result<PaperRow> {
    let authors: String = row.get(3)?;
    let line_count: i64 = row.get(17)?;
    Ok(PaperRow {
        paper_num: row.get(0)?,
        year: row.get(1)?,
        title: row.get(2)?,
        authors: decode_authors(&authors),
        target_group: row.get(4)?,
        intent: row.get(5)?,
        url: row.get(6)?,
        document_date: row.get(7)?,
        mailing_date: row.get(8)?,
        disposition: row.get(9)?,
        previous_version: row.get(10)?,
        source_file: row.get(11)?,
        markdown_path: row.get(12)?,
        dissect_path: row.get(13)?,
        agora_path: row.get(14)?,
        assay_path: row.get(15)?,
        citations_extracted_at: row.get(16)?,
        line_count: u64::try_from(line_count)
            .map_err(|_| rusqlite::Error::IntegralValueOutOfRange(17, line_count))?,
        status: row.get(18)?,
        error: row.get(19)?,
    })
}

fn backend_err(e: rusqlite::Error) -> Error {
    // The `#[from]` impl is on the boxed trait object; the coercion has to
    // be named before `Error::from` will resolve to it.
    let boxed: Box<dyn std::error::Error + Send + Sync> = Box::new(e);
    Error::from(boxed)
}
