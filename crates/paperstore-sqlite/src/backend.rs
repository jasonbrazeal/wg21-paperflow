//! The SQLite-backed [`StorageBackend`] implementation.

use std::path::{Path, PathBuf};

use paperstore::{
    Error, PaperNum, PaperRow, Result, StorageBackend, decode_authors, default_workspace_dir,
};
use rusqlite::{Connection, OptionalExtension};

use crate::schema;

/// SQLite-backed [`StorageBackend`] over `<workspace_dir>/paperstore.db`.
#[derive(Debug)]
pub struct SqliteBackend {
    conn: Connection,
    workspace_dir: PathBuf,
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
        Ok(Self {
            conn,
            workspace_dir: workspace_dir.to_path_buf(),
        })
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
            .prepare(&format!(
                "SELECT {ROW_COLUMNS} FROM papers WHERE paper_id = ?"
            ))
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

    /// Returns every stored paper number, sorted ascending.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Backend`] when the query fails.
    fn list_all_paper_ids(&self) -> Result<Vec<PaperNum>> {
        self.query_paper_ids("SELECT paper_id FROM papers", [])
    }

    /// Returns the paper numbers for mailing year `year`, sorted ascending.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Backend`] when the query fails.
    fn list_paper_ids_for_year(&self, year: u16) -> Result<Vec<PaperNum>> {
        self.query_paper_ids(
            "SELECT paper_id FROM papers WHERE year = ?",
            [year.to_string()],
        )
    }

    /// Returns every revision of the paper number of `num`, newest first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Backend`] when the query fails.
    fn list_revisions(&self, num: &PaperNum) -> Result<Vec<PaperRow>> {
        let (base, _) = num.split_revision();
        let mut stmt = self
            .conn
            .prepare(&format!(
                "SELECT {ROW_COLUMNS} FROM papers \
                 WHERE paper_id = ?1 OR paper_id LIKE ?1 || 'R%'"
            ))
            .map_err(backend_err)?;
        let rows = stmt
            .query_map([base], row_to_paper)
            .map_err(backend_err)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(backend_err)?;
        // LIKE does not check that only digits follow the `R`; dropping rows
        // that fail to parse keeps exactly this number's revisions, since any
        // matching id that parses normalizes to `<base>` or `<base>R<digits>`.
        let mut keyed: Vec<(Option<u32>, PaperRow)> = rows
            .into_iter()
            .filter_map(|row| {
                let rev = PaperNum::parse(&row.paper_num).ok()?.split_revision().1;
                Some((rev, row))
            })
            .collect();
        keyed.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| a.1.paper_num.cmp(&b.1.paper_num))
        });
        Ok(keyed.into_iter().map(|(_, row)| row).collect())
    }

    fn tool_dir(&self, tool: &str) -> PathBuf {
        self.workspace_dir.join(tool)
    }
}

const ROW_COLUMNS: &str = "paper_id, year, title, authors, target_group, intent, url, \
     document_date, mailing_date, disposition, previous_version, \
     source_file, markdown_path, dissect_path, agora_path, assay_path, \
     citations_extracted_at, line_count, status, error";

impl SqliteBackend {
    fn query_paper_ids(&self, sql: &str, params: impl rusqlite::Params) -> Result<Vec<PaperNum>> {
        let mut stmt = self.conn.prepare(sql).map_err(backend_err)?;
        let raw = stmt
            .query_map(params, |row| row.get::<_, String>(0))
            .map_err(backend_err)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(backend_err)?;
        // Non-paper-number ids such as `SD-1` live in the same table; they
        // are not addressable through `PaperNum`, so listings skip them.
        let mut ids: Vec<PaperNum> = raw
            .iter()
            .filter_map(|id| PaperNum::parse(id).ok())
            .collect();
        ids.sort();
        ids.dedup();
        Ok(ids)
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
