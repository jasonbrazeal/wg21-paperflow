//! The abstract storage backend trait.

use std::path::PathBuf;

use crate::error::Result;
use crate::num::PaperNum;
use crate::row::PaperRow;

/// Abstract storage interface for paperflow artifacts.
///
/// Port of the Python `StorageBackend` ABC, reduced to the read slice the
/// Rust workspace needs. Consumers depend on this trait; only backend
/// crates touch SQL or paths.
pub trait StorageBackend {
    /// Returns the metadata row for `num`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::MissingPaper`] when no row exists for `num`.
    fn meta(&self, num: &PaperNum) -> Result<PaperRow>;

    /// Returns the converted markdown for `num`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::MissingMarkdown`] when the paper has no
    /// converted markdown.
    fn paper_md(&self, num: &PaperNum) -> Result<String>;

    /// Returns every stored paper number, sorted ascending.
    ///
    /// Stored ids that are not valid paper numbers (e.g. `SD-1`) are
    /// skipped.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Backend`] when the store cannot be read.
    fn list_all_paper_ids(&self) -> Result<Vec<PaperNum>>;

    /// Returns the paper numbers whose mailing year is `year`, sorted
    /// ascending, with the same skipping rule as
    /// [`StorageBackend::list_all_paper_ids`].
    ///
    /// A year with no papers gives an empty list.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Backend`] when the store cannot be read.
    fn list_paper_ids_for_year(&self, year: u16) -> Result<Vec<PaperNum>>;

    /// Returns the metadata row of every stored revision of the paper number
    /// of `num`, newest first by numeric revision.
    ///
    /// The revision of `num` itself is ignored, and a row with no revision
    /// sorts last. An unknown paper number gives an empty list.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Backend`] when the store cannot be read.
    fn list_revisions(&self, num: &PaperNum) -> Result<Vec<PaperRow>>;

    /// Returns the per-tool directory for `tool` under the workspace.
    ///
    /// The directory is not created. `tool` should be a single path
    /// component.
    fn tool_dir(&self, tool: &str) -> PathBuf;
}
