//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Building and maintaining the LanceDB chunk index.
//!
//! The index lives in [`INDEX_DIR_NAME`] under the paperstore workspace,
//! beside (never inside) paperstore's own files. It holds three tables:
//!
//! - [`CHUNKS_TABLE`]: one row per chunk of every text-bearing paper, plus
//!   one `has_text = false` row per metadata-only paper.
//! - [`PAPERS_TABLE`]: the per-paper manifest recording what each paper's
//!   rows were built from (content hash, chunker version, model id).
//! - [`MANIFEST_TABLE`]: one row recording the model id, vector dimension,
//!   and chunker version of the whole index, written when a run completes.
//!
//! [`Indexer::run`] is incremental and resumable: a paper is re-chunked and
//! re-embedded only when its Markdown, its indexed paperstore metadata, the
//! chunker version, or the model id changed, and each paper's chunks are committed before its manifest row, so
//! an interrupted run leaves completed papers indexed.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use futures::TryStreamExt;
use lancedb::arrow::arrow_array::builder::{
    BooleanBuilder, FixedSizeListBuilder, Float32Builder, Int32Builder, ListBuilder, StringBuilder,
    UInt32Builder,
};
use lancedb::arrow::arrow_array::cast::AsArray;
use lancedb::arrow::arrow_array::types::UInt32Type;
use lancedb::arrow::arrow_array::{ArrayRef, RecordBatch, TimestampMicrosecondArray};
use lancedb::arrow::arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef, TimeUnit};
use lancedb::index::Index;
use lancedb::index::scalar::{BTreeIndexBuilder, BitmapIndexBuilder, FtsIndexBuilder};
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::{AddDataMode, OptimizeAction};
use lancedb::{Connection, Table};
use paperstore::{PaperNum, PaperRow, StorageBackend, default_workspace_dir};
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::chunk::{CHUNKER_VERSION, Chunk, PrefixMeta, chunk_markdown};
use crate::embed::{EmbedError, Embedder};
use crate::revision::{latest_set, revision_number, revisionless_id};

/// Directory under the paperstore workspace holding the index.
pub const INDEX_DIR_NAME: &str = "paperindex";

/// Table of chunk rows searched by the server.
pub const CHUNKS_TABLE: &str = "chunks";

/// Per-paper manifest table.
pub const PAPERS_TABLE: &str = "papers";

/// Index-level manifest table.
pub const MANIFEST_TABLE: &str = "index_manifest";

/// Command that rebuilds the index for the current model and chunker.
pub const REBUILD_COMMAND: &str = "paper-mcp index";

/// Separator joining a paper's authors in the `authors` column.
pub const AUTHORS_SEPARATOR: &str = "; ";

/// Filter columns given a B-tree scalar index (high cardinality or ranges).
const BTREE_COLUMNS: [&str; 5] = [
    "paper_id",
    "revisionless_id",
    "year",
    "mailing_date",
    "target_group",
];

/// Filter columns given a bitmap scalar index (low cardinality).
const BITMAP_COLUMNS: [&str; 4] = ["is_latest", "has_text", "intent", "chunk_kind"];

/// Column holding the full-text-indexed chunk text.
pub(crate) const FTS_COLUMN: &str = "indexed_text";

/// Column holding chunk embeddings.
pub(crate) const VECTOR_COLUMN: &str = "vector";

/// Chunk ID suffix of a metadata-only paper's single row.
const METADATA_CHUNK_SUFFIX: &str = "meta";

/// Errors from building, opening, or checking the index.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    /// LanceDB failed to read or write the index.
    #[error("index storage error: {0}")]
    Lance(Box<lancedb::Error>),
    /// An Arrow batch could not be built or read.
    #[error("index data error: {0}")]
    Arrow(#[from] ArrowError),
    /// Paperstore could not be read.
    #[error(transparent)]
    Store(#[from] paperstore::Error),
    /// The embedder failed.
    #[error(transparent)]
    Embed(#[from] EmbedError),
    /// The index path is not valid UTF-8, which LanceDB requires.
    #[error("index path {0:?} is not valid UTF-8")]
    NonUtf8Path(PathBuf),
    /// A table's columns do not have the types this build writes.
    #[error(
        "index table {table} has an unexpected schema: {detail}; rebuild with `{REBUILD_COMMAND}`"
    )]
    Schema {
        /// Table name.
        table: &'static str,
        /// What was wrong.
        detail: String,
    },
    /// A value does not fit its index column.
    #[error("{0} does not fit in its index column")]
    OutOfRange(&'static str),
    /// The index has no manifest: it was never built or its last run did not
    /// complete.
    #[error("the paper index has no manifest; build it with `{REBUILD_COMMAND}`")]
    ManifestMissing,
    /// The index was built with a different model or chunker.
    #[error(
        "the paper index was built with {field} {indexed} but this build uses {current}; rebuild it with `{REBUILD_COMMAND}`"
    )]
    ManifestMismatch {
        /// The mismatched setting.
        field: ManifestField,
        /// Value recorded in the index.
        indexed: String,
        /// Value of the running build.
        current: String,
    },
}

impl From<lancedb::Error> for IndexError {
    fn from(e: lancedb::Error) -> Self {
        Self::Lance(Box::new(e))
    }
}

/// A setting recorded in the index manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ManifestField {
    /// Embedding model identifier.
    ModelId,
    /// Embedding vector dimension.
    Dim,
    /// Chunker version.
    ChunkerVersion,
}

impl fmt::Display for ManifestField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ModelId => "embedding model",
            Self::Dim => "embedding dimension",
            Self::ChunkerVersion => "chunker version",
        })
    }
}

/// A paper the indexer skipped because it could not be indexed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaperFailure {
    /// Paper ID as stored in paperstore.
    pub paper_id: String,
    /// Description of the failure.
    pub error: String,
}

/// Outcome of one [`Indexer::run`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexReport {
    /// Papers chunked and embedded this run, in paper ID order.
    pub embedded: Vec<String>,
    /// Papers stored as metadata-only rows this run, in paper ID order.
    pub metadata_only: Vec<String>,
    /// Number of papers whose rows were already current.
    pub unchanged: usize,
    /// Papers no longer in paperstore whose rows were deleted.
    pub removed: Vec<String>,
    /// Papers skipped because of an error. Their rows were deleted, so they
    /// are absent from the index until a later run succeeds.
    pub failures: Vec<PaperFailure>,
}

impl IndexReport {
    /// Return whether every paper was indexed.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.failures.is_empty()
    }
}

impl fmt::Display for IndexReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} embedded, {} metadata-only, {} unchanged, {} removed, {} failed",
            self.embedded.len(),
            self.metadata_only.len(),
            self.unchanged,
            self.removed.len(),
            self.failures.len(),
        )?;
        for failure in &self.failures {
            write!(f, "\n  {}: {}", failure.paper_id, failure.error)?;
        }
        Ok(())
    }
}

/// Return the index directory for a paperstore workspace.
#[must_use]
pub fn index_dir(workspace: &Path) -> PathBuf {
    workspace.join(INDEX_DIR_NAME)
}

/// Return the index directory under `$WG21_DATA_DIR`.
///
/// # Errors
///
/// Returns [`IndexError::Store`] when `$WG21_DATA_DIR` is unset or empty.
pub fn default_index_dir() -> Result<PathBuf, IndexError> {
    Ok(index_dir(&default_workspace_dir()?))
}

/// What one paper's rows were built from.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ManifestRow {
    content_hash: String,
    metadata_hash: String,
    chunker_version: u32,
    model_id: String,
}

/// Settings recorded in the index-level manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
struct IndexManifest {
    model_id: String,
    dim: u32,
    chunker_version: u32,
}

/// An opened paper index directory.
pub struct PaperIndex {
    conn: Connection,
    dir: PathBuf,
}

impl fmt::Debug for PaperIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PaperIndex")
            .field("dir", &self.dir)
            .finish_non_exhaustive()
    }
}

impl PaperIndex {
    /// Open the index at `dir`. Tables are created by the first
    /// [`Indexer::run`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::NonUtf8Path`] for a non-UTF-8 path and
    /// [`IndexError::Lance`] when the directory cannot be opened.
    pub async fn open(dir: &Path) -> Result<Self, IndexError> {
        let uri = dir
            .to_str()
            .ok_or_else(|| IndexError::NonUtf8Path(dir.to_path_buf()))?;
        let conn = lancedb::connect(uri).execute().await?;
        Ok(Self {
            conn,
            dir: dir.to_path_buf(),
        })
    }

    /// Directory the index lives in.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Open the chunks table.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::Lance`] when the table does not exist or cannot
    /// be opened.
    pub async fn chunks_table(&self) -> Result<Table, IndexError> {
        Ok(self.conn.open_table(CHUNKS_TABLE).execute().await?)
    }

    /// Open the per-paper manifest table.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::Lance`] when the table does not exist or cannot
    /// be opened.
    pub async fn papers_table(&self) -> Result<Table, IndexError> {
        Ok(self.conn.open_table(PAPERS_TABLE).execute().await?)
    }

    /// Check that the index was completely built for `embedder` and the
    /// current [`CHUNKER_VERSION`].
    ///
    /// # Errors
    ///
    /// Returns [`IndexError::ManifestMissing`] when no run has completed, and
    /// [`IndexError::ManifestMismatch`] naming the first differing setting
    /// (model id, then dimension, then chunker version). Both name
    /// [`REBUILD_COMMAND`].
    pub async fn check_manifest(&self, embedder: &dyn Embedder) -> Result<(), IndexError> {
        self.check_manifest_with(embedder, CHUNKER_VERSION).await
    }

    /// Check the manifest against `embedder` and an explicit chunker version.
    ///
    /// # Errors
    ///
    /// As [`PaperIndex::check_manifest`].
    pub async fn check_manifest_with(
        &self,
        embedder: &dyn Embedder,
        chunker_version: u32,
    ) -> Result<(), IndexError> {
        let current = IndexManifest {
            model_id: embedder.model_id().to_owned(),
            dim: u32::try_from(embedder.dim()).map_err(|_| IndexError::OutOfRange("dim"))?,
            chunker_version,
        };
        let indexed = self
            .read_index_manifest()
            .await?
            .ok_or(IndexError::ManifestMissing)?;
        let mismatch = |field, indexed: String, current: String| IndexError::ManifestMismatch {
            field,
            indexed,
            current,
        };
        if indexed.model_id != current.model_id {
            return Err(mismatch(
                ManifestField::ModelId,
                indexed.model_id,
                current.model_id,
            ));
        }
        if indexed.dim != current.dim {
            return Err(mismatch(
                ManifestField::Dim,
                indexed.dim.to_string(),
                current.dim.to_string(),
            ));
        }
        if indexed.chunker_version != current.chunker_version {
            return Err(mismatch(
                ManifestField::ChunkerVersion,
                indexed.chunker_version.to_string(),
                current.chunker_version.to_string(),
            ));
        }
        Ok(())
    }

    async fn table_names(&self) -> Result<BTreeSet<String>, IndexError> {
        Ok(self
            .conn
            .table_names()
            .execute()
            .await?
            .into_iter()
            .collect())
    }

    async fn read_index_manifest(&self) -> Result<Option<IndexManifest>, IndexError> {
        if !self.table_names().await?.contains(MANIFEST_TABLE) {
            return Ok(None);
        }
        let table = self.conn.open_table(MANIFEST_TABLE).execute().await?;
        let batches = collect(&table, None, &["model_id", "dim", "chunker_version"]).await?;
        let Some(batch) = batches.iter().find(|b| b.num_rows() > 0) else {
            return Ok(None);
        };
        let model_ids = string_column(batch, MANIFEST_TABLE, "model_id")?;
        let dims = u32_column(batch, MANIFEST_TABLE, "dim")?;
        let versions = u32_column(batch, MANIFEST_TABLE, "chunker_version")?;
        Ok(Some(IndexManifest {
            model_id: model_ids.value(0).to_owned(),
            dim: dims.value(0),
            chunker_version: versions.value(0),
        }))
    }

    /// Create missing tables for vectors of `dim`, dropping the chunk and
    /// paper tables when their vectors have a different dimension, and
    /// clear the index manifest when it does not match `current`.
    async fn prepare(&self, current: &IndexManifest) -> Result<(), IndexError> {
        let mut names = self.table_names().await?;
        let dim = i32::try_from(current.dim).map_err(|_| IndexError::OutOfRange("dim"))?;
        let stale_chunks = names.contains(CHUNKS_TABLE) && {
            let schema = self.chunks_table().await?.schema().await?;
            vector_dim(&schema) != Some(current.dim) || !same_columns(&schema, &chunks_schema(dim))
        };
        let stale_papers = names.contains(PAPERS_TABLE)
            && !same_columns(
                &self.papers_table().await?.schema().await?,
                &papers_schema(),
            );
        if stale_chunks || stale_papers {
            info!(
                "vector dimension is not {} or the chunk or paper table layout changed; dropping chunk and paper tables",
                current.dim
            );
            for name in [CHUNKS_TABLE, PAPERS_TABLE] {
                if names.remove(name) {
                    self.conn.drop_table(name, &[]).await?;
                }
            }
        }
        for (name, schema) in [
            (CHUNKS_TABLE, chunks_schema(dim)),
            (PAPERS_TABLE, papers_schema()),
            (MANIFEST_TABLE, manifest_schema()),
        ] {
            if !names.contains(name) {
                self.conn.create_empty_table(name, schema).execute().await?;
            }
        }
        if let Some(indexed) = self.read_index_manifest().await?
            && indexed != *current
        {
            // A run that changes settings and then dies must not leave a
            // manifest vouching for a partly rebuilt index.
            self.conn
                .open_table(MANIFEST_TABLE)
                .execute()
                .await?
                .delete("true")
                .await?;
        }
        Ok(())
    }

    async fn write_index_manifest(&self, manifest: &IndexManifest) -> Result<(), IndexError> {
        let batch = RecordBatch::try_new(
            manifest_schema(),
            vec![
                string_array([manifest.model_id.as_str()]),
                u32_array([manifest.dim]),
                u32_array([manifest.chunker_version]),
            ],
        )?;
        self.conn
            .open_table(MANIFEST_TABLE)
            .execute()
            .await?
            .add(batch)
            .mode(AddDataMode::Overwrite)
            .execute()
            .await?;
        Ok(())
    }
}

/// Builds and incrementally refreshes a [`PaperIndex`] from paperstore.
pub struct Indexer<'a, B: StorageBackend + ?Sized, E: Embedder + ?Sized> {
    backend: &'a B,
    embedder: &'a E,
    index: &'a PaperIndex,
    chunker_version: u32,
}

impl<B: StorageBackend + ?Sized, E: Embedder + ?Sized> fmt::Debug for Indexer<'_, B, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Indexer")
            .field("model_id", &self.embedder.model_id())
            .field("index", &self.index)
            .field("chunker_version", &self.chunker_version)
            .finish_non_exhaustive()
    }
}

/// Paper metadata copied onto each of the paper's chunk rows.
struct PaperFields {
    paper_id: String,
    title: String,
    revisionless_id: String,
    revision: Option<u32>,
    is_latest: bool,
    year: Option<i32>,
    mailing_date: String,
    target_group: String,
    authors: String,
    intent: String,
}

/// Rows to write for one paper.
struct PaperRows {
    batch: RecordBatch,
    has_text: bool,
}

impl<'a, B: StorageBackend + ?Sized, E: Embedder + ?Sized> Indexer<'a, B, E> {
    /// Create an indexer reading `backend`, embedding with `embedder`, and
    /// writing `index`, using the current [`CHUNKER_VERSION`].
    pub fn new(backend: &'a B, embedder: &'a E, index: &'a PaperIndex) -> Self {
        Self {
            backend,
            embedder,
            index,
            chunker_version: CHUNKER_VERSION,
        }
    }

    /// Record `version` as the chunker version instead of
    /// [`CHUNKER_VERSION`]. Papers indexed under another version are
    /// re-chunked.
    #[must_use]
    pub fn with_chunker_version(mut self, version: u32) -> Self {
        self.chunker_version = version;
        self
    }

    /// Bring the index up to date with paperstore.
    ///
    /// Papers are processed one at a time in paper ID order. A paper whose
    /// Markdown hash, indexed metadata (title, authors, year, mailing date,
    /// target group, intent), chunker version, and model id match its
    /// manifest row is left alone; any other paper has its rows replaced.
    /// Papers with `status < 0` or no Markdown get one `has_text = false`
    /// row. Papers no longer in paperstore are deleted, including chunk rows
    /// left without a manifest row by an interrupted run. A paper that fails is logged, has
    /// its rows deleted, and is listed in [`IndexReport::failures`]; the run
    /// continues. Afterwards `is_latest` is recomputed for every row, the
    /// full-text and scalar indexes are rebuilt, and the index manifest is
    /// written. Paperstore is never written.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError`] when paperstore cannot be listed or the index
    /// itself cannot be read or written. Per-paper problems are reported in
    /// the returned [`IndexReport`] instead.
    pub async fn run(&self) -> Result<IndexReport, IndexError> {
        let manifest = IndexManifest {
            model_id: self.embedder.model_id().to_owned(),
            dim: u32::try_from(self.embedder.dim()).map_err(|_| IndexError::OutOfRange("dim"))?,
            chunker_version: self.chunker_version,
        };
        let rows = self.backend.list_papers()?;
        let latest = latest_set(&rows);
        self.index.prepare(&manifest).await?;
        let chunks = self.index.chunks_table().await?;
        let papers = self.index.papers_table().await?;
        let existing = read_paper_manifest(&papers).await?;
        let mut report = IndexReport::default();

        let present: BTreeSet<&str> = rows.iter().map(|r| r.paper_num.as_str()).collect();
        let mut indexed_ids = read_chunk_paper_ids(&chunks).await?;
        indexed_ids.extend(existing.keys().cloned());
        for paper_id in indexed_ids
            .iter()
            .filter(|id| !present.contains(id.as_str()))
        {
            purge(&chunks, &papers, paper_id).await?;
            info!(paper_id = %paper_id, "removed paper no longer in paperstore");
            report.removed.push(paper_id.clone());
        }

        for row in &rows {
            let paper_id = row.paper_num.as_str();
            match self
                .index_paper(row, existing.get(paper_id), &latest, &chunks, &papers)
                .await
            {
                Ok(None) => report.unchanged += 1,
                Ok(Some(true)) => report.embedded.push(paper_id.to_owned()),
                Ok(Some(false)) => report.metadata_only.push(paper_id.to_owned()),
                Err(e) => {
                    warn!(paper_id = %paper_id, error = %e, "skipping paper");
                    purge(&chunks, &papers, paper_id).await?;
                    report.failures.push(PaperFailure {
                        paper_id: paper_id.to_owned(),
                        error: e.to_string(),
                    });
                }
            }
        }

        refresh_latest(&chunks, &latest).await?;
        for table in [&chunks, &papers] {
            table.optimize(OptimizeAction::All).await?;
        }
        rebuild_indexes(&chunks).await?;
        self.index.write_index_manifest(&manifest).await?;
        info!(%report, "index run complete");
        Ok(report)
    }

    /// Index one paper, returning `None` when its rows were current, or
    /// `Some(has_text)` after replacing them.
    async fn index_paper(
        &self,
        row: &PaperRow,
        existing: Option<&ManifestRow>,
        latest: &BTreeSet<String>,
        chunks: &Table,
        papers: &Table,
    ) -> Result<Option<bool>, IndexError> {
        let num = PaperNum::parse(&row.paper_num)?;
        let markdown = self.read_markdown(row, &num)?;
        let content_hash = markdown.as_deref().map(sha256_hex).unwrap_or_default();
        let current = ManifestRow {
            content_hash,
            metadata_hash: metadata_hash(row),
            chunker_version: self.chunker_version,
            model_id: self.embedder.model_id().to_owned(),
        };
        if existing == Some(&current) {
            return Ok(None);
        }
        let fields = paper_fields(row, &num, latest);
        let rows = self.build_rows(row, &fields, markdown.as_deref())?;
        let manifest = manifest_batch(&fields.paper_id, &current)?;
        commit_paper(chunks, papers, &fields.paper_id, rows.batch, manifest).await?;
        Ok(Some(rows.has_text))
    }

    /// Read a paper's Markdown, or `None` when it is metadata-only.
    fn read_markdown(&self, row: &PaperRow, num: &PaperNum) -> Result<Option<String>, IndexError> {
        if row.status < 0 || row.markdown_path.trim().is_empty() {
            return Ok(None);
        }
        match self.backend.paper_md(num) {
            Ok(md) => Ok(Some(md)),
            Err(paperstore::Error::MissingMarkdown { .. }) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn build_rows(
        &self,
        row: &PaperRow,
        fields: &PaperFields,
        markdown: Option<&str>,
    ) -> Result<PaperRows, IndexError> {
        let meta = PrefixMeta {
            paper_id: fields.paper_id.clone(),
            title: row.title.clone(),
            authors: row.authors.clone(),
        };
        let paper_chunks = markdown
            .map(|md| chunk_markdown(md, &meta))
            .unwrap_or_default();
        if paper_chunks.is_empty() {
            return Ok(PaperRows {
                batch: chunk_batch(fields, &[], &[], self.embedder.dim())?,
                has_text: false,
            });
        }
        let texts: Vec<String> = paper_chunks
            .iter()
            .map(|c| c.indexed_text.clone())
            .collect();
        let vectors = self.embedder.embed_documents(&texts)?;
        if vectors.len() != texts.len() {
            return Err(EmbedError::CountMismatch {
                expected: texts.len(),
                actual: vectors.len(),
            }
            .into());
        }
        if let Some(bad) = vectors.iter().find(|v| v.len() != self.embedder.dim()) {
            return Err(EmbedError::DimMismatch {
                expected: self.embedder.dim(),
                actual: bad.len(),
            }
            .into());
        }
        Ok(PaperRows {
            batch: chunk_batch(fields, &paper_chunks, &vectors, self.embedder.dim())?,
            has_text: true,
        })
    }
}

fn paper_fields(row: &PaperRow, num: &PaperNum, latest: &BTreeSet<String>) -> PaperFields {
    PaperFields {
        paper_id: num.as_str().to_owned(),
        title: row.title.clone(),
        revisionless_id: revisionless_id(num),
        revision: revision_number(num),
        is_latest: latest.contains(num.as_str()),
        year: parse_year(num, &row.year),
        mailing_date: row.mailing_date.clone(),
        target_group: row.target_group.clone(),
        authors: row.authors.join(AUTHORS_SEPARATOR),
        intent: row.intent.clone(),
    }
}

/// Parse a paperstore year, treating an empty value as unknown and logging a
/// non-empty value that is not a year.
fn parse_year(num: &PaperNum, year: &str) -> Option<i32> {
    let year = year.trim();
    if year.is_empty() {
        return None;
    }
    match year.parse() {
        Ok(parsed) => Some(parsed),
        Err(e) => {
            warn!(paper_id = %num.as_str(), year, error = %e, "unparseable year stored as null");
            None
        }
    }
}

fn sha256_hex(text: &str) -> String {
    hex(&Sha256::digest(text.as_bytes()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut hex, b| {
        let _ = write!(hex, "{b:02x}");
        hex
    })
}

/// SHA-256 of the paperstore metadata copied into a paper's rows or its
/// contextual prefix. `is_latest` is excluded: it is recomputed every run.
fn metadata_hash(row: &PaperRow) -> String {
    let mut hasher = Sha256::new();
    let mut field = |value: &str| {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    };
    field(&row.title);
    field(&row.year);
    field(&row.mailing_date);
    field(&row.target_group);
    field(&row.intent);
    field(&row.authors.len().to_string());
    for author in &row.authors {
        field(author);
    }
    hex(&hasher.finalize())
}

/// Replace a paper's rows: chunk rows first, then its manifest row, so a
/// crash between the two leaves a paper with no manifest row, which the next
/// run rewrites.
async fn commit_paper(
    chunks: &Table,
    papers: &Table,
    paper_id: &str,
    batch: RecordBatch,
    manifest: RecordBatch,
) -> Result<(), IndexError> {
    purge(chunks, papers, paper_id).await?;
    chunks.add(batch).execute().await?;
    papers.add(manifest).execute().await?;
    Ok(())
}

/// Quote `value` as an SQL string literal.
fn sql_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn eq_predicate(column: &str, value: &str) -> String {
    format!("{column} = {}", sql_string(value))
}

/// Delete a paper's chunk rows and manifest row, manifest first.
async fn purge(chunks: &Table, papers: &Table, paper_id: &str) -> Result<(), IndexError> {
    let predicate = eq_predicate("paper_id", paper_id);
    papers.delete(&predicate).await?;
    chunks.delete(&predicate).await?;
    Ok(())
}

/// Set `is_latest` on every chunk row to membership in `latest`, touching
/// only rows whose flag is wrong.
async fn refresh_latest(chunks: &Table, latest: &BTreeSet<String>) -> Result<(), IndexError> {
    if latest.is_empty() {
        chunks
            .update()
            .only_if("is_latest = true")
            .column("is_latest", "false")
            .execute()
            .await?;
        return Ok(());
    }
    let ids = latest
        .iter()
        .map(|id| sql_string(id))
        .collect::<Vec<_>>()
        .join(", ");
    chunks
        .update()
        .only_if(format!("is_latest = false AND paper_id IN ({ids})"))
        .column("is_latest", "true")
        .execute()
        .await?;
    chunks
        .update()
        .only_if(format!("is_latest = true AND paper_id NOT IN ({ids})"))
        .column("is_latest", "false")
        .execute()
        .await?;
    Ok(())
}

async fn rebuild_indexes(chunks: &Table) -> Result<(), IndexError> {
    if chunks.count_rows(None).await? == 0 {
        return Ok(());
    }
    chunks
        .create_index(&[FTS_COLUMN], Index::FTS(FtsIndexBuilder::default()))
        .replace(true)
        .execute()
        .await?;
    for column in BTREE_COLUMNS {
        chunks
            .create_index(&[column], Index::BTree(BTreeIndexBuilder::default()))
            .replace(true)
            .execute()
            .await?;
    }
    for column in BITMAP_COLUMNS {
        chunks
            .create_index(&[column], Index::Bitmap(BitmapIndexBuilder::default()))
            .replace(true)
            .execute()
            .await?;
    }
    Ok(())
}

async fn collect(
    table: &Table,
    filter: Option<&str>,
    columns: &[&str],
) -> Result<Vec<RecordBatch>, IndexError> {
    let mut query = table.query().select(Select::columns(columns));
    if let Some(filter) = filter {
        query = query.only_if(filter);
    }
    Ok(query.execute().await?.try_collect().await?)
}

/// Distinct paper IDs with chunk rows, whether or not they have a manifest
/// row.
async fn read_chunk_paper_ids(chunks: &Table) -> Result<BTreeSet<String>, IndexError> {
    let batches = collect(chunks, None, &["paper_id"]).await?;
    let mut ids = BTreeSet::new();
    for batch in &batches {
        let column = string_column(batch, CHUNKS_TABLE, "paper_id")?;
        ids.extend(column.iter().flatten().map(str::to_owned));
    }
    Ok(ids)
}

async fn read_paper_manifest(papers: &Table) -> Result<BTreeMap<String, ManifestRow>, IndexError> {
    let batches = collect(
        papers,
        None,
        &[
            "paper_id",
            "content_hash",
            "metadata_hash",
            "chunker_version",
            "model_id",
        ],
    )
    .await?;
    let mut rows = BTreeMap::new();
    for batch in &batches {
        let ids = string_column(batch, PAPERS_TABLE, "paper_id")?;
        let hashes = string_column(batch, PAPERS_TABLE, "content_hash")?;
        let metadata_hashes = string_column(batch, PAPERS_TABLE, "metadata_hash")?;
        let versions = u32_column(batch, PAPERS_TABLE, "chunker_version")?;
        let models = string_column(batch, PAPERS_TABLE, "model_id")?;
        for i in 0..batch.num_rows() {
            rows.insert(
                ids.value(i).to_owned(),
                ManifestRow {
                    content_hash: hashes.value(i).to_owned(),
                    metadata_hash: metadata_hashes.value(i).to_owned(),
                    chunker_version: versions.value(i),
                    model_id: models.value(i).to_owned(),
                },
            );
        }
    }
    Ok(rows)
}

pub(crate) fn string_column<'b>(
    batch: &'b RecordBatch,
    table: &'static str,
    column: &str,
) -> Result<&'b lancedb::arrow::arrow_array::StringArray, IndexError> {
    batch
        .column_by_name(column)
        .and_then(|c| c.as_string_opt::<i32>())
        .ok_or_else(|| IndexError::Schema {
            table,
            detail: format!("{column} is not a string column"),
        })
}

pub(crate) fn u32_column<'b>(
    batch: &'b RecordBatch,
    table: &'static str,
    column: &str,
) -> Result<&'b lancedb::arrow::arrow_array::UInt32Array, IndexError> {
    batch
        .column_by_name(column)
        .and_then(|c| c.as_primitive_opt::<UInt32Type>())
        .ok_or_else(|| IndexError::Schema {
            table,
            detail: format!("{column} is not a u32 column"),
        })
}

fn vector_dim(schema: &SchemaRef) -> Option<u32> {
    match schema.field_with_name(VECTOR_COLUMN).ok()?.data_type() {
        DataType::FixedSizeList(_, size) => u32::try_from(*size).ok(),
        _ => None,
    }
}

/// Return whether `schema` has the column names and types of `expected`.
fn same_columns(schema: &SchemaRef, expected: &SchemaRef) -> bool {
    let columns = |s: &SchemaRef| {
        s.fields()
            .iter()
            .map(|f| (f.name().clone(), f.data_type().clone()))
            .collect::<Vec<_>>()
    };
    columns(schema) == columns(expected)
}

fn utf8(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::Utf8, nullable)
}

fn chunks_schema(dim: i32) -> SchemaRef {
    Arc::new(Schema::new(vec![
        utf8("chunk_id", false),
        utf8("paper_id", false),
        utf8("title", false),
        utf8("revisionless_id", false),
        Field::new("revision", DataType::UInt32, true),
        Field::new("is_latest", DataType::Boolean, false),
        Field::new("has_text", DataType::Boolean, false),
        Field::new("year", DataType::Int32, true),
        utf8("mailing_date", false),
        utf8("target_group", false),
        utf8("authors", false),
        utf8("intent", false),
        utf8("chunk_kind", true),
        Field::new(
            "section_path",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            false,
        ),
        Field::new("start_line", DataType::UInt32, false),
        Field::new("end_line", DataType::UInt32, false),
        utf8("text", false),
        utf8(FTS_COLUMN, false),
        Field::new(
            VECTOR_COLUMN,
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim),
            true,
        ),
    ]))
}

fn papers_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        utf8("paper_id", false),
        utf8("content_hash", false),
        utf8("metadata_hash", false),
        Field::new("chunker_version", DataType::UInt32, false),
        utf8("model_id", false),
        Field::new(
            "indexed_at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
    ]))
}

fn manifest_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        utf8("model_id", false),
        Field::new("dim", DataType::UInt32, false),
        Field::new("chunker_version", DataType::UInt32, false),
    ]))
}

fn string_array<'s>(values: impl IntoIterator<Item = &'s str>) -> ArrayRef {
    let mut builder = StringBuilder::new();
    for value in values {
        builder.append_value(value);
    }
    Arc::new(builder.finish())
}

fn u32_array(values: impl IntoIterator<Item = u32>) -> ArrayRef {
    let mut builder = UInt32Builder::new();
    builder.extend(values.into_iter().map(Some));
    Arc::new(builder.finish())
}

fn line_number(line: usize) -> Result<u32, IndexError> {
    u32::try_from(line).map_err(|_| IndexError::OutOfRange("line number"))
}

/// Build a paper's chunk rows. With no chunks, builds the single
/// metadata-only row (`has_text = false`, no text, null vector).
fn chunk_batch(
    fields: &PaperFields,
    paper_chunks: &[Chunk],
    vectors: &[Vec<f32>],
    dim: usize,
) -> Result<RecordBatch, IndexError> {
    let has_text = !paper_chunks.is_empty();
    let n = paper_chunks.len().max(1);
    let list_size = i32::try_from(dim).map_err(|_| IndexError::OutOfRange("dim"))?;
    let repeat = |value: &str| string_array(std::iter::repeat_n(value, n));

    let mut chunk_ids = StringBuilder::new();
    let mut kinds = StringBuilder::new();
    let mut sections = ListBuilder::new(StringBuilder::new());
    let mut starts = UInt32Builder::new();
    let mut ends = UInt32Builder::new();
    let mut texts = StringBuilder::new();
    let mut indexed = StringBuilder::new();
    let mut vector_builder = FixedSizeListBuilder::new(Float32Builder::new(), list_size);
    if has_text {
        for (ordinal, (chunk, vector)) in paper_chunks.iter().zip(vectors).enumerate() {
            chunk_ids.append_value(format!("{}:{ordinal}", fields.paper_id));
            kinds.append_value(chunk.chunk_kind.as_str());
            for heading in &chunk.section_path {
                sections.values().append_value(heading);
            }
            sections.append(true);
            starts.append_value(line_number(chunk.start_line)?);
            ends.append_value(line_number(chunk.end_line)?);
            texts.append_value(&chunk.text);
            indexed.append_value(&chunk.indexed_text);
            vector_builder.values().append_slice(vector);
            vector_builder.append(true);
        }
    } else {
        chunk_ids.append_value(format!("{}:{METADATA_CHUNK_SUFFIX}", fields.paper_id));
        kinds.append_null();
        sections.append(true);
        starts.append_value(0);
        ends.append_value(0);
        texts.append_value("");
        indexed.append_value("");
        vector_builder.values().append_nulls(dim);
        vector_builder.append(false);
    }

    let mut revisions = UInt32Builder::new();
    let mut years = Int32Builder::new();
    let mut latest = BooleanBuilder::new();
    let mut text_flags = BooleanBuilder::new();
    for _ in 0..n {
        revisions.append_option(fields.revision);
        years.append_option(fields.year);
        latest.append_value(fields.is_latest);
        text_flags.append_value(has_text);
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(chunk_ids.finish()),
        repeat(&fields.paper_id),
        repeat(&fields.title),
        repeat(&fields.revisionless_id),
        Arc::new(revisions.finish()),
        Arc::new(latest.finish()),
        Arc::new(text_flags.finish()),
        Arc::new(years.finish()),
        repeat(&fields.mailing_date),
        repeat(&fields.target_group),
        repeat(&fields.authors),
        repeat(&fields.intent),
        Arc::new(kinds.finish()),
        Arc::new(sections.finish()),
        Arc::new(starts.finish()),
        Arc::new(ends.finish()),
        Arc::new(texts.finish()),
        Arc::new(indexed.finish()),
        Arc::new(vector_builder.finish()),
    ];
    Ok(RecordBatch::try_new(chunks_schema(list_size), columns)?)
}

fn manifest_batch(paper_id: &str, current: &ManifestRow) -> Result<RecordBatch, IndexError> {
    let micros = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| IndexError::OutOfRange("indexed_at"))?
        .as_micros();
    let micros = i64::try_from(micros).map_err(|_| IndexError::OutOfRange("indexed_at"))?;
    Ok(RecordBatch::try_new(
        papers_schema(),
        vec![
            string_array([paper_id]),
            string_array([current.content_hash.as_str()]),
            string_array([current.metadata_hash.as_str()]),
            u32_array([current.chunker_version]),
            string_array([current.model_id.as_str()]),
            Arc::new(TimestampMicrosecondArray::from(vec![micros]).with_timezone("UTC")),
        ],
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sql_string_doubles_quotes() {
        assert_eq!(sql_string("P2300R10"), "'P2300R10'");
        assert_eq!(sql_string("a'b"), "'a''b'");
    }

    #[test]
    fn sha256_hex_is_lowercase_hex_of_the_bytes() {
        assert_eq!(
            sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn vector_dim_reads_the_fixed_size_list_length() {
        assert_eq!(vector_dim(&chunks_schema(16)), Some(16));
        assert_eq!(vector_dim(&papers_schema()), None);
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|e| panic!("build runtime: {e}"))
            .block_on(future)
    }

    fn fields(paper_id: &str) -> PaperFields {
        PaperFields {
            paper_id: paper_id.to_owned(),
            title: String::new(),
            revisionless_id: paper_id.to_owned(),
            revision: None,
            is_latest: true,
            year: None,
            mailing_date: String::new(),
            target_group: String::new(),
            authors: String::new(),
            intent: String::new(),
        }
    }

    #[test]
    fn commit_paper_writes_chunks_before_the_manifest_row() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
        block_on(async {
            let index = PaperIndex::open(dir.path())
                .await
                .unwrap_or_else(|e| panic!("open: {e}"));
            let manifest = IndexManifest {
                model_id: "test/hash".to_owned(),
                dim: 4,
                chunker_version: CHUNKER_VERSION,
            };
            index
                .prepare(&manifest)
                .await
                .unwrap_or_else(|e| panic!("prepare: {e}"));
            // A manifest table that rejects the row simulates a crash between
            // the chunk write and the manifest write.
            index
                .conn
                .drop_table(PAPERS_TABLE, &[])
                .await
                .unwrap_or_else(|e| panic!("drop papers: {e}"));
            let incompatible = Arc::new(Schema::new(vec![utf8("paper_id", false)]));
            index
                .conn
                .create_empty_table(PAPERS_TABLE, incompatible)
                .execute()
                .await
                .unwrap_or_else(|e| panic!("create papers: {e}"));
            let chunks = index.chunks_table().await.unwrap_or_else(|e| panic!("{e}"));
            let papers = index.papers_table().await.unwrap_or_else(|e| panic!("{e}"));

            let row = ManifestRow {
                content_hash: String::new(),
                metadata_hash: String::new(),
                chunker_version: CHUNKER_VERSION,
                model_id: manifest.model_id.clone(),
            };
            let batch = chunk_batch(&fields("P1R0"), &[], &[], 4)
                .unwrap_or_else(|e| panic!("chunk batch: {e}"));
            let manifest_row =
                manifest_batch("P1R0", &row).unwrap_or_else(|e| panic!("manifest batch: {e}"));
            let result = commit_paper(&chunks, &papers, "P1R0", batch, manifest_row).await;
            assert!(result.is_err(), "manifest write should fail");
            assert_eq!(
                chunks
                    .count_rows(Some("paper_id = 'P1R0'".to_owned()))
                    .await
                    .unwrap_or_else(|e| panic!("count: {e}")),
                1,
                "chunk rows are committed before the manifest row"
            );
        });
    }

    #[test]
    fn parse_year_treats_empty_and_invalid_as_null() {
        let num = PaperNum::parse("P1R0").unwrap_or_else(|e| panic!("parse: {e}"));
        assert_eq!(parse_year(&num, " 2025 "), Some(2025));
        assert_eq!(parse_year(&num, "  "), None);
        assert_eq!(parse_year(&num, "20x5"), None);
    }

    #[test]
    fn metadata_hash_tracks_indexed_fields() {
        let row = PaperRow {
            paper_num: "P1R0".to_owned(),
            title: "Title".to_owned(),
            authors: vec!["A".to_owned(), "B".to_owned()],
            ..PaperRow::default()
        };
        let base = metadata_hash(&row);
        assert_eq!(metadata_hash(&row.clone()), base);
        let changed = [
            PaperRow {
                title: "Other".to_owned(),
                ..row.clone()
            },
            PaperRow {
                year: "2025".to_owned(),
                ..row.clone()
            },
            PaperRow {
                mailing_date: "2025-01".to_owned(),
                ..row.clone()
            },
            PaperRow {
                target_group: "LEWG".to_owned(),
                ..row.clone()
            },
            PaperRow {
                intent: "ask".to_owned(),
                ..row.clone()
            },
            PaperRow {
                authors: vec!["AB".to_owned()],
                ..row.clone()
            },
        ];
        for other in &changed {
            assert_ne!(metadata_hash(other), base, "{other:?}");
        }
        let untracked = PaperRow {
            error: "boom".to_owned(),
            ..row.clone()
        };
        assert_eq!(metadata_hash(&untracked), base);
    }

    #[test]
    fn index_dir_is_beside_paperstore() {
        assert_eq!(
            index_dir(Path::new("/data/wg21")),
            PathBuf::from("/data/wg21/paperindex")
        );
    }
}
