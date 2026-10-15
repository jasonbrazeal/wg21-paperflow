// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! The ingest pipeline: paperstore papers into index records.
//!
//! [`index_papers`] turns each selected paper revision into one
//! [`PaperRecords`] write: its metadata row, plus its Markdown chunked,
//! prefixed and embedded. Papers whose index records already match their
//! current Markdown and the current [`IndexFormatVersion`] are skipped.
//! [`IndexLock`] keeps a second writer out of the index directory.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::{File, OpenOptions, TryLockError};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use paperstore::{PaperNum, PaperRow, StorageBackend};

use crate::chunk::chunk_markdown;
use crate::error::{Error, Result};
use crate::format::{IndexFormatVersion, content_hash};
use crate::model::Embedder;
use crate::prefix::{build_prefix, prefixed_text};
use crate::store::{ChunkRecord, IndexStore, MetaRecord, PaperFields, PaperRecords};

/// Paperstore tool name whose directory holds the index.
pub const TOOL_NAME: &str = "paper-mcp";

/// Name of the writer lock file inside the index directory.
pub const LOCK_FILE: &str = "index.lock";

/// Largest chunk, in embedder tokens, before the prefix is added.
///
/// Changing it changes every chunk, so it must ship with a
/// [`crate::CHUNKER_VERSION`] bump.
pub const DEFAULT_CHUNK_BUDGET: usize = 512;

/// Embedder tokens kept free for the prefix when the embedder's input limit
/// is what bounds the chunk budget.
pub const PREFIX_TOKEN_RESERVE: usize = 128;

/// Exclusive hold on an index directory's writer lock.
///
/// The lock is released when the value is dropped, and by the operating
/// system if the process dies.
#[derive(Debug)]
pub struct IndexLock {
    _file: File,
    path: PathBuf,
}

impl IndexLock {
    /// Take the writer lock of the index in `tool_dir`, creating the directory
    /// and lock file when missing.
    ///
    /// Fails at once rather than waiting when another writer holds the lock.
    ///
    /// # Errors
    ///
    /// Returns [`Error::IndexLocked`] when another handle, in this or any
    /// other process, holds the lock, and [`Error::LockIo`] when the lock file
    /// cannot be created or locked.
    pub fn acquire(tool_dir: &Path) -> Result<Self> {
        let path = tool_dir.join(LOCK_FILE);
        let io_err = |source| Error::LockIo {
            path: path.clone(),
            source,
        };
        std::fs::create_dir_all(tool_dir).map_err(io_err)?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(io_err)?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file, path }),
            Err(TryLockError::WouldBlock) => Err(Error::IndexLocked(path)),
            Err(TryLockError::Error(source)) => Err(io_err(source)),
        }
    }

    /// Return the path of the held lock file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Settings for one [`index_papers`] run.
#[derive(Clone, Debug)]
pub struct IndexOptions {
    /// Reindex papers even when their records are current.
    pub force: bool,
    /// Most papers prepared (chunked and embedded) at once. Store writes are
    /// always serialized.
    pub workers: NonZeroUsize,
    /// Largest chunk, in embedder tokens, before the prefix is added.
    pub chunk_budget: usize,
    /// Format version recorded with, and compared against, every record.
    pub format: IndexFormatVersion,
}

impl IndexOptions {
    /// Return options for `embedder` and the reranker named `reranker_id`:
    /// no forcing, one worker, and [`default_chunk_budget`].
    #[must_use]
    pub fn new(embedder: &dyn Embedder, reranker_id: &str) -> Self {
        Self {
            force: false,
            workers: NonZeroUsize::MIN,
            chunk_budget: default_chunk_budget(embedder),
            format: IndexFormatVersion::current(embedder.id(), reranker_id),
        }
    }
}

/// Return the chunk budget for `embedder`: [`DEFAULT_CHUNK_BUDGET`], or less
/// when the embedder's input limit leaves no room for it plus
/// [`PREFIX_TOKEN_RESERVE`].
///
/// The budget is at least one token.
#[must_use]
pub fn default_chunk_budget(embedder: &dyn Embedder) -> usize {
    DEFAULT_CHUNK_BUDGET
        .min(embedder.max_tokens().saturating_sub(PREFIX_TOKEN_RESERVE))
        .max(1)
}

/// What an index run did with one paper.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Chunks and metadata were written.
    Indexed {
        /// Number of chunks written.
        chunks: usize,
    },
    /// The paper has no Markdown, so only its metadata record was written.
    NoContent,
    /// The paper's records were already current.
    Skipped,
    /// The paper could not be indexed.
    Failed(String),
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Indexed { chunks: 1 } => f.write_str("indexed (1 chunk)"),
            Self::Indexed { chunks } => write!(f, "indexed ({chunks} chunks)"),
            Self::NoContent => f.write_str("no content (metadata only)"),
            Self::Skipped => f.write_str("skipped (up to date)"),
            Self::Failed(reason) => write!(f, "failed: {reason}"),
        }
    }
}

/// The outcome for one paper of an index run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaperReport {
    /// The paper revision.
    pub paper: PaperNum,
    /// What happened to it.
    pub outcome: Outcome,
}

/// Per-paper outcomes of an index run, in selection order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexReport {
    /// One entry per selected paper.
    pub papers: Vec<PaperReport>,
    /// Why the final store optimization failed, if it did. The records
    /// reported as written are still in the store.
    pub optimize_error: Option<String>,
}

impl IndexReport {
    /// Return how many papers failed.
    #[must_use]
    pub fn failed(&self) -> usize {
        self.papers
            .iter()
            .filter(|p| matches!(p.outcome, Outcome::Failed(_)))
            .count()
    }

    /// Return whether any paper failed or the final optimization failed.
    #[must_use]
    pub fn has_failures(&self) -> bool {
        self.failed() > 0 || self.optimize_error.is_some()
    }

    /// Return the operator summary: one line per paper, a totals line, and
    /// a line for a failed optimization.
    #[must_use]
    pub fn summary_lines(&self) -> Vec<String> {
        let count =
            |want: fn(&Outcome) -> bool| self.papers.iter().filter(|p| want(&p.outcome)).count();
        let mut lines: Vec<String> = self
            .papers
            .iter()
            .map(|p| format!("{}: {}", p.paper, p.outcome))
            .collect();
        lines.push(format!(
            "{} papers: {} indexed, {} metadata only, {} skipped, {} failed",
            self.papers.len(),
            count(|o| matches!(o, Outcome::Indexed { .. })),
            count(|o| *o == Outcome::NoContent),
            count(|o| *o == Outcome::Skipped),
            self.failed(),
        ));
        if let Some(reason) = &self.optimize_error {
            lines.push(format!("index optimization failed: {reason}"));
        }
        lines
    }

    /// Return the outcome recorded for `paper`, if it was selected.
    #[must_use]
    pub fn outcome(&self, paper: &PaperNum) -> Option<&Outcome> {
        self.papers
            .iter()
            .find(|p| &p.paper == paper)
            .map(|p| &p.outcome)
    }
}

/// Index `nums` from `backend` into `store`.
///
/// Each paper revision is processed independently: its metadata row and
/// Markdown are read, it is skipped when its stored content hash and format
/// version match the current ones (unless `options.force`), and otherwise
/// its records are replaced. A paper without Markdown gets only its metadata
/// record. A failure on one paper is recorded in the report and the run goes
/// on. After the writes, every indexed revision of each written paper number
/// has its latest flag recomputed from the revisions `backend` lists, and
/// the store is optimized.
///
/// When the latest flags of a paper number cannot be updated, its papers
/// written in this run are reported as failed and their records are
/// removed, so the next run reindexes them instead of skipping them. A
/// failed optimization is recorded in [`IndexReport::optimize_error`];
/// every failure leaves [`IndexReport::has_failures`] true.
pub async fn index_papers<S: IndexStore>(
    backend: &dyn StorageBackend,
    store: &S,
    embedder: &dyn Embedder,
    nums: &[PaperNum],
    options: &IndexOptions,
) -> IndexReport {
    let format = options.format.to_string();
    let mut report = IndexReport::default();
    let mut latest_cache: BTreeMap<String, Option<PaperNum>> = BTreeMap::new();
    let mut written: BTreeMap<String, Vec<PaperNum>> = BTreeMap::new();

    for window in nums.chunks(options.workers.get()) {
        let mut slots: Vec<(PaperNum, std::result::Result<Loaded, Outcome>)> = Vec::new();
        for paper in window {
            let loaded = load(
                backend,
                store,
                paper,
                &format,
                options.force,
                &mut latest_cache,
            )
            .await
            .map_err(|e| Outcome::Failed(e.to_string()))
            .and_then(|l| l.ok_or(Outcome::Skipped));
            slots.push((paper.clone(), loaded));
        }

        let prepared = prepare_all(slots, embedder, options.chunk_budget, &format);

        for (paper, records) in prepared {
            let outcome = match records {
                Err(outcome) => outcome,
                Ok(records) => {
                    let outcome = if records.meta.has_markdown {
                        Outcome::Indexed {
                            chunks: records.chunks.len(),
                        }
                    } else {
                        Outcome::NoContent
                    };
                    match store.upsert_paper(records).await {
                        Ok(()) => {
                            written
                                .entry(paper.split_revision().0.to_owned())
                                .or_default()
                                .push(paper.clone());
                            outcome
                        }
                        Err(e) => Outcome::Failed(e.to_string()),
                    }
                }
            };
            tracing::debug!(paper = %paper, %outcome, "paper processed");
            report.papers.push(PaperReport { paper, outcome });
        }
    }

    for (number, papers) in &written {
        let latest = match latest_cache.get(number) {
            Some(Some(latest)) => latest.clone(),
            _ => continue,
        };
        if let Err(e) = store.set_latest(number, &latest).await {
            let reason = format!("cannot update latest-revision flags: {e}");
            for paper in papers {
                let outcome = match store.delete_paper(paper).await {
                    Ok(()) => Outcome::Failed(reason.clone()),
                    Err(delete) => Outcome::Failed(format!(
                        "{reason}; cannot remove its records for reindexing: {delete}"
                    )),
                };
                if let Some(entry) = report.papers.iter_mut().find(|p| &p.paper == paper) {
                    entry.outcome = outcome;
                }
            }
        }
    }

    if let Err(e) = store.optimize().await {
        report.optimize_error = Some(e.to_string());
    }
    report
}

/// A paper read from paperstore and found stale, ready to prepare.
struct Loaded {
    row: PaperRow,
    markdown: Option<String>,
    hash: String,
    is_latest: bool,
}

/// Read `paper` and decide whether it needs indexing.
///
/// Returns `None` when its records are current and `force` is off.
async fn load<S: IndexStore>(
    backend: &dyn StorageBackend,
    store: &S,
    paper: &PaperNum,
    format: &str,
    force: bool,
    latest_cache: &mut BTreeMap<String, Option<PaperNum>>,
) -> Result<Option<Loaded>> {
    let row = backend.meta(paper)?;
    let markdown = match backend.paper_md(paper) {
        Ok(md) => Some(md),
        Err(paperstore::Error::MissingMarkdown { .. }) => None,
        Err(e) => return Err(e.into()),
    };
    let hash = markdown.as_deref().map(content_hash).unwrap_or_default();
    if !force
        && let Some((stored_hash, stored_format)) = store.freshness(paper).await?
        && stored_hash == hash
        && stored_format == format
    {
        return Ok(None);
    }
    let number = paper.split_revision().0.to_owned();
    let latest = match latest_cache.get(&number) {
        Some(latest) => latest.clone(),
        None => {
            let latest = latest_listed(backend, paper)?;
            latest_cache.insert(number, latest.clone());
            latest
        }
    };
    Ok(Some(Loaded {
        is_latest: latest.as_ref() == Some(paper),
        row,
        markdown,
        hash,
    }))
}

/// Return the newest revision paperstore lists for the paper number of `paper`.
fn latest_listed(backend: &dyn StorageBackend, paper: &PaperNum) -> Result<Option<PaperNum>> {
    backend
        .list_revisions(paper)?
        .first()
        .map(|row| PaperNum::parse(&row.paper_num))
        .transpose()
        .map_err(Error::from)
}

type Slot = (PaperNum, std::result::Result<Loaded, Outcome>);
type Prepared = (PaperNum, std::result::Result<PaperRecords, Outcome>);

/// Prepare every loaded slot, one thread per slot when there are several.
///
/// Output order matches input order.
fn prepare_all(
    slots: Vec<Slot>,
    embedder: &dyn Embedder,
    budget: usize,
    format: &str,
) -> Vec<Prepared> {
    let run = |paper: PaperNum, loaded: std::result::Result<Loaded, Outcome>| {
        let records = loaded.and_then(|l| {
            prepare(&paper, l, embedder, budget, format).map_err(|e| Outcome::Failed(e.to_string()))
        });
        (paper, records)
    };
    if slots.len() <= 1 {
        return slots.into_iter().map(|(p, l)| run(p, l)).collect();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = slots
            .into_iter()
            .map(|(paper, loaded)| {
                let fallback = paper.clone();
                (fallback, scope.spawn(move || run(paper, loaded)))
            })
            .collect();
        handles
            .into_iter()
            .map(|(paper, handle)| {
                handle.join().unwrap_or_else(|_| {
                    (
                        paper,
                        Err(Outcome::Failed("paper preparation panicked".to_owned())),
                    )
                })
            })
            .collect()
    })
}

/// Chunk, prefix and embed one loaded paper into its records.
fn prepare(
    paper: &PaperNum,
    loaded: Loaded,
    embedder: &dyn Embedder,
    budget: usize,
    format: &str,
) -> Result<PaperRecords> {
    let Loaded {
        row,
        markdown,
        hash,
        is_latest,
    } = loaded;
    let (number, revision) = paper.split_revision();
    let fields = PaperFields::from_row(&row);
    let mut chunks = Vec::new();
    if let Some(md) = &markdown {
        let pieces = chunk_markdown(md, budget, &|text| embedder.count_tokens(text));
        let prefixed: Vec<String> = pieces
            .iter()
            .map(|c| prefixed_text(&build_prefix(&row, &c.heading_path), &c.text))
            .collect();
        let vectors = embedder.embed_documents(&prefixed)?;
        if vectors.len() != pieces.len() {
            return Err(Error::Inference(format!(
                "embedder returned {} vectors for {} chunks",
                vectors.len(),
                pieces.len()
            )));
        }
        chunks = pieces
            .into_iter()
            .zip(prefixed)
            .zip(vectors)
            .map(|((chunk, prefixed_text), vector)| ChunkRecord {
                paper_id: paper.clone(),
                revision,
                paper_number: number.to_owned(),
                is_latest,
                heading_path: chunk.heading_path,
                start: chunk.start,
                end: chunk.end,
                text: chunk.text,
                prefixed_text,
                vector,
                content_hash: hash.clone(),
                format_version: format.to_owned(),
                fields: fields.clone(),
            })
            .collect();
    }
    Ok(PaperRecords {
        meta: MetaRecord {
            row,
            is_latest,
            has_markdown: markdown.is_some(),
            content_hash: hash,
            format_version: format.to_owned(),
        },
        chunks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::fake::HashEmbedder;

    #[test]
    fn chunk_budget_is_capped_by_default_and_embedder_limit() {
        assert_eq!(
            default_chunk_budget(&HashEmbedder::new(8, 100_000)),
            DEFAULT_CHUNK_BUDGET
        );
        assert_eq!(
            default_chunk_budget(&HashEmbedder::new(8, PREFIX_TOKEN_RESERVE + 200)),
            200
        );
        assert_eq!(default_chunk_budget(&HashEmbedder::new(8, 10)), 1);
    }

    #[test]
    fn outcome_display_names_the_result() {
        assert_eq!(
            Outcome::Indexed { chunks: 1 }.to_string(),
            "indexed (1 chunk)"
        );
        assert_eq!(
            Outcome::Indexed { chunks: 3 }.to_string(),
            "indexed (3 chunks)"
        );
        assert_eq!(
            Outcome::Failed("boom".to_owned()).to_string(),
            "failed: boom"
        );
    }

    fn report(outcomes: &[(&str, Outcome)], optimize_error: Option<&str>) -> IndexReport {
        IndexReport {
            papers: outcomes
                .iter()
                .map(|(raw, outcome)| PaperReport {
                    paper: PaperNum::parse(raw).unwrap_or_else(|e| panic!("parse {raw}: {e}")),
                    outcome: outcome.clone(),
                })
                .collect(),
            optimize_error: optimize_error.map(str::to_owned),
        }
    }

    #[test]
    fn summary_lists_each_paper_then_totals() {
        let report = report(
            &[
                ("P2300R10", Outcome::Indexed { chunks: 3 }),
                ("N4950", Outcome::NoContent),
                ("P2300R4", Outcome::Skipped),
                ("P2322R4", Outcome::Failed("boom".to_owned())),
            ],
            None,
        );
        assert_eq!(
            report.summary_lines(),
            vec![
                "P2300R10: indexed (3 chunks)",
                "N4950: no content (metadata only)",
                "P2300R4: skipped (up to date)",
                "P2322R4: failed: boom",
                "4 papers: 1 indexed, 1 metadata only, 1 skipped, 1 failed",
            ]
        );
        assert!(report.has_failures());
    }

    #[test]
    fn optimize_error_is_summarized_and_counts_as_failure() {
        let report = report(
            &[("P2300R10", Outcome::Indexed { chunks: 1 })],
            Some("disk full"),
        );
        assert_eq!(report.failed(), 0);
        assert!(report.has_failures());
        assert_eq!(
            report.summary_lines().last().map(String::as_str),
            Some("index optimization failed: disk full")
        );
    }

    #[test]
    fn clean_report_has_no_failures() {
        let report = report(&[("P2300R10", Outcome::Skipped)], None);
        assert!(!report.has_failures());
    }
}
