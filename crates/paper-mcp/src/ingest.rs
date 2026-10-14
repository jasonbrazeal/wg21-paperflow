//! The ingest command (§8): copies the paperstore into Qdrant.

use std::fmt::Write;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags, params_from_iter};
use sha2::{Digest, Sha256};
use tracing::{error, info, warn};

use crate::bm25;
use crate::config::Config;
use crate::dates::{Date, parse_date, parse_month};
use crate::embed::{Embedder, MODEL_NAME};
use crate::paper_id::PaperId;
use crate::qdrant::{ChunkPayload, Citation, MetadataPayload, Store, StoreError, chunk_point_id};
use crate::text::{
    ChunkSettings, Document, PaperLabel, SPECIAL_TOKENS, TokenCounter, chunk_document,
};

/// Bump when the splitter, block, or chunk rules change, so every paper is
/// chunked again.
const CHUNKER_CODE_VERSION: &str = "2";

/// Bump when the context header format changes.
const HEADER_FORMAT_VERSION: &str = "1";

/// What to ingest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Targets {
    /// Every paper.
    All,
    /// Papers of these four-digit years.
    Years(Vec<String>),
    /// These papers.
    Papers(Vec<PaperId>),
}

/// Invalid ingest targets.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TargetError {
    /// No target was given.
    #[error("give `all`, one or more years, or one or more paper ids")]
    Empty,
    /// `all` was combined with something else.
    #[error("`all` cannot be combined with other targets")]
    AllWithOthers,
    /// Years and paper ids were mixed.
    #[error("years and paper ids cannot be mixed")]
    Mixed,
    /// A target is neither a year nor a paper id.
    #[error("not a year or paper id: {0:?}")]
    Invalid(String),
}

fn is_year(s: &str) -> bool {
    s.len() == 4 && s.bytes().all(|b| b.is_ascii_digit())
}

/// Parses command-line targets.
///
/// # Errors
///
/// Returns [`TargetError`] for an empty list, `all` with other targets,
/// mixed years and ids, or an unrecognised target.
pub fn parse_targets(args: &[String]) -> Result<Targets, TargetError> {
    if args.is_empty() {
        return Err(TargetError::Empty);
    }
    if args.iter().any(|a| a == "all") {
        return if args.len() == 1 {
            Ok(Targets::All)
        } else {
            Err(TargetError::AllWithOthers)
        };
    }
    let years = args.iter().filter(|a| is_year(a)).count();
    if years == args.len() {
        return Ok(Targets::Years(args.to_vec()));
    }
    if years > 0 {
        return Err(TargetError::Mixed);
    }
    args.iter()
        .map(|a| a.parse().map_err(|_| TargetError::Invalid(a.clone())))
        .collect::<Result<_, _>>()
        .map(Targets::Papers)
}

/// The chunker version: a hash of everything that shapes chunks.
#[must_use]
pub fn chunker_version(settings: ChunkSettings) -> String {
    let input = format!(
        "code={CHUNKER_CODE_VERSION};target={};overlap={};max={};header={HEADER_FORMAT_VERSION};model={MODEL_NAME}",
        settings.target_tokens, settings.overlap_max_tokens, settings.max_tokens
    );
    let digest = Sha256::digest(input.as_bytes());
    let mut hex = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// An ingest failure that stops the run.
#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    /// The paperstore could not be read.
    #[error("cannot read paperstore {path}: {source}")]
    Paperstore {
        /// The database file.
        path: PathBuf,
        /// The cause.
        source: rusqlite::Error,
    },
    /// Qdrant failed.
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// One `papers` row.
#[derive(Debug, Clone)]
struct PaperRow {
    paper_id: String,
    year: String,
    title: String,
    authors: String,
    target_group: String,
    intent: String,
    url: String,
    document_date: String,
    mailing_date: String,
    disposition: String,
    previous_version: String,
    markdown_path: String,
    citations_extracted_at: String,
}

const PAPER_COLUMNS: &str = "paper_id, COALESCE(year, ''), COALESCE(title, ''), COALESCE(authors, ''), \
     COALESCE(target_group, ''), COALESCE(intent, ''), COALESCE(url, ''), COALESCE(document_date, ''), \
     COALESCE(mailing_date, ''), COALESCE(disposition, ''), COALESCE(previous_version, ''), \
     COALESCE(markdown_path, ''), COALESCE(citations_extracted_at, '')";

/// Opens `paperstore.db` read-only. It never runs migrations or writes.
fn open_paperstore(path: &Path) -> rusqlite::Result<Connection> {
    Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
}

fn select_rows(conn: &Connection, targets: &Targets) -> rusqlite::Result<Vec<PaperRow>> {
    let (clause, args): (String, Vec<String>) = match targets {
        Targets::All => (String::new(), Vec::new()),
        Targets::Years(years) => (
            format!("WHERE year IN ({})", vec!["?"; years.len()].join(", ")),
            years.clone(),
        ),
        Targets::Papers(ids) => (
            format!(
                "WHERE lower(paper_id) IN ({})",
                vec!["?"; ids.len()].join(", ")
            ),
            ids.iter().map(|id| id.as_str().to_owned()).collect(),
        ),
    };
    let sql = format!("SELECT {PAPER_COLUMNS} FROM papers {clause} ORDER BY paper_id");
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(args.iter()), |row| {
        Ok(PaperRow {
            paper_id: row.get(0)?,
            year: row.get(1)?,
            title: row.get(2)?,
            authors: row.get(3)?,
            target_group: row.get(4)?,
            intent: row.get(5)?,
            url: row.get(6)?,
            document_date: row.get(7)?,
            mailing_date: row.get(8)?,
            disposition: row.get(9)?,
            previous_version: row.get(10)?,
            markdown_path: row.get(11)?,
            citations_extracted_at: row.get(12)?,
        })
    })?;
    rows.collect()
}

fn select_citations(conn: &Connection, paper_id: &str) -> rusqlite::Result<Vec<Citation>> {
    let mut stmt = conn.prepare_cached(
        "SELECT cited_paper_id, COALESCE(count, 1) FROM paper_citations WHERE paper_id = ?1 ORDER BY cited_paper_id",
    )?;
    let rows = stmt.query_map([paper_id], |row| {
        Ok(Citation {
            paper_id: row.get::<_, String>(0)?.to_lowercase(),
            count: row.get::<_, i64>(1)?.try_into().unwrap_or(0),
        })
    })?;
    rows.collect()
}

fn parse_authors(paper_id: &str, raw: &str) -> Vec<String> {
    if raw.trim().is_empty() {
        return Vec::new();
    }
    if let Ok(authors) = serde_json::from_str::<Vec<String>>(raw) {
        authors
    } else {
        warn!(
            paper = paper_id,
            value = raw,
            "authors is not a JSON array; using it as one name"
        );
        vec![raw.trim().to_owned()]
    }
}

/// Builds a metadata record without markdown.
fn metadata_record(id: &PaperId, row: &PaperRow, cites: Vec<Citation>) -> MetadataPayload {
    let document_date = parse_date(row.document_date.trim()).map(Date::iso);
    if document_date.is_none() {
        warn!(
            paper = id.as_str(),
            value = row.document_date.as_str(),
            "document_date is not a valid YYYY-MM-DD date"
        );
    }
    let mailing_date_filter = parse_month(row.mailing_date.trim()).map(Date::iso);
    let authors = parse_authors(id.as_str(), &row.authors);
    MetadataPayload {
        paper_id: id.as_str().to_owned(),
        title: row.title.clone(),
        authors_folded: authors.iter().map(|a| bm25::fold(a)).collect(),
        authors,
        document_date,
        mailing_date: row.mailing_date.trim().to_owned(),
        mailing_date_filter,
        year: row.year.clone(),
        group: row.target_group.clone(),
        intent: row.intent.clone(),
        url: row.url.clone(),
        disposition: row.disposition.clone(),
        previous_version: row.previous_version.trim().to_lowercase(),
        has_markdown: false,
        total_lines: None,
        total_sections: None,
        series: id.series().as_str().to_owned(),
        number: id.number(),
        revision: id.revision(),
        cited_ids: cites.iter().map(|c| c.paper_id.clone()).collect(),
        cites,
        citations_extracted: !row.citations_extracted_at.trim().is_empty(),
        markdown: None,
        chunker_version: None,
    }
}

/// Counts reported at the end of a run.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Report {
    /// Papers whose chunks were written.
    pub indexed: usize,
    /// Papers whose chunks were current; their metadata was refreshed.
    pub skipped: usize,
    /// Papers that failed.
    pub failed: usize,
    /// Papers with no markdown; their metadata was written.
    pub no_markdown: Vec<String>,
}

enum Outcome {
    Indexed,
    Skipped,
    NoMarkdown,
}

/// A failure confined to one paper.
#[derive(Debug, thiserror::Error)]
enum PaperError {
    #[error("cannot read markdown {path}: {source}")]
    Markdown {
        path: String,
        source: std::io::Error,
    },
    #[error("cannot read citations: {0}")]
    Citations(rusqlite::Error),
    #[error("cannot build chunk point {0}")]
    Point(usize),
    #[error("chunk at lines {first}-{last} has {tokens} tokens, over the model limit of {limit}")]
    ChunkTooLong {
        first: usize,
        last: usize,
        tokens: usize,
        limit: usize,
    },
    #[error(transparent)]
    Embed(#[from] crate::embed::EmbedError),
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// Runs the ingest.
pub struct Ingest<'a> {
    config: &'a Config,
    store: &'a Store,
    embedder: &'a Embedder,
    force: bool,
    version: String,
}

impl std::fmt::Debug for Ingest<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ingest")
            .field("force", &self.force)
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl<'a> Ingest<'a> {
    /// Prepares an ingest run.
    #[must_use]
    pub fn new(config: &'a Config, store: &'a Store, embedder: &'a Embedder, force: bool) -> Self {
        Self {
            config,
            store,
            embedder,
            force,
            version: chunker_version(config.chunk_settings()),
        }
    }

    /// Ingests `targets` from the paperstore in `data_dir`.
    ///
    /// # Errors
    ///
    /// Returns [`IngestError`] when the paperstore cannot be read or Qdrant
    /// cannot be set up. Failures confined to one paper are logged and
    /// counted instead.
    pub async fn run(&self, data_dir: &Path, targets: &Targets) -> Result<Report, IngestError> {
        let db_path = data_dir.join("paperstore.db");
        let paperstore_error = |source| IngestError::Paperstore {
            path: db_path.clone(),
            source,
        };
        let conn = open_paperstore(&db_path).map_err(paperstore_error)?;
        let rows = select_rows(&conn, targets).map_err(paperstore_error)?;
        self.store.ensure_collections().await?;

        let mut report = Report::default();
        if let Targets::Papers(ids) = targets {
            for id in ids {
                if !rows
                    .iter()
                    .any(|r| r.paper_id.eq_ignore_ascii_case(id.as_str()))
                {
                    error!(paper = id.as_str(), "not in the paperstore");
                    report.failed += 1;
                }
            }
        }
        info!(
            papers = rows.len(),
            version = self.version.as_str(),
            "ingest starting"
        );
        for row in &rows {
            let Ok(id) = row.paper_id.parse::<PaperId>() else {
                error!(paper = row.paper_id.as_str(), "not a valid paper id");
                report.failed += 1;
                continue;
            };
            match self.paper(&conn, &id, row).await {
                Ok(Outcome::Indexed) => report.indexed += 1,
                Ok(Outcome::Skipped) => report.skipped += 1,
                Ok(Outcome::NoMarkdown) => report.no_markdown.push(id.as_str().to_owned()),
                Err(PaperError::Store(e)) => {
                    self.store.health().await?;
                    error!(paper = id.as_str(), error = %e, "paper failed");
                    report.failed += 1;
                }
                Err(e) => {
                    error!(paper = id.as_str(), error = %e, "paper failed");
                    report.failed += 1;
                }
            }
        }
        Ok(report)
    }

    async fn paper(
        &self,
        conn: &Connection,
        id: &PaperId,
        row: &PaperRow,
    ) -> Result<Outcome, PaperError> {
        let cites = select_citations(conn, &row.paper_id).map_err(PaperError::Citations)?;
        let mut record = metadata_record(id, row, cites);
        let path = row.markdown_path.trim();
        if path.is_empty() {
            self.store.upsert_metadata(&record).await?;
            info!(paper = id.as_str(), "metadata written; no markdown");
            return Ok(Outcome::NoMarkdown);
        }
        let markdown = std::fs::read_to_string(path).map_err(|source| PaperError::Markdown {
            path: path.to_owned(),
            source,
        })?;
        let doc = Document::parse(&markdown);
        let sections = doc.sections();
        record.has_markdown = true;
        record.total_lines = u64::try_from(doc.line_count()).ok();
        record.total_sections = u64::try_from(sections.len()).ok();

        let stored = self.store.chunker_version(id.as_str()).await?;
        let current = stored.as_ref().and_then(Option::as_deref) == Some(self.version.as_str());
        let outcome = if self.force || !current {
            let chunks = self.write_chunks(id, &record, &doc, &sections).await?;
            info!(
                paper = id.as_str(),
                chunks,
                lines = doc.line_count(),
                sections = sections.len(),
                "indexed"
            );
            Outcome::Indexed
        } else {
            info!(paper = id.as_str(), "chunks current; metadata refreshed");
            Outcome::Skipped
        };
        record.markdown = Some(markdown.clone());
        record.chunker_version = Some(self.version.clone());
        self.store.upsert_metadata(&record).await?;
        Ok(outcome)
    }

    async fn write_chunks(
        &self,
        id: &PaperId,
        record: &MetadataPayload,
        doc: &Document<'_>,
        sections: &[crate::text::Section],
    ) -> Result<usize, PaperError> {
        let counter = self.embedder.token_counter();
        let label = PaperLabel {
            paper_id: id.as_str(),
            title: &record.title,
        };
        let chunks = chunk_document(doc, sections, self.config.chunk_settings(), &counter, label);
        let limit = self.config.model.max_length;
        let mut points = Vec::with_capacity(chunks.len());
        for batch in chunks.chunks(self.config.ingest.embed_batch) {
            let mut texts = Vec::with_capacity(batch.len());
            for chunk in batch {
                let text = chunk.embed_text(doc);
                // The chunker keeps chunks far below this; a longer one
                // would be truncated by the model, so it is a chunker bug.
                let tokens = counter.count(&text) + SPECIAL_TOKENS;
                if tokens > limit {
                    return Err(PaperError::ChunkTooLong {
                        first: chunk.first.get(),
                        last: chunk.last.get(),
                        tokens,
                        limit,
                    });
                }
                texts.push(text);
            }
            let vectors = self.embedder.embed(texts).await?;
            for (chunk, dense) in batch.iter().zip(vectors) {
                let payload = ChunkPayload {
                    paper_id: id.as_str().to_owned(),
                    title: record.title.clone(),
                    url: record.url.clone(),
                    document_date: record.document_date.clone(),
                    mailing_date: record.mailing_date.clone(),
                    mailing_date_filter: record.mailing_date_filter.clone(),
                    section_index: chunk.section.get() as u64,
                    heading_path: chunk.heading_path_text(),
                    first_line: chunk.first.get() as u64,
                    last_line: chunk.last.get() as u64,
                    text: chunk.text(doc),
                };
                let point_id = chunk_point_id(id.as_str(), chunk.section.get(), chunk.number.get());
                let point = self
                    .store
                    .chunk_point(point_id, dense, &payload)
                    .ok_or(PaperError::Point(points.len()))?;
                points.push(point);
            }
        }
        let count = points.len();
        self.store.replace_chunks(id.as_str(), points).await?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn parses_targets() {
        assert_eq!(parse_targets(&args(&["all"])), Ok(Targets::All));
        assert_eq!(
            parse_targets(&args(&["2025", "2026"])),
            Ok(Targets::Years(args(&["2025", "2026"])))
        );
        assert!(
            matches!(parse_targets(&args(&["P3045R9", "n5008"])), Ok(Targets::Papers(ids)) if ids.len() == 2)
        );
        assert_eq!(parse_targets(&[]), Err(TargetError::Empty));
        assert_eq!(
            parse_targets(&args(&["all", "2026"])),
            Err(TargetError::AllWithOthers)
        );
        assert_eq!(
            parse_targets(&args(&["2026", "p1r0"])),
            Err(TargetError::Mixed)
        );
        assert_eq!(
            parse_targets(&args(&["p12"])),
            Err(TargetError::Invalid("p12".to_owned()))
        );
    }

    #[test]
    fn chunker_version_tracks_settings() {
        let base = ChunkSettings {
            target_tokens: 400,
            overlap_max_tokens: 200,
            max_tokens: 1024,
        };
        let other = ChunkSettings {
            target_tokens: 300,
            ..base
        };
        assert_eq!(chunker_version(base), chunker_version(base));
        assert_ne!(chunker_version(base), chunker_version(other));
    }
}
