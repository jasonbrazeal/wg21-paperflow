//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! MCP tool handlers.
//!
//! [`PaperTools`] registers exactly five tools and no resources or prompts:
//! `search_papers`, `get_paper_metadata`, `list_revisions`, `list_papers`,
//! and `get_paper_lines`. Every tool returns structured JSON with a declared
//! output schema. Invalid input (an unknown or revisionless paper ID, an
//! unknown search type, a malformed filter) is reported as an MCP tool
//! error naming the input, never as a protocol error.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use paperstore::{PaperNum, PaperRow, StorageBackend};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ContentBlock, IntoContents, Tool};
use rmcp::{Json, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::embed::Embedder;
use crate::filter::{FilterInput, SearchFilters};
use crate::index::PaperIndex;
use crate::lines::slice_lines;
use crate::revision::{parse_revision_query, revision_number, revisionless_id};
use crate::search::{
    DEFAULT_SEARCH_LIMIT, SearchHit, SearchRequest, SearchType, WG21_LINK_BASE, search,
};

/// Papers returned by `list_papers` when a request does not set a limit.
pub const DEFAULT_LIST_LIMIT: usize = 50;

/// Largest page a `list_papers` request may set.
pub const MAX_LIST_LIMIT: usize = 200;

/// A tool failure reported to the client as an MCP tool error.
///
/// The message names the offending input field and value where there is
/// one.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ToolError(String);

impl ToolError {
    fn new(message: impl fmt::Display) -> Self {
        Self(message.to_string())
    }
}

impl IntoContents for ToolError {
    fn into_contents(self) -> Vec<ContentBlock> {
        vec![ContentBlock::text(self.0)]
    }
}

/// Metadata filters shared by `search_papers` and `list_papers`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FilterParams {
    /// Exact paper IDs with revision (for example `P2300R10`, `N4950`).
    /// When given, the revision scope is ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paper_ids: Option<Vec<String>>,
    /// `latest` (default) for only the latest revision of each paper, or
    /// `all` for every revision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision_scope: Option<String>,
    /// Earliest paper year, inclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub year_from: Option<i32>,
    /// Latest paper year, inclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub year_to: Option<i32>,
    /// Mailing as `YYYY-MM`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mailing: Option<String>,
    /// Case-insensitive substring of the target subgroup (for example
    /// `LEWG`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_group: Option<String>,
    /// Case-insensitive substring of an author name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// Paper intent, for example `info` or `ask`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    /// Chunk kind: `body`, `wording`, `code`, `revision_history`, or
    /// `front_matter`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_kind: Option<String>,
}

impl From<&FilterParams> for FilterInput {
    fn from(params: &FilterParams) -> Self {
        Self {
            paper_ids: params.paper_ids.clone(),
            revision_scope: params.revision_scope.clone(),
            year_from: params.year_from,
            year_to: params.year_to,
            mailing: params.mailing.clone(),
            target_group: params.target_group.clone(),
            author: params.author.clone(),
            intent: params.intent.clone(),
            chunk_kind: params.chunk_kind.clone(),
        }
    }
}

/// Input of `search_papers`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SearchPapersParams {
    /// Query text.
    pub query: String,
    /// `hybrid` (keyword and vector fused by reciprocal rank), `vector`
    /// (dense embeddings), or `keyword` (BM25).
    pub search_type: String,
    /// Filters applied before ranking.
    #[serde(flatten)]
    pub filters: FilterParams,
    /// Maximum hits, from 1 to 50. Defaults to 10.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

/// Input of the tools taking one paper ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PaperIdParams {
    /// Paper ID. Must be an exact revision (`P2300R10`, `N4950`, `SD-5`),
    /// except for `list_revisions`, which also accepts `P2300`.
    pub paper_id: String,
}

/// Input of `list_papers`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ListPapersParams {
    /// Filters selecting the papers.
    #[serde(flatten)]
    pub filters: FilterParams,
    /// Number of matching papers to skip. Defaults to 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
    /// Page size, from 1 to 200. Defaults to 50.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

/// Input of `get_paper_lines`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GetPaperLinesParams {
    /// Exact paper ID with revision.
    pub paper_id: String,
    /// First line, 0-based and inclusive; negative counts from the end.
    /// Defaults to the first line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<i64>,
    /// Line after the last, exclusive; negative counts from the end.
    /// Defaults to through the last line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<i64>,
}

/// One ranked chunk in a `search_papers` result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SearchHitOutput {
    /// Exact paper ID.
    pub paper_id: String,
    /// Revision number, or null for N and SD papers.
    pub revision: Option<u32>,
    /// Paper title.
    pub title: String,
    /// Author names.
    pub authors: Vec<String>,
    /// Enclosing heading titles, outermost first.
    pub section_path: Vec<String>,
    /// Chunk kind.
    pub chunk_kind: String,
    /// First line of the chunk, 0-based; pass to `get_paper_lines` as
    /// `start`.
    pub start_line: u32,
    /// Line after the chunk (exclusive); pass to `get_paper_lines` as `end`.
    pub end_line: u32,
    /// Chunk text.
    pub text: String,
    /// Relevance, higher is better; comparable only within one search type.
    pub score: f32,
    /// Canonical paper URL.
    pub url: String,
}

impl From<SearchHit> for SearchHitOutput {
    fn from(hit: SearchHit) -> Self {
        Self {
            paper_id: hit.paper_id,
            revision: hit.revision,
            title: hit.title,
            authors: hit.authors,
            section_path: hit.section_path,
            chunk_kind: hit.chunk_kind.as_str().to_owned(),
            start_line: hit.start_line,
            end_line: hit.end_line,
            text: hit.text,
            score: hit.score,
            url: hit.url,
        }
    }
}

/// Result of `search_papers`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SearchPapersOutput {
    /// Hits by descending score.
    pub hits: Vec<SearchHitOutput>,
}

/// Metadata of one paper revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PaperMetadata {
    /// Exact paper ID.
    pub paper_id: String,
    /// Revision number, or null for N and SD papers.
    pub revision: Option<u32>,
    /// Paper title.
    pub title: String,
    /// Author names.
    pub authors: Vec<String>,
    /// Mailing year, or null when unknown.
    pub year: Option<i32>,
    /// Document date, as recorded.
    pub document_date: String,
    /// Mailing as `YYYY-MM`.
    pub mailing_date: String,
    /// Target subgroup.
    pub target_group: String,
    /// Paper intent.
    pub intent: String,
    /// Disposition, as recorded.
    pub disposition: String,
    /// Previous revision's paper ID, or empty.
    pub previous_version: String,
    /// Canonical paper URL.
    pub url: String,
    /// Whether the paper's text is available to `get_paper_lines` and
    /// search.
    pub has_text: bool,
}

impl From<&PaperRow> for PaperMetadata {
    fn from(row: &PaperRow) -> Self {
        Self {
            paper_id: row.paper_num.clone(),
            revision: revision_of(row),
            title: row.title.clone(),
            authors: row.authors.clone(),
            year: row.year.trim().parse().ok(),
            document_date: row.document_date.clone(),
            mailing_date: row.mailing_date.clone(),
            target_group: row.target_group.clone(),
            intent: row.intent.clone(),
            disposition: row.disposition.clone(),
            previous_version: row.previous_version.clone(),
            url: format!("{WG21_LINK_BASE}{}", row.paper_num),
            has_text: has_text(row),
        }
    }
}

/// One revision in a paper's lineage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RevisionSummary {
    /// Exact paper ID.
    pub paper_id: String,
    /// Revision number, or null for N and SD papers.
    pub revision: Option<u32>,
    /// Document date, as recorded.
    pub document_date: String,
    /// Mailing as `YYYY-MM`.
    pub mailing_date: String,
    /// Disposition, as recorded.
    pub disposition: String,
    /// Previous revision's paper ID, or empty.
    pub previous_version: String,
}

impl From<&PaperRow> for RevisionSummary {
    fn from(row: &PaperRow) -> Self {
        Self {
            paper_id: row.paper_num.clone(),
            revision: revision_of(row),
            document_date: row.document_date.clone(),
            mailing_date: row.mailing_date.clone(),
            disposition: row.disposition.clone(),
            previous_version: row.previous_version.clone(),
        }
    }
}

/// Result of `get_paper_metadata`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PaperMetadataOutput {
    /// The requested revision.
    pub paper: PaperMetadata,
    /// Every revision of the paper, by ascending revision number.
    pub revisions: Vec<RevisionSummary>,
}

/// Result of `list_revisions`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RevisionsOutput {
    /// Paper ID without its revision suffix.
    pub paper: String,
    /// Every revision of the paper, by ascending revision number.
    pub revisions: Vec<RevisionSummary>,
}

/// Result of `list_papers`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ListPapersOutput {
    /// This page of matching papers, by paper ID.
    pub papers: Vec<PaperMetadata>,
    /// Number of matching papers across all pages.
    pub total: usize,
    /// Offset of this page.
    pub offset: usize,
    /// Offset of the next page, or null on the last page.
    pub next_offset: Option<usize>,
}

/// Result of `get_paper_lines`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PaperLinesOutput {
    /// Exact paper ID.
    pub paper_id: String,
    /// Resolved first line, 0-based.
    pub start: usize,
    /// Resolved end line, exclusive; equal to `start` for an empty range.
    pub end: usize,
    /// Number of lines in the paper.
    pub total_lines: usize,
    /// Lines `start..end` joined with newlines.
    pub text: String,
}

/// The paper MCP tools over a paperstore backend and its chunk index.
#[derive(Clone)]
pub struct PaperTools {
    backend: Arc<Mutex<Box<dyn StorageBackend + Send>>>,
    index: Arc<PaperIndex>,
    embedder: Arc<dyn Embedder>,
    tool_router: ToolRouter<Self>,
}

impl fmt::Debug for PaperTools {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PaperTools")
            .field("index", &self.index)
            .field("model_id", &self.embedder.model_id())
            .finish_non_exhaustive()
    }
}

#[tool_router]
impl PaperTools {
    /// Create the tools over `backend`, an `index` built from it, and the
    /// `embedder` the index was built with.
    pub fn new(
        backend: impl StorageBackend + Send + 'static,
        index: PaperIndex,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        Self {
            backend: Arc::new(Mutex::new(Box::new(backend))),
            index: Arc::new(index),
            embedder,
            tool_router: Self::tool_router(),
        }
    }

    /// The registered tools, sorted by name.
    #[must_use]
    pub fn tools(&self) -> Vec<Tool> {
        self.tool_router.list_all()
    }

    /// Search paper text by keyword, vector, or hybrid retrieval.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] naming the input for an unknown search type,
    /// an empty query, an out-of-range limit, an invalid filter, or a paper
    /// ID filter naming no paper, and a [`ToolError`] when the index cannot
    /// be searched.
    #[tool(
        description = "Search WG21 paper text. search_type is hybrid, vector, or keyword. Returns ranked chunks whose start_line/end_line can be passed to get_paper_lines."
    )]
    pub async fn search_papers(
        &self,
        Parameters(params): Parameters<SearchPapersParams>,
    ) -> Result<Json<SearchPapersOutput>, ToolError> {
        let search_type: SearchType = params.search_type.parse().map_err(ToolError::new)?;
        let filters = self.validate_filters(&params.filters)?;
        let request = SearchRequest {
            limit: params.limit.unwrap_or(DEFAULT_SEARCH_LIMIT),
            ..SearchRequest::new(params.query, search_type, filters)
        };
        let hits = search(&self.index, self.embedder.as_ref(), &request)
            .await
            .map_err(ToolError::new)?;
        Ok(Json(SearchPapersOutput {
            hits: hits.into_iter().map(SearchHitOutput::from).collect(),
        }))
    }

    /// Return one paper revision's metadata and its paper's revisions.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] naming `paper_id` when it is not an exact
    /// paperstore paper ID.
    #[tool(
        description = "Get metadata for one exact paper revision (for example P2300R10) and the list of all its revisions."
    )]
    pub async fn get_paper_metadata(
        &self,
        Parameters(params): Parameters<PaperIdParams>,
    ) -> Result<Json<PaperMetadataOutput>, ToolError> {
        let row = self.exact_paper(&params.paper_id)?;
        let num = PaperNum::parse(&row.paper_num).map_err(ToolError::new)?;
        let revisions = self.revisions(&revisionless_id(&num))?;
        Ok(Json(PaperMetadataOutput {
            paper: PaperMetadata::from(&row),
            revisions: revisions.iter().map(RevisionSummary::from).collect(),
        }))
    }

    /// List every revision of a paper.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] naming `paper_id` when it is not a paper ID
    /// or no revision of it is in paperstore.
    #[tool(
        description = "List all revisions of a paper in order. Accepts a paper ID with or without its R<n> suffix (P2300 or P2300R3)."
    )]
    pub async fn list_revisions(
        &self,
        Parameters(params): Parameters<PaperIdParams>,
    ) -> Result<Json<RevisionsOutput>, ToolError> {
        let raw = &params.paper_id;
        let paper = parse_revision_query(raw)
            .map_err(|_| ToolError::new(format!("paper_id: {raw:?} is not a paper ID")))?;
        let revisions = self.revisions(&paper)?;
        if revisions.is_empty() {
            return Err(ToolError::new(format!(
                "paper_id: {raw:?} matches no paper in paperstore"
            )));
        }
        Ok(Json(RevisionsOutput {
            paper,
            revisions: revisions.iter().map(RevisionSummary::from).collect(),
        }))
    }

    /// List paper metadata matching filters, one page at a time.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] naming the input for an out-of-range limit,
    /// an invalid filter, or a paper ID filter naming no paper, and a
    /// [`ToolError`] when the index cannot be read.
    #[tool(
        description = "List paper metadata matching the search_papers filters, paginated by offset and limit, ordered by paper ID."
    )]
    pub async fn list_papers(
        &self,
        Parameters(params): Parameters<ListPapersParams>,
    ) -> Result<Json<ListPapersOutput>, ToolError> {
        let limit = params.limit.unwrap_or(DEFAULT_LIST_LIMIT);
        if !(1..=MAX_LIST_LIMIT).contains(&limit) {
            return Err(ToolError::new(format!(
                "limit: {limit} is not between 1 and {MAX_LIST_LIMIT}"
            )));
        }
        let offset = params.offset.unwrap_or(0);
        let filters = self.validate_filters(&params.filters)?;
        let predicate = filters.to_listing_predicate();
        let ids = self
            .index
            .paper_ids_matching(predicate.as_deref())
            .await
            .map_err(ToolError::new)?;
        let rows: BTreeMap<String, PaperRow> = self
            .with_backend(|backend| backend.list_papers())?
            .into_iter()
            .map(|row| (row.paper_num.clone(), row))
            .collect();
        let matching: Vec<&PaperRow> = ids.iter().filter_map(|id| rows.get(id)).collect();
        let total = matching.len();
        let papers: Vec<PaperMetadata> = matching
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(PaperMetadata::from)
            .collect();
        let end = offset.saturating_add(papers.len());
        Ok(Json(ListPapersOutput {
            papers,
            total,
            offset,
            next_offset: (end < total).then_some(end),
        }))
    }

    /// Return a line range of a paper's Markdown with Python slice
    /// semantics.
    ///
    /// # Errors
    ///
    /// Returns a [`ToolError`] naming `paper_id` when it is not an exact
    /// paperstore paper ID or the paper has no text.
    #[tool(
        description = "Read lines of a paper's Markdown (front matter included) with Python slice semantics: 0-based start inclusive, end exclusive, negatives count from the end, omitted bounds mean the start or end of the paper."
    )]
    pub async fn get_paper_lines(
        &self,
        Parameters(params): Parameters<GetPaperLinesParams>,
    ) -> Result<Json<PaperLinesOutput>, ToolError> {
        let row = self.exact_paper(&params.paper_id)?;
        let no_text = || {
            ToolError::new(format!(
                "paper_id: {:?} has no text available; it is a metadata-only paper",
                row.paper_num
            ))
        };
        if !has_text(&row) {
            return Err(no_text());
        }
        let num = PaperNum::parse(&row.paper_num).map_err(ToolError::new)?;
        let md = match self.with_backend(|backend| backend.paper_md(&num)) {
            Err(BackendError::Store(paperstore::Error::MissingMarkdown { .. })) => {
                return Err(no_text());
            }
            other => other?,
        };
        let (start, end, total_lines, text) = slice_lines(&md, params.start, params.end);
        Ok(Json(PaperLinesOutput {
            paper_id: row.paper_num,
            start,
            end,
            total_lines,
            text,
        }))
    }
}

#[allow(
    clippy::unused_async_trait_impl,
    reason = "rmcp's tool_handler macro generates the async list_tools"
)]
#[tool_handler]
impl ServerHandler for PaperTools {}

/// A failure reading paperstore through the shared backend.
#[derive(Debug, thiserror::Error)]
enum BackendError {
    #[error(transparent)]
    Store(#[from] paperstore::Error),
    #[error("the paperstore backend is unavailable after a panic in another request")]
    Poisoned,
}

impl From<BackendError> for ToolError {
    fn from(e: BackendError) -> Self {
        Self::new(e)
    }
}

impl PaperTools {
    fn with_backend<T>(
        &self,
        f: impl FnOnce(&dyn StorageBackend) -> paperstore::Result<T>,
    ) -> Result<T, BackendError> {
        let backend = self.backend.lock().map_err(|_| BackendError::Poisoned)?;
        Ok(f(backend.as_ref())?)
    }

    /// Look up the paperstore row whose ID is exactly `raw`.
    fn exact_paper(&self, raw: &str) -> Result<PaperRow, ToolError> {
        let num = PaperNum::parse(raw)
            .map_err(|_| ToolError::new(format!("paper_id: {raw:?} is not a paper ID")))?;
        self.lookup("paper_id", raw, &num)
    }

    fn lookup(&self, field: &str, raw: &str, num: &PaperNum) -> Result<PaperRow, ToolError> {
        match self.with_backend(|backend| backend.meta(num)) {
            Err(BackendError::Store(paperstore::Error::MissingPaper { .. })) => {
                Err(ToolError::new(unknown_paper(field, raw, num)))
            }
            other => Ok(other?),
        }
    }

    /// Validate filter input, including that every paper ID filter names an
    /// exact paperstore paper.
    fn validate_filters(&self, params: &FilterParams) -> Result<SearchFilters, ToolError> {
        let filters =
            SearchFilters::from_input(&FilterInput::from(params)).map_err(ToolError::new)?;
        for num in filters.paper_ids() {
            self.lookup("paper_ids", num.as_str(), num)?;
        }
        Ok(filters)
    }

    /// Every paperstore row whose revisionless ID is `paper`, by revision
    /// number then paper ID.
    fn revisions(&self, paper: &str) -> Result<Vec<PaperRow>, ToolError> {
        let mut rows: Vec<PaperRow> = self
            .with_backend(|backend| backend.list_papers())?
            .into_iter()
            .filter(|row| {
                PaperNum::parse(&row.paper_num).is_ok_and(|num| revisionless_id(&num) == paper)
            })
            .collect();
        rows.sort_by(|a, b| {
            revision_of(a)
                .cmp(&revision_of(b))
                .then_with(|| a.paper_num.cmp(&b.paper_num))
        });
        Ok(rows)
    }
}

/// The tool error message for a well-formed paper ID with no paperstore row.
fn unknown_paper(field: &str, raw: &str, num: &PaperNum) -> String {
    if revision_number(num).is_none() {
        format!(
            "{field}: {raw:?} matches no paper in paperstore; P papers need an exact revision such as {}R0, and list_revisions lists them",
            num.as_str()
        )
    } else {
        format!("{field}: {raw:?} matches no paper in paperstore")
    }
}

fn revision_of(row: &PaperRow) -> Option<u32> {
    PaperNum::parse(&row.paper_num)
        .ok()
        .as_ref()
        .and_then(revision_number)
}

/// Whether a paper has converted text, by the rule the indexer applies.
fn has_text(row: &PaperRow) -> bool {
    row.status >= 0 && !row.markdown_path.trim().is_empty()
}
