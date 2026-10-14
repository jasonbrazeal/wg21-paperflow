//! The MCP tools (§6).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::time::Instant;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{Implementation, ServerCapabilities, ServerConfig};
use rmcp::{Json, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::bm25;
use crate::config::SearchConfig;
use crate::cursor;
use crate::dates::{Date, parse_date, parse_month};
use crate::embed::Embedder;
use crate::paper_id::{PaperId, RevisionTarget, Series};
use crate::qdrant::{ChunkSearch, Citation, DateField, Filters, MetadataPayload, Store, field};
use crate::text::{Document, LineNo, OutlineEntry, Section, outline, resolve_range};

use super::error::ToolError;

type ToolResult<T> = Result<Json<T>, ToolError>;

// ---------------------------------------------------------------- arguments

/// A paper id argument.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PaperArgs {
    /// Exact paper id, any case: P<n>R<r>, D<n>R<r>, N<n>, or SD-<n>.
    pub paper: String,
}

/// `list_revisions` arguments.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RevisionsArgs {
    /// Exact paper id, or a bare P<n> or D<n> for every revision.
    pub paper: String,
}

/// A paper id with an optional Lua-style range.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RangeArgs {
    /// Exact paper id.
    pub paper: String,
    /// First item, inclusive; negative counts from the end. Default 1.
    pub first: Option<i64>,
    /// Last item, inclusive; negative counts from the end. Default -1.
    pub last: Option<i64>,
}

/// Citation direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Ids this paper mentions.
    Cites,
    /// Papers that mention this exact id.
    CitedBy,
    /// Both.
    Both,
    /// Any other value; rejected with `invalid_argument`.
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// `get_citations` arguments.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CitationsArgs {
    /// Exact paper id.
    pub paper: String,
    /// `cites`, `cited_by`, or `both`.
    pub direction: Direction,
}

/// Search type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SearchType {
    /// BM25 full-text search.
    Fulltext,
    /// Dense vector search.
    Vector,
    /// Both, fused by reciprocal rank fusion.
    Hybrid,
    /// Any other value; rejected with `invalid_argument`.
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// The date a date range applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DateFieldArg {
    /// The document date (default).
    DocumentDate,
    /// The mailing month, compared as its first day.
    MailingDate,
    /// Any other value; rejected with `invalid_argument`.
    #[serde(other)]
    #[schemars(skip)]
    Unknown,
}

/// Filters and paging shared by both searches.
#[derive(Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct FilterArgs {
    /// Restrict to these exact paper ids.
    pub papers: Option<Vec<String>>,
    /// Earliest date, YYYY-MM-DD, inclusive.
    pub start: Option<String>,
    /// Latest date, YYYY-MM-DD, inclusive.
    pub end: Option<String>,
    /// `document_date` (default) or `mailing_date`.
    pub date_field: Option<DateFieldArg>,
    /// Matches any author whose name contains this text, ignoring case and
    /// accents.
    pub author: Option<String>,
    /// Page size. Default 10.
    pub limit: Option<u64>,
    /// The `nextCursor` of the previous page, passed back unchanged.
    #[serde(skip_serializing)]
    pub cursor: Option<String>,
}

/// `search` arguments.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchArgs {
    /// Search text. Required.
    pub query: Option<String>,
    /// `fulltext`, `vector`, or `hybrid`. Required.
    #[serde(rename = "type")]
    pub search_type: Option<SearchType>,
    /// Filters and paging.
    #[serde(flatten)]
    pub filters: FilterArgs,
}

/// `search_metadata` arguments.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchMetadataArgs {
    /// Search text over titles, authors, and other metadata. Without it, at
    /// least one filter is required and results are newest first.
    pub query: Option<String>,
    /// Filters and paging.
    #[serde(flatten)]
    pub filters: FilterArgs,
}

// ------------------------------------------------------------------ results

/// One paper's metadata.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct Metadata {
    /// Lower-case paper id.
    pub paper_id: String,
    /// Title.
    pub title: String,
    /// Author names.
    pub authors: Vec<String>,
    /// YYYY-MM-DD, or empty when unknown.
    pub document_date: String,
    /// Mailing month, YYYY-MM.
    pub mailing_date: String,
    /// Year.
    pub year: String,
    /// Target group.
    pub group: String,
    /// Intent.
    pub intent: String,
    /// Source URL.
    pub url: String,
    /// Disposition.
    pub disposition: String,
    /// Previous revision id, or empty.
    pub previous_version: String,
    /// Whether the paper has converted markdown.
    pub has_markdown: bool,
    /// Lines in the markdown, when it exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_lines: Option<u64>,
    /// Sections in the markdown, when it exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_sections: Option<u64>,
}

impl From<MetadataPayload> for Metadata {
    fn from(p: MetadataPayload) -> Self {
        Self {
            paper_id: p.paper_id,
            title: p.title,
            authors: p.authors,
            document_date: p.document_date.unwrap_or_default(),
            mailing_date: p.mailing_date,
            year: p.year,
            group: p.group,
            intent: p.intent,
            url: p.url,
            disposition: p.disposition,
            previous_version: p.previous_version,
            has_markdown: p.has_markdown,
            total_lines: p.total_lines,
            total_sections: p.total_sections,
        }
    }
}

/// A `search_metadata` result.
#[derive(Debug, Serialize, JsonSchema)]
pub struct MetadataHit {
    /// The paper's metadata.
    #[serde(flatten)]
    pub metadata: Metadata,
    /// Match score, present only when a query was given.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

/// A page of `search_metadata` results.
#[derive(Debug, Serialize, JsonSchema)]
pub struct MetadataPage {
    /// Results, best match first, or newest first without a query.
    pub results: Vec<MetadataHit>,
    /// Cursor for the next page; absent on the last page.
    #[serde(rename = "nextCursor", skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

/// One revision in a chain.
#[derive(Debug, Serialize, JsonSchema)]
pub struct Revision {
    /// Lower-case paper id.
    pub paper_id: String,
    /// YYYY-MM-DD, or empty.
    pub document_date: String,
    /// YYYY-MM, or empty.
    pub mailing_date: String,
    /// Whether a record for it exists.
    pub indexed: bool,
}

/// A revision chain.
#[derive(Debug, Serialize, JsonSchema)]
pub struct Revisions {
    /// The requested paper, lower case.
    pub paper: String,
    /// Revisions, oldest first.
    pub revisions: Vec<Revision>,
}

/// A paper's outline.
#[derive(Debug, Serialize, JsonSchema)]
pub struct Outline {
    /// Lower-case paper id.
    pub paper_id: String,
    /// Lines in the paper.
    pub total_lines: u64,
    /// Sections in the paper.
    pub total_sections: u64,
    /// Every heading, in order.
    pub headings: Vec<OutlineEntry>,
}

/// A range of lines.
#[derive(Debug, Serialize, JsonSchema)]
pub struct Lines {
    /// Lower-case paper id.
    pub paper_id: String,
    /// The lines joined with newlines.
    pub text: String,
    /// The resolved first line.
    pub first: u64,
    /// The resolved last line; below `first` when the range is empty.
    pub last: u64,
    /// Lines in the paper.
    pub total_lines: u64,
}

/// One markdown section.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SectionText {
    /// 1-based section index.
    pub section_index: u64,
    /// The cleaned heading, empty for front matter.
    pub heading: String,
    /// First line.
    pub first_line: u64,
    /// Last line.
    pub last_line: u64,
    /// The section's full markdown, heading line included.
    pub text: String,
}

/// A range of sections.
#[derive(Debug, Serialize, JsonSchema)]
pub struct Sections {
    /// Lower-case paper id.
    pub paper_id: String,
    /// Sections in the paper.
    pub total_sections: u64,
    /// The requested sections.
    pub sections: Vec<SectionText>,
}

/// A paper's citations.
#[derive(Debug, Serialize, JsonSchema)]
pub struct Citations {
    /// Lower-case paper id.
    pub paper_id: String,
    /// Whether citation extraction has run for this paper.
    pub citations_extracted: bool,
    /// Set when extraction has not run, instead of an empty `cites` list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Ids this paper mentions, most mentioned first.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cites: Option<Vec<Citation>>,
    /// Papers that mention this exact id, with their mention counts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cited_by: Option<Vec<Citation>>,
}

/// One search hit: a chunk with its provenance.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchHit {
    /// Lower-case paper id.
    pub paper_id: String,
    /// Title.
    pub title: String,
    /// Source URL.
    pub url: String,
    /// The section heading and nested headings above the chunk.
    pub heading_path: String,
    /// The chunk's section.
    pub section_index: u64,
    /// The chunk's first line.
    pub first_line: u64,
    /// The chunk's last line.
    pub last_line: u64,
    /// Lines in the chunk containing a query term.
    pub match_lines: Vec<u64>,
    /// Exactly the paper's lines `first_line` to `last_line`.
    pub text: String,
    /// Score.
    pub score: f32,
}

/// A page of `search` results.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SearchPage {
    /// Hits, best first.
    pub results: Vec<SearchHit>,
    /// Requested papers that have no markdown.
    pub no_markdown: Vec<String>,
    /// Cursor for the next page; absent on the last page.
    #[serde(rename = "nextCursor", skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

// ------------------------------------------------------------------- server

/// The MCP server: one per process, cloned per request.
#[derive(Clone)]
pub struct PaperServer {
    store: Store,
    embedder: Embedder,
    search: SearchConfig,
    tool_router: ToolRouter<Self>,
}

impl std::fmt::Debug for PaperServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PaperServer").finish_non_exhaustive()
    }
}

const INSTRUCTIONS: &str = "WG21 papers: metadata, full text, citations, and search. \
Paper ids are P<n>R<r>, D<n>R<r>, N<n>, or SD-<n>, in any case; results use lower case. \
Lines and sections are 1-based and agree across every tool and search hit. \
Text is never truncated: get_lines and get_sections return everything requested in one call.";

fn parse_id(raw: &str) -> Result<PaperId, ToolError> {
    Ok(raw.parse::<PaperId>()?)
}

fn to_u64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// A chain entry sort key: the document date, else the first of the
/// mailing month.
fn date_key(record: &MetadataPayload) -> Option<Date> {
    record
        .document_date
        .as_deref()
        .and_then(parse_date)
        .or_else(|| parse_month(&record.mailing_date))
}

fn section_text(doc: &Document<'_>, section: &Section) -> SectionText {
    SectionText {
        section_index: to_u64(section.index.get()),
        heading: section.heading.clone(),
        first_line: to_u64(section.first.get()),
        last_line: to_u64(section.last.get()),
        text: doc.text(section.first, section.last),
    }
}

struct ParsedFilters {
    filters: Filters,
    limit: u64,
    requested: Option<Vec<PaperId>>,
}

#[derive(Serialize)]
struct Fingerprint<'a> {
    tool: &'a str,
    query: Option<&'a str>,
    search_type: Option<SearchType>,
    filters: &'a FilterArgs,
}

impl PaperServer {
    /// Builds the server.
    #[must_use]
    pub fn new(store: Store, embedder: Embedder, search: SearchConfig) -> Self {
        Self {
            store,
            embedder,
            search,
            tool_router: Self::tool_router(),
        }
    }

    async fn observe<T, F>(&self, tool: &'static str, papers: &[&str], work: F) -> ToolResult<T>
    where
        F: Future<Output = ToolResult<T>>,
    {
        let started = Instant::now();
        let result = work.await;
        let code = result.as_ref().err().map_or("ok", ToolError::code);
        info!(
            tool,
            papers = ?papers,
            code,
            duration_ms = started.elapsed().as_millis(),
            "tool call"
        );
        result
    }

    async fn record(
        &self,
        id: &PaperId,
        with_markdown: bool,
    ) -> Result<MetadataPayload, ToolError> {
        self.store
            .get_one(id.as_str(), with_markdown)
            .await?
            .ok_or_else(|| ToolError::PaperNotIndexed(id.to_string()))
    }

    async fn markdown(&self, raw: &str) -> Result<(PaperId, String), ToolError> {
        let id = parse_id(raw)?;
        let record = self.record(&id, true).await?;
        match record.markdown {
            Some(markdown) if record.has_markdown => Ok((id, markdown)),
            _ => Err(ToolError::MarkdownNotConverted(id.to_string())),
        }
    }

    fn parse_filters(&self, args: &FilterArgs) -> Result<ParsedFilters, ToolError> {
        let date = |raw: &Option<String>, name: &str| -> Result<Option<Date>, ToolError> {
            raw.as_deref()
                .map(|s| {
                    parse_date(s).ok_or_else(|| {
                        ToolError::InvalidArgument(format!(
                            "{name} must be a YYYY-MM-DD date, got {s:?}"
                        ))
                    })
                })
                .transpose()
        };
        let start = date(&args.start, "start")?;
        let end = date(&args.end, "end")?;
        let date_field = match args.date_field {
            None | Some(DateFieldArg::DocumentDate) => DateField::Document,
            Some(DateFieldArg::MailingDate) => DateField::Mailing,
            Some(DateFieldArg::Unknown) => {
                return Err(ToolError::InvalidArgument(
                    "date_field must be document_date or mailing_date".to_owned(),
                ));
            }
        };
        let requested = args
            .papers
            .as_ref()
            .map(|ids| {
                ids.iter()
                    .map(|id| parse_id(id))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        let limit = args.limit.unwrap_or(self.search.default_limit);
        if limit == 0 || limit > self.search.max_limit {
            return Err(ToolError::InvalidArgument(format!(
                "limit must be between 1 and {}",
                self.search.max_limit
            )));
        }
        let author = args
            .author
            .as_deref()
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .map(bm25::fold);
        Ok(ParsedFilters {
            filters: Filters {
                paper_ids: requested
                    .as_ref()
                    .map(|ids| ids.iter().map(|id| id.as_str().to_owned()).collect()),
                start,
                end,
                date_field: Some(date_field),
                author,
            },
            limit,
            requested,
        })
    }

    fn page_offset(cursor: Option<&str>, fingerprint: &str) -> Result<u64, ToolError> {
        cursor.map_or(Ok(0), |c| {
            cursor::decode(c, fingerprint).map_err(|e| ToolError::InvalidArgument(e.to_string()))
        })
    }

    async fn get_metadata_impl(&self, paper: &str) -> ToolResult<Metadata> {
        let id = parse_id(paper)?;
        Ok(Json(self.record(&id, false).await?.into()))
    }

    async fn search_metadata_impl(&self, args: SearchMetadataArgs) -> ToolResult<MetadataPage> {
        let query = args
            .query
            .as_deref()
            .map(str::trim)
            .filter(|q| !q.is_empty());
        let parsed = self.parse_filters(&args.filters)?;
        if query.is_none() && parsed.filters.is_empty() {
            return Err(ToolError::InvalidArgument(
                "search_metadata needs a query or at least one filter".to_owned(),
            ));
        }
        let fingerprint = cursor::fingerprint(&Fingerprint {
            tool: "search_metadata",
            query,
            search_type: None,
            filters: &args.filters,
        });
        let offset = Self::page_offset(args.filters.cursor.as_deref(), &fingerprint)?;
        let limit = parsed.limit;
        let filter = parsed.filters.to_filter(true);

        let (results, more) = if let Some(query) = query {
            let mut hits = self
                .store
                .search_metadata(query, filter, offset, limit + 1)
                .await?;
            let more = to_u64(hits.len()) > limit;
            hits.truncate(usize::try_from(limit).unwrap_or(usize::MAX));
            let results = hits
                .into_iter()
                .map(|hit| MetadataHit {
                    metadata: hit.payload.into(),
                    score: Some(hit.score),
                })
                .collect();
            (results, more)
        } else {
            let mut keys = self
                .store
                .scroll_metadata(
                    filter,
                    Some(&[field::PAPER_ID, field::DOCUMENT_DATE, field::MAILING_DATE]),
                )
                .await?;
            keys.sort_by(|a, b| {
                date_key(b)
                    .cmp(&date_key(a))
                    .then_with(|| a.paper_id.cmp(&b.paper_id))
            });
            let start = usize::try_from(offset)
                .unwrap_or(usize::MAX)
                .min(keys.len());
            let end = start
                .saturating_add(usize::try_from(limit).unwrap_or(usize::MAX))
                .min(keys.len());
            let page: Vec<String> = keys[start..end]
                .iter()
                .map(|k| k.paper_id.clone())
                .collect();
            let mut records: HashMap<String, MetadataPayload> = self
                .store
                .get_metadata(&page, false)
                .await?
                .into_iter()
                .map(|r| (r.paper_id.clone(), r))
                .collect();
            let results = page
                .iter()
                .filter_map(|id| records.remove(id))
                .map(|r| MetadataHit {
                    metadata: r.into(),
                    score: None,
                })
                .collect();
            (results, end < keys.len())
        };
        Ok(Json(MetadataPage {
            results,
            next_cursor: more.then(|| cursor::encode(&fingerprint, offset + limit)),
        }))
    }

    async fn list_revisions_impl(&self, paper: &str) -> ToolResult<Revisions> {
        let target: RevisionTarget = paper.parse()?;
        let (series, number, requested) = match &target {
            RevisionTarget::Exact(id) => (id.series(), id.number(), id.to_string()),
            RevisionTarget::Bare { series, number } => {
                (*series, *number, paper.to_ascii_lowercase())
            }
        };
        if let RevisionTarget::Exact(id) = &target {
            self.record(id, false).await?;
        }
        let fields = [
            field::PAPER_ID,
            field::DOCUMENT_DATE,
            field::MAILING_DATE,
            field::PREVIOUS_VERSION,
            field::SERIES,
            field::NUMBER,
        ];
        let mut chain: BTreeMap<String, MetadataPayload> = BTreeMap::new();
        let mut missing: BTreeSet<String> = BTreeSet::new();
        let mut numbers_done: BTreeSet<(String, u64)> = BTreeSet::new();
        let mut numbers: Vec<(Series, u64)> = vec![(series, number)];
        loop {
            while let Some((series, number)) = numbers.pop() {
                if !numbers_done.insert((series.as_str().to_owned(), number)) {
                    continue;
                }
                let filter = qdrant_client::qdrant::Filter::must([
                    qdrant_client::qdrant::Condition::matches(
                        field::SERIES,
                        series.as_str().to_owned(),
                    ),
                    qdrant_client::qdrant::Condition::matches(
                        field::NUMBER,
                        i64::try_from(number).unwrap_or(i64::MAX),
                    ),
                ]);
                for record in self
                    .store
                    .scroll_metadata(Some(filter), Some(&fields))
                    .await?
                {
                    chain.entry(record.paper_id.clone()).or_insert(record);
                }
            }
            let mut grew = false;
            let wanted: Vec<String> = chain
                .values()
                .map(|r| r.previous_version.clone())
                .filter(|p| !p.is_empty() && !chain.contains_key(p) && !missing.contains(p))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let found = self.store.get_metadata(&wanted, false).await?;
            for id in &wanted {
                if !found.iter().any(|r| &r.paper_id == id) {
                    missing.insert(id.clone());
                }
            }
            for record in found {
                if let Ok(id) = record.paper_id.parse::<PaperId>() {
                    numbers.push((id.series(), id.number()));
                }
                chain.insert(record.paper_id.clone(), record);
                grew = true;
            }
            let ids: Vec<String> = chain.keys().cloned().collect();
            let successors = self
                .store
                .scroll_metadata(
                    Some(qdrant_client::qdrant::Filter::must([
                        qdrant_client::qdrant::Condition::matches(field::PREVIOUS_VERSION, ids),
                    ])),
                    Some(&fields),
                )
                .await?;
            for record in successors {
                if !chain.contains_key(&record.paper_id) {
                    if let Ok(id) = record.paper_id.parse::<PaperId>() {
                        numbers.push((id.series(), id.number()));
                    }
                    chain.insert(record.paper_id.clone(), record);
                    grew = true;
                }
            }
            if !grew && numbers.is_empty() {
                break;
            }
        }
        if chain.is_empty() {
            return Err(ToolError::PaperNotIndexed(requested));
        }
        Ok(Json(Revisions {
            paper: requested,
            revisions: order_chain(&chain, &missing),
        }))
    }

    async fn get_outline_impl(&self, paper: &str) -> ToolResult<Outline> {
        let (id, markdown) = self.markdown(paper).await?;
        let doc = Document::parse(&markdown);
        let sections = doc.sections();
        Ok(Json(Outline {
            paper_id: id.to_string(),
            total_lines: to_u64(doc.line_count()),
            total_sections: to_u64(sections.len()),
            headings: outline(&doc, &sections),
        }))
    }

    async fn get_lines_impl(&self, args: RangeArgs) -> ToolResult<Lines> {
        let (id, markdown) = self.markdown(&args.paper).await?;
        let doc = Document::parse(&markdown);
        let range = resolve_range(args.first, args.last, doc.line_count());
        let text = range
            .bounds()
            .map(|(first, last)| doc.text(first, last))
            .unwrap_or_default();
        Ok(Json(Lines {
            paper_id: id.to_string(),
            text,
            first: to_u64(range.first),
            last: to_u64(range.last),
            total_lines: to_u64(doc.line_count()),
        }))
    }

    async fn get_sections_impl(&self, args: RangeArgs) -> ToolResult<Sections> {
        let (id, markdown) = self.markdown(&args.paper).await?;
        let doc = Document::parse(&markdown);
        let sections = doc.sections();
        let range = resolve_range(args.first, args.last, sections.len());
        let selected = range
            .bounds()
            .map(|(first, last)| {
                sections[first.zero_based()..last.get()]
                    .iter()
                    .map(|s| section_text(&doc, s))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Json(Sections {
            paper_id: id.to_string(),
            total_sections: to_u64(sections.len()),
            sections: selected,
        }))
    }

    async fn get_citations_impl(&self, args: CitationsArgs) -> ToolResult<Citations> {
        let id = parse_id(&args.paper)?;
        let (want_cites, want_cited_by) = match args.direction {
            Direction::Cites => (true, false),
            Direction::CitedBy => (false, true),
            Direction::Both => (true, true),
            Direction::Unknown => {
                return Err(ToolError::InvalidArgument(
                    "direction must be cites, cited_by, or both".to_owned(),
                ));
            }
        };
        let record = self.record(&id, false).await?;
        let extracted = record.citations_extracted;
        let cites = (want_cites && extracted).then(|| {
            let mut cites = record.cites.clone();
            cites.sort_by(|a, b| {
                b.count
                    .cmp(&a.count)
                    .then_with(|| a.paper_id.cmp(&b.paper_id))
            });
            cites
        });
        let cited_by =
            if want_cited_by {
                let filter = qdrant_client::qdrant::Filter::must([
                    qdrant_client::qdrant::Condition::matches(field::CITED_IDS, id.to_string()),
                ]);
                let citing = self
                    .store
                    .scroll_metadata(Some(filter), Some(&[field::PAPER_ID, "cites"]))
                    .await?;
                let mut list: Vec<Citation> = citing
                    .into_iter()
                    .map(|r| Citation {
                        count: r
                            .cites
                            .iter()
                            .find(|c| c.paper_id == id.as_str())
                            .map_or(0, |c| c.count),
                        paper_id: r.paper_id,
                    })
                    .collect();
                list.sort_by(|a, b| a.paper_id.cmp(&b.paper_id));
                Some(list)
            } else {
                None
            };
        Ok(Json(Citations {
            paper_id: id.to_string(),
            citations_extracted: extracted,
            message: (!extracted)
                .then(|| "citation extraction has not run for this paper".to_owned()),
            cites,
            cited_by,
        }))
    }

    /// The requested papers that have no markdown.
    async fn no_markdown(&self, requested: Option<&[PaperId]>) -> Result<Vec<String>, ToolError> {
        let Some(requested) = requested else {
            return Ok(Vec::new());
        };
        let ids: Vec<String> = requested.iter().map(|id| id.as_str().to_owned()).collect();
        let records = self.store.get_metadata(&ids, false).await?;
        let mut out = Vec::new();
        for id in ids {
            match records.iter().find(|r| r.paper_id == id) {
                None => return Err(ToolError::PaperNotIndexed(id)),
                Some(r) if !r.has_markdown => out.push(id),
                Some(_) => {}
            }
        }
        Ok(out)
    }

    /// Replaces the author filter, which `paper_chunks` cannot apply, with
    /// the matching paper ids. Returns `None` when no paper can match.
    async fn resolve_author(&self, mut filters: Filters) -> Result<Option<Filters>, ToolError> {
        let Some(author) = filters.author.take() else {
            return Ok(Some(filters));
        };
        let author_filter = Filters {
            author: Some(author),
            ..Filters::default()
        };
        let matching: BTreeSet<String> = self
            .store
            .scroll_metadata(author_filter.to_filter(true), Some(&[field::PAPER_ID]))
            .await?
            .into_iter()
            .map(|r| r.paper_id)
            .collect();
        let ids: Vec<String> = match filters.paper_ids.take() {
            Some(requested) => requested
                .into_iter()
                .filter(|id| matching.contains(id))
                .collect(),
            None => matching.into_iter().collect(),
        };
        if ids.is_empty() {
            return Ok(None);
        }
        filters.paper_ids = Some(ids);
        Ok(Some(filters))
    }

    async fn search_impl(&self, args: SearchArgs) -> ToolResult<SearchPage> {
        let query = args
            .query
            .as_deref()
            .map(str::trim)
            .filter(|q| !q.is_empty())
            .ok_or_else(|| ToolError::InvalidArgument("search needs a query".to_owned()))?;
        let search_type = match args.search_type {
            None => return Err(ToolError::InvalidArgument("search needs a type".to_owned())),
            Some(SearchType::Unknown) => {
                return Err(ToolError::InvalidArgument(
                    "type must be fulltext, vector, or hybrid".to_owned(),
                ));
            }
            Some(t) => t,
        };
        let parsed = self.parse_filters(&args.filters)?;
        let fingerprint = cursor::fingerprint(&Fingerprint {
            tool: "search",
            query: Some(query),
            search_type: Some(search_type),
            filters: &args.filters,
        });
        let offset = Self::page_offset(args.filters.cursor.as_deref(), &fingerprint)?;
        let limit = parsed.limit;

        let no_markdown = self.no_markdown(parsed.requested.as_deref()).await?;

        let Some(filters) = self.resolve_author(parsed.filters).await? else {
            return Ok(Json(SearchPage {
                results: Vec::new(),
                no_markdown,
                next_cursor: None,
            }));
        };

        let search = match search_type {
            SearchType::Fulltext | SearchType::Unknown => ChunkSearch::Fulltext,
            SearchType::Vector => ChunkSearch::Vector(self.embedder.embed_one(query).await?),
            SearchType::Hybrid => ChunkSearch::Hybrid(self.embedder.embed_one(query).await?),
        };
        let mut hits = self
            .store
            .search_chunks(query, search, filters.to_filter(false), offset, limit + 1)
            .await?;
        let more = to_u64(hits.len()) > limit;
        hits.truncate(usize::try_from(limit).unwrap_or(usize::MAX));
        let results = hits
            .into_iter()
            .map(|hit| {
                let chunk = hit.payload;
                let first = usize::try_from(chunk.first_line)
                    .ok()
                    .and_then(LineNo::new)
                    .unwrap_or(LineNo::FIRST);
                SearchHit {
                    match_lines: bm25::match_lines(&chunk.text, first, query)
                        .into_iter()
                        .map(|l| to_u64(l.get()))
                        .collect(),
                    paper_id: chunk.paper_id,
                    title: chunk.title,
                    url: chunk.url,
                    heading_path: chunk.heading_path,
                    section_index: chunk.section_index,
                    first_line: chunk.first_line,
                    last_line: chunk.last_line,
                    text: chunk.text,
                    score: hit.score,
                }
            })
            .collect();
        Ok(Json(SearchPage {
            results,
            no_markdown,
            next_cursor: more.then(|| cursor::encode(&fingerprint, offset + limit)),
        }))
    }
}

/// Orders a revision chain oldest first: indexed records by date, then
/// series, number, and revision; each unindexed id goes just before the
/// first record whose `previous_version` names it.
fn order_chain(
    chain: &BTreeMap<String, MetadataPayload>,
    missing: &BTreeSet<String>,
) -> Vec<Revision> {
    let mut records: Vec<&MetadataPayload> = chain.values().collect();
    let sort_key = |r: &MetadataPayload| {
        let id = r.paper_id.parse::<PaperId>().ok();
        (
            date_key(r),
            id.as_ref().map(|i| (i.series(), i.number(), i.revision())),
            r.paper_id.clone(),
        )
    };
    records.sort_by_key(|r| sort_key(r));
    let mut out = Vec::new();
    let mut placed: BTreeSet<&str> = BTreeSet::new();
    for record in records {
        let previous = record.previous_version.as_str();
        if missing.contains(previous) && placed.insert(previous) {
            out.push(Revision {
                paper_id: previous.to_owned(),
                document_date: String::new(),
                mailing_date: String::new(),
                indexed: false,
            });
        }
        out.push(Revision {
            paper_id: record.paper_id.clone(),
            document_date: record.document_date.clone().unwrap_or_default(),
            mailing_date: record.mailing_date.clone(),
            indexed: true,
        });
    }
    out
}

#[tool_router]
impl PaperServer {
    /// get_metadata
    #[tool(
        name = "get_metadata",
        description = "Return one paper's metadata: paper_id, title, authors, document_date, mailing_date, year, group, intent, url, disposition, previous_version, has_markdown, and, when markdown exists, total_lines and total_sections."
    )]
    async fn get_metadata(
        &self,
        Parameters(args): Parameters<PaperArgs>,
    ) -> Result<Json<Metadata>, ToolError> {
        self.observe(
            "get_metadata",
            &[&args.paper],
            self.get_metadata_impl(&args.paper),
        )
        .await
    }

    /// search_metadata
    #[tool(
        name = "search_metadata",
        description = "Search the metadata of every paper (titles, authors, ids, groups, intent, disposition, dates). Returns results with get_metadata's fields; with a query they are ranked best first and carry a score, without one they are newest first and at least one filter is required. Supports papers, start, end, date_field, author, limit, and cursor; pass nextCursor back as cursor for the next page."
    )]
    async fn search_metadata(
        &self,
        Parameters(args): Parameters<SearchMetadataArgs>,
    ) -> Result<Json<MetadataPage>, ToolError> {
        let papers: Vec<String> = args.filters.papers.clone().unwrap_or_default();
        let refs: Vec<&str> = papers.iter().map(String::as_str).collect();
        self.observe(
            "search_metadata",
            &refs,
            Box::pin(self.search_metadata_impl(args)),
        )
        .await
    }

    /// list_revisions
    #[tool(
        name = "list_revisions",
        description = "Return a paper's revision chain, oldest first, each with paper_id, document_date, mailing_date, and indexed. Accepts an exact id or a bare P<number> or D<number>; follows renumbered papers through previous_version."
    )]
    async fn list_revisions(
        &self,
        Parameters(args): Parameters<RevisionsArgs>,
    ) -> Result<Json<Revisions>, ToolError> {
        self.observe(
            "list_revisions",
            &[&args.paper],
            self.list_revisions_impl(&args.paper),
        )
        .await
    }

    /// get_outline
    #[tool(
        name = "get_outline",
        description = "Return total_lines, total_sections, and every heading in order, each with its line, level, cleaned text, the section_index of the get_sections section that contains it, and whether it starts that section."
    )]
    async fn get_outline(
        &self,
        Parameters(args): Parameters<PaperArgs>,
    ) -> Result<Json<Outline>, ToolError> {
        self.observe(
            "get_outline",
            &[&args.paper],
            self.get_outline_impl(&args.paper),
        )
        .await
    }

    /// get_lines
    #[tool(
        name = "get_lines",
        description = "Return the whole paper or a range of lines in one call, never truncated. first and last work like Lua string.sub: inclusive, default 1 and -1, negative values count from the end, out-of-range values clamp. Returns paper_id, text, the resolved first and last, and total_lines."
    )]
    async fn get_lines(
        &self,
        Parameters(args): Parameters<RangeArgs>,
    ) -> Result<Json<Lines>, ToolError> {
        let paper = args.paper.clone();
        self.observe("get_lines", &[&paper], self.get_lines_impl(args))
            .await
    }

    /// get_sections
    #[tool(
        name = "get_sections",
        description = "Return the whole paper or a range of sections as markdown, in one call, never truncated. first and last index sections like get_lines indexes lines. Each section has section_index, heading, first_line, last_line, and its full text."
    )]
    async fn get_sections(
        &self,
        Parameters(args): Parameters<RangeArgs>,
    ) -> Result<Json<Sections>, ToolError> {
        let paper = args.paper.clone();
        self.observe("get_sections", &[&paper], self.get_sections_impl(args))
            .await
    }

    /// get_citations
    #[tool(
        name = "get_citations",
        description = "Return the paper ids this paper mentions (cites), the papers that mention this exact id (cited_by), or both, with mention counts. A citation records a mention, not support or rebuttal. When extraction has not run for the paper, citations_extracted is false and cites is omitted."
    )]
    async fn get_citations(
        &self,
        Parameters(args): Parameters<CitationsArgs>,
    ) -> Result<Json<Citations>, ToolError> {
        let paper = args.paper.clone();
        self.observe("get_citations", &[&paper], self.get_citations_impl(args))
            .await
    }

    /// search
    #[tool(
        name = "search",
        description = "Search paper text. query and type (fulltext, vector, or hybrid) are required. Each hit is a chunk with paper_id, title, url, heading_path, section_index, first_line, last_line, match_lines, text (exactly get_lines over that range, except that a chunk of a line too long for one chunk holds only its part of that line), and score. Supports papers, start, end, date_field, author, limit, and cursor; pass nextCursor back as cursor for the next page. Requested papers without markdown are listed in no_markdown."
    )]
    async fn search(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> Result<Json<SearchPage>, ToolError> {
        let papers: Vec<String> = args.filters.papers.clone().unwrap_or_default();
        let refs: Vec<&str> = papers.iter().map(String::as_str).collect();
        self.observe("search", &refs, Box::pin(self.search_impl(args)))
            .await
    }
}

#[tool_handler(router = self.tool_router)]
#[allow(clippy::unused_async_trait_impl)]
impl ServerHandler for PaperServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("paper-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, date: &str, previous: &str) -> MetadataPayload {
        MetadataPayload {
            paper_id: id.to_owned(),
            document_date: (!date.is_empty()).then(|| date.to_owned()),
            previous_version: previous.to_owned(),
            ..MetadataPayload::default()
        }
    }

    #[test]
    fn orders_chains_by_date_and_places_missing_links() {
        let mut chain = BTreeMap::new();
        for r in [
            record("p4353r0", "2026-01-10", "p2034r9"),
            record("p2034r8", "2024-01-10", "p2034r7"),
            record("p2034r9", "2025-01-10", "p2034r8"),
        ] {
            chain.insert(r.paper_id.clone(), r);
        }
        let missing: BTreeSet<String> = ["p2034r7".to_owned()].into();
        let order: Vec<(String, bool)> = order_chain(&chain, &missing)
            .into_iter()
            .map(|r| (r.paper_id, r.indexed))
            .collect();
        assert_eq!(
            order,
            vec![
                ("p2034r7".to_owned(), false),
                ("p2034r8".to_owned(), true),
                ("p2034r9".to_owned(), true),
                ("p4353r0".to_owned(), true),
            ]
        );
    }

    #[test]
    fn every_tool_declares_an_output_schema() {
        let tools = PaperServer::tool_router().list_all();
        let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "get_citations",
                "get_lines",
                "get_metadata",
                "get_outline",
                "get_sections",
                "list_revisions",
                "search",
                "search_metadata"
            ]
        );
        assert!(tools.iter().all(|t| t.output_schema.is_some()));
    }

    #[test]
    fn unknown_enum_values_deserialize_for_invalid_argument() {
        let args: SearchArgs = serde_json::from_value(serde_json::json!({
            "query": "x", "type": "semantic", "date_field": "bogus"
        }))
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(args.search_type, Some(SearchType::Unknown));
        assert_eq!(args.filters.date_field, Some(DateFieldArg::Unknown));
    }
}
