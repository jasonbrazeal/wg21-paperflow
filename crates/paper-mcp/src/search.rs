//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Keyword, vector, and hybrid search over the chunk index.
//!
//! [`search`] ranks chunks with LanceDB BM25 full-text search for
//! [`SearchType::Keyword`], exact cosine nearest-neighbor search for
//! [`SearchType::Vector`], or both fused with reciprocal rank fusion for
//! [`SearchType::Hybrid`]. Ties at the limit are broken by paper ID and
//! start line, never by LanceDB's internal order, so the same index and
//! request always select the same hits. The request's [`SearchFilters`] predicate is
//! applied before ranking in every case, so a filtered search fills its
//! limit whenever enough matching chunks exist.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::fmt;
use std::str::FromStr;

use futures::TryStreamExt;
use lancedb::arrow::arrow_array::cast::AsArray;
use lancedb::arrow::arrow_array::types::Float32Type;
use lancedb::arrow::arrow_array::{Array, Float32Array, RecordBatch};
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{DistanceType, Table};

use crate::chunk::{ChunkKind, UnknownChunkKind};
use crate::embed::{EmbedError, Embedder};
use crate::filter::SearchFilters;
use crate::index::{
    AUTHORS_SEPARATOR, CHUNKS_TABLE, FTS_COLUMN, IndexError, PaperIndex, VECTOR_COLUMN,
    string_column, u32_column,
};

/// Hits returned when a request does not set a limit.
pub const DEFAULT_SEARCH_LIMIT: usize = 10;

/// Largest limit a request may set.
pub const MAX_SEARCH_LIMIT: usize = 50;

/// Prefix of a paper's canonical URL; the paper ID follows it.
pub const WG21_LINK_BASE: &str = "https://wg21.link/";

/// Score column LanceDB adds to full-text search results (BM25, higher is
/// better).
const KEYWORD_SCORE_COLUMN: &str = "_score";

/// Distance column LanceDB adds to vector search results (lower is better).
const DISTANCE_COLUMN: &str = "_distance";

/// Rank offset in reciprocal rank fusion, LanceDB's `RRFReranker` default.
const RRF_K: f32 = 60.0;

/// Factor by which a ranked list's fetch limit grows while the chunks it
/// returns are all tied with the last kept chunk.
const OVERFETCH_FACTOR: usize = 2;

/// Chunk columns read into each [`SearchHit`].
const HIT_COLUMNS: [&str; 10] = [
    "chunk_id",
    "paper_id",
    "revision",
    "title",
    "authors",
    "section_path",
    "chunk_kind",
    "start_line",
    "end_line",
    "text",
];

/// Retrieval method of a search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SearchType {
    /// Keyword and vector search fused with reciprocal rank fusion.
    Hybrid,
    /// Dense embedding search.
    Vector,
    /// BM25 full-text search.
    Keyword,
}

impl SearchType {
    /// Every search type, in the order they are documented.
    pub const ALL: [Self; 3] = [Self::Hybrid, Self::Vector, Self::Keyword];

    /// Name accepted by [`SearchType::from_str`].
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::Vector => "vector",
            Self::Keyword => "keyword",
        }
    }
}

impl fmt::Display for SearchType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A search type name that is not one of the [`SearchType`] names.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
#[error("search_type: {0:?} is not a search type; expected hybrid, vector, or keyword")]
pub struct UnknownSearchType(pub String);

impl FromStr for SearchType {
    type Err = UnknownSearchType;

    /// Parse a name produced by [`SearchType::as_str`], exactly as named.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|t| t.as_str() == s)
            .ok_or_else(|| UnknownSearchType(s.to_owned()))
    }
}

/// A validated-filter search over the chunk index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchRequest {
    /// Query text. Must contain a non-whitespace character.
    pub query: String,
    /// Retrieval method.
    pub search_type: SearchType,
    /// Filters applied before ranking.
    pub filters: SearchFilters,
    /// Maximum hits to return, from 1 to [`MAX_SEARCH_LIMIT`].
    pub limit: usize,
}

impl SearchRequest {
    /// Create a request returning up to [`DEFAULT_SEARCH_LIMIT`] hits.
    pub fn new(query: impl Into<String>, search_type: SearchType, filters: SearchFilters) -> Self {
        Self {
            query: query.into(),
            search_type,
            filters,
            limit: DEFAULT_SEARCH_LIMIT,
        }
    }
}

/// One ranked chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    /// Exact paper ID, for example `P2300R10`.
    pub paper_id: String,
    /// Revision number, or `None` for N and SD papers.
    pub revision: Option<u32>,
    /// Paper title.
    pub title: String,
    /// Author names in paperstore order.
    pub authors: Vec<String>,
    /// Heading titles enclosing the chunk, outermost first.
    pub section_path: Vec<String>,
    /// Category of the chunk.
    pub chunk_kind: ChunkKind,
    /// First line of the chunk in the paper's Markdown, 0-based.
    pub start_line: u32,
    /// Line after the chunk's last line (exclusive).
    pub end_line: u32,
    /// Chunk text, without the contextual prefix.
    pub text: String,
    /// Relevance; higher is better. BM25 for keyword search, cosine
    /// similarity for vector search, and the fused rank score for hybrid
    /// search, so scores are comparable only within one search type.
    pub score: f32,
    /// Canonical URL, [`WG21_LINK_BASE`] followed by the paper ID.
    pub url: String,
}

/// Errors from [`search`].
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    /// The query is empty or whitespace only.
    #[error("query: value is empty")]
    EmptyQuery,
    /// The limit is 0 or above [`MAX_SEARCH_LIMIT`].
    #[error("limit: {0} is not between 1 and {MAX_SEARCH_LIMIT}")]
    Limit(usize),
    /// The query could not be embedded.
    #[error(transparent)]
    Embed(#[from] EmbedError),
    /// The index could not be queried or returned malformed rows.
    #[error(transparent)]
    Index(#[from] IndexError),
}

impl From<lancedb::Error> for SearchError {
    fn from(e: lancedb::Error) -> Self {
        Self::Index(e.into())
    }
}

/// Search the chunk index.
///
/// Only chunks matching `request.filters` are ranked, so the result holds
/// `request.limit` hits whenever at least that many matching chunks match
/// the query. Hits are ordered by descending score, then paper ID, then
/// start line.
///
/// # Errors
///
/// Returns [`SearchError::EmptyQuery`] or [`SearchError::Limit`] for an
/// invalid request, [`SearchError::Embed`] when the query cannot be
/// embedded, and [`SearchError::Index`] when the index cannot be queried.
pub async fn search(
    index: &PaperIndex,
    embedder: &dyn Embedder,
    request: &SearchRequest,
) -> Result<Vec<SearchHit>, SearchError> {
    let query = request.query.trim();
    if query.is_empty() {
        return Err(SearchError::EmptyQuery);
    }
    if !(1..=MAX_SEARCH_LIMIT).contains(&request.limit) {
        return Err(SearchError::Limit(request.limit));
    }
    let predicate = request.filters.to_predicate();
    let table = index.chunks_table().await?;
    let mut hits = match request.search_type {
        SearchType::Keyword => {
            leg_candidates(&table, &predicate, &Leg::Keyword(query), request.limit).await?
        }
        SearchType::Vector => {
            let vector = embed_query(embedder, query)?;
            leg_candidates(&table, &predicate, &Leg::Vector(&vector), request.limit).await?
        }
        SearchType::Hybrid => {
            let vector = embed_query(embedder, query)?;
            let keyword =
                leg_candidates(&table, &predicate, &Leg::Keyword(query), request.limit).await?;
            let dense =
                leg_candidates(&table, &predicate, &Leg::Vector(&vector), request.limit).await?;
            reciprocal_rank_fusion(dense, keyword)
        }
    };
    hits.sort_by(candidate_order);
    hits.truncate(request.limit);
    Ok(hits.into_iter().map(|c| c.hit).collect())
}

/// One ranked list feeding a search: BM25 over the query text, or cosine
/// nearest neighbors of the query vector.
enum Leg<'q> {
    Keyword(&'q str),
    Vector(&'q [f32]),
}

/// A hit with the chunk ID that identifies it across ranked lists.
struct Candidate {
    chunk_id: String,
    hit: SearchHit,
}

/// Rank the filtered chunks for `leg`, keeping the best `keep` and every
/// chunk tied with the last of them, in [`candidate_order`].
///
/// LanceDB cuts a query at its limit in its own order, so a tie at the
/// limit would let the library pick which tied chunks survive. The query
/// limit is grown until a fetched chunk scores strictly below the `keep`th,
/// or the filtered chunks run out, so the kept set does not depend on that
/// order.
async fn leg_candidates(
    table: &Table,
    predicate: &str,
    leg: &Leg<'_>,
    keep: usize,
) -> Result<Vec<Candidate>, SearchError> {
    let mut fetch = keep;
    loop {
        let mut candidates = run_leg(table, predicate, leg, fetch).await?;
        candidates.sort_by(candidate_order);
        let cutoff = candidates.get(keep - 1).map(|c| c.hit.score);
        let past_ties = match (cutoff, candidates.last()) {
            (Some(cutoff), Some(last)) => last.hit.score.total_cmp(&cutoff) == Ordering::Less,
            _ => false,
        };
        if candidates.len() < fetch || past_ties {
            if let Some(cutoff) = cutoff {
                candidates.retain(|c| c.hit.score.total_cmp(&cutoff) != Ordering::Less);
            }
            return Ok(candidates);
        }
        fetch = fetch.saturating_mul(OVERFETCH_FACTOR);
    }
}

/// Run one LanceDB query for `leg` returning up to `fetch` filtered chunks.
async fn run_leg(
    table: &Table,
    predicate: &str,
    leg: &Leg<'_>,
    fetch: usize,
) -> Result<Vec<Candidate>, SearchError> {
    let base = table
        .query()
        .only_if(predicate)
        .limit(fetch)
        .select(Select::columns(&HIT_COLUMNS));
    let batches: Vec<RecordBatch> = match leg {
        Leg::Keyword(query) => {
            let full_text = FullTextSearchQuery::new((*query).to_owned())
                .with_column(FTS_COLUMN.to_owned())
                .map_err(|e| SearchError::Index(schema_error(e.to_string())))?;
            base.full_text_search(full_text)
                .execute()
                .await?
                .try_collect()
                .await?
        }
        Leg::Vector(vector) => {
            base.nearest_to(vector.to_vec())?
                .column(VECTOR_COLUMN)
                .distance_type(DistanceType::Cosine)
                .execute()
                .await?
                .try_collect()
                .await?
        }
    };
    let mut candidates = Vec::new();
    for batch in batches.iter().filter(|b| b.num_rows() > 0) {
        read_candidates(batch, leg, &mut candidates)?;
    }
    Ok(candidates)
}

/// Fuse two lists in [`candidate_order`] by reciprocal rank: each chunk
/// scores the sum of `1 / (rank + RRF_K)` over the lists holding it, with
/// rank 0-based.
fn reciprocal_rank_fusion(dense: Vec<Candidate>, keyword: Vec<Candidate>) -> Vec<Candidate> {
    let mut fused: BTreeMap<String, Candidate> = BTreeMap::new();
    for list in [dense, keyword] {
        for (rank, mut candidate) in list.into_iter().enumerate() {
            #[allow(clippy::cast_precision_loss, reason = "ranks are far below 2^24")]
            let score = 1.0 / (rank as f32 + RRF_K);
            match fused.entry(candidate.chunk_id.clone()) {
                Entry::Occupied(mut e) => e.get_mut().hit.score += score,
                Entry::Vacant(e) => {
                    candidate.hit.score = score;
                    e.insert(candidate);
                }
            }
        }
    }
    fused.into_values().collect()
}

/// Embed `query`, checking the vector has the embedder's dimension.
fn embed_query(embedder: &dyn Embedder, query: &str) -> Result<Vec<f32>, EmbedError> {
    let vector = embedder.embed_query(query)?;
    if vector.len() != embedder.dim() {
        return Err(EmbedError::DimMismatch {
            expected: embedder.dim(),
            actual: vector.len(),
        });
    }
    Ok(vector)
}

/// Descending score, then ascending paper ID, then ascending start line.
fn hit_order(a: &SearchHit, b: &SearchHit) -> Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.paper_id.cmp(&b.paper_id))
        .then_with(|| a.start_line.cmp(&b.start_line))
}

/// [`hit_order`], then ascending chunk ID, a total order over chunks.
fn candidate_order(a: &Candidate, b: &Candidate) -> Ordering {
    hit_order(&a.hit, &b.hit).then_with(|| a.chunk_id.cmp(&b.chunk_id))
}

fn schema_error(detail: String) -> IndexError {
    IndexError::Schema {
        table: CHUNKS_TABLE,
        detail,
    }
}

fn f32_column<'b>(batch: &'b RecordBatch, column: &str) -> Result<&'b Float32Array, IndexError> {
    batch
        .column_by_name(column)
        .and_then(|c| c.as_primitive_opt::<Float32Type>())
        .ok_or_else(|| schema_error(format!("{column} is not an f32 column")))
}

/// Append the chunks in `batch`, a result of `leg`, to `candidates`.
fn read_candidates(
    batch: &RecordBatch,
    leg: &Leg<'_>,
    candidates: &mut Vec<Candidate>,
) -> Result<(), IndexError> {
    let chunk_ids = string_column(batch, CHUNKS_TABLE, "chunk_id")?;
    let paper_ids = string_column(batch, CHUNKS_TABLE, "paper_id")?;
    let revisions = u32_column(batch, CHUNKS_TABLE, "revision")?;
    let titles = string_column(batch, CHUNKS_TABLE, "title")?;
    let authors = string_column(batch, CHUNKS_TABLE, "authors")?;
    let kinds = string_column(batch, CHUNKS_TABLE, "chunk_kind")?;
    let starts = u32_column(batch, CHUNKS_TABLE, "start_line")?;
    let ends = u32_column(batch, CHUNKS_TABLE, "end_line")?;
    let texts = string_column(batch, CHUNKS_TABLE, "text")?;
    let sections = batch
        .column_by_name("section_path")
        .and_then(|c| c.as_list_opt::<i32>())
        .ok_or_else(|| schema_error("section_path is not a list column".to_owned()))?;
    let (scores, is_distance) = match leg {
        Leg::Keyword(_) => (f32_column(batch, KEYWORD_SCORE_COLUMN)?, false),
        Leg::Vector(_) => (f32_column(batch, DISTANCE_COLUMN)?, true),
    };
    for i in 0..batch.num_rows() {
        let paper_id = paper_ids.value(i).to_owned();
        let kind = if kinds.is_null(i) {
            return Err(schema_error(format!(
                "{paper_id} has a text chunk with no chunk_kind"
            )));
        } else {
            kinds
                .value(i)
                .parse()
                .map_err(|UnknownChunkKind(kind)| schema_error(format!("chunk_kind {kind:?}")))?
        };
        let section_path = sections.value(i);
        let section_path = section_path
            .as_string_opt::<i32>()
            .ok_or_else(|| schema_error("section_path items are not strings".to_owned()))?
            .iter()
            .map(|s| s.unwrap_or_default().to_owned())
            .collect();
        let score = scores.value(i);
        let hit = SearchHit {
            url: format!("{WG21_LINK_BASE}{paper_id}"),
            paper_id,
            revision: (!revisions.is_null(i)).then(|| revisions.value(i)),
            title: titles.value(i).to_owned(),
            authors: split_authors(authors.value(i)),
            section_path,
            chunk_kind: kind,
            start_line: starts.value(i),
            end_line: ends.value(i),
            text: texts.value(i).to_owned(),
            score: if is_distance { 1.0 - score } else { score },
        };
        candidates.push(Candidate {
            chunk_id: chunk_ids.value(i).to_owned(),
            hit,
        });
    }
    Ok(())
}

/// Split an `authors` column value back into names.
fn split_authors(joined: &str) -> Vec<String> {
    if joined.is_empty() {
        return Vec::new();
    }
    joined.split(AUTHORS_SEPARATOR).map(str::to_owned).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_type_names_round_trip() {
        for search_type in SearchType::ALL {
            assert_eq!(search_type.as_str().parse(), Ok(search_type));
        }
        assert_eq!(
            ["hybrid", "vector", "keyword"].map(str::parse::<SearchType>),
            [
                Ok(SearchType::Hybrid),
                Ok(SearchType::Vector),
                Ok(SearchType::Keyword)
            ]
        );
    }

    #[test]
    fn unknown_search_type_names_the_input() {
        for bad in ["bm25", "Hybrid", " keyword", ""] {
            let err = bad.parse::<SearchType>().err();
            assert_eq!(err, Some(UnknownSearchType(bad.to_owned())));
            let message = err.map(|e| e.to_string()).unwrap_or_default();
            assert!(message.contains(&format!("{bad:?}")), "{message}");
        }
    }

    #[test]
    fn new_request_uses_the_default_limit() {
        let request = SearchRequest::new("q", SearchType::Keyword, SearchFilters::default());
        assert_eq!(request.limit, DEFAULT_SEARCH_LIMIT);
        const { assert!(DEFAULT_SEARCH_LIMIT <= MAX_SEARCH_LIMIT) };
    }

    fn hit(paper_id: &str, start_line: u32, score: f32) -> SearchHit {
        SearchHit {
            paper_id: paper_id.to_owned(),
            revision: None,
            title: String::new(),
            authors: Vec::new(),
            section_path: Vec::new(),
            chunk_kind: ChunkKind::Body,
            start_line,
            end_line: start_line + 1,
            text: String::new(),
            score,
            url: String::new(),
        }
    }

    #[test]
    fn hits_order_by_score_then_paper_then_line() {
        let mut hits = [
            hit("P2R0", 5, 0.5),
            hit("P1R0", 9, 0.5),
            hit("P1R0", 2, 0.5),
            hit("P3R0", 0, 0.9),
            hit("P0R0", 0, 0.1),
        ];
        hits.sort_by(hit_order);
        let order: Vec<(&str, u32)> = hits
            .iter()
            .map(|h| (h.paper_id.as_str(), h.start_line))
            .collect();
        assert_eq!(
            order,
            [
                ("P3R0", 0),
                ("P1R0", 2),
                ("P1R0", 9),
                ("P2R0", 5),
                ("P0R0", 0)
            ]
        );
    }

    fn candidate(chunk_id: &str, score: f32) -> Candidate {
        Candidate {
            chunk_id: chunk_id.to_owned(),
            hit: hit(chunk_id, 0, score),
        }
    }

    #[test]
    fn fusion_sums_reciprocal_ranks_across_lists() {
        let dense = vec![candidate("A", 0.9), candidate("B", 0.8)];
        let keyword = vec![candidate("B", 7.0), candidate("C", 3.0)];
        let mut fused = reciprocal_rank_fusion(dense, keyword);
        fused.sort_by(candidate_order);
        let scores: Vec<(&str, f32)> = fused
            .iter()
            .map(|c| (c.chunk_id.as_str(), c.hit.score))
            .collect();
        assert_eq!(
            scores,
            [
                ("B", 1.0 / (1.0 + RRF_K) + 1.0 / RRF_K),
                ("A", 1.0 / RRF_K),
                ("C", 1.0 / (1.0 + RRF_K))
            ]
        );
    }

    #[test]
    fn split_authors_inverts_the_join() {
        assert_eq!(split_authors(""), Vec::<String>::new());
        assert_eq!(split_authors("Ada Lovelace"), ["Ada Lovelace"]);
        assert_eq!(
            split_authors(&["A, B", "C"].join(AUTHORS_SEPARATOR)),
            ["A, B", "C"]
        );
    }
}
