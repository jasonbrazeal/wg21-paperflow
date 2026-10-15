// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! The search pipeline over indexed chunks and paper-level records.
//!
//! [`search_chunks`] runs a vector, keyword or hybrid search, fuses the two
//! sides of a hybrid search with reciprocal rank fusion, and lets the
//! reranker reorder the head of the list. Nothing is dropped after
//! retrieval, so the caller can page to the end of the full ranked list.
//! [`search_metadata`] returns every paper-level record matching keywords
//! on title and authors. Both apply the same latest-revision rule.

use std::collections::HashMap;

use paperstore::PaperNum;

use crate::error::{Error, Result};
use crate::filter::SearchFilter;
use crate::model::{Embedder, Reranker};
use crate::store::{ChunkHit, IndexStore, MetaRecord};

/// Default number of nearest chunks a vector search retrieves.
pub const DEFAULT_TOP_K: usize = 1000;

/// Default number of leading results the reranker reorders.
pub const DEFAULT_RERANK_DEPTH: usize = 150;

/// Rank offset of reciprocal rank fusion: a hit at 1-based rank `r` on one
/// side contributes `1 / (RRF_K + r)`.
pub const RRF_K: f32 = 60.0;

/// Minimum distance between the lowest reranked score and the highest
/// score past the rerank depth, once tail scores are shifted below the head.
pub const TAIL_SCORE_GAP: f32 = 1.0;

/// Which retrieval sides a chunk search uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMode {
    /// The `top_k` nearest chunks by embedding.
    Vector,
    /// Every chunk matching the query by keyword.
    Keyword,
    /// Every keyword match plus the `top_k` nearest chunks, fused with RRF.
    Hybrid,
}

/// One search call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchRequest {
    /// Query text. Metadata searches match its keywords on title and authors.
    pub query: String,
    /// Retrieval sides for chunk searches; metadata searches ignore it.
    pub mode: SearchMode,
    /// Field filters. Its `latest_only` flag is ignored: the latest-revision
    /// rule comes from `all_revisions` and the paper id.
    pub filter: SearchFilter,
    /// Search every indexed revision instead of only the latest of each paper.
    pub all_revisions: bool,
    /// Per-call override of [`SearchParams::top_k`].
    pub top_k: Option<usize>,
}

/// Server-wide search limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchParams {
    /// Nearest chunks a vector search retrieves when the request sets none.
    pub top_k: usize,
    /// Leading results the reranker reorders.
    pub rerank_depth: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            top_k: DEFAULT_TOP_K,
            rerank_depth: DEFAULT_RERANK_DEPTH,
        }
    }
}

/// One chunk search result.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchHit {
    /// Paper id with revision.
    pub paper_id: PaperNum,
    /// Paper title.
    pub title: String,
    /// Headings enclosing the chunk, outermost first.
    pub heading_path: Vec<String>,
    /// First source line, 1-based and inclusive.
    pub start: usize,
    /// Source line after the chunk, 1-based and exclusive.
    pub end: usize,
    /// Chunk text without its prefix.
    pub text: String,
    /// Relevance; higher is better. Reranked hits carry the reranker score.
    /// Past the rerank depth, hits carry the score they were ranked by
    /// before reranking, shifted when needed so that it falls below every
    /// reranked score. Scores never increase down the result list and
    /// compare only within one search.
    pub score: f32,
}

impl From<ChunkHit> for SearchHit {
    fn from(hit: ChunkHit) -> Self {
        let chunk = hit.chunk;
        Self {
            paper_id: chunk.paper_id,
            title: chunk.fields.title,
            heading_path: chunk.heading_path,
            start: chunk.start,
            end: chunk.end,
            text: chunk.text,
            score: hit.score,
        }
    }
}

/// Return the filter a request searches with, latest-revision rule applied.
///
/// Only the latest revision of each paper matches unless the request sets
/// `all_revisions` or names a paper id with a revision.
#[must_use]
pub fn effective_filter(req: &SearchRequest) -> SearchFilter {
    let versioned = req
        .filter
        .paper_id
        .as_ref()
        .is_some_and(|id| id.split_revision().1.is_some());
    SearchFilter {
        latest_only: !req.all_revisions && !versioned,
        ..req.filter.clone()
    }
}

/// Run a chunk search and return the full ranked result list.
///
/// Keyword mode returns every match, vector mode the `top_k` nearest
/// chunks, and hybrid mode the deduplicated union of both, ordered by RRF.
/// The reranker then reorders the first `rerank_depth` hits; the rest keep
/// their order.
///
/// # Errors
///
/// Returns [`Error::Index`] when the store fails and [`Error::Inference`]
/// when a model fails or the reranker returns the wrong number of scores.
pub async fn search_chunks<S, E, R>(
    store: &S,
    embedder: &E,
    reranker: &R,
    params: &SearchParams,
    req: &SearchRequest,
) -> Result<Vec<SearchHit>>
where
    S: IndexStore + ?Sized,
    E: Embedder + ?Sized,
    R: Reranker + ?Sized,
{
    let filter = effective_filter(req);
    let top_k = req.top_k.unwrap_or(params.top_k);
    let hits = match req.mode {
        SearchMode::Keyword => store.search_keyword(&req.query, &filter).await?,
        SearchMode::Vector => {
            let vector = embedder.embed_query(&req.query)?;
            store.search_vector(&vector, &filter, top_k).await?
        }
        SearchMode::Hybrid => {
            let keyword = store.search_keyword(&req.query, &filter).await?;
            let vector = embedder.embed_query(&req.query)?;
            let nearest = store.search_vector(&vector, &filter, top_k).await?;
            fuse_rrf(keyword, nearest)
        }
    };
    let hits = rerank_head(reranker, &req.query, hits, params.rerank_depth)?;
    Ok(hits.into_iter().map(SearchHit::from).collect())
}

/// Return every paper-level record matching the request.
///
/// The query's keywords match title and authors; a query without keywords
/// matches every record that passes the filter. The latest-revision rule
/// of [`effective_filter`] applies, and the request's mode and `top_k` are
/// ignored.
///
/// # Errors
///
/// Returns [`Error::Index`] when the store fails.
pub async fn search_metadata<S>(store: &S, req: &SearchRequest) -> Result<Vec<MetaRecord>>
where
    S: IndexStore + ?Sized,
{
    let hits = store
        .search_metadata(&req.query, &effective_filter(req))
        .await?;
    Ok(hits.into_iter().map(|h| h.meta).collect())
}

/// Identity of a chunk across result lists.
type ChunkKey = (PaperNum, usize, usize);

fn chunk_key(hit: &ChunkHit) -> ChunkKey {
    (hit.chunk.paper_id.clone(), hit.chunk.start, hit.chunk.end)
}

/// Fuse two ranked lists with reciprocal rank fusion.
///
/// Each chunk appears once, scored by the sum of its RRF contributions
/// from the lists that hold it. The result is ordered by descending score,
/// ties by paper id and position.
pub(crate) fn fuse_rrf(first: Vec<ChunkHit>, second: Vec<ChunkHit>) -> Vec<ChunkHit> {
    let mut fused: Vec<ChunkHit> = Vec::with_capacity(first.len() + second.len());
    let mut index: HashMap<ChunkKey, usize> = HashMap::new();
    for list in [first, second] {
        // An f32 counts ranks exactly up to 2^24, beyond any result list.
        let mut rank = 0.0_f32;
        for hit in list {
            rank += 1.0;
            let contribution = 1.0 / (RRF_K + rank);
            if let Some(&i) = index.get(&chunk_key(&hit)) {
                fused[i].score += contribution;
            } else {
                index.insert(chunk_key(&hit), fused.len());
                fused.push(ChunkHit {
                    score: contribution,
                    ..hit
                });
            }
        }
    }
    fused.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.chunk.paper_id.cmp(&b.chunk.paper_id))
            .then_with(|| a.chunk.start.cmp(&b.chunk.start))
    });
    fused
}

/// Let the reranker reorder the first `depth` hits by their prefixed text.
fn rerank_head<R: Reranker + ?Sized>(
    reranker: &R,
    query: &str,
    hits: Vec<ChunkHit>,
    depth: usize,
) -> Result<Vec<ChunkHit>> {
    let depth = depth.min(hits.len());
    if depth == 0 {
        return Ok(hits);
    }
    let documents: Vec<String> = hits[..depth]
        .iter()
        .map(|h| h.chunk.prefixed_text.clone())
        .collect();
    let scores = reranker.rerank(query, &documents)?;
    if scores.len() != depth {
        return Err(Error::Inference(format!(
            "reranker returned {} scores for {depth} documents",
            scores.len()
        )));
    }
    Ok(splice_reranked(hits, &scores))
}

/// Reorder the first `scores.len()` hits by descending score and keep the rest.
///
/// Reordered hits take their new score; equal scores keep their prior order.
/// The remaining hits keep their order and their score gaps, but when the
/// first of them does not score below the lowest reordered hit, all of
/// them are shifted down to [`TAIL_SCORE_GAP`] below it, so scores never
/// increase down the list. The remaining hits must be in descending score
/// order.
pub(crate) fn splice_reranked(hits: Vec<ChunkHit>, scores: &[f32]) -> Vec<ChunkHit> {
    let depth = scores.len().min(hits.len());
    let mut hits = hits;
    let mut tail = hits.split_off(depth);
    let mut head: Vec<ChunkHit> = hits
        .into_iter()
        .zip(scores)
        .map(|(hit, &score)| ChunkHit { score, ..hit })
        .collect();
    head.sort_by(|a, b| b.score.total_cmp(&a.score));
    if let (Some(lowest), Some(first)) = (head.last(), tail.first()) {
        let shift = (lowest.score - TAIL_SCORE_GAP) - first.score;
        if shift < 0.0 {
            for hit in &mut tail {
                hit.score += shift;
            }
        }
    }
    head.extend(tail);
    head
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{ChunkRecord, PaperFields};

    fn hit(paper: &str, start: usize, score: f32) -> ChunkHit {
        let paper_id = PaperNum::parse(paper).unwrap_or_else(|e| panic!("{paper}: {e}"));
        let (number, revision) = paper_id.split_revision();
        ChunkHit {
            chunk: ChunkRecord {
                paper_number: number.to_owned(),
                revision,
                paper_id,
                is_latest: true,
                heading_path: vec!["Intro".to_owned()],
                start,
                end: start + 2,
                text: format!("{paper}:{start}"),
                prefixed_text: format!("prefix {paper}:{start}"),
                vector: Vec::new(),
                content_hash: String::new(),
                format_version: String::new(),
                fields: PaperFields::default(),
            },
            score,
        }
    }

    fn keys(hits: &[ChunkHit]) -> Vec<String> {
        hits.iter().map(|h| h.chunk.text.clone()).collect()
    }

    fn rrf(ranks: &[u16]) -> f32 {
        ranks.iter().map(|r| 1.0 / (RRF_K + f32::from(*r))).sum()
    }

    /// Scores each document by its position, reversing the input order.
    struct ReverseReranker;

    impl Reranker for ReverseReranker {
        fn id(&self) -> &'static str {
            "reverse"
        }

        fn rerank(&self, _query: &str, documents: &[String]) -> Result<Vec<f32>> {
            Ok((0_u16..)
                .zip(documents)
                .map(|(i, _)| f32::from(i))
                .collect())
        }
    }

    /// Returns one score too few.
    struct ShortReranker;

    impl Reranker for ShortReranker {
        fn id(&self) -> &'static str {
            "short"
        }

        fn rerank(&self, _query: &str, documents: &[String]) -> Result<Vec<f32>> {
            Ok(vec![0.0; documents.len().saturating_sub(1)])
        }
    }

    #[test]
    fn rrf_sums_contributions_and_deduplicates() {
        let keyword = vec![
            hit("P1R0", 1, 9.0),
            hit("P1R0", 5, 8.0),
            hit("P2R0", 1, 7.0),
        ];
        let vector = vec![hit("P2R0", 1, 0.9), hit("P3R0", 1, 0.8)];
        let fused = fuse_rrf(keyword, vector);
        assert_eq!(keys(&fused), vec!["P2R0:1", "P1R0:1", "P1R0:5", "P3R0:1"]);
        let scores: Vec<f32> = fused.iter().map(|h| h.score).collect();
        assert_eq!(scores, vec![rrf(&[3, 1]), rrf(&[1]), rrf(&[2]), rrf(&[2])]);
    }

    #[test]
    fn rrf_ties_break_by_paper_and_position() {
        let fused = fuse_rrf(vec![hit("P9R0", 1, 1.0)], vec![hit("P1R0", 3, 1.0)]);
        assert_eq!(keys(&fused), vec!["P1R0:3", "P9R0:1"]);
    }

    #[test]
    fn rrf_keeps_one_sided_lists_in_order() {
        let list = vec![
            hit("P1R0", 1, 3.0),
            hit("P1R0", 4, 2.0),
            hit("P1R0", 9, 1.0),
        ];
        assert_eq!(keys(&fuse_rrf(list.clone(), Vec::new())), keys(&list));
        assert_eq!(keys(&fuse_rrf(Vec::new(), list.clone())), keys(&list));
        assert!(fuse_rrf(Vec::new(), Vec::new()).is_empty());
    }

    #[test]
    fn rrf_treats_same_start_different_end_as_distinct() {
        let mut other = hit("P1R0", 1, 1.0);
        other.chunk.end += 1;
        other.chunk.text = "other".to_owned();
        assert_eq!(fuse_rrf(vec![hit("P1R0", 1, 1.0)], vec![other]).len(), 2);
    }

    fn five() -> Vec<ChunkHit> {
        (1_u16..=5)
            .map(|i| hit("P1R0", usize::from(i), 10.0 - f32::from(i)))
            .collect()
    }

    #[test]
    fn rerank_reorders_only_the_head() -> Result<()> {
        let out = rerank_head(&ReverseReranker, "q", five(), 3)?;
        assert_eq!(
            keys(&out),
            vec!["P1R0:3", "P1R0:2", "P1R0:1", "P1R0:4", "P1R0:5"]
        );
        let scores: Vec<f32> = out.iter().map(|h| h.score).collect();
        assert_eq!(scores, vec![2.0, 1.0, 0.0, -1.0, -2.0]);
        Ok(())
    }

    #[test]
    fn tail_below_head_keeps_its_scores() {
        let out = splice_reranked(five(), &[20.0, 30.0]);
        let scores: Vec<f32> = out.iter().map(|h| h.score).collect();
        assert_eq!(scores, vec![30.0, 20.0, 7.0, 6.0, 5.0]);
    }

    #[test]
    fn identity_rerank_scores_never_increase() {
        let hits = fuse_rrf(
            vec![hit("P1R0", 1, 9.0), hit("P1R0", 2, 8.0)],
            vec![hit("P1R0", 3, 0.9), hit("P1R0", 4, 0.8)],
        );
        let out = splice_reranked(hits, &[0.0, -1.0]);
        assert!(out.windows(2).all(|w| w[0].score >= w[1].score));
        assert!(out[2].score < out[1].score);
    }

    #[test]
    fn rerank_depth_beyond_length_reorders_everything() -> Result<()> {
        let out = rerank_head(&ReverseReranker, "q", five(), 150)?;
        assert_eq!(
            keys(&out),
            vec!["P1R0:5", "P1R0:4", "P1R0:3", "P1R0:2", "P1R0:1"]
        );
        Ok(())
    }

    #[test]
    fn zero_rerank_depth_keeps_the_list() -> Result<()> {
        let out = rerank_head(&ShortReranker, "q", five(), 0)?;
        assert_eq!(out, five());
        Ok(())
    }

    #[test]
    fn equal_rerank_scores_keep_prior_order() {
        let out = splice_reranked(five(), &[1.0, 1.0, 1.0, 2.0]);
        assert_eq!(
            keys(&out),
            vec!["P1R0:4", "P1R0:1", "P1R0:2", "P1R0:3", "P1R0:5"]
        );
    }

    #[test]
    fn wrong_score_count_is_an_error() {
        assert!(matches!(
            rerank_head(&ShortReranker, "q", five(), 3),
            Err(Error::Inference(_))
        ));
    }

    fn request(paper: Option<&str>, all_revisions: bool) -> SearchRequest {
        SearchRequest {
            query: "q".to_owned(),
            mode: SearchMode::Hybrid,
            filter: SearchFilter {
                paper_id: paper.and_then(|p| PaperNum::parse(p).ok()),
                ..SearchFilter::default()
            },
            all_revisions,
            top_k: None,
        }
    }

    #[test]
    fn latest_only_unless_all_revisions_or_versioned_id() {
        assert!(effective_filter(&request(None, false)).latest_only);
        assert!(effective_filter(&request(Some("P2300"), false)).latest_only);
        assert!(!effective_filter(&request(Some("P2300"), true)).latest_only);
        assert!(!effective_filter(&request(Some("P2300R4"), false)).latest_only);
        let mut req = request(None, true);
        req.filter.latest_only = true;
        assert!(!effective_filter(&req).latest_only);
    }

    #[test]
    fn defaults_match_functional_specification() {
        assert_eq!(SearchParams::default().top_k, 1000);
        assert_eq!(SearchParams::default().rerank_depth, 150);
    }
}
