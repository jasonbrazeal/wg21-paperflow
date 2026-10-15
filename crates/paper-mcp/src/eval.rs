// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Retrieval evaluation against a hand-labeled golden query set.
//!
//! A [`GoldenSet`] is parsed from JSON Lines, one [`GoldenQuery`] per line,
//! each labeled with the paper revisions and line ranges that answer it.
//! [`evaluate`] runs every query through [`search_chunks`] in each
//! [`EvalConfig`] and returns an [`EvalReport`] with recall@k and search
//! latency per configuration. [`Throughput`] turns an [`IndexReport`] and
//! the run's wall time into indexing throughput. The functions return data;
//! the `paper-mcp-eval` binary prints the report.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use paperstore::{PaperNum, StorageBackend};
use serde::Deserialize;

use crate::error::{Error, Result};
use crate::filter::SearchFilter;
use crate::ingest::{IndexReport, Outcome};
use crate::model::{Embedder, Reranker};
use crate::search::{SearchHit, SearchMode, SearchParams, SearchRequest, search_chunks};
use crate::selection::resolve_latest;
use crate::store::IndexStore;

/// Fewest queries a golden set may hold.
pub const MIN_GOLDEN_QUERIES: usize = 50;

/// Result-list depth at which recall is reported by default.
pub const DEFAULT_RECALL_K: usize = 20;

/// Percentile of per-query latency the report gives.
pub const LATENCY_PERCENTILE: f64 = 0.95;

/// What kind of question a golden query asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    /// A natural-language question answered by explanatory prose.
    Prose,
    /// An exact C++ identifier or signature.
    Identifier,
    /// A phrase quoted from proposed standard wording.
    WordingDiff,
    /// A question about one revision of a paper.
    Revision,
}

impl Category {
    /// Every category, in report order.
    pub const ALL: [Self; 4] = [
        Self::Prose,
        Self::Identifier,
        Self::WordingDiff,
        Self::Revision,
    ];

    /// Return the name used in the golden set and the report.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Prose => "prose",
            Self::Identifier => "identifier",
            Self::WordingDiff => "wording_diff",
            Self::Revision => "revision",
        }
    }
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A line range of one paper revision that answers a golden query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabeledRange {
    /// Paper id with revision.
    pub paper_id: PaperNum,
    /// First line, 1-based and inclusive.
    pub start: usize,
    /// Line after the range, 1-based and exclusive.
    pub end: usize,
}

impl LabeledRange {
    /// Return whether a search hit lies in the same paper revision and
    /// shares at least one line with this range.
    #[must_use]
    pub fn is_hit_by(&self, hit: &SearchHit) -> bool {
        hit.paper_id == self.paper_id && hit.start < self.end && self.start < hit.end
    }
}

/// One labeled query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldenQuery {
    /// Unique identifier within the set.
    pub id: String,
    /// What kind of question it asks.
    pub category: Category,
    /// Query text as a client would send it.
    pub query: String,
    /// Search every revision instead of only the latest of each paper.
    pub all_revisions: bool,
    /// Paper id filter, with or without a revision.
    pub paper_id: Option<PaperNum>,
    /// Ranges that answer the query. Never empty.
    pub relevant: Vec<LabeledRange>,
}

impl GoldenQuery {
    /// Return the search request that runs this query in `mode`.
    #[must_use]
    pub fn request(&self, mode: SearchMode) -> SearchRequest {
        SearchRequest {
            query: self.query.clone(),
            mode,
            filter: SearchFilter {
                paper_id: self.paper_id.clone(),
                ..SearchFilter::default()
            },
            all_revisions: self.all_revisions,
            top_k: None,
        }
    }

    /// Return whether the search a request for this query runs is limited
    /// to the latest revision of each paper.
    fn latest_only(&self) -> bool {
        !self.all_revisions
            && self
                .paper_id
                .as_ref()
                .is_none_or(|id| id.split_revision().1.is_none())
    }
}

/// A validated golden query set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldenSet {
    /// Queries in file order.
    pub queries: Vec<GoldenQuery>,
}

/// One line of the golden set file, before validation.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawQuery {
    id: String,
    category: Category,
    query: String,
    #[serde(default)]
    all_revisions: bool,
    #[serde(default)]
    paper_id: Option<String>,
    relevant: Vec<RawRange>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRange {
    paper_id: String,
    start: usize,
    end: usize,
}

impl GoldenSet {
    /// Parse and validate a golden set from JSON Lines.
    ///
    /// Blank lines are skipped. Every query needs a unique non-empty id,
    /// non-blank query text and at least one relevant range; every range
    /// names a paper id and has `1 <= start < end`. The set must hold at
    /// least [`MIN_GOLDEN_QUERIES`] queries and every [`Category`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Golden`] naming the first problem and, for a
    /// problem with one query, its 1-based line number.
    pub fn parse(text: &str) -> Result<Self> {
        let mut queries = Vec::new();
        let mut ids = BTreeSet::new();
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let at = |reason: String| Error::Golden(format!("line {}: {reason}", index + 1));
            let raw: RawQuery = serde_json::from_str(line).map_err(|e| at(e.to_string()))?;
            let query = validate(raw).map_err(at)?;
            if !ids.insert(query.id.clone()) {
                return Err(at(format!("duplicate id {:?}", query.id)));
            }
            queries.push(query);
        }
        if queries.len() < MIN_GOLDEN_QUERIES {
            return Err(Error::Golden(format!(
                "{} queries; at least {MIN_GOLDEN_QUERIES} are required",
                queries.len()
            )));
        }
        let missing: Vec<&str> = Category::ALL
            .into_iter()
            .filter(|c| !queries.iter().any(|q| q.category == *c))
            .map(Category::name)
            .collect();
        if !missing.is_empty() {
            return Err(Error::Golden(format!(
                "no queries in categories: {}",
                missing.join(", ")
            )));
        }
        Ok(Self { queries })
    }

    /// Check every labeled range against the papers in `store`.
    ///
    /// Returns one message per problem, empty when the set matches the
    /// corpus: a labeled paper that is absent or has no Markdown, a range
    /// that ends past the paper's last line, or, for a query searching only
    /// latest revisions, a labeled revision that is not the latest.
    #[must_use]
    pub fn check_corpus(&self, store: &dyn StorageBackend) -> Vec<String> {
        let mut problems = Vec::new();
        for query in &self.queries {
            for range in &query.relevant {
                let id = &range.paper_id;
                match store.paper_md(id) {
                    Ok(md) => {
                        let limit = md.lines().count() + 1;
                        if range.end > limit {
                            problems.push(format!(
                                "{}: {id} lines {}..{} end past line {}",
                                query.id,
                                range.start,
                                range.end,
                                limit - 1
                            ));
                        }
                    }
                    Err(e) => problems.push(format!("{}: {id}: {e}", query.id)),
                }
                if query.latest_only() {
                    let number = PaperNum::parse(id.split_revision().0);
                    match number.map_err(Error::from).and_then(|n| resolve_latest(store, &n)) {
                        Ok(latest) if &latest == id => {}
                        Ok(latest) => problems.push(format!(
                            "{}: {id} is not the latest revision ({latest}) but the query searches only latest revisions",
                            query.id
                        )),
                        Err(e) => problems.push(format!("{}: {id}: {e}", query.id)),
                    }
                }
            }
        }
        problems
    }
}

/// Validate one raw query, returning the reason it is invalid.
fn validate(raw: RawQuery) -> std::result::Result<GoldenQuery, String> {
    if raw.id.trim().is_empty() {
        return Err("empty id".to_owned());
    }
    if raw.query.trim().is_empty() {
        return Err(format!("{}: empty query", raw.id));
    }
    if raw.relevant.is_empty() {
        return Err(format!("{}: no relevant ranges", raw.id));
    }
    let paper_id = raw
        .paper_id
        .as_deref()
        .map(PaperNum::parse)
        .transpose()
        .map_err(|e| format!("{}: {e}", raw.id))?;
    let relevant = raw
        .relevant
        .into_iter()
        .map(|r| {
            if r.start == 0 || r.start >= r.end {
                return Err(format!(
                    "{}: range {}..{} of {} must have 1 <= start < end",
                    raw.id, r.start, r.end, r.paper_id
                ));
            }
            let paper_id = PaperNum::parse(&r.paper_id).map_err(|e| format!("{}: {e}", raw.id))?;
            Ok(LabeledRange {
                paper_id,
                start: r.start,
                end: r.end,
            })
        })
        .collect::<std::result::Result<_, _>>()?;
    Ok(GoldenQuery {
        id: raw.id,
        category: raw.category,
        query: raw.query,
        all_revisions: raw.all_revisions,
        paper_id,
        relevant,
    })
}

/// Return the fraction of `relevant` ranges hit by at least one of the
/// first `k` hits.
///
/// A hit counts for a range when it is in the same paper revision and
/// shares a line with it. Returns `0.0` when `relevant` is empty.
#[must_use]
pub fn recall_at_k(hits: &[SearchHit], relevant: &[LabeledRange], k: usize) -> f64 {
    let head = &hits[..k.min(hits.len())];
    let found = relevant
        .iter()
        .filter(|range| head.iter().any(|hit| range.is_hit_by(hit)))
        .count();
    ratio(found, relevant.len())
}

/// Return `num / den` as a float, `0.0` when `den` is zero.
#[expect(
    clippy::cast_precision_loss,
    reason = "counts of queries and ranges are far below 2^52"
)]
fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

/// Return the nearest-rank percentile of `samples`, `None` when empty.
///
/// `p` is a fraction in `(0, 1]`; values outside are clamped.
#[must_use]
pub fn percentile(samples: &[Duration], p: f64) -> Option<Duration> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = nearest_rank(sorted.len(), p);
    sorted.get(rank - 1).copied()
}

/// Return the 1-based nearest rank of fraction `p` in `n` samples.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the rank is clamped to 1..=n, and n is far below 2^52"
)]
fn nearest_rank(n: usize, p: f64) -> usize {
    let rank = (p.clamp(0.0, 1.0) * n as f64).ceil() as usize;
    rank.clamp(1, n)
}

/// One retrieval configuration the harness compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalConfig {
    /// Vector search alone, no reranking.
    Dense,
    /// Keyword (BM25) search alone, no reranking.
    Bm25,
    /// Vector and keyword fused with RRF, no reranking.
    Hybrid,
    /// Hybrid search with the head reranked, as `search_papers` serves it.
    HybridRerank,
}

impl EvalConfig {
    /// Every configuration, in report order.
    pub const ALL: [Self; 4] = [Self::Dense, Self::Bm25, Self::Hybrid, Self::HybridRerank];

    /// Return the name used in the report.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Bm25 => "bm25",
            Self::Hybrid => "hybrid",
            Self::HybridRerank => "hybrid+rerank",
        }
    }

    /// Return the search mode the configuration runs.
    #[must_use]
    pub fn mode(self) -> SearchMode {
        match self {
            Self::Dense => SearchMode::Vector,
            Self::Bm25 => SearchMode::Keyword,
            Self::Hybrid | Self::HybridRerank => SearchMode::Hybrid,
        }
    }

    /// Return `base` with reranking disabled unless the configuration reranks.
    #[must_use]
    pub fn params(self, base: SearchParams) -> SearchParams {
        match self {
            Self::HybridRerank => base,
            Self::Dense | Self::Bm25 | Self::Hybrid => SearchParams {
                rerank_depth: 0,
                ..base
            },
        }
    }
}

impl fmt::Display for EvalConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The outcome of one query in one configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryResult {
    /// The golden query's id.
    pub id: String,
    /// The golden query's category.
    pub category: Category,
    /// Recall at the report's `k`.
    pub recall: f64,
    /// Wall time of the search.
    pub latency: Duration,
}

/// Every query's outcome in one configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfigResult {
    /// The configuration.
    pub config: EvalConfig,
    /// One result per golden query, in set order.
    pub queries: Vec<QueryResult>,
}

impl ConfigResult {
    /// Return the mean recall over every query.
    #[must_use]
    pub fn recall(&self) -> f64 {
        mean_recall(self.queries.iter())
    }

    /// Return the mean recall over the queries of `category`, `None` when
    /// it has none.
    #[must_use]
    pub fn category_recall(&self, category: Category) -> Option<f64> {
        let mut matching = self
            .queries
            .iter()
            .filter(|q| q.category == category)
            .peekable();
        matching.peek()?;
        Some(mean_recall(matching))
    }

    /// Return the [`LATENCY_PERCENTILE`] search latency, `None` without queries.
    #[must_use]
    pub fn latency_percentile(&self) -> Option<Duration> {
        let latencies: Vec<Duration> = self.queries.iter().map(|q| q.latency).collect();
        percentile(&latencies, LATENCY_PERCENTILE)
    }
}

fn mean_recall<'a>(queries: impl Iterator<Item = &'a QueryResult>) -> f64 {
    let (sum, count) = queries.fold((0.0, 0), |(sum, count), q| (sum + q.recall, count + 1));
    if count == 0 {
        0.0
    } else {
        sum / ratio(count, 1)
    }
}

/// Indexing throughput of one index run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Throughput {
    /// Papers whose chunks were written.
    pub papers: usize,
    /// Chunks written.
    pub chunks: usize,
    /// Wall time of the run.
    pub elapsed: Duration,
}

impl Throughput {
    /// Count the papers and chunks `report` records as indexed.
    ///
    /// Skipped, metadata-only and failed papers count toward neither.
    #[must_use]
    pub fn from_report(report: &IndexReport, elapsed: Duration) -> Self {
        let (papers, chunks) = report
            .papers
            .iter()
            .filter_map(|p| match p.outcome {
                Outcome::Indexed { chunks } => Some(chunks),
                _ => None,
            })
            .fold((0, 0), |(papers, total), chunks| {
                (papers + 1, total + chunks)
            });
        Self {
            papers,
            chunks,
            elapsed,
        }
    }

    /// Return chunks written per second, `0.0` for a zero elapsed time.
    #[must_use]
    pub fn chunks_per_sec(&self) -> f64 {
        per_sec(self.chunks, self.elapsed)
    }

    /// Return papers indexed per second, `0.0` for a zero elapsed time.
    #[must_use]
    pub fn papers_per_sec(&self) -> f64 {
        per_sec(self.papers, self.elapsed)
    }
}

fn per_sec(count: usize, elapsed: Duration) -> f64 {
    let secs = elapsed.as_secs_f64();
    if secs > 0.0 {
        ratio(count, 1) / secs
    } else {
        0.0
    }
}

/// The full evaluation report.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalReport {
    /// Result-list depth recall is measured at.
    pub k: usize,
    /// One entry per configuration, in [`EvalConfig::ALL`] order.
    pub configs: Vec<ConfigResult>,
    /// Throughput of the index run that built the index, when the harness ran one.
    pub throughput: Option<Throughput>,
}

impl EvalReport {
    /// Return the result of `config`, if it was evaluated.
    #[must_use]
    pub fn config(&self, config: EvalConfig) -> Option<&ConfigResult> {
        self.configs.iter().find(|c| c.config == config)
    }

    /// Return the [`LATENCY_PERCENTILE`] latency of the configuration
    /// `search_papers` serves, hybrid plus rerank.
    #[must_use]
    pub fn search_papers_latency(&self) -> Option<Duration> {
        self.config(EvalConfig::HybridRerank)
            .and_then(ConfigResult::latency_percentile)
    }

    /// Return the report as text lines: one line per configuration with
    /// overall and per-category recall and latency, then throughput and
    /// `search_papers` latency.
    #[must_use]
    pub fn summary_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        let queries = self.configs.first().map_or(0, |c| c.queries.len());
        lines.push(format!("{queries} queries, recall@{}", self.k));
        for result in &self.configs {
            let categories: Vec<String> = Category::ALL
                .into_iter()
                .filter_map(|c| result.category_recall(c).map(|r| format!("{c} {r:.3}")))
                .collect();
            lines.push(format!(
                "{:<14} recall {:.3} ({}); p95 {}",
                result.config.name(),
                result.recall(),
                categories.join(", "),
                fmt_latency(result.latency_percentile()),
            ));
        }
        lines.push(match &self.throughput {
            Some(t) => format!(
                "indexing: {} papers, {} chunks in {:.1}s ({:.2} papers/s, {:.1} chunks/s)",
                t.papers,
                t.chunks,
                t.elapsed.as_secs_f64(),
                t.papers_per_sec(),
                t.chunks_per_sec()
            ),
            None => "indexing: not measured (no index run)".to_owned(),
        });
        lines.push(format!(
            "search_papers p95 latency: {}",
            fmt_latency(self.search_papers_latency())
        ));
        lines
    }
}

fn fmt_latency(latency: Option<Duration>) -> String {
    latency.map_or_else(
        || "n/a".to_owned(),
        |d| format!("{:.1} ms", d.as_secs_f64() * 1000.0),
    )
}

/// Run every golden query in every [`EvalConfig`] and measure recall@`k`.
///
/// Each query searches with [`GoldenQuery::request`] and `params`, with
/// reranking disabled in the configurations that do not rerank. Queries
/// run one at a time, so each latency is one search's wall time. The
/// report carries no throughput; the caller sets it after an index run.
///
/// # Errors
///
/// Returns the first error a search returns.
///
/// # Panics
///
/// Panics when called outside a tokio runtime.
pub async fn evaluate<S, E, R>(
    store: &S,
    embedder: &Arc<E>,
    reranker: &Arc<R>,
    params: SearchParams,
    golden: &GoldenSet,
    k: usize,
) -> Result<EvalReport>
where
    S: IndexStore + ?Sized,
    E: Embedder + ?Sized + 'static,
    R: Reranker + ?Sized + 'static,
{
    let mut configs = Vec::with_capacity(EvalConfig::ALL.len());
    for config in EvalConfig::ALL {
        let config_params = config.params(params);
        let mut queries = Vec::with_capacity(golden.queries.len());
        for query in &golden.queries {
            let req = query.request(config.mode());
            let started = Instant::now();
            let hits = search_chunks(store, embedder, reranker, &config_params, &req).await?;
            let latency = started.elapsed();
            queries.push(QueryResult {
                id: query.id.clone(),
                category: query.category,
                recall: recall_at_k(&hits, &query.relevant, k),
                latency,
            });
        }
        configs.push(ConfigResult { config, queries });
    }
    Ok(EvalReport {
        k,
        configs,
        throughput: None,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use paperstore::PaperRow;

    use super::*;
    use crate::ingest::PaperReport;

    /// An in-memory paperstore: each paper maps to its Markdown, `None`
    /// for a paper stored without Markdown.
    struct MemStore(BTreeMap<PaperNum, Option<String>>);

    impl MemStore {
        fn new(papers: &[(&str, Option<usize>)]) -> Self {
            Self(
                papers
                    .iter()
                    .map(|(id, lines)| (num(id), lines.map(markdown)))
                    .collect(),
            )
        }

        fn row(n: &PaperNum) -> PaperRow {
            PaperRow {
                paper_num: n.as_str().to_owned(),
                ..PaperRow::default()
            }
        }
    }

    /// Markdown of exactly `lines` lines, with a trailing newline.
    fn markdown(lines: usize) -> String {
        "line\n".repeat(lines)
    }

    impl StorageBackend for MemStore {
        fn meta(&self, n: &PaperNum) -> paperstore::Result<PaperRow> {
            self.0
                .get(n)
                .map(|_| Self::row(n))
                .ok_or_else(|| paperstore::Error::missing_paper(n.clone()))
        }

        fn paper_md(&self, n: &PaperNum) -> paperstore::Result<String> {
            match self.0.get(n) {
                Some(Some(md)) => Ok(md.clone()),
                Some(None) => Err(paperstore::Error::missing_markdown(n.clone())),
                None => Err(paperstore::Error::missing_paper(n.clone())),
            }
        }

        fn list_all_paper_ids(&self) -> paperstore::Result<Vec<PaperNum>> {
            Ok(self.0.keys().cloned().collect())
        }

        fn list_paper_ids_for_year(&self, _year: u16) -> paperstore::Result<Vec<PaperNum>> {
            Ok(Vec::new())
        }

        fn list_revisions(&self, n: &PaperNum) -> paperstore::Result<Vec<PaperRow>> {
            let base = n.split_revision().0;
            Ok(self
                .0
                .keys()
                .filter(|k| k.split_revision().0 == base)
                .map(Self::row)
                .collect())
        }

        fn tool_dir(&self, tool: &str) -> PathBuf {
            PathBuf::from(tool)
        }
    }

    fn corpus() -> MemStore {
        MemStore::new(&[
            ("P1R0", Some(10)),
            ("P1R2", Some(10)),
            ("P1R10", Some(10)),
            ("P2R0", None),
        ])
    }

    fn query(
        all_revisions: bool,
        paper_id: Option<&str>,
        relevant: Vec<LabeledRange>,
    ) -> GoldenQuery {
        GoldenQuery {
            id: "q".to_owned(),
            category: Category::Prose,
            query: "q".to_owned(),
            all_revisions,
            paper_id: paper_id.map(num),
            relevant,
        }
    }

    fn problems(queries: Vec<GoldenQuery>) -> Vec<String> {
        GoldenSet { queries }.check_corpus(&corpus())
    }

    #[test]
    fn check_corpus_accepts_ranges_up_to_the_last_line() {
        let relevant = vec![range("P1R10", 1, 11), range("P1R10", 10, 11)];
        assert_eq!(
            problems(vec![query(false, None, relevant)]),
            Vec::<String>::new()
        );
    }

    #[test]
    fn check_corpus_reports_a_range_past_the_last_line() {
        let found = problems(vec![query(false, None, vec![range("P1R10", 5, 12)])]);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("end past line 10"), "{found:?}");
    }

    #[test]
    fn check_corpus_reports_missing_papers_and_markdown() {
        for paper in ["P2R0", "P9R0"] {
            let found = problems(vec![query(true, None, vec![range(paper, 1, 2)])]);
            assert_eq!(found.len(), 1, "{paper}: {found:?}");
            assert!(found[0].contains(paper), "{found:?}");
        }
    }

    #[test]
    fn check_corpus_requires_the_latest_revision_for_latest_only_queries() {
        for paper_id in [None, Some("P1")] {
            let found = problems(vec![query(false, paper_id, vec![range("P1R2", 1, 2)])]);
            assert_eq!(found.len(), 1, "{paper_id:?}: {found:?}");
            assert!(
                found[0].contains("not the latest revision (P1R10)"),
                "{found:?}"
            );
            let latest = problems(vec![query(false, paper_id, vec![range("P1R10", 1, 2)])]);
            assert_eq!(latest, Vec::<String>::new(), "{paper_id:?}");
        }
    }

    #[test]
    fn check_corpus_allows_older_revisions_outside_latest_only_queries() {
        let older = || vec![range("P1R2", 1, 2), range("P1R0", 1, 2)];
        assert_eq!(
            problems(vec![query(true, None, older())]),
            Vec::<String>::new()
        );
        assert_eq!(
            problems(vec![query(true, Some("P1"), older())]),
            Vec::<String>::new()
        );
        assert_eq!(
            problems(vec![query(false, Some("P1R2"), older())]),
            Vec::<String>::new()
        );
    }

    fn result(id: &str, category: Category, recall: f64) -> QueryResult {
        QueryResult {
            id: id.to_owned(),
            category,
            recall,
            latency: Duration::ZERO,
        }
    }

    #[test]
    fn config_recall_averages_over_all_and_per_category() {
        let config = ConfigResult {
            config: EvalConfig::Hybrid,
            queries: vec![
                result("a", Category::Prose, 1.0),
                result("b", Category::Prose, 0.5),
                result("c", Category::Identifier, 0.0),
                result("d", Category::Revision, 0.25),
            ],
        };
        assert!((config.recall() - 0.4375).abs() < 1e-9);
        let prose = config.category_recall(Category::Prose);
        assert!(prose.is_some_and(|r| (r - 0.75).abs() < 1e-9), "{prose:?}");
        let identifier = config.category_recall(Category::Identifier);
        assert!(identifier.is_some_and(|r| r.abs() < 1e-9), "{identifier:?}");
        assert_eq!(config.category_recall(Category::WordingDiff), None);
    }

    #[test]
    fn config_recall_without_queries_is_zero() {
        let config = ConfigResult {
            config: EvalConfig::Dense,
            queries: Vec::new(),
        };
        assert!(config.recall().abs() < f64::EPSILON);
        assert_eq!(config.category_recall(Category::Prose), None);
        assert_eq!(config.latency_percentile(), None);
    }

    /// The golden set shipped with the crate.
    const SHIPPED: &str = include_str!("../eval/golden.jsonl");

    fn num(raw: &str) -> PaperNum {
        PaperNum::parse(raw).unwrap_or_else(|e| panic!("{raw}: {e}"))
    }

    fn range(paper: &str, start: usize, end: usize) -> LabeledRange {
        LabeledRange {
            paper_id: num(paper),
            start,
            end,
        }
    }

    fn hit(paper: &str, start: usize, end: usize) -> SearchHit {
        SearchHit {
            paper_id: num(paper),
            title: String::new(),
            heading_path: Vec::new(),
            start,
            end,
            text: String::new(),
            score: 0.0,
        }
    }

    fn line(id: &str, category: &str) -> String {
        format!(
            r#"{{"id":"{id}","category":"{category}","query":"q {id}","relevant":[{{"paper_id":"P2300R10","start":10,"end":20}}]}}"#
        )
    }

    /// A valid set of `n` queries cycling through every category.
    fn valid_lines(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| line(&format!("q{i}"), Category::ALL[i % 4].name()))
            .collect()
    }

    fn parse_lines(lines: &[String]) -> Result<GoldenSet> {
        GoldenSet::parse(&lines.join("\n"))
    }

    fn golden_error(lines: &[String]) -> String {
        match parse_lines(lines) {
            Err(Error::Golden(reason)) => reason,
            other => panic!("expected a golden-set error, got {other:?}"),
        }
    }

    #[test]
    fn shipped_set_is_valid_and_large_enough() -> Result<()> {
        let set = GoldenSet::parse(SHIPPED)?;
        assert!(set.queries.len() >= MIN_GOLDEN_QUERIES);
        for category in Category::ALL {
            assert!(set.queries.iter().any(|q| q.category == category));
        }
        Ok(())
    }

    #[test]
    fn parses_fields_and_defaults() -> Result<()> {
        let mut lines = valid_lines(MIN_GOLDEN_QUERIES);
        lines[0] = r#"{"id":"r1","category":"revision","query":"what changed","all_revisions":true,"paper_id":"p2900","relevant":[{"paper_id":"p2900r7","start":29,"end":45},{"paper_id":"N4950","start":1,"end":2}]}"#.to_owned();
        let set = parse_lines(&lines)?;
        let first = &set.queries[0];
        assert_eq!(first.category, Category::Revision);
        assert!(first.all_revisions);
        assert_eq!(first.paper_id, Some(num("P2900")));
        assert_eq!(
            first.relevant,
            vec![range("P2900R7", 29, 45), range("N4950", 1, 2)]
        );
        let second = &set.queries[1];
        assert!(!second.all_revisions);
        assert_eq!(second.paper_id, None);
        Ok(())
    }

    #[test]
    fn blank_lines_are_skipped() -> Result<()> {
        let mut lines = valid_lines(MIN_GOLDEN_QUERIES);
        lines.insert(3, "   ".to_owned());
        lines.push(String::new());
        assert_eq!(parse_lines(&lines)?.queries.len(), MIN_GOLDEN_QUERIES);
        Ok(())
    }

    #[test]
    fn too_few_queries_is_an_error() {
        let reason = golden_error(&valid_lines(MIN_GOLDEN_QUERIES - 1));
        assert!(reason.contains("at least"), "{reason}");
    }

    #[test]
    fn missing_category_is_an_error() {
        let lines: Vec<String> = (0..MIN_GOLDEN_QUERIES)
            .map(|i| line(&format!("q{i}"), Category::ALL[i % 3].name()))
            .collect();
        let reason = golden_error(&lines);
        assert!(reason.contains("revision"), "{reason}");
    }

    #[test]
    fn malformed_ranges_are_errors() {
        for (start, end) in [(0, 5), (5, 5), (9, 5)] {
            let mut lines = valid_lines(MIN_GOLDEN_QUERIES);
            lines[7] = format!(
                r#"{{"id":"bad","category":"prose","query":"q","relevant":[{{"paper_id":"P1R0","start":{start},"end":{end}}}]}}"#
            );
            let reason = golden_error(&lines);
            assert!(reason.starts_with("line 8:"), "{start}..{end}: {reason}");
        }
    }

    #[test]
    fn invalid_queries_are_errors() {
        let cases = [
            r#"{"id":"","category":"prose","query":"q","relevant":[{"paper_id":"P1R0","start":1,"end":2}]}"#,
            r#"{"id":"x","category":"prose","query":"  ","relevant":[{"paper_id":"P1R0","start":1,"end":2}]}"#,
            r#"{"id":"x","category":"prose","query":"q","relevant":[]}"#,
            r#"{"id":"x","category":"prose","query":"q","relevant":[{"paper_id":"not a paper","start":1,"end":2}]}"#,
            r#"{"id":"x","category":"prose","query":"q","paper_id":"???","relevant":[{"paper_id":"P1R0","start":1,"end":2}]}"#,
            r#"{"id":"x","category":"poetry","query":"q","relevant":[{"paper_id":"P1R0","start":1,"end":2}]}"#,
            r#"{"id":"x","category":"prose","query":"q","extra":1,"relevant":[{"paper_id":"P1R0","start":1,"end":2}]}"#,
            r#"{"id":"x","category":"prose""#,
        ];
        for case in cases {
            let mut lines = valid_lines(MIN_GOLDEN_QUERIES);
            lines[2] = case.to_owned();
            let reason = golden_error(&lines);
            assert!(reason.starts_with("line 3:"), "{case}: {reason}");
        }
    }

    #[test]
    fn duplicate_ids_are_errors() {
        let mut lines = valid_lines(MIN_GOLDEN_QUERIES);
        lines[5] = line("q1", "prose");
        let reason = golden_error(&lines);
        assert!(reason.contains("duplicate"), "{reason}");
    }

    #[test]
    fn recall_counts_ranges_hit_by_overlap() {
        let relevant = [range("P1R0", 10, 20), range("P2R0", 1, 5)];
        let touching_start = [hit("P1R0", 5, 11)];
        assert!((recall_at_k(&touching_start, &relevant, 20) - 0.5).abs() < f64::EPSILON);
        let inside = [hit("P1R0", 12, 14), hit("P2R0", 4, 30)];
        assert!((recall_at_k(&inside, &relevant, 20) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_ignores_adjacent_ranges_and_other_revisions() {
        let relevant = [range("P1R1", 10, 20)];
        let misses = [
            hit("P1R1", 1, 10),
            hit("P1R1", 20, 30),
            hit("P1R0", 10, 20),
            hit("P9R1", 10, 20),
        ];
        assert!(recall_at_k(&misses, &relevant, 20).abs() < f64::EPSILON);
    }

    #[test]
    fn recall_only_looks_at_the_first_k_hits() {
        let relevant = [range("P1R0", 10, 20)];
        let mut hits: Vec<SearchHit> = (0..20).map(|i| hit("P9R0", i + 1, i + 2)).collect();
        hits.push(hit("P1R0", 10, 20));
        assert!(recall_at_k(&hits, &relevant, 20).abs() < f64::EPSILON);
        assert!((recall_at_k(&hits, &relevant, 21) - 1.0).abs() < f64::EPSILON);
        assert!((recall_at_k(&hits, &relevant, 100) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn one_hit_covers_several_ranges_and_duplicates_count_once() {
        let relevant = [
            range("P1R0", 1, 5),
            range("P1R0", 5, 9),
            range("P1R0", 50, 60),
        ];
        let hits = [hit("P1R0", 3, 7), hit("P1R0", 3, 7)];
        assert!((recall_at_k(&hits, &relevant, 20) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let samples: Vec<Duration> = (1..=20).rev().map(Duration::from_millis).collect();
        assert_eq!(percentile(&samples, 0.95), Some(Duration::from_millis(19)));
        assert_eq!(percentile(&samples, 1.0), Some(Duration::from_millis(20)));
        assert_eq!(
            percentile(&[Duration::from_millis(7)], 0.95),
            Some(Duration::from_millis(7))
        );
        assert_eq!(percentile(&[], 0.95), None);
    }

    #[test]
    fn only_hybrid_rerank_reranks() {
        let base = SearchParams::default();
        for config in EvalConfig::ALL {
            let reranks = config.params(base).rerank_depth > 0;
            assert_eq!(reranks, config == EvalConfig::HybridRerank, "{config}");
        }
        assert_eq!(EvalConfig::Dense.mode(), SearchMode::Vector);
        assert_eq!(EvalConfig::Bm25.mode(), SearchMode::Keyword);
    }

    #[test]
    fn throughput_counts_only_indexed_papers() {
        let report = IndexReport {
            papers: vec![
                PaperReport {
                    paper: num("P1R0"),
                    outcome: Outcome::Indexed { chunks: 30 },
                },
                PaperReport {
                    paper: num("P2R0"),
                    outcome: Outcome::Indexed { chunks: 10 },
                },
                PaperReport {
                    paper: num("P3R0"),
                    outcome: Outcome::Skipped,
                },
                PaperReport {
                    paper: num("P4R0"),
                    outcome: Outcome::Failed("x".to_owned()),
                },
            ],
            optimize_error: None,
        };
        let t = Throughput::from_report(&report, Duration::from_secs(4));
        assert_eq!((t.papers, t.chunks), (2, 40));
        assert!((t.chunks_per_sec() - 10.0).abs() < f64::EPSILON);
        assert!((t.papers_per_sec() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn request_applies_filter_and_revision_flags() {
        let query = GoldenQuery {
            id: "r".to_owned(),
            category: Category::Revision,
            query: "what changed".to_owned(),
            all_revisions: true,
            paper_id: Some(num("P2900")),
            relevant: vec![range("P2900R7", 1, 2)],
        };
        let req = query.request(SearchMode::Hybrid);
        assert_eq!(req.filter.paper_id, Some(num("P2900")));
        assert!(req.all_revisions);
        assert_eq!(req.mode, SearchMode::Hybrid);
        assert_eq!(req.top_k, None);
    }
}
