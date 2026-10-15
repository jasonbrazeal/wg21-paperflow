// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! `index_papers` and `IndexLock` against a copy of the fixture workspace.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use paper_mcp::model::fake::{HashEmbedder, IdentityReranker};
use paper_mcp::{
    ChunkHit, Embedder, Error, IndexLock, IndexOptions, IndexReport, IndexStore, LanceStore,
    LanceStoreOptions, MetaHit, Outcome, PaperRecords, Reranker, SearchFilter, TOOL_NAME,
    content_hash, index_papers, parse_selectors, resolve,
};
use paperstore::{PaperNum, PaperRow, StorageBackend};
use paperstore_sqlite::SqliteBackend;
use tempfile::TempDir;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

/// Large fixture papers shared with the chunker tests, copied in by name.
const SHARED_MARKDOWN: [&str; 2] = ["p2169r4.md", "p2322r4.md"];

pub(crate) const ALL_PAPERS: [&str; 14] = [
    "N4950", "P2169R4", "P2300R0", "P2300R1", "P2300R10", "P2300R2", "P2300R3", "P2300R4",
    "P2300R5", "P2300R6", "P2300R7", "P2300R8", "P2300R9", "P2322R4",
];

/// A temporary paperstore workspace built from `tests/fixtures/workspace`.
pub(crate) struct Workspace {
    dir: TempDir,
    pub(crate) backend: SqliteBackend,
}

impl Workspace {
    pub(crate) fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let papers = dir.path().join("paperstore");
        std::fs::create_dir_all(&papers)?;
        let fixture = Path::new(FIXTURES).join("workspace");
        for entry in std::fs::read_dir(fixture.join("paperstore"))? {
            let entry = entry?;
            std::fs::copy(entry.path(), papers.join(entry.file_name()))?;
        }
        for name in SHARED_MARKDOWN {
            std::fs::copy(Path::new(FIXTURES).join(name), papers.join(name))?;
        }
        let backend = SqliteBackend::new(dir.path())?;
        let ws = dir
            .path()
            .to_str()
            .ok_or("temp dir is not UTF-8")?
            .replace('\'', "''");
        let sql = std::fs::read_to_string(fixture.join("papers.sql"))?.replace("{ws}", &ws);
        rusqlite::Connection::open(dir.path().join("paperstore.db"))?.execute_batch(&sql)?;
        Ok(Self { dir, backend })
    }

    pub(crate) fn path(&self) -> &Path {
        self.dir.path()
    }

    fn tool_dir(&self) -> PathBuf {
        self.backend.tool_dir(TOOL_NAME)
    }

    pub(crate) fn markdown(&self, paper: &str) -> PathBuf {
        self.dir
            .path()
            .join("paperstore")
            .join(format!("{}.md", paper.to_ascii_lowercase()))
    }

    pub(crate) async fn store(&self) -> paper_mcp::Result<LanceStore> {
        let options = LanceStoreOptions {
            read_consistency_interval: Duration::ZERO,
            ..LanceStoreOptions::default()
        };
        LanceStore::open_with(&self.tool_dir(), HashEmbedder::default().dim(), options).await
    }

    /// Open the existing index read-only, as `serve` does, checking for new
    /// index runs once `read_consistency_interval` has passed since the
    /// last check.
    pub(crate) async fn read_only_store(
        &self,
        read_consistency_interval: Duration,
    ) -> paper_mcp::Result<LanceStore> {
        let options = LanceStoreOptions {
            read_consistency_interval,
            ..LanceStoreOptions::default()
        };
        LanceStore::open_read_only(&self.tool_dir(), HashEmbedder::default().dim(), options).await
    }
}

pub(crate) fn num(raw: &str) -> PaperNum {
    PaperNum::parse(raw).unwrap_or_else(|e| panic!("parse {raw}: {e}"))
}

pub(crate) fn nums(raws: &[&str]) -> Vec<PaperNum> {
    raws.iter().map(|r| num(r)).collect()
}

pub(crate) fn options() -> IndexOptions {
    IndexOptions::new(&HashEmbedder::default(), IdentityReranker.id())
}

fn forced() -> IndexOptions {
    IndexOptions {
        force: true,
        ..options()
    }
}

/// Index `papers`, turning a failed optimization into an error.
pub(crate) async fn run(
    backend: &dyn StorageBackend,
    store: &LanceStore,
    papers: &[PaperNum],
    options: &IndexOptions,
) -> paper_mcp::Result<IndexReport> {
    let report = index_papers(backend, store, &HashEmbedder::default(), papers, options).await;
    match &report.optimize_error {
        Some(reason) => Err(Error::Index(reason.clone())),
        None => Ok(report),
    }
}

fn ids(report: &IndexReport) -> Vec<PaperNum> {
    report.papers.iter().map(|p| p.paper.clone()).collect()
}

fn skipped(report: &IndexReport) -> Vec<String> {
    report
        .papers
        .iter()
        .filter(|p| p.outcome == Outcome::Skipped)
        .map(|p| p.paper.to_string())
        .collect()
}

fn not_skipped(report: &IndexReport) -> Vec<String> {
    report
        .papers
        .iter()
        .filter(|p| p.outcome != Outcome::Skipped)
        .map(|p| p.paper.to_string())
        .collect()
}

fn by_paper(raw: &str) -> SearchFilter {
    SearchFilter {
        paper_id: Some(num(raw)),
        ..SearchFilter::default()
    }
}

async fn latest_flags(store: &LanceStore, number: &str) -> paper_mcp::Result<Vec<(String, bool)>> {
    Ok(store
        .search_metadata("", &by_paper(number))
        .await?
        .into_iter()
        .map(|m| (m.meta.row.paper_num, m.meta.is_latest))
        .collect())
}

#[tokio::test]
async fn selection_by_ids_year_and_all_matches_paperstore_listings() -> TestResult {
    let ws = Workspace::new()?;
    let store = ws.store().await?;
    let cases: [(&[&str], Vec<PaperNum>); 3] = [
        (&["all"], ws.backend.list_all_paper_ids()?),
        (&["2023"], ws.backend.list_paper_ids_for_year(2023)?),
        (&["p2300", "P2169R4"], nums(&["P2169R4", "P2300R10"])),
    ];
    assert_eq!(cases[0].1, nums(&ALL_PAPERS));
    assert_eq!(
        cases[1].1,
        nums(&["N4950", "P2169R4", "P2300R6", "P2300R7"])
    );
    for (args, expected) in cases {
        let args: Vec<String> = args.iter().map(|a| (*a).to_owned()).collect();
        let selected = resolve(&ws.backend, &parse_selectors(&args)?)?;
        assert_eq!(selected, expected, "{args:?}");
        let report = run(&ws.backend, &store, &selected, &forced()).await?;
        assert_eq!(ids(&report), expected, "{args:?}");
        assert!(!report.has_failures(), "{args:?}: {report:?}");
    }
    Ok(())
}

#[tokio::test]
async fn first_run_indexes_content_and_records_metadata_only_papers() -> TestResult {
    let ws = Workspace::new()?;
    let store = ws.store().await?;
    let report = run(&ws.backend, &store, &nums(&ALL_PAPERS), &options()).await?;

    assert_eq!(report.outcome(&num("N4950")), Some(&Outcome::NoContent));
    for paper in ALL_PAPERS.iter().filter(|p| **p != "N4950") {
        assert!(
            matches!(report.outcome(&num(paper)), Some(Outcome::Indexed { chunks }) if *chunks > 0),
            "{paper}: {:?}",
            report.outcome(&num(paper))
        );
    }

    let meta = store.search_metadata("", &by_paper("N4950")).await?;
    assert_eq!(meta.len(), 1);
    assert!(!meta[0].meta.has_markdown);
    let html = store
        .search_keyword("fold_left", &by_paper("P2322R4"))
        .await?;
    assert!(!html.is_empty(), "HTML-origin paper has keyword hits");
    let pdf = store
        .search_keyword("placeholder", &by_paper("P2169R4"))
        .await?;
    assert!(!pdf.is_empty(), "PDF-origin paper has keyword hits");
    for hit in html.iter().chain(&pdf) {
        let md = std::fs::read_to_string(ws.markdown(hit.chunk.paper_id.as_str()))?;
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(
            lines[hit.chunk.start - 1..hit.chunk.end - 1].join("\n"),
            hit.chunk.text,
            "stored range reproduces the chunk text"
        );
        assert_eq!(hit.chunk.content_hash, content_hash(&md));
    }
    Ok(())
}

#[tokio::test]
async fn rerun_skips_unchanged_and_force_reindexes() -> TestResult {
    let ws = Workspace::new()?;
    let store = ws.store().await?;
    let papers = nums(&ALL_PAPERS);
    run(&ws.backend, &store, &papers, &options()).await?;

    let rerun = run(&ws.backend, &store, &papers, &options()).await?;
    assert_eq!(not_skipped(&rerun), Vec::<String>::new());

    let forced_run = run(&ws.backend, &store, &papers, &forced()).await?;
    assert_eq!(skipped(&forced_run), Vec::<String>::new());
    assert!(!forced_run.has_failures());
    Ok(())
}

#[tokio::test]
async fn changing_format_version_marks_every_paper_stale() -> TestResult {
    let ws = Workspace::new()?;
    let store = ws.store().await?;
    let papers = nums(&ALL_PAPERS);
    run(&ws.backend, &store, &papers, &options()).await?;

    let mut bumped = options();
    bumped.format.chunker += 1;
    let report = run(&ws.backend, &store, &papers, &bumped).await?;
    assert_eq!(skipped(&report), Vec::<String>::new());
    Ok(())
}

#[tokio::test]
async fn changed_markdown_triggers_reindex_of_that_paper_only() -> TestResult {
    let ws = Workspace::new()?;
    let store = ws.store().await?;
    let papers = nums(&ALL_PAPERS);
    run(&ws.backend, &store, &papers, &options()).await?;

    let path = ws.markdown("P2300R3");
    let changed = format!(
        "{}\n## Appendix\n\nA reconverted paragraph about zygomorphic senders.\n",
        std::fs::read_to_string(&path)?
    );
    std::fs::write(&path, &changed)?;

    let report = run(&ws.backend, &store, &papers, &options()).await?;
    assert_eq!(not_skipped(&report), vec!["P2300R3"]);
    let fresh = store.freshness(&num("P2300R3")).await?;
    assert_eq!(fresh.map(|(hash, _)| hash), Some(content_hash(&changed)));
    let hits = store
        .search_keyword("zygomorphic", &SearchFilter::default())
        .await?;
    let hit_ids: Vec<_> = hits.iter().map(|h| h.chunk.paper_id.to_string()).collect();
    assert_eq!(hit_ids, vec!["P2300R3"]);
    Ok(())
}

/// Delegates to a real backend but fails reading the Markdown of one paper.
struct FailingBackend<'a> {
    inner: &'a SqliteBackend,
    broken: PaperNum,
}

impl StorageBackend for FailingBackend<'_> {
    fn meta(&self, num: &PaperNum) -> paperstore::Result<PaperRow> {
        self.inner.meta(num)
    }

    fn paper_md(&self, num: &PaperNum) -> paperstore::Result<String> {
        if *num == self.broken {
            return Err(std::io::Error::other("injected read failure").into());
        }
        self.inner.paper_md(num)
    }

    fn list_all_paper_ids(&self) -> paperstore::Result<Vec<PaperNum>> {
        self.inner.list_all_paper_ids()
    }

    fn list_paper_ids_for_year(&self, year: u16) -> paperstore::Result<Vec<PaperNum>> {
        self.inner.list_paper_ids_for_year(year)
    }

    fn list_revisions(&self, num: &PaperNum) -> paperstore::Result<Vec<PaperRow>> {
        self.inner.list_revisions(num)
    }

    fn tool_dir(&self, tool: &str) -> PathBuf {
        self.inner.tool_dir(tool)
    }
}

#[tokio::test]
async fn per_paper_failure_is_reported_while_other_papers_index() -> TestResult {
    let ws = Workspace::new()?;
    let store = ws.store().await?;
    let failing = FailingBackend {
        inner: &ws.backend,
        broken: num("P2322R4"),
    };
    let papers = nums(&ALL_PAPERS);
    let report = run(&failing, &store, &papers, &options()).await?;

    assert!(report.has_failures());
    assert_eq!(report.failed(), 1);
    assert!(
        matches!(report.outcome(&num("P2322R4")), Some(Outcome::Failed(reason)) if reason.contains("injected read failure")),
        "{:?}",
        report.outcome(&num("P2322R4"))
    );
    assert_eq!(ids(&report), papers, "every paper is reported in order");
    assert_eq!(store.freshness(&num("P2322R4")).await?, None);
    for paper in ALL_PAPERS.iter().filter(|p| **p != "P2322R4") {
        assert!(store.freshness(&num(paper)).await?.is_some(), "{paper}");
    }

    let retry = run(&ws.backend, &store, &papers, &options()).await?;
    assert_eq!(
        not_skipped(&retry),
        vec!["P2322R4"],
        "rerun resumes the work"
    );
    assert!(!retry.has_failures());
    Ok(())
}

#[tokio::test]
async fn latest_flags_follow_paperstore_revisions() -> TestResult {
    let ws = Workspace::new()?;
    let store = ws.store().await?;

    run(&ws.backend, &store, &nums(&["P2300R4"]), &options()).await?;
    assert_eq!(
        latest_flags(&store, "P2300").await?,
        vec![("P2300R4".to_owned(), false)],
        "an older revision is not latest while paperstore lists a newer one"
    );

    run(&ws.backend, &store, &nums(&["P2300R10"]), &options()).await?;
    assert_eq!(
        latest_flags(&store, "P2300").await?,
        vec![("P2300R10".to_owned(), true), ("P2300R4".to_owned(), false)]
    );

    let latest_only = SearchFilter {
        latest_only: true,
        ..by_paper("P2300")
    };
    let hits = store.search_keyword("scheduler", &latest_only).await?;
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.chunk.paper_id == num("P2300R10")));
    Ok(())
}

#[tokio::test]
async fn parallel_workers_write_the_same_records_as_serial() -> TestResult {
    let serial_ws = Workspace::new()?;
    let parallel_ws = Workspace::new()?;
    let serial_store = serial_ws.store().await?;
    let parallel_store = parallel_ws.store().await?;
    let papers = nums(&ALL_PAPERS);

    let serial = run(&serial_ws.backend, &serial_store, &papers, &options()).await?;
    let parallel_options = IndexOptions {
        workers: NonZeroUsize::new(4).ok_or("zero workers")?,
        ..options()
    };
    let parallel = run(
        &parallel_ws.backend,
        &parallel_store,
        &papers,
        &parallel_options,
    )
    .await?;
    assert_eq!(serial, parallel);

    let all = SearchFilter::default();
    let key = |hits: Vec<paper_mcp::ChunkHit>| {
        let mut rows: Vec<_> = hits
            .into_iter()
            .map(|h| {
                (
                    h.chunk.paper_id.to_string(),
                    h.chunk.start,
                    h.chunk.end,
                    h.chunk.text,
                )
            })
            .collect();
        rows.sort();
        rows
    };
    assert_eq!(
        key(serial_store.search_keyword("sender", &all).await?),
        key(parallel_store.search_keyword("sender", &all).await?)
    );
    Ok(())
}

#[test]
fn held_lock_blocks_a_second_writer() -> TestResult {
    let ws = Workspace::new()?;
    let first = IndexLock::acquire(&ws.tool_dir())?;
    assert_eq!(first.path(), ws.tool_dir().join("index.lock"));
    match IndexLock::acquire(&ws.tool_dir()) {
        Err(Error::IndexLocked(path)) => assert_eq!(path, first.path()),
        other => panic!("second writer should be blocked, got {other:?}"),
    }
    drop(first);
    IndexLock::acquire(&ws.tool_dir())?;
    Ok(())
}

/// Delegates to a real store but fails the chosen maintenance writes.
struct FaultyStore<'a> {
    inner: &'a LanceStore,
    fail_set_latest: bool,
    fail_optimize: bool,
}

impl IndexStore for FaultyStore<'_> {
    async fn freshness(&self, paper: &PaperNum) -> paper_mcp::Result<Option<(String, String)>> {
        self.inner.freshness(paper).await
    }

    async fn upsert_paper(&self, records: PaperRecords) -> paper_mcp::Result<()> {
        self.inner.upsert_paper(records).await
    }

    async fn delete_paper(&self, paper: &PaperNum) -> paper_mcp::Result<()> {
        self.inner.delete_paper(paper).await
    }

    async fn set_latest(&self, paper_number: &str, latest: &PaperNum) -> paper_mcp::Result<()> {
        if self.fail_set_latest {
            return Err(Error::Index("injected set_latest failure".to_owned()));
        }
        self.inner.set_latest(paper_number, latest).await
    }

    async fn search_vector(
        &self,
        vector: &[f32],
        filter: &SearchFilter,
        limit: usize,
    ) -> paper_mcp::Result<Vec<ChunkHit>> {
        self.inner.search_vector(vector, filter, limit).await
    }

    async fn search_keyword(
        &self,
        query: &str,
        filter: &SearchFilter,
    ) -> paper_mcp::Result<Vec<ChunkHit>> {
        self.inner.search_keyword(query, filter).await
    }

    async fn count_keyword_matches(
        &self,
        query: &str,
        filter: &SearchFilter,
    ) -> paper_mcp::Result<usize> {
        self.inner.count_keyword_matches(query, filter).await
    }

    async fn search_metadata(
        &self,
        query: &str,
        filter: &SearchFilter,
    ) -> paper_mcp::Result<Vec<MetaHit>> {
        self.inner.search_metadata(query, filter).await
    }

    async fn optimize(&self) -> paper_mcp::Result<()> {
        if self.fail_optimize {
            return Err(Error::Index("injected optimize failure".to_owned()));
        }
        self.inner.optimize().await
    }
}

#[tokio::test]
async fn failed_latest_flags_leave_written_papers_stale_for_the_next_run() -> TestResult {
    let ws = Workspace::new()?;
    let store = ws.store().await?;
    run(&ws.backend, &store, &nums(&["P2169R4"]), &options()).await?;

    let faulty = FaultyStore {
        inner: &store,
        fail_set_latest: true,
        fail_optimize: false,
    };
    let papers = nums(&["P2169R4", "P2300R4", "P2300R10"]);
    let report = index_papers(
        &ws.backend,
        &faulty,
        &HashEmbedder::default(),
        &papers,
        &options(),
    )
    .await;

    assert_eq!(report.outcome(&num("P2169R4")), Some(&Outcome::Skipped));
    for paper in ["P2300R4", "P2300R10"] {
        assert!(
            matches!(report.outcome(&num(paper)), Some(Outcome::Failed(reason)) if reason.contains("injected set_latest failure")),
            "{paper}: {:?}",
            report.outcome(&num(paper))
        );
        assert_eq!(store.freshness(&num(paper)).await?, None, "{paper}");
    }
    assert!(report.has_failures());

    let retry = run(&ws.backend, &store, &papers, &options()).await?;
    assert_eq!(not_skipped(&retry), vec!["P2300R4", "P2300R10"]);
    assert!(!retry.has_failures());
    assert_eq!(
        latest_flags(&store, "P2300").await?,
        vec![("P2300R10".to_owned(), true), ("P2300R4".to_owned(), false)]
    );
    Ok(())
}

#[tokio::test]
async fn failed_optimize_keeps_the_per_paper_report() -> TestResult {
    let ws = Workspace::new()?;
    let store = ws.store().await?;
    let faulty = FaultyStore {
        inner: &store,
        fail_set_latest: false,
        fail_optimize: true,
    };
    let papers = nums(&["N4950", "P2169R4"]);
    let report = index_papers(
        &ws.backend,
        &faulty,
        &HashEmbedder::default(),
        &papers,
        &options(),
    )
    .await;

    assert_eq!(ids(&report), papers);
    assert_eq!(report.failed(), 0);
    assert_eq!(report.outcome(&num("N4950")), Some(&Outcome::NoContent));
    assert!(matches!(
        report.outcome(&num("P2169R4")),
        Some(Outcome::Indexed { .. })
    ));
    assert!(
        report
            .optimize_error
            .as_deref()
            .is_some_and(|e| e.contains("injected optimize failure")),
        "{report:?}"
    );
    assert!(report.has_failures());
    Ok(())
}
