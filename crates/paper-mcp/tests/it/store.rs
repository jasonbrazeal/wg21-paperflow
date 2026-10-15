// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! `LanceStore` against temp-dir indexes, with the hash test embedder.

use std::path::Path;
use std::time::Duration;

use paper_mcp::model::fake::HashEmbedder;
use paper_mcp::{
    ChunkHit, ChunkRecord, Embedder, Error, IndexStore, LanceStore, LanceStoreOptions, MetaRecord,
    PaperFields, PaperRecords, SearchFilter, build_prefix, prefixed_text,
};
use paperstore::{PaperNum, PaperRow};
use tempfile::TempDir;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const FORMAT: &str = "test-format-v1";

fn options() -> LanceStoreOptions {
    LanceStoreOptions {
        read_consistency_interval: Duration::ZERO,
        ..LanceStoreOptions::default()
    }
}

async fn open(dir: &Path) -> paper_mcp::Result<LanceStore> {
    LanceStore::open_with(dir, HashEmbedder::default().dim(), options()).await
}

fn num(raw: &str) -> PaperNum {
    PaperNum::parse(raw).unwrap_or_else(|e| panic!("parse {raw}: {e}"))
}

fn row(id: &str) -> PaperRow {
    PaperRow {
        paper_num: id.to_owned(),
        year: "2024".to_owned(),
        title: format!("Title of {id}"),
        target_group: "LEWG Library Evolution".to_owned(),
        document_date: "2024-05-01".to_owned(),
        mailing_date: "2024-05".to_owned(),
        authors: vec!["Alice Liddell".to_owned()],
        line_count: 10,
        ..PaperRow::default()
    }
}

fn chunk(row: &PaperRow, ordinal: usize, text: &str) -> ChunkRecord {
    let paper_id = num(&row.paper_num);
    let (number, revision) = paper_id.split_revision();
    let heading_path = vec!["Intro".to_owned(), format!("Part {ordinal}")];
    let prefixed = prefixed_text(&build_prefix(row, &heading_path), text);
    let vector = HashEmbedder::default()
        .embed_documents(std::slice::from_ref(&prefixed))
        .unwrap_or_else(|e| panic!("embed: {e}"))
        .pop()
        .unwrap_or_default();
    ChunkRecord {
        paper_number: number.to_owned(),
        revision,
        paper_id,
        is_latest: true,
        heading_path,
        start: ordinal * 10 + 1,
        end: ordinal * 10 + 5,
        text: text.to_owned(),
        prefixed_text: prefixed,
        vector,
        content_hash: format!("hash-{}", row.paper_num),
        format_version: FORMAT.to_owned(),
        fields: PaperFields::from_row(row),
    }
}

fn records_for(row: &PaperRow, texts: &[&str]) -> PaperRecords {
    PaperRecords {
        meta: MetaRecord {
            row: row.clone(),
            is_latest: true,
            has_markdown: !texts.is_empty(),
            content_hash: format!("hash-{}", row.paper_num),
            format_version: FORMAT.to_owned(),
        },
        chunks: texts
            .iter()
            .enumerate()
            .map(|(i, t)| chunk(row, i, t))
            .collect(),
    }
}

fn records(id: &str, texts: &[&str]) -> PaperRecords {
    records_for(&row(id), texts)
}

fn ids(hits: &[ChunkHit]) -> Vec<String> {
    hits.iter()
        .map(|h| format!("{}@{}", h.chunk.paper_id, h.chunk.start))
        .collect()
}

fn by_paper(id: &str) -> SearchFilter {
    SearchFilter {
        paper_id: Some(num(id)),
        ..SearchFilter::default()
    }
}

#[tokio::test]
async fn upsert_round_trips_chunk_fields() -> TestResult {
    let dir = TempDir::new()?;
    let store = open(dir.path()).await?;
    let recs = records("P1000R0", &["alpha bravo", "charlie delta"]);
    let expected = recs.chunks[1].clone();
    store.upsert_paper(recs).await?;

    let hits = store
        .search_keyword("charlie", &SearchFilter::default())
        .await?;
    assert_eq!(hits.len(), 1);
    let got = &hits[0].chunk;
    assert_eq!(got.text, "charlie delta");
    assert!(got.vector.is_empty());
    assert_eq!(
        ChunkRecord {
            vector: expected.vector.clone(),
            ..got.clone()
        },
        expected
    );
    assert!(hits[0].score > 0.0);
    Ok(())
}

#[tokio::test]
async fn upsert_replaces_every_chunk_of_one_revision() -> TestResult {
    let dir = TempDir::new()?;
    let store = open(dir.path()).await?;
    store
        .upsert_paper(records(
            "P1000R0",
            &["alpha one", "alpha two", "alpha three"],
        ))
        .await?;
    store
        .upsert_paper(records("P2000R0", &["alpha other"]))
        .await?;
    store
        .upsert_paper(records("P1000R0", &["alpha replaced"]))
        .await?;

    let hits = store.search_keyword("alpha", &by_paper("P1000R0")).await?;
    assert_eq!(ids(&hits), vec!["P1000R0@1"]);
    assert_eq!(hits[0].chunk.text, "alpha replaced");
    assert!(
        store
            .search_keyword("three", &SearchFilter::default())
            .await?
            .is_empty()
    );
    assert_eq!(
        store
            .search_keyword("other", &SearchFilter::default())
            .await?
            .len(),
        1
    );

    // Replacing with no chunks leaves a metadata-only paper.
    store.upsert_paper(records("P1000R0", &[])).await?;
    assert!(
        store
            .search_keyword("alpha", &by_paper("P1000R0"))
            .await?
            .is_empty()
    );
    let meta = store.search_metadata("", &by_paper("P1000R0")).await?;
    assert_eq!(meta.len(), 1);
    assert!(!meta[0].meta.has_markdown);
    Ok(())
}

#[tokio::test]
async fn delete_removes_chunks_and_metadata_of_one_paper() -> TestResult {
    let dir = TempDir::new()?;
    let store = open(dir.path()).await?;
    store
        .upsert_paper(records("P1000R0", &["alpha one"]))
        .await?;
    store
        .upsert_paper(records("P2000R0", &["alpha two"]))
        .await?;
    store.delete_paper(&num("P1000R0")).await?;

    assert_eq!(store.freshness(&num("P1000R0")).await?, None);
    let hits = store
        .search_keyword("alpha", &SearchFilter::default())
        .await?;
    assert_eq!(ids(&hits), vec!["P2000R0@1"]);
    let meta = store.search_metadata("", &SearchFilter::default()).await?;
    let meta_ids: Vec<_> = meta.iter().map(|m| m.meta.row.paper_num.clone()).collect();
    assert_eq!(meta_ids, vec!["P2000R0"]);
    Ok(())
}

#[tokio::test]
async fn freshness_reports_hash_and_format() -> TestResult {
    let dir = TempDir::new()?;
    let store = open(dir.path()).await?;
    assert_eq!(store.freshness(&num("P1000R0")).await?, None);

    store.upsert_paper(records("P1000R0", &["alpha"])).await?;
    assert_eq!(
        store.freshness(&num("P1000R0")).await?,
        Some(("hash-P1000R0".to_owned(), FORMAT.to_owned()))
    );

    let mut newer = records("P1000R0", &["alpha"]);
    newer.meta.content_hash = "hash-2".to_owned();
    newer.meta.format_version = "format-2".to_owned();
    store.upsert_paper(newer).await?;
    assert_eq!(
        store.freshness(&num("P1000R0")).await?,
        Some(("hash-2".to_owned(), "format-2".to_owned()))
    );
    assert_eq!(store.freshness(&num("P1000R1")).await?, None);
    Ok(())
}

#[tokio::test]
async fn set_latest_recomputes_every_revision() -> TestResult {
    let dir = TempDir::new()?;
    let store = open(dir.path()).await?;
    for id in ["P2300R0", "P2300R1", "P2300R2", "P9999R0"] {
        store
            .upsert_paper(records(id, &["sender receiver"]))
            .await?;
    }
    store.set_latest("P2300", &num("P2300R1")).await?;

    let latest = SearchFilter {
        latest_only: true,
        ..SearchFilter::default()
    };
    let hits = store.search_keyword("sender", &latest).await?;
    let mut got = ids(&hits);
    got.sort();
    assert_eq!(got, vec!["P2300R1@1", "P9999R0@1"]);

    let meta = store.search_metadata("", &by_paper("P2300")).await?;
    let flags: Vec<_> = meta
        .iter()
        .map(|m| (m.meta.row.paper_num.clone(), m.meta.is_latest))
        .collect();
    assert_eq!(
        flags,
        vec![
            ("P2300R0".to_owned(), false),
            ("P2300R1".to_owned(), true),
            ("P2300R2".to_owned(), false),
        ]
    );

    store.set_latest("P2300", &num("P2300R2")).await?;
    let hits = store
        .search_keyword(
            "sender",
            &SearchFilter {
                paper_id: Some(num("P2300")),
                latest_only: true,
                ..SearchFilter::default()
            },
        )
        .await?;
    assert_eq!(ids(&hits), vec!["P2300R2@1"]);
    Ok(())
}

/// Filters over the papers of `filters_restrict_every_search_side`, each
/// with the paper ids it must match.
fn filter_cases() -> Vec<(SearchFilter, Vec<&'static str>)> {
    vec![
        (
            SearchFilter {
                author: Some("o'neal".to_owned()),
                ..SearchFilter::default()
            },
            vec!["P1000R0"],
        ),
        (
            SearchFilter {
                author: Some("' OR '1'='1".to_owned()),
                ..SearchFilter::default()
            },
            vec![],
        ),
        (
            SearchFilter {
                target_group: Some("lewg".to_owned()),
                ..SearchFilter::default()
            },
            vec!["P2000R0", "P2000R1"],
        ),
        (
            SearchFilter {
                intent: Some("INFORM".to_owned()),
                ..SearchFilter::default()
            },
            vec!["P1000R0"],
        ),
        (
            SearchFilter {
                year: Some(2021),
                ..SearchFilter::default()
            },
            vec!["P1000R0"],
        ),
        (
            SearchFilter {
                mailing_date: Some("2024-05".to_owned()),
                ..SearchFilter::default()
            },
            vec!["P2000R0", "P2000R1"],
        ),
        (
            SearchFilter {
                document_date_from: Some("2021-01-01".to_owned()),
                document_date_to: Some("2021-12-31".to_owned()),
                ..SearchFilter::default()
            },
            vec!["P1000R0"],
        ),
        (
            SearchFilter {
                revision: Some(1),
                ..SearchFilter::default()
            },
            vec!["P2000R1"],
        ),
        (by_paper("P2000"), vec!["P2000R0", "P2000R1"]),
        (by_paper("P2000R0"), vec!["P2000R0"]),
    ]
}

#[tokio::test]
async fn filters_restrict_every_search_side() -> TestResult {
    let dir = TempDir::new()?;
    let store = open(dir.path()).await?;
    let mut quoted = row("P1000R0");
    quoted.authors = vec!["Shaquille O'Neal".to_owned()];
    quoted.target_group = "SG1 Concurrency".to_owned();
    quoted.intent = "Inform".to_owned();
    quoted.year = "2021".to_owned();
    quoted.mailing_date = "2021-02".to_owned();
    quoted.document_date = "2021-02-12".to_owned();
    store
        .upsert_paper(records_for(&quoted, &["shared words"]))
        .await?;
    store
        .upsert_paper(records("P2000R0", &["shared words"]))
        .await?;
    store
        .upsert_paper(records("P2000R1", &["shared words"]))
        .await?;

    let query = HashEmbedder::default().embed_query("shared words")?;
    for (filter, expected) in filter_cases() {
        let mut kw: Vec<_> = store
            .search_keyword("shared", &filter)
            .await?
            .iter()
            .map(|h| h.chunk.paper_id.to_string())
            .collect();
        kw.sort();
        assert_eq!(kw, expected, "keyword {filter:?}");
        let mut vec_ids: Vec<_> = store
            .search_vector(&query, &filter, 10)
            .await?
            .iter()
            .map(|h| h.chunk.paper_id.to_string())
            .collect();
        vec_ids.sort();
        assert_eq!(vec_ids, expected, "vector {filter:?}");
        let meta: Vec<_> = store
            .search_metadata("", &filter)
            .await?
            .iter()
            .map(|m| m.meta.row.paper_num.clone())
            .collect();
        assert_eq!(meta, expected, "metadata {filter:?}");
    }
    Ok(())
}

#[tokio::test]
async fn keyword_search_matches_cpp_identifiers() -> TestResult {
    let dir = TempDir::new()?;
    let store = open(dir.path()).await?;
    store
        .upsert_paper(records(
            "P0288R9",
            &[
                "Add `std::move_only_function`, a move-only wrapper.",
                "Unlike std::function, it is not copyable.",
                "A move only function type.",
                "Specialize std::move_only_function<R(Args...)>; for each cv.",
                "See (std::move_only_function).",
            ],
        ))
        .await?;

    let hits = store
        .search_keyword("std::move_only_function", &SearchFilter::default())
        .await?;
    let mut got = ids(&hits);
    got.sort();
    assert_eq!(got, vec!["P0288R9@1", "P0288R9@41"]);
    let generic = store
        .search_keyword(
            "std::move_only_function<R(Args...)>;",
            &SearchFilter::default(),
        )
        .await?;
    assert_eq!(ids(&generic), vec!["P0288R9@31"]);
    assert!(
        store
            .search_keyword("  `` ", &SearchFilter::default())
            .await?
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
async fn keyword_search_returns_every_match() -> TestResult {
    let dir = TempDir::new()?;
    let store = open(dir.path()).await?;
    let texts: Vec<String> = (0..40).map(|i| format!("needle number {i}")).collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    store.upsert_paper(records("P1000R0", &refs)).await?;
    store.upsert_paper(records("P2000R0", &refs[..15])).await?;
    store
        .upsert_paper(records("P3000R0", &["haystack only"]))
        .await?;

    let all = SearchFilter::default();
    assert_eq!(store.count_keyword_matches("needle", &all).await?, 55);
    let hits = store.search_keyword("needle", &all).await?;
    assert_eq!(hits.len(), 55);
    assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
    assert_eq!(
        store
            .count_keyword_matches("needle", &by_paper("P2000R0"))
            .await?,
        15
    );
    assert_eq!(store.count_keyword_matches("absent", &all).await?, 0);
    assert!(store.search_keyword("absent", &all).await?.is_empty());

    // Rows written after the last optimize are still matched.
    store.optimize().await?;
    store
        .upsert_paper(records("P4000R0", &["needle late"]))
        .await?;
    assert_eq!(store.search_keyword("needle", &all).await?.len(), 56);
    Ok(())
}

#[tokio::test]
async fn metadata_search_by_title_and_author() -> TestResult {
    let dir = TempDir::new()?;
    let store = open(dir.path()).await?;
    let mut a = row("P1000R0");
    a.title = "Executors for `std::execution`".to_owned();
    a.authors = vec!["Eric Niebler".to_owned(), "Lewis Baker".to_owned()];
    let mut b = row("P2000R0");
    b.title = "Contracts for C++".to_owned();
    b.authors = vec!["Joshua Berne".to_owned()];
    let mut c = row("P3000R0");
    c.title = "Reflection".to_owned();
    c.authors = vec!["Barry Revzin".to_owned(), "Lewis Baker".to_owned()];
    store.upsert_paper(records_for(&a, &["body"])).await?;
    store.upsert_paper(records_for(&b, &[])).await?;
    store.upsert_paper(records_for(&c, &["body"])).await?;

    let paper_ids = |hits: &[paper_mcp::MetaHit]| -> Vec<String> {
        hits.iter().map(|h| h.meta.row.paper_num.clone()).collect()
    };
    let all = SearchFilter::default();
    assert_eq!(
        paper_ids(&store.search_metadata("std::execution", &all).await?),
        vec!["P1000R0"]
    );
    assert_eq!(
        paper_ids(&store.search_metadata("contracts", &all).await?),
        vec!["P2000R0"]
    );
    let mut baker = paper_ids(&store.search_metadata("Baker", &all).await?);
    baker.sort();
    assert_eq!(baker, vec!["P1000R0", "P3000R0"]);
    assert_eq!(
        paper_ids(&store.search_metadata("", &all).await?),
        vec!["P1000R0", "P2000R0", "P3000R0"]
    );

    let hit = &store.search_metadata("berne", &all).await?[0];
    assert_eq!(hit.meta.row.title, "Contracts for C++");
    assert_eq!(hit.meta.row.authors, vec!["Joshua Berne".to_owned()]);
    assert_eq!(hit.meta.row.line_count, 10);
    assert!(!hit.meta.has_markdown);
    Ok(())
}

#[tokio::test]
async fn vector_search_orders_by_similarity_and_honors_limit() -> TestResult {
    let dir = TempDir::new()?;
    let store = open(dir.path()).await?;
    store
        .upsert_paper(records(
            "P1000R0",
            &[
                "unicode normalization identifiers",
                "sender receiver scheduler executor",
                "allocator awareness",
            ],
        ))
        .await?;
    let query = HashEmbedder::default().embed_query("sender receiver scheduler executor")?;
    let hits = store
        .search_vector(&query, &SearchFilter::default(), 2)
        .await?;
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].chunk.text, "sender receiver scheduler executor");
    assert!(hits[0].score >= hits[1].score);
    assert!(
        store
            .search_vector(&query, &SearchFilter::default(), 0)
            .await?
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
async fn upsert_rejects_inconsistent_records() -> TestResult {
    let dir = TempDir::new()?;
    let store = open(dir.path()).await?;
    let mut wrong_dim = records("P1000R0", &["alpha"]);
    wrong_dim.chunks[0].vector.pop();
    assert!(matches!(
        store.upsert_paper(wrong_dim).await,
        Err(Error::Index(_))
    ));
    let mut wrong_paper = records("P1000R0", &["alpha"]);
    wrong_paper.chunks.push(chunk(&row("P2000R0"), 1, "beta"));
    assert!(matches!(
        store.upsert_paper(wrong_paper).await,
        Err(Error::Index(_))
    ));
    assert_eq!(store.freshness(&num("P1000R0")).await?, None);
    Ok(())
}

#[tokio::test]
async fn reopening_with_another_dimension_fails() -> TestResult {
    let dir = TempDir::new()?;
    drop(open(dir.path()).await?);
    let other =
        LanceStore::open_with(dir.path(), HashEmbedder::default().dim() + 1, options()).await;
    assert!(matches!(other, Err(Error::Index(_))));
    Ok(())
}

#[tokio::test]
async fn second_handle_sees_writes_after_optimize() -> TestResult {
    let dir = TempDir::new()?;
    let writer = open(dir.path()).await?;
    let reader = open(dir.path()).await?;
    assert!(
        reader
            .search_keyword("alpha", &SearchFilter::default())
            .await?
            .is_empty()
    );

    writer
        .upsert_paper(records("P1000R0", &["alpha bravo"]))
        .await?;
    writer.set_latest("P1000", &num("P1000R0")).await?;
    writer.optimize().await?;

    let hits = reader
        .search_keyword("alpha", &SearchFilter::default())
        .await?;
    assert_eq!(ids(&hits), vec!["P1000R0@1"]);
    assert!(reader.freshness(&num("P1000R0")).await?.is_some());
    assert_eq!(
        reader
            .search_metadata("title", &SearchFilter::default())
            .await?
            .len(),
        1
    );
    Ok(())
}

#[tokio::test]
async fn optimize_builds_the_vector_index_once_enough_rows_exist() -> TestResult {
    let dir = TempDir::new()?;
    let store = LanceStore::open_with(
        dir.path(),
        HashEmbedder::default().dim(),
        LanceStoreOptions {
            vector_index_min_rows: 300,
            ..options()
        },
    )
    .await?;
    let texts: Vec<String> = (0..320)
        .map(|i| format!("chunk {i} topic{}", i % 7))
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    store.upsert_paper(records("P1000R0", &refs[..299])).await?;
    store.optimize().await?;
    assert!(!store.has_vector_index().await?);

    store.upsert_paper(records("P1000R0", &refs)).await?;
    store.optimize().await?;
    assert!(store.has_vector_index().await?);
    // A second optimize updates, rather than recreates, the index.
    store.optimize().await?;
    assert!(store.has_vector_index().await?);

    let query = HashEmbedder::default().embed_query("chunk 5 topic5")?;
    let hits = store
        .search_vector(&query, &SearchFilter::default(), 5)
        .await?;
    assert_eq!(hits.len(), 5);
    let filtered = store.search_vector(&query, &by_paper("P2000R0"), 5).await?;
    assert!(filtered.is_empty());
    assert_eq!(
        store
            .search_keyword("topic5", &SearchFilter::default())
            .await?
            .len(),
        45
    );
    Ok(())
}

#[tokio::test]
async fn read_only_open_never_creates_the_index() -> TestResult {
    let dir = TempDir::new()?;
    let dim = HashEmbedder::default().dim();
    let missing = LanceStore::open_read_only(dir.path(), dim, options()).await;
    assert!(
        matches!(&missing, Err(Error::NoIndex(path)) if path == &dir.path().join("lancedb")),
        "{missing:?}"
    );
    assert!(!dir.path().join("lancedb").exists(), "nothing was created");

    open(dir.path()).await?;
    let store = LanceStore::open_read_only(dir.path(), dim, options()).await?;
    assert!(
        store
            .search_metadata("", &SearchFilter::default())
            .await?
            .is_empty()
    );
    let wrong_dim = LanceStore::open_read_only(dir.path(), dim + 1, options()).await;
    assert!(matches!(wrong_dim, Err(Error::Index(_))), "{wrong_dim:?}");
    Ok(())
}
