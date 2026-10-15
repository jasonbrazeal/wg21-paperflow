// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! `search_chunks` and `search_metadata` over the indexed fixture workspace.

use std::collections::BTreeSet;
use std::sync::Arc;

use paper_mcp::model::fake::{HashEmbedder, IdentityReranker};
use paper_mcp::{
    IndexStore, LanceStore, SearchFilter, SearchHit, SearchMode, SearchParams, SearchRequest,
    effective_filter, search_chunks, search_metadata,
};
use paperstore::{PaperNum, StorageBackend};

use crate::ingest::{ALL_PAPERS, Workspace, num, nums, options, run};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const MODES: [SearchMode; 3] = [SearchMode::Vector, SearchMode::Keyword, SearchMode::Hybrid];

/// A query with keyword matches in many P2300 revisions and other papers.
const QUERY: &str = "sender scheduler";

/// Small enough that the vector side is shorter than the keyword side.
const SMALL_TOP_K: usize = 5;

async fn indexed() -> Result<(Workspace, LanceStore), Box<dyn std::error::Error>> {
    let ws = Workspace::new()?;
    let store = ws.store().await?;
    run(&ws.backend, &store, &nums(&ALL_PAPERS), &options()).await?;
    Ok((ws, store))
}

fn request(mode: SearchMode, filter: SearchFilter, all_revisions: bool) -> SearchRequest {
    SearchRequest {
        query: QUERY.to_owned(),
        mode,
        filter,
        all_revisions,
        top_k: None,
    }
}

fn by_paper(raw: &str) -> SearchFilter {
    SearchFilter {
        paper_id: Some(num(raw)),
        ..SearchFilter::default()
    }
}

async fn search(store: &LanceStore, req: &SearchRequest) -> paper_mcp::Result<Vec<SearchHit>> {
    search_chunks(
        store,
        &Arc::new(HashEmbedder::default()),
        &Arc::new(IdentityReranker),
        &SearchParams::default(),
        req,
    )
    .await
}

type Key = (PaperNum, usize, usize);

fn key_set(keys: impl Iterator<Item = Key>) -> BTreeSet<Key> {
    keys.collect()
}

fn hit_keys(hits: &[SearchHit]) -> BTreeSet<Key> {
    key_set(hits.iter().map(|h| (h.paper_id.clone(), h.start, h.end)))
}

fn papers(hits: &[SearchHit]) -> BTreeSet<String> {
    hits.iter().map(|h| h.paper_id.to_string()).collect()
}

#[tokio::test]
async fn result_counts_follow_the_mode() -> TestResult {
    let (_ws, store) = indexed().await?;
    let mut req = request(SearchMode::Keyword, SearchFilter::default(), true);
    req.top_k = Some(SMALL_TOP_K);
    let filter = effective_filter(&req);

    let keyword = search(&store, &req).await?;
    let expected_matches = store.count_keyword_matches(QUERY, &filter).await?;
    assert!(expected_matches > SMALL_TOP_K, "{expected_matches}");
    assert_eq!(keyword.len(), expected_matches);

    req.mode = SearchMode::Vector;
    let vector = search(&store, &req).await?;
    assert_eq!(vector.len(), SMALL_TOP_K);

    req.mode = SearchMode::Hybrid;
    let hybrid = search(&store, &req).await?;
    let union: BTreeSet<Key> = hit_keys(&keyword)
        .union(&hit_keys(&vector))
        .cloned()
        .collect();
    assert_eq!(hybrid.len(), union.len(), "hybrid holds no duplicates");
    assert_eq!(hit_keys(&hybrid), union);
    assert!(union.len() > keyword.len(), "the vector side adds chunks");
    Ok(())
}

#[tokio::test]
async fn top_k_defaults_to_the_server_parameter() -> TestResult {
    let (_ws, store) = indexed().await?;
    let params = SearchParams {
        top_k: 7,
        ..SearchParams::default()
    };
    let req = request(SearchMode::Vector, SearchFilter::default(), true);
    let hits = search_chunks(
        &store,
        &Arc::new(HashEmbedder::default()),
        &Arc::new(IdentityReranker),
        &params,
        &req,
    )
    .await?;
    assert_eq!(hits.len(), 7);
    Ok(())
}

#[tokio::test]
async fn default_search_returns_only_the_latest_revision() -> TestResult {
    let (_ws, store) = indexed().await?;
    for mode in MODES {
        let hits = search(&store, &request(mode, by_paper("P2300"), false)).await?;
        assert!(!hits.is_empty(), "{mode:?}");
        assert_eq!(
            papers(&hits),
            BTreeSet::from(["P2300R10".to_owned()]),
            "{mode:?}"
        );

        let unscoped = search(&store, &request(mode, SearchFilter::default(), false)).await?;
        let older: Vec<_> = unscoped
            .iter()
            .filter(|h| h.paper_id.as_str().starts_with("P2300") && h.paper_id != num("P2300R10"))
            .collect();
        assert!(older.is_empty(), "{mode:?}: {older:?}");
    }
    Ok(())
}

#[tokio::test]
async fn all_revisions_returns_other_revisions() -> TestResult {
    let (_ws, store) = indexed().await?;
    for mode in MODES {
        let hits = search(&store, &request(mode, by_paper("P2300"), true)).await?;
        let found = papers(&hits);
        assert!(found.len() > 1, "{mode:?}: {found:?}");
        assert!(found.iter().all(|p| p.starts_with("P2300")), "{mode:?}");
    }
    Ok(())
}

#[tokio::test]
async fn versioned_id_restricts_to_that_revision() -> TestResult {
    let (_ws, store) = indexed().await?;
    for mode in MODES {
        let hits = search(&store, &request(mode, by_paper("P2300R4"), false)).await?;
        assert!(!hits.is_empty(), "{mode:?}");
        assert_eq!(
            papers(&hits),
            BTreeSet::from(["P2300R4".to_owned()]),
            "{mode:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn filters_apply_to_every_side() -> TestResult {
    let (_ws, store) = indexed().await?;
    let core = SearchFilter {
        target_group: Some("cwg".to_owned()),
        ..SearchFilter::default()
    };
    let year = SearchFilter {
        year: Some(2021),
        ..SearchFilter::default()
    };
    let year_2021 = ["P2300R0", "P2300R1", "P2300R2", "P2300R3", "P2322R4"];
    for mode in MODES {
        let mut req = request(mode, core.clone(), true);
        req.query = "placeholder sender".to_owned();
        let hits = search(&store, &req).await?;
        assert!(!hits.is_empty(), "{mode:?}");
        assert_eq!(
            papers(&hits),
            BTreeSet::from(["P2169R4".to_owned()]),
            "{mode:?}"
        );

        let hits = search(&store, &request(mode, year.clone(), true)).await?;
        assert!(!hits.is_empty(), "{mode:?}");
        let found = papers(&hits);
        assert!(
            found.iter().all(|p| year_2021.contains(&p.as_str())),
            "{mode:?}: {found:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn hits_carry_unprefixed_source_text() -> TestResult {
    let (ws, store) = indexed().await?;
    let hits = search(
        &store,
        &request(SearchMode::Hybrid, by_paper("P2300R10"), false),
    )
    .await?;
    assert!(!hits.is_empty());
    for hit in &hits {
        let md = std::fs::read_to_string(ws.backend.meta(&hit.paper_id)?.markdown_path)?;
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[hit.start - 1..hit.end - 1].join("\n"), hit.text);
        assert_eq!(hit.title, "`std::execution`");
        assert!(!hit.heading_path.is_empty());
    }
    Ok(())
}

#[tokio::test]
async fn metadata_search_returns_every_matching_row() -> TestResult {
    let (_ws, store) = indexed().await?;
    let mut req = request(SearchMode::Keyword, by_paper("P2300"), false);
    req.query = String::new();
    let ids = |rows: Vec<paper_mcp::MetaRecord>| -> Vec<String> {
        rows.into_iter().map(|m| m.row.paper_num).collect()
    };

    assert_eq!(ids(search_metadata(&store, &req).await?), vec!["P2300R10"]);

    req.all_revisions = true;
    assert_eq!(search_metadata(&store, &req).await?.len(), 11);

    req.filter = by_paper("P2300R4");
    req.all_revisions = false;
    assert_eq!(ids(search_metadata(&store, &req).await?), vec!["P2300R4"]);

    req.filter = SearchFilter::default();
    req.query = "Revzin".to_owned();
    assert_eq!(ids(search_metadata(&store, &req).await?), vec!["P2322R4"]);
    Ok(())
}
