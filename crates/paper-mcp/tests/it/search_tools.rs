// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! The search tools and end-to-end server behavior over rmcp Streamable HTTP.

use std::collections::BTreeSet;
use std::io::{Read as _, Write as _};
use std::net::{SocketAddr, TcpStream};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use paper_mcp::model::fake::{HashEmbedder, IdentityReranker};
use paper_mcp::{
    CursorConfig, IndexStore as _, MCP_PATH, SearchFilter, SearchMode, SearchParams, SearchRequest,
    effective_filter, search_chunks,
};
use serde_json::{Value, json};

use crate::ingest::{ALL_PAPERS, nums, options, run};
use crate::tools::Harness;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A query with keyword matches in many P2300 revisions and other papers.
const QUERY: &str = "sender scheduler";

/// A query matching P2300 revisions and the late paper P2169R4.
const LATE_QUERY: &str = "sender placeholder";

/// Small enough that every mode spans several pages.
const PAGE_SIZE: NonZeroUsize = match NonZeroUsize::new(7) {
    Some(n) => n,
    None => NonZeroUsize::MIN,
};

/// Vector-side depth: several pages, and fewer than the keyword matches.
const TOP_K: usize = 30;

/// Papers indexed before the server starts in the concurrent-index tests.
const EARLY_PAPERS: [&str; 12] = [
    "N4950", "P2300R0", "P2300R1", "P2300R10", "P2300R2", "P2300R3", "P2300R4", "P2300R5",
    "P2300R6", "P2300R7", "P2300R8", "P2300R9",
];

/// Papers indexed while the server is running.
const LATE_PAPERS: [&str; 2] = ["P2169R4", "P2322R4"];

fn paging() -> CursorConfig {
    CursorConfig {
        page_size: PAGE_SIZE,
        ..CursorConfig::default()
    }
}

fn mode_name(mode: SearchMode) -> &'static str {
    match mode {
        SearchMode::Vector => "vector",
        SearchMode::Keyword => "keyword",
        SearchMode::Hybrid => "hybrid",
    }
}

fn papers_args(mode: SearchMode) -> Value {
    query_args(QUERY, mode)
}

fn query_args(query: &str, mode: SearchMode) -> Value {
    json!({"query": query, "mode": mode_name(mode), "all_revisions": true, "top_k": TOP_K})
}

fn request(mode: SearchMode) -> SearchRequest {
    query_request(QUERY, mode)
}

fn query_request(query: &str, mode: SearchMode) -> SearchRequest {
    SearchRequest {
        query: query.to_owned(),
        mode,
        filter: SearchFilter::default(),
        all_revisions: true,
        top_k: Some(TOP_K),
    }
}

/// One page's results and next cursor, checking the page size.
fn split_page(page: &Value) -> Result<(Vec<Value>, Option<String>), Box<dyn std::error::Error>> {
    let results = page["results"]
        .as_array()
        .ok_or_else(|| format!("no results array: {page}"))?
        .clone();
    assert!(results.len() <= PAGE_SIZE.get(), "oversized page: {page}");
    let next = page["next_cursor"].as_str().map(str::to_owned);
    if next.is_some() {
        assert_eq!(results.len(), PAGE_SIZE.get(), "short page before the end");
    }
    Ok((results, next))
}

/// Follow `cursor` to the last page, returning the results in order.
async fn rest(
    h: &Harness,
    tool: &str,
    args: &Value,
    mut cursor: Option<String>,
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let mut all = Vec::new();
    while let Some(c) = cursor {
        let mut args = args.clone();
        args["cursor"] = json!(c);
        let (results, next) = split_page(&h.structured(tool, args).await?)?;
        all.extend(results);
        cursor = next;
    }
    Ok(all)
}

/// Run `tool` and page it to the end, returning every result in order.
async fn page_all(
    h: &Harness,
    tool: &str,
    args: &Value,
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let (mut all, next) = split_page(&h.structured(tool, args.clone()).await?)?;
    all.extend(rest(h, tool, args, next).await?);
    Ok(all)
}

type Key = (String, u64, u64);

fn key(hit: &Value) -> Key {
    (
        hit["paper_id"].as_str().unwrap_or_default().to_owned(),
        hit["start"].as_u64().unwrap_or_default(),
        hit["end"].as_u64().unwrap_or_default(),
    )
}

fn keys(hits: &[Value]) -> Vec<Key> {
    hits.iter().map(key).collect()
}

/// The full ranked list a direct library search returns for `req`.
async fn direct(h: &Harness, req: &SearchRequest) -> Result<Vec<Key>, Box<dyn std::error::Error>> {
    let hits = search_chunks(
        &h.ws.store().await?,
        &Arc::new(HashEmbedder::default()),
        &Arc::new(IdentityReranker),
        &SearchParams::default(),
        req,
    )
    .await?;
    let mut keys = Vec::with_capacity(hits.len());
    for hit in hits {
        keys.push((
            hit.paper_id.to_string(),
            u64::try_from(hit.start)?,
            u64::try_from(hit.end)?,
        ));
    }
    Ok(keys)
}

/// Send `body` to the MCP endpoint without a token; return the status line.
fn unauthenticated_post(addr: SocketAddr, body: &str) -> std::io::Result<String> {
    let mut stream = TcpStream::connect(addr)?;
    write!(
        stream,
        "POST {MCP_PATH} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Accept: application/json, text/event-stream\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response.lines().next().unwrap_or_default().to_owned())
}

#[tokio::test]
async fn all_eight_tools_need_the_token() -> TestResult {
    let h = Harness::start_with(&ALL_PAPERS, paging()).await?;
    let calls = [
        ("get_paper", json!({"paper_id": "P2300"})),
        (
            "get_paper_lines",
            json!({"paper_id": "P2300", "start": 1, "end": 3}),
        ),
        ("get_paper_metadata", json!({"paper_id": "P2300"})),
        ("get_paper_outline", json!({"paper_id": "P2300"})),
        ("list_revisions", json!({"paper_id": "P2300"})),
        (
            "diff_revisions",
            json!({"from_id": "P2300R9", "to_id": "P2300"}),
        ),
        ("search_papers", json!({"query": QUERY, "mode": "hybrid"})),
        ("search_metadata", json!({"query": "Revzin"})),
    ];
    for (id, (tool, args)) in calls.iter().enumerate() {
        let result = h.call(tool, args.clone()).await?;
        assert_ne!(result.is_error, Some(true), "{tool}: {result:?}");

        let body = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        })
        .to_string();
        let addr = h.addr;
        let status =
            tokio::task::spawn_blocking(move || unauthenticated_post(addr, &body)).await??;
        assert!(status.contains(" 401 "), "{tool}: {status}");
    }
    let metadata = h
        .structured("search_metadata", json!({"query": "Revzin"}))
        .await?;
    assert_eq!(metadata["results"][0]["paper_id"], "P2322R4");
    h.stop().await
}

#[tokio::test]
async fn every_hit_range_reads_back_its_text() -> TestResult {
    let h = Harness::start_with(&ALL_PAPERS, paging()).await?;
    let hits = page_all(&h, "search_papers", &papers_args(SearchMode::Hybrid)).await?;
    assert!(hits.len() > PAGE_SIZE.get(), "{}", hits.len());
    for hit in &hits {
        let lines = h
            .structured(
                "get_paper_lines",
                json!({"paper_id": hit["paper_id"], "start": hit["start"], "end": hit["end"]}),
            )
            .await?;
        assert_eq!(lines["text"], hit["text"], "{hit}");
        assert_eq!(lines["paper_id"], hit["paper_id"]);
    }
    h.stop().await
}

#[tokio::test]
async fn paging_each_mode_yields_the_full_ranked_list() -> TestResult {
    let h = Harness::start_with(&ALL_PAPERS, paging()).await?;
    let mut by_mode = Vec::new();
    for mode in [SearchMode::Keyword, SearchMode::Vector, SearchMode::Hybrid] {
        let paged = keys(&page_all(&h, "search_papers", &papers_args(mode)).await?);
        assert_eq!(paged, direct(&h, &request(mode)).await?, "{mode:?}");
        by_mode.push(paged);
    }
    let [keyword, vector, hybrid] = by_mode.as_slice() else {
        return Err("three modes".into());
    };
    let req = request(SearchMode::Keyword);
    let matches =
        h.ws.store()
            .await?
            .count_keyword_matches(QUERY, &effective_filter(&req))
            .await?;
    assert!(matches > TOP_K, "{matches}");
    assert_eq!(keyword.len(), matches);
    assert_eq!(vector.len(), TOP_K);
    let union: BTreeSet<&Key> = keyword.iter().chain(vector).collect();
    assert_eq!(hybrid.len(), union.len());
    assert_eq!(hybrid.iter().collect::<BTreeSet<_>>(), union);

    let all = page_all(
        &h,
        "search_metadata",
        &json!({"paper_id": "P2300", "all_revisions": true}),
    )
    .await?;
    let ids: BTreeSet<&str> = all.iter().filter_map(|m| m["paper_id"].as_str()).collect();
    let want: BTreeSet<String> = (0..=10).map(|r| format!("P2300R{r}")).collect();
    assert_eq!(ids, want.iter().map(String::as_str).collect());
    assert_eq!(all.len(), want.len());

    let (_, next) = split_page(
        &h.structured("search_papers", papers_args(SearchMode::Keyword))
            .await?,
    )?;
    let crossed = json!({"cursor": next.ok_or("several pages")?});
    let message = h.error("search_metadata", crossed).await?;
    assert!(message.contains("search_papers"), "{message}");
    h.stop().await
}

#[tokio::test]
async fn cursor_pages_survive_an_index_run() -> TestResult {
    let h = Harness::start_with(&EARLY_PAPERS, paging()).await?;
    let args = query_args(LATE_QUERY, SearchMode::Keyword);
    let req = query_request(LATE_QUERY, SearchMode::Keyword);
    let before = direct(&h, &req).await?;
    let (first, next) = split_page(&h.structured("search_papers", args.clone()).await?)?;
    assert!(next.is_some());

    run(
        &h.ws.backend,
        &h.ws.store().await?,
        &nums(&LATE_PAPERS),
        &options(),
    )
    .await?;
    let after = direct(&h, &req).await?;
    assert_ne!(after, before, "the index run changes fresh results");

    let mut paged = keys(&first);
    paged.extend(keys(&rest(&h, "search_papers", &args, next).await?));
    assert_eq!(paged, before, "an existing cursor keeps its list");
    h.await_fresh_reads().await;
    let fresh = keys(&page_all(&h, "search_papers", &args).await?);
    assert_eq!(fresh, after, "a new search sees the new data");
    h.stop().await
}

#[tokio::test]
async fn server_keeps_answering_during_an_index_run() -> TestResult {
    let h = Harness::start_with(&EARLY_PAPERS, paging()).await?;
    let revzin = json!({"query": "Revzin"});
    let found = h.structured("search_metadata", revzin.clone()).await?;
    assert_eq!(found["results"], json!([]), "not indexed yet");

    let done = AtomicBool::new(false);
    let index = async {
        let result = run(
            &h.ws.backend,
            &h.ws.store().await?,
            &nums(&LATE_PAPERS),
            &options(),
        )
        .await;
        done.store(true, Ordering::SeqCst);
        result
    };
    // Only answers that complete before the index run ends count, so the
    // test fails when no request overlaps the run.
    let serve = async {
        let mut during_run = 0_usize;
        while !done.load(Ordering::SeqCst) {
            h.structured("search_papers", papers_args(SearchMode::Hybrid))
                .await?;
            h.structured("get_paper_metadata", json!({"paper_id": "P2300"}))
                .await?;
            if !done.load(Ordering::SeqCst) {
                during_run += 1;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        Ok::<_, Box<dyn std::error::Error>>(during_run)
    };
    let (indexed, during_run) = tokio::join!(index, serve);
    indexed?;
    let during_run = during_run?;
    assert!(during_run >= 1, "no request completed during the index run");
    h.await_fresh_reads().await;

    let found = h.structured("search_metadata", revzin).await?;
    assert_eq!(found["results"][0]["paper_id"], "P2322R4");
    let hits = h
        .structured(
            "search_papers",
            json!({"query": QUERY, "mode": "vector", "paper_id": "P2169R4"}),
        )
        .await?;
    let results = hits["results"].as_array().ok_or("results")?;
    assert!(!results.is_empty());
    assert!(results.iter().all(|r| r["paper_id"] == "P2169R4"));
    h.stop().await
}

#[tokio::test]
async fn bad_cursors_are_tool_errors() -> TestResult {
    let expiring = CursorConfig {
        ttl: Duration::ZERO,
        ..paging()
    };
    let h = Harness::start_with(&ALL_PAPERS, expiring).await?;
    let args = papers_args(SearchMode::Keyword);
    let mut unknown = args.clone();
    unknown["cursor"] = json!("no-such-cursor.7");
    let message = h.error("search_papers", unknown).await?;
    assert!(message.contains("cursor"), "{message}");

    let (_, next) = split_page(&h.structured("search_papers", args.clone()).await?)?;
    tokio::time::sleep(Duration::from_millis(5)).await;
    let mut stale = args;
    stale["cursor"] = json!(next.ok_or("several pages")?);
    let message = h.error("search_papers", stale).await?;
    assert!(message.contains("expired"), "{message}");

    let rejected = h
        .call("search_papers", json!({"query": QUERY, "mode": "fuzzy"}))
        .await
        .map_or(true, |r| r.is_error == Some(true));
    assert!(rejected, "an unknown mode is rejected");
    let message = h
        .error(
            "search_papers",
            json!({"query": QUERY, "mode": "keyword", "paper_id": "not a paper"}),
        )
        .await?;
    assert!(message.contains("not a paper"), "{message}");
    h.stop().await
}
