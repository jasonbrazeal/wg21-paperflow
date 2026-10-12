//! Integration tests: ingest a fixture paperstore into a real Qdrant and
//! call every tool over MCP, plus golden snapshots of real papers.
//!
//! The MCP test needs Qdrant 1.19 or later and the BGE-M3 model cache:
//! set `PAPER_MCP_QDRANT_URL` (default `http://localhost:6334`),
//! `PAPER_MCP_QDRANT_API_KEY` when Qdrant has one, and
//! `PAPER_MCP_MODEL_CACHE_DIR`, then run
//! `cargo test -p paper-mcp --test it -- --ignored`.

mod fixture;
mod golden;

use std::sync::Arc;

use paper_mcp::config::Config;
use paper_mcp::embed::Embedder;
use paper_mcp::ingest::{Ingest, Targets};
use paper_mcp::qdrant::Store;
use paper_mcp::serve::{Keys, PaperServer, router};
use rmcp::model::CallToolRequestParams;
use rmcp::service::RunningService;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{RoleClient, ServiceExt};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

const TOKEN: &str = "integration-test-token";

fn config() -> Config {
    let mut config = Config::default();
    if let Ok(url) = std::env::var("PAPER_MCP_QDRANT_URL") {
        config.qdrant.url = url;
    }
    config.qdrant.api_key = std::env::var("PAPER_MCP_QDRANT_API_KEY").ok();
    config.qdrant.collection_prefix = format!("it_{}_", uuid::Uuid::new_v4().simple());
    config.model.cache_dir = Some(
        std::env::var("PAPER_MCP_MODEL_CACHE_DIR")
            .unwrap_or_else(|_| panic!("set PAPER_MCP_MODEL_CACHE_DIR"))
            .into(),
    );
    config
}

type Client = RunningService<RoleClient, ()>;

/// Calls a tool, returning whether it failed and its structured content.
async fn call(client: &Client, tool: &'static str, arguments: Value) -> (bool, Value) {
    let Value::Object(arguments) = arguments else {
        panic!("arguments must be an object");
    };
    let result = client
        .call_tool(CallToolRequestParams::new(tool).with_arguments(arguments))
        .await
        .unwrap_or_else(|e| panic!("{tool}: {e}"));
    (
        result.is_error.unwrap_or(false),
        result.structured_content.unwrap_or(Value::Null),
    )
}

async fn ok(client: &Client, tool: &'static str, arguments: Value) -> Value {
    let (failed, value) = call(client, tool, arguments.clone()).await;
    assert!(!failed, "{tool}({arguments}) failed: {value}");
    value
}

async fn error_code(client: &Client, tool: &'static str, arguments: Value) -> String {
    let (failed, value) = call(client, tool, arguments.clone()).await;
    assert!(failed, "{tool}({arguments}) should fail, got {value}");
    value["code"].as_str().unwrap_or_default().to_owned()
}

fn ids(value: &Value, list: &str) -> Vec<String> {
    value[list]
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|item| item["paper_id"].as_str().unwrap_or_default().to_owned())
                .collect()
        })
        .unwrap_or_default()
}

/// Every hit's text equals `get_lines` over its range.
async fn check_hits(client: &Client, page: &Value) {
    for hit in page["results"].as_array().into_iter().flatten() {
        let lines = ok(
            client,
            "get_lines",
            json!({"paper": hit["paper_id"], "first": hit["first_line"], "last": hit["last_line"]}),
        )
        .await;
        assert_eq!(
            hit["text"], lines["text"],
            "hit text differs from get_lines: {hit}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Qdrant 1.19+ and the BGE-M3 model cache"]
async fn ingest_and_call_every_tool() {
    let workspace = fixture::workspace();
    let config = config();
    let store = Store::connect(&config).unwrap_or_else(|e| panic!("connect: {e}"));
    let embedder = Embedder::start(
        config.model_cache_dir().unwrap_or_else(|e| panic!("{e}")),
        config.model.max_length,
        config.model.batch_tokens,
        config.model.queue_depth,
    )
    .unwrap_or_else(|e| panic!("model: {e}"));

    let report = Ingest::new(&config, &store, &embedder, false)
        .run(workspace.path(), &Targets::All)
        .await
        .unwrap_or_else(|e| panic!("ingest: {e}"));
    assert_eq!((report.indexed, report.skipped, report.failed), (4, 0, 0));
    assert_eq!(report.no_markdown, vec!["p1000r0".to_owned()]);
    let again = Ingest::new(&config, &store, &embedder, false)
        .run(workspace.path(), &Targets::All)
        .await
        .unwrap_or_else(|e| panic!("ingest: {e}"));
    assert_eq!((again.indexed, again.skipped, again.failed), (0, 4, 0));

    let keys_path = workspace.path().join("keys");
    std::fs::write(&keys_path, format!("# test\n{TOKEN}\n")).unwrap_or_else(|e| panic!("{e}"));
    let keys = Keys::load(&keys_path).unwrap_or_else(|e| panic!("{e}"));
    let shutdown = CancellationToken::new();
    let server = PaperServer::new(store.clone(), embedder.clone(), config.search);
    let app = router(&config, server, Arc::clone(&keys), shutdown.child_token());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap_or_else(|e| panic!("bind: {e}"));
    let url = format!(
        "http://127.0.0.1:{}{}",
        listener.local_addr().map(|a| a.port()).unwrap_or_default(),
        config.serve.path
    );
    let stop = shutdown.clone();
    let serving = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move { stop.cancelled().await })
            .await
    });

    let unauthorized = reqwest::Client::new()
        .post(&url)
        .header("content-type", "application/json")
        .body("{}")
        .send()
        .await
        .unwrap_or_else(|e| panic!("post: {e}"));
    assert_eq!(unauthorized.status().as_u16(), 401);

    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(url.as_str()).auth_header(TOKEN),
    );
    let client = ().serve(transport).await.unwrap_or_else(|e| panic!("client: {e}"));

    let tools = client
        .list_all_tools()
        .await
        .unwrap_or_else(|e| panic!("list tools: {e}"));
    assert_eq!(tools.len(), 8);
    assert!(tools.iter().all(|t| t.output_schema.is_some()));

    check_metadata(&client).await;
    check_text(&client).await;
    check_revisions_and_citations(&client).await;
    check_search(&client).await;
    check_search_metadata(&client).await;

    client.cancel().await.ok();
    shutdown.cancel();
    serving.await.ok();
    store
        .drop_collections()
        .await
        .unwrap_or_else(|e| panic!("drop: {e}"));
}

async fn check_metadata(client: &Client) {
    let meta = ok(client, "get_metadata", json!({"paper": "P1000R1"})).await;
    assert_eq!(meta["paper_id"], "p1000r1");
    assert_eq!(meta["authors"], json!(["Eric Niebler", "Murat Can Çağrı"]));
    assert_eq!(meta["document_date"], "2024-05-01");
    assert_eq!(meta["mailing_date"], "2024-05");
    assert_eq!(meta["previous_version"], "p1000r0");
    assert_eq!(meta["has_markdown"], true);
    assert_eq!(meta["total_lines"], 31);

    let bad_date = ok(client, "get_metadata", json!({"paper": "p3000r0"})).await;
    assert_eq!(bad_date["document_date"], "");

    let no_md = ok(client, "get_metadata", json!({"paper": "p1000r0"})).await;
    assert_eq!(no_md["has_markdown"], false);
    assert!(no_md.get("total_lines").is_none());

    for bad in [
        " p1000r1",
        "p1000r1.md",
        "https://wg21.link/p1000r1",
        "p1000",
    ] {
        assert_eq!(
            error_code(client, "get_metadata", json!({"paper": bad})).await,
            "invalid_paper_id"
        );
    }
    assert_eq!(
        error_code(client, "get_metadata", json!({"paper": "p9999r9"})).await,
        "paper_not_indexed"
    );
}

async fn check_text(client: &Client) {
    let whole = ok(client, "get_lines", json!({"paper": "p1000r1"})).await;
    let expected = fixture::P1000R1
        .strip_suffix('\n')
        .unwrap_or(fixture::P1000R1);
    assert_eq!(whole["text"], expected);
    assert_eq!(
        (whole["first"].as_u64(), whole["last"].as_u64()),
        (Some(1), Some(31))
    );

    let tail = ok(
        client,
        "get_lines",
        json!({"paper": "p1000r1", "first": -2}),
    )
    .await;
    assert_eq!(tail["text"], "\nThanks to everyone.");
    let empty = ok(
        client,
        "get_lines",
        json!({"paper": "p1000r1", "first": 40}),
    )
    .await;
    assert_eq!(empty["text"], "");

    let sections = ok(client, "get_sections", json!({"paper": "p1000r1"})).await;
    assert_eq!(sections["total_sections"], 5);
    let joined: Vec<&str> = sections["sections"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| s["text"].as_str())
        .collect();
    assert_eq!(joined.join("\n"), expected);
    let last = ok(
        client,
        "get_sections",
        json!({"paper": "p1000r1", "first": -1}),
    )
    .await;
    assert_eq!(last["sections"][0]["heading"], "4 Acknowledgements");

    let outline = ok(client, "get_outline", json!({"paper": "p1000r1"})).await;
    let headings: Vec<(u64, String, u64, bool)> = outline["headings"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|h| {
            (
                h["line"].as_u64().unwrap_or_default(),
                h["text"].as_str().unwrap_or_default().to_owned(),
                h["section_index"].as_u64().unwrap_or_default(),
                h["starts_section"].as_bool().unwrap_or_default(),
            )
        })
        .collect();
    assert_eq!(
        headings,
        vec![
            (6, "1 Motivation".to_owned(), 2, true),
            (11, "2 Design".to_owned(), 3, true),
            (13, "2.1 Units of measure".to_owned(), 3, false),
            (22, "3 Wording".to_owned(), 4, true),
            (29, "4 Acknowledgements".to_owned(), 5, true),
        ]
    );

    for tool in ["get_lines", "get_sections", "get_outline"] {
        assert_eq!(
            error_code(client, tool, json!({"paper": "p1000r0"})).await,
            "markdown_not_converted"
        );
    }
}

async fn check_revisions_and_citations(client: &Client) {
    let chain = ok(client, "list_revisions", json!({"paper": "P1000"})).await;
    assert_eq!(
        ids(&chain, "revisions"),
        vec!["p1000r0", "p1000r1", "p3000r0"]
    );
    let from_end = ok(client, "list_revisions", json!({"paper": "p3000r0"})).await;
    assert_eq!(
        ids(&from_end, "revisions"),
        vec!["p1000r0", "p1000r1", "p3000r0"]
    );
    let single = ok(client, "list_revisions", json!({"paper": "n5000"})).await;
    assert_eq!(ids(&single, "revisions"), vec!["n5000"]);
    assert_eq!(
        error_code(client, "list_revisions", json!({"paper": "p4242"})).await,
        "paper_not_indexed"
    );

    let both = ok(
        client,
        "get_citations",
        json!({"paper": "p1000r1", "direction": "both"}),
    )
    .await;
    assert_eq!(
        both["cites"],
        json!([{"paper_id": "p2000r0", "count": 3}, {"paper_id": "d0263r1", "count": 1}])
    );
    assert_eq!(
        both["cited_by"],
        json!([{"paper_id": "p3000r0", "count": 2}])
    );
    let none = ok(
        client,
        "get_citations",
        json!({"paper": "p1000r0", "direction": "cites"}),
    )
    .await;
    assert_eq!(none["citations_extracted"], false);
    assert!(none.get("cites").is_none());
    assert_eq!(
        error_code(
            client,
            "get_citations",
            json!({"paper": "p1000r1", "direction": "up"})
        )
        .await,
        "invalid_argument"
    );
}

async fn check_search(client: &Client) {
    let fulltext = ok(
        client,
        "search",
        json!({"query": "frobnicate", "type": "fulltext"}),
    )
    .await;
    let hits = fulltext["results"].as_array().cloned().unwrap_or_default();
    assert!(!hits.is_empty());
    for hit in &hits {
        assert!(
            !hit["match_lines"].as_array().is_none_or(Vec::is_empty),
            "fulltext hit without match_lines: {hit}"
        );
    }
    check_hits(client, &fulltext).await;

    for kind in ["vector", "hybrid"] {
        let page = ok(
            client,
            "search",
            json!({"query": "units of measure", "type": kind}),
        )
        .await;
        assert!(
            !page["results"].as_array().is_none_or(Vec::is_empty),
            "{kind}"
        );
        check_hits(client, &page).await;
    }

    let first = ok(
        client,
        "search",
        json!({"query": "frobnication", "type": "hybrid", "limit": 1}),
    )
    .await;
    let cursor = first["nextCursor"].as_str().unwrap_or_default().to_owned();
    assert!(!cursor.is_empty());
    let second = ok(
        client,
        "search",
        json!({"query": "frobnication", "type": "hybrid", "limit": 1, "cursor": cursor}),
    )
    .await;
    assert_ne!(first["results"][0], second["results"][0]);
    assert_eq!(
        error_code(
            client,
            "search",
            json!({"query": "other", "type": "hybrid", "limit": 1, "cursor": cursor})
        )
        .await,
        "invalid_argument"
    );

    let by_author = ok(
        client,
        "search",
        json!({"query": "frobnication", "type": "fulltext", "author": "CAGRI"}),
    )
    .await;
    assert!(ids(&by_author, "results").iter().all(|id| id == "p1000r1"));
    let scoped = ok(
        client,
        "search",
        json!({"query": "frobnicate", "type": "fulltext", "papers": ["p1000r0", "P1000R1"]}),
    )
    .await;
    assert_eq!(scoped["no_markdown"], json!(["p1000r0"]));
    let dated = ok(
        client,
        "search",
        json!({"query": "frobnication", "type": "fulltext", "start": "2024-01-01"}),
    )
    .await;
    assert!(ids(&dated, "results").iter().all(|id| id == "p1000r1"));
    let by_mailing = ok(
        client,
        "search",
        json!({"query": "frobnication", "type": "fulltext", "start": "2025-01-01", "date_field": "mailing_date"}),
    )
    .await;
    assert_eq!(ids(&by_mailing, "results"), vec!["p3000r0"]);

    for args in [
        json!({"query": "x"}),
        json!({"type": "fulltext"}),
        json!({"query": "x", "type": "semantic"}),
        json!({"query": "x", "type": "fulltext", "start": "2024-1-1"}),
        json!({"query": "x", "type": "fulltext", "cursor": "garbage"}),
    ] {
        assert_eq!(error_code(client, "search", args).await, "invalid_argument");
    }
}

async fn check_search_metadata(client: &Client) {
    let ranked = ok(client, "search_metadata", json!({"query": "Niebler"})).await;
    let found = ids(&ranked, "results");
    assert!(found.contains(&"p1000r1".to_owned()) && found.contains(&"p1000r0".to_owned()));
    assert!(ranked["results"][0]["score"].is_number());

    let newest = ok(
        client,
        "search_metadata",
        json!({"start": "2024-01-01", "date_field": "mailing_date"}),
    )
    .await;
    assert_eq!(
        ids(&newest, "results"),
        vec!["p3000r0", "p1000r1", "p2000r0"]
    );
    assert!(newest["results"][0].get("score").is_none());

    let paged = ok(
        client,
        "search_metadata",
        json!({"author": "azman", "limit": 1}),
    )
    .await;
    assert_eq!(ids(&paged, "results"), vec!["p2000r0"]);
    assert!(paged.get("nextCursor").is_none());

    assert_eq!(
        error_code(client, "search_metadata", json!({})).await,
        "invalid_argument"
    );
}
