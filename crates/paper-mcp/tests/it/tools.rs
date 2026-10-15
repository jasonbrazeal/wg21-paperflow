// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! The read tools, called through an rmcp Streamable HTTP client against a
//! server on an ephemeral port over the fixture workspace.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use paper_mcp::model::fake::{HashEmbedder, IdentityReranker};
use paper_mcp::server::config::default_allowed_hosts;
use paper_mcp::{
    BearerToken, CursorConfig, MCP_PATH, PaperServer, SearchIndex, SearchParams, ServeConfig,
    ServeOptions, build_app, outline, slice_lines,
};
use paperstore_sqlite::SqliteBackend;
use rmcp::ServiceExt as _;
use rmcp::model::{CallToolRequestParams, CallToolResult, ClientConfig};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::ingest::{Workspace, nums, options, run};

type TestResult = Result<(), Box<dyn std::error::Error>>;

pub(crate) const SECRET: &str = "read-tools-token";

/// How stale the served index may be. Short, but not zero, so that the
/// server takes the same periodic freshness check as the production
/// default instead of checking on every read.
const SERVE_READ_CONSISTENCY_INTERVAL: Duration = Duration::from_millis(50);

const TOOLS: [&str; 8] = [
    "diff_revisions",
    "get_paper",
    "get_paper_lines",
    "get_paper_metadata",
    "get_paper_outline",
    "list_revisions",
    "search_metadata",
    "search_papers",
];

/// A running server over a fixture workspace, and a connected client.
pub(crate) struct Harness {
    pub(crate) ws: Workspace,
    pub(crate) addr: SocketAddr,
    client: RunningService<RoleClient, ClientConfig>,
    shutdown: CancellationToken,
}

impl Harness {
    async fn start() -> Result<Self, Box<dyn std::error::Error>> {
        Self::start_with(&[], CursorConfig::default()).await
    }

    /// Index `papers` with the test-double models, then serve the workspace
    /// with `cursor` paging limits from an index opened read-only, as
    /// `serve` opens it.
    pub(crate) async fn start_with(
        papers: &[&str],
        cursor: CursorConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ws = Workspace::new()?;
        let writer = ws.store().await?;
        if !papers.is_empty() {
            run(&ws.backend, &writer, &nums(papers), &options()).await?;
        }
        let index = SearchIndex {
            store: ws.read_only_store(SERVE_READ_CONSISTENCY_INTERVAL).await?,
            embedder: Arc::new(HashEmbedder::default()),
            reranker: Arc::new(IdentityReranker),
            params: SearchParams::default(),
        };
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let shutdown = CancellationToken::new();
        let server = PaperServer::new(SqliteBackend::new(ws.path())?, index, cursor);
        let app = build_app(&config(addr, &ws)?, server, shutdown.clone());
        tokio::spawn(
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown.clone().cancelled_owned())
                .into_future(),
        );
        let transport = StreamableHttpClientTransport::from_config(
            StreamableHttpClientTransportConfig::with_uri(format!("http://{addr}{MCP_PATH}"))
                .auth_header(SECRET),
        );
        let client = ClientConfig::default().serve(transport).await?;
        Ok(Self {
            ws,
            addr,
            client,
            shutdown,
        })
    }

    pub(crate) async fn call(
        &self,
        tool: &str,
        args: Value,
    ) -> Result<CallToolResult, Box<dyn std::error::Error>> {
        let Value::Object(args) = args else {
            return Err("tool arguments must be an object".into());
        };
        let params = CallToolRequestParams::new(tool.to_owned()).with_arguments(args);
        Ok(self.client.call_tool(params).await?)
    }

    /// Call `tool`, requiring success, and return its text content.
    pub(crate) async fn text(
        &self,
        tool: &str,
        args: Value,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let result = self.call(tool, args).await?;
        let text = text_of(&result)?;
        if result.is_error == Some(true) {
            return Err(format!("{tool} failed: {text}").into());
        }
        Ok(text)
    }

    /// Call `tool`, requiring success, and return its structured content.
    pub(crate) async fn structured(
        &self,
        tool: &str,
        args: Value,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let result = self.call(tool, args).await?;
        if result.is_error == Some(true) {
            return Err(format!("{tool} failed: {}", text_of(&result)?).into());
        }
        result
            .structured_content
            .ok_or_else(|| format!("{tool} returned no structured content").into())
    }

    /// Call `tool`, requiring a tool error, and return its message.
    pub(crate) async fn error(
        &self,
        tool: &str,
        args: Value,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let result = self.call(tool, args.clone()).await?;
        let text = text_of(&result)?;
        if result.is_error != Some(true) {
            return Err(format!("{tool} {args} should fail, got {text}").into());
        }
        Ok(text)
    }

    fn markdown(&self, paper: &str) -> std::io::Result<String> {
        std::fs::read_to_string(self.ws.markdown(paper))
    }

    /// Wait until the server's next read checks for new index runs.
    pub(crate) async fn await_fresh_reads(&self) {
        tokio::time::sleep(SERVE_READ_CONSISTENCY_INTERVAL * 2).await;
    }

    pub(crate) async fn stop(self) -> TestResult {
        self.client.cancel().await?;
        self.shutdown.cancel();
        Ok(())
    }
}

fn config(bind: SocketAddr, ws: &Workspace) -> Result<ServeConfig, paper_mcp::Error> {
    let defaults = ServeOptions::default();
    Ok(ServeConfig {
        bind,
        token: BearerToken::new(SECRET)?,
        workspace: ws.path().to_path_buf(),
        tool_dir: ws.path().join("paper-mcp"),
        model_dir: "/nonexistent/models".into(),
        allowed_hosts: default_allowed_hosts(),
        top_k: defaults.top_k,
        rerank_depth: defaults.rerank_depth,
        cursor: CursorConfig::default(),
    })
}

pub(crate) fn text_of(result: &CallToolResult) -> Result<String, Box<dyn std::error::Error>> {
    let texts: Vec<&str> = result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
        .collect();
    match texts.as_slice() {
        [one] => Ok((*one).to_owned()),
        _ => Err(format!("expected one text block, got {:?}", result.content).into()),
    }
}

#[tokio::test]
async fn lists_exactly_the_eight_tools() -> TestResult {
    let h = Harness::start().await?;
    let mut names: Vec<String> = h
        .client
        .list_all_tools()
        .await?
        .into_iter()
        .map(|t| t.name.to_string())
        .collect();
    names.sort();
    assert_eq!(names, TOOLS);
    h.stop().await
}

#[tokio::test]
async fn get_paper_returns_the_whole_markdown() -> TestResult {
    let h = Harness::start().await?;
    let got = h.text("get_paper", json!({"paper_id": "p2300"})).await?;
    assert_eq!(
        got,
        h.markdown("P2300R10")?,
        "unversioned id resolves to the latest"
    );
    let got = h.text("get_paper", json!({"paper_id": "P2169R4"})).await?;
    assert_eq!(got, h.markdown("P2169R4")?, "large paper is not capped");

    let message = h.error("get_paper", json!({"paper_id": "P9999"})).await?;
    assert!(message.contains("P9999"), "{message}");
    let message = h.error("get_paper", json!({"paper_id": "N4950"})).await?;
    assert!(message.contains("N4950"), "no markdown: {message}");
    let message = h
        .error("get_paper", json!({"paper_id": "not a paper"}))
        .await?;
    assert!(message.contains("not a paper"), "{message}");
    h.stop().await
}

#[tokio::test]
async fn get_paper_lines_slices_like_python() -> TestResult {
    let h = Harness::start().await?;
    let md = h.markdown("P2169R4")?;
    for (start, end) in [
        (Some(1), Some(4)),
        (Some(-5), None),
        (None, Some(-3)),
        (Some(10), Some(5)),
    ] {
        let want = slice_lines(&md, start, end)?;
        let got = h
            .structured(
                "get_paper_lines",
                json!({"paper_id": "p2169r4", "start": start, "end": end}),
            )
            .await?;
        assert_eq!(
            got,
            json!({
                "paper_id": "P2169R4",
                "start": want.start,
                "end": want.end,
                "total_lines": want.total_lines,
                "text": want.text,
            }),
            "start={start:?} end={end:?}"
        );
    }
    let got = h
        .structured(
            "get_paper_lines",
            json!({"paper_id": "P2300", "start": 1, "end": 2}),
        )
        .await?;
    assert_eq!(got["paper_id"], "P2300R10");

    for args in [
        json!({"paper_id": "P2169R4", "start": 0, "end": 3}),
        json!({"paper_id": "P2169R4"}),
    ] {
        let message = h.error("get_paper_lines", args).await?;
        assert!(message.contains("invalid line range"), "{message}");
    }
    let message = h
        .error(
            "get_paper_lines",
            json!({"paper_id": "P9999R0", "start": 1}),
        )
        .await?;
    assert!(message.contains("P9999R0"), "{message}");
    h.stop().await
}

#[tokio::test]
async fn get_paper_metadata_returns_the_full_row() -> TestResult {
    let h = Harness::start().await?;
    let got = h
        .structured("get_paper_metadata", json!({"paper_id": "P2300"}))
        .await?;
    let path = h.ws.path().to_str().ok_or("temp dir is not UTF-8")?;
    let expected = json!({
        "paper_id": "P2300R10",
        "year": "2024",
        "title": "`std::execution`",
        "authors": ["Michal Dominiak", "Eric Niebler"],
        "target_group": "LEWG Library Evolution",
        "intent": "info",
        "url": "https://wg21.link/p2300r10",
        "document_date": "2024-06-28",
        "mailing_date": "2024-07",
        "previous_version": "p2300r9",
        "source_file": format!("{path}/paperstore/p2300r10.html"),
        "markdown_path": format!("{path}/paperstore/p2300r10.md"),
    });
    for (key, value) in expected.as_object().ok_or("object")? {
        assert_eq!(&got[key], value, "{key}");
    }
    for key in [
        "disposition",
        "dissect_path",
        "agora_path",
        "assay_path",
        "citations_extracted_at",
        "error",
        "line_count",
        "status",
    ] {
        assert!(got.get(key).is_some(), "metadata lacks {key}: {got}");
    }
    let message = h
        .error("get_paper_metadata", json!({"paper_id": "P9999"}))
        .await?;
    assert!(message.contains("P9999"), "{message}");
    h.stop().await
}

#[tokio::test]
async fn list_revisions_is_newest_first() -> TestResult {
    let h = Harness::start().await?;
    let got = h
        .structured("list_revisions", json!({"paper_id": "P2300R4"}))
        .await?;
    let revisions = got["revisions"].as_array().ok_or("revisions array")?;
    let ids: Vec<&str> = revisions
        .iter()
        .filter_map(|r| r["paper_id"].as_str())
        .collect();
    let want: Vec<String> = (0..=10).rev().map(|r| format!("P2300R{r}")).collect();
    assert_eq!(ids, want);
    assert_eq!(
        revisions[0],
        json!({
            "paper_id": "P2300R10",
            "date": "2024-06-28",
            "mailing": "2024-07",
            "previous_version": "p2300r9",
        })
    );
    assert_eq!(revisions[10]["previous_version"], "");

    let got = h
        .structured("list_revisions", json!({"paper_id": "P2169"}))
        .await?;
    assert_eq!(got["revisions"].as_array().map(Vec::len), Some(1));
    let message = h
        .error("list_revisions", json!({"paper_id": "P9999"}))
        .await?;
    assert!(message.contains("P9999"), "{message}");
    h.stop().await
}

#[tokio::test]
async fn get_paper_outline_matches_the_chunker_outline() -> TestResult {
    let h = Harness::start().await?;
    let md = h.markdown("P2322R4")?;
    let want: Vec<Value> = outline(&md)
        .into_iter()
        .map(|s| json!({"level": s.level, "heading": s.heading, "start": s.start, "end": s.end}))
        .collect();
    assert!(!want.is_empty());
    let got = h
        .structured("get_paper_outline", json!({"paper_id": "p2322r4"}))
        .await?;
    assert_eq!(got, json!({"paper_id": "P2322R4", "sections": want}));

    let message = h
        .error("get_paper_outline", json!({"paper_id": "N4950"}))
        .await?;
    assert!(message.contains("N4950"), "{message}");
    h.stop().await
}

#[tokio::test]
async fn diff_revisions_gives_a_unified_diff() -> TestResult {
    let h = Harness::start().await?;
    let got = h
        .text(
            "diff_revisions",
            json!({"from_id": "P2300R9", "to_id": "p2300"}),
        )
        .await?;
    assert!(got.starts_with("--- P2300R9\n+++ P2300R10\n@@ "), "{got}");
    let (old, new) = (h.markdown("P2300R9")?, h.markdown("P2300R10")?);
    let removed = got
        .lines()
        .filter(|l| l.starts_with('-') && !l.starts_with("---"));
    let added = got
        .lines()
        .filter(|l| l.starts_with('+') && !l.starts_with("+++"));
    assert!(removed.clone().count() + added.clone().count() > 0, "{got}");
    for line in removed {
        assert!(old.lines().any(|l| l == &line[1..]), "{line}");
    }
    for line in added {
        assert!(new.lines().any(|l| l == &line[1..]), "{line}");
    }
    let same = h
        .text(
            "diff_revisions",
            json!({"from_id": "P2300R9", "to_id": "P2300R9"}),
        )
        .await?;
    assert_eq!(same, "");

    let message = h
        .error(
            "diff_revisions",
            json!({"from_id": "P2300R9", "to_id": "P2169R4"}),
        )
        .await?;
    assert!(
        message.contains("P2300R9")
            && message.contains("P2169R4")
            && message.contains("same paper"),
        "{message}"
    );
    let message = h
        .error(
            "diff_revisions",
            json!({"from_id": "P2300R9", "to_id": "P2300R42"}),
        )
        .await?;
    assert!(message.contains("P2300R42"), "{message}");
    h.stop().await
}
