// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! The authenticated MCP HTTP app, and token secrecy in its logs.

use std::io::Write;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError};

use axum::Router;
use axum::body::Body;
use axum::http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HOST};
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt as _;
use paper_mcp::model::fake::{HashEmbedder, IdentityReranker};
use paper_mcp::server::config::{TOKEN_ENV, default_allowed_hosts};
use paper_mcp::{
    BearerToken, CursorConfig, Embedder as _, LanceStore, MCP_PATH, PaperServer, SearchIndex,
    SearchParams, ServeConfig, ServeOptions, build_app,
};
use paperstore::{PaperNum, PaperRow, StorageBackend};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt as _;
use tracing_subscriber::fmt::MakeWriter;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const SECRET: &str = "s3cret-Token_value";

const INIT_BODY: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#;

/// A paperstore with no papers, for tests that call no tools.
struct NoPapers;

impl StorageBackend for NoPapers {
    fn meta(&self, num: &PaperNum) -> paperstore::Result<PaperRow> {
        Err(paperstore::Error::missing_paper(num.clone()))
    }

    fn paper_md(&self, num: &PaperNum) -> paperstore::Result<String> {
        Err(paperstore::Error::missing_markdown(num.clone()))
    }

    fn list_all_paper_ids(&self) -> paperstore::Result<Vec<PaperNum>> {
        Ok(Vec::new())
    }

    fn list_paper_ids_for_year(&self, _year: u16) -> paperstore::Result<Vec<PaperNum>> {
        Ok(Vec::new())
    }

    fn list_revisions(&self, _num: &PaperNum) -> paperstore::Result<Vec<PaperRow>> {
        Ok(Vec::new())
    }

    fn tool_dir(&self, tool: &str) -> PathBuf {
        PathBuf::from(tool)
    }
}

fn config(bind: SocketAddr) -> Result<ServeConfig, paper_mcp::Error> {
    let defaults = ServeOptions::default();
    Ok(ServeConfig {
        bind,
        token: BearerToken::new(SECRET)?,
        workspace: "/nonexistent/workspace".into(),
        tool_dir: "/nonexistent/workspace/paper-mcp".into(),
        model_dir: "/nonexistent/models".into(),
        allowed_hosts: default_allowed_hosts(),
        top_k: defaults.top_k,
        rerank_depth: defaults.rerank_depth,
        cursor: CursorConfig::default(),
    })
}

/// A server with no papers over an empty index in `dir`.
async fn server(dir: &TempDir) -> Result<PaperServer, paper_mcp::Error> {
    let index = SearchIndex {
        store: LanceStore::open(dir.path(), HashEmbedder::default().dim()).await?,
        embedder: Arc::new(HashEmbedder::default()),
        reranker: Arc::new(IdentityReranker),
        params: SearchParams::default(),
    };
    Ok(PaperServer::new(NoPapers, index, CursorConfig::default()))
}

async fn app(dir: &TempDir) -> Result<Router, paper_mcp::Error> {
    Ok(build_app(
        &config(ServeOptions::default().bind)?,
        server(dir).await?,
        CancellationToken::new(),
    ))
}

async fn post(
    path: &str,
    authorization: Option<&str>,
) -> Result<(StatusCode, String), Box<dyn std::error::Error>> {
    let mut request = Request::post(path)
        .header(HOST, "localhost")
        .header(CONTENT_TYPE, "application/json")
        .header(ACCEPT, "application/json, text/event-stream");
    if let Some(value) = authorization {
        request = request.header(AUTHORIZATION, value);
    }
    let dir = TempDir::new()?;
    let response = app(&dir)
        .await?
        .oneshot(request.body(Body::from(INIT_BODY))?)
        .await?;
    let status = response.status();
    let body = response.into_body().collect().await?.to_bytes();
    Ok((status, String::from_utf8_lossy(&body).into_owned()))
}

#[tokio::test]
async fn mcp_endpoint_requires_the_token() -> TestResult {
    let bearer = format!("Bearer {SECRET}");
    let wrong = format!("Bearer {SECRET}-not");
    for authorization in [
        None,
        Some("Bearer"),
        Some(wrong.as_str()),
        Some("Basic Zm9vOmJhcg=="),
    ] {
        let (status, body) = post(MCP_PATH, authorization).await?;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{authorization:?}");
        assert!(!body.contains(SECRET));
    }
    let (status, _) = post("/elsewhere", None).await?;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "every route is guarded");

    let (status, body) = post(MCP_PATH, Some(&bearer)).await?;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.contains("serverInfo"), "initialize result: {body}");
    assert!(body.contains("\"tools\""), "tools capability: {body}");
    assert!(!body.contains("\"resources\""), "no resources: {body}");
    assert!(!body.contains("\"prompts\""), "no prompts: {body}");
    Ok(())
}

/// Log sink shared between a subscriber and the test.
#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Captured {
    fn text(&self) -> String {
        let bytes = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl Write for Captured {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for Captured {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// Set in the child process that runs the log-capture scenario alone.
const LOG_CHILD_ENV: &str = "PAPER_MCP_TOKEN_LOG_CHILD";

/// Libtest name of [`token_never_appears_in_logs`] inside this binary.
const LOG_TEST_NAME: &str = "server::token_never_appears_in_logs";

// tracing caches each callsite's interest process-wide. A concurrent test
// that first hits a callsite on a thread without a subscriber can race
// `set_default` here and cache `never`, silently dropping the event. The
// scenario therefore runs in a re-executed test binary where it is the only
// test.
#[test]
fn token_never_appears_in_logs() -> TestResult {
    if std::env::var_os(LOG_CHILD_ENV).is_some() {
        return tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(token_log_scenario());
    }
    let output = Command::new(std::env::current_exe()?)
        .args([LOG_TEST_NAME, "--exact", "--test-threads=1", "--nocapture"])
        .env(LOG_CHILD_ENV, "1")
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed"),
        "log scenario failed in child process ({}):\n{stdout}\n{stderr}",
        output.status
    );
    Ok(())
}

// The current-thread runtime lets the thread-local subscriber see every
// task's events.
async fn token_log_scenario() -> TestResult {
    let logs = Captured::default();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(logs.clone())
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let bearer = format!("Bearer {SECRET}");
    let near_miss = format!("Bearer {SECRET}x");
    for authorization in [None, Some(near_miss.as_str()), Some(bearer.as_str())] {
        post(MCP_PATH, authorization).await?;
    }

    // Startup failure after the configuration is logged: the port is taken.
    let taken = TcpListener::bind("127.0.0.1:0").await?;
    let config = config(taken.local_addr()?)?;
    let dir = TempDir::new()?;
    let result = paper_mcp::serve(&config, server(&dir).await?, CancellationToken::new()).await;
    let error = result.err().ok_or("binding a taken port should fail")?;
    tracing::error!(%error, ?error, "startup failed");

    // Startup validation failure with the token set.
    let workspace = TempDir::new()?;
    let error = temp_env::with_vars(
        [
            (TOKEN_ENV, Some(SECRET)),
            ("WG21_DATA_DIR", workspace.path().to_str()),
        ],
        || ServeConfig::load(ServeOptions::default()),
    )
    .err()
    .ok_or("a workspace without a database should fail")?;
    tracing::error!(%error, ?error, "startup failed");

    let text = logs.text();
    for expected in [
        "rejected unauthenticated request",
        "starting MCP server",
        "cannot listen on",
        "no paperstore database",
        "[redacted]",
    ] {
        assert!(text.contains(expected), "logs lack {expected:?}:\n{text}");
    }
    assert!(!text.contains(SECRET), "token in logs:\n{text}");
    Ok(())
}
