//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! End-to-end tests of the `paper-mcp` binary: `index` over the fixture
//! corpus, then `serve` driven by an rmcp client over Streamable HTTP.
//!
//! The binary runs with the hidden hash test embedder so no model is
//! downloaded.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use paper_mcp::auth::KEYS_FILE_ENV;
use paper_mcp::chunk::CHUNKER_VERSION;
use paper_mcp::embed::{
    HashEmbedder, TEST_EMBEDDER_DIM, TEST_EMBEDDER_ENV, TEST_EMBEDDER_HASH, TEST_EMBEDDER_MODEL_ID,
};
use paper_mcp::index::{INDEX_DIR_NAME, Indexer, PaperIndex, REBUILD_COMMAND};
use paperstore::WORKSPACE_ENV_VAR;
use reqwest::StatusCode;
use rmcp::model::{CallToolRequestParams, CallToolResult, ClientConfig, Tool};
use rmcp::service::RunningService;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{RoleClient, ServiceExt};
use serde_json::{Map, Value, json};

use super::{open_from_env, with_fixture};

const BIN: &str = env!("CARGO_BIN_EXE_paper-mcp");
const LISTENING_MARKER: &str = "listening on ";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);
const RELOAD_TIMEOUT: Duration = Duration::from_secs(10);
const RELOAD_POLL: Duration = Duration::from_millis(100);
const KEY: &str = "alpha-key";
const ADDED_KEY: &str = "beta-key";
const TOOL_NAMES: [&str; 5] = [
    "get_paper_lines",
    "get_paper_metadata",
    "list_papers",
    "list_revisions",
    "search_papers",
];

type Client = RunningService<RoleClient, ClientConfig>;

fn block_on<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| panic!("build runtime: {e}"))
        .block_on(future)
}

fn command(workspace: &Path, args: &[&str]) -> Command {
    let mut command = Command::new(BIN);
    command
        .args(args)
        .env(WORKSPACE_ENV_VAR, workspace)
        .env(TEST_EMBEDDER_ENV, TEST_EMBEDDER_HASH)
        .env_remove(KEYS_FILE_ENV)
        .env_remove("RUST_LOG")
        .stdin(Stdio::null());
    command
}

fn output_text(output: &Output) -> String {
    format!(
        "status {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

/// Build the fixture index with `paper-mcp index`.
fn index_with_binary(workspace: &Path) {
    let output = command(workspace, &["index"])
        .output()
        .unwrap_or_else(|e| panic!("run paper-mcp index: {e}"));
    let text = output_text(&output);
    assert!(output.status.success(), "{text}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("6 embedded"), "{text}");
    assert!(stdout.contains("0 failed"), "{text}");
}

/// Build the fixture index in-process with a chosen model id and chunker
/// version.
fn index_with_library(workspace: &Path, model_id: &str, chunker_version: u32) {
    block_on(async {
        let index = PaperIndex::open(&workspace.join(INDEX_DIR_NAME))
            .await
            .unwrap_or_else(|e| panic!("open index: {e}"));
        let embedder = HashEmbedder::new(model_id, TEST_EMBEDDER_DIM);
        let backend = open_from_env();
        let report = Indexer::new(&backend, &embedder, &index)
            .with_chunker_version(chunker_version)
            .run()
            .await
            .unwrap_or_else(|e| panic!("index run: {e}"));
        assert!(report.is_success(), "{report:?}");
    });
}

fn write_keys(workspace: &Path, text: &str) -> PathBuf {
    let path = workspace.join("keys");
    fs::write(&path, text).unwrap_or_else(|e| panic!("write keys: {e}"));
    path
}

/// A running `paper-mcp serve`, killed on drop.
struct Server {
    child: Child,
    url: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn forward_lines(reader: impl std::io::Read + Send + 'static) -> Receiver<String> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(reader).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    rx
}

/// Start `paper-mcp serve` on an ephemeral loopback port.
///
/// Returns the server once it logs its URL, or everything it wrote to
/// stderr when it exits first.
fn spawn_server(workspace: &Path, extra: &[&str]) -> Result<Server, String> {
    let mut args = vec!["serve", "--host", "127.0.0.1", "--port", "0"];
    args.extend_from_slice(extra);
    let mut child = command(workspace, &args)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn paper-mcp serve: {e}"));
    let stderr = child
        .stderr
        .take()
        .unwrap_or_else(|| panic!("serve stderr not piped"));
    let lines = forward_lines(stderr);
    let mut seen = Vec::new();
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match lines.recv_timeout(remaining) {
            Ok(line) => {
                if let Some((_, url)) = line.split_once(LISTENING_MARKER) {
                    let url = url.trim().to_owned();
                    thread::spawn(move || for _ in lines {});
                    return Ok(Server { child, url });
                }
                seen.push(line);
            }
            Err(RecvTimeoutError::Disconnected) => {
                let status = child
                    .wait()
                    .unwrap_or_else(|e| panic!("wait for serve: {e}"));
                seen.push(format!("exit: {status}"));
                return Err(seen.join("\n"));
            }
            Err(RecvTimeoutError::Timeout) => {
                let _ = child.kill();
                panic!("serve did not start in time:\n{}", seen.join("\n"));
            }
        }
    }
}

fn expect_server(workspace: &Path, extra: &[&str]) -> Server {
    spawn_server(workspace, extra).unwrap_or_else(|stderr| panic!("serve failed:\n{stderr}"))
}

fn expect_startup_failure(workspace: &Path, extra: &[&str], needles: &[&str]) {
    let Err(stderr) = spawn_server(workspace, extra) else {
        panic!("serve {extra:?} should refuse to start");
    };
    assert!(!stderr.ends_with("exit: exit status: 0"), "{stderr}");
    for needle in needles {
        assert!(stderr.contains(needle), "{stderr:?} should name {needle:?}");
    }
}

/// HTTP status of an MCP `initialize` POST with an optional bearer key.
async fn initialize_status(url: &str, key: Option<&str>) -> StatusCode {
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "paper-mcp-test", "version": "0"},
        },
    });
    let mut request = reqwest::Client::new()
        .post(url)
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream")
        .body(body.to_string());
    if let Some(key) = key {
        request = request.header("authorization", format!("Bearer {key}"));
    }
    request
        .send()
        .await
        .unwrap_or_else(|e| panic!("POST {url}: {e}"))
        .status()
}

async fn connect(url: &str, key: &str) -> Client {
    let config = StreamableHttpClientTransportConfig::with_uri(url).auth_header(key);
    ClientConfig::default()
        .serve(StreamableHttpClientTransport::from_config(config))
        .await
        .unwrap_or_else(|e| panic!("connect to {url}: {e}"))
}

fn arguments(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        other => panic!("arguments must be an object, got {other}"),
    }
}

/// Call `name` and validate its structured result against the tool's
/// declared output schema.
async fn call(client: &Client, tools: &[Tool], name: &'static str, args: Value) -> Value {
    let result: CallToolResult = client
        .call_tool(CallToolRequestParams::new(name).with_arguments(arguments(args)))
        .await
        .unwrap_or_else(|e| panic!("call {name}: {e}"));
    assert_ne!(result.is_error, Some(true), "{name}: {result:?}");
    let content = result
        .structured_content
        .unwrap_or_else(|| panic!("{name} returned no structured content"));
    let tool = tools
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("{name} not listed"));
    let schema = Value::Object(
        tool.output_schema
            .as_deref()
            .cloned()
            .unwrap_or_else(|| panic!("{name} declares no output schema")),
    );
    let validator =
        jsonschema::validator_for(&schema).unwrap_or_else(|e| panic!("{name} schema: {e}"));
    if let Err(e) = validator.validate(&content) {
        panic!("{name} result does not match its output schema: {e}\n{content}");
    }
    content
}

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value
        .get(key)
        .unwrap_or_else(|| panic!("no {key:?} in {value}"))
}

#[test]
fn client_lists_and_calls_every_tool_over_streamable_http() {
    with_fixture(|workspace| {
        index_with_binary(workspace);
        let keys = write_keys(workspace, &format!("{KEY}\n"));
        let keys = keys.to_str().unwrap_or_else(|| panic!("non-UTF-8 path"));
        let server = expect_server(workspace, &["--keys-file", keys]);
        block_on(async {
            let client = connect(&server.url, KEY).await;
            let tools = client
                .list_all_tools()
                .await
                .unwrap_or_else(|e| panic!("list tools: {e}"));
            let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
            names.sort_unstable();
            assert_eq!(names, TOOL_NAMES);

            let search = call(
                &client,
                &tools,
                "search_papers",
                json!({"query": "ring buffer capacity", "search_type": "hybrid"}),
            )
            .await;
            let hits = field(&search, "hits")
                .as_array()
                .unwrap_or_else(|| panic!("hits is not an array"));
            let hit = hits.first().unwrap_or_else(|| panic!("no hits: {search}"));

            let lines = call(
                &client,
                &tools,
                "get_paper_lines",
                json!({
                    "paper_id": field(hit, "paper_id"),
                    "start": field(hit, "start_line"),
                    "end": field(hit, "end_line"),
                }),
            )
            .await;
            assert_eq!(field(&lines, "text"), field(hit, "text"));

            let metadata = call(
                &client,
                &tools,
                "get_paper_metadata",
                json!({"paper_id": "P1000R2"}),
            )
            .await;
            assert_eq!(field(field(&metadata, "paper"), "paper_id"), "P1000R2");

            let revisions = call(
                &client,
                &tools,
                "list_revisions",
                json!({"paper_id": "P1000"}),
            )
            .await;
            let revision_ids: Vec<&Value> = field(&revisions, "revisions")
                .as_array()
                .unwrap_or_else(|| panic!("revisions is not an array"))
                .iter()
                .map(|r| field(r, "paper_id"))
                .collect();
            assert_eq!(revision_ids, ["P1000R0", "P1000R1", "P1000R2"]);

            let listing = call(
                &client,
                &tools,
                "list_papers",
                json!({"revision_scope": "all"}),
            )
            .await;
            assert_eq!(field(&listing, "total"), 7);

            let _ = client.cancel().await;
        });
    });
}

#[test]
fn index_exits_nonzero_on_a_per_paper_failure() {
    with_fixture(|workspace| {
        fs::write(workspace.join("paperstore/p2000r0.md"), [0xff, 0xfe, 0xfd])
            .unwrap_or_else(|e| panic!("corrupt markdown: {e}"));
        let output = command(workspace, &["index"])
            .output()
            .unwrap_or_else(|e| panic!("run paper-mcp index: {e}"));
        let text = output_text(&output);
        assert!(!output.status.success(), "{text}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("5 embedded"), "{text}");
        assert!(stdout.contains("1 failed"), "{text}");
        assert!(stdout.contains("P2000R0"), "{text}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("1 papers failed to index"), "{text}");
    });
}

#[cfg(unix)]
#[test]
fn bearer_keys_are_enforced_and_reloaded_on_sighup() {
    with_fixture(|workspace| {
        index_with_binary(workspace);
        let keys_path = write_keys(workspace, &format!("# operators\n{KEY}\n"));
        let keys = keys_path
            .to_str()
            .unwrap_or_else(|| panic!("non-UTF-8 path"));
        let server = expect_server(workspace, &["--keys-file", keys]);
        block_on(async {
            let url = server.url.as_str();
            assert_eq!(initialize_status(url, None).await, StatusCode::UNAUTHORIZED);
            assert_eq!(
                initialize_status(url, Some("wrong")).await,
                StatusCode::UNAUTHORIZED
            );
            assert_eq!(initialize_status(url, Some(KEY)).await, StatusCode::OK);
            assert_eq!(
                initialize_status(url, Some(ADDED_KEY)).await,
                StatusCode::UNAUTHORIZED
            );

            fs::write(&keys_path, format!("# operators\n{KEY}\n{ADDED_KEY}\n"))
                .unwrap_or_else(|e| panic!("append key: {e}"));
            let status = Command::new("kill")
                .args(["-HUP", &server.child.id().to_string()])
                .status()
                .unwrap_or_else(|e| panic!("send SIGHUP: {e}"));
            assert!(status.success(), "kill -HUP: {status}");

            let deadline = Instant::now() + RELOAD_TIMEOUT;
            while initialize_status(url, Some(ADDED_KEY)).await != StatusCode::OK {
                assert!(
                    Instant::now() < deadline,
                    "added key not accepted after SIGHUP"
                );
                tokio::time::sleep(RELOAD_POLL).await;
            }
            assert_eq!(initialize_status(url, Some(KEY)).await, StatusCode::OK);
            let client = connect(url, ADDED_KEY).await;
            let _ = client.cancel().await;
        });
    });
}

#[test]
fn startup_without_keys_fails_unless_no_auth() {
    with_fixture(|workspace| {
        index_with_binary(workspace);
        let keys = write_keys(workspace, "# no keys yet\n\n");
        let keys = keys.to_str().unwrap_or_else(|| panic!("non-UTF-8 path"));
        expect_startup_failure(workspace, &["--keys-file", keys], &[keys, "--no-auth"]);

        let server = expect_server(workspace, &["--keys-file", keys, "--no-auth"]);
        block_on(async {
            assert_eq!(initialize_status(&server.url, None).await, StatusCode::OK);
        });
    });
}

#[test]
fn startup_fails_on_manifest_model_mismatch() {
    with_fixture(|workspace| {
        index_with_library(workspace, "other/model", CHUNKER_VERSION);
        expect_startup_failure(
            workspace,
            &["--no-auth"],
            &[
                "embedding model",
                "other/model",
                TEST_EMBEDDER_MODEL_ID,
                REBUILD_COMMAND,
            ],
        );
    });
}

#[test]
fn startup_fails_on_manifest_chunker_mismatch() {
    with_fixture(|workspace| {
        index_with_library(workspace, TEST_EMBEDDER_MODEL_ID, CHUNKER_VERSION + 1);
        expect_startup_failure(
            workspace,
            &["--no-auth"],
            &["chunker version", REBUILD_COMMAND],
        );
    });
}

#[test]
fn startup_fails_without_an_index() {
    with_fixture(|workspace| {
        expect_startup_failure(workspace, &["--no-auth"], &[REBUILD_COMMAND]);
    });
}
