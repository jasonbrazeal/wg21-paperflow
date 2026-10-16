#
# Copyright (c) 2026 Jason Brazeal
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#

"""Run cpp-mcp and paper-mcp side by side on one keys file.

Starts both servers on distinct loopback ports with a single shared keys
file, as `crates/paper-mcp/docs/server-setup.md` deploys them, and checks
that one key authenticates against both and an unknown key is rejected by
both.

The `paper-mcp` binary is taken from `$PAPER_MCP_BIN`, else from
`debug/` under the target directory Cargo reports (honoring
`$CARGO_TARGET_DIR` and `build.target-dir` config), falling back to the
repository's `target/` when `cargo` is unavailable. The test is skipped
when the binary is absent; build it with
`cargo build -p paper-mcp --locked`. It runs with the hash test embedder
against the crate's fixture corpus, so no model is downloaded.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.shared.exceptions import McpError

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "crates" / "paper-mcp" / "tests" / "fixtures"
PAPER_MCP_BIN_ENV = "PAPER_MCP_BIN"
PAPER_MCP_EXE = "paper-mcp.exe" if os.name == "nt" else "paper-mcp"

HOST = "127.0.0.1"
MCP_PATH = "/mcp"
SHARED_KEY = "shared-coexist-key"
UNKNOWN_KEY = "unknown-coexist-key"

CARGO_METADATA_TIMEOUT_S = 60
INDEX_TIMEOUT_S = 300
STARTUP_TIMEOUT_S = 120
SHUTDOWN_TIMEOUT_S = 10
CONNECT_POLL_S = 0.1
CLIENT_TIMEOUT_S = 30

PAPER_MCP_TOOL = "search_papers"
CPP_MCP_TOOL = "lookup_section"
LISTENING_RE = re.compile(r"listening on (http://\S+)")
HTTP_UNAUTHORIZED_RE = re.compile(r"\b401\b")
# cpp-mcp's middleware rejects during `initialize`; FastMCP masks the reason
# as -32602 on the wire and logs it server-side.
CPP_MCP_REJECTION_LOG = "Unauthorized: invalid API key"


def _cargo_target_dir() -> Path:
    cargo = shutil.which("cargo")
    if cargo is None:
        return Path(os.environ.get("CARGO_TARGET_DIR", "").strip() or REPO_ROOT / "target")
    metadata = subprocess.run(
        [cargo, "metadata", "--no-deps", "--format-version", "1", "--locked"],
        cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=CARGO_METADATA_TIMEOUT_S,
    )
    if metadata.returncode != 0:
        pytest.fail(f"cargo metadata failed:\n{metadata.stderr}")
    return Path(json.loads(metadata.stdout)["target_directory"])


def _paper_mcp_bin() -> Path:
    configured = os.environ.get(PAPER_MCP_BIN_ENV, "").strip()
    if configured:
        path = Path(configured)
        if not path.is_file():
            pytest.fail(f"${PAPER_MCP_BIN_ENV} points at a missing file: {path}")
        return path
    default = _cargo_target_dir() / "debug" / PAPER_MCP_EXE
    if not default.is_file():
        pytest.skip(
            f"paper-mcp binary not built ({default}); "
            f"run `cargo build -p paper-mcp --locked` or set ${PAPER_MCP_BIN_ENV}"
        )
    return default


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in ("CPP_MCP_KEYS_FILE", "CPP_MCP_DATA_DIR", "CPP_MCP_DEFAULT_DRAFT", "RUST_LOG"):
        env.pop(name, None)
    return env


def _paper_mcp_env(workspace: Path) -> dict[str, str]:
    env = _base_env()
    env["WG21_DATA_DIR"] = str(workspace)
    env["PAPER_MCP_TEST_EMBEDDER"] = "hash"
    return env


def _stage_fixture(workspace: Path) -> None:
    """Copy the fixture corpus and make its artifact paths absolute."""
    workspace.mkdir(parents=True)
    shutil.copy(FIXTURE_DIR / "paperstore.db", workspace / "paperstore.db")
    shutil.copytree(FIXTURE_DIR / "paperstore", workspace / "paperstore")
    prefix = f"{workspace.as_posix()}/"
    conn = sqlite3.connect(workspace / "paperstore.db")
    try:
        with conn:
            for column in ("source_file", "markdown_path"):
                conn.execute(
                    f"UPDATE papers SET {column} = ? || {column} WHERE {column} != ''",
                    (prefix,),
                )
    finally:
        conn.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def _wait_for_port(proc: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"cpp-mcp exited with {proc.returncode} before listening")
        try:
            with socket.create_connection((HOST, port), timeout=CONNECT_POLL_S):
                return
        except OSError:
            time.sleep(CONNECT_POLL_S)
    pytest.fail(f"cpp-mcp did not listen on {HOST}:{port} within {STARTUP_TIMEOUT_S}s")


def _wait_for_listening_url(proc: subprocess.Popen[str]) -> str:
    """Read paper-mcp's stderr until it logs its bound URL, then keep draining it."""
    assert proc.stderr is not None
    stderr = proc.stderr
    lines: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        for line in stderr:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    seen: list[str] = []
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty:
            break
        if line is None:
            pytest.fail(f"paper-mcp exited before listening:\n{''.join(seen)}")
        seen.append(line)
        if match := LISTENING_RE.search(line):
            return match.group(1)
    pytest.fail(f"paper-mcp did not listen within {STARTUP_TIMEOUT_S}s:\n{''.join(seen)}")


def _stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=SHUTDOWN_TIMEOUT_S)


async def _list_tool_names(url: str, key: str) -> set[str]:
    transport = StreamableHttpTransport(url, headers={"Authorization": f"Bearer {key}"})
    async with Client(transport, timeout=CLIENT_TIMEOUT_S) as client:
        return {tool.name for tool in await client.list_tools()}


@pytest.fixture
def keys_file(tmp_path: Path) -> Path:
    path = tmp_path / "keys"
    path.write_text(f"# shared by cpp-mcp and paper-mcp\n\n{SHARED_KEY}\n", encoding="utf-8")
    return path


@pytest.fixture
def paper_mcp_url(tmp_path: Path, keys_file: Path) -> Iterator[str]:
    binary = _paper_mcp_bin()
    workspace = tmp_path / "wg21"
    _stage_fixture(workspace)
    env = _paper_mcp_env(workspace)
    indexed = subprocess.run(
        [str(binary), "index"],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=INDEX_TIMEOUT_S,
    )
    assert indexed.returncode == 0, f"paper-mcp index failed:\n{indexed.stdout}\n{indexed.stderr}"

    proc = subprocess.Popen(
        [str(binary), "serve", "--host", HOST, "--port", "0", "--keys-file", str(keys_file)],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield _wait_for_listening_url(proc)
    finally:
        _stop(proc)


@pytest.fixture
def cpp_mcp_log(tmp_path: Path) -> Path:
    return tmp_path / "cpp-mcp.log"


@pytest.fixture
def cpp_mcp_url(tmp_path: Path, keys_file: Path, cpp_mcp_log: Path) -> Iterator[str]:
    port = _free_port()
    log = cpp_mcp_log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "cpp_mcp",
            "--data-dir", str(tmp_path / "cpp-mcp"),
            "serve", "--host", HOST, "--port", str(port), "--keys-file", str(keys_file),
        ],
        env=_base_env(),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_port(proc, port)
        yield f"http://{HOST}:{port}{MCP_PATH}"
    finally:
        _stop(proc)
        log.close()


def test_shared_key_authenticates_against_both(paper_mcp_url: str, cpp_mcp_url: str) -> None:
    paper_tools = asyncio.run(_list_tool_names(paper_mcp_url, SHARED_KEY))
    cpp_tools = asyncio.run(_list_tool_names(cpp_mcp_url, SHARED_KEY))
    assert PAPER_MCP_TOOL in paper_tools
    assert CPP_MCP_TOOL in cpp_tools


def test_unknown_key_rejected_by_paper_mcp(paper_mcp_url: str, cpp_mcp_url: str) -> None:
    with pytest.raises(Exception) as excinfo:
        asyncio.run(_list_tool_names(paper_mcp_url, UNKNOWN_KEY))
    assert HTTP_UNAUTHORIZED_RE.search(str(excinfo.value)), repr(excinfo.value)


def test_unknown_key_rejected_by_cpp_mcp(
    paper_mcp_url: str, cpp_mcp_url: str, cpp_mcp_log: Path
) -> None:
    with pytest.raises(McpError):
        asyncio.run(_list_tool_names(cpp_mcp_url, UNKNOWN_KEY))
    assert CPP_MCP_REJECTION_LOG in cpp_mcp_log.read_text(encoding="utf-8")
