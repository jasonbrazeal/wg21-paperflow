// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Startup checks of the `paper-mcp` binary, which need no models.

use std::process::{Command, Output};

use tempfile::TempDir;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const WORKSPACE_ENV: &str = "WG21_DATA_DIR";
const MODEL_DIR_ENV: &str = "PAPER_MCP_MODEL_DIR";
const TOKEN_ENV: &str = "PAPER_MCP_TOKEN";
const SECRET: &str = "s3cret-Token_value";

fn index(workspace: Option<&TempDir>, model_dir: Option<&TempDir>) -> std::io::Result<Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_paper-mcp"));
    command
        .args(["index", "all"])
        .env_remove(WORKSPACE_ENV)
        .env_remove(MODEL_DIR_ENV);
    if let Some(dir) = workspace {
        command.env(WORKSPACE_ENV, dir.path());
    }
    if let Some(dir) = model_dir {
        command.arg("--model-dir").arg(dir.path());
    }
    command.output()
}

fn assert_fails_with(output: &Output, needle: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "should fail: {stderr}");
    assert!(stderr.contains(needle), "stderr lacks {needle:?}: {stderr}");
}

#[test]
fn index_fails_without_workspace_env() -> TestResult {
    assert_fails_with(&index(None, None)?, WORKSPACE_ENV);
    Ok(())
}

#[test]
fn index_fails_without_paperstore_db() -> TestResult {
    let workspace = TempDir::new()?;
    let output = index(Some(&workspace), None)?;
    assert_fails_with(&output, "no paperstore database");
    assert!(
        !workspace.path().join("paperstore").exists(),
        "nothing is created in a workspace without a database"
    );
    Ok(())
}

#[test]
fn index_fails_without_model_files() -> TestResult {
    let workspace = TempDir::new()?;
    std::fs::write(workspace.path().join("paperstore.db"), b"")?;
    let models = TempDir::new()?;
    assert_fails_with(&index(Some(&workspace), Some(&models))?, "is missing");
    assert_fails_with(&index(Some(&workspace), None)?, MODEL_DIR_ENV);
    Ok(())
}

fn serve(token: Option<&str>, workspace: Option<&TempDir>) -> std::io::Result<Output> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_paper-mcp"));
    command
        .args(["serve", "--bind", "127.0.0.1:0"])
        .env("RUST_LOG", "trace")
        .env_remove(WORKSPACE_ENV)
        .env_remove(MODEL_DIR_ENV)
        .env_remove(TOKEN_ENV);
    if let Some(token) = token {
        command.env(TOKEN_ENV, token);
    }
    if let Some(dir) = workspace {
        command.env(WORKSPACE_ENV, dir.path());
    }
    command.output()
}

fn assert_secret_hidden(output: &Output) {
    for stream in [&output.stdout, &output.stderr] {
        assert!(!String::from_utf8_lossy(stream).contains(SECRET));
    }
}

#[test]
fn serve_fails_without_token() -> TestResult {
    assert_fails_with(&serve(None, None)?, TOKEN_ENV);
    Ok(())
}

#[test]
fn serve_startup_errors_hide_the_token() -> TestResult {
    let output = serve(Some(SECRET), None)?;
    assert_fails_with(&output, WORKSPACE_ENV);
    assert_secret_hidden(&output);

    let workspace = TempDir::new()?;
    let output = serve(Some(SECRET), Some(&workspace))?;
    assert_fails_with(&output, "no paperstore database");
    assert_secret_hidden(&output);

    std::fs::write(workspace.path().join("paperstore.db"), b"")?;
    let output = serve(Some(SECRET), Some(&workspace))?;
    assert_fails_with(&output, MODEL_DIR_ENV);
    assert_secret_hidden(&output);
    Ok(())
}
