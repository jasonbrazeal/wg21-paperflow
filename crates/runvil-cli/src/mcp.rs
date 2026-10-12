//! The MCP server file: where `runvil` finds it and what it reads from it.
//!
//! The file uses the `mcp.json` format: a JSON object whose `mcpServers`
//! object maps each server's name to its entry. `--mcp` names the file
//! outright. Without it, the file is searched for the way the gateway
//! searches for `gateway.toml`: beside the executable, then in the working
//! directory, then in the user profile's `.promptforge` directory. With no
//! file found, the run has no MCP servers.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::Value;

/// The file name searched for at each discovery location.
pub(crate) const MCP_FILE_NAME: &str = "mcp.json";

/// Returns the MCP file the run reads: `explicit` when given, else the
/// first discovery location that holds an `mcp.json`.
///
/// # Errors
/// Returns an error when `explicit` names a path that is not a file.
pub(crate) fn locate(explicit: Option<&Path>) -> Result<Option<PathBuf>> {
    if let Some(path) = explicit {
        if !path.is_file() {
            bail!("the --mcp file {} does not exist", path.display());
        }
        return Ok(Some(path.to_owned()));
    }
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let cwd = std::env::current_dir().ok();
    let profile = std::env::home_dir().map(|home| home.join(".promptforge"));
    Ok(discover(&[exe_dir, cwd, profile]))
}

/// Returns `mcp.json` in the first of `dirs` that holds one.
fn discover(dirs: &[Option<PathBuf>]) -> Option<PathBuf> {
    dirs.iter()
        .flatten()
        .map(|dir| dir.join(MCP_FILE_NAME))
        .find(|path| path.is_file())
}

/// Reads the `mcpServers` entries of the file at `path`, in name order, as
/// each server's name and its entry.
///
/// The entries are passed on as they stand; the MCP Plugin validates each
/// one when it is installed.
///
/// # Errors
/// Returns an error when the file cannot be read, is not JSON, or has no
/// `mcpServers` object.
pub(crate) fn read_servers(path: &Path) -> Result<Vec<(String, Value)>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read the MCP file {}", path.display()))?;
    parse_servers(&text).with_context(|| format!("read the MCP file {}", path.display()))
}

/// Returns the `mcpServers` entries of an `mcp.json` document.
fn parse_servers(text: &str) -> Result<Vec<(String, Value)>> {
    let document: Value = serde_json::from_str(text).context("the file is not valid JSON")?;
    let Value::Object(mut top) = document else {
        bail!("the file must be a JSON object, got {}", kind(&document));
    };
    match top.remove("mcpServers") {
        Some(Value::Object(servers)) => Ok(servers.into_iter().collect()),
        Some(other) => bail!("`mcpServers` must be a JSON object, got {}", kind(&other)),
        None => bail!("the file has no `mcpServers` object"),
    }
}

/// Names a JSON value's type for an error message.
fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tempfile::TempDir;

    use super::{MCP_FILE_NAME, discover, locate, parse_servers};

    #[test]
    fn parse_servers_reads_every_server_entry() {
        let text = r#"{"mcpServers": {
            "wg21-papers": {"url": "https://example.org/mcp"},
            "Other": {"url": "https://other.example/mcp", "headers": {"x": "y"}}
        }}"#;
        let mut servers = parse_servers(text).unwrap_or_else(|e| panic!("parse: {e}"));
        servers.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            servers,
            [
                (
                    "Other".to_owned(),
                    json!({"url": "https://other.example/mcp", "headers": {"x": "y"}})
                ),
                (
                    "wg21-papers".to_owned(),
                    json!({"url": "https://example.org/mcp"})
                ),
            ]
        );
    }

    #[test]
    fn parse_servers_rejects_malformed_documents() {
        for (text, needle) in [
            ("not json", "not valid JSON"),
            ("[]", "must be a JSON object, got an array"),
            ("{}", "no `mcpServers` object"),
            (
                r#"{"mcpServers": []}"#,
                "`mcpServers` must be a JSON object",
            ),
        ] {
            let error = parse_servers(text)
                .err()
                .unwrap_or_else(|| panic!("{text:?} should be refused"));
            assert!(format!("{error:#}").contains(needle), "{text:?}: {error:#}");
        }
    }

    #[test]
    fn discover_takes_the_first_location_holding_the_file() {
        let empty = TempDir::new().unwrap_or_else(|e| panic!("temp dir: {e}"));
        let second = TempDir::new().unwrap_or_else(|e| panic!("temp dir: {e}"));
        let third = TempDir::new().unwrap_or_else(|e| panic!("temp dir: {e}"));
        for dir in [&second, &third] {
            std::fs::write(dir.path().join(MCP_FILE_NAME), "{}")
                .unwrap_or_else(|e| panic!("write: {e}"));
        }
        let dirs = [
            None,
            Some(empty.path().to_owned()),
            Some(second.path().to_owned()),
            Some(third.path().to_owned()),
        ];
        assert_eq!(discover(&dirs), Some(second.path().join(MCP_FILE_NAME)));
        assert_eq!(discover(&dirs[..2]), None);
    }

    #[test]
    fn locate_refuses_a_missing_explicit_file() {
        let dir = TempDir::new().unwrap_or_else(|e| panic!("temp dir: {e}"));
        assert!(locate(Some(&dir.path().join("absent.json"))).is_err());
        let present = dir.path().join(MCP_FILE_NAME);
        std::fs::write(&present, "{}").unwrap_or_else(|e| panic!("write: {e}"));
        assert_eq!(
            locate(Some(&present)).unwrap_or_else(|e| panic!("locate: {e}")),
            Some(present)
        );
    }
}
