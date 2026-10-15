// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Configuration of the `serve` command and its startup validation.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use paperstore::{StorageBackend as _, default_workspace_dir};
use paperstore_sqlite::SqliteBackend;
use rmcp::transport::streamable_http_server::StreamableHttpServerConfig;

use crate::cursor::{CursorConfig, DEFAULT_CURSOR_CAPACITY, DEFAULT_CURSOR_TTL, DEFAULT_PAGE_SIZE};
use crate::error::{Error, Result};
use crate::ingest::TOOL_NAME;
use crate::model::onnx::check_model_files;
use crate::model::resolve_model_dir;
use crate::search::{DEFAULT_RERANK_DEPTH, DEFAULT_TOP_K};
use crate::store::lance::LANCEDB_DIR;

/// Environment variable holding the bearer token when `--token-file` is absent.
pub const TOKEN_ENV: &str = "PAPER_MCP_TOKEN";

/// Default listen address: loopback only, since a reverse proxy in front
/// of the server terminates TLS.
pub const DEFAULT_BIND: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8080);

/// Return rmcp's default accepted `Host` header values (the loopback names),
/// which guard against DNS rebinding.
#[must_use]
pub fn default_allowed_hosts() -> Vec<String> {
    StreamableHttpServerConfig::default().allowed_hosts
}

/// Name of the paperstore database inside the workspace directory.
const PAPERSTORE_DB: &str = "paperstore.db";

/// Shown in place of the token by `Debug` and `Display`.
const REDACTED: &str = "[redacted]";

/// A bearer token that never reveals itself in `Debug` or `Display` output.
#[derive(Clone)]
pub struct BearerToken(String);

impl BearerToken {
    /// Wrap `token`, trimming surrounding whitespace such as a trailing newline.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidToken`] when the trimmed token is empty or
    /// holds characters other than visible ASCII, which an `Authorization`
    /// header cannot carry verbatim.
    pub fn new(token: &str) -> Result<Self> {
        let token = token.trim();
        if token.is_empty() {
            return Err(Error::InvalidToken("the token is empty"));
        }
        if !token.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(Error::InvalidToken(
                "the token must be visible ASCII with no spaces",
            ));
        }
        Ok(Self(token.to_owned()))
    }

    /// Return the token's bytes, for comparison against a request.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl fmt::Debug for BearerToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BearerToken").field(&REDACTED).finish()
    }
}

impl fmt::Display for BearerToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

/// Return the bearer token: the contents of `file` when given, otherwise [`TOKEN_ENV`].
///
/// An empty environment value counts as unset.
///
/// # Errors
///
/// Returns [`Error::TokenFileRead`] when `file` cannot be read,
/// [`Error::NoToken`] when neither source is set, and
/// [`Error::InvalidToken`] when the token is unusable.
pub fn resolve_token(file: Option<&Path>) -> Result<BearerToken> {
    if let Some(path) = file {
        let contents = std::fs::read_to_string(path).map_err(|source| Error::TokenFileRead {
            path: path.to_path_buf(),
            source,
        })?;
        return BearerToken::new(&contents);
    }
    let value = std::env::var(TOKEN_ENV).unwrap_or_default();
    if value.is_empty() {
        return Err(Error::NoToken);
    }
    BearerToken::new(&value)
}

/// Operator-supplied `serve` settings, before validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServeOptions {
    /// Address to listen on.
    pub bind: SocketAddr,
    /// File holding the bearer token; [`TOKEN_ENV`] is used when absent.
    pub token_file: Option<PathBuf>,
    /// Model directory; the model directory environment variable is used when absent.
    pub model_dir: Option<PathBuf>,
    /// Accepted `Host` header values, guarding against DNS rebinding.
    pub allowed_hosts: Vec<String>,
    /// Default number of nearest chunks a vector or hybrid search returns.
    pub top_k: NonZeroUsize,
    /// Number of leading results the reranker reorders.
    pub rerank_depth: NonZeroUsize,
    /// Items per result page.
    pub page_size: NonZeroUsize,
    /// Idle time after which a stored result list expires.
    pub cursor_ttl: Duration,
    /// Maximum number of stored result lists.
    pub cursor_capacity: NonZeroUsize,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self {
            bind: DEFAULT_BIND,
            token_file: None,
            model_dir: None,
            allowed_hosts: default_allowed_hosts(),
            top_k: NonZeroUsize::new(DEFAULT_TOP_K).unwrap_or(NonZeroUsize::MIN),
            rerank_depth: NonZeroUsize::new(DEFAULT_RERANK_DEPTH).unwrap_or(NonZeroUsize::MIN),
            page_size: DEFAULT_PAGE_SIZE,
            cursor_ttl: DEFAULT_CURSOR_TTL,
            cursor_capacity: DEFAULT_CURSOR_CAPACITY,
        }
    }
}

/// Validated `serve` configuration.
#[derive(Clone, Debug)]
pub struct ServeConfig {
    /// Address to listen on.
    pub bind: SocketAddr,
    /// The bearer token every request must carry.
    pub token: BearerToken,
    /// The paperstore workspace directory.
    pub workspace: PathBuf,
    /// The index's tool directory inside the workspace.
    pub tool_dir: PathBuf,
    /// The model directory, holding every required model file.
    pub model_dir: PathBuf,
    /// Accepted `Host` header values.
    pub allowed_hosts: Vec<String>,
    /// Default number of nearest chunks a vector or hybrid search returns.
    pub top_k: NonZeroUsize,
    /// Number of leading results the reranker reorders.
    pub rerank_depth: NonZeroUsize,
    /// Paging limits for stored result lists.
    pub cursor: CursorConfig,
}

impl ServeConfig {
    /// Validate `options` against the environment and the filesystem.
    ///
    /// Checks run in order: token, workspace, paperstore database, model
    /// files, index. A missing paperstore database is never created, but an
    /// existing one is opened through [`SqliteBackend`], which brings its
    /// schema up to date. Nothing else is created or modified.
    ///
    /// # Errors
    ///
    /// Returns the token errors of [`resolve_token`],
    /// [`paperstore::Error::WorkspaceEnvUnset`] when `WG21_DATA_DIR` is
    /// unset, [`Error::NoPaperstoreDb`] when the workspace has no database,
    /// [`Error::NoModelDir`] or [`Error::MissingModelFiles`] for absent
    /// models, and [`Error::NoIndex`] when nothing has been indexed.
    pub fn load(options: ServeOptions) -> Result<Self> {
        let token = resolve_token(options.token_file.as_deref())?;
        let workspace = default_workspace_dir()?;
        let db = workspace.join(PAPERSTORE_DB);
        if !db.is_file() {
            return Err(Error::NoPaperstoreDb(db));
        }
        let model_dir = resolve_model_dir(options.model_dir.as_deref())?;
        check_model_files(&model_dir)?;
        let tool_dir = SqliteBackend::new(&workspace)?.tool_dir(TOOL_NAME);
        let index = tool_dir.join(LANCEDB_DIR);
        if !index.is_dir() {
            return Err(Error::NoIndex(index));
        }
        Ok(Self {
            bind: options.bind,
            token,
            workspace,
            tool_dir,
            model_dir,
            allowed_hosts: options.allowed_hosts,
            top_k: options.top_k,
            rerank_depth: options.rerank_depth,
            cursor: CursorConfig {
                page_size: options.page_size,
                ttl: options.cursor_ttl,
                capacity: options.cursor_capacity,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use paperstore::WORKSPACE_ENV_VAR;
    use tempfile::TempDir;

    use super::*;
    use crate::model::MODEL_DIR_ENV;
    use crate::model::onnx::{EMBEDDER_SUBDIR, ONNX_FILE, RERANKER_SUBDIR, TOKENIZER_FILES};

    const SECRET: &str = "s3cret-Token_value";

    /// A workspace with a paperstore database, an index directory and a
    /// model directory holding every required file, each removable.
    struct Fixture {
        workspace: TempDir,
        models: TempDir,
    }

    impl Fixture {
        fn new() -> Self {
            let workspace = TempDir::new().unwrap_or_else(|e| panic!("tempdir: {e}"));
            let models = TempDir::new().unwrap_or_else(|e| panic!("tempdir: {e}"));
            assert!(SqliteBackend::new(workspace.path()).is_ok());
            let index = workspace.path().join(TOOL_NAME).join(LANCEDB_DIR);
            assert!(std::fs::create_dir_all(index).is_ok());
            for sub in [EMBEDDER_SUBDIR, RERANKER_SUBDIR] {
                let dir = models.path().join(sub);
                assert!(std::fs::create_dir_all(&dir).is_ok());
                for name in std::iter::once(ONNX_FILE).chain(TOKENIZER_FILES) {
                    assert!(std::fs::write(dir.join(name), b"x").is_ok());
                }
            }
            Self { workspace, models }
        }

        fn options(&self) -> ServeOptions {
            ServeOptions {
                model_dir: Some(self.models.path().to_path_buf()),
                ..ServeOptions::default()
            }
        }

        fn load_with(&self, token: Option<&str>, options: ServeOptions) -> Result<ServeConfig> {
            temp_env::with_vars(
                [
                    (TOKEN_ENV, token),
                    (WORKSPACE_ENV_VAR, self.workspace.path().to_str()),
                    (MODEL_DIR_ENV, None),
                ],
                || ServeConfig::load(options),
            )
        }

        fn load(&self) -> Result<ServeConfig> {
            self.load_with(Some(SECRET), self.options())
        }
    }

    fn assert_err_without_secret(result: Result<ServeConfig>, check: impl Fn(&Error) -> bool) {
        match result {
            Ok(config) => panic!("expected an error, got {config:?}"),
            Err(e) => {
                assert!(check(&e), "unexpected error {e:?}");
                assert!(!e.to_string().contains(SECRET), "token in {e}");
                assert!(!format!("{e:?}").contains(SECRET), "token in {e:?}");
            }
        }
    }

    #[test]
    fn defaults_follow_the_specification() {
        let options = ServeOptions::default();
        assert_eq!(options.top_k.get(), 1000);
        assert_eq!(options.rerank_depth.get(), 150);
        assert_eq!(options.page_size.get(), 20);
        assert_eq!(options.cursor_ttl, Duration::from_mins(15));
        assert_eq!(options.cursor_capacity.get(), 64);
        assert!(options.bind.ip().is_loopback());
    }

    #[test]
    fn valid_setup_loads() {
        let fixture = Fixture::new();
        let config = fixture.load().unwrap_or_else(|e| panic!("load: {e}"));
        assert_eq!(config.token.as_bytes(), SECRET.as_bytes());
        assert_eq!(config.workspace, fixture.workspace.path());
        assert_eq!(config.tool_dir, fixture.workspace.path().join(TOOL_NAME));
        assert_eq!(config.model_dir, fixture.models.path());
        assert_eq!(config.cursor, CursorConfig::default());
    }

    #[test]
    fn token_redacts_itself() {
        let fixture = Fixture::new();
        let config = fixture.load().unwrap_or_else(|e| panic!("load: {e}"));
        assert_eq!(config.token.to_string(), REDACTED);
        for shown in [format!("{:?}", config.token), format!("{config:?}")] {
            assert!(!shown.contains(SECRET), "token in {shown}");
            assert!(shown.contains(REDACTED));
        }
    }

    #[test]
    fn missing_token_fails() {
        let fixture = Fixture::new();
        for unset in [None, Some("")] {
            assert_err_without_secret(fixture.load_with(unset, fixture.options()), |e| {
                matches!(e, Error::NoToken) && e.to_string().contains(TOKEN_ENV)
            });
        }
    }

    #[test]
    fn blank_or_spaced_token_fails() {
        let fixture = Fixture::new();
        for bad in ["   ", "two words", "tab\there"] {
            assert_err_without_secret(fixture.load_with(Some(bad), fixture.options()), |e| {
                matches!(e, Error::InvalidToken(_))
                    && (bad.trim().is_empty() || !e.to_string().contains(bad.trim()))
            });
        }
    }

    #[test]
    fn token_file_wins_and_is_trimmed() {
        let fixture = Fixture::new();
        let file = fixture.models.path().join("token");
        assert!(std::fs::write(&file, b"from-file\n").is_ok());
        let options = ServeOptions {
            token_file: Some(file),
            ..fixture.options()
        };
        let config = fixture
            .load_with(Some(SECRET), options)
            .unwrap_or_else(|e| panic!("load: {e}"));
        assert_eq!(config.token.as_bytes(), b"from-file");
    }

    #[test]
    fn unreadable_token_file_fails() {
        let fixture = Fixture::new();
        let options = ServeOptions {
            token_file: Some(fixture.models.path().join("absent")),
            ..fixture.options()
        };
        assert_err_without_secret(fixture.load_with(Some(SECRET), options), |e| {
            matches!(e, Error::TokenFileRead { .. })
        });
    }

    #[test]
    fn missing_workspace_env_fails() {
        let fixture = Fixture::new();
        let result = temp_env::with_vars(
            [(TOKEN_ENV, Some(SECRET)), (WORKSPACE_ENV_VAR, None)],
            || ServeConfig::load(fixture.options()),
        );
        assert_err_without_secret(result, |e| {
            matches!(e, Error::Store(paperstore::Error::WorkspaceEnvUnset))
                && e.to_string().contains(WORKSPACE_ENV_VAR)
        });
    }

    #[test]
    fn missing_database_fails_without_creating_it() {
        let fixture = Fixture::new();
        let db = fixture.workspace.path().join(PAPERSTORE_DB);
        assert!(std::fs::remove_file(&db).is_ok());
        assert_err_without_secret(fixture.load(), |e| matches!(e, Error::NoPaperstoreDb(_)));
        assert!(!db.exists(), "validation must not create the database");
    }

    #[test]
    fn missing_model_dir_or_files_fail() {
        let fixture = Fixture::new();
        let options = ServeOptions {
            model_dir: None,
            ..fixture.options()
        };
        assert_err_without_secret(fixture.load_with(Some(SECRET), options), |e| {
            matches!(e, Error::NoModelDir)
        });
        let onnx = fixture.models.path().join(RERANKER_SUBDIR).join(ONNX_FILE);
        assert!(std::fs::remove_file(onnx).is_ok());
        assert_err_without_secret(
            fixture.load(),
            |e| matches!(e, Error::MissingModelFiles { dir, .. } if dir.ends_with(RERANKER_SUBDIR)),
        );
    }

    #[test]
    fn missing_index_fails() {
        let fixture = Fixture::new();
        let tool_dir = fixture.workspace.path().join(TOOL_NAME);
        assert!(std::fs::remove_dir_all(&tool_dir).is_ok());
        assert_err_without_secret(
            fixture.load(),
            |e| matches!(e, Error::NoIndex(path) if path == &tool_dir.join(LANCEDB_DIR)),
        );
    }
}
