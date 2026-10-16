//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Bearer-key authentication for the HTTP server.
//!
//! Compatible with `cpp-mcp` (`packages/cpp-mcp/src/cpp_mcp/server.py`
//! `_load_keys` and `_BearerKeyMiddleware`): the same keys file format, the
//! same `$CPP_MCP_KEYS_FILE` override, and the same `Authorization: Bearer
//! <key>` header, so one key from the shared keys file authenticates against
//! both servers.

use std::collections::BTreeSet;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError, RwLock};
use std::task::{Context, Poll};

use axum::http::{Request, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures::future::BoxFuture;
use tower::{Layer, Service};

/// Environment variable naming the keys file, shared with `cpp-mcp`.
pub const KEYS_FILE_ENV: &str = "CPP_MCP_KEYS_FILE";

/// Keys file used when neither `--keys-file` nor [`KEYS_FILE_ENV`] is set:
/// the path `cpp-mcp` deployments use.
pub const DEFAULT_KEYS_FILE: &str = "/etc/cpp-mcp/keys";

/// Case-insensitive scheme prefix of the `Authorization` header.
const BEARER_PREFIX: &str = "bearer ";

/// Errors from reading a keys file.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    /// The keys file exists but could not be read as UTF-8 text.
    #[error("failed to read keys file {path}: {source}")]
    Read {
        /// The keys file path.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: io::Error,
    },
}

/// Resolve the keys file path: `cli` if given, else a non-blank
/// [`KEYS_FILE_ENV`], else [`DEFAULT_KEYS_FILE`].
#[must_use]
pub fn resolve_keys_file(cli: Option<&Path>) -> PathBuf {
    if let Some(path) = cli {
        return path.to_path_buf();
    }
    match std::env::var(KEYS_FILE_ENV) {
        Ok(value) if !value.trim().is_empty() => PathBuf::from(value.trim()),
        _ => PathBuf::from(DEFAULT_KEYS_FILE),
    }
}

/// Load bearer keys from `path`: one key per line, surrounding whitespace
/// trimmed, blank lines and lines starting with `#` ignored.
///
/// A path that is not an existing file yields an empty set with a warning,
/// as `cpp-mcp` does; the server refuses to start on an empty set unless
/// authentication is disabled.
///
/// # Errors
///
/// Returns [`AuthError::Read`] when the file exists but cannot be read as
/// UTF-8 text.
pub fn load_keys(path: &Path) -> Result<BTreeSet<String>, AuthError> {
    if !path.is_file() {
        tracing::warn!(
            "Keys file {} does not exist; no keys loaded",
            path.display()
        );
        return Ok(BTreeSet::new());
    }
    let text = std::fs::read_to_string(path).map_err(|source| AuthError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let keys: BTreeSet<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect();
    tracing::info!("Loaded {} API key(s) from {}", keys.len(), path.display());
    Ok(keys)
}

/// A shared, swappable set of bearer keys loaded from a keys file.
///
/// Clones share the same set, so a [`KeyStore::reload`] through any clone
/// is seen by every [`BearerKeyLayer`] built from it.
#[derive(Debug, Clone)]
pub struct KeyStore {
    path: PathBuf,
    keys: Arc<RwLock<Arc<BTreeSet<String>>>>,
}

impl KeyStore {
    /// Load the keys file at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`AuthError::Read`] as [`load_keys`] does.
    pub fn open(path: &Path) -> Result<Self, AuthError> {
        let keys = load_keys(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            keys: Arc::new(RwLock::new(Arc::new(keys))),
        })
    }

    /// The keys file this store loads from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Re-read the keys file and swap in its keys, returning the new count.
    ///
    /// # Errors
    ///
    /// Returns [`AuthError::Read`] as [`load_keys`] does; the previous keys
    /// stay in effect.
    pub fn reload(&self) -> Result<usize, AuthError> {
        let keys = load_keys(&self.path)?;
        let count = keys.len();
        *self.keys.write().unwrap_or_else(PoisonError::into_inner) = Arc::new(keys);
        Ok(count)
    }

    /// Whether `key` is currently accepted.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.snapshot().contains(key)
    }

    /// Number of keys currently loaded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.snapshot().len()
    }

    /// Whether no keys are loaded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshot().is_empty()
    }

    fn snapshot(&self) -> Arc<BTreeSet<String>> {
        Arc::clone(&self.keys.read().unwrap_or_else(PoisonError::into_inner))
    }
}

/// Extract the bearer token from an `Authorization` header value: the text
/// after a case-insensitive `Bearer ` prefix, untrimmed, as `cpp-mcp` reads
/// it.
fn bearer_token(value: &str) -> Option<&str> {
    let prefix = value.get(..BEARER_PREFIX.len())?;
    prefix
        .eq_ignore_ascii_case(BEARER_PREFIX)
        .then(|| &value[BEARER_PREFIX.len()..])
}

/// Tower layer rejecting requests without a known bearer key.
///
/// A request whose `Authorization` header is missing, is not a `Bearer`
/// credential, or carries a key not in the [`KeyStore`] receives
/// `401 Unauthorized` and never reaches the inner service.
#[derive(Debug, Clone)]
pub struct BearerKeyLayer {
    keys: KeyStore,
}

impl BearerKeyLayer {
    /// Build a layer checking requests against `keys`.
    #[must_use]
    pub fn new(keys: KeyStore) -> Self {
        Self { keys }
    }
}

impl<S> Layer<S> for BearerKeyLayer {
    type Service = BearerKeyService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        BearerKeyService {
            inner,
            keys: self.keys.clone(),
        }
    }
}

/// The service produced by [`BearerKeyLayer`].
#[derive(Debug, Clone)]
pub struct BearerKeyService<S> {
    inner: S,
    keys: KeyStore,
}

impl<S> BearerKeyService<S> {
    fn is_authorized<B>(&self, request: &Request<B>) -> bool {
        request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(bearer_token)
            .is_some_and(|token| self.keys.contains(token))
    }
}

impl<S, B> Service<Request<B>> for BearerKeyService<S>
where
    S: Service<Request<B>>,
    S::Response: IntoResponse,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future = BoxFuture<'static, Result<Response, S::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request<B>) -> Self::Future {
        if !self.is_authorized(&request) {
            return Box::pin(async { Ok(StatusCode::UNAUTHORIZED.into_response()) });
        }
        let future = self.inner.call(request);
        Box::pin(async move { future.await.map(IntoResponse::into_response) })
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::fs;

    use axum::body::Body;
    use tempfile::TempDir;
    use tower::ServiceExt;
    use tower::service_fn;

    use super::*;

    fn write_keys(dir: &TempDir, text: &str) -> PathBuf {
        let path = dir.path().join("keys");
        fs::write(&path, text).unwrap_or_else(|err| panic!("write keys: {err}"));
        path
    }

    fn set(keys: &[&str]) -> BTreeSet<String> {
        keys.iter().map(|key| (*key).to_owned()).collect()
    }

    fn tempdir() -> TempDir {
        TempDir::new().unwrap_or_else(|err| panic!("tempdir: {err}"))
    }

    #[test]
    fn load_keys_skips_comments_and_blank_lines_and_trims() {
        let dir = tempdir();
        let path = write_keys(
            &dir,
            "# operator keys\n\nalpha\n  beta  \n\t\n   # indented comment\r\ngamma\r\nalpha\n",
        );
        let keys = load_keys(&path).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(keys, set(&["alpha", "beta", "gamma"]));
    }

    #[test]
    fn load_keys_keeps_inner_hash() {
        let dir = tempdir();
        let path = write_keys(&dir, "key#with#hash\n");
        let keys = load_keys(&path).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(keys, set(&["key#with#hash"]));
    }

    #[test]
    fn load_keys_missing_file_is_empty() {
        let dir = tempdir();
        let keys = load_keys(&dir.path().join("absent")).unwrap_or_else(|err| panic!("{err}"));
        assert!(keys.is_empty());
    }

    #[test]
    fn load_keys_rejects_non_utf8() {
        let dir = tempdir();
        let path = dir.path().join("keys");
        fs::write(&path, [0xff, 0xfe, b'\n']).unwrap_or_else(|err| panic!("{err}"));
        assert!(matches!(load_keys(&path), Err(AuthError::Read { .. })));
    }

    #[test]
    fn reload_picks_up_added_key_across_clones() {
        let dir = tempdir();
        let path = write_keys(&dir, "alpha\n");
        let store = KeyStore::open(&path).unwrap_or_else(|err| panic!("{err}"));
        let shared = store.clone();
        assert!(shared.contains("alpha"));
        assert!(!shared.contains("beta"));

        fs::write(&path, "alpha\nbeta\n").unwrap_or_else(|err| panic!("{err}"));
        assert!(!shared.contains("beta"), "keys change only on reload");
        assert_eq!(store.reload().unwrap_or_else(|err| panic!("{err}")), 2);
        assert!(shared.contains("beta"));
        assert_eq!(shared.len(), 2);
    }

    #[test]
    fn reload_failure_keeps_previous_keys() {
        let dir = tempdir();
        let path = write_keys(&dir, "alpha\n");
        let store = KeyStore::open(&path).unwrap_or_else(|err| panic!("{err}"));
        fs::write(&path, [0xff, 0xfe]).unwrap_or_else(|err| panic!("{err}"));
        assert!(store.reload().is_err());
        assert!(store.contains("alpha"));
    }

    #[test]
    fn resolve_prefers_cli_then_env_then_default() {
        let cli = PathBuf::from("/cli/keys");
        temp_env::with_var(KEYS_FILE_ENV, Some("/env/keys"), || {
            assert_eq!(resolve_keys_file(Some(&cli)), cli);
            assert_eq!(resolve_keys_file(None), PathBuf::from("/env/keys"));
        });
        temp_env::with_var(KEYS_FILE_ENV, Some("   "), || {
            assert_eq!(resolve_keys_file(None), PathBuf::from(DEFAULT_KEYS_FILE));
        });
        temp_env::with_var_unset(KEYS_FILE_ENV, || {
            assert_eq!(resolve_keys_file(None), PathBuf::from(DEFAULT_KEYS_FILE));
        });
    }

    async fn status_for(store: &KeyStore, authorization: Option<&str>) -> StatusCode {
        let inner = service_fn(|_request: Request<Body>| async {
            Ok::<_, Infallible>(StatusCode::OK.into_response())
        });
        let service = BearerKeyLayer::new(store.clone()).layer(inner);
        let mut request = Request::builder().uri("/mcp");
        if let Some(value) = authorization {
            request = request.header(header::AUTHORIZATION, value);
        }
        let request = request
            .body(Body::empty())
            .unwrap_or_else(|err| panic!("request: {err}"));
        match service.oneshot(request).await {
            Ok(response) => response.status(),
            Err(never) => match never {},
        }
    }

    #[tokio::test]
    async fn layer_rejects_missing_and_wrong_keys_and_accepts_valid() {
        let dir = tempdir();
        let path = write_keys(&dir, "alpha\n");
        let store = KeyStore::open(&path).unwrap_or_else(|err| panic!("{err}"));

        assert_eq!(status_for(&store, None).await, StatusCode::UNAUTHORIZED);
        assert_eq!(
            status_for(&store, Some("Bearer wrong")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status_for(&store, Some("Basic alpha")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status_for(&store, Some("Bearer alpha ")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status_for(&store, Some("Bearer")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status_for(&store, Some("Bearer alpha")).await,
            StatusCode::OK
        );
        assert_eq!(
            status_for(&store, Some("bearer alpha")).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn layer_accepts_key_added_after_reload() {
        let dir = tempdir();
        let path = write_keys(&dir, "alpha\n");
        let store = KeyStore::open(&path).unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(
            status_for(&store, Some("Bearer beta")).await,
            StatusCode::UNAUTHORIZED
        );

        fs::write(&path, "alpha\nbeta\n").unwrap_or_else(|err| panic!("{err}"));
        store.reload().unwrap_or_else(|err| panic!("{err}"));
        assert_eq!(
            status_for(&store, Some("Bearer beta")).await,
            StatusCode::OK
        );
    }
}
