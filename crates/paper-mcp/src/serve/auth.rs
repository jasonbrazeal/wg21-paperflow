//! Bearer-token authentication (§9).
//!
//! Tokens come from a keys file, one per line, with blank lines and `#`
//! comments ignored: the format the Python `paper_mcp` uses. SIGHUP reloads
//! the file; a reload that yields no tokens is rejected.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use subtle::ConstantTimeEq;
use tracing::{info, warn};

const BEARER_PREFIX: &str = "bearer ";

/// A keys file problem.
#[derive(Debug, thiserror::Error)]
pub enum KeysError {
    /// The file could not be read.
    #[error("cannot read keys file {path}: {source}")]
    Read {
        /// The file.
        path: PathBuf,
        /// The cause.
        source: std::io::Error,
    },
    /// The file holds no tokens.
    #[error("keys file {0} holds no tokens")]
    Empty(PathBuf),
}

/// Parses keys file text into tokens.
#[must_use]
pub fn parse_keys(text: &str) -> Vec<Vec<u8>> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| line.as_bytes().to_vec())
        .collect()
}

fn load(path: &Path) -> Result<Vec<Vec<u8>>, KeysError> {
    let text = std::fs::read_to_string(path).map_err(|source| KeysError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let keys = parse_keys(&text);
    if keys.is_empty() {
        return Err(KeysError::Empty(path.to_path_buf()));
    }
    Ok(keys)
}

/// The current token set, shared by every request.
#[derive(Debug)]
pub struct Keys {
    path: PathBuf,
    tokens: RwLock<Vec<Vec<u8>>>,
}

impl Keys {
    /// Loads the keys file.
    ///
    /// # Errors
    ///
    /// Returns [`KeysError`] when the file cannot be read or holds no
    /// tokens.
    pub fn load(path: &Path) -> Result<Arc<Self>, KeysError> {
        let tokens = load(path)?;
        info!(count = tokens.len(), path = %path.display(), "loaded bearer keys");
        Ok(Arc::new(Self {
            path: path.to_path_buf(),
            tokens: RwLock::new(tokens),
        }))
    }

    /// Re-reads the keys file, keeping the current tokens when the file
    /// cannot be read or holds none.
    pub fn reload(&self) {
        match load(&self.path) {
            Ok(tokens) => {
                let count = tokens.len();
                if let Ok(mut current) = self.tokens.write() {
                    *current = tokens;
                    info!(count, "reloaded bearer keys");
                }
            }
            Err(e) => warn!(error = %e, "keys reload rejected; previous keys stay in force"),
        }
    }

    /// Whether `token` is a current key, compared in constant time against
    /// every key.
    #[must_use]
    pub fn verify(&self, token: &[u8]) -> bool {
        if token.is_empty() {
            return false;
        }
        let Ok(tokens) = self.tokens.read() else {
            return false;
        };
        let mut matched = subtle::Choice::from(0);
        for key in tokens.iter() {
            matched |= key.as_slice().ct_eq(token);
        }
        matched.into()
    }
}

fn bearer(value: &str) -> Option<&str> {
    let prefix = value.get(..BEARER_PREFIX.len())?;
    if !prefix.eq_ignore_ascii_case(BEARER_PREFIX) {
        return None;
    }
    let token = value[BEARER_PREFIX.len()..].trim();
    (!token.is_empty()).then_some(token)
}

/// Rejects requests without a valid bearer token with HTTP 401.
pub async fn require_bearer(
    State(keys): State<Arc<Keys>>,
    request: Request,
    next: Next,
) -> Response {
    let authorized = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(bearer)
        .is_some_and(|token| keys.verify(token.as_bytes()));
    if authorized {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            "missing or invalid bearer token",
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tokens_and_skips_comments() {
        let keys = parse_keys("# comment\n\n  abc  \ndef\n#x\n");
        assert_eq!(keys, vec![b"abc".to_vec(), b"def".to_vec()]);
    }

    #[test]
    fn extracts_bearer_tokens() {
        assert_eq!(bearer("Bearer abc"), Some("abc"));
        assert_eq!(bearer("bearer  abc "), Some("abc"));
        assert_eq!(bearer("Basic abc"), None);
        assert_eq!(bearer("Bearer "), None);
        assert_eq!(bearer("Bear"), None);
    }

    #[test]
    fn verifies_and_reload_keeps_keys_on_empty_file() {
        let dir = std::env::temp_dir().join(format!("paper-mcp-keys-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{e}"));
        let path = dir.join("keys");
        std::fs::write(&path, "one\ntwo\n").unwrap_or_else(|e| panic!("{e}"));
        let keys = Keys::load(&path).unwrap_or_else(|e| panic!("{e}"));
        assert!(keys.verify(b"two"));
        assert!(!keys.verify(b"three"));
        assert!(!keys.verify(b""));

        std::fs::write(&path, "# nothing\n").unwrap_or_else(|e| panic!("{e}"));
        keys.reload();
        assert!(keys.verify(b"one"));

        std::fs::write(&path, "three\n").unwrap_or_else(|e| panic!("{e}"));
        keys.reload();
        assert!(keys.verify(b"three"));
        assert!(!keys.verify(b"one"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
