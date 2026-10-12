//! Opaque paging cursors (§7.6).
//!
//! A cursor carries the offset of the next page and a fingerprint of the
//! arguments that produced it (query, type, filters, and page size), so a
//! cursor passed back with different arguments is rejected.

use std::fmt::Write;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Hex digits of the argument hash kept in a cursor.
const FINGERPRINT_LEN: usize = 32;

/// Why a cursor was rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CursorError {
    /// The cursor is not one this server issued.
    #[error("cursor is malformed")]
    Malformed,
    /// The cursor was issued for different arguments.
    #[error("cursor was issued for different arguments")]
    ArgumentsChanged,
}

#[derive(Serialize, Deserialize)]
struct Payload {
    f: String,
    o: u64,
}

/// A stable fingerprint of a call's arguments.
#[must_use]
pub fn fingerprint<T: Serialize>(arguments: &T) -> String {
    let bytes = serde_json::to_vec(arguments).unwrap_or_default();
    let digest = Sha256::digest(&bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex.truncate(FINGERPRINT_LEN);
    hex
}

/// Encodes the cursor for the page starting at `offset`.
#[must_use]
pub fn encode(fingerprint: &str, offset: u64) -> String {
    let payload = Payload {
        f: fingerprint.to_owned(),
        o: offset,
    };
    URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap_or_default())
}

/// Decodes a cursor and checks it against the current arguments' fingerprint.
///
/// # Errors
///
/// Returns [`CursorError::Malformed`] for a string this server did not
/// issue, and [`CursorError::ArgumentsChanged`] when the fingerprint
/// differs.
pub fn decode(cursor: &str, fingerprint: &str) -> Result<u64, CursorError> {
    let bytes = URL_SAFE_NO_PAD
        .decode(cursor)
        .map_err(|_| CursorError::Malformed)?;
    let payload: Payload = serde_json::from_slice(&bytes).map_err(|_| CursorError::Malformed)?;
    if payload.f != fingerprint {
        return Err(CursorError::ArgumentsChanged);
    }
    Ok(payload.o)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize)]
    struct Args<'a> {
        query: &'a str,
        limit: u32,
    }

    #[test]
    fn round_trips() {
        let fp = fingerprint(&Args {
            query: "q",
            limit: 10,
        });
        let cursor = encode(&fp, 30);
        assert_eq!(decode(&cursor, &fp), Ok(30));
    }

    #[test]
    fn rejects_other_arguments_and_garbage() {
        let fp = fingerprint(&Args {
            query: "q",
            limit: 10,
        });
        let other = fingerprint(&Args {
            query: "q",
            limit: 20,
        });
        assert_ne!(fp, other);
        let cursor = encode(&fp, 10);
        assert_eq!(decode(&cursor, &other), Err(CursorError::ArgumentsChanged));
        assert_eq!(decode("not a cursor!", &fp), Err(CursorError::Malformed));
        assert_eq!(
            decode(&URL_SAFE_NO_PAD.encode(b"{}"), &fp),
            Err(CursorError::Malformed)
        );
    }
}
