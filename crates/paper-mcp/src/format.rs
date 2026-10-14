// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Content hashing and the index-format version.
//!
//! An indexed paper is fresh when both its content hash and the index-format
//! version recorded with it match the current ones; any mismatch marks it
//! stale.

use std::fmt::{self, Write as _};

use sha2::{Digest as _, Sha256};

use crate::chunk::CHUNKER_VERSION;
use crate::prefix::PREFIX_VERSION;

/// Return the SHA-256 digest of `md` as 64 lowercase hex digits.
#[must_use]
pub fn content_hash(md: &str) -> String {
    Sha256::digest(md.as_bytes())
        .iter()
        .fold(String::new(), |mut hex, byte| {
            // Writing to a String cannot fail.
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

/// Identity of everything that shapes indexed records.
///
/// Its [`Display`](fmt::Display) rendering is the stable string stored in
/// index records:
/// `chunker=<n>;prefix=<n>;embedder=<len>:<id>;reranker=<len>:<id>`, where
/// `<len>` is the id's length in bytes. The length prefixes keep distinct
/// versions from rendering the same even when an id contains separators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexFormatVersion {
    /// Chunking rules version, normally [`CHUNKER_VERSION`].
    pub chunker: u32,
    /// Prefix layout version, normally [`PREFIX_VERSION`].
    pub prefix: u32,
    /// Identity of the embedding model.
    pub embedder_id: String,
    /// Identity of the reranking model.
    pub reranker_id: String,
}

impl IndexFormatVersion {
    /// Return the version for this build's chunker and prefix with the given models.
    #[must_use]
    pub fn current(embedder_id: &str, reranker_id: &str) -> Self {
        Self {
            chunker: CHUNKER_VERSION,
            prefix: PREFIX_VERSION,
            embedder_id: embedder_id.to_owned(),
            reranker_id: reranker_id.to_owned(),
        }
    }
}

impl fmt::Display for IndexFormatVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "chunker={};prefix={};embedder={}:{};reranker={}:{}",
            self.chunker,
            self.prefix,
            self.embedder_id.len(),
            self.embedder_id,
            self.reranker_id.len(),
            self.reranker_id,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version() -> IndexFormatVersion {
        IndexFormatVersion {
            chunker: 1,
            prefix: 1,
            embedder_id: "bge-small-en-v1.5".to_owned(),
            reranker_id: "bge-reranker-base".to_owned(),
        }
    }

    #[test]
    fn content_hash_matches_known_sha256() {
        assert_eq!(
            content_hash(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            content_hash("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn content_hash_is_stable_and_input_sensitive() {
        let md = "## Intro\n\nText.\n";
        assert_eq!(content_hash(md), content_hash(md));
        assert_ne!(content_hash(md), content_hash("## Intro\n\nText.\n\n"));
    }

    #[test]
    fn current_uses_module_versions() {
        let v = IndexFormatVersion::current("e", "r");
        assert_eq!(v.chunker, CHUNKER_VERSION);
        assert_eq!(v.prefix, PREFIX_VERSION);
        assert_eq!(v.embedder_id, "e");
        assert_eq!(v.reranker_id, "r");
    }

    #[test]
    fn rendering_is_fixed() {
        assert_eq!(
            version().to_string(),
            "chunker=1;prefix=1;embedder=17:bge-small-en-v1.5;reranker=17:bge-reranker-base"
        );
    }

    #[test]
    fn changing_any_component_changes_rendering() {
        let base = version().to_string();
        let variants = [
            IndexFormatVersion {
                chunker: 2,
                ..version()
            },
            IndexFormatVersion {
                prefix: 2,
                ..version()
            },
            IndexFormatVersion {
                embedder_id: "other".to_owned(),
                ..version()
            },
            IndexFormatVersion {
                reranker_id: "other".to_owned(),
                ..version()
            },
        ];
        for v in variants {
            assert_ne!(v.to_string(), base, "{v:?}");
        }
    }

    #[test]
    fn ids_containing_separators_cannot_collide() {
        let a = IndexFormatVersion {
            embedder_id: "e;reranker=1:x".to_owned(),
            reranker_id: "y".to_owned(),
            ..version()
        };
        let b = IndexFormatVersion {
            embedder_id: "e".to_owned(),
            reranker_id: "x;reranker=1:y".to_owned(),
            ..version()
        };
        assert_ne!(a.to_string(), b.to_string());
    }
}
