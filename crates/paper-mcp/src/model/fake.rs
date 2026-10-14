// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Deterministic stand-in models for tests and CI.
//!
//! They need no model files and produce the same output on every run and
//! platform, so tests that index and search can assert exact results.

use crate::error::Result;
use crate::model::{Embedder, Reranker};

/// Version of the hashing scheme, part of [`HashEmbedder`] ids.
const HASH_EMBEDDER_VERSION: u32 = 1;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Vector length of [`HashEmbedder::default`].
pub const DEFAULT_HASH_DIM: usize = 64;

/// Token budget of [`HashEmbedder::default`].
pub const DEFAULT_HASH_MAX_TOKENS: usize = 512;

/// Embedder that hashes each token into a signed bucket of a unit vector.
///
/// A token is a maximal run of alphanumeric characters or `_`, lowercased.
/// Each token adds plus or minus one to the bucket picked by its FNV-1a
/// hash, and the result is scaled to unit length, so texts sharing tokens
/// land close together. Text without tokens embeds as the zero vector.
#[derive(Clone, Debug)]
pub struct HashEmbedder {
    id: String,
    dim: usize,
    max_tokens: usize,
}

impl HashEmbedder {
    /// Return a hash embedder producing `dim`-long vectors.
    ///
    /// # Panics
    ///
    /// Panics when `dim` is zero.
    #[must_use]
    pub fn new(dim: usize, max_tokens: usize) -> Self {
        assert!(dim > 0, "HashEmbedder dimension must be positive");
        Self {
            id: format!("hash-embedder-v{HASH_EMBEDDER_VERSION}-d{dim}"),
            dim,
            max_tokens,
        }
    }
}

impl HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0_f32; self.dim];
        let buckets = u64::try_from(self.dim).unwrap_or(u64::MAX);
        for token in tokens(text) {
            let h = fnv1a(token.as_bytes());
            let bucket = usize::try_from(h % buckets).unwrap_or(0);
            v[bucket] += if h >> 63 == 1 { -1.0 } else { 1.0 };
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new(DEFAULT_HASH_DIM, DEFAULT_HASH_MAX_TOKENS)
    }
}

impl Embedder for HashEmbedder {
    fn id(&self) -> &str {
        &self.id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn count_tokens(&self, text: &str) -> usize {
        tokens(text).count()
    }

    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed(t)).collect())
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        Ok(self.embed(query))
    }
}

/// Reranker that keeps the input order: scores strictly decrease by position.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityReranker;

impl Reranker for IdentityReranker {
    fn id(&self) -> &'static str {
        "identity-reranker-v1"
    }

    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        let _ = query;
        let mut score = 0.0_f32;
        Ok(documents
            .iter()
            .map(|_| {
                let s = score;
                score -= 1.0;
                s
            })
            .collect())
    }
}

/// Yield the lowercased tokens of `text`.
fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET_BASIS, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(FNV_PRIME)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs(texts: &[&str]) -> Vec<String> {
        texts.iter().map(|t| (*t).to_owned()).collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn embed_one(e: &HashEmbedder, text: &str) -> Vec<f32> {
        e.embed_documents(&docs(&[text]))
            .ok()
            .and_then(|mut v| v.pop())
            .unwrap_or_default()
    }

    #[test]
    fn vectors_have_the_declared_dimension() {
        for dim in [1, 7, DEFAULT_HASH_DIM, 384] {
            let e = HashEmbedder::new(dim, DEFAULT_HASH_MAX_TOKENS);
            assert_eq!(e.dim(), dim);
            let vs = e
                .embed_documents(&docs(&["alpha beta", "", "std::move_only_function"]))
                .ok()
                .unwrap_or_default();
            assert_eq!(vs.len(), 3);
            assert!(vs.iter().all(|v| v.len() == dim), "dim {dim}");
            assert_eq!(e.embed_query("alpha").ok().map(|v| v.len()), Some(dim));
        }
    }

    #[test]
    fn embedding_is_deterministic_and_pinned() {
        let a = HashEmbedder::default();
        let b = HashEmbedder::default();
        let text = "Contracts for std::move_only_function";
        assert_eq!(embed_one(&a, text), embed_one(&b, text));
        assert_eq!(a.embed_query(text).ok(), Some(embed_one(&a, text)));
        // Pin the bucket layout so a hashing change cannot slip through
        // unnoticed: FNV-1a of "x" is 0xaf63_f54c_8602_1707, so bucket
        // 3 of 4 and, with the top bit set, a negative sign.
        let v = embed_one(&HashEmbedder::new(4, 8), "x");
        assert_eq!(v, vec![0.0, 0.0, 0.0, -1.0]);
    }

    #[test]
    fn vectors_are_unit_length_or_zero() {
        let e = HashEmbedder::default();
        let v = embed_one(&e, "alpha beta gamma alpha");
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-5);
        assert!(embed_one(&e, " ... ").iter().all(|x| *x == 0.0));
    }

    #[test]
    fn lexically_similar_texts_land_closer() {
        let e = HashEmbedder::default();
        let q = e
            .embed_query("executor scheduler sender")
            .ok()
            .unwrap_or_default();
        let near = embed_one(&e, "the sender and scheduler of an executor");
        let far = embed_one(&e, "unicode normalization of identifiers");
        assert!(cosine(&q, &near) > cosine(&q, &far));
    }

    #[test]
    fn tokens_are_case_insensitive_and_counted() {
        let e = HashEmbedder::default();
        assert_eq!(embed_one(&e, "Alpha BETA"), embed_one(&e, "alpha beta"));
        assert_eq!(e.count_tokens("std::move_only_function, x2!"), 3);
        assert_eq!(e.count_tokens(""), 0);
    }

    #[test]
    fn id_names_the_dimension() {
        assert_ne!(HashEmbedder::new(8, 8).id(), HashEmbedder::new(16, 8).id());
        assert_eq!(
            HashEmbedder::default().max_tokens(),
            DEFAULT_HASH_MAX_TOKENS
        );
    }

    #[test]
    fn identity_reranker_keeps_order() {
        let scores = IdentityReranker
            .rerank("q", &docs(&["a", "b", "c", "d"]))
            .ok()
            .unwrap_or_default();
        assert_eq!(scores.len(), 4);
        assert!(scores.windows(2).all(|w| w[0] > w[1]));
        assert_eq!(
            IdentityReranker
                .rerank("other", &docs(&["a", "b", "c", "d"]))
                .ok(),
            Some(scores)
        );
        assert_eq!(IdentityReranker.rerank("q", &[]).ok(), Some(vec![]));
    }
}
