//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Text embedding: the [`Embedder`] trait and its implementations.
//!
//! [`FastEmbedder`] is the production embedder, running
//! `nomic-embed-text-v1.5` locally through fastembed. [`HashEmbedder`] is a
//! deterministic stand-in for tests that never loads a model.

use std::path::PathBuf;
use std::sync::Mutex;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use sha2::{Digest, Sha256};

/// Task prefix nomic-embed-text-v1.5 expects on indexed documents.
pub const DOCUMENT_PREFIX: &str = "search_document: ";

/// Task prefix nomic-embed-text-v1.5 expects on search queries.
pub const QUERY_PREFIX: &str = "search_query: ";

/// Model identifier recorded in the index manifest for [`FastEmbedder`].
pub const FAST_EMBEDDER_MODEL_ID: &str = "nomic-ai/nomic-embed-text-v1.5";

/// Output dimension of nomic-embed-text-v1.5.
pub const FAST_EMBEDDER_DIM: usize = 768;

/// Tokenizer truncation length for [`FastEmbedder`] inputs.
///
/// Chunks are capped at 1024 tokens before the contextual prefix and task
/// prefix are added; this leaves headroom so no indexed text is truncated.
pub const FAST_EMBEDDER_MAX_LENGTH: usize = 2048;

/// Texts per inference batch in [`FastEmbedder::embed_documents`].
pub const FAST_EMBEDDER_BATCH_SIZE: usize = 16;

/// Environment variable that, set to [`TEST_EMBEDDER_HASH`], makes the
/// `paper-mcp` binary use a [`HashEmbedder`] instead of loading the model.
/// For end-to-end tests only.
#[doc(hidden)]
pub const TEST_EMBEDDER_ENV: &str = "PAPER_MCP_TEST_EMBEDDER";

/// Value of [`TEST_EMBEDDER_ENV`] selecting the test [`HashEmbedder`].
#[doc(hidden)]
pub const TEST_EMBEDDER_HASH: &str = "hash";

/// Model identifier of the binary's test [`HashEmbedder`].
#[doc(hidden)]
pub const TEST_EMBEDDER_MODEL_ID: &str = "test/hash";

/// Output dimension of the binary's test [`HashEmbedder`].
#[doc(hidden)]
pub const TEST_EMBEDDER_DIM: usize = 16;

/// Errors raised while producing embeddings.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// The underlying embedding model failed to load or run.
    #[error("embedding model error: {0}")]
    Model(#[from] fastembed::Error),
    /// A previous embedding call panicked while holding the model lock.
    #[error("embedding model lock poisoned")]
    Poisoned,
    /// The model returned a different number of vectors than inputs.
    #[error("embedding model returned {actual} vectors for {expected} inputs")]
    CountMismatch {
        /// Number of input texts.
        expected: usize,
        /// Number of vectors returned.
        actual: usize,
    },
    /// The model returned a vector of an unexpected dimension.
    #[error("embedding model returned a {actual}-dim vector, expected {expected}")]
    DimMismatch {
        /// Declared dimension.
        expected: usize,
        /// Dimension returned.
        actual: usize,
    },
}

/// A text embedding model.
///
/// Documents and queries are embedded separately so asymmetric models can
/// apply their task-specific input formatting.
pub trait Embedder: Send + Sync {
    /// Stable identifier of the model, recorded in the index manifest.
    fn model_id(&self) -> &str;

    /// Length of every vector this embedder returns.
    fn dim(&self) -> usize;

    /// Embed texts for indexing, returning one vector per text in order.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedError`] when the model fails or returns malformed
    /// output.
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError>;

    /// Embed a search query.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedError`] when the model fails or returns malformed
    /// output.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
}

/// Local nomic-embed-text-v1.5 embedder backed by fastembed.
pub struct FastEmbedder {
    model: Mutex<TextEmbedding>,
}

impl std::fmt::Debug for FastEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedder")
            .field("model_id", &FAST_EMBEDDER_MODEL_ID)
            .finish_non_exhaustive()
    }
}

impl FastEmbedder {
    /// Load the model, downloading it into `cache_dir` on first use.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedError::Model`] when the model cannot be downloaded or
    /// loaded.
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self, EmbedError> {
        let mut options = TextInitOptions::new(EmbeddingModel::NomicEmbedTextV15)
            .with_max_length(FAST_EMBEDDER_MAX_LENGTH)
            .with_show_download_progress(false);
        if let Some(dir) = cache_dir {
            options = options.with_cache_dir(dir);
        }
        let model = TextEmbedding::try_new(options)?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    /// Format a document as the model input for indexing.
    #[must_use]
    pub fn document_input(text: &str) -> String {
        format!("{DOCUMENT_PREFIX}{text}")
    }

    /// Format a query as the model input for search.
    #[must_use]
    pub fn query_input(text: &str) -> String {
        format!("{QUERY_PREFIX}{text}")
    }

    fn embed_inputs(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let vectors = {
            let mut model = self.model.lock().map_err(|_| EmbedError::Poisoned)?;
            model.embed(inputs, Some(FAST_EMBEDDER_BATCH_SIZE))?
        };
        check_shape(&vectors, inputs.len(), FAST_EMBEDDER_DIM)?;
        Ok(vectors)
    }
}

impl Embedder for FastEmbedder {
    fn model_id(&self) -> &str {
        FAST_EMBEDDER_MODEL_ID
    }

    fn dim(&self) -> usize {
        FAST_EMBEDDER_DIM
    }

    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let inputs: Vec<String> = texts.iter().map(|t| Self::document_input(t)).collect();
        self.embed_inputs(&inputs)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let inputs = [Self::query_input(text)];
        let mut vectors = self.embed_inputs(&inputs)?;
        vectors.pop().ok_or(EmbedError::CountMismatch {
            expected: 1,
            actual: 0,
        })
    }
}

fn check_shape(vectors: &[Vec<f32>], expected: usize, dim: usize) -> Result<(), EmbedError> {
    if vectors.len() != expected {
        return Err(EmbedError::CountMismatch {
            expected,
            actual: vectors.len(),
        });
    }
    if let Some(bad) = vectors.iter().find(|v| v.len() != dim) {
        return Err(EmbedError::DimMismatch {
            expected: dim,
            actual: bad.len(),
        });
    }
    Ok(())
}

/// Deterministic test embedder deriving vectors from SHA-256 of the input.
///
/// Equal inputs always produce equal unit-length vectors and different
/// inputs produce different vectors. Vectors carry no semantic similarity.
/// Documents and queries embed identically.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    model_id: String,
    dim: usize,
}

impl HashEmbedder {
    /// Create an embedder reporting `model_id` and producing `dim`-length
    /// vectors.
    #[must_use]
    pub fn new(model_id: impl Into<String>, dim: usize) -> Self {
        Self {
            model_id: model_id.into(),
            dim,
        }
    }

    fn vector(&self, text: &str) -> Vec<f32> {
        let mut values = Vec::with_capacity(self.dim);
        let mut block: u64 = 0;
        while values.len() < self.dim {
            let digest = Sha256::new()
                .chain_update(block.to_le_bytes())
                .chain_update(text.as_bytes())
                .finalize();
            for word in digest.as_chunks::<4>().0 {
                if values.len() == self.dim {
                    break;
                }
                let bits = u32::from_le_bytes(*word);
                values.push(f64::from(bits) / f64::from(u32::MAX) * 2.0 - 1.0);
            }
            block += 1;
        }
        let norm = values.iter().map(|x| x * x).sum::<f64>().sqrt();
        #[allow(clippy::cast_possible_truncation)]
        let unit = values
            .into_iter()
            .map(|x| if norm > 0.0 { (x / norm) as f32 } else { 0.0 })
            .collect();
        unit
    }
}

impl Embedder for HashEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts.iter().map(|t| self.vector(t)).collect())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(self.vector(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_MODEL_ID: &str = "test/hash";

    fn texts(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn hash_embedder_is_deterministic() -> Result<(), EmbedError> {
        let embedder = HashEmbedder::new(TEST_MODEL_ID, 64);
        let input = texts(&["coroutines", "contracts"]);
        let first = embedder.embed_documents(&input)?;
        let second = HashEmbedder::new(TEST_MODEL_ID, 64).embed_documents(&input)?;
        assert_eq!(first, second);
        assert_eq!(embedder.embed_query("coroutines")?, first[0]);
        Ok(())
    }

    #[test]
    fn hash_embedder_has_declared_dim_and_model_id() -> Result<(), EmbedError> {
        for dim in [1, 7, 8, 9, 64, 768] {
            let embedder = HashEmbedder::new(TEST_MODEL_ID, dim);
            assert_eq!(embedder.dim(), dim);
            assert_eq!(embedder.model_id(), TEST_MODEL_ID);
            let vectors = embedder.embed_documents(&texts(&["a", "b", "c"]))?;
            assert_eq!(vectors.len(), 3);
            assert!(vectors.iter().all(|v| v.len() == dim));
            assert_eq!(embedder.embed_query("a")?.len(), dim);
        }
        Ok(())
    }

    #[test]
    fn hash_embedder_distinguishes_inputs() -> Result<(), EmbedError> {
        let embedder = HashEmbedder::new(TEST_MODEL_ID, 32);
        let vectors = embedder.embed_documents(&texts(&["P2300R10", "P2300R9", "p2300r10", ""]))?;
        for (i, a) in vectors.iter().enumerate() {
            for b in &vectors[i + 1..] {
                assert_ne!(a, b);
            }
        }
        Ok(())
    }

    #[test]
    fn hash_embedder_vectors_are_unit_length() -> Result<(), EmbedError> {
        let vector = HashEmbedder::new(TEST_MODEL_ID, 48).embed_query("executors")?;
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
        Ok(())
    }

    #[test]
    fn fast_embedder_applies_task_prefixes() {
        assert_eq!(
            FastEmbedder::document_input("Wording for std::execution"),
            "search_document: Wording for std::execution"
        );
        assert_eq!(
            FastEmbedder::query_input("sender adaptors"),
            "search_query: sender adaptors"
        );
    }

    #[test]
    fn check_shape_rejects_wrong_count_and_dim() {
        let vectors = vec![vec![0.0_f32; 4], vec![0.0_f32; 4]];
        assert!(check_shape(&vectors, 2, 4).is_ok());
        assert!(matches!(
            check_shape(&vectors, 3, 4),
            Err(EmbedError::CountMismatch {
                expected: 3,
                actual: 2
            })
        ));
        assert!(matches!(
            check_shape(&vectors, 2, 5),
            Err(EmbedError::DimMismatch {
                expected: 5,
                actual: 4
            })
        ));
    }
}
