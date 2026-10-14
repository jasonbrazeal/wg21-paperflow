// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Embedding and reranking models behind backend-neutral traits.
//!
//! [`onnx`] holds the production backends, which load open-weight models
//! from a local model directory and never download. The `fake` module,
//! built for tests and under the `test-models` feature, holds deterministic
//! stand-ins that need no model files.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

#[cfg(any(test, feature = "test-models"))]
pub mod fake;
pub mod onnx;

/// Environment variable naming the model directory when `--model-dir` is absent.
pub const MODEL_DIR_ENV: &str = "PAPER_MCP_MODEL_DIR";

/// A text embedding model.
pub trait Embedder: Send + Sync {
    /// Return the model identity recorded in the index-format version.
    ///
    /// Two embedders with the same id must produce the same vectors.
    fn id(&self) -> &str;

    /// Return the length of every vector this embedder produces.
    fn dim(&self) -> usize;

    /// Return the largest input, in tokens, the model embeds without truncation.
    fn max_tokens(&self) -> usize;

    /// Return the number of model tokens in `text`, special tokens included.
    fn count_tokens(&self, text: &str) -> usize;

    /// Embed document texts, returning one vector per text in input order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Inference`] when the model fails.
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Embed a search query, applying any query-side instruction the model expects.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Inference`] when the model fails.
    fn embed_query(&self, query: &str) -> Result<Vec<f32>>;
}

/// A query-document relevance model.
pub trait Reranker: Send + Sync {
    /// Return the model identity recorded in the index-format version.
    fn id(&self) -> &str;

    /// Score each document against `query`.
    ///
    /// Returns one score per document, in input order; a higher score means
    /// more relevant. Scores are comparable only within one call.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Inference`] when the model fails.
    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>>;
}

/// Return the model directory: `flag` when given, otherwise [`MODEL_DIR_ENV`].
///
/// An empty environment value counts as unset.
///
/// # Errors
///
/// Returns [`Error::NoModelDir`] when neither is set.
pub fn resolve_model_dir(flag: Option<&Path>) -> Result<PathBuf> {
    if let Some(dir) = flag {
        return Ok(dir.to_path_buf());
    }
    std::env::var_os(MODEL_DIR_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .ok_or(Error::NoModelDir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_wins_over_env() {
        temp_env::with_var(MODEL_DIR_ENV, Some("/from/env"), || {
            let dir = resolve_model_dir(Some(Path::new("/from/flag"))).ok();
            assert_eq!(dir, Some(PathBuf::from("/from/flag")));
        });
    }

    #[test]
    fn env_used_without_flag() {
        temp_env::with_var(MODEL_DIR_ENV, Some("/from/env"), || {
            assert_eq!(
                resolve_model_dir(None).ok(),
                Some(PathBuf::from("/from/env"))
            );
        });
    }

    #[test]
    fn neither_is_an_error() {
        for unset in [None, Some("")] {
            temp_env::with_var(MODEL_DIR_ENV, unset, || {
                assert!(matches!(resolve_model_dir(None), Err(Error::NoModelDir)));
            });
        }
    }
}
