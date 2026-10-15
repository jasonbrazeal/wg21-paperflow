// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Error and result types for the paper-mcp library.

use std::path::PathBuf;

/// Errors returned by the paper-mcp library.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// No selector was given.
    #[error("at least one selector is required: paper ids, a year, or `all`")]
    EmptySelection,

    /// `all` was combined with other selectors.
    #[error("`all` cannot be combined with other selectors")]
    AllWithOthers,

    /// A selector was neither `all`, a four-digit year, nor a paper id.
    #[error("{0:?} is not a paper id, a four-digit year, or `all`")]
    BadSelector(String),

    /// Neither `--model-dir` nor the model directory environment variable is set.
    #[error("no model directory: pass --model-dir or set PAPER_MCP_MODEL_DIR")]
    NoModelDir,

    /// Required model files are absent from a model directory.
    #[error(
        "model directory {} is missing {}; it must contain {}",
        dir.display(),
        missing.join(", "),
        expected.join(", ")
    )]
    MissingModelFiles {
        /// The model directory that was searched.
        dir: PathBuf,
        /// File names, relative to `dir`, that were not found.
        missing: Vec<String>,
        /// Every required file name, relative to `dir`.
        expected: Vec<String>,
    },

    /// A model file exists but could not be read.
    #[error("cannot read model file {}: {source}", path.display())]
    ModelFileRead {
        /// The unreadable file.
        path: PathBuf,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// A model configuration file is malformed or lacks a required value.
    #[error("invalid model file {}: {reason}", path.display())]
    ModelConfig {
        /// The offending file.
        path: PathBuf,
        /// What is wrong with it.
        reason: String,
    },

    /// The inference runtime rejected the model files in a model directory.
    #[error("cannot load model from {}: {message}", dir.display())]
    ModelLoad {
        /// The model directory.
        dir: PathBuf,
        /// The runtime's error message.
        message: String,
    },

    /// Embedding or reranking failed at inference time.
    #[error("model inference failed: {0}")]
    Inference(String),

    /// An index store operation failed or found inconsistent data.
    #[error("index store error: {0}")]
    Index(String),

    /// A paperstore operation failed.
    #[error(transparent)]
    Store(#[from] paperstore::Error),
}

/// Result alias for paper-mcp library operations.
pub type Result<T> = std::result::Result<T, Error>;
