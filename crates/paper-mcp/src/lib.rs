// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Index WG21 papers held in paperstore and serve search over MCP.
//!
//! The library holds the pieces shared by the `index` and `serve`
//! subcommands of the `paper-mcp` binary. [`selection`] turns operator
//! selectors (paper ids, a year, or `all`) into paper numbers through the
//! [`paperstore::StorageBackend`] abstraction. [`chunk`] splits a paper's
//! tomd Markdown into heading-scoped chunks with exact source line ranges.
//! [`prefix`] builds the deterministic paper and section context prepended
//! to each chunk for embedding and BM25, and [`format`](mod@format) provides the content
//! hash and index-format version that decide when a paper is stale.
//! [`model`] defines the [`Embedder`] and [`Reranker`] traits and their
//! local ONNX backends.

pub mod chunk;
pub mod error;
pub mod format;
pub mod model;
pub mod prefix;
pub mod selection;

pub use chunk::{CHUNKER_VERSION, Chunk, Section, chunk_markdown, outline};
pub use error::{Error, Result};
pub use format::{IndexFormatVersion, content_hash};
pub use model::onnx::{OnnxEmbedder, OnnxReranker};
pub use model::{Embedder, MODEL_DIR_ENV, Reranker, resolve_model_dir};
pub use prefix::{PREFIX_VERSION, build_prefix, prefixed_text};
pub use selection::{Selector, parse_selectors, resolve, resolve_latest};
