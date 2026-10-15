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
//! local ONNX backends. [`store`] defines the [`IndexStore`] trait and its
//! LanceDB backend, and [`filter`] the metadata filter shared by every
//! search side. [`ingest`] ties them together into the `index` run, and
//! [`search`] runs vector, keyword and hybrid searches with reranking, and
//! [`cursor`] pages ranked search results through stable server-side cursors.
//! [`lines`] cuts line ranges out of a paper with Python slice semantics.
//! [`server`] serves the MCP tools over bearer-authenticated Streamable HTTP.

pub mod chunk;
pub mod cursor;
pub mod error;
pub mod filter;
pub mod format;
pub mod ingest;
pub mod lines;
pub mod model;
pub mod prefix;
pub mod search;
pub mod selection;
pub mod server;
pub mod store;

pub use chunk::{CHUNKER_VERSION, Chunk, Section, chunk_markdown, outline};
pub use cursor::{
    Clock, CursorCache, CursorConfig, CursorError, DEFAULT_CURSOR_CAPACITY, DEFAULT_CURSOR_TTL,
    DEFAULT_PAGE_SIZE, Page, SystemClock,
};
pub use error::{Error, Result};
pub use filter::SearchFilter;
pub use format::{IndexFormatVersion, content_hash};
pub use ingest::{
    IndexLock, IndexOptions, IndexReport, Outcome, PaperReport, TOOL_NAME, index_papers,
};
pub use lines::{LineSlice, slice_lines};
pub use model::onnx::{OnnxEmbedder, OnnxReranker};
pub use model::{Embedder, MODEL_DIR_ENV, Reranker, resolve_model_dir};
pub use prefix::{PREFIX_VERSION, build_prefix, prefixed_text};
pub use search::{
    DEFAULT_RERANK_DEPTH, DEFAULT_TOP_K, RRF_K, SearchHit, SearchMode, SearchParams, SearchRequest,
    effective_filter, search_chunks, search_metadata,
};
pub use selection::{Selector, parse_selectors, resolve, resolve_latest};
pub use server::{BearerToken, MCP_PATH, PaperServer, ServeConfig, ServeOptions, build_app, serve};
pub use store::lance::{LanceStore, LanceStoreOptions};
pub use store::{
    ChunkHit, ChunkRecord, IndexStore, MetaHit, MetaRecord, PaperFields, PaperRecords,
};
