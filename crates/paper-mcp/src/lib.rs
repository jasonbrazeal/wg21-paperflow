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

pub mod chunk;
pub mod error;
pub mod selection;

pub use chunk::{CHUNKER_VERSION, Chunk, Section, chunk_markdown, outline};
pub use error::{Error, Result};
pub use selection::{Selector, parse_selectors, resolve, resolve_latest};
