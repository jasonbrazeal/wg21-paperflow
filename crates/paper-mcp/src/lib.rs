// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Index WG21 papers held in paperstore and serve search over MCP.
//!
//! The library holds the pieces shared by the `index` and `serve`
//! subcommands of the `paper-mcp` binary. [`selection`] turns operator
//! selectors (paper ids, a year, or `all`) into paper numbers through the
//! [`paperstore::StorageBackend`] abstraction.

pub mod error;
pub mod selection;

pub use error::{Error, Result};
pub use selection::{Selector, parse_selectors, resolve, resolve_latest};
