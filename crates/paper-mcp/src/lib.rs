//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Search and read the WG21 paper corpus over MCP.
//!
//! Indexes the paper Markdown stored in paperstore into a LanceDB table
//! supporting keyword, vector, and hybrid search with metadata filters, and
//! exposes search, paper text, and metadata as MCP tools over Streamable
//! HTTP.

pub mod auth;
pub mod chunk;
pub mod embed;
pub mod filter;
pub mod index;
pub mod lines;
pub mod revision;
pub mod search;
pub mod tools;
