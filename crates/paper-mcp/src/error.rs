// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Error and result types for the paper-mcp library.

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

    /// A paperstore operation failed.
    #[error(transparent)]
    Store(#[from] paperstore::Error),
}

/// Result alias for paper-mcp library operations.
pub type Result<T> = std::result::Result<T, Error>;
