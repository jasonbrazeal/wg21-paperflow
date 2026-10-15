// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! The index store: chunk and paper-metadata records and their searches.
//!
//! [`IndexStore`] is the backend-neutral interface the ingest and search
//! pipelines use; [`lance::LanceStore`] implements it over embedded
//! LanceDB. One paper revision owns one [`MetaRecord`] and zero or more
//! [`ChunkRecord`]s, written and replaced together by
//! [`IndexStore::upsert_paper`].

use std::future::Future;

use paperstore::{PaperNum, PaperRow};

use crate::error::Result;
use crate::filter::SearchFilter;

pub mod lance;

/// Paper-level fields copied onto every chunk, for filters and results.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PaperFields {
    /// Paper title.
    pub title: String,
    /// Author names.
    pub authors: Vec<String>,
    /// Target subgroup.
    pub target_group: String,
    /// Mailing year.
    pub year: String,
    /// Mailing date (`YYYY-MM`).
    pub mailing_date: String,
    /// Document date.
    pub document_date: String,
    /// Intent.
    pub intent: String,
}

impl PaperFields {
    /// Copy the chunk-level fields out of a paperstore row.
    #[must_use]
    pub fn from_row(row: &PaperRow) -> Self {
        Self {
            title: row.title.clone(),
            authors: row.authors.clone(),
            target_group: row.target_group.clone(),
            year: row.year.clone(),
            mailing_date: row.mailing_date.clone(),
            document_date: row.document_date.clone(),
            intent: row.intent.clone(),
        }
    }
}

/// One indexed chunk of one paper revision.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkRecord {
    /// Paper id with revision.
    pub paper_id: PaperNum,
    /// Revision, `None` for unversioned papers.
    pub revision: Option<u32>,
    /// Paper number without revision, shared by every revision.
    pub paper_number: String,
    /// Whether this revision is the latest of its paper number.
    pub is_latest: bool,
    /// Headings enclosing the chunk, outermost first.
    pub heading_path: Vec<String>,
    /// First source line, 1-based and inclusive.
    pub start: usize,
    /// Source line after the chunk, 1-based and exclusive.
    pub end: usize,
    /// Chunk text as returned to clients, without the prefix.
    pub text: String,
    /// Prefix plus chunk text, as embedded and keyword-indexed.
    pub prefixed_text: String,
    /// Embedding of `prefixed_text`. Search results carry an empty vector.
    pub vector: Vec<f32>,
    /// Content hash of the Markdown the chunk was cut from.
    pub content_hash: String,
    /// Index-format version the chunk was built with.
    pub format_version: String,
    /// Paper-level fields for filters and results.
    pub fields: PaperFields,
}

/// The paper-level record of one paper revision.
#[derive(Clone, Debug)]
pub struct MetaRecord {
    /// The full paperstore metadata row.
    pub row: PaperRow,
    /// Whether this revision is the latest of its paper number.
    pub is_latest: bool,
    /// Whether the paper had Markdown when indexed.
    pub has_markdown: bool,
    /// Content hash of the indexed Markdown, empty without Markdown.
    pub content_hash: String,
    /// Index-format version the record was built with.
    pub format_version: String,
}

/// Every record of one paper revision, written together.
#[derive(Clone, Debug)]
pub struct PaperRecords {
    /// The paper-level record; its row names the revision.
    pub meta: MetaRecord,
    /// The revision's chunks in document order.
    pub chunks: Vec<ChunkRecord>,
}

/// A chunk returned by a search.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkHit {
    /// The chunk, with an empty `vector`.
    pub chunk: ChunkRecord,
    /// Relevance; higher is better. Vector hits score cosine similarity and
    /// keyword hits score BM25, so scores compare only within one search.
    pub score: f32,
}

/// A paper-level record returned by a metadata search.
#[derive(Clone, Debug)]
pub struct MetaHit {
    /// The record.
    pub meta: MetaRecord,
    /// BM25 relevance on title and authors; `0.0` when the search had no
    /// keywords.
    pub score: f32,
}

/// Storage for indexed chunks and paper-level metadata.
pub trait IndexStore: Send + Sync {
    /// Return the content hash and index-format version recorded for a revision.
    ///
    /// Returns `None` when the revision is not indexed.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Index`] when the store cannot be read.
    fn freshness(
        &self,
        paper: &PaperNum,
    ) -> impl Future<Output = Result<Option<(String, String)>>> + Send;

    /// Replace every record of one revision with `records`.
    ///
    /// Chunks of the revision absent from `records` are removed.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Index`] when a chunk belongs to another paper,
    /// a vector has the wrong length, or the write fails.
    fn upsert_paper(&self, records: PaperRecords) -> impl Future<Output = Result<()>> + Send;

    /// Remove every record of one revision.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Index`] when the write fails.
    fn delete_paper(&self, paper: &PaperNum) -> impl Future<Output = Result<()>> + Send;

    /// Flag `latest` as the latest revision of `paper_number` and every other
    /// indexed revision of it as not latest.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Index`] when the write fails.
    fn set_latest(
        &self,
        paper_number: &str,
        latest: &PaperNum,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Return the `limit` chunks nearest to `vector` that match `filter`.
    ///
    /// Hits are ordered by descending score, ties by paper id and position.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Index`] when the search fails.
    fn search_vector(
        &self,
        vector: &[f32],
        filter: &SearchFilter,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<ChunkHit>>> + Send;

    /// Return every chunk matching `query` by keyword and matching `filter`.
    ///
    /// Hits are ordered by descending score, ties by paper id and position.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Index`] when the search fails.
    fn search_keyword(
        &self,
        query: &str,
        filter: &SearchFilter,
    ) -> impl Future<Output = Result<Vec<ChunkHit>>> + Send;

    /// Return how many chunks match `query` by keyword and match `filter`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Index`] when the search fails.
    fn count_keyword_matches(
        &self,
        query: &str,
        filter: &SearchFilter,
    ) -> impl Future<Output = Result<usize>> + Send;

    /// Return every paper-level record matching `query` on title or authors
    /// and matching `filter`.
    ///
    /// A query without keywords matches every record that passes `filter`,
    /// ordered by paper id. Otherwise hits are ordered by descending score,
    /// ties by paper id.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Index`] when the search fails.
    fn search_metadata(
        &self,
        query: &str,
        filter: &SearchFilter,
    ) -> impl Future<Output = Result<Vec<MetaHit>>> + Send;

    /// Compact storage and bring every index up to date with written rows.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Index`] when optimization fails.
    fn optimize(&self) -> impl Future<Output = Result<()>> + Send;
}
