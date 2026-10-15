// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! [`IndexStore`] over embedded LanceDB.
//!
//! The database lives in `lancedb/` under the tool directory and holds two
//! tables: `chunks`, one row per chunk, and `paper_meta`, one row per paper
//! revision. Keyword search runs over a whitespace-tokenized copy of the
//! prefixed chunk text (and of the title and authors), with the punctuation
//! Markdown wraps around words trimmed off each token, so C++ identifiers
//! such as `std::move_only_function` or `std::vector<int>` stay whole.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::types::Float32Type;
use arrow_array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Int64Array, ListArray,
    RecordBatch, RecordBatchIterator, StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt as _;
use lancedb::index::Index;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::vector::IvfHnswSqIndexBuilder;
use lancedb::query::{ExecutableQuery as _, QueryBase as _, Select};
use lancedb::table::OptimizeAction;
use lancedb::{Connection, DistanceType, Table, connect};
use paperstore::{PaperNum, PaperRow};

use crate::error::{Error, Result};
use crate::filter::{
    COL_AUTHORS_TEXT, COL_DOCUMENT_DATE, COL_INTENT, COL_IS_LATEST, COL_MAILING_DATE, COL_PAPER_ID,
    COL_PAPER_NUMBER, COL_REVISION, COL_TARGET_GROUP, COL_YEAR, SearchFilter, eq,
};
use crate::store::{
    ChunkHit, ChunkRecord, IndexStore, MetaHit, MetaRecord, PaperFields, PaperRecords,
};

/// Subdirectory of the tool directory holding the database.
pub const LANCEDB_DIR: &str = "lancedb";

/// Table of chunk records.
const CHUNKS_TABLE: &str = "chunks";

/// Table of paper-level metadata records.
const META_TABLE: &str = "paper_meta";

/// Default [`LanceStoreOptions::read_consistency_interval`].
pub const DEFAULT_READ_CONSISTENCY_INTERVAL: Duration = Duration::from_secs(5);

/// Default [`LanceStoreOptions::vector_index_min_rows`]. Below it an exact
/// scan is fast, and an approximate index would have too few rows to train
/// its partitions on.
pub const DEFAULT_VECTOR_INDEX_MIN_ROWS: usize = 10_000;

/// Base tokenizer of every full-text index: split on whitespace only.
const FTS_BASE_TOKENIZER: &str = "whitespace";

/// Whether full-text indexes lowercase tokens.
const FTS_LOWER_CASE: bool = true;

/// Characters trimmed from both ends of each whitespace-separated token
/// before keyword indexing and querying: Markdown emphasis and code marks,
/// quotes, brackets and sentence punctuation.
const KEYWORD_EDGE_PUNCTUATION: &[char] = &[
    '`', '*', '"', '\'', '\u{201c}', '\u{201d}', '\u{2018}', '\u{2019}', '(', ')', '[', ']', '{',
    '}', ',', '.', ';', ':', '!', '?', '|',
];

/// Separator between author names in the joined authors column.
const AUTHORS_SEPARATOR: &str = "; ";

/// Width of the zero-padded chunk position in chunk ids.
const CHUNK_ORDINAL_WIDTH: usize = 6;

const COL_CHUNK_ID: &str = "chunk_id";
const COL_HEADING_PATH: &str = "heading_path";
const COL_LINE_START: &str = "line_start";
const COL_LINE_END: &str = "line_end";
const COL_TEXT: &str = "text";
const COL_PREFIXED_TEXT: &str = "prefixed_text";
const COL_FTS_TEXT: &str = "fts_text";
const COL_VECTOR: &str = "vector";
const COL_CONTENT_HASH: &str = "content_hash";
const COL_FORMAT_VERSION: &str = "format_version";
const COL_TITLE: &str = "title";
const COL_AUTHORS: &str = "authors";
const COL_HAS_MARKDOWN: &str = "has_markdown";
const COL_URL: &str = "url";
const COL_DISPOSITION: &str = "disposition";
const COL_PREVIOUS_VERSION: &str = "previous_version";
const COL_SOURCE_FILE: &str = "source_file";
const COL_MARKDOWN_PATH: &str = "markdown_path";
const COL_DISSECT_PATH: &str = "dissect_path";
const COL_AGORA_PATH: &str = "agora_path";
const COL_ASSAY_PATH: &str = "assay_path";
const COL_CITATIONS_EXTRACTED_AT: &str = "citations_extracted_at";
const COL_ERROR: &str = "error";
const COL_LINE_COUNT: &str = "line_count";
const COL_STATUS: &str = "status";
const COL_FTS_TITLE: &str = "fts_title";
const COL_FTS_AUTHORS: &str = "fts_authors";

/// Column LanceDB adds to vector results: the distance to the query.
const COL_DISTANCE: &str = "_distance";

/// Column LanceDB adds to full-text results: the BM25 score.
const COL_SCORE: &str = "_score";

/// Tuning for a [`LanceStore`].
#[derive(Clone, Debug)]
pub struct LanceStoreOptions {
    /// How stale a read may be: after this long, a read first checks for
    /// writes from other handles and processes. Zero checks on every read.
    pub read_consistency_interval: Duration,
    /// Row count at which [`IndexStore::optimize`] first builds the
    /// approximate vector index. Smaller tables are searched exactly.
    pub vector_index_min_rows: usize,
}

impl Default for LanceStoreOptions {
    fn default() -> Self {
        Self {
            read_consistency_interval: DEFAULT_READ_CONSISTENCY_INTERVAL,
            vector_index_min_rows: DEFAULT_VECTOR_INDEX_MIN_ROWS,
        }
    }
}

/// An [`IndexStore`] backed by an embedded LanceDB database.
///
/// Several handles, in one or more processes, may read the same database;
/// only one may write at a time.
pub struct LanceStore {
    chunks: Table,
    meta: Table,
    dim: usize,
    options: LanceStoreOptions,
}

impl std::fmt::Debug for LanceStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceStore")
            .field("dim", &self.dim)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

impl LanceStore {
    /// Open the store under `tool_dir` with default options.
    ///
    /// See [`LanceStore::open_with`].
    ///
    /// # Errors
    ///
    /// As [`LanceStore::open_with`].
    pub async fn open(tool_dir: &Path, dim: usize) -> Result<Self> {
        Self::open_with(tool_dir, dim, LanceStoreOptions::default()).await
    }

    /// Open the store in `tool_dir/lancedb` for `dim`-long vectors.
    ///
    /// Missing tables are created empty, with their full-text indexes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Index`] when `dim` is zero, the path is not UTF-8,
    /// the database cannot be opened or created, or the existing chunk
    /// vectors have another length.
    pub async fn open_with(
        tool_dir: &Path,
        dim: usize,
        options: LanceStoreOptions,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Index("vector dimension must be positive".to_owned()));
        }
        let path = tool_dir.join(LANCEDB_DIR);
        let db = connect_db(&path, &options).await?;
        let names = table_names(&db).await?;
        let chunks = open_or_create(
            &db,
            &names,
            CHUNKS_TABLE,
            chunks_schema(dim)?,
            &[COL_FTS_TEXT],
        )
        .await?;
        let meta = open_or_create(
            &db,
            &names,
            META_TABLE,
            meta_schema(),
            &[COL_FTS_TITLE, COL_FTS_AUTHORS],
        )
        .await?;
        Self::from_tables(chunks, meta, dim, options).await
    }

    /// Open the existing store in `tool_dir/lancedb` for reading.
    ///
    /// Unlike [`LanceStore::open_with`], nothing is created: a missing
    /// database or table is an error. The handle can still write, so a
    /// reader must not call the writing methods of [`IndexStore`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoIndex`] when the database or one of its tables is
    /// missing, and [`Error::Index`] when `dim` is zero, the path is not
    /// UTF-8, the database cannot be opened, or the stored chunk vectors
    /// have another length.
    pub async fn open_read_only(
        tool_dir: &Path,
        dim: usize,
        options: LanceStoreOptions,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Index("vector dimension must be positive".to_owned()));
        }
        let path = tool_dir.join(LANCEDB_DIR);
        if !path.is_dir() {
            return Err(Error::NoIndex(path));
        }
        let db = connect_db(&path, &options).await?;
        let names = table_names(&db).await?;
        if ![CHUNKS_TABLE, META_TABLE]
            .iter()
            .all(|table| names.iter().any(|n| n == table))
        {
            return Err(Error::NoIndex(path));
        }
        let chunks = open_table(&db, CHUNKS_TABLE).await?;
        let meta = open_table(&db, META_TABLE).await?;
        Self::from_tables(chunks, meta, dim, options).await
    }

    async fn from_tables(
        chunks: Table,
        meta: Table,
        dim: usize,
        options: LanceStoreOptions,
    ) -> Result<Self> {
        let stored = vector_dim(&chunks).await?;
        if stored != dim {
            return Err(Error::Index(format!(
                "index holds {stored}-long vectors but the embedder produces {dim}; \
                 rebuild the index for this embedder"
            )));
        }
        Ok(Self {
            chunks,
            meta,
            dim,
            options,
        })
    }

    /// Return whether the chunks table has an approximate vector index.
    ///
    /// Without one, vector search scans every row exactly.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Index`] when the indexes cannot be listed.
    pub async fn has_vector_index(&self) -> Result<bool> {
        let indices = self
            .chunks
            .list_indices()
            .await
            .map_err(index_err("list indexes"))?;
        Ok(indices
            .iter()
            .any(|i| i.columns.iter().any(|c| c == COL_VECTOR)))
    }

    /// Run a keyword query over chunks matching `filter`, returning at most
    /// `limit` matches with `select` columns.
    ///
    /// The filter is applied before ranking, so a `limit` of at least the
    /// number of filtered matches returns every one of them.
    async fn keyword_batches(
        &self,
        terms: String,
        filter: &SearchFilter,
        select: Select,
        limit: usize,
    ) -> Result<Vec<RecordBatch>> {
        let fts = FullTextSearchQuery::new(terms)
            .with_column(COL_FTS_TEXT.to_owned())
            .map_err(index_err("build keyword query"))?
            .limit(Some(as_i64(limit)));
        let mut q = self
            .chunks
            .query()
            .full_text_search(fts)
            .select(select)
            .limit(limit);
        if let Some(predicate) = filter.to_predicate() {
            q = q.only_if(predicate);
        }
        collect(q.execute().await).await
    }
}

impl IndexStore for LanceStore {
    async fn freshness(&self, paper: &PaperNum) -> Result<Option<(String, String)>> {
        let batches = collect(
            self.meta
                .query()
                .only_if(eq(COL_PAPER_ID, paper.as_str()))
                .select(Select::columns(&[COL_CONTENT_HASH, COL_FORMAT_VERSION]))
                .limit(1)
                .execute()
                .await,
        )
        .await?;
        for batch in &batches {
            if batch.num_rows() > 0 {
                return Ok(Some((
                    string_col(batch, COL_CONTENT_HASH)?.value(0).to_owned(),
                    string_col(batch, COL_FORMAT_VERSION)?.value(0).to_owned(),
                )));
            }
        }
        Ok(None)
    }

    async fn upsert_paper(&self, records: PaperRecords) -> Result<()> {
        let paper = PaperNum::parse(&records.meta.row.paper_num)
            .map_err(|e| Error::Index(format!("metadata record has a bad paper id: {e}")))?;
        for chunk in &records.chunks {
            if chunk.paper_id != paper {
                return Err(Error::Index(format!(
                    "chunk of {} written with records of {paper}",
                    chunk.paper_id
                )));
            }
            if chunk.vector.len() != self.dim {
                return Err(Error::Index(format!(
                    "chunk of {paper} has a {}-long vector, expected {}",
                    chunk.vector.len(),
                    self.dim
                )));
            }
        }
        let owned = eq(COL_PAPER_ID, paper.as_str());
        // Chunks are written before the metadata record, which carries the
        // freshness data: an interrupted write leaves the paper stale, so
        // the next index run rewrites it.
        if records.chunks.is_empty() {
            self.chunks
                .delete(owned.as_str())
                .await
                .map_err(index_err("delete chunks"))?;
        } else {
            let batch = chunks_batch(self.dim, &paper, &records.chunks)?;
            let mut merge = self.chunks.merge_insert(&[COL_CHUNK_ID]);
            merge
                .when_matched_update_all(None)
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(Some(owned));
            merge
                .execute(reader(batch))
                .await
                .map_err(index_err("write chunks"))?;
        }
        let batch = meta_batch(&paper, &records.meta)?;
        let mut merge = self.meta.merge_insert(&[COL_PAPER_ID]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        merge
            .execute(reader(batch))
            .await
            .map_err(index_err("write metadata"))?;
        Ok(())
    }

    async fn delete_paper(&self, paper: &PaperNum) -> Result<()> {
        let predicate = eq(COL_PAPER_ID, paper.as_str());
        self.chunks
            .delete(predicate.as_str())
            .await
            .map_err(index_err("delete chunks"))?;
        self.meta
            .delete(predicate.as_str())
            .await
            .map_err(index_err("delete metadata"))?;
        Ok(())
    }

    async fn set_latest(&self, paper_number: &str, latest: &PaperNum) -> Result<()> {
        let scope = eq(COL_PAPER_NUMBER, paper_number);
        let flag = eq(COL_PAPER_ID, latest.as_str());
        for table in [&self.chunks, &self.meta] {
            table
                .update()
                .only_if(scope.clone())
                .column(COL_IS_LATEST, flag.clone())
                .execute()
                .await
                .map_err(index_err("update latest flags"))?;
        }
        Ok(())
    }

    async fn search_vector(
        &self,
        vector: &[f32],
        filter: &SearchFilter,
        limit: usize,
    ) -> Result<Vec<ChunkHit>> {
        if vector.len() != self.dim {
            return Err(Error::Index(format!(
                "query vector is {}-long, expected {}",
                vector.len(),
                self.dim
            )));
        }
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut q = self
            .chunks
            .query()
            .nearest_to(vector)
            .map_err(index_err("build vector query"))?
            .column(COL_VECTOR)
            .distance_type(DistanceType::Cosine)
            .select(chunk_select())
            .limit(limit);
        if let Some(predicate) = filter.to_predicate() {
            q = q.only_if(predicate);
        }
        let batches = collect(q.execute().await).await?;
        let mut hits = decode_chunk_hits(&batches, COL_DISTANCE, |d| 1.0 - d)?;
        sort_chunk_hits(&mut hits);
        Ok(hits)
    }

    async fn search_keyword(&self, query: &str, filter: &SearchFilter) -> Result<Vec<ChunkHit>> {
        let matches = self.count_keyword_matches(query, filter).await?;
        if matches == 0 {
            return Ok(Vec::new());
        }
        let batches = self
            .keyword_batches(keyword_text(query), filter, chunk_select(), matches)
            .await?;
        let mut hits = decode_chunk_hits(&batches, COL_SCORE, |s| s)?;
        sort_chunk_hits(&mut hits);
        Ok(hits)
    }

    async fn count_keyword_matches(&self, query: &str, filter: &SearchFilter) -> Result<usize> {
        let terms = keyword_text(query);
        if terms.is_empty() {
            return Ok(0);
        }
        let total = count(&self.chunks).await?;
        if total == 0 {
            return Ok(0);
        }
        // No query can match more rows than the table holds.
        let batches = self
            .keyword_batches(terms, filter, Select::columns(&[COL_CHUNK_ID]), total)
            .await?;
        Ok(batches.iter().map(RecordBatch::num_rows).sum())
    }

    async fn search_metadata(&self, query: &str, filter: &SearchFilter) -> Result<Vec<MetaHit>> {
        let total = count(&self.meta).await?;
        if total == 0 {
            return Ok(Vec::new());
        }
        let terms = keyword_text(query);
        let mut q = self.meta.query().select(meta_select()).limit(total);
        let has_terms = !terms.is_empty();
        if has_terms {
            let fts = FullTextSearchQuery::new(terms)
                .with_columns(&[COL_FTS_TITLE.to_owned(), COL_FTS_AUTHORS.to_owned()])
                .map_err(index_err("build metadata query"))?
                .limit(Some(as_i64(total)));
            q = q.full_text_search(fts);
        }
        if let Some(predicate) = filter.to_predicate() {
            q = q.only_if(predicate);
        }
        let batches = collect(q.execute().await).await?;
        let mut hits = Vec::new();
        for batch in &batches {
            let scores = if has_terms {
                Some(f32_col(batch, COL_SCORE)?)
            } else {
                None
            };
            for i in 0..batch.num_rows() {
                hits.push(MetaHit {
                    meta: decode_meta(batch, i)?,
                    score: scores.map_or(0.0, |s| s.value(i)),
                });
            }
        }
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.meta.row.paper_num.cmp(&b.meta.row.paper_num))
        });
        Ok(hits)
    }

    async fn optimize(&self) -> Result<()> {
        for table in [&self.chunks, &self.meta] {
            table
                .optimize(OptimizeAction::All)
                .await
                .map_err(index_err("optimize"))?;
        }
        if !self.has_vector_index().await?
            && count(&self.chunks).await? >= self.options.vector_index_min_rows
        {
            self.chunks
                .create_index(
                    &[COL_VECTOR],
                    Index::IvfHnswSq(
                        IvfHnswSqIndexBuilder::default().distance_type(DistanceType::Cosine),
                    ),
                )
                .execute()
                .await
                .map_err(index_err("create vector index"))?;
        }
        Ok(())
    }
}

/// Return `text` prepared for keyword indexing or querying.
///
/// Each whitespace-separated token loses the Markdown code and emphasis
/// marks, quotes, brackets and sentence punctuation at both ends; tokens
/// left empty are dropped, and the rest join with one space. Characters
/// inside a token, such as `::`, `_` and `<>`, are kept.
#[must_use]
pub fn keyword_text(text: &str) -> String {
    text.split_whitespace()
        .map(|t| t.trim_matches(KEYWORD_EDGE_PUNCTUATION))
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn index_err<E: std::fmt::Display>(context: &'static str) -> impl FnOnce(E) -> Error {
    move |e| Error::Index(format!("{context}: {e}"))
}

fn fts_params() -> FtsIndexBuilder {
    FtsIndexBuilder::default()
        .base_tokenizer(FTS_BASE_TOKENIZER.to_owned())
        .lower_case(FTS_LOWER_CASE)
        .stem(false)
        .remove_stop_words(false)
        .max_token_length(None)
}

async fn connect_db(path: &Path, options: &LanceStoreOptions) -> Result<Connection> {
    let uri = path
        .to_str()
        .ok_or_else(|| Error::Index(format!("index path {} is not UTF-8", path.display())))?;
    connect(uri)
        .read_consistency_interval(options.read_consistency_interval)
        .execute()
        .await
        .map_err(index_err("open database"))
}

async fn table_names(db: &Connection) -> Result<Vec<String>> {
    db.table_names()
        .execute()
        .await
        .map_err(index_err("list tables"))
}

async fn open_table(db: &Connection, name: &str) -> Result<Table> {
    db.open_table(name)
        .execute()
        .await
        .map_err(index_err("open table"))
}

async fn open_or_create(
    db: &Connection,
    names: &[String],
    name: &str,
    schema: SchemaRef,
    fts_columns: &[&str],
) -> Result<Table> {
    if names.iter().any(|n| n == name) {
        return open_table(db, name).await;
    }
    let table = db
        .create_empty_table(name, schema)
        .execute()
        .await
        .map_err(index_err("create table"))?;
    for column in fts_columns {
        table
            .create_index(&[*column], Index::FTS(fts_params()))
            .execute()
            .await
            .map_err(index_err("create keyword index"))?;
    }
    Ok(table)
}

async fn vector_dim(table: &Table) -> Result<usize> {
    let schema = table.schema().await.map_err(index_err("read schema"))?;
    match schema.field_with_name(COL_VECTOR).map(Field::data_type) {
        Ok(DataType::FixedSizeList(_, n)) => {
            usize::try_from(*n).map_err(|_| Error::Index(format!("bad vector dimension {n}")))
        }
        _ => Err(Error::Index("chunks table has no vector column".to_owned())),
    }
}

async fn count(table: &Table) -> Result<usize> {
    table
        .count_rows(None)
        .await
        .map_err(index_err("count rows"))
}

async fn collect(
    stream: lancedb::Result<impl futures::Stream<Item = lancedb::Result<RecordBatch>>>,
) -> Result<Vec<RecordBatch>> {
    stream
        .map_err(index_err("run query"))?
        .try_collect()
        .await
        .map_err(index_err("read results"))
}

fn as_i64(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

fn reader(batch: RecordBatch) -> Box<dyn arrow_array::RecordBatchReader + Send> {
    let schema = batch.schema();
    Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema))
}

fn utf8(name: &str) -> Field {
    Field::new(name, DataType::Utf8, false)
}

fn utf8_list(name: &str) -> Field {
    Field::new(
        name,
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        false,
    )
}

/// Columns both tables share, in schema order.
fn common_fields() -> Vec<Field> {
    vec![
        utf8(COL_PAPER_ID),
        utf8(COL_PAPER_NUMBER),
        Field::new(COL_REVISION, DataType::UInt32, true),
        Field::new(COL_IS_LATEST, DataType::Boolean, false),
        utf8(COL_CONTENT_HASH),
        utf8(COL_FORMAT_VERSION),
        utf8(COL_TITLE),
        utf8_list(COL_AUTHORS),
        utf8(COL_AUTHORS_TEXT),
        utf8(COL_TARGET_GROUP),
        utf8(COL_YEAR),
        utf8(COL_MAILING_DATE),
        utf8(COL_DOCUMENT_DATE),
        utf8(COL_INTENT),
    ]
}

/// Every chunk column except the vector, in schema order.
fn chunk_fields() -> Vec<Field> {
    let mut fields = vec![utf8(COL_CHUNK_ID)];
    fields.extend(common_fields());
    fields.extend([
        utf8_list(COL_HEADING_PATH),
        Field::new(COL_LINE_START, DataType::UInt64, false),
        Field::new(COL_LINE_END, DataType::UInt64, false),
        utf8(COL_TEXT),
        utf8(COL_PREFIXED_TEXT),
        utf8(COL_FTS_TEXT),
    ]);
    fields
}

fn chunks_schema(dim: usize) -> Result<SchemaRef> {
    let size = i32::try_from(dim)
        .map_err(|_| Error::Index(format!("vector dimension {dim} is too large")))?;
    let mut fields = chunk_fields();
    fields.push(Field::new(
        COL_VECTOR,
        DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), size),
        false,
    ));
    Ok(Arc::new(Schema::new(fields)))
}

fn meta_schema() -> SchemaRef {
    let mut fields = common_fields();
    fields.extend([
        Field::new(COL_HAS_MARKDOWN, DataType::Boolean, false),
        utf8(COL_URL),
        utf8(COL_DISPOSITION),
        utf8(COL_PREVIOUS_VERSION),
        utf8(COL_SOURCE_FILE),
        utf8(COL_MARKDOWN_PATH),
        utf8(COL_DISSECT_PATH),
        utf8(COL_AGORA_PATH),
        utf8(COL_ASSAY_PATH),
        utf8(COL_CITATIONS_EXTRACTED_AT),
        utf8(COL_ERROR),
        Field::new(COL_LINE_COUNT, DataType::UInt64, false),
        Field::new(COL_STATUS, DataType::Int64, false),
        utf8(COL_FTS_TITLE),
        utf8(COL_FTS_AUTHORS),
    ]);
    Arc::new(Schema::new(fields))
}

/// Every chunk column except the vector and keyword text, which results
/// never carry.
fn chunk_select() -> Select {
    Select::Columns(
        chunk_fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| n != COL_FTS_TEXT)
            .collect(),
    )
}

fn meta_select() -> Select {
    Select::Columns(
        meta_schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| n != COL_FTS_TITLE && n != COL_FTS_AUTHORS)
            .collect(),
    )
}

fn strings<'a>(values: impl Iterator<Item = &'a str>) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(values))
}

fn string_lists<'a>(values: impl Iterator<Item = &'a [String]>) -> ArrayRef {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for list in values {
        for item in list {
            builder.values().append_value(item);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

/// Columns of [`common_fields`] for rows of one paper revision.
struct Common<'a> {
    paper: &'a PaperNum,
    is_latest: bool,
    content_hash: &'a str,
    format_version: &'a str,
    fields: &'a PaperFields,
}

fn common_columns(rows: &[Common<'_>]) -> Vec<ArrayRef> {
    let revisions: UInt32Array = rows.iter().map(|r| r.paper.split_revision().1).collect();
    let latest: BooleanArray = rows.iter().map(|r| Some(r.is_latest)).collect();
    vec![
        strings(rows.iter().map(|r| r.paper.as_str())),
        strings(rows.iter().map(|r| r.paper.split_revision().0)),
        Arc::new(revisions),
        Arc::new(latest),
        strings(rows.iter().map(|r| r.content_hash)),
        strings(rows.iter().map(|r| r.format_version)),
        strings(rows.iter().map(|r| r.fields.title.as_str())),
        string_lists(rows.iter().map(|r| r.fields.authors.as_slice())),
        Arc::new(StringArray::from_iter_values(
            rows.iter()
                .map(|r| r.fields.authors.join(AUTHORS_SEPARATOR)),
        )),
        strings(rows.iter().map(|r| r.fields.target_group.as_str())),
        strings(rows.iter().map(|r| r.fields.year.as_str())),
        strings(rows.iter().map(|r| r.fields.mailing_date.as_str())),
        strings(rows.iter().map(|r| r.fields.document_date.as_str())),
        strings(rows.iter().map(|r| r.fields.intent.as_str())),
    ]
}

fn u64s(values: impl Iterator<Item = usize>) -> ArrayRef {
    Arc::new(UInt64Array::from_iter_values(
        values.map(|v| u64::try_from(v).unwrap_or(u64::MAX)),
    ))
}

fn chunks_batch(dim: usize, paper: &PaperNum, chunks: &[ChunkRecord]) -> Result<RecordBatch> {
    let schema = chunks_schema(dim)?;
    let size = i32::try_from(dim).unwrap_or(i32::MAX);
    let common: Vec<Common<'_>> = chunks
        .iter()
        .map(|c| Common {
            paper,
            is_latest: c.is_latest,
            content_hash: &c.content_hash,
            format_version: &c.format_version,
            fields: &c.fields,
        })
        .collect();
    let mut columns = vec![Arc::new(StringArray::from_iter_values(
        (0..chunks.len()).map(|i| format!("{paper}#{i:0CHUNK_ORDINAL_WIDTH$}")),
    )) as ArrayRef];
    columns.extend(common_columns(&common));
    columns.extend([
        string_lists(chunks.iter().map(|c| c.heading_path.as_slice())),
        u64s(chunks.iter().map(|c| c.start)),
        u64s(chunks.iter().map(|c| c.end)),
        strings(chunks.iter().map(|c| c.text.as_str())),
        strings(chunks.iter().map(|c| c.prefixed_text.as_str())),
        Arc::new(StringArray::from_iter_values(
            chunks.iter().map(|c| keyword_text(&c.prefixed_text)),
        )),
        Arc::new(
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                chunks
                    .iter()
                    .map(|c| Some(c.vector.iter().copied().map(Some).collect::<Vec<_>>())),
                size,
            ),
        ),
    ]);
    RecordBatch::try_new(schema, columns).map_err(index_err("build chunk rows"))
}

fn meta_batch(paper: &PaperNum, meta: &MetaRecord) -> Result<RecordBatch> {
    let row = &meta.row;
    let fields = PaperFields::from_row(row);
    let mut columns = common_columns(&[Common {
        paper,
        is_latest: meta.is_latest,
        content_hash: &meta.content_hash,
        format_version: &meta.format_version,
        fields: &fields,
    }]);
    let one = |v: &str| strings(std::iter::once(v));
    columns.extend([
        Arc::new(BooleanArray::from(vec![meta.has_markdown])) as ArrayRef,
        one(&row.url),
        one(&row.disposition),
        one(&row.previous_version),
        one(&row.source_file),
        one(&row.markdown_path),
        one(&row.dissect_path),
        one(&row.agora_path),
        one(&row.assay_path),
        one(&row.citations_extracted_at),
        one(&row.error),
        Arc::new(UInt64Array::from(vec![row.line_count])),
        Arc::new(Int64Array::from(vec![row.status])),
        one(&keyword_text(&row.title)),
        one(&keyword_text(&row.authors.join(AUTHORS_SEPARATOR))),
    ]);
    RecordBatch::try_new(meta_schema(), columns).map_err(index_err("build metadata row"))
}

fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> Result<&'a T> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<T>())
        .ok_or_else(|| Error::Index(format!("result lacks a well-typed {name} column")))
}

fn string_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    column(batch, name)
}

fn f32_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float32Array> {
    column(batch, name)
}

fn string(batch: &RecordBatch, name: &str, i: usize) -> Result<String> {
    Ok(string_col(batch, name)?.value(i).to_owned())
}

fn string_list(batch: &RecordBatch, name: &str, i: usize) -> Result<Vec<String>> {
    let list = column::<ListArray>(batch, name)?.value(i);
    let items = list
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| Error::Index(format!("{name} items are not strings")))?;
    Ok(items
        .iter()
        .map(|s| s.unwrap_or_default().to_owned())
        .collect())
}

fn usize_at(batch: &RecordBatch, name: &str, i: usize) -> Result<usize> {
    let v = column::<UInt64Array>(batch, name)?.value(i);
    usize::try_from(v).map_err(|_| Error::Index(format!("{name} {v} overflows usize")))
}

fn bool_at(batch: &RecordBatch, name: &str, i: usize) -> Result<bool> {
    Ok(column::<BooleanArray>(batch, name)?.value(i))
}

fn paper_at(batch: &RecordBatch, i: usize) -> Result<PaperNum> {
    let raw = string_col(batch, COL_PAPER_ID)?.value(i);
    PaperNum::parse(raw).map_err(|e| Error::Index(format!("stored paper id {raw:?}: {e}")))
}

fn fields_at(batch: &RecordBatch, i: usize) -> Result<PaperFields> {
    Ok(PaperFields {
        title: string(batch, COL_TITLE, i)?,
        authors: string_list(batch, COL_AUTHORS, i)?,
        target_group: string(batch, COL_TARGET_GROUP, i)?,
        year: string(batch, COL_YEAR, i)?,
        mailing_date: string(batch, COL_MAILING_DATE, i)?,
        document_date: string(batch, COL_DOCUMENT_DATE, i)?,
        intent: string(batch, COL_INTENT, i)?,
    })
}

fn decode_chunk_hits(
    batches: &[RecordBatch],
    score_column: &str,
    to_score: impl Fn(f32) -> f32,
) -> Result<Vec<ChunkHit>> {
    let mut hits = Vec::new();
    for batch in batches {
        let scores = f32_col(batch, score_column)?;
        let revisions = column::<UInt32Array>(batch, COL_REVISION)?;
        for i in 0..batch.num_rows() {
            hits.push(ChunkHit {
                chunk: ChunkRecord {
                    paper_id: paper_at(batch, i)?,
                    revision: (!revisions.is_null(i)).then(|| revisions.value(i)),
                    paper_number: string(batch, COL_PAPER_NUMBER, i)?,
                    is_latest: bool_at(batch, COL_IS_LATEST, i)?,
                    heading_path: string_list(batch, COL_HEADING_PATH, i)?,
                    start: usize_at(batch, COL_LINE_START, i)?,
                    end: usize_at(batch, COL_LINE_END, i)?,
                    text: string(batch, COL_TEXT, i)?,
                    prefixed_text: string(batch, COL_PREFIXED_TEXT, i)?,
                    vector: Vec::new(),
                    content_hash: string(batch, COL_CONTENT_HASH, i)?,
                    format_version: string(batch, COL_FORMAT_VERSION, i)?,
                    fields: fields_at(batch, i)?,
                },
                score: to_score(scores.value(i)),
            });
        }
    }
    Ok(hits)
}

fn sort_chunk_hits(hits: &mut [ChunkHit]) {
    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.chunk.paper_id.cmp(&b.chunk.paper_id))
            .then_with(|| a.chunk.start.cmp(&b.chunk.start))
    });
}

fn decode_meta(batch: &RecordBatch, i: usize) -> Result<MetaRecord> {
    let fields = fields_at(batch, i)?;
    let row = PaperRow {
        paper_num: string(batch, COL_PAPER_ID, i)?,
        year: fields.year,
        title: fields.title,
        target_group: fields.target_group,
        intent: fields.intent,
        url: string(batch, COL_URL, i)?,
        document_date: fields.document_date,
        mailing_date: fields.mailing_date,
        disposition: string(batch, COL_DISPOSITION, i)?,
        previous_version: string(batch, COL_PREVIOUS_VERSION, i)?,
        source_file: string(batch, COL_SOURCE_FILE, i)?,
        markdown_path: string(batch, COL_MARKDOWN_PATH, i)?,
        dissect_path: string(batch, COL_DISSECT_PATH, i)?,
        agora_path: string(batch, COL_AGORA_PATH, i)?,
        assay_path: string(batch, COL_ASSAY_PATH, i)?,
        citations_extracted_at: string(batch, COL_CITATIONS_EXTRACTED_AT, i)?,
        error: string(batch, COL_ERROR, i)?,
        authors: fields.authors,
        line_count: column::<UInt64Array>(batch, COL_LINE_COUNT)?.value(i),
        status: column::<Int64Array>(batch, COL_STATUS)?.value(i),
    };
    Ok(MetaRecord {
        row,
        is_latest: bool_at(batch, COL_IS_LATEST, i)?,
        has_markdown: bool_at(batch, COL_HAS_MARKDOWN, i)?,
        content_hash: string(batch, COL_CONTENT_HASH, i)?,
        format_version: string(batch, COL_FORMAT_VERSION, i)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyword_text_trims_markdown_around_identifiers() {
        assert_eq!(
            keyword_text("Add `std::move_only_function`, a (move-only) **wrapper**."),
            "Add std::move_only_function a move-only wrapper"
        );
        assert_eq!(
            keyword_text("std::vector<int> and __cpp_lib_ranges;"),
            "std::vector<int> and __cpp_lib_ranges"
        );
        assert_eq!(keyword_text("  `` ... \n"), "");
    }

    #[test]
    fn chunk_select_omits_vector_and_keyword_columns() {
        let Select::Columns(cols) = chunk_select() else {
            panic!("expected explicit columns");
        };
        assert!(!cols.iter().any(|c| c == COL_VECTOR || c == COL_FTS_TEXT));
        assert!(cols.iter().any(|c| c == COL_PREFIXED_TEXT));
    }
}
