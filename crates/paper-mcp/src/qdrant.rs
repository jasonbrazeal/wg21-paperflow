//! Qdrant: collection setup, payload types, writes, and queries (§7).

use std::collections::HashMap;
use std::time::Duration;

use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, DatetimeRange,
    DeletePointsBuilder, Distance, Document, FieldType, Filter, Fusion, GetPointsBuilder, Modifier,
    NamedVectors, PayloadExcludeSelector, PayloadIncludeSelector, PointId, PointStruct,
    PrefetchQueryBuilder, Query, QueryPointsBuilder, RetrievedPoint, RrfBuilder, ScoredPoint,
    ScrollPointsBuilder, SparseVectorParamsBuilder, SparseVectorsConfigBuilder, Timestamp,
    UpsertPointsBuilder, VectorParamsBuilder, VectorsConfigBuilder, with_payload_selector,
};
use qdrant_client::{Payload, Qdrant, QdrantError};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::bm25;
use crate::config::{Bm25Config, Config, SearchConfig};
use crate::dates::Date;
use crate::embed::DIMENSIONS;

/// Payload field names.
pub mod field {
    /// Lower-case paper id.
    pub const PAPER_ID: &str = "paper_id";
    /// Document date, `YYYY-MM-DD`, absent when unknown.
    pub const DOCUMENT_DATE: &str = "document_date";
    /// Mailing month as a date, for filtering.
    pub const MAILING_DATE_FILTER: &str = "mailing_date_filter";
    /// Mailing month, `YYYY-MM`.
    pub const MAILING_DATE: &str = "mailing_date";
    /// Paper series.
    pub const SERIES: &str = "series";
    /// Paper number.
    pub const NUMBER: &str = "number";
    /// Previous revision's id.
    pub const PREVIOUS_VERSION: &str = "previous_version";
    /// Ids the paper cites.
    pub const CITED_IDS: &str = "cited_ids";
    /// Folded author names.
    pub const AUTHORS_FOLDED: &str = "authors_folded";
    /// The full markdown.
    pub const MARKDOWN: &str = "markdown";
    /// Version of the chunker that produced the paper's chunks.
    pub const CHUNKER_VERSION: &str = "chunker_version";
}

const DENSE: &str = "dense";
const BM25: &str = "bm25";
const TITLE: &str = "title";
const AUTHORS: &str = "authors";
const OTHER: &str = "other";

/// Points fetched per scroll request.
const SCROLL_PAGE: u32 = 512;

/// Chunk points written per upsert request.
const UPSERT_BATCH: usize = 128;

/// Namespace for point ids, so the same paper always gets the same id.
const POINT_NAMESPACE: Uuid = Uuid::from_u128(0x6d1c_5f0e_2b7a_4c3e_9a51_0f8e_7d24_b6a9);

/// A Qdrant failure.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// A request failed.
    #[error("qdrant request failed: {0}")]
    Qdrant(#[from] Box<QdrantError>),
    /// A stored payload did not have the expected shape.
    #[error("unexpected payload in {collection}: {reason}")]
    Payload {
        /// The collection.
        collection: String,
        /// Why.
        reason: String,
    },
}

impl From<QdrantError> for StoreError {
    fn from(error: QdrantError) -> Self {
        Self::Qdrant(Box::new(error))
    }
}

type Result<T> = std::result::Result<T, StoreError>;

/// One citation: a cited paper id and how often it is mentioned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Citation {
    /// The lower-case cited id.
    pub paper_id: String,
    /// Mentions of it.
    pub count: u64,
}

/// A `paper_metadata` point's payload (§7.5).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MetadataPayload {
    /// Lower-case paper id.
    pub paper_id: String,
    /// Title.
    pub title: String,
    /// Author names.
    pub authors: Vec<String>,
    /// `YYYY-MM-DD`, absent when the paperstore value is not a valid date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_date: Option<String>,
    /// `YYYY-MM`.
    pub mailing_date: String,
    /// The first day of the mailing month, for filtering.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mailing_date_filter: Option<String>,
    /// Year.
    pub year: String,
    /// Target group.
    pub group: String,
    /// Intent.
    pub intent: String,
    /// Source URL.
    pub url: String,
    /// Disposition.
    pub disposition: String,
    /// Lower-case previous revision id, or empty.
    pub previous_version: String,
    /// Whether the paper has converted markdown.
    pub has_markdown: bool,
    /// Lines in the markdown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_lines: Option<u64>,
    /// Sections in the markdown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_sections: Option<u64>,
    /// `p`, `d`, `n`, or `sd`.
    pub series: String,
    /// Paper number.
    pub number: u64,
    /// Revision, absent for N and SD papers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<u64>,
    /// Folded author names, for substring matching.
    pub authors_folded: Vec<String>,
    /// Papers this one cites.
    pub cites: Vec<Citation>,
    /// The cited ids alone.
    pub cited_ids: Vec<String>,
    /// Whether citations were extracted.
    pub citations_extracted: bool,
    /// The full markdown, written only once the paper's chunks are complete.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub markdown: Option<String>,
    /// The chunker version that produced the chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunker_version: Option<String>,
}

/// A `paper_chunks` point's payload (§7.1).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkPayload {
    /// Lower-case paper id.
    pub paper_id: String,
    /// Title.
    pub title: String,
    /// Source URL.
    pub url: String,
    /// Document date, when valid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_date: Option<String>,
    /// Mailing month.
    pub mailing_date: String,
    /// Mailing month as a date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mailing_date_filter: Option<String>,
    /// Section index.
    pub section_index: u64,
    /// Heading path, joined.
    pub heading_path: String,
    /// First line.
    pub first_line: u64,
    /// Last line.
    pub last_line: u64,
    /// The chunk text, exactly lines `first_line` to `last_line`.
    pub text: String,
}

/// The point id of a paper's metadata record.
#[must_use]
pub fn metadata_point_id(paper_id: &str) -> Uuid {
    Uuid::new_v5(&POINT_NAMESPACE, format!("paper/{paper_id}").as_bytes())
}

/// The point id of one chunk.
#[must_use]
pub fn chunk_point_id(paper_id: &str, section: usize, chunk: usize) -> Uuid {
    Uuid::new_v5(
        &POINT_NAMESPACE,
        format!("chunk/{paper_id}/{section}/{chunk}").as_bytes(),
    )
}

fn timestamp(date: Date) -> Option<Timestamp> {
    Timestamp::date(i64::from(date.year), date.month, date.day).ok()
}

/// The date field a date range applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DateField {
    /// The document date.
    Document,
    /// The mailing month.
    Mailing,
}

/// Filters shared by `search` and `search_metadata` (§6.8).
#[derive(Debug, Clone, Default)]
pub struct Filters {
    /// Restrict to these lower-case paper ids.
    pub paper_ids: Option<Vec<String>>,
    /// Inclusive date range, applied when either end is set.
    pub start: Option<Date>,
    /// See `start`.
    pub end: Option<Date>,
    /// Which date the range applies to.
    pub date_field: Option<DateField>,
    /// Folded author substring, applied directly on `paper_metadata`.
    pub author: Option<String>,
}

impl Filters {
    /// Whether any filter is set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.paper_ids.is_none()
            && self.start.is_none()
            && self.end.is_none()
            && self.author.is_none()
    }

    /// The Qdrant filter, with the author condition only when
    /// `with_author` is set (it exists only on `paper_metadata`).
    #[must_use]
    pub fn to_filter(&self, with_author: bool) -> Option<Filter> {
        let mut must = Vec::new();
        if let Some(ids) = &self.paper_ids {
            must.push(Condition::matches(field::PAPER_ID, ids.clone()));
        }
        if self.start.is_some() || self.end.is_some() {
            let key = match self.date_field.unwrap_or(DateField::Document) {
                DateField::Document => field::DOCUMENT_DATE,
                DateField::Mailing => field::MAILING_DATE_FILTER,
            };
            must.push(Condition::datetime_range(
                key,
                DatetimeRange {
                    gte: self.start.and_then(timestamp),
                    lte: self.end.and_then(timestamp),
                    ..Default::default()
                },
            ));
        }
        if with_author && let Some(author) = &self.author {
            must.push(Condition::matches_text(
                field::AUTHORS_FOLDED,
                author.clone(),
            ));
        }
        (!must.is_empty()).then(|| Filter::must(must))
    }
}

/// How `search` ranks chunks (§7.3).
#[derive(Debug, Clone)]
pub enum ChunkSearch {
    /// BM25 only.
    Fulltext,
    /// Dense vector only, with the embedded query.
    Vector(Vec<f32>),
    /// Both, fused by RRF, with the embedded query.
    Hybrid(Vec<f32>),
}

/// A scored hit.
#[derive(Debug, Clone)]
pub struct Scored<T> {
    /// The score.
    pub score: f32,
    /// The payload.
    pub payload: T,
}

/// The Qdrant client and the two collection names.
#[derive(Clone)]
pub struct Store {
    client: Qdrant,
    chunks: String,
    metadata: String,
    bm25: Bm25Config,
    search: SearchConfig,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("chunks", &self.chunks)
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

fn decode<T: DeserializeOwned>(
    collection: &str,
    payload: HashMap<String, qdrant_client::qdrant::Value>,
) -> Result<T> {
    Payload::from(payload)
        .deserialize()
        .map_err(|e| StoreError::Payload {
            collection: collection.to_owned(),
            reason: e.to_string(),
        })
}

fn document(text: &str, avg_len: f64) -> Document {
    let options: HashMap<String, qdrant_client::qdrant::Value> = match bm25::options(avg_len) {
        Value::Object(map) => map.into_iter().map(|(k, v)| (k, v.into())).collect(),
        _ => HashMap::new(),
    };
    let mut doc = Document::new(text, bm25::BM25_MODEL);
    doc.options = options;
    doc
}

fn exclude_markdown() -> with_payload_selector::SelectorOptions {
    PayloadExcludeSelector::new(vec![field::MARKDOWN.to_owned()]).into()
}

impl Store {
    /// Connects to Qdrant.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] when the client cannot be built.
    pub fn connect(config: &Config) -> Result<Self> {
        // The client already accepts responses of any size; requests carry
        // at most one paper's markdown, far below Qdrant's 32 MB default.
        let client = Qdrant::from_url(&config.qdrant.url)
            .api_key(config.qdrant.api_key.clone())
            .timeout(Duration::from_secs(config.qdrant.timeout_secs))
            .skip_compatibility_check()
            .build()?;
        Ok(Self {
            client,
            chunks: config.chunks_collection(),
            metadata: config.metadata_collection(),
            bm25: config.bm25,
            search: config.search,
        })
    }

    /// Checks that Qdrant answers.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] when it does not.
    pub async fn health(&self) -> Result<()> {
        self.client.health_check().await?;
        Ok(())
    }

    /// Creates both collections and their payload indexes when missing.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] when a request fails.
    pub async fn ensure_collections(&self) -> Result<()> {
        if !self.client.collection_exists(&self.chunks).await? {
            let mut vectors = VectorsConfigBuilder::default();
            vectors.add_named_vector_params(
                DENSE,
                VectorParamsBuilder::new(DIMENSIONS, Distance::Cosine),
            );
            let mut sparse = SparseVectorsConfigBuilder::default();
            sparse.add_named_vector_params(
                BM25,
                SparseVectorParamsBuilder::default().modifier(Modifier::Idf),
            );
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.chunks)
                        .vectors_config(vectors)
                        .sparse_vectors_config(sparse)
                        .on_disk_payload(true),
                )
                .await?;
            self.index(&self.chunks, field::PAPER_ID, FieldType::Keyword)
                .await?;
            self.index(&self.chunks, field::DOCUMENT_DATE, FieldType::Datetime)
                .await?;
            self.index(
                &self.chunks,
                field::MAILING_DATE_FILTER,
                FieldType::Datetime,
            )
            .await?;
        }
        if !self.client.collection_exists(&self.metadata).await? {
            let mut sparse = SparseVectorsConfigBuilder::default();
            for name in [TITLE, AUTHORS, OTHER] {
                sparse.add_named_vector_params(
                    name,
                    SparseVectorParamsBuilder::default().modifier(Modifier::Idf),
                );
            }
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&self.metadata)
                        .sparse_vectors_config(sparse)
                        .on_disk_payload(true),
                )
                .await?;
            for (name, kind) in [
                (field::PAPER_ID, FieldType::Keyword),
                (field::SERIES, FieldType::Keyword),
                (field::PREVIOUS_VERSION, FieldType::Keyword),
                (field::CITED_IDS, FieldType::Keyword),
                (field::NUMBER, FieldType::Integer),
                (field::DOCUMENT_DATE, FieldType::Datetime),
                (field::MAILING_DATE_FILTER, FieldType::Datetime),
            ] {
                self.index(&self.metadata, name, kind).await?;
            }
        }
        Ok(())
    }

    async fn index(&self, collection: &str, name: &str, kind: FieldType) -> Result<()> {
        self.client
            .create_field_index(
                CreateFieldIndexCollectionBuilder::new(collection, name, kind).wait(true),
            )
            .await?;
        Ok(())
    }

    /// Deletes both collections. Used by tests.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] when a request fails.
    pub async fn drop_collections(&self) -> Result<()> {
        for name in [&self.chunks, &self.metadata] {
            if self.client.collection_exists(name).await? {
                self.client.delete_collection(name).await?;
            }
        }
        Ok(())
    }

    /// The stored chunker version of a paper: `None` when the paper has no
    /// record, `Some(None)` when its record holds no markdown.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] when the request fails.
    pub async fn chunker_version(&self, paper_id: &str) -> Result<Option<Option<String>>> {
        let points = self
            .client
            .get_points(
                GetPointsBuilder::new(
                    &self.metadata,
                    vec![PointId::from(metadata_point_id(paper_id))],
                )
                .with_payload(PayloadIncludeSelector::new(vec![
                    field::CHUNKER_VERSION.to_owned(),
                ])),
            )
            .await?
            .result;
        Ok(points.into_iter().next().map(|point| {
            point
                .payload
                .get(field::CHUNKER_VERSION)
                .and_then(|v| v.as_str().map(ToOwned::to_owned))
        }))
    }

    /// Replaces a paper's chunks: deletes every point with its id, then
    /// writes `points`.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] when a request fails.
    pub async fn replace_chunks(&self, paper_id: &str, points: Vec<PointStruct>) -> Result<()> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.chunks)
                    .points(Filter::must([Condition::matches(
                        field::PAPER_ID,
                        paper_id.to_owned(),
                    )]))
                    .wait(true),
            )
            .await?;
        let mut points = points;
        while !points.is_empty() {
            let rest = points.split_off(points.len().min(UPSERT_BATCH));
            self.client
                .upsert_points(UpsertPointsBuilder::new(&self.chunks, points).wait(true))
                .await?;
            points = rest;
        }
        Ok(())
    }

    /// Builds one chunk point.
    #[must_use]
    pub fn chunk_point(
        &self,
        id: Uuid,
        dense: Vec<f32>,
        payload: &ChunkPayload,
    ) -> Option<PointStruct> {
        let vectors = NamedVectors::default()
            .add_vector(DENSE, dense)
            .add_vector(BM25, document(&payload.text, self.bm25.chunk_avg_len));
        let payload = Payload::try_from(serde_json::to_value(payload).ok()?).ok()?;
        Some(PointStruct::new(id, vectors, payload))
    }

    /// Writes a paper's metadata record, replacing any existing one.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] when the payload cannot be encoded or the
    /// request fails.
    pub async fn upsert_metadata(&self, record: &MetadataPayload) -> Result<()> {
        let other = [
            record.paper_id.as_str(),
            record.group.as_str(),
            record.intent.as_str(),
            record.disposition.as_str(),
            record.previous_version.as_str(),
            record.document_date.as_deref().unwrap_or_default(),
            record.mailing_date.as_str(),
        ]
        .join(" ");
        let vectors = NamedVectors::default()
            .add_vector(TITLE, document(&record.title, self.bm25.title_avg_len))
            .add_vector(
                AUTHORS,
                document(&record.authors.join(", "), self.bm25.authors_avg_len),
            )
            .add_vector(OTHER, document(&other, self.bm25.other_avg_len));
        let encode_error = |reason: String| StoreError::Payload {
            collection: self.metadata.clone(),
            reason,
        };
        let value = serde_json::to_value(record).map_err(|e| encode_error(e.to_string()))?;
        let payload = Payload::try_from(value).map_err(|e| encode_error(e.to_string()))?;
        let point = PointStruct::new(metadata_point_id(&record.paper_id), vectors, payload);
        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.metadata, vec![point]).wait(true))
            .await?;
        Ok(())
    }

    /// Fetches metadata records by paper id, in no particular order.
    /// Missing ids are skipped.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] when the request fails.
    pub async fn get_metadata(
        &self,
        paper_ids: &[String],
        with_markdown: bool,
    ) -> Result<Vec<MetadataPayload>> {
        if paper_ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<PointId> = paper_ids
            .iter()
            .map(|id| metadata_point_id(id).into())
            .collect();
        let selector: with_payload_selector::SelectorOptions = if with_markdown {
            true.into()
        } else {
            exclude_markdown()
        };
        let points: Vec<RetrievedPoint> = self
            .client
            .get_points(GetPointsBuilder::new(&self.metadata, ids).with_payload(selector))
            .await?
            .result;
        points
            .into_iter()
            .map(|p| decode(&self.metadata, p.payload))
            .collect()
    }

    /// Fetches one metadata record.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] when the request fails.
    pub async fn get_one(
        &self,
        paper_id: &str,
        with_markdown: bool,
    ) -> Result<Option<MetadataPayload>> {
        Ok(self
            .get_metadata(&[paper_id.to_owned()], with_markdown)
            .await?
            .into_iter()
            .next())
    }

    /// Every metadata record matching `filter`, without markdown. With
    /// `fields`, only those payload fields are read.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] when a request fails.
    pub async fn scroll_metadata(
        &self,
        filter: Option<Filter>,
        fields: Option<&[&str]>,
    ) -> Result<Vec<MetadataPayload>> {
        let mut out = Vec::new();
        let mut offset: Option<PointId> = None;
        loop {
            let selector: with_payload_selector::SelectorOptions = match fields {
                Some(fields) => PayloadIncludeSelector::new(
                    fields.iter().map(|f| (*f).to_owned()).collect::<Vec<_>>(),
                )
                .into(),
                None => exclude_markdown(),
            };
            let mut request = ScrollPointsBuilder::new(&self.metadata)
                .limit(SCROLL_PAGE)
                .with_payload(selector);
            if let Some(filter) = &filter {
                request = request.filter(filter.clone());
            }
            if let Some(offset) = offset.take() {
                request = request.offset(offset);
            }
            let response = self.client.scroll(request).await?;
            for point in response.result {
                out.push(decode(&self.metadata, point.payload)?);
            }
            match response.next_page_offset {
                Some(next) => offset = Some(next),
                None => break,
            }
        }
        Ok(out)
    }

    /// Ranks chunks for `query`, returning hits `offset` to
    /// `offset + limit`.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] when the request fails.
    pub async fn search_chunks(
        &self,
        query: &str,
        search: ChunkSearch,
        filter: Option<Filter>,
        offset: u64,
        limit: u64,
    ) -> Result<Vec<Scored<ChunkPayload>>> {
        let bm25 = || Query::new_nearest(document(query, self.bm25.chunk_avg_len));
        let mut request = QueryPointsBuilder::new(&self.chunks)
            .limit(limit)
            .offset(offset)
            .with_payload(true);
        match search {
            ChunkSearch::Fulltext => request = request.query(bm25()).using(BM25),
            ChunkSearch::Vector(dense) => {
                request = request.query(Query::new_nearest(dense)).using(DENSE);
            }
            ChunkSearch::Hybrid(dense) => {
                for (query, using) in [(bm25(), BM25), (Query::new_nearest(dense), DENSE)] {
                    let mut prefetch = PrefetchQueryBuilder::default()
                        .query(query)
                        .using(using)
                        .limit(self.search.fused_candidates);
                    if let Some(filter) = &filter {
                        prefetch = prefetch.filter(filter.clone());
                    }
                    request = request.add_prefetch(prefetch);
                }
                request = request.query(Query::new_fusion(Fusion::Rrf));
            }
        }
        if let Some(filter) = filter {
            request = request.filter(filter);
        }
        let points: Vec<ScoredPoint> = self.client.query(request).await?.result;
        points
            .into_iter()
            .map(|p| {
                Ok(Scored {
                    score: p.score,
                    payload: decode(&self.chunks, p.payload)?,
                })
            })
            .collect()
    }

    /// Ranks metadata records for `query` over the title, authors, and
    /// other vectors, fused by weighted RRF.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] when the request fails.
    pub async fn search_metadata(
        &self,
        query: &str,
        filter: Option<Filter>,
        offset: u64,
        limit: u64,
    ) -> Result<Vec<Scored<MetadataPayload>>> {
        let mut request = QueryPointsBuilder::new(&self.metadata)
            .limit(limit)
            .offset(offset)
            .with_payload(exclude_markdown());
        for (name, avg_len) in [
            (TITLE, self.bm25.title_avg_len),
            (AUTHORS, self.bm25.authors_avg_len),
            (OTHER, self.bm25.other_avg_len),
        ] {
            let mut prefetch = PrefetchQueryBuilder::default()
                .query(Query::new_nearest(document(query, avg_len)))
                .using(name)
                .limit(self.search.fused_candidates);
            if let Some(filter) = &filter {
                prefetch = prefetch.filter(filter.clone());
            }
            request = request.add_prefetch(prefetch);
        }
        let weights = vec![
            self.search.title_weight,
            self.search.authors_weight,
            self.search.other_weight,
        ];
        request = request.query(Query::new_rrf(RrfBuilder::new().weights(weights)));
        if let Some(filter) = filter {
            request = request.filter(filter);
        }
        let points: Vec<ScoredPoint> = self.client.query(request).await?.result;
        points
            .into_iter()
            .map(|p| {
                Ok(Scored {
                    score: p.score,
                    payload: decode(&self.metadata, p.payload)?,
                })
            })
            .collect()
    }
}
