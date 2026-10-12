//! The BGE-M3 embedding model and its worker thread (§7.2, §10.4).
//!
//! One thread owns the model. Callers send texts over a bounded channel and
//! await the vectors on a oneshot channel, so embedding never runs on the
//! async executor and a full queue makes callers wait.

use std::path::Path;
use std::sync::Arc;
use std::thread;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot};

use crate::text::TokenCounter;

/// The model's name, part of the chunker version.
pub const MODEL_NAME: &str = "BAAI/bge-m3";

/// Dense vector dimensions.
pub const DIMENSIONS: u64 = 1024;

/// An embedding failure.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// The model could not be loaded.
    #[error("cannot load {MODEL_NAME}: {0}")]
    Load(String),
    /// The model failed on a batch.
    #[error("embedding failed: {0}")]
    Model(String),
    /// The worker thread is gone.
    #[error("the embedding worker stopped")]
    WorkerStopped,
}

struct Request {
    texts: Vec<String>,
    reply: oneshot::Sender<Result<Vec<Vec<f32>>, EmbedError>>,
}

/// A handle to the embedding worker. Cloning shares the worker.
#[derive(Clone)]
pub struct Embedder {
    requests: mpsc::Sender<Request>,
    tokenizer: Arc<Tokenizer>,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

/// Counts tokens with the model's own tokenizer, untruncated.
#[derive(Clone)]
pub struct ModelTokens(Arc<Tokenizer>);

impl std::fmt::Debug for ModelTokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelTokens").finish_non_exhaustive()
    }
}

impl TokenCounter for ModelTokens {
    fn count(&self, text: &str) -> usize {
        // A tokenizer failure is treated as one token per byte, which can
        // only overstate the size.
        self.0
            .encode(text, false)
            .map_or(text.len(), |encoding| encoding.len())
    }
}

impl Embedder {
    /// Loads the model, downloading it into `cache_dir` on first use, and
    /// starts the worker thread. Blocks until the model is loaded.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedError::Load`] when the model cannot be downloaded or
    /// loaded.
    pub fn start(
        cache_dir: &Path,
        max_length: usize,
        batch_tokens: usize,
        queue_depth: usize,
    ) -> Result<Self, EmbedError> {
        let options = TextInitOptions::new(EmbeddingModel::BGEM3)
            .with_cache_dir(cache_dir.to_path_buf())
            .with_max_length(max_length)
            .with_show_download_progress(false);
        let model = TextEmbedding::try_new(options).map_err(|e| EmbedError::Load(e.to_string()))?;
        let mut counting = model.tokenizer.clone();
        counting
            .with_truncation(None)
            .map_err(|e| EmbedError::Load(e.to_string()))?;
        counting.with_padding(None);
        let tokenizer = Arc::new(counting);
        let (requests, receiver) = mpsc::channel(queue_depth);
        let worker_tokens = ModelTokens(Arc::clone(&tokenizer));
        thread::Builder::new()
            .name("embed".to_owned())
            .spawn(move || run_worker(model, &worker_tokens, max_length, batch_tokens, receiver))
            .map_err(|e| EmbedError::Load(e.to_string()))?;
        Ok(Self {
            requests,
            tokenizer,
        })
    }

    /// The model's token counter.
    #[must_use]
    pub fn token_counter(&self) -> ModelTokens {
        ModelTokens(Arc::clone(&self.tokenizer))
    }

    /// Embeds `texts`, returning one vector per text in order.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedError`] when the model fails or the worker is gone.
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let (reply, response) = oneshot::channel();
        self.requests
            .send(Request { texts, reply })
            .await
            .map_err(|_| EmbedError::WorkerStopped)?;
        response.await.map_err(|_| EmbedError::WorkerStopped)?
    }

    /// Embeds one text.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedError`] when the model fails or the worker is gone.
    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed(vec![text.to_owned()])
            .await?
            .pop()
            .ok_or_else(|| EmbedError::Model("no vector returned".to_owned()))
    }
}

fn run_worker(
    mut model: TextEmbedding,
    tokens: &ModelTokens,
    max_length: usize,
    batch_tokens: usize,
    mut receiver: mpsc::Receiver<Request>,
) {
    while let Some(request) = receiver.blocking_recv() {
        let result = embed_batched(&mut model, tokens, max_length, batch_tokens, &request.texts);
        // The caller may have gone away; there is nobody left to tell.
        let _ = request.reply.send(result);
    }
}

/// Embeds texts in batches of similar length, so short texts are not padded
/// to the length of the longest one in the request.
fn embed_batched(
    model: &mut TextEmbedding,
    tokens: &ModelTokens,
    max_length: usize,
    batch_tokens: usize,
    texts: &[String],
) -> Result<Vec<Vec<f32>>, EmbedError> {
    let lengths: Vec<usize> = texts
        .iter()
        .map(|t| tokens.count(t).clamp(1, max_length))
        .collect();
    let mut order: Vec<usize> = (0..texts.len()).collect();
    order.sort_by_key(|&i| lengths[i]);

    let mut vectors: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && (end - start + 1) * lengths[order[end]] <= batch_tokens {
            end += 1;
        }
        let batch: Vec<&str> = order[start..end]
            .iter()
            .map(|&i| texts[i].as_str())
            .collect();
        let embedded = model
            .embed(&batch, Some(batch.len()))
            .map_err(|e| EmbedError::Model(e.to_string()))?;
        if embedded.len() != batch.len() {
            return Err(EmbedError::Model("vector count mismatch".to_owned()));
        }
        for (&index, vector) in order[start..end].iter().zip(embedded) {
            vectors[index] = Some(vector);
        }
        start = end;
    }
    vectors
        .into_iter()
        .map(|v| v.ok_or_else(|| EmbedError::Model("missing vector".to_owned())))
        .collect()
}
