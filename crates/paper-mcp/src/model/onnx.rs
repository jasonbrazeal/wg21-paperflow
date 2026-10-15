// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! ONNX Runtime backends built on fastembed's user-defined model APIs.
//!
//! A model directory holds an [`EMBEDDER_SUBDIR`] and a [`RERANKER_SUBDIR`]
//! subdirectory, each a Hugging Face style ONNX export containing
//! [`ONNX_FILE`] and the [`TOKENIZER_FILES`]. An embedder directory may also
//! hold [`EMBEDDER_POOLING_FILE`] (sentence-transformers pooling config;
//! mean pooling when absent), [`QUERY_INSTRUCTION_FILE`] (text prepended to
//! queries, not documents) and [`EXTERNAL_DATA_FILE`] (external weights).
//! Files are only ever read from that directory; nothing is downloaded.

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};
use std::fs::{self, File};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use fastembed::{
    InitOptionsUserDefined, Pooling, RerankInitOptionsUserDefined, TextEmbedding, TextRerank,
    TokenizerFiles, UserDefinedEmbeddingModel, UserDefinedRerankingModel,
};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use tokenizers::Tokenizer;

use crate::error::{Error, Result};
use crate::model::{Embedder, Reranker};

/// Subdirectory of the model directory holding the embedding model.
pub const EMBEDDER_SUBDIR: &str = "embedder";

/// Subdirectory of the model directory holding the reranking model.
pub const RERANKER_SUBDIR: &str = "reranker";

/// The ONNX graph file of a model.
pub const ONNX_FILE: &str = "model.onnx";

/// Tokenizer and model configuration files every model needs.
pub const TOKENIZER_FILES: [&str; 4] = [
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

/// Optional external weights referenced by [`ONNX_FILE`].
pub const EXTERNAL_DATA_FILE: &str = "model.onnx_data";

/// Optional sentence-transformers pooling config of an embedder.
pub const EMBEDDER_POOLING_FILE: &str = "1_Pooling/config.json";

/// Optional text an embedder prepends to queries.
pub const QUERY_INSTRUCTION_FILE: &str = "query_instruction.txt";

/// Text embedded once at load time to learn the vector length.
const DIM_PROBE_TEXT: &str = "dimension probe";

/// Hex digits of the file digest kept in a model id.
const MODEL_ID_HASH_HEX_DIGITS: usize = 16;

/// Largest `model_max_length` taken as a real limit. Tokenizers without a
/// limit store a huge sentinel (Hugging Face uses 1e30), and no supported
/// model accepts anywhere near a billion tokens.
const MAX_PLAUSIBLE_MODEL_LENGTH: f64 = 1e9;

/// Read buffer size used when hashing weight files without keeping them.
const HASH_BUFFER_BYTES: usize = 1 << 20;

/// Embedder running a local ONNX export through fastembed.
pub struct OnnxEmbedder {
    id: String,
    dim: usize,
    max_tokens: usize,
    query_instruction: String,
    counter: Tokenizer,
    model: Mutex<TextEmbedding>,
}

impl OnnxEmbedder {
    /// Load the embedder from the [`EMBEDDER_SUBDIR`] of `model_dir`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingModelFiles`] naming every absent required
    /// file, [`Error::ModelFileRead`] for an unreadable file,
    /// [`Error::ModelConfig`] for malformed configuration, and
    /// [`Error::ModelLoad`] when ONNX Runtime rejects the model.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let files = ModelFiles::<InMemoryWeights>::read(
            model_dir,
            EMBEDDER_SUBDIR,
            &[EMBEDDER_POOLING_FILE, QUERY_INSTRUCTION_FILE],
        )?;
        let pooling = files.pooling()?;
        let query_instruction = files.query_instruction()?;
        let ModelFiles {
            dir,
            id,
            max_tokens,
            weights: InMemoryWeights { onnx, external },
            tokenizer,
            ..
        } = files;
        let mut model = UserDefinedEmbeddingModel::new(onnx, tokenizer).with_pooling(pooling);
        if let Some(data) = external {
            model = model.with_external_initializer(EXTERNAL_DATA_FILE.to_owned(), data);
        }
        let options = InitOptionsUserDefined::new().with_max_length(max_tokens);
        let load_error = |message: String| Error::ModelLoad {
            dir: dir.clone(),
            message,
        };
        let mut embedding = TextEmbedding::try_new_from_user_defined(model, options)
            .map_err(|e| load_error(e.to_string()))?;
        let dim = embedding
            .embed([DIM_PROBE_TEXT], None)
            .map_err(|e| load_error(e.to_string()))?
            .first()
            .map_or(0, Vec::len);
        if dim == 0 {
            return Err(load_error("model produced an empty embedding".to_owned()));
        }
        let mut counter = embedding.tokenizer.clone();
        counter
            .with_truncation(None)
            .map_err(|e| load_error(e.to_string()))?;
        counter.with_padding(None);
        Ok(Self {
            id,
            dim,
            max_tokens,
            query_instruction,
            counter,
            model: Mutex::new(embedding),
        })
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let vectors = self
            .model
            .lock()
            .map_err(|_| Error::Inference("embedder lock poisoned".to_owned()))?
            .embed(texts, None)
            .map_err(|e| Error::Inference(e.to_string()))?;
        if vectors.len() != texts.len() || vectors.iter().any(|v| v.len() != self.dim) {
            return Err(Error::Inference(format!(
                "embedder returned {} vectors for {} texts, expected each of length {}",
                vectors.len(),
                texts.len(),
                self.dim
            )));
        }
        Ok(vectors)
    }
}

impl fmt::Debug for OnnxEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnnxEmbedder")
            .field("id", &self.id)
            .field("dim", &self.dim)
            .field("max_tokens", &self.max_tokens)
            .field("query_instruction", &self.query_instruction)
            .finish_non_exhaustive()
    }
}

impl Embedder for OnnxEmbedder {
    fn id(&self) -> &str {
        &self.id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Count tokens with the model's tokenizer, without truncation.
    ///
    /// Should the tokenizer fail, the byte length stands in: no token is
    /// shorter than one byte, so budgets stay conservative.
    fn count_tokens(&self, text: &str) -> usize {
        match self.counter.encode(text, true) {
            Ok(encoding) => encoding.len(),
            Err(e) => {
                tracing::warn!(error = %e, "tokenizer failed; counting bytes instead");
                text.len()
            }
        }
    }

    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let text = format!("{}{query}", self.query_instruction);
        self.embed(&[text])?
            .pop()
            .ok_or_else(|| Error::Inference("embedder returned no vector".to_owned()))
    }
}

/// Cross-encoder reranker running a local ONNX export through fastembed.
pub struct OnnxReranker {
    id: String,
    model: Mutex<TextRerank>,
}

impl OnnxReranker {
    /// Load the reranker from the [`RERANKER_SUBDIR`] of `model_dir`.
    ///
    /// # Errors
    ///
    /// As for [`OnnxEmbedder::load`].
    pub fn load(model_dir: &Path) -> Result<Self> {
        let ModelFiles {
            dir,
            id,
            max_tokens,
            tokenizer,
            ..
        } = ModelFiles::<OnDiskWeights>::read(model_dir, RERANKER_SUBDIR, &[])?;
        // Loading from the file lets ONNX Runtime resolve external weights
        // next to it, so the weights are only hashed, never held in memory.
        let model = UserDefinedRerankingModel::new(dir.join(ONNX_FILE), tokenizer);
        let options = RerankInitOptionsUserDefined::new().with_max_length(max_tokens);
        let rerank = TextRerank::try_new_from_user_defined(model, options).map_err(|e| {
            Error::ModelLoad {
                dir: dir.clone(),
                message: e.to_string(),
            }
        })?;
        Ok(Self {
            id,
            model: Mutex::new(rerank),
        })
    }
}

impl fmt::Debug for OnnxReranker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OnnxReranker")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl Reranker for OnnxReranker {
    fn id(&self) -> &str {
        &self.id
    }

    fn rerank(&self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let results = self
            .model
            .lock()
            .map_err(|_| Error::Inference("reranker lock poisoned".to_owned()))?
            .rerank(query.to_owned(), documents, false, None)
            .map_err(|e| Error::Inference(e.to_string()))?;
        let mut scores = vec![None; documents.len()];
        for result in results {
            if let Some(slot) = scores.get_mut(result.index) {
                *slot = Some(result.score);
            }
        }
        scores
            .into_iter()
            .collect::<Option<Vec<f32>>>()
            .ok_or_else(|| {
                Error::Inference(format!(
                    "reranker did not score all {} documents",
                    documents.len()
                ))
            })
    }
}

/// Incremental digest over the files of one model, each framed by its name
/// and byte length so that no two file sets collide by concatenation.
struct FileDigest(Sha256);

impl FileDigest {
    /// Hash `bytes` as the contents of the file `name`.
    fn add(&mut self, name: &str, bytes: &[u8]) {
        self.begin(name);
        self.0.update(bytes);
        self.end(bytes.len());
    }

    /// Hash the file `name` in `dir` without keeping its contents, giving the
    /// same digest as [`FileDigest::add`] on its bytes.
    fn add_streamed(&mut self, dir: &Path, name: &str) -> Result<()> {
        let path = dir.join(name);
        let read_error = |source| Error::ModelFileRead {
            path: path.clone(),
            source,
        };
        let mut file = File::open(&path).map_err(read_error)?;
        let mut buffer = vec![0; HASH_BUFFER_BYTES];
        let mut len = 0;
        self.begin(name);
        loop {
            let n = file.read(&mut buffer).map_err(read_error)?;
            if n == 0 {
                break;
            }
            self.0.update(&buffer[..n]);
            len += n;
        }
        self.end(len);
        Ok(())
    }

    fn begin(&mut self, name: &str) {
        self.0.update((name.len() as u64).to_le_bytes());
        self.0.update(name.as_bytes());
    }

    fn end(&mut self, len: usize) {
        self.0.update((len as u64).to_le_bytes());
    }
}

/// Read `name` from `dir`, adding it to `digest`.
fn read_hashed(dir: &Path, name: &str, digest: &mut FileDigest) -> Result<Vec<u8>> {
    let path = dir.join(name);
    let bytes = fs::read(&path).map_err(|source| Error::ModelFileRead { path, source })?;
    digest.add(name, &bytes);
    Ok(bytes)
}

/// How a model's weights, [`ONNX_FILE`] and an optional
/// [`EXTERNAL_DATA_FILE`], are taken from its directory.
trait Weights: Sized {
    /// Take the weights from `dir`, adding every weight file to `digest`.
    fn take(dir: &Path, digest: &mut FileDigest) -> Result<Self>;
}

/// Weights read into memory, for backends that load from bytes.
struct InMemoryWeights {
    onnx: Vec<u8>,
    external: Option<Vec<u8>>,
}

impl Weights for InMemoryWeights {
    fn take(dir: &Path, digest: &mut FileDigest) -> Result<Self> {
        let onnx = read_hashed(dir, ONNX_FILE, digest)?;
        let external = dir
            .join(EXTERNAL_DATA_FILE)
            .exists()
            .then(|| read_hashed(dir, EXTERNAL_DATA_FILE, digest))
            .transpose()?;
        Ok(Self { onnx, external })
    }
}

/// Weights left on disk and only hashed, for backends that load from the
/// file path.
struct OnDiskWeights;

impl Weights for OnDiskWeights {
    fn take(dir: &Path, digest: &mut FileDigest) -> Result<Self> {
        digest.add_streamed(dir, ONNX_FILE)?;
        if dir.join(EXTERNAL_DATA_FILE).exists() {
            digest.add_streamed(dir, EXTERNAL_DATA_FILE)?;
        }
        Ok(Self)
    }
}

/// Check that `model_dir` holds every required file of both models,
/// without reading or loading them.
///
/// # Errors
///
/// Returns [`Error::MissingModelFiles`] for the first model subdirectory
/// that lacks a required file, naming every absent file in it.
pub fn check_model_files(model_dir: &Path) -> Result<()> {
    for sub in [EMBEDDER_SUBDIR, RERANKER_SUBDIR] {
        require_files(model_dir, sub)?;
    }
    Ok(())
}

/// Return `model_dir/sub` when it holds every required model file.
fn require_files(model_dir: &Path, sub: &str) -> Result<PathBuf> {
    let dir = model_dir.join(sub);
    let expected: Vec<String> = std::iter::once(ONNX_FILE)
        .chain(TOKENIZER_FILES)
        .map(str::to_owned)
        .collect();
    let missing: Vec<String> = expected
        .iter()
        .filter(|name| !dir.join(name).exists())
        .cloned()
        .collect();
    if missing.is_empty() {
        Ok(dir)
    } else {
        Err(Error::MissingModelFiles {
            dir,
            missing,
            expected,
        })
    }
}

/// The files of one model subdirectory, read and validated.
struct ModelFiles<W> {
    dir: PathBuf,
    id: String,
    max_tokens: usize,
    weights: W,
    tokenizer: TokenizerFiles,
    optional: BTreeMap<&'static str, Vec<u8>>,
}

impl<W: Weights> ModelFiles<W> {
    /// Read the weights, the required files and whichever of `optional`
    /// exist under `model_dir/sub`.
    ///
    /// The model id is `sub` plus a digest of every file read, so changing
    /// any of them changes the id. The id does not depend on `W`.
    fn read(model_dir: &Path, sub: &str, optional: &[&'static str]) -> Result<Self> {
        let dir = require_files(model_dir, sub)?;
        let mut digest = FileDigest(Sha256::new());
        let weights = W::take(&dir, &mut digest)?;
        let [
            tokenizer_file,
            config_file,
            special_tokens_map_file,
            tokenizer_config_file,
        ] = [0, 1, 2, 3].map(|i| read_hashed(&dir, TOKENIZER_FILES[i], &mut digest));
        let tokenizer = TokenizerFiles {
            tokenizer_file: tokenizer_file?,
            config_file: config_file?,
            special_tokens_map_file: special_tokens_map_file?,
            tokenizer_config_file: tokenizer_config_file?,
        };
        let mut optional_files = BTreeMap::new();
        for name in optional {
            if dir.join(name).exists() {
                optional_files.insert(*name, read_hashed(&dir, name, &mut digest)?);
            }
        }
        let id = format!(
            "{sub}-{}",
            &hex(&digest.0.finalize())[..MODEL_ID_HASH_HEX_DIGITS]
        );

        let parse = |name: &str, bytes: &[u8]| -> Result<Value> {
            serde_json::from_slice(bytes).map_err(|e| Error::ModelConfig {
                path: dir.join(name),
                reason: e.to_string(),
            })
        };
        parse(TOKENIZER_FILES[0], &tokenizer.tokenizer_file)?;
        let config = parse(TOKENIZER_FILES[1], &tokenizer.config_file)?;
        parse(TOKENIZER_FILES[2], &tokenizer.special_tokens_map_file)?;
        let tokenizer_config = parse(TOKENIZER_FILES[3], &tokenizer.tokenizer_config_file)?;
        let max_tokens =
            max_tokens(&config, &tokenizer_config).ok_or_else(|| Error::ModelConfig {
                path: dir.join(TOKENIZER_FILES[3]),
                reason: "neither model_max_length here nor max_position_embeddings in \
                         config.json gives a positive input length"
                    .to_owned(),
            })?;

        Ok(Self {
            dir,
            id,
            max_tokens,
            weights,
            tokenizer,
            optional: optional_files,
        })
    }

    /// Return the pooling named by [`EMBEDDER_POOLING_FILE`], or mean pooling.
    fn pooling(&self) -> Result<Pooling> {
        let Some(bytes) = self.optional.get(EMBEDDER_POOLING_FILE) else {
            return Ok(Pooling::Mean);
        };
        let path = self.dir.join(EMBEDDER_POOLING_FILE);
        let config: Value = serde_json::from_slice(bytes).map_err(|e| Error::ModelConfig {
            path: path.clone(),
            reason: e.to_string(),
        })?;
        let enabled = |key: &str| config.get(key).and_then(Value::as_bool) == Some(true);
        if enabled("pooling_mode_cls_token") {
            Ok(Pooling::Cls)
        } else if enabled("pooling_mode_mean_tokens") {
            Ok(Pooling::Mean)
        } else {
            Err(Error::ModelConfig {
                path,
                reason: "only CLS and mean pooling are supported".to_owned(),
            })
        }
    }

    /// Return the text of [`QUERY_INSTRUCTION_FILE`] without its line
    /// ending, or `""`.
    fn query_instruction(&self) -> Result<String> {
        let Some(bytes) = self.optional.get(QUERY_INSTRUCTION_FILE) else {
            return Ok(String::new());
        };
        let text = std::str::from_utf8(bytes).map_err(|e| Error::ModelConfig {
            path: self.dir.join(QUERY_INSTRUCTION_FILE),
            reason: e.to_string(),
        })?;
        Ok(text.trim_end_matches(['\n', '\r']).to_owned())
    }
}

/// Return the smaller of the tokenizer's `model_max_length` and the model's
/// `max_position_embeddings`, ignoring absent or non-positive values and a
/// `model_max_length` above [`MAX_PLAUSIBLE_MODEL_LENGTH`] ("no limit").
fn max_tokens(config: &Value, tokenizer_config: &Value) -> Option<usize> {
    let from_tokenizer = tokenizer_config
        .get("model_max_length")
        .and_then(Value::as_f64)
        .filter(|n| (1.0..=MAX_PLAUSIBLE_MODEL_LENGTH).contains(n))
        .map(truncating_usize);
    let from_config = config
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .filter(|n| *n >= 1)
        .map(|n| usize::try_from(n).unwrap_or(usize::MAX));
    match (from_tokenizer, from_config) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// Convert a float in `1.0..=MAX_PLAUSIBLE_MODEL_LENGTH` to `usize`,
/// dropping the fraction.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the caller bounds the value well inside `usize`; `as` drops the fraction"
)]
fn truncating_usize(n: f64) -> usize {
    n as usize
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut out, byte| {
        // Writing to a String cannot fail.
        let _ = write!(out, "{byte:02x}");
        out
    })
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::model::MODEL_DIR_ENV;

    /// Proxy settings pointing at a closed local port, so any attempted
    /// download fails fast instead of reaching the network.
    const DEAD_PROXY: &str = "http://127.0.0.1:9";

    fn required() -> Vec<String> {
        std::iter::once(ONNX_FILE)
            .chain(TOKENIZER_FILES)
            .map(str::to_owned)
            .collect()
    }

    fn write_files(dir: &Path, names: &[&str], contents: &str) {
        for name in names {
            let path = dir.join(name);
            if let Some(parent) = path.parent() {
                assert!(fs::create_dir_all(parent).is_ok());
            }
            assert!(fs::write(path, contents).is_ok());
        }
    }

    fn tempdir() -> TempDir {
        TempDir::new().ok().unwrap_or_else(|| panic!("tempdir"))
    }

    /// Run `f` with every download path closed off, then check that no
    /// download cache appeared.
    fn offline<T>(f: impl FnOnce() -> T) -> T {
        let cache = tempdir();
        let fastembed_cache = cache.path().join("fastembed");
        let hf_home = cache.path().join("hf");
        let out = temp_env::with_vars(
            [
                ("HTTP_PROXY", Some(DEAD_PROXY)),
                ("HTTPS_PROXY", Some(DEAD_PROXY)),
                ("ALL_PROXY", Some(DEAD_PROXY)),
                ("HF_HUB_OFFLINE", Some("1")),
                ("FASTEMBED_CACHE_DIR", fastembed_cache.to_str()),
                ("HF_HOME", hf_home.to_str()),
            ],
            f,
        );
        assert!(!fastembed_cache.exists(), "fastembed cache was created");
        assert!(!hf_home.exists(), "hugging face cache was created");
        out
    }

    fn missing_of(result: Result<impl fmt::Debug>) -> (PathBuf, Vec<String>, Vec<String>, String) {
        match result {
            Err(e @ Error::MissingModelFiles { .. }) => {
                let message = e.to_string();
                let Error::MissingModelFiles {
                    dir,
                    missing,
                    expected,
                } = e
                else {
                    unreachable!()
                };
                (dir, missing, expected, message)
            }
            other => panic!("expected MissingModelFiles, got {other:?}"),
        }
    }

    #[test]
    fn empty_dir_names_every_required_file() {
        let root = tempdir();
        for (sub, result) in [
            (
                EMBEDDER_SUBDIR,
                offline(|| OnnxEmbedder::load(root.path()).map(|_| ())),
            ),
            (
                RERANKER_SUBDIR,
                offline(|| OnnxReranker::load(root.path()).map(|_| ())),
            ),
        ] {
            let (dir, missing, expected, message) = missing_of(result);
            assert_eq!(dir, root.path().join(sub));
            assert_eq!(missing, required());
            assert_eq!(expected, required());
            for name in required() {
                assert!(message.contains(&name), "{message}");
            }
            assert!(message.contains(&dir.display().to_string()), "{message}");
        }
    }

    #[test]
    fn partial_dir_names_only_absent_files() {
        let root = tempdir();
        let present = [ONNX_FILE, "tokenizer.json"];
        write_files(&root.path().join(EMBEDDER_SUBDIR), &present, "{}");
        write_files(&root.path().join(RERANKER_SUBDIR), &present, "{}");
        for result in [
            offline(|| OnnxEmbedder::load(root.path()).map(|_| ())),
            offline(|| OnnxReranker::load(root.path()).map(|_| ())),
        ] {
            let (_, missing, expected, _) = missing_of(result);
            assert_eq!(
                missing,
                [
                    "config.json",
                    "special_tokens_map.json",
                    "tokenizer_config.json"
                ]
            );
            assert_eq!(expected, required());
        }
    }

    #[test]
    fn unreadable_file_names_its_path() {
        let root = tempdir();
        let sub = root.path().join(EMBEDDER_SUBDIR);
        write_files(&sub, &TOKENIZER_FILES, "{}");
        // A directory where a file is expected exists but cannot be read.
        assert!(fs::create_dir_all(sub.join(ONNX_FILE)).is_ok());
        match offline(|| OnnxEmbedder::load(root.path())) {
            Err(Error::ModelFileRead { path, .. }) => assert_eq!(path, sub.join(ONNX_FILE)),
            other => panic!("expected ModelFileRead, got {other:?}"),
        }
    }

    #[test]
    fn malformed_config_is_reported_before_runtime_load() {
        let root = tempdir();
        for sub in [EMBEDDER_SUBDIR, RERANKER_SUBDIR] {
            let dir = root.path().join(sub);
            write_files(&dir, &[ONNX_FILE], "not an onnx graph");
            write_files(&dir, &TOKENIZER_FILES, "not json");
        }
        for result in [
            offline(|| OnnxEmbedder::load(root.path()).map(|_| ())),
            offline(|| OnnxReranker::load(root.path()).map(|_| ())),
        ] {
            match result {
                Err(Error::ModelConfig { path, .. }) => {
                    assert!(path.starts_with(root.path()), "{}", path.display());
                }
                other => panic!("expected ModelConfig, got {other:?}"),
            }
        }
    }

    #[test]
    fn garbage_graph_fails_to_load_without_network() {
        let root = tempdir();
        let dir = root.path().join(RERANKER_SUBDIR);
        write_files(&dir, &[ONNX_FILE], "not an onnx graph");
        write_files(&dir, &["tokenizer.json"], MINIMAL_TOKENIZER);
        write_files(&dir, &["config.json"], r#"{"max_position_embeddings": 16}"#);
        write_files(&dir, &["special_tokens_map.json"], "{}");
        write_files(
            &dir,
            &["tokenizer_config.json"],
            r#"{"model_max_length": 16, "pad_token": "[PAD]"}"#,
        );
        match offline(|| OnnxReranker::load(root.path())) {
            Err(Error::ModelLoad { dir: d, .. }) => assert_eq!(d, dir),
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    fn max_of(config: &str, tokenizer_config: &str) -> Option<usize> {
        let parse = |s: &str| serde_json::from_str::<Value>(s).unwrap_or(Value::Null);
        max_tokens(&parse(config), &parse(tokenizer_config))
    }

    #[test]
    fn max_tokens_takes_the_smaller_valid_limit() {
        let pos = |n: &str| format!(r#"{{"max_position_embeddings": {n}}}"#);
        let len = |n: &str| format!(r#"{{"model_max_length": {n}}}"#);
        assert_eq!(max_of(&pos("512"), &len("8192")), Some(512));
        assert_eq!(max_of(&pos("8192"), &len("512")), Some(512));
        assert_eq!(max_of(&pos("512"), "{}"), Some(512));
        assert_eq!(max_of("{}", &len("512")), Some(512));
        assert_eq!(max_of("{}", &len("512.9")), Some(512));
        assert_eq!(max_of(&pos("0"), &len("256")), Some(256));
        assert_eq!(max_of(&pos("256"), &len("0")), Some(256));
        assert_eq!(max_of(&pos("256"), &len("-1")), Some(256));
        assert_eq!(max_of(&pos("256"), &len("\"512\"")), Some(256));
        assert_eq!(max_of("{}", "{}"), None);
        assert_eq!(max_of(&pos("0"), &len("0")), None);
    }

    #[test]
    fn max_tokens_treats_no_limit_sentinel_as_absent() {
        let sentinel = r#"{"model_max_length": 1e30}"#;
        assert_eq!(
            max_of(r#"{"max_position_embeddings": 512}"#, sentinel),
            Some(512)
        );
        assert_eq!(max_of("{}", sentinel), None);
        assert_eq!(
            max_of("{}", r#"{"model_max_length": 1000000000}"#),
            Some(1_000_000_000)
        );
        assert_eq!(max_of("{}", r#"{"model_max_length": 1000000001}"#), None);
    }

    #[test]
    fn sentinel_without_config_limit_fails_to_load() {
        let root = tempdir();
        let dir = write_fixture(root.path(), EMBEDDER_SUBDIR);
        write_files(&dir, &["config.json"], "{}");
        write_files(
            &dir,
            &["tokenizer_config.json"],
            r#"{"model_max_length": 1e30}"#,
        );
        match ModelFiles::<OnDiskWeights>::read(root.path(), EMBEDDER_SUBDIR, &[]) {
            Err(Error::ModelConfig { path, .. }) => {
                assert_eq!(path, dir.join("tokenizer_config.json"));
            }
            Err(e) => panic!("expected ModelConfig, got {e:?}"),
            Ok(_) => panic!("expected ModelConfig, got Ok"),
        }
    }

    /// Write a model directory `root/sub` whose files parse but whose graph
    /// is not a real model, and return its path.
    fn write_fixture(root: &Path, sub: &str) -> PathBuf {
        let dir = root.join(sub);
        write_files(&dir, &[ONNX_FILE], "graph");
        write_files(&dir, &["tokenizer.json", "special_tokens_map.json"], "{}");
        write_files(
            &dir,
            &["config.json"],
            r#"{"max_position_embeddings": 512}"#,
        );
        write_files(
            &dir,
            &["tokenizer_config.json"],
            r#"{"model_max_length": 256}"#,
        );
        dir
    }

    fn embedder_files(root: &Path) -> ModelFiles<InMemoryWeights> {
        match ModelFiles::read(
            root,
            EMBEDDER_SUBDIR,
            &[EMBEDDER_POOLING_FILE, QUERY_INSTRUCTION_FILE],
        ) {
            Ok(files) => files,
            Err(e) => panic!("{e}"),
        }
    }

    #[test]
    fn fixture_reads_limits_and_weights() {
        let root = tempdir();
        let dir = write_fixture(root.path(), EMBEDDER_SUBDIR);
        let files = embedder_files(root.path());
        assert_eq!(files.max_tokens, 256);
        assert_eq!(files.weights.onnx, b"graph");
        assert!(files.weights.external.is_none());
        write_files(&dir, &[EXTERNAL_DATA_FILE], "weights");
        let files = embedder_files(root.path());
        assert_eq!(files.weights.external.as_deref(), Some(&b"weights"[..]));
    }

    #[test]
    fn pooling_follows_the_pooling_config() {
        let root = tempdir();
        let dir = write_fixture(root.path(), EMBEDDER_SUBDIR);
        let pooling = || embedder_files(root.path()).pooling();
        assert!(matches!(pooling(), Ok(Pooling::Mean)));
        for (config, expected) in [
            (r#"{"pooling_mode_cls_token": true}"#, Pooling::Cls),
            (r#"{"pooling_mode_mean_tokens": true}"#, Pooling::Mean),
            (
                r#"{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": true}"#,
                Pooling::Cls,
            ),
            (
                r#"{"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true}"#,
                Pooling::Mean,
            ),
        ] {
            write_files(&dir, &[EMBEDDER_POOLING_FILE], config);
            match pooling() {
                Ok(p) => assert_eq!(format!("{p:?}"), format!("{expected:?}"), "{config}"),
                Err(e) => panic!("{config}: {e}"),
            }
        }
        for config in [
            r#"{"pooling_mode_max_tokens": true}"#,
            r#"{"pooling_mode_cls_token": "true"}"#,
            "{}",
            "not json",
        ] {
            write_files(&dir, &[EMBEDDER_POOLING_FILE], config);
            match pooling() {
                Err(Error::ModelConfig { path, .. }) => {
                    assert_eq!(path, dir.join(EMBEDDER_POOLING_FILE), "{config}");
                }
                other => panic!("{config}: expected ModelConfig, got {other:?}"),
            }
        }
    }

    #[test]
    fn query_instruction_drops_only_the_line_ending() {
        let root = tempdir();
        let dir = write_fixture(root.path(), EMBEDDER_SUBDIR);
        let instruction = || embedder_files(root.path()).query_instruction().ok();
        assert_eq!(instruction().as_deref(), Some(""));
        for (text, expected) in [
            ("query: ", "query: "),
            ("query: \n", "query: "),
            ("query: \r\n", "query: "),
            ("  Represent this: \r\n\n", "  Represent this: "),
        ] {
            write_files(&dir, &[QUERY_INSTRUCTION_FILE], text);
            assert_eq!(instruction().as_deref(), Some(expected), "{text:?}");
        }
        assert!(fs::write(dir.join(QUERY_INSTRUCTION_FILE), [0xff, 0xfe]).is_ok());
        match embedder_files(root.path()).query_instruction() {
            Err(Error::ModelConfig { path, .. }) => {
                assert_eq!(path, dir.join(QUERY_INSTRUCTION_FILE));
            }
            other => panic!("expected ModelConfig, got {other:?}"),
        }
    }

    #[test]
    fn id_changes_with_every_file_read() {
        let root = tempdir();
        let dir = write_fixture(root.path(), EMBEDDER_SUBDIR);
        write_files(
            &dir,
            &[EMBEDDER_POOLING_FILE],
            r#"{"pooling_mode_cls_token": true}"#,
        );
        write_files(&dir, &[QUERY_INSTRUCTION_FILE], "query: ");
        write_files(&dir, &[EXTERNAL_DATA_FILE], "weights");
        let id = || embedder_files(root.path()).id;
        let base = id();
        assert!(base.starts_with("embedder-"), "{base}");
        assert_eq!(base.len(), "embedder-".len() + MODEL_ID_HASH_HEX_DIGITS);
        assert_eq!(id(), base, "id is not stable");
        let mut seen = vec![base.clone()];
        let files = required().into_iter().chain(
            [
                EMBEDDER_POOLING_FILE,
                QUERY_INSTRUCTION_FILE,
                EXTERNAL_DATA_FILE,
            ]
            .map(str::to_owned),
        );
        for name in files {
            let path = dir.join(&name);
            let original = fs::read(&path).ok().unwrap_or_default();
            let mut changed = original.clone();
            changed.push(b' ');
            assert!(fs::write(&path, &changed).is_ok());
            let new = id();
            assert!(
                !seen.contains(&new),
                "changing {name} did not change the id"
            );
            seen.push(new);
            assert!(fs::write(&path, &original).is_ok());
            assert_eq!(id(), base, "restoring {name} did not restore the id");
        }
        assert!(fs::remove_file(dir.join(EXTERNAL_DATA_FILE)).is_ok());
        assert_ne!(
            id(),
            base,
            "removing an optional file did not change the id"
        );
    }

    #[test]
    fn id_ignores_unrequested_files() {
        let root = tempdir();
        let dir = write_fixture(root.path(), RERANKER_SUBDIR);
        let id = || match ModelFiles::<OnDiskWeights>::read(root.path(), RERANKER_SUBDIR, &[]) {
            Ok(files) => files.id,
            Err(e) => panic!("{e}"),
        };
        let base = id();
        write_files(&dir, &[QUERY_INSTRUCTION_FILE], "query: ");
        assert_eq!(id(), base);
    }

    #[test]
    fn streamed_and_in_memory_weights_give_the_same_id() {
        let root = tempdir();
        let dir = write_fixture(root.path(), EMBEDDER_SUBDIR);
        let in_memory =
            || match ModelFiles::<InMemoryWeights>::read(root.path(), EMBEDDER_SUBDIR, &[]) {
                Ok(files) => files.id,
                Err(e) => panic!("{e}"),
            };
        let on_disk = || match ModelFiles::<OnDiskWeights>::read(root.path(), EMBEDDER_SUBDIR, &[])
        {
            Ok(files) => files.id,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(in_memory(), on_disk());
        // Larger than one read buffer, so streaming spans several reads.
        let big: Vec<u8> = (0..HASH_BUFFER_BYTES * 2 + 7)
            .map(|i| u8::try_from(i % 251).unwrap_or_default())
            .collect();
        assert!(fs::write(dir.join(ONNX_FILE), &big).is_ok());
        write_files(&dir, &[EXTERNAL_DATA_FILE], "weights");
        assert_eq!(in_memory(), on_disk());
    }

    #[test]
    fn unreadable_weight_names_its_path_when_streamed() {
        let root = tempdir();
        let sub = root.path().join(RERANKER_SUBDIR);
        write_files(&sub, &TOKENIZER_FILES, "{}");
        assert!(fs::create_dir_all(sub.join(ONNX_FILE)).is_ok());
        match offline(|| OnnxReranker::load(root.path())) {
            Err(Error::ModelFileRead { path, .. }) => assert_eq!(path, sub.join(ONNX_FILE)),
            other => panic!("expected ModelFileRead, got {other:?}"),
        }
    }

    /// Smallest valid `tokenizer.json`: whitespace-split word-level vocabulary.
    const MINIMAL_TOKENIZER: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"[PAD]": 0, "[UNK]": 1}, "unk_token": "[UNK]"}
    }"#;

    fn real_model_dir() -> PathBuf {
        std::env::var_os(MODEL_DIR_ENV).map_or_else(
            || panic!("set {MODEL_DIR_ENV} to run this test"),
            PathBuf::from,
        )
    }

    #[test]
    #[ignore = "needs real ONNX models in PAPER_MCP_MODEL_DIR"]
    fn real_embedder_loads_and_embeds() {
        let e = match offline(|| OnnxEmbedder::load(&real_model_dir())) {
            Ok(e) => e,
            Err(err) => panic!("{err}"),
        };
        assert!(e.dim() > 0);
        assert!(e.max_tokens() > 0);
        assert!(e.count_tokens("std::move_only_function") > 0);
        let texts = vec![
            "The sender and scheduler model for asynchronous execution.".to_owned(),
            "Unicode normalization of C++ identifiers.".to_owned(),
        ];
        let a = e.embed_documents(&texts).ok().unwrap_or_default();
        let b = e.embed_documents(&texts).ok().unwrap_or_default();
        assert_eq!(a, b, "embedding is not deterministic");
        assert_eq!(a.len(), 2);
        assert!(a.iter().all(|v| v.len() == e.dim()));
        let q = e
            .embed_query("asynchronous senders")
            .ok()
            .unwrap_or_default();
        assert_eq!(q.len(), e.dim());
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, r)| p * r).sum::<f32>();
        assert!(dot(&q, &a[0]) > dot(&q, &a[1]));
        assert_eq!(e.embed_documents(&[]).ok(), Some(vec![]));
    }

    #[test]
    #[ignore = "needs real ONNX models in PAPER_MCP_MODEL_DIR"]
    fn real_reranker_scores_in_input_order() {
        let r = match offline(|| OnnxReranker::load(&real_model_dir())) {
            Ok(r) => r,
            Err(err) => panic!("{err}"),
        };
        let docs = vec![
            "Unicode normalization of C++ identifiers.".to_owned(),
            "The sender and scheduler model for asynchronous execution.".to_owned(),
        ];
        let scores = r
            .rerank("asynchronous senders", &docs)
            .ok()
            .unwrap_or_default();
        assert_eq!(scores.len(), 2);
        assert!(scores[1] > scores[0]);
        assert_eq!(r.rerank("q", &[]).ok(), Some(vec![]));
    }
}
