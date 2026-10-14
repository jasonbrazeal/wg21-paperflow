//! Typed configuration, read once at startup from an optional TOML file and
//! the environment. Environment variables override the file.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::text::{ChunkSettings, MIN_MAX_TOKENS};

/// Environment variable naming the TOML configuration file.
pub const CONFIG_ENV: &str = "PAPER_MCP_CONFIG";

/// The embedding model's input limit, in tokens.
pub const MODEL_MAX_TOKENS: usize = 8192;

/// A configuration error, naming the setting at fault.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The configuration file could not be read.
    #[error("cannot read config file {path}: {source}")]
    Read {
        /// The file.
        path: PathBuf,
        /// The cause.
        source: std::io::Error,
    },
    /// The configuration file is not valid TOML for this schema.
    #[error("invalid config file {path}: {source}")]
    Parse {
        /// The file.
        path: PathBuf,
        /// The cause.
        source: Box<toml::de::Error>,
    },
    /// A setting is missing.
    #[error("setting {setting} is not set ({hint})")]
    Missing {
        /// The setting's name in the file.
        setting: &'static str,
        /// Where else it can be set.
        hint: &'static str,
    },
    /// A setting has an invalid value.
    #[error("setting {setting} is invalid: {reason}")]
    Invalid {
        /// The setting's name in the file.
        setting: &'static str,
        /// Why.
        reason: String,
    },
}

/// Qdrant connection settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QdrantConfig {
    /// gRPC URL. Env: `PAPER_MCP_QDRANT_URL`.
    pub url: String,
    /// API key. Env: `PAPER_MCP_QDRANT_API_KEY`.
    pub api_key: Option<String>,
    /// Prefix for both collection names. Env: `PAPER_MCP_COLLECTION_PREFIX`.
    pub collection_prefix: String,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_owned(),
            api_key: None,
            collection_prefix: String::new(),
            timeout_secs: 120,
        }
    }
}

/// Embedding model settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelConfig {
    /// Where the model weights are cached. Env: `PAPER_MCP_MODEL_CACHE_DIR`.
    pub cache_dir: Option<PathBuf>,
    /// The longest input the model accepts, in tokens. Chunks are kept to
    /// `chunks.max_tokens`; a longer search query is truncated.
    pub max_length: usize,
    /// The most padded tokens embedded in one batch.
    pub batch_tokens: usize,
    /// Depth of the queue of embedding requests waiting for the model.
    pub queue_depth: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            cache_dir: None,
            max_length: MODEL_MAX_TOKENS,
            batch_tokens: 8192,
            queue_depth: 64,
        }
    }
}

/// Chunking settings (§5.5).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChunkConfig {
    /// Target chunk size in tokens.
    pub target_tokens: usize,
    /// Largest block repeated as overlap, in tokens.
    pub overlap_max_tokens: usize,
    /// The most tokens in one chunk's embedded text. Larger blocks are
    /// split.
    pub max_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_tokens: 400,
            overlap_max_tokens: 200,
            max_tokens: 1024,
        }
    }
}

/// BM25 `avg_len` per vector (§7.2).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Bm25Config {
    /// Chunk text.
    pub chunk_avg_len: f64,
    /// Titles.
    pub title_avg_len: f64,
    /// Author names.
    pub authors_avg_len: f64,
    /// The `other` metadata field.
    pub other_avg_len: f64,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            chunk_avg_len: 300.0,
            title_avg_len: 6.0,
            authors_avg_len: 4.0,
            other_avg_len: 12.0,
        }
    }
}

/// Search and paging settings (§7.5, §7.6).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SearchConfig {
    /// Candidates each prefetch of a fused query retrieves.
    pub fused_candidates: u64,
    /// Page size when `limit` is not given.
    pub default_limit: u64,
    /// Largest accepted `limit`.
    pub max_limit: u64,
    /// RRF weight of the title vector in `search_metadata`.
    pub title_weight: f32,
    /// RRF weight of the authors vector in `search_metadata`.
    pub authors_weight: f32,
    /// RRF weight of the other vector in `search_metadata`.
    pub other_weight: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            fused_candidates: 1000,
            default_limit: 10,
            max_limit: 100,
            title_weight: 3.0,
            authors_weight: 1.0,
            other_weight: 1.0,
        }
    }
}

/// Server settings (§9).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServeConfig {
    /// Listen address. Env: `PAPER_MCP_BIND`.
    pub bind: String,
    /// HTTP path of the MCP endpoint.
    pub path: String,
    /// Bearer keys file, one token per line. Env: `PAPER_MCP_KEYS_FILE`.
    pub keys_file: Option<PathBuf>,
    /// Hostnames accepted in the `Host` header. Env:
    /// `PAPER_MCP_ALLOWED_HOSTS`, comma separated.
    pub allowed_hosts: Vec<String>,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8003".to_owned(),
            path: "/papers/mcp".to_owned(),
            keys_file: None,
            allowed_hosts: vec!["localhost".to_owned(), "127.0.0.1".to_owned()],
        }
    }
}

/// Ingest settings (§8).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct IngestConfig {
    /// The paperflow workspace holding `paperstore.db`. Env: `WG21_DATA_DIR`.
    pub data_dir: Option<PathBuf>,
    /// Chunks embedded per request to the model worker.
    pub embed_batch: usize,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            data_dir: None,
            embed_batch: 64,
        }
    }
}

/// The whole configuration.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Qdrant connection.
    pub qdrant: QdrantConfig,
    /// Embedding model.
    pub model: ModelConfig,
    /// Chunking.
    pub chunks: ChunkConfig,
    /// BM25.
    pub bm25: Bm25Config,
    /// Search and paging.
    pub search: SearchConfig,
    /// Server.
    pub serve: ServeConfig,
    /// Ingest.
    pub ingest: IngestConfig,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}

impl Config {
    /// Loads the configuration from `path` (or `$PAPER_MCP_CONFIG`) when
    /// given, then applies environment overrides and validates it.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] naming the file or setting at fault.
    pub fn load(path: Option<&Path>) -> Result<Self, ConfigError> {
        let path = path
            .map(Path::to_path_buf)
            .or_else(|| env(CONFIG_ENV).map(PathBuf::from));
        let mut config = match path {
            Some(path) => {
                let text = std::fs::read_to_string(&path).map_err(|source| ConfigError::Read {
                    path: path.clone(),
                    source,
                })?;
                Self::parse(&text).map_err(|source| ConfigError::Parse {
                    path,
                    source: Box::new(source),
                })?
            }
            None => Self::default(),
        };
        config.apply_env();
        config.validate()?;
        Ok(config)
    }

    /// Parses TOML text without environment overrides or validation.
    ///
    /// # Errors
    ///
    /// Returns the TOML error for invalid text or unknown settings.
    pub fn parse(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }

    fn apply_env(&mut self) {
        if let Some(v) = env("PAPER_MCP_QDRANT_URL") {
            self.qdrant.url = v;
        }
        if let Some(v) = env("PAPER_MCP_QDRANT_API_KEY") {
            self.qdrant.api_key = Some(v);
        }
        if let Some(v) = env("PAPER_MCP_COLLECTION_PREFIX") {
            self.qdrant.collection_prefix = v;
        }
        if let Some(v) = env("PAPER_MCP_MODEL_CACHE_DIR") {
            self.model.cache_dir = Some(PathBuf::from(v));
        }
        if let Some(v) = env("PAPER_MCP_BIND") {
            self.serve.bind = v;
        }
        if let Some(v) = env("PAPER_MCP_KEYS_FILE") {
            self.serve.keys_file = Some(PathBuf::from(v));
        }
        if let Some(v) = env("PAPER_MCP_ALLOWED_HOSTS") {
            self.serve.allowed_hosts = v
                .split(',')
                .map(str::trim)
                .filter(|h| !h.is_empty())
                .map(str::to_owned)
                .collect();
        }
        if let Some(v) = env("WG21_DATA_DIR") {
            self.ingest.data_dir = Some(PathBuf::from(v));
        }
    }

    /// Checks every setting that has a fixed valid range.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Invalid`] for the first bad setting.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let invalid = |setting, reason: &str| {
            Err(ConfigError::Invalid {
                setting,
                reason: reason.to_owned(),
            })
        };
        if self.qdrant.url.is_empty() {
            return invalid("qdrant.url", "must not be empty");
        }
        if self.model.max_length == 0 || self.model.max_length > MODEL_MAX_TOKENS {
            return invalid("model.max_length", "must be between 1 and 8192");
        }
        if self.model.batch_tokens < self.model.max_length {
            return invalid("model.batch_tokens", "must be at least model.max_length");
        }
        if self.model.queue_depth == 0 {
            return invalid("model.queue_depth", "must be positive");
        }
        if self.chunks.max_tokens < MIN_MAX_TOKENS || self.chunks.max_tokens > self.model.max_length
        {
            return invalid(
                "chunks.max_tokens",
                &format!("must be between {MIN_MAX_TOKENS} and model.max_length"),
            );
        }
        if self.chunks.target_tokens == 0 || self.chunks.target_tokens > self.chunks.max_tokens / 2
        {
            return invalid(
                "chunks.target_tokens",
                "must be positive and at most half of chunks.max_tokens",
            );
        }
        if self.chunks.overlap_max_tokens >= self.chunks.target_tokens {
            return invalid(
                "chunks.overlap_max_tokens",
                "must be below chunks.target_tokens",
            );
        }
        for (setting, value) in [
            ("bm25.chunk_avg_len", self.bm25.chunk_avg_len),
            ("bm25.title_avg_len", self.bm25.title_avg_len),
            ("bm25.authors_avg_len", self.bm25.authors_avg_len),
            ("bm25.other_avg_len", self.bm25.other_avg_len),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return invalid(setting, "must be a positive number");
            }
        }
        if self.search.fused_candidates == 0 {
            return invalid("search.fused_candidates", "must be positive");
        }
        if self.search.max_limit == 0 || self.search.max_limit > self.search.fused_candidates {
            return invalid(
                "search.max_limit",
                "must be between 1 and search.fused_candidates",
            );
        }
        if self.search.default_limit == 0 || self.search.default_limit > self.search.max_limit {
            return invalid(
                "search.default_limit",
                "must be between 1 and search.max_limit",
            );
        }
        for (setting, value) in [
            ("search.title_weight", self.search.title_weight),
            ("search.authors_weight", self.search.authors_weight),
            ("search.other_weight", self.search.other_weight),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return invalid(setting, "must be a positive number");
            }
        }
        if self.ingest.embed_batch == 0 {
            return invalid("ingest.embed_batch", "must be positive");
        }
        if !self.serve.path.starts_with('/') {
            return invalid("serve.path", "must start with /");
        }
        Ok(())
    }

    /// The chunker settings.
    #[must_use]
    pub fn chunk_settings(&self) -> ChunkSettings {
        ChunkSettings {
            target_tokens: self.chunks.target_tokens,
            overlap_max_tokens: self.chunks.overlap_max_tokens,
            max_tokens: self.chunks.max_tokens,
        }
    }

    /// The model cache directory.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Missing`] when it is not set.
    pub fn model_cache_dir(&self) -> Result<&Path, ConfigError> {
        self.model.cache_dir.as_deref().ok_or(ConfigError::Missing {
            setting: "model.cache_dir",
            hint: "or PAPER_MCP_MODEL_CACHE_DIR",
        })
    }

    /// The paperflow workspace directory.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Missing`] when it is not set.
    pub fn data_dir(&self) -> Result<&Path, ConfigError> {
        self.ingest.data_dir.as_deref().ok_or(ConfigError::Missing {
            setting: "ingest.data_dir",
            hint: "or WG21_DATA_DIR",
        })
    }

    /// The bearer keys file.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Missing`] when it is not set.
    pub fn keys_file(&self) -> Result<&Path, ConfigError> {
        self.serve.keys_file.as_deref().ok_or(ConfigError::Missing {
            setting: "serve.keys_file",
            hint: "or PAPER_MCP_KEYS_FILE",
        })
    }

    /// The listen address.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Invalid`] when it is not a socket address.
    pub fn bind_addr(&self) -> Result<SocketAddr, ConfigError> {
        self.serve
            .bind
            .parse()
            .map_err(|e: std::net::AddrParseError| ConfigError::Invalid {
                setting: "serve.bind",
                reason: e.to_string(),
            })
    }

    /// The chunk collection's name.
    #[must_use]
    pub fn chunks_collection(&self) -> String {
        format!("{}paper_chunks", self.qdrant.collection_prefix)
    }

    /// The metadata collection's name.
    #[must_use]
    pub fn metadata_collection(&self) -> String {
        format!("{}paper_metadata", self.qdrant.collection_prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        assert!(Config::default().validate().is_ok());
    }

    #[test]
    fn parses_a_partial_file() {
        let config =
            Config::parse("[chunks]\ntarget_tokens = 300\n[serve]\nbind = \"0.0.0.0:9000\"\n")
                .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(config.chunks.target_tokens, 300);
        assert_eq!(config.chunks.overlap_max_tokens, 200);
        assert_eq!(config.serve.bind, "0.0.0.0:9000");
    }

    #[test]
    fn rejects_unknown_settings_and_bad_values() {
        assert!(Config::parse("[chunks]\ntarget = 3\n").is_err());
        let mut config = Config::default();
        config.chunks.overlap_max_tokens = 500;
        let err = config
            .validate()
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("chunks.overlap_max_tokens"), "{err}");
    }

    #[test]
    fn chunk_size_limits_are_checked() {
        let error = |config: &Config| {
            config
                .validate()
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default()
        };
        let mut config = Config::default();
        config.chunks.max_tokens = 9000;
        assert!(error(&config).contains("chunks.max_tokens"));
        config.chunks.max_tokens = 64;
        assert!(error(&config).contains("chunks.max_tokens"));
        config.chunks.max_tokens = 600;
        assert!(error(&config).contains("chunks.target_tokens"));
    }
}
