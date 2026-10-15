# WG21 Paper RAG MCP Server (Rust)

<product-contract>

## Product Requirements

A Rust service indexes the WG21 paper corpus already held in paperstore and lets LLM clients search and read the papers and their metadata. Clients connect to a shared MCP server over Streamable HTTP with a bearer token. Indexing runs only when an operator asks for it, and the operator chooses the scope: all papers, one year, or a list of paper ids.

- Problem and users: LLM clients (and the people driving them) need to find and cite WG21 papers by their content and metadata across every revision. Today the corpus exists only as Markdown files and SQLite rows (`data/paperstore/`, `data/paperstore.db`), with no semantic or lexical search and no way for an LLM to reach it.
- Goals:
  - Hybrid retrieval (dense vectors plus BM25) over every indexed revision of every paper, filterable by paper metadata.
  - Read access: whole papers or line ranges, paper outlines, metadata, revision lists and diffs between revisions.
  - A shared, network-reachable MCP server protected by a bearer token.
  - Indexing on demand, scoped the same way `paperflow convert` is scoped (`packages/tomd/README.md`): one or more paper ids, a year, or `all`.
- Non-goals:
  - MCP resources and prompts. The server exposes tools only.
  - The stdio transport.
  - TLS handling. A reverse proxy in front of the server terminates TLS, and the server speaks plain HTTP.
  - Automatic or scheduled indexing, and ingest triggered by the server.
  - Scraping, downloading or converting papers. Those stay in the existing Python pipeline.
  - Any network access by the index command or the server: no web search, no metadata or paper downloads, no runtime model downloads. Paperstore is the only source of papers and metadata.
  - LLM-generated chunk context.
- Success criteria:
  - Every tool in the Functional Specification works against an index built from real paperstore data.
  - On a golden query set that includes code-identifier, wording and revision-specific queries, recall@20 for the chosen configuration beats dense-only and BM25-only baselines on the same set.
  - The server rejects every request that lacks a valid bearer token.
- Constraints:
  - Rust, as new members of the existing Cargo workspace (`Cargo.toml`: edition 2024, `rust-version = "1.89"`, resolver 3; `rust-toolchain.toml`: channel 1.98.0). The workspace `rust-version` rises to 1.91 because lancedb 0.40.0 requires it (crates.io API).
  - All paper storage access goes through the paperstore abstraction (CLAUDE.md invariants). The workspace directory comes from `WG21_DATA_DIR` (`crates/paperstore/src/workspace.rs`).
  - Models are self-hosted and open-weight. There are no cloud embedding or LLM APIs (CLAUDE.md "Model sovereignty").
  - Source files carry BSL-1.0 headers and contain no em dashes (CLAUDE.md "Style").
- Open questions: None

## Functional Specification

Operators run an index command over a selection of papers. It chunks each paper's Markdown along its structure, adds a deterministic context prefix to each chunk, embeds it, and writes the chunks and paper-level metadata records to the index. A long-running server exposes eight MCP tools over authenticated Streamable HTTP. Search tools read the index. Read tools read paperstore directly, so they always return current content.

- Actors and workflows:
  - Operator indexes papers: `index <selector>... [--force] [--workers N]`. A selector is one or more paper ids, a four-digit year, or `all`, mirroring `paperflow convert` (`packages/tomd/README.md`). A paper is skipped when its index records match its current Markdown content hash and the current index-format version, unless `--force` is given.
  - Operator runs the server: `serve`, configured with a bind address, the bearer token, the index location and the model location.
  - An LLM client calls tools over Streamable HTTP with `Authorization: Bearer <token>`.
- Inputs and outputs (MCP tools):
  - `search_papers`: content search over chunks. It takes a query, a required `mode` (`vector`, `keyword` or `hybrid`), filters (paper id, revision, subgroup/target_group, author, year, mailing_date, document_date range, intent), an `all_revisions` flag (default false), an optional `top_k` override and an optional `cursor`.
    - Result set per mode: `keyword` returns every chunk matching the query. `vector` returns the `top_k` nearest chunks. `hybrid` returns every keyword match plus the `top_k` nearest chunks, fused with RRF. `top_k` defaults to the server's configured value (default 1000) and can be overridden per call.
    - In every mode, the reranker reorders the first `rerank_depth` results (server-configured, default 150), and results past that depth follow in their original order. Nothing is dropped, so paging reaches the end of the result set. By default only the latest revision of each paper is searched. `all_revisions: true` searches every indexed revision without collapsing results, and a fully versioned paper id (e.g. `P2300R4`) searches only that revision. Each result carries the paper id with revision, title, section heading path, `start`/`end` in the paper's Markdown, the chunk text without its prefix, and a score.
  - `search_metadata`: search over paper-level records, using keywords on title and authors plus the same field filters, with the same `cursor` paging. It returns every matching metadata row, one per paper revision.
  - Paging follows MCP's cursor convention. The first call runs the search, stores the complete ranked list on the server, and returns the first page plus an opaque `next_cursor`. Later calls pass `cursor` and read from the stored list, so pages stay stable even if an index run lands mid-way. The server sets the page size (configurable, default 20). Stored lists expire after a configurable idle TTL (default 15 minutes), and the cache holds a configurable maximum number of lists (default 64, least recently used evicted first). An expired or unknown cursor returns a tool error telling the client to rerun the search.
  - `get_paper`: the full Markdown of one paper revision, with no size cap.
  - `get_paper_lines`: a line slice of one paper revision. At least one of `start` and `end` is required. Lines are numbered from 1 and sliced like Python: `start` is inclusive, `end` is exclusive, a missing `start` begins at the first line, a missing `end` runs to the last line, and negative values count back from the end. Examples: `start=1` returns the whole paper; `start=1, end=3` returns lines 1 and 2; `start=-5` returns the last five lines; `end=-1` returns every line except the last. `start=0` is invalid. Out-of-range bounds clamp as Python slices do. The response includes `total_lines`.
  - Every line range the server reports (search results, outline sections) uses the same convention, so it can be passed straight to `get_paper_lines`.
  - `get_paper_metadata`: the full paperstore metadata row for one paper revision.
  - `list_revisions`: every known revision of a paper number, with date, mailing and `previous_version`, newest first.
  - `get_paper_outline`: the paper's heading tree, giving level, heading text and `start`/`end` (end exclusive) for each section.
  - `diff_revisions`: a unified diff of the Markdown of two revisions of the same paper, with no size cap.
  - Paper ids are parsed with `PaperNum::parse`, which trims and uppercases (`crates/paperstore/src/num.rs`). An id without a revision (e.g. `P2300`) resolves to the latest known revision.
- States and validation:
  - A paper with a metadata row but no Markdown (`MissingMarkdown`, `crates/paperstore-sqlite/src/backend.rs`) gets only its paper-level metadata record. The index run reports it as having no content.
  - Line numbers in search results refer to the Markdown as of indexing. Each chunk record stores the Markdown content hash it was built from.
  - Latest revision means the highest revision of a paper number that paperstore lists, the same rule used to resolve unversioned ids. Each indexed revision carries an `is_latest` flag. When an index run writes a revision, it recomputes `is_latest` for every indexed revision of that paper number.
- Errors and recovery:
  - Indexing processes papers independently. A failure on one paper is reported per paper and does not abort the run, and the command exits non-zero if any paper failed. Rerunning the same selector resumes the work, because papers that are already current are skipped.
  - Tool errors (unknown paper, invalid range, revision mismatch in `diff_revisions`) return MCP tool errors with a message the client can act on, not transport errors.
  - The server and the index command fail at startup with a clear message when the token, index, model files or `WG21_DATA_DIR` are missing.
- Security and privacy behavior:
  - Every HTTP request needs a valid bearer token, compared in constant time, and gets HTTP 401 otherwise. The token is supplied through the environment or a file and is never logged.
  - Tools are read-only. No tool writes to paperstore or the index.
- Acceptance criteria:
  - Running `index` with ids, with a year and with `all` selects the same papers that paperstore lists for those selectors. Rerunning without `--force` skips every unchanged paper.
  - Every `search_papers` result's `start`/`end`, passed to `get_paper_lines`, returns that chunk's text.
  - `get_paper_lines` matches Python slicing of the paper's lines (with 1-based positive indices) for every combination of positive, negative and missing bounds.
  - With P2300R0 through R10 indexed, a default `search_papers` call returns only P2300R10 chunks for P2300, and `all_revisions: true` can return chunks from any revision.
  - Code blocks and tables are never split across chunks unless a single block exceeds the chunk budget.
  - A request without a valid token gets 401, and all eight tools work with a valid token.

</product-contract>
<implementation-contract>

## Technical Design

One new Rust crate provides an `index` command and a `serve` command. Both share a library containing the selection, chunking, contextual-prefix, embedding, index-store and search modules. The paperstore crates gain listing methods so that selection and revision lookups go through the abstraction. Retrieval follows contextual retrieval, except that the per-chunk context is a deterministic prefix instead of LLM-written text. The prefixed text feeds both the embedding and the BM25 index, the two result lists are fused with rank fusion, and a local cross-encoder reranks the fused candidates.

- Architecture:
  - Ingest: paperstore selection, then `paper_md` and `meta`, then structure-aware chunking, then the deterministic prefix, then local embedding, then upserts to the index store (chunk records plus one paper-level metadata record per revision).
  - Serve: an axum app with bearer-auth middleware in front of rmcp's `StreamableHttpService` (rmcp 3.x, feature `transport-streamable-http-server`). It exposes tools only, defined with `#[tool_router(server_handler)]`.
  - Search: depending on `mode`, a dense query embedding, a BM25 query over prefixed chunk text, or both fused with RRF. The same metadata filter (including `is_latest = true` unless `all_revisions` is set or a versioned id is given) applies to every side. Local cross-encoder reranking covers the first `rerank_depth` results, and the full ranked list goes into the cursor cache for paging.
  - Server configuration: `top_k`, `rerank_depth`, page size, cursor TTL and cursor cache size, each with the default given in the Functional Specification.
- Modules and interfaces:
  - Selection: resolves ids, years and `all` to `PaperNum`s through the new paperstore listing methods.
  - Chunker: parses tomd Markdown and drops YAML front matter and `<style>` blocks from chunk bodies, while still counting their lines. It splits on `##`/`###` headings and treats fenced code blocks, GFM pipe tables and `<!-- tomd:mixed-table -->` HTML tables as atomic blocks. Sections over budget are split at block boundaries. The chunk token budget fits within the embedding model's context window and is tuned by the golden-set evaluation. Each chunk records its heading path and its line range in the original file (1-based start, exclusive end). The tomd Markdown structure is described in `data/paperstore/p2169r4.md` and `data/paperstore/p2322r4.md`.
  - Prefix: a deterministic string built from paper id with revision, title, authors, target_group, document date and section heading path, prepended to the chunk text for both embedding and BM25. The text returned to clients is unprefixed.
  - Embedder and reranker: local open-weight models loaded from a configured model directory, with no downloads at runtime. Embedding is behind a trait so that a candle backend can replace an ONNX-based one.
  - Index store: behind a trait covering upsert by paper, delete by paper, hybrid search with filter, metadata search and freshness lookup (content hash plus index-format version). The backend is embedded LanceDB 0.40 (`lancedb` crate): one chunks table and one paper-metadata table, each with a vector or FTS index as needed. Hybrid queries use `only_if`, `full_text_search`, `rerank` and `execute_hybrid`. The FTS index uses a tokenizer configured for C++ text (stemming and stop-word removal off, chosen by the golden-set evaluation), because the default splits, stems and drops tokens in identifiers like `std::move_only_function`.
- File and public API changes:
  - `crates/paperstore/src/backend.rs`: `StorageBackend` gains listing methods for all paper ids, ids for a year, and revisions of a paper number. `crates/paperstore-sqlite` implements them. The Python equivalents are `list_all_paper_ids` and `list_papers_for_year` in `packages/paperstore/src/paperstore/sqlite_backend.py`.
  - A new workspace member under `crates/` that builds one binary with `index` and `serve` subcommands.
  - CI: `.github/workflows/tests.yml` builds and tests the new crate and the paperstore crates.
- Data, persistence, failure, security, and privacy constraints:
  - The index lives under `$WG21_DATA_DIR` (gitignored, `.gitignore`), in its own subdirectory.
  - The index records the index-format version (chunker, prefix and embedding model identity). A change in that version marks every paper stale.
  - One index writer runs at a time, enforced by a lock file in the index directory, because LanceDB multi-process writer safety is not established. The server serves reads while an index run is in progress and picks up new data without a restart by setting `read_consistency_interval`. Each index run ends with `optimize` so that newly added rows join the FTS and vector indexes instead of being flat-scanned.
  - The bearer token is never logged or echoed in errors.

</implementation-contract>
<verification-contract>

## Testing Plan

Unit tests cover parsing, chunking, prefixing and auth. Integration tests build a small index from fixture papers and exercise every tool through a real rmcp client over HTTP. A retrieval evaluation on a golden query set picks the embedding and reranker models and guards against regressions.

- Unit:
  - Selector parsing (ids, year, `all`, mixed and invalid input) and resolving unversioned ids to the latest revision.
  - Chunker: code fences, pipe tables and mixed HTML tables kept whole; front matter and `<style>` excluded; heading paths and line ranges exact; budget splitting only at block boundaries.
  - Prefix output is deterministic for a given metadata row and heading path.
  - Bearer middleware: missing, malformed and wrong tokens give 401, and a correct token passes.
  - `get_paper_lines` slicing against a Python-slice oracle for positive, negative and missing bounds, including `start=0` rejection and clamping.
  - Cursor cache: page stability, page-size boundaries, TTL expiry, LRU eviction and the unknown-cursor error.
- Integration and end-to-end:
  - Index a fixture set that includes a multi-revision paper, one PDF-origin and one HTML-origin paper, and a paper with no Markdown. Assert skip-on-rerun, `--force` and per-paper failure reporting.
  - Run `serve`, connect with an rmcp Streamable HTTP client and call all eight tools. Check that `search_papers` line ranges round-trip through `get_paper_lines`.
  - Run an index while the server is serving, and check that the server keeps answering and then sees the new data.
  - For each `mode`, page a `search_papers` result to its end and check that the result count equals every keyword match (keyword), `top_k` (vector) or their deduplicated union (hybrid). Check that an index run between pages does not change later pages of an existing cursor.
- Regression, security, and performance:
  - A golden query set of at least 50 hand-labeled queries, each with its relevant paper revisions and line ranges, covering prose questions, exact C++ identifiers, wording-diff phrases and revision-specific questions, measuring recall@20 for dense-only, BM25-only, hybrid and hybrid plus rerank.
  - Record indexing throughput and p95 `search_papers` latency on the full corpus.
  - The token never appears in logs, as asserted by a captured-log test.
- Exit criteria:
  - All unit and integration tests pass in CI.
  - The chosen configuration meets the recall success criterion, and a full-corpus index completes and serves.

</verification-contract>
<decision-record>

## Decision Record

- Decisions:
  - Build on the existing paperstore for metadata and Markdown. User: "of course use the existing paperstore".
  - Ingest is selector-driven, never automatic, and works like tomd/`paperflow convert`. User: "i can process all papers or one year or 1+ paper ids at a time" and "never automatic, always on demand processing for 1 or many papers".
  - The shared server uses Streamable HTTP with a simple bearer token. User: "it will be a shared server. use a simple bearer token for auth".
  - There is no stdio transport. User: "no stdio transport".
  - TLS terminates at a reverse proxy, and the server does no TLS. User: "TLS does terminuates at a reverse proxy in front of this, no TLS/SSL handling needed here".
  - The MCP surface is tools only: `search_papers`, `search_metadata`, `get_paper`, `get_paper_lines`, `get_paper_metadata`, `list_revisions`, `get_paper_outline`, `diff_revisions`. User: "no resources, just tools", "we must be able to fetch partial papers md by line number", "get_paper_outline to return the outline".
  - No arbitrary output limits: full papers and diffs are uncapped, and search results are reranked and paged to the end. User: "nothing should be arbitrarily restricted here. yes do the reranking, but give me paging to look through all serch results returned" and "there hsould be a way to get a full paper and also a paper by lines numbers. again no arbitrary defaults".
  - `get_paper_lines` takes `start` or `end` (at least one required), 1-indexed with strict Python slice semantics. User: "indices sholud work like 1-indexed but slice like Python", "start=1, end=3 gives lines 1 and 2 of the paper", and "like python, leave out end to go to the end of the paper. and leave out start tot start from the beginning".
  - Every search names its mode. User: "every serach should specify the type: vector, keyword, or hybrid". The vector side returns a tunable `top_k` (default 1000), because vector search ranks the whole corpus and needs a cutoff. User: "make it easily tunable, but do set a sensible default for htis use case".
  - Reranking covers the first 150 results by default, with the rest kept in order. Rationale: cross-encoder cost grows linearly with the number of candidates, so reranking every keyword match would make broad queries take minutes, while dropping results would contradict the no-arbitrary-limits decision.
  - Paging uses stable server-side cursors with a server-set page size. User: "whatevers standard, i think stable is more standard than requiring page sdize". MCP's own list pagination uses opaque cursors with server-determined page sizes.
  - Contextual retrieval uses a deterministic per-chunk prefix instead of LLM-written context. User: "follow Anthropic's contextual retrieval blog post closely (no LLM, just a deterministic prefix)". The prefixed text feeds both embeddings and BM25, and the pipeline includes fusion and a reranking stage.
  - Embedding and reranker models are chosen by the golden-set evaluation, with fitness for mixed code, prose and table content as the criterion. User: "no preference here, just make sure it will work fine with the standards papers".
  - Search reads the index while read tools read paperstore, so reads always reflect current Markdown.
  - `search_papers` searches only the latest revision by default, and `all_revisions: true` searches every indexed revision without collapsing. User chose option "a" over collapsing all-revision results per paper. Rationale: default results reflect each proposal's current text, while history stays one flag away and `diff_revisions`/`list_revisions` cover change questions.
  - rmcp is the MCP SDK because it is the official SDK and by far the most used.
  - The vector store is embedded LanceDB. User: "Lance DB if thats what you recommend". Rationale: a single binary with no extra service, documented cross-process read freshness, and a Rust hybrid API with filter and rerank.
  - Paperstore is the only data source, with no network access. User: "paperstore only of course. this MCP server should not serach the web for anything or download anything, only use paperstore to get the papers and metadata".
- Rejected alternatives:
  - MCP resources and prompts: rejected by the user. Revisit if a client needs resource subscriptions.
  - LLM-generated chunk context: rejected by the user. Revisit if the deterministic prefix fails the recall criterion.
  - Fixed-size chunking: it splits code and tables. Revisit only as a fallback for documents without headings.
  - Qdrant as a local server: it is a second service that needs an API key, block storage and version-matched snapshots. Revisit if other services need to query the index or LanceDB 0.x API churn becomes costly.
  - wg21.link `index.json` metadata: rejected by the user, because paperstore is the only source.
  - Qdrant Edge (in-process): beta API. Revisit when it is stable.
  - DIY tantivy plus a separate ANN crate: two indexes with filtering across both and custom fusion. Revisit if LanceDB does not meet the recall or latency needs.
- Assumptions, risks, and notes:
  - Of 9,272 source files, 9,060 have Markdown (`data/paperstore/`), so papers without Markdown get metadata-only records.
  - Line numbers come from the Markdown at index time. Reconverting a paper makes its ranges stale until it is reindexed, and the content-hash check detects this.
  - `ort` 2.x, under fastembed, is still a release candidate, so pin it exactly. The candle backend is the fallback.
  - LanceDB is 0.x with frequent releases and a heavy Arrow/DataFusion dependency tree, so pin it exactly and expect longer builds.
  - LanceDB facts the design relies on come from docs.rs lancedb 0.40.0 and the docs.lancedb.com consistency, full-text-search and hybrid-search pages:
    - With `read_consistency_interval` set, readers see writes from other processes.
    - Rows appended after FTS indexing are flat-scanned until `optimize` runs.
    - The Rust `VectorQuery` exposes `only_if`, `full_text_search`, `rerank` and `execute_hybrid`.
  - LanceDB full-text queries take a result limit. To return every keyword match, keyword queries must set a limit at least as large as the chunk count matching the filter.
  - When paperstore lists a newer revision than the index holds, the indexed older revision has `is_latest = false`, so that paper drops out of default search until the newer revision is indexed.
  - Model files are provisioned into the model directory out of band, because the binaries never download.
  - Paperstore has no mailing field. `mailing_date` (`YYYY-MM`) serves as the mailing (`crates/paperstore/src/row.rs`).
- Decisions added during decomposition (not confirmed by the user):
  - The new crate and binary are named `paper-mcp` (`crates/paper-mcp`), and the evaluation harness is a second binary, `paper-mcp-eval`, in the same crate.
  - The index lives in `$WG21_DATA_DIR/paper-mcp/` (LanceDB under `lancedb/`, writer lock `index.lock`), located through a new `StorageBackend::tool_dir` method so no caller builds paths from the workspace dir.
  - The bearer token comes from `PAPER_MCP_TOKEN` or `--token-file`, and the model directory from `PAPER_MCP_MODEL_DIR` or `--model-dir`.
  - Integration tests and CI use deterministic test-double models behind a `test-models` cargo feature, so CI needs no model files.
  - Hybrid fusion uses RRF with a named constant `RRF_K` (default 60).
  - The writer lock uses `std::fs::File::try_lock` (stable since Rust 1.89).

</decision-record>
<project-survey>

## Project Survey

- Status: complete
- Build command: Rust `cargo build --workspace --locked` (CI builds `cargo build --release -p paperweight --locked`; needs the `vendor/promptforge` submodule checked out). Python has no build step: `uv sync --all-packages --group dev`.
- Focused test command pattern: Rust `cargo test -p <crate> <test_name_filter> --locked`; Python `uv run pytest <path/to/test_file.py>::<test_name>`.
- Component test command pattern: Rust `cargo test -p <crate> --locked`; Python `uv run pytest packages/<package>/tests` (or `uv run --package <package> pytest`).
- Full-suite test command: `cargo test --workspace --locked` plus `uv run pytest` (pytest testpaths cover every package except `whisker`, plus `tests/`; `network`-marked tests are deselected by default).
- Linter command: Rust `cargo clippy --workspace --all-targets --locked` (workspace lints deny `clippy::all`, `unwrap_used`, `expect_used`; warn `pedantic`; forbid `unsafe_code`). Python `uv run ruff check packages tests` and `uv run pyright packages` (pyright is non-blocking in CI).
- Formatter check command: Rust `cargo fmt --all --check` (`rustfmt.toml`: `style_edition = "2024"`). Python: None (CI runs no format check).
- Docs command: Rust `cargo doc --workspace --no-deps` (rustdoc denies broken and private intra-doc links; `missing_docs` warns). Python: None.
- Test placement and naming conventions: Rust unit tests live in `#[cfg(test)] mod tests` at the bottom of each source file; integration tests live in a single `tests/it/main.rs` binary per crate (`crates/paperstore-sqlite/tests/it/`); dev-deps `tempfile` and `temp-env` for filesystem and env isolation. Python tests live in `packages/<name>/tests/test_<topic>.py` with a per-package `conftest.py` and `fixtures/` dir; cross-package integration tests live in top-level `tests/` (`test_end_to_end_*.py`); live-network tests carry `@pytest.mark.network`.
- Directory map:
  - `crates/`: Rust Cargo workspace members. `paperstore` (abstract storage API: `StorageBackend` trait, `PaperRow`, `PaperNum`, `Error`, `WG21_DATA_DIR` resolution), `paperstore-sqlite` (rusqlite backend), `paperweight` (CLI running a promptforge prompt against a paper; extra bin `promptforge-cli`).
  - `vendor/promptforge/`: git submodule, its own Cargo workspace, excluded from the root workspace; `paperweight` depends on its `harness` crate by path.
  - `packages/`: Python uv workspace members: `paperstore`, `mailing`, `tomd`, `pipeline`, `agora`, `assay`, `cli`, `preview`, `cpp-mcp` (Python MCP server over C++ sources), `whisker`.
  - `tests/`: cross-package Python integration tests.
  - `study/`, `research/`, `herald/`: experiments, research notes, design docs (not production).
  - `data/`: local workspace sample (`paperstore.db`, `paperstore/` with `<pid>.md` etc.).
  - `vibe/`: planning artifacts. `html/`: static output. `target/`: Cargo build output (untracked).
  - Root docs: `CLAUDE.md`, `ARCHITECTURE.md`, `DESIGN.md`, `MODELS.md`, `SERVICES.toml` (service and classifier inventory), `justfile` (mounts `packages/tomd/justfile`).
  - `.github/workflows/tests.yml`: ruff, pyright, per-package pytest matrix (ubuntu, windows), `paperweight` cargo build and test, integration tests.
- Component boundaries: Rust `paperstore` has no internal deps; `paperstore-sqlite` depends on `paperstore`; `paperweight` depends on `paperstore`, `paperstore-sqlite`, and vendored `harness`. Python `paperstore` has no internal deps; `mailing`, `tomd`, `pipeline` depend on `paperstore`; `agora`, `assay` (and dissect) depend on `pipeline`; `cli` sits on top and orchestrates. Production packages never import from `study`. The Rust and Python sides share only the on-disk workspace (`paperstore.db` plus `paperstore/` files), not code.
- Conventions summary: storage only through the `StorageBackend` abstraction, never hand-built paths; library code returns data and callers (CLI) persist; Rust edition 2024, workspace-inherited package metadata and `[lints] workspace = true` in every crate, `thiserror` errors in libraries and `anyhow` in binaries, no `unwrap`/`expect`, `//!` crate docs and doc comments on public items; Python 3.12, file-level imports only, `logging` not `print`, named module-level constants for tunables, `__init__.py` carries re-exports only, BSL-1.0 headers on new files; LLM calls go through `pipeline.run_agent`/`run_task` with serial-by-default determinism; no em dashes in prose.

</project-survey>
<execution-plan>

## Execution Instructions

Components in dependency order: paperstore listing (1), paper-mcp core library and `index` command (2 to 7), search and paging (8 to 9), MCP server (10 to 12), retrieval evaluation and model selection (13 to 14), CI (15). Every step lands as one local commit containing its code and tests, and must pass `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked` and the focused `cargo test -p <crate> --locked` command for the crates it touches. New `.rs` files carry the BSL-1.0 header and no em dashes.

<step-1>

### Step 1: Paperstore listing methods and tool directory [completed]

- Component: paperstore-listing
- Placement: first, because selection, unversioned-id resolution, `list_revisions` and the index location all go through the paperstore abstraction.
- Pieces: one piece, built sequentially (trait, then backend).
- `crates/paperstore/src/backend.rs`: add to `StorageBackend` the methods `list_all_paper_ids(&self) -> Result<Vec<PaperNum>>`, `list_paper_ids_for_year(&self, year: u16) -> Result<Vec<PaperNum>>`, `list_revisions(&self, num: &PaperNum) -> Result<Vec<PaperRow>>` (every revision of the paper number of `num`, newest first by revision) and `tool_dir(&self, tool: &str) -> PathBuf` (a per-tool subdirectory under the workspace, so callers never build paths from the workspace dir). Add a helper on `PaperNum` (`crates/paperstore/src/num.rs`) that returns the paper number without revision and the revision number, used to group revisions.
- `crates/paperstore-sqlite/src/backend.rs`: implement the methods with SQL mirroring `list_all_paper_ids` and `list_papers_for_year` in `packages/paperstore/src/paperstore/sqlite_backend.py`, sorted deterministically.
- Tests: unit tests for the `PaperNum` helper in `num.rs`; integration tests in `crates/paperstore-sqlite/tests/it/` against a tempdir SQLite fixture covering all ids, one year, an empty year, multi-revision ordering, an unknown paper number and `tool_dir` placement.

</step-1>

<step-2>

### Step 2: paper-mcp crate scaffold and paper selection [completed]

- Component: paper-mcp-core
- Placement: after paperstore-listing, which selection depends on. The core component is the indexing half of the system and is useful on its own (it builds an index that later components read).
- Pieces: scaffold and selection (step 2), chunk and prefix (steps 3 to 4), models (step 5), index store (step 6), ingest command (step 7). Built sequentially, because each piece consumes the previous one's types.
- `crates/paper-mcp/Cargo.toml`: workspace-inherited package metadata, `[lints] workspace = true`, deps `paperstore`, `paperstore-sqlite`, `clap`, `anyhow`, `thiserror`, `tracing`, `tracing-subscriber`; dev-deps `tempfile`, `temp-env`. Update `Cargo.lock`.
- `crates/paper-mcp/src/lib.rs` (crate docs, module re-exports), `src/main.rs` (clap `Cli` with `index` and `serve` subcommands, `serve` returning a not-yet-implemented error until step 10), `src/error.rs` (`thiserror` library error enum).
- `src/selection.rs`: `Selector` enum (`Ids(Vec<PaperNum>)`, `Year(u16)`, `All`), `parse_selectors(&[String]) -> Result<Vec<Selector>>` (rejects mixing `all` with others and malformed input), `resolve(&dyn StorageBackend, &[Selector]) -> Result<Vec<PaperNum>>` (deduplicated, sorted), `resolve_latest(&dyn StorageBackend, &PaperNum) -> Result<PaperNum>` (unversioned id to highest listed revision; versioned id passes through after an existence check).
- Tests: unit tests in `selection.rs` for ids, year, `all`, mixed and invalid input and latest-revision resolution using an in-memory `StorageBackend` test double.

</step-2>

<step-3>

### Step 3: Structure-aware Markdown chunker [completed]

- Component: paper-mcp-core
- `src/chunk.rs`: `Chunk { heading_path: Vec<String>, start: usize, end: usize, text: String }` with 1-based inclusive `start` and exclusive `end`; `Section { level, heading, start, end }` and `outline(md) -> Vec<Section>` (reused by `get_paper_outline`); `chunk_markdown(md: &str, budget: usize, count_tokens: &dyn Fn(&str) -> usize) -> Vec<Chunk>`. It skips YAML front matter and `<style>` blocks from chunk bodies while counting their lines, splits on `##`/`###` headings, treats fenced code blocks, GFM pipe tables and `<!-- tomd:mixed-table -->` HTML tables as atomic blocks, splits over-budget sections only at block boundaries, and emits a lone oversized block as its own chunk split at line boundaries. A `CHUNKER_VERSION` constant feeds the index-format version.
- Tests: unit tests with inline Markdown plus fixtures derived from `data/paperstore/p2169r4.md` and `data/paperstore/p2322r4.md` copied into `crates/paper-mcp/tests/fixtures/`: code fences, pipe tables and mixed HTML tables kept whole; front matter and `<style>` excluded; heading paths and line ranges exact (slicing the source lines by each range reproduces the chunk text); budget splitting only at block boundaries; headingless documents.

</step-3>

<step-4>

### Step 4: Deterministic prefix, content hash and index-format version [completed]

- Component: paper-mcp-core
- `src/prefix.rs`: `build_prefix(row: &PaperRow, heading_path: &[String]) -> String` from paper id with revision, title, authors, target_group, document date and heading path in a fixed order and layout; `prefixed_text(prefix, chunk) -> String`; `PREFIX_VERSION` constant.
- `src/format.rs`: `content_hash(md: &str) -> String` (SHA-256 hex via `sha2`) and `IndexFormatVersion { chunker, prefix, embedder_id, reranker_id }` with a stable string rendering stored in index records.
- Tests: prefix output is byte-identical across calls for a given row and heading path, field order and empty-field handling are fixed, the hash is stable for known input, and changing any version component changes the rendered format version.

</step-4>

<step-5>

### Step 5: Embedder and reranker traits with local backends [completed]

- Component: paper-mcp-core
- `src/model.rs`: `Embedder` trait (`id() -> &str`, `dim() -> usize`, `max_tokens() -> usize`, `count_tokens(&str) -> usize`, `embed_documents(&[String])`, `embed_query(&str)`) and `Reranker` trait (`id()`, `rerank(query, &[String]) -> Vec<f32>`).
- `src/model/onnx.rs`: `OnnxEmbedder` and `OnnxReranker` built on `fastembed` with `ort` pinned exactly (`=2.0.0-rc.x`), loading only from files in a configured model directory (`--model-dir` or `PAPER_MCP_MODEL_DIR`) through fastembed's user-defined model APIs, never downloading. Missing or unreadable model files produce a clear startup error naming the expected files. The candle backend stays a fallback behind the same trait and is not built now.
- `src/model/fake.rs` behind cargo feature `test-models`: `HashEmbedder` (deterministic token-hash vectors, so lexically similar text lands close) and `IdentityReranker`, used by integration tests and CI so they need no model files.
- Tests: fakes are deterministic and dimension-correct; loading from an empty or partial model directory fails with the expected error and performs no network access; an `#[ignore]` test exercises the real ONNX backends when `PAPER_MCP_MODEL_DIR` is set.

</step-5>

<step-6>

### Step 6: LanceDB index store [completed]

- Component: paper-mcp-core
- `Cargo.toml` (root): raise workspace `rust-version` to 1.91. `crates/paper-mcp/Cargo.toml`: add `lancedb = "=0.40.0"` and the matching `arrow-array`/`arrow-schema` versions, pinned exactly; update `Cargo.lock`.
- `src/store.rs`: `IndexStore` trait with `freshness(&PaperNum) -> Option<(content_hash, format_version)>`, `upsert_paper(PaperRecords)` (replaces every chunk and the metadata record of one revision), `delete_paper`, `set_latest(paper_number, latest: &PaperNum)` (recomputes `is_latest` for every indexed revision of the number), `search_vector`, `search_keyword`, `count_keyword_matches`, `search_metadata`, `optimize`. Record types `ChunkRecord` (paper id, revision, paper number, `is_latest`, heading path, start, end, unprefixed text, prefixed text, vector, content hash, format version, filter fields) and `MetaRecord` (the full `PaperRow` plus `is_latest`, `has_markdown`, content hash, format version).
- `src/filter.rs`: `SearchFilter` (paper id, revision, target_group, author, year, mailing_date, document_date range, intent, latest-only flag) rendered to a LanceDB SQL predicate with string escaping, shared by every search side.
- `src/store/lance.rs`: `LanceStore` opening `$tool_dir/lancedb` with `read_consistency_interval` set; `chunks` and `paper_meta` tables; vector index on chunk vectors; FTS indexes on prefixed chunk text and on metadata title and authors with a tokenizer configured for C++ text (stemming and stop-word removal off, base tokenizer and lowercasing per the step 14 evaluation, defaulting to whitespace-based splitting that keeps `::`, `_` and `<>`-bearing identifiers searchable); keyword queries set the FTS limit from `count_keyword_matches` so every match returns.
- Tests: tempdir integration tests in `crates/paper-mcp/tests/it/` using `HashEmbedder`: upsert, replace and delete by paper; freshness lookup; `is_latest` recomputation; filter predicates including quote escaping; keyword search finds `std::move_only_function` and returns every match; metadata search by title and author; reads from a second `LanceStore` handle see writes after `optimize`.

</step-6>

<step-7>

### Step 7: Ingest pipeline and `index` command [completed]

- Component: paper-mcp-core
- `src/ingest.rs`: `IndexLock` holding an exclusive lock on `$tool_dir/index.lock` via `std::fs::File::try_lock`, failing fast when another writer holds it; `index_papers(backend, store, embedder, nums, force, workers) -> IndexReport` running per paper: `meta`, `paper_md`, freshness check against content hash and format version (skip unless `--force`), chunk, prefix, embed, upsert, then `set_latest` per touched paper number using `list_revisions`; `MissingMarkdown` writes only the metadata record and is reported as no content; per-paper errors are collected without aborting; `optimize` runs at the end. `--workers N` bounds concurrent paper preparation while store writes stay serialized.
- `src/main.rs`: `index <selector>... [--force] [--workers N] [--model-dir]` wiring with startup checks for `WG21_DATA_DIR` and model files, a per-paper summary on stderr, and a non-zero exit if any paper failed.
- Tests: integration tests with a fixture workspace under `crates/paper-mcp/tests/fixtures/workspace/` (SQLite db plus Markdown) containing a multi-revision paper (P2300R0 to R10 metadata, small Markdown), one PDF-origin and one HTML-origin paper and one paper without Markdown, using `HashEmbedder`: selection by ids, year and `all` matches paperstore listings; rerun without `--force` skips every unchanged paper; `--force` reindexes; changing Markdown triggers reindex; an injected per-paper failure is reported and yields a non-zero result while other papers index; a held lock blocks a second writer.

</step-7>

<step-8>

### Step 8: Cursor cache [completed]

- Component: search-and-paging
- Placement: after paper-mcp-core, whose store it queries. Search and paging is a reusable query library that the server and the evaluation harness both call.
- Pieces: cursor cache (step 8) and search pipeline (step 9), built sequentially: the cache is pure and generic, and the pipeline produces the result lists it stores.
- `src/cursor.rs`: `CursorCache<T>` with configurable page size (default 20), idle TTL (default 15 minutes) and capacity (default 64, LRU eviction), `start(results: Vec<T>) -> Page<T>` and `next(cursor: &str) -> Result<Page<T>, CursorError>`; `Page { items, next_cursor: Option<String> }`; opaque random cursor ids; `CursorError::Expired`/`Unknown` with a message telling the client to rerun the search. Time comes from an injectable clock for tests.
- Tests: unit tests for page stability, exact page-size boundaries (empty, one page, exact multiple), TTL expiry, LRU eviction order and the unknown-cursor error.

</step-8>

<step-9>

### Step 9: Search pipeline [completed]

- Component: search-and-paging
- `src/search.rs`: `SearchMode` (`Vector`, `Keyword`, `Hybrid`), `SearchRequest` (query, mode, `SearchFilter`, `all_revisions`, `top_k` override), `SearchParams` (`top_k` default 1000, `rerank_depth` default 150), `SearchHit` (paper id with revision, title, heading path, start, end, unprefixed text, score), `search_chunks(store, embedder, reranker, params, req) -> Vec<SearchHit>`: latest-only filter unless `all_revisions` or a versioned paper id; keyword returns every match, vector returns `top_k`, hybrid fuses every keyword match with the `top_k` nearest via RRF (`RRF_K` constant, deduplicated by chunk); the reranker reorders the first `rerank_depth` hits and the rest keep their order. `search_metadata(store, req) -> Vec<MetaRecord>` returns every matching metadata row.
- Tests: unit tests for RRF fusion and rerank-depth splicing; integration tests on the step 7 fixture index: result counts equal every keyword match, `top_k`, and their deduplicated union for the three modes; default search for P2300 returns only P2300R10 chunks, `all_revisions` returns other revisions, and `P2300R4` restricts to that revision; filters apply to every side.

</step-9>

<step-10>

### Step 10: Server configuration, bearer auth and Streamable HTTP host [completed]

- Component: mcp-server
- Placement: after search-and-paging, whose search functions and cursor cache back the tools. This component is the shippable shared service.
- Pieces: host and auth (step 10), read tools (step 11), search tools (step 12). Sequential: tools need the host, and read tools come first because search-tool tests round-trip through `get_paper_lines`.
- `crates/paper-mcp/Cargo.toml`: add `rmcp` 3.x (features `server`, `macros`, `transport-streamable-http-server`, plus `client` and `transport-streamable-http-client` as dev features), `axum`, `tokio`, `subtle`, `schemars`, `serde`.
- `src/server/config.rs`: `ServeConfig` (bind address, token from `PAPER_MCP_TOKEN` or `--token-file`, model dir, `top_k`, `rerank_depth`, page size, cursor TTL, cursor capacity) with Functional Specification defaults; startup validation failing clearly when the token, index, model files or `WG21_DATA_DIR` are missing. The token type redacts itself in `Debug` and `Display`.
- `src/server/auth.rs`: axum middleware comparing `Authorization: Bearer <token>` in constant time with `subtle`, returning 401 otherwise, never logging or echoing the token.
- `src/server/mod.rs`: `PaperServer` with `#[tool_router(server_handler)]` (tools only, no resources or prompts), `build_app(config) -> axum::Router` mounting rmcp `StreamableHttpService` behind the middleware; `serve` subcommand in `src/main.rs`.
- Tests: unit tests for missing, malformed, wrong-scheme and wrong tokens giving 401 and a correct token passing; config validation errors; a captured-log test (`tracing` test subscriber) asserting the token never appears in logs, including on auth failure and startup errors.

</step-10>

<step-11>

### Step 11: Read tools [completed]

- Component: mcp-server
- `src/lines.rs`: `slice_lines(md, start: Option<i64>, end: Option<i64>) -> Result<LineSlice>` with 1-based Python slice semantics (`start` inclusive, `end` exclusive, negatives from the end, clamping, `start=0` and both-missing rejected), returning text and `total_lines`.
- `src/server/tools.rs`: tool methods and `schemars` parameter types for `get_paper` (full Markdown, uncapped), `get_paper_lines`, `get_paper_metadata` (full `PaperRow`), `list_revisions` (date, mailing, `previous_version`, newest first), `get_paper_outline` (via `chunk::outline`, end exclusive) and `diff_revisions` (unified diff via `similar`, uncapped, error when the two ids are different paper numbers). All read paperstore directly, resolve unversioned ids with `selection::resolve_latest`, and map unknown papers, invalid ranges and revision mismatches to MCP tool errors.
- Tests: `slice_lines` against a Python-slice oracle over every combination of positive, negative and missing bounds in a range around the line count (oracle computed in Rust from 0-based slice rules); integration tests in `tests/it/` start the server on an ephemeral port against the fixture workspace and call each read tool through an rmcp Streamable HTTP client, including the error cases.

</step-11>

<step-12>

### Step 12: Search tools and end-to-end server behavior [completed]

- Component: mcp-server
- `src/server/tools.rs`: `search_papers` (query, required `mode`, filters, `all_revisions`, `top_k`, `cursor`) and `search_metadata` (keywords on title and authors, same filters and `cursor`), each running the search on the first call, storing the full ranked list in a shared `CursorCache` and returning the first page plus `next_cursor`; an expired or unknown cursor returns a tool error. The server opens the index read-only through `LanceStore` with `read_consistency_interval`, so it picks up new index runs without a restart.
- Tests: integration tests over rmcp HTTP: all eight tools succeed with a valid token and every request without one gets 401; every `search_papers` hit's `start`/`end` passed to `get_paper_lines` returns that hit's text; paging each mode to its end yields the step 9 counts; an index run between pages leaves later pages of an existing cursor unchanged; running `index_papers` while the server serves keeps it answering and new data becomes visible afterwards.

</step-12>

<step-13>

### Step 13: Golden query set and evaluation harness

- Component: retrieval-evaluation
- Placement: after the server, because it measures the complete search stack and server latency. It is a standalone tool for choosing and guarding configurations.
- Pieces: golden set and harness (step 13), then model selection (step 14). Sequential, because selection runs the harness.
- `crates/paper-mcp/eval/golden.jsonl`: at least 50 hand-labeled queries, each with relevant paper revisions and line ranges, covering prose questions, exact C++ identifiers, wording-diff phrases and revision-specific questions, labeled against the real corpus in `data/`.
- `crates/paper-mcp/src/bin/paper-mcp-eval.rs` and `src/eval.rs`: loads the golden set, runs dense-only, BM25-only, hybrid and hybrid plus rerank through `search::search_chunks` against a built index, and reports recall@20 per configuration, indexing throughput (from the index report) and p95 `search_papers` latency. Library code returns data; the binary prints the report.
- `crates/paper-mcp/README.md`: model provisioning into the model directory out of band, index and serve usage, eval usage.
- Tests: unit tests for golden-set parsing and validation (at least 50 queries, every category present, ranges well-formed) and for recall@k computation, including hits that overlap a labeled line range.

</step-13>

<step-14>

### Step 14: Model and tokenizer selection on the full corpus

- Component: retrieval-evaluation
- Provision candidate open-weight embedding and cross-encoder models into the model directory out of band (development only), index the full corpus for each candidate configuration, and sweep chunk budget and FTS tokenizer settings with `paper-mcp-eval`.
- Record candidates, recall@20 for all four configurations, throughput and p95 latency in `crates/paper-mcp/eval/RESULTS.md`. Pin the chosen embedder, reranker, chunk budget and tokenizer settings as named constants in `src/model.rs`, `src/chunk.rs` and `src/store/lance.rs`, which updates `IndexFormatVersion`. Add the chosen models to this plan's decision record.
- Exit check: the chosen configuration's recall@20 beats dense-only and BM25-only, and a full-corpus index completes and serves.
- Tests: a regression test running the harness on a small fixture subset with `HashEmbedder` to guard the harness, plus an `#[ignore]` test asserting the recorded recall criterion when `PAPER_MCP_MODEL_DIR` and the full index are present.

</step-14>

<step-15>

### Step 15: CI for paper-mcp and paperstore crates

- Component: ci
- Placement: last, because it builds and tests every crate the earlier components add or change.
- Pieces: one piece.
- `.github/workflows/tests.yml`: a cargo job that builds `paper-mcp`, `paperstore` and `paperstore-sqlite` with `--locked` and runs `cargo test -p paperstore -p paperstore-sqlite -p paper-mcp --features paper-mcp/test-models --locked`, with the toolchain from `rust-toolchain.toml` and cargo caching. Model-dependent and full-corpus tests stay `#[ignore]` and need no model files or network.
- Tests: the workflow passes locally via the same commands; validate the YAML with a local parse. Do not push.

</step-15>

</execution-plan>
