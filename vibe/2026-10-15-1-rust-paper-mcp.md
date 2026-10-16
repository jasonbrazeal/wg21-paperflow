# WG21 Paper RAG and MCP Server (Rust)

<product-contract>

## Product Requirements

LLM clients need grounded, citable access to the WG21 paper corpus: paper text, every revision, and the metadata committee work depends on (authors, subgroup, mailing, disposition, revision lineage). This plan adds a Rust retrieval system that indexes the corpus paperflow already stores in paperstore and exposes search, paper text, and metadata as MCP (Model Context Protocol) tools. Retrieval supports keyword (BM25), vector, and hybrid search with metadata filtering, and it is revision-aware so version-sensitive questions resolve to the right revision. The server can be deployed and run alongside the existing Python `cpp-mcp` server on the same host, using the same authentication.

- Problem and users:
  - LLM clients (desktop assistants, coding agents, and the repo's own agents) answer questions about WG21 proposals from model memory, which is stale and cannot distinguish revisions. The users are the operator and other committee-facing tooling that speaks MCP.
- Goals:
  - Index the full text of every paper revision in paperstore that has Markdown, with per-revision metadata.
  - Provide keyword, vector, and hybrid retrieval with metadata pre-filtering (paper IDs, revision scope, year, mailing, subgroup, author, intent, chunk kind).
  - Make retrieval revision-aware: corpus-wide searches default to the latest revision of each paper with an opt-in `all` scope, and searches filtered to paper IDs return only those exact papers.
  - Expose papers, metadata, line ranges, and search as MCP tools.
  - Run fully in-process with local open-weight embeddings and no external service at query time.
- Non-goals:
  - Generating answers. The server returns evidence, and the client LLM writes the answer.
  - Re-implementing paperflow's scraping, downloading, or HTML/PDF conversion in Rust.
  - Modifying paperstore data. The RAG system reads it only.
  - MCP resources and prompts.
- Success criteria:
  - Over the fixture corpus, every MCP tool in the Functional Specification returns schema-valid results over Streamable HTTP.
  - Over the fixture corpus, for each of the three search types, a search filtered to a paper ID (for example `P2300R10`) returns chunks only from that paper, and a search with no paper ID filter returns only latest-revision chunks by default.
  - Incremental reindexing re-embeds only papers whose Markdown, chunker version, or embedding model changed.
- Constraints:
  - All paper text and metadata come from paperstore under `$WG21_DATA_DIR` (`crates/paperstore/src/workspace.rs`), accessed through the `StorageBackend` abstraction (`crates/paperstore/src/backend.rs`, `crates/paperstore-sqlite`). Nothing fetches paper data from the web. The only permitted network download is the embedding model weights.
  - Corpus scope is exactly what paperstore holds: currently 9,339 revisions from 2011 to 2026 (7,601 P, 1,734 N, 4 SD), one row per revision (`packages/mailing/src/mailing/scrape.py` `MAILING_EARLIEST_YEAR = 2011`; `data/paperstore.db`).
  - Rust, in the existing Cargo workspace (`Cargo.toml`: `members = ["crates/*"]`, edition 2024, `rust-version = "1.89"`, toolchain 1.98.0 per `rust-toolchain.toml`, BSL-1.0 license).
  - Workspace lints apply: `unsafe_code` forbidden, `clippy::all`, `unwrap_used`, `expect_used` denied, `pedantic` and `missing_docs` warned (`Cargo.toml`). There is no `[workspace.dependencies]` table, so the crate pins its own versions.
  - No em dashes in code, docs, or output; spell names `paperflow`, `tomd`, `WG21` (`CLAUDE.md`, Style).
- Open questions:
  - None

## Functional Specification

The system has two entry points in one binary: an indexer that builds and incrementally refreshes a local index from paperstore, and an MCP server that answers tool calls from that index and from paperstore. Search returns ranked chunks with citation metadata and line ranges; line and metadata access are exact lookups. HTTP clients authenticate exactly as they do against `cpp-mcp`.

- Actors and workflows:
  - Operator runs paperflow (`paperflow mailing|download|convert`, `README.md`) to populate `$WG21_DATA_DIR`, then runs `paper-mcp index`, then starts `paper-mcp serve`.
  - MCP client connects over Streamable HTTP, lists tools, and calls them. Streamable HTTP is the only transport.
- Inputs and outputs:
  - Indexer input: `$WG21_DATA_DIR/paperstore.db` table `papers` (paper_id, year, title, authors, target_group, intent, url, document_date, mailing_date, disposition, previous_version, markdown_path, line_count, status; `packages/paperstore/src/paperstore/sqlite_backend.py`) and `$WG21_DATA_DIR/paperstore/<pid>.md` (tomd Markdown with fixed-order YAML front matter: title, document, date, intent, audience, reply-to).
  - Indexer output: a local index directory under `$WG21_DATA_DIR` (see Technical Design).
  - MCP tools (names and parameter names are the contract):
    - `search_papers`: required `query` and required `search_type` in {`hybrid`, `vector`, `keyword`}; optional filters (one or more exact paper IDs, revision scope `latest` | `all`, year range, mailing, subgroup/target_group, author substring, intent, chunk kind) and `limit`. `keyword` is BM25 full-text search, `vector` is dense embedding search, `hybrid` fuses both with RRF (reciprocal rank fusion). Returns ranked hits, each with paper_id, revision number, title, authors, section path, chunk kind, `start_line`/`end_line` (usable directly with `get_paper_lines`), text, score, and canonical URL `https://wg21.link/<pid>` (computed, not fetched).
    - `get_paper_metadata`: one paper_id. Returns metadata for that revision plus the list of all revisions of that paper.
    - `list_revisions`: a paper ID with or without its `R<n>` suffix (`P2300` and `P2300R3` are both valid). This is the only tool that accepts an ID without a revision. Returns all revisions in paperstore in order with date, mailing, disposition, and previous_version.
    - `list_papers`: the `search_papers` filters minus `query` and `search_type`, paginated. Returns metadata rows.
    - `get_paper_lines`: paper_id plus optional integer `start` and `end`, with Python slice semantics over the lines of `<pid>.md` (front matter included): 0-based, `start` inclusive, `end` exclusive, omitted `start` means the first line, omitted `end` means through the last line, negative values count from the end, and out-of-range values clamp rather than error (an empty range returns no text). Returns the resolved `start` and `end`, `total_lines`, and the text.
  - MCP resources: None. MCP prompts: None.
- States and validation:
  - paper_id input is validated with `PaperNum::parse` (`crates/paperstore/src/num.rs`) and, for every tool except `list_revisions`, must match a paperstore row exactly. P papers always carry an `R<n>` revision; N and SD papers have no revision suffix (e.g. `N4950`), per `data/paperstore.db`.
  - A paper is `latest` when it has the highest revision number among paperstore rows whose paper_ids differ from it only in the `R<n>` suffix (its revisions). N and SD papers have no revisions and are always latest.
  - A paper ID filter matches only those exact papers. Revision scope applies only when no paper ID filter is given: `latest` (the default) restricts to latest revisions and `all` includes every revision.
  - Papers with `status < 0` or no Markdown are indexed as metadata-only (`has_text = false`): visible to metadata tools, excluded from search, and `get_paper_lines` returns a tool error saying no text is available.
  - The server refuses to start when the index manifest's embedding model, dimension, or chunker version differs from its compiled configuration, and reports the rebuild command.
- Errors and recovery:
  - Unknown paper_id, or an invalid `search_type`, returns an MCP tool error naming the input, not a protocol error.
  - Indexer failures on a single paper are logged with the paper_id and skipped; the run continues and exits nonzero with a summary if any paper failed.
  - Indexer runs are resumable: a crash mid-run leaves completed papers indexed and the next run picks up the rest.
- Security and privacy behavior:
  - The indexer and server never write to paperstore.
  - HTTP authentication matches `cpp-mcp` (`packages/cpp-mcp/src/cpp_mcp/server.py:79-93` `_load_keys`, `:130-160` `_BearerKeyMiddleware`): clients send `Authorization: Bearer <key>`; keys load from `--keys-file` or `$CPP_MCP_KEYS_FILE` (same file, `/etc/cpp-mcp/keys`, so one key works for both servers), one key per line with `#` comments; keys reload on SIGHUP; the HTTP server refuses to start without at least one key unless `--no-auth` is passed.
  - Paper text is untrusted third-party content and is returned as data; the server does not interpret it.
- Acceptance criteria:
  - Each tool above works over Streamable HTTP against the fixture corpus.
  - Filters are applied before ranking (pre-filter) for all three search types, so a filtered search returns up to `limit` hits when that many matching chunks exist.
  - `get_paper_lines` matches Python `lines[start:end]` for a table of cases covering omitted, negative, out-of-range, and reversed bounds.
  - HTTP requests with a missing or unknown bearer key are rejected; a key added to the keys file is accepted after SIGHUP.
  - `paper-mcp` and `cpp-mcp` run concurrently on one host, each on its own port, and one key from the shared keys file authenticates against both.

</product-contract>
<implementation-contract>

## Technical Design

One new crate, `crates/paper-mcp`, holds a library (chunking, embedding, indexing, search, tool handlers) and a binary (operator CLI and MCP server). LanceDB provides the embedded store with BM25 full-text search, vector search, hybrid RRF, and SQL-style pre-filters; fastembed provides local ONNX embeddings; rmcp provides a tools-only MCP server over Streamable HTTP. The existing `paperstore` crate gains a listing API.

- Architecture:
  - `paperstore` / `paperstore-sqlite` (existing) supply metadata and Markdown.
  - `paper-mcp` (new): `src/lib.rs` modules for chunking, embedding, index, search, and the rmcp tool router; `src/main.rs` is a clap CLI with `index` and `serve --host --port --keys-file [--no-auth]`.
  - HTTP serving uses rmcp `StreamableHttpService` mounted on axum at `/mcp`, with bearer-key middleware as a Tower layer.
- Modules and interfaces:
  - `Embedder` trait: `model_id()`, `dim()`, `embed_documents(&[String])`, `embed_query(&str)`. Production wraps fastembed; tests use a deterministic hash embedder.
  - Chunking: split tomd Markdown on headings into sections; within a section, pack paragraphs to a 512-token target with a 1024-token hard maximum; never split inside a fenced code block or table unless it alone exceeds the maximum, then split on line boundaries. Each chunk records `section_path`, `start_line`/`end_line` (0-based, end exclusive, matching `get_paper_lines`), and a `chunk_kind` in {`body`, `wording`, `code`, `revision_history`, `front_matter`} assigned by heading and content rules.
  - Contextual prefix: each chunk's indexed text (used for both embedding and BM25) is prefixed with a deterministic header of paper_id, title, authors, and section path, built from paperstore metadata. No LLM-generated context. The stored display text omits the prefix.
  - Search: LanceDB query per `search_type` (FTS only, vector only, or hybrid with RRF) with an `only_if` filter built from validated filter inputs, never string-concatenated user text.
- File and public API changes:
  - `crates/paperstore`: add a `StorageBackend` method listing all `PaperRow`s, implemented in `crates/paperstore-sqlite`, and extend `PaperNum::parse` (`crates/paperstore/src/num.rs`) to accept `SD-<n>` IDs, which it rejects today although paperstore holds 4 SD rows (SD-1, SD-2, SD-3, SD-5). Both changes are additive.
  - `crates/paper-mcp/` (new).
  - A setup doc in `crates/paper-mcp/docs/` describing how to run `paper-mcp` on the `cpp-mcp` host (`packages/cpp-mcp/docs/server-setup.md`) without conflicting with it: its own `--host`/`--port` and the shared keys file.
  - `.github/workflows/tests.yml`: add `cargo build`/`cargo test --locked` for `paper-mcp` and `paperstore*` alongside the existing paperweight job.
  - `justfile`: recipes for index and serve.
- Data, persistence, failure, security, and privacy constraints:
  - Index lives at `$WG21_DATA_DIR/paperindex/` with a LanceDB `chunks` table (chunk id, paper_id, title (copied onto each row so search hits carry it without a paperstore lookup), revisionless_id (paper_id with any `R<n>` suffix removed, internal grouping key), revision number, is_latest, has_text, year, mailing_date, target_group, authors, intent, chunk_kind, section_path, start_line, end_line, text, indexed_text, vector) and a `papers` manifest table (paper_id, content hash, chunker version, model id, indexed_at).
  - Content hash is SHA-256 of the Markdown bytes; a paper is re-embedded when its hash, the chunker version, or the model id changes. `is_latest` is recomputed across each paper's revisions after every run.
  - An FTS index on `indexed_text` and scalar indexes on filter columns are rebuilt at the end of each indexing run.
  - The index is derived data and may be deleted and rebuilt at any time. On the server host the operator builds it from that host's `$WG21_DATA_DIR`.

</implementation-contract>
<verification-contract>

## Testing Plan

Verification runs against a checked-in fixture corpus and a deterministic embedder, so most tests need no network. One opt-in test downloads the real embedding model and runs it on a named sample of at most 20 real papers. The MCP surface is tested end to end through rmcp's client over Streamable HTTP, including bearer-key auth.

- Unit:
  - Chunker: heading splits, paragraph packing limits, code fences and tables kept whole, oversized code split on lines, `chunk_kind` assignment, line ranges, contextual prefix content.
  - Paper IDs: `SD-<n>` parsing alongside existing `PaperNum` cases.
  - Revision logic: revision grouping, `is_latest` computation, `list_revisions` input with and without `R<n>`, and revision scope.
  - `get_paper_lines` slice resolution against a table of Python-equivalent cases.
  - Filter construction: each filter maps to the expected predicate; hostile strings (quotes, SQL fragments) are escaped or rejected.
  - Keys file parsing: comments, blank lines, reload.
- Integration and end-to-end:
  - Fixture corpus under `crates/paper-mcp/tests/fixtures/`: a small `paperstore.db` plus Markdown for at least three revisions of one paper, one N paper, one PDF-derived paper, and one metadata-only paper with `status < 0`.
  - Index the fixture, then for each `search_type` assert exact paper ID filtering, the `latest` default and `all` scope without a paper filter, and metadata-only exclusion; assert incremental reindex re-embeds only a touched paper.
  - Start `paper-mcp serve`; call every tool with an rmcp client and validate result schemas; confirm a hit's `start_line`/`end_line` passed to `get_paper_lines` returns that chunk's text.
  - HTTP auth: no key and wrong key rejected, valid key accepted, newly added key accepted after SIGHUP, startup without keys fails unless `--no-auth`.
  - Opt-in real-model test (enabled by an environment variable): download `nomic-embed-text-v1.5`, index P1048R0 and P1048R1 from `$WG21_DATA_DIR`, and assert a scoped-enumeration type trait query with `latest` scope returns P1048R1 hits for each search type (operator reduced the sample from P2300R0-R10 to two small revisions for speed).
- Regression, security, and performance:
  - Server refuses to start on manifest model or chunker mismatch.
- Operational limits:
  - Target hardware: the Ubuntu development VM, CPU only.
  - Downloads: the embedding model (about 0.5 GB) is permitted; nothing substantially larger may be downloaded. No paper data is downloaded.
  - Data: fixtures plus the 2-paper P1048 sample (P1048R0, P1048R1). No command processes the full corpus.
  - Longest command: a clean `cargo build` of the LanceDB dependency tree may exceed 10 minutes; no other command is expected to exceed 10 minutes.
  - Disk: index and model for the sample stay under 2 GB.
- Exit criteria:
  - `cargo clippy --all-targets` clean under workspace lints and `cargo test --locked` passing for `paperstore`, `paperstore-sqlite`, and `paper-mcp`, with the real-model test run once and passing.
  - All Functional Specification acceptance criteria pass.

</verification-contract>
<decision-record>

## Decision Record

- Decisions:
  - MCP SDK is `rmcp` 3.5.1 (official SDK, dominant adoption, Streamable HTTP via `transport-streamable-http-server` mountable on axum), as a tools-only, Streamable HTTP-only server. User: "MCP surface noresources, no propts"; "nostdio neededm juyst http".
  - `search_papers` takes a required `search_type` of `hybrid`, `vector`, or `keyword`. User: "make search take a type that rqeuired, hybrid,vector or keyword".
  - `get_paper_lines` with optional `start`/`end` and Python slice semantics. User: "add get_paper_lines with start and end params, leave them out to go from start or to end it should work liek Python slicing".
  - All paper data comes from paperstore; no web fetches of paper data. User: "this should use data from paperstore and not grab ANY paper data from the web".
  - Deployed alongside `cpp-mcp` with the same auth (shared keys file, bearer header, SIGHUP reload). User: "it should be deployed alongside the exsintg python paper mcp server using the same auth".
  - Contextual retrieval uses only a deterministic prefix of title, authors, section path, and paper_id. User: "use contextual retrieval but NO LLM generated contexts, jsut add a simple pre3fix with the title, author, section path, id".
  - Vector store is LanceDB 0.40.0, embedded: keyword, vector, and hybrid queries with pre-filtering, no server process.
  - Embeddings via fastembed 7.1.1 with `nomic-embed-text-v1.5` (768 dimensions, 8192-token context so 1024-token chunks are not truncated, Apache-2.0, about 0.5 GB, within the user's download limit). It uses the `search_document:` / `search_query:` task prefixes.
  - Index source is tomd Markdown, so chunking handles one normalized format instead of raw HTML, PDF, and bikeshed sources.
  - Every revision is indexed. A search filtered to paper IDs returns only those exact papers; a corpus-wide search defaults to latest revisions, with `revision_scope: all` to include older ones. User: "if its scoped to P2300R10....ti ONLY reutrns antyhgin fromtha tpaper"; on the latest default with `all` opt-in: "this it should work illike this".
  - `list_revisions` is the only tool accepting an ID without `R<n>`; every other tool requires an exact paper ID. User: "list revisions can work without the r at the end, so P2300 is valid to list reevisions. but NOT anythign else that needs an atual paper id".
  - No `get_paper` tool: `get_paper_lines` with omitted bounds returns the whole paper.
  - Revision lineage uses metadata (revision number, `previous_version`) instead of a version graph.
  - MCP tools return structured JSON content with declared output schemas (rmcp structured output), so result shapes are a versioned client contract and "schema-valid results" is checkable. Added during decomposition; changing shapes later breaks clients.
- Rejected alternatives:
  - Qdrant (server): adds an external service; revisit if the index must be shared across hosts.
  - Qdrant Edge: beta API; revisit when stable.
  - tantivy plus a separate ANN crate: two indexes and hand-rolled fusion and filtering; revisit only if LanceDB proves unworkable.
  - Repo's MiniLM-L12-v2 embedder (`SERVICES.toml` `[embedders.minilm]`): trained on short inputs, unsuitable for 512 to 1024-token chunks.
  - `rust-mcp-sdk`: about 1% of rmcp's adoption; no feature gap.
  - VersionRAG-style version graph: revisit if revision-scoped filtering fails version-sensitive queries in practice.
  - LLM-generated chunk contexts: excluded by the user.
  - Reusing `study/section-chunks` code: Python study scripts tuned for classification routing, not retrieval.
- Assumptions, risks, and notes:
  - "The existing Python paper MCP server" is taken to be `cpp-mcp` (`packages/cpp-mcp`), the only MCP server in the repository; it serves the C++ standard, runs as systemd `cpp-mcp.service` on 127.0.0.1:8001 behind Caddy at `https://mcpserver1.cpp.al/mcp` (`packages/cpp-mcp/docs/server-setup.md`).
  - `ort` (under fastembed) is at 2.0.0-rc.14 with no stable 2.x; pin exactly. Fallback is fastembed's candle backend.
  - LanceDB pulls Arrow and DataFusion, increasing build time and binary size.
  - Paper text may contain prompt-injection content; clients must treat returned text as data.
  - CI's OS matrix for the Rust job must build LanceDB and ONNX Runtime.

### Deferred and Out of Scope

- Deferred: embedding model selection by experiment (candidates nomic-embed-text-v1.5, bge-base-en-v1.5, Qwen3-Embedding-0.6B); revisit when an operator-supplied retrieval evaluation set exists.
- Deferred: cross-encoder reranking; revisit if hybrid RRF precision is insufficient.
- Deferred: full-corpus indexing on the server host is an operator-run follow-up, not build work.
- Out of scope: answer generation, scraping, conversion, any web retrieval of paper data, MCP resources and prompts.

</decision-record>
<project-survey>

## Project Survey

- Status: complete
- Build command: `cargo build --workspace --locked` (CI builds `cargo build --release -p paperweight --locked`; requires the `vendor/promptforge` submodule checked out, `git submodule update --init --recursive`). Python side: `uv sync --all-packages --group dev`. Note: `cargo` is not on PATH in this VM; `rust-toolchain.toml` pins 1.98.0 with clippy and rustfmt.
- Focused test command pattern: `cargo test -p <crate> --locked <test_name_filter>`; Python: `uv run pytest packages/<pkg>/tests/<file>.py::<test>`.
- Component test command pattern: `cargo test -p <crate> --locked`; Python: `uv run pytest packages/<pkg>/tests` (or `uv run --package <pkg> pytest`).
- Full-suite test command: `cargo test --workspace --locked` plus `uv run pytest` (pytest `addopts` deselects `network`-marked tests).
- Linter command: `cargo clippy --workspace --all-targets --locked` (workspace lints in root `Cargo.toml`: `clippy::all`, `unwrap_used`, `expect_used` deny; `pedantic`, `missing_docs` warn; `unsafe_code` forbid). Python: `uv run ruff check packages tests` and `uv run pyright packages` (CI, pyright non-blocking). No clippy step in CI.
- Formatter check command: `cargo fmt --all --check` (`rustfmt.toml`: `style_edition = "2024"`; no CI step). Python: None (ruff format not configured or run).
- Docs command: None as a recipe or CI step; `[workspace.lints.rustdoc]` denies broken and private intra-doc links, which `cargo doc --workspace --no-deps` enforces.
- Test placement and naming conventions: Rust unit tests inline as `#[cfg(test)] mod tests { use super::*; ... }` at the bottom of the source file (`crates/paperstore/src/{row,num,error,workspace}.rs`, `crates/paperweight/src/app.rs`); integration tests as a single `tests/it/main.rs` target with a `//!` doc header (`crates/paperstore-sqlite/tests/it/main.rs`), using `tempfile::TempDir` and `temp-env` for `$WG21_DATA_DIR`. Python tests in `packages/<pkg>/tests/test_<topic>.py` (often `test_<pkg>_<topic>.py`), fixtures in `tests/fixtures/` and `conftest.py`; cross-package integration in top-level `tests/`; live-network tests marked `@pytest.mark.network`.
- Directory map:
  - `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `rustfmt.toml`: Rust workspace (`members = ["crates/*"]`, excludes `vendor/promptforge`).
  - `crates/paperstore`: Rust abstract storage API (error, `PaperNum`, row, `StorageBackend` trait, workspace resolution).
  - `crates/paperstore-sqlite`: `SqliteBackend` over rusqlite (bundled), schema and migration.
  - `crates/paperweight`: CLI binary running a promptforge prompt against a paper (clap, tokio, reqwest rustls); plus `promptforge-cli` bin.
  - `vendor/promptforge`: git submodule, separate Cargo workspace (provides `harness`).
  - `packages/`: uv workspace Python packages: `paperstore`, `mailing`, `tomd`, `pipeline`, `agora`, `assay`, `cli`, `preview`, `cpp-mcp` (FastMCP server), `whisker` (tomd benchmarking).
  - `tests/`: Python cross-package integration tests.
  - `data/`: local workspace data (`paperstore.db`, `paperstore/`).
  - `study/`, `research/`, `herald/`, `html/`, `vibe/`: experiments, research notes, design docs, plans.
  - `SERVICES.toml`: LLM service and classifier inventory. `justfile`: command hub mounting `packages/tomd/justfile`.
  - `.github/workflows/tests.yml`: ruff, pyright, per-package pytest matrix (Ubuntu and Windows), paperweight cargo build and test, integration tests.
- Component boundaries:
  - Rust: `paperstore` (no internal deps) <- `paperstore-sqlite` <- `paperweight`; `paperweight` also depends on `vendor/promptforge` `harness`.
  - Python: `paperstore` is the base; `mailing`, `tomd`, `preview` depend on it; `pipeline` is standalone; `agora`, `assay` depend on `paperstore` and `pipeline`; `whisker` depends on `paperstore` and `tomd`; `cli` depends on `paperstore`, `mailing`, `tomd`, `pipeline`, `agora`, `assay`; `cpp-mcp` depends only on `fastmcp`.
  - Rust and Python share the on-disk paperstore (`$WG21_DATA_DIR/paperstore.db`, `paperstore/`), with Rust reading the schema Python creates.
- Conventions summary: BSL-1.0 headers on new files; Rust edition 2024, crates inherit `[workspace.package]` and `[lints] workspace = true`, each crate pins its own dependency versions (no `[workspace.dependencies]`), `//!` crate and module docs and `///` on public items, `thiserror` errors in libraries and `anyhow` in binaries, no `unwrap`/`expect`, no unsafe. Python 3.12, src layout with hatchling, storage only through `StorageBackend`, `logging` not `print`, no lazy imports, named module-level constants for thresholds, `__init__.py` re-exports only, Google-style docstrings. Per-package rules in `packages/<name>/src/<name>/CLAUDE.md`. No em dashes; spellings `paperflow`, `tomd`, `WG21`.

</project-survey>
<execution-plan>

## Execution Instructions

Components, in dependency order:

1. `paperstore-listing`: the additive `paperstore` and `paperstore-sqlite` API. It comes first because every `paper-mcp` component reads paperstore through it, and it is useful on its own to any Rust consumer of paperstore.
2. `paper-mcp-index`: crate scaffold, fixture corpus, embedding, chunking, revision logic, and incremental LanceDB indexing. It depends on the listing API and produces the index that search reads. Pieces: foundations (scaffold, embedder, fixtures) are built first because chunking and indexing tests need them; chunking and revision logic are independent and built jointly; indexing comes last because it composes both.
3. `paper-mcp-search`: filter construction and the three search types. It depends on a built index. Pieces are sequential: filter predicates first, then queries that consume them.
4. `paper-mcp-server`: MCP tool handlers, bearer auth, and the CLI and HTTP server. It depends on search and paperstore reads. Tool handlers and auth are independent and built jointly; the CLI and HTTP server come last because they compose both.
5. `paper-mcp-ops`: setup doc, justfile, CI, coexistence with `cpp-mcp`, the opt-in real-model test, and exit checks. It depends on the final CLI flags and a complete binary.

Each step's tests pass before the next step starts. Run Rust commands through the toolchain pinned in `rust-toolchain.toml` (`cargo` is not on PATH in this VM, so install or locate the pinned toolchain first). Every new file carries a BSL-1.0 header; no em dashes anywhere.

<step-1>

### Step 1: Accept SD paper IDs [completed]

- Component: paperstore-listing
- Extend `PaperNum::parse` in `crates/paperstore/src/num.rs` to accept `SD-<n>` (for example `SD-1`, `SD-5`) alongside existing P and N forms, with `as_str` and `stem` behaving consistently for SD IDs.
- Tests: inline `#[cfg(test)] mod tests` cases in `num.rs` for valid and invalid SD IDs plus the existing `PaperNum` cases unchanged.
- Commit: `paperstore: accept SD-<n> paper IDs`.

</step-1>

<step-2>

### Step 2: List all paper rows [completed]

- Component: paperstore-listing
- Add a `StorageBackend` method in `crates/paperstore/src/backend.rs` (for example `fn list_papers(&self) -> Result<Vec<PaperRow>>`) returning every `PaperRow` in a stable order (by paper_id), and implement it in `crates/paperstore-sqlite/src/backend.rs`. Ensure `PaperRow` (`crates/paperstore/src/row.rs`) exposes every column the indexer needs: paper_id, year, title, authors, target_group, intent, url, document_date, mailing_date, disposition, previous_version, markdown_path, line_count, status. The change is additive and read-only.
- Tests: `crates/paperstore-sqlite/tests/it/main.rs` cases listing an empty store, a mixed P/N/SD store, and rows with `status < 0` or no Markdown.
- Commit: `paperstore: add list_papers to StorageBackend`.

</step-2>

<step-3>

### Step 3: Scaffold paper-mcp with the Embedder trait [completed]

- Component: paper-mcp-index
- Create `crates/paper-mcp/` (`Cargo.toml` inheriting `[workspace.package]` and `[lints] workspace = true`, pinning `lancedb = "=0.40.0"`, `fastembed = "=7.1.1"` with `ort` pinned exactly at `2.0.0-rc.14`, `rmcp = "=3.5.1"` with `transport-streamable-http-server`, axum, tokio, clap, thiserror, anyhow, sha2, `tempfile` and `temp-env` as dev dependencies; `Cargo.lock` updated).
- `src/lib.rs` with `//!` docs declaring modules `embed`, `chunk`, `revision`, `index`, `search`, `filter`, `lines`, `auth`, `tools`, and a `src/main.rs` stub.
- `src/embed.rs`: `Embedder` trait (`model_id()`, `dim()`, `embed_documents(&[String])`, `embed_query(&str)`); `FastEmbedder` wrapping fastembed `nomic-embed-text-v1.5` (768 dims) applying `search_document:` and `search_query:` prefixes; `HashEmbedder`, a deterministic test embedder with configurable dim and model id; a `thiserror` `EmbedError`.
- Tests: inline unit tests that `HashEmbedder` is deterministic, has the declared dim, and distinguishes different inputs; a unit test that `FastEmbedder` applies the task prefixes (prefix construction tested without downloading the model).
- Commit: `paper-mcp: scaffold crate and Embedder trait`.

</step-3>

<step-4>

### Step 4: Build the fixture corpus [completed]

- Component: paper-mcp-index
- Add `crates/paper-mcp/tests/fixtures/paperstore.db` and `crates/paper-mcp/tests/fixtures/paperstore/<pid>.md` using the schema Python paperstore creates (`packages/paperstore/src/paperstore/sqlite_backend.py`). Contents: at least three revisions of one P paper (with `previous_version` lineage), one N paper, one SD paper, one PDF-derived paper, and one metadata-only paper with `status < 0`. Markdown files are short tomd output with fixed-order front matter and include headings, a fenced code block, a table, a wording section, and a revision history section. Add `crates/paper-mcp/tests/fixtures/README.md` describing the contents and how the DB was produced.
- Add `crates/paper-mcp/tests/it/main.rs` (single integration target with a `//!` header) with a helper that copies the fixture into a `TempDir` and sets `$WG21_DATA_DIR` via `temp-env`.
- Tests: the fixture opens through `SqliteBackend`, `list_papers` returns the expected rows, and `paper_md` reads each text-bearing paper.
- Commit: `paper-mcp: add fixture corpus`.

</step-4>

<step-5>

### Step 5: Chunk tomd Markdown with contextual prefixes [completed]

- Component: paper-mcp-index
- `src/chunk.rs`: `CHUNKER_VERSION` constant, `TARGET_TOKENS = 512` and `MAX_TOKENS = 1024` constants, `ChunkKind` enum {`body`, `wording`, `code`, `revision_history`, `front_matter`}, `Chunk` struct (section_path, start_line, end_line 0-based end exclusive, chunk_kind, text, indexed_text), and `chunk_markdown(md: &str, meta: &PrefixMeta) -> Vec<Chunk>`. Split on headings into sections, pack paragraphs to the target, never split fenced code or tables unless one alone exceeds the maximum (then split on line boundaries), and assign `chunk_kind` by heading and content rules.
- `build_prefix(meta)`: deterministic header of paper_id, title, authors, and section path; `indexed_text` = prefix + text; `text` omits the prefix. No LLM involvement.
- Tests: inline unit tests for heading splits, packing limits, whole code fences and tables, oversized code split on lines, each `chunk_kind`, line ranges equal to the chunk's slice of the Markdown lines, and prefix content.
- Commit: `paper-mcp: chunk Markdown with contextual prefixes`.

</step-5>

<step-6>

### Step 6: Revision grouping and latest computation [completed]

- Component: paper-mcp-index
- `src/revision.rs`: `revisionless_id(&PaperNum) -> String`, `revision_number(&PaperNum) -> Option<u32>` (None for N and SD), `latest_set(&[PaperRow]) -> BTreeSet<String>` computing `is_latest` per revision group (N and SD always latest), `RevisionScope` enum {`Latest` (default), `All`}, and `parse_revision_query(&str)` accepting IDs with or without `R<n>` for `list_revisions` only.
- Tests: inline unit tests for grouping, `is_latest` (including R10 vs R9 numeric ordering), N/SD handling, `list_revisions` input with and without `R<n>`, and scope defaults.
- Commit: `paper-mcp: add revision grouping and latest computation`.

</step-6>

<step-7>

### Step 7: Incremental LanceDB indexing [completed]

- Component: paper-mcp-index
- `src/index.rs`: `Indexer` over a `StorageBackend` and an `Embedder`, writing `$WG21_DATA_DIR/paperindex/` (path resolved from the paperstore workspace, never written into paperstore) with a `chunks` table (chunk id, paper_id, revisionless_id, revision number, is_latest, has_text, year, mailing_date, target_group, authors, intent, chunk_kind, section_path, start_line, end_line, text, indexed_text, vector) and a `papers` manifest table (paper_id, content hash, chunker version, model id, indexed_at), plus an index-level manifest recording model id, dim, and chunker version.
- Behavior: SHA-256 content hash; re-embed only when hash, chunker version, or model id changes; metadata-only rows (`status < 0` or no Markdown) stored with `has_text = false`; per-paper writes committed before the manifest row so a crash leaves completed papers indexed and the next run resumes; per-paper failures logged with paper_id, skipped, and summarized in an `IndexReport` that makes the CLI exit nonzero; `is_latest` recomputed via `revision::latest_set` after every run; FTS index on `indexed_text` and scalar indexes on filter columns rebuilt at the end. `check_manifest(embedder)` returns a mismatch error naming the rebuild command.
- Tests (in `tests/it/`, using `HashEmbedder` and the fixture): full index produces expected chunk and paper counts; metadata-only paper has `has_text = false` and no searchable chunks; reindex after touching one Markdown file re-embeds only that paper (counting embed calls); changing chunker version or model id re-embeds all; an interrupted run resumes; `check_manifest` rejects model, dim, and chunker mismatches; paperstore files are unmodified after indexing.
- Commit: `paper-mcp: incremental LanceDB indexer`.

</step-7>

<step-8>

### Step 8: Validated search filters [completed]

- Component: paper-mcp-search
- `src/filter.rs`: `SearchFilters` struct (paper_ids, revision_scope, year range, mailing, target_group, author substring, intent, chunk_kind), validated from tool input (paper IDs through `PaperNum::parse`, enums parsed, numbers typed), and `to_predicate(&self) -> String` producing the LanceDB `only_if` SQL predicate. String values are escaped (quote doubling, `LIKE` wildcard escaping) or rejected; no user text is concatenated unescaped. Revision scope adds `is_latest = true` only when no paper ID filter is present, and `has_text = true` is always applied for search.
- Tests: inline unit tests mapping each filter to its expected predicate, combinations, scope behavior with and without paper IDs, and hostile strings (quotes, `--`, `; DROP`, `%`, `_`) escaped or rejected.
- Commit: `paper-mcp: validated search filter predicates`.

</step-8>

<step-9>

### Step 9: Keyword, vector, and hybrid search [completed]

- Component: paper-mcp-search
- `src/search.rs`: `SearchType` enum {`hybrid`, `vector`, `keyword`}, `SearchRequest` (query, search_type, filters, limit with a named default and maximum constant), `SearchHit` (paper_id, revision number, title, authors, section_path, chunk_kind, start_line, end_line, text, score, url `https://wg21.link/<pid>` computed), and `search(index, embedder, &SearchRequest)` issuing a LanceDB FTS, vector, or hybrid RRF query with the filter predicate applied as a pre-filter. Results are ordered deterministically (score, then paper_id, then start_line).
- Tests (in `tests/it/`, fixture plus `HashEmbedder`), for each search type: a paper ID filter (for example the fixture's newest revision) returns only that paper; no paper filter returns only latest revisions by default; `all` scope includes older revisions; metadata-only papers never appear; a filtered search returns `limit` hits when that many matching chunks exist (pre-filter, not post-filter); hit URL format.
- Commit: `paper-mcp: keyword, vector, and hybrid search`.

</step-9>

<step-10>

### Step 10: MCP tool handlers [completed]

- Component: paper-mcp-server
- `src/lines.rs`: `resolve_slice(total: usize, start: Option<i64>, end: Option<i64>) -> (usize, usize)` with Python slice semantics (0-based, end exclusive, negatives from the end, clamping, reversed bounds yield empty).
- `src/tools.rs`: rmcp tool router `PaperTools` (holding the backend, index, and embedder) exposing exactly `search_papers`, `get_paper_metadata`, `list_revisions`, `list_papers` (paginated, the search filters minus `query` and `search_type`), and `get_paper_lines` (returns resolved `start`, `end`, `total_lines`, text over `<pid>.md` including front matter), with parameter names as in the Functional Specification and structured JSON output with declared output schemas. Unknown paper_id, IDs without `R<n>` outside `list_revisions`, invalid `search_type`, and `get_paper_lines` on a metadata-only paper return MCP tool errors naming the input. No resources or prompts are registered.
- Tests: inline table tests for `resolve_slice` matching Python `lines[start:end]` for omitted, negative, out-of-range, and reversed bounds; `tests/it/` tests calling each handler directly against the indexed fixture for results and tool-error cases.
- Commit: `paper-mcp: MCP tool handlers`.

</step-10>

<step-11>

### Step 11: cpp-mcp-compatible bearer auth [completed]

- Component: paper-mcp-server
- `src/auth.rs`: `load_keys(path) -> Result<BTreeSet<String>>` matching `packages/cpp-mcp/src/cpp_mcp/server.py` `_load_keys` (one key per line, `#` comments, blank lines ignored); `KeyStore` (shared, swappable key set) with `reload()`; a Tower layer `BearerKeyLayer` rejecting requests whose `Authorization: Bearer <key>` is missing or unknown with HTTP 401 Unauthorized (the Functional Specification requires rejection only; cpp-mcp's `_BearerKeyMiddleware` returns a JSON-RPC error inside HTTP 200 and skips notifications, which this layer deliberately does not copy); keys file resolved from `--keys-file`, else `$CPP_MCP_KEYS_FILE`, else `/etc/cpp-mcp/keys`.
- Tests: inline unit tests for keys-file parsing (comments, blank lines, whitespace) and reload picking up an added key; a Tower service test for missing, wrong, and valid keys.
- Commit: `paper-mcp: bearer key auth compatible with cpp-mcp`.

</step-11>

<step-12>

### Step 12: CLI and Streamable HTTP server [completed]

- Component: paper-mcp-server
- `src/main.rs`: clap CLI with `index` (runs `Indexer` with `FastEmbedder`, prints the `IndexReport`, exits nonzero on any per-paper failure) and `serve --host --port --keys-file [--no-auth]`. `serve` checks the index manifest and refuses to start on model, dim, or chunker mismatch with the rebuild command; refuses to start without at least one key unless `--no-auth`; mounts rmcp `StreamableHttpService` on axum at `/mcp` behind `BearerKeyLayer`; reloads keys on SIGHUP. Uses `anyhow` and `logging` via `tracing` or `log`, writing diagnostics to stderr only. A hidden test-only embedder selection (for example a `PAPER_MCP_TEST_EMBEDDER=hash` environment variable) lets end-to-end tests run without the model download.
- Tests (in `tests/it/`, spawning the built binary against the fixture index): an rmcp client over Streamable HTTP lists exactly the five tools and calls each, validating results against the declared schemas; a search hit's `start_line`/`end_line` passed to `get_paper_lines` returns that hit's text; no key and wrong key rejected, valid key accepted, a key appended to the keys file accepted after SIGHUP; startup without keys fails and succeeds with `--no-auth`; startup fails on manifest model or chunker mismatch.
- Commit: `paper-mcp: CLI and Streamable HTTP server`.

</step-12>

<step-13>

### Step 13: Deployment doc, justfile, CI, and cpp-mcp coexistence [completed]

- Component: paper-mcp-ops
- `crates/paper-mcp/docs/server-setup.md`: running `paper-mcp` on the `cpp-mcp` host per `packages/cpp-mcp/docs/server-setup.md`: its own systemd unit and `--host`/`--port` distinct from 127.0.0.1:8001, the shared `/etc/cpp-mcp/keys` file and SIGHUP reload, a Caddy route, building the index from that host's `$WG21_DATA_DIR`, and the rebuild command on manifest mismatch.
- `justfile`: `paper-mcp-index` and `paper-mcp-serve` recipes.
- `.github/workflows/tests.yml`: Rust job steps `cargo build -p paper-mcp --locked` and `cargo test --locked -p paperstore -p paperstore-sqlite -p paper-mcp` alongside the paperweight job, on an OS matrix that builds LanceDB and ONNX Runtime; the real-model test stays disabled in CI.
- Coexistence test: top-level `tests/test_paper_mcp_cpp_mcp_coexist.py` that starts `cpp-mcp` and the built `paper-mcp` binary (hash embedder, fixture index) on two ports with one shared keys file and asserts one key authenticates against both and an unknown key is rejected by both.
- Tests: the coexistence test passes; `cargo doc -p paper-mcp --no-deps` builds without broken intra-doc links.
- Commit: `paper-mcp: deployment doc, CI, and justfile recipes`.

</step-13>

<step-14>

### Step 14: Opt-in real-model test and exit checks [completed]

- Component: paper-mcp-ops
- `tests/it/` test gated by an environment variable (for example `PAPER_MCP_REAL_MODEL=1`) that downloads `nomic-embed-text-v1.5` (about 0.5 GB, the only permitted download), indexes exactly P1048R0 and P1048R1 from `$WG21_DATA_DIR` into a temporary index directory (no full-corpus run), and asserts a "type trait to detect scoped enumerations" query with `latest` scope returns P1048R1 hits for each search type. Index plus model stay under 2 GB.
- Run the test once and report its command and outcome in the coding return (the run ledger records it). Run `cargo fmt --all --check`, `cargo clippy --all-targets --locked` clean under workspace lints, and `cargo test --locked` for `paperstore`, `paperstore-sqlite`, and `paper-mcp`, and confirm every Functional Specification acceptance criterion maps to a passing test.
- Commit: `paper-mcp: opt-in real-model test`.

</step-14>

</execution-plan>
