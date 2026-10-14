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

</decision-record>
<project-survey>

## Project Survey

None

</project-survey>
<execution-plan>

## Execution Instructions

- Work items:
  - Raise the workspace `rust-version` in `Cargo.toml` to 1.91.
  - Add paperstore listing methods to the trait and the SQLite backend.
  - Create the new crate with `index` and `serve` subcommands.
  - Implement selection, chunker, prefix, embedder and reranker, index store and search modules.
  - Implement the eight MCP tools on rmcp with axum and bearer middleware.
  - Build the golden query set and the evaluation harness, and select the models.
  - Add CI build and test steps for the new and paperstore crates.
- Dependencies and verification:
  - The paperstore listing methods are needed by selection, `list_revisions` and unversioned-id resolution.
  - Model selection depends on the golden query set and the evaluation harness.
  - Verify with the Testing Plan exit criteria.

</execution-plan>
