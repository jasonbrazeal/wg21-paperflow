# paperflow

Paperflow fetches WG21 C++ standards papers from open-std.org, converts them to markdown, and stores the results in a local SQLite-backed workspace. It is a uv-managed monorepo of four packages that share a common storage backend.

## Commands

```bash
# Full pipeline for a year (scrape + download + convert)
paperflow 2026

# Individual stages
paperflow mailing 2026          # scrape mailing indexes (no downloads)
paperflow mailing all           # scrape all years >= 2011
paperflow download 2026         # fetch source files (PDF/HTML)
paperflow download P3642R4      # fetch a specific paper
paperflow convert 2026          # convert staged sources to markdown
paperflow full 2026             # all three stages
paperflow full all              # everything not yet done

# Idempotency: re-running any command skips already-complete work
paperflow download all          # downloads only what's not yet staged
paperflow convert all           # converts only what's not yet converted
```

### Flags

| Flag | Commands | Description |
|------|----------|-------------|
| `--force` / `-f` | mailing, download, convert, full | Redo stage even if already complete |
| `--verify` | download, full | HEAD-check staged files against Content-Length |
| `--concurrency N` | download, convert, full | Parallel workers (defaults vary) |
| `--extract-vector-images` | convert, full | Opt in to vector-figure extraction (heuristic; see Images below) |
| `--vector-whiteout-text` | convert, full | When extracting vector figures, paint over text inside each cluster |
| `--workspace-dir DIR` | all | Backend root (default: `$WG21_DATA_DIR`) |

All commands and flags are shown by running `paperflow` with no arguments.

## Install

```bash
uv sync && source .venv/bin/activate
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Outside an activated venv, prefix commands with `uv run`.

Set `WG21_DATA_DIR` to point to your workspace directory.

### Run paperweight locally

`paperweight` (in `crates/`) runs the paperweight prompt against one paper
through a [runvil](https://github.com/cppalliance/runvil) gateway.
It needs no Docker, only a gateway to talk to.

Get the binary in one of two ways:

- Download `paperweight-<tag>-x86_64-unknown-linux-gnu.tar.gz` from a paperflow
  release (x86_64 Linux only).
- Build it. Install [rustup](https://rustup.rs/); the toolchain pinned in
  `rust-toolchain.toml` installs itself on first use. runvil is vendored
  as the `vendor/runvil` submodule:

  ```bash
  git submodule update --init
  cargo build --release -p paperweight   # target/release/paperweight
  ```

Point it at a gateway and run it on a paper:

```bash
export RUNVIL_GATEWAY_URL=http://127.0.0.1:8085
export RUNVIL_GATEWAY_API_KEY=...       # the gateway's server.api_key
paperweight --file paper.md                    # or: paperweight P4003R2
```

- The gateway can be one you run yourself (the runvil release
  installer's Gateway component) or a shared one reached over an SSH tunnel,
  for example `ssh -L 8085:127.0.0.1:8085 <user>@stage.wg21.org`.
- A paper number is looked up in the paper store under `WG21_DATA_DIR`;
  `--file` reads a markdown file and needs no workspace.
- `--model <id>` picks the gateway model. Without it, paperweight uses the
  first chat model the gateway lists.
- The report goes to stdout, or to `--output <path>`. Its first line is the
  `Weight:`.
- `paperweight --help` lists the rest.

To move runvil, check out the new commit inside `vendor/runvil`,
fix the crate against it, and commit both in one PR. The pin must not be
newer than the runvil commit the deployed gateway runs.

### Run paper-mcp locally

`paper-mcp` (in `crates/`) is the MCP server that serves paper metadata, full
text, citations, and search from Qdrant. `paper-mcp ingest` copies the
paperstore into Qdrant; `paper-mcp serve` answers MCP calls from Qdrant alone.
It needs Qdrant 1.19 or later, and downloads the BGE-M3 embedding model (about
2.3 GB) into its cache directory on first use. paper-mcp is its own Cargo
workspace, so build and test it from `crates/paper-mcp`.

`compose.yaml` runs paper-mcp and Qdrant in Docker; it reads bearer tokens
from `~/.config/paper-mcp/keys`:

```bash
docker compose up --build
docker compose exec paper-mcp paper-mcp ingest P3045R9 P2728R14
```

To run it directly instead:

```bash
cargo build --release --manifest-path crates/paper-mcp/Cargo.toml   # crates/paper-mcp/target/release/paper-mcp
export PAPER_MCP_QDRANT_URL=http://localhost:6334
export PAPER_MCP_QDRANT_API_KEY=...     # when Qdrant has one
export PAPER_MCP_MODEL_CACHE_DIR=~/.cache/paper-mcp
paper-mcp ingest P3045R9 P2728R14         # or: all, or years such as 2026
export PAPER_MCP_KEYS_FILE=secrets/paper-mcp-keys   # one bearer token per line
paper-mcp serve                           # http://127.0.0.1:8003/papers/mcp
```

- Ingest reads `paperstore.db` under `WG21_DATA_DIR` read-only. It runs only
  by hand, and skips papers whose chunks are current unless given `--force`.
- Every setting can also come from a TOML file passed with `--config` or
  `PAPER_MCP_CONFIG`; environment variables override it. The settings and
  their defaults are in `crates/paper-mcp/src/config.rs`.
- SIGHUP reloads the keys file.
- `cargo test --test it -- --ignored`, run in `crates/paper-mcp`, runs the integration and
  golden tests; they need Qdrant, the model cache, and `WG21_DATA_DIR`.

## Tests

```bash
uv run pytest
```

## Packages

- **paperstore** - SQLite storage backend (`SqliteBackend`). All metadata in `paperstore.db`; source files and markdown in the `paperstore/` subdirectory.
- **mailing** - Scrapes the open-std.org mailing index and downloads paper sources.
- **tomd** - Converts paper PDFs and HTML to clean markdown.
- **cli** - Ingestion and conversion CLI (`paperflow`).

## Images

PDFs and HTML papers with embedded raster images get those images extracted to `paperstore/<pid>-fig{page}-{n}.{ext}` and referenced in the converted markdown as `![caption](file)`. PDF captions come from "Figure N: ..."-style labels near the image; HTML captions come from `<figcaption>` or the `alt` attribute. HTML papers also get a `<pid>.html-images.json` sidecar manifest that records the mailing-to-tomd handoff.

**Vector diagrams** drawn with PDF path/line operators (flowcharts, graph diagrams) can be extracted under the opt-in `--extract-vector-images` flag. The extractor clusters spatially adjacent path operators per page, rejects clusters that look like decoration (table borders, running-header rules, ins/del-coloured strokes, regions overlapping text blocks), and rasterises survivors to PNG via `page.get_pixmap`. The output is heuristic by design: each converted paper carries a trailing `<!-- tomd:vector-extraction-uncertain: ... -->` HTML comment disclosing per-paper rejection counts so a reader can see why a diagram might be missed. The opt-in default is deliberate; flip when a fresh corpus re-survey or the layout-aware path (see `packages/tomd/improvements.md` §4) justifies it.

**Out of scope:**

- **Scanned-page PDFs** whose body is one image per page. See `packages/tomd/improvements.md` §4.

Papers with more than 20 unique embedded images keep the first 20 in source order and append a `<!-- tomd:images-truncated: ... -->` HTML comment at end-of-body recording the cap.

## License

Boost Software License 1.0
