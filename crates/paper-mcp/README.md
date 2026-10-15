# paper-mcp

Index the WG21 papers held in paperstore and serve search over MCP (Model Context Protocol) Streamable HTTP. The crate builds two binaries: `paper-mcp`, with the `index` and `serve` subcommands, and `paper-mcp-eval`, which measures retrieval quality against a hand-labeled golden query set.

Both read the paperstore workspace named by `$WG21_DATA_DIR` (`paperstore.db` plus `paperstore/<pid>.md`). The index lives in the workspace's `paper-mcp` tool directory.

## Model provisioning

Models are never downloaded at runtime. Provision them out of band into a model directory and point `--model-dir` or `$PAPER_MCP_MODEL_DIR` at it. Keep the directory outside the repository.

```
<model-dir>/
  embedder/
    model.onnx
    model.onnx_data            # optional, external weights
    tokenizer.json
    config.json
    special_tokens_map.json
    tokenizer_config.json
    1_Pooling/config.json      # optional, sentence-transformers pooling; mean pooling when absent
    query_instruction.txt      # optional, text prepended to queries only
  reranker/
    model.onnx
    model.onnx_data            # optional
    tokenizer.json
    config.json
    special_tokens_map.json
    tokenizer_config.json
```

Each subdirectory is a Hugging Face style ONNX export of an open-weight model: a text embedding model under `embedder/` and a cross-encoder reranker under `reranker/`. Fetch the export with whatever tooling you use (for example `huggingface-cli download <repo> --local-dir <model-dir>/embedder`) on a machine with network access, then copy it into place. The model identity, derived from the files, is part of the index-format version, so swapping either model marks every indexed paper stale.

## Index

```
paper-mcp index P2300R10 P2996        # paper ids; an unversioned id selects its latest revision
paper-mcp index 2025                  # every paper of a mailing year
paper-mcp index all                   # the whole corpus
paper-mcp index all --force           # reindex papers that are already current
paper-mcp index all --workers 4       # chunk and embed four papers at once
```

Papers whose content hash and index-format version match the index are skipped. Only one index run may write at a time; a second run fails on the lock file. Each run ends by optimizing the index.

## Serve

```
paper-mcp serve --token-file /path/to/token
```

The bearer token comes from `--token-file` or `$PAPER_MCP_TOKEN`, and every request must carry it as `Authorization: Bearer <token>`. The MCP endpoint is `/mcp` on `--bind` (default `127.0.0.1:8080`). Behind a proxy, pass the public host with `--allowed-host`. Search limits are set with `--top-k`, `--rerank-depth`, `--page-size`, `--cursor-ttl-secs` and `--cursor-capacity`; `paper-mcp serve --help` lists their defaults. The server reads the index while an index run writes it and picks up new data without a restart.

## Evaluate

`eval/golden.jsonl` holds the golden query set, labeled against the real corpus. Each line is one query:

```json
{"id": "revision-01", "category": "revision", "query": "which revision ...", "all_revisions": true, "paper_id": "P2900", "relevant": [{"paper_id": "P2900R7", "start": 29, "end": 45}]}
```

- `category` is `prose` (a natural-language question), `identifier` (an exact C++ identifier or signature), `wording_diff` (a phrase from proposed wording) or `revision` (a question about one revision).
- `all_revisions` (default `false`) and `paper_id` (optional filter, with or without a revision) shape the search exactly as the `search_papers` tool parameters do.
- `relevant` lists the paper revisions and line ranges (1-based start, exclusive end) that answer the query. A result counts for a range when it is in the same revision and shares at least one line with it.

The set must hold at least 50 queries and every category. The harness checks each label against the corpus before searching: the paper must have Markdown, the range must fit in it, and a query that searches only latest revisions must label latest revisions.

```
paper-mcp-eval                                  # evaluate the existing index
paper-mcp-eval --build all                      # rebuild the index first and report indexing throughput
paper-mcp-eval --index-dir /tmp/pm --build all  # build and evaluate a scratch index
paper-mcp-eval --top-k 500 --rerank-depth 100 --k 20
```

It runs every query in four configurations through the same search code the server uses: `dense` (vector only), `bm25` (keyword only), `hybrid` (both, fused with RRF) and `hybrid+rerank` (what `search_papers` serves). The report gives mean recall@k overall and per category, p95 search latency per configuration, the `search_papers` p95 latency (the `hybrid+rerank` search), and, after `--build`, indexing throughput in papers and chunks per second. Queries run one at a time. To compare models or chunking settings, build each candidate into its own `--index-dir`.
