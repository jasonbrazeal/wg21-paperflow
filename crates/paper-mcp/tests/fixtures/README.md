# paper-mcp fixture corpus

A small paperstore workspace for the `paper-mcp` integration tests
(`tests/it/main.rs`). `paperstore.db` was created by the Python
`paperstore.SqliteBackend`, so it carries the full Python schema (every
table, including the migrated `assay_path` and `citations_extracted_at`
columns).

## Contents

| paper_id | Kind | Source | Markdown | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| P1000R0 | P paper, revision 0 | HTML | `p1000r0.md` | 2 | Start of the lineage |
| P1000R1 | P paper, revision 1 | HTML | `p1000r1.md` | 2 | `previous_version = p1000r0` |
| P1000R2 | P paper, revision 2 | HTML | `p1000r2.md` | 2 | `previous_version = p1000r1`, mailed in 2025 |
| N5001 | N paper | HTML | `n5001.md` | 2 | |
| SD-4 | SD paper | HTML | `sd-4.md` | 2 | |
| P2000R0 | PDF-derived P paper | PDF | `p2000r0.md` | 2 | |
| P3000R0 | Metadata only | none | none | -1 | Failed download, `error` set |

Each Markdown file is short tomd-style output: fixed-order YAML front matter
followed by H2 sections including a `## Revision History` section, a fenced
`cpp` code block, a table, and a `## Wording` section. `line_count` in the
database matches each file.

## Paths

The Python backend stores absolute `source_file` and `markdown_path` values.
The fixture stores them relative to the workspace (`paperstore/<pid>.md`) so
the corpus can be copied anywhere. The test helper copies `paperstore.db` and
`paperstore/` into a temp dir, prefixes both columns with the temp workspace
path, and sets `$WG21_DATA_DIR` to it. No source files (`.html`, `.pdf`) are
staged; only their recorded paths are.

## Regenerating `paperstore.db`

Edit the Markdown files, then run from the repository root (the Python
`paperstore` package has no dependencies):

```bash
PYTHONPATH=packages/paperstore/src python - <<'PY'
import shutil
import sqlite3
import tempfile
from pathlib import Path

from paperstore import SqliteBackend

FIXTURES = Path("crates/paper-mcp/tests/fixtures")
URL = "https://www.open-std.org/jtc1/sc22/wg21/docs/papers"
P1000_AUTHORS = ["Ada Lovelace", "Charles Babbage"]
PAPERS = {
    "2024": [
        dict(paper_id="P1000R0", title="A fixed-capacity ring buffer", authors=P1000_AUTHORS,
             subgroup="LEWG", intent="ask", url=f"{URL}/2024/p1000r0.html",
             document_date="2024-01-12", mailing_date="2024-01"),
        dict(paper_id="P1000R1", title="std::ring_span, a fixed-capacity circular view",
             authors=P1000_AUTHORS, subgroup="LEWG", intent="ask", url=f"{URL}/2024/p1000r1.html",
             document_date="2024-05-20", mailing_date="2024-05", previous_version="p1000r0"),
        dict(paper_id="N5001", title="Library Fundamentals Working Draft", authors=["Grace Hopper"],
             subgroup="All of WG21", url=f"{URL}/2024/n5001.html",
             document_date="2024-12-16", mailing_date="2024-12"),
        dict(paper_id="SD-4", title="WG21 Practices and Procedures", authors=["Herb Sutter"],
             url=f"{URL}/2024/sd-4.htm", document_date="2024-09-02", mailing_date="2024-09"),
    ],
    "2025": [
        dict(paper_id="P1000R2", title="std::ring_span, a fixed-capacity circular view",
             authors=P1000_AUTHORS, subgroup="LEWG", intent="ask", url=f"{URL}/2025/p1000r2.html",
             document_date="2025-01-13", mailing_date="2025-01", previous_version="p1000r1"),
        dict(paper_id="P2000R0", title="Hazard pointer cells", authors=["Leslie Lamport"],
             subgroup="SG1 Concurrency,LEWG Library Evolution", intent="info", url=f"{URL}/2025/p2000r0.pdf",
             document_date="2025-03-17", mailing_date="2025-03"),
        dict(paper_id="P3000R0", title="Contracts for coroutines", authors=["Barbara Liskov"],
             subgroup="EWG", url=f"{URL}/2025/p3000r0.html",
             document_date="2025-03-17", mailing_date="2025-03"),
    ],
}
SOURCES = {"N5001": ".html", "P1000R0": ".html", "P1000R1": ".html", "P1000R2": ".html",
           "P2000R0": ".pdf", "SD-4": ".html"}
FAILED = ("P3000R0", 0, "Client error '401 Unauthorized' for url "
          f"'{URL}/2025/p3000r0.html'")
STAMP = "2025-04-01T00:00:00+00:00"

with tempfile.TemporaryDirectory() as tmp:
    workspace = Path(tmp)
    backend = SqliteBackend(workspace)
    for year, papers in PAPERS.items():
        backend.upsert_year(year, papers)
    for pid, suffix in SOURCES.items():
        backend.record_source(pid, workspace / "paperstore" / f"{pid.lower()}{suffix}")
        md = (FIXTURES / "paperstore" / f"{pid.lower()}.md").read_text(encoding="utf-8")
        backend.write_paper_md(pid, md)
        backend.advance_status(pid, 0, 2)
    backend.fail_paper(*FAILED)
    backend.close()

    conn = sqlite3.connect(workspace / "paperstore.db")
    prefix = f"{workspace}/"
    with conn:
        for column in ("source_file", "markdown_path"):
            conn.execute(
                f"UPDATE papers SET {column} = substr({column}, ?) WHERE {column} != ''",
                (len(prefix) + 1,),
            )
        conn.execute("UPDATE papers SET updated_at = ? WHERE updated_at != ''", (STAMP,))
        conn.execute("UPDATE years SET added = ?", (STAMP,))
    conn.execute("VACUUM")
    conn.close()
    shutil.copy(workspace / "paperstore.db", FIXTURES / "paperstore.db")
PY
```

The script writes the Markdown through `write_paper_md` (which records
`line_count`), advances converted papers to status 2, fails P3000R0 at stage
0, then strips the temp workspace prefix from the path columns and pins the
timestamps so the file is reproducible.
