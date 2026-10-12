//! A small paperstore workspace written to a temporary directory.

use std::path::Path;

use rusqlite::Connection;
use tempfile::TempDir;

/// Markdown of p1000r1: H2 sections, a code fence hiding a heading, a stray
/// `\r`, and a trailing newline.
pub(crate) const P1000R1: &str = "---\ntitle: \"Frobnication for ranges\"\ndocument: P1000R1\n---\n\n## 1 Motivation\n\nWe propose to frobnicate views.\nThe word frobnicate appears here.\n\n## 2 Design\n\n### 2.1 Units \r of measure\n\nUnits design text about quantities.\n\n```cpp\n## not a heading\nauto x = std::views::iota(0);\n```\n\n## 3 Wording\n\nModify [range.adaptors] as follows.\n\n- item one\n- item two\n\n## 4 Acknowledgements\n\nThanks to everyone.\n";

/// Markdown of p3000r0, the renumbered continuation of p1000r1.
pub(crate) const P3000R0: &str = "## Introduction\n\nThis paper continues frobnication.\n\n## Changes\n\nNone.\n\n## Proposal\n\nAdopt it.\n\n## Wording\n\nSee above.";

/// Markdown of p2000r0.
pub(crate) const P2000R0: &str = "## Abstract\n\nExecutors and senders for asynchronous code.\n";

/// Markdown of n5000.
pub(crate) const N5000: &str = "# Minutes\n\nMeeting minutes with straw polls.\n";

struct Row<'a> {
    paper_id: &'a str,
    year: &'a str,
    title: &'a str,
    authors: &'a str,
    document_date: &'a str,
    mailing_date: &'a str,
    previous_version: &'a str,
    markdown: Option<&'a str>,
    citations_extracted_at: &'a str,
}

const ROWS: [Row<'static>; 5] = [
    Row {
        paper_id: "P1000R0",
        year: "2023",
        title: "Frobnication, first draft",
        authors: r#"["Eric Niebler"]"#,
        document_date: "2023-01-02",
        mailing_date: "2023-01",
        previous_version: "",
        markdown: None,
        citations_extracted_at: "",
    },
    Row {
        paper_id: "P1000R1",
        year: "2024",
        title: "Frobnication for ranges",
        authors: r#"["Eric Niebler", "Murat Can Çağrı"]"#,
        document_date: "2024-05-01",
        mailing_date: "2024-05",
        previous_version: "p1000r0",
        markdown: Some(P1000R1),
        citations_extracted_at: "2024-06-01T00:00:00+00:00",
    },
    Row {
        paper_id: "P3000R0",
        year: "2025",
        title: "Frobnication continued",
        authors: r#"["Barry Revzin"]"#,
        document_date: "Barry Revzin",
        mailing_date: "2025-01",
        previous_version: "p1000r1",
        markdown: Some(P3000R0),
        citations_extracted_at: "2025-02-01T00:00:00+00:00",
    },
    Row {
        paper_id: "P2000R0",
        year: "2024",
        title: "Executors",
        authors: r#"["Gašper Ažman"]"#,
        document_date: "2024-02-03",
        mailing_date: "2024-02",
        previous_version: "",
        markdown: Some(P2000R0),
        citations_extracted_at: "2024-03-01T00:00:00+00:00",
    },
    Row {
        paper_id: "N5000",
        year: "2022",
        title: "Minutes",
        authors: r#"["Nina Ranns"]"#,
        document_date: "2022-03-04",
        mailing_date: "2022-03",
        previous_version: "",
        markdown: Some(N5000),
        citations_extracted_at: "2022-04-01T00:00:00+00:00",
    },
];

const CITATIONS: [(&str, &str, i64); 3] = [
    ("P1000R1", "P2000R0", 3),
    ("P1000R1", "D0263R1", 1),
    ("P3000R0", "P1000R1", 2),
];

/// Writes the workspace: `paperstore.db` and one markdown file per paper
/// that has markdown.
pub(crate) fn workspace() -> TempDir {
    let dir = TempDir::new().unwrap_or_else(|e| panic!("temp dir: {e}"));
    write(dir.path());
    dir
}

fn write(root: &Path) {
    let store = root.join("paperstore");
    std::fs::create_dir_all(&store).unwrap_or_else(|e| panic!("mkdir: {e}"));
    let conn = Connection::open(root.join("paperstore.db")).unwrap_or_else(|e| panic!("open: {e}"));
    conn.execute_batch(
        "CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY, year TEXT DEFAULT '', title TEXT DEFAULT '',
            authors TEXT DEFAULT '', target_group TEXT DEFAULT '', intent TEXT DEFAULT '',
            url TEXT DEFAULT '', document_date TEXT DEFAULT '', mailing_date TEXT DEFAULT '',
            disposition TEXT DEFAULT '', previous_version TEXT DEFAULT '',
            markdown_path TEXT DEFAULT '', citations_extracted_at TEXT DEFAULT '');
         CREATE TABLE paper_citations (
            paper_id TEXT NOT NULL, cited_paper_id TEXT NOT NULL, count INTEGER DEFAULT 1,
            PRIMARY KEY (paper_id, cited_paper_id));",
    )
    .unwrap_or_else(|e| panic!("schema: {e}"));
    for row in ROWS {
        let markdown_path = row.markdown.map_or_else(String::new, |text| {
            let path = store.join(format!("{}.md", row.paper_id.to_lowercase()));
            std::fs::write(&path, text).unwrap_or_else(|e| panic!("write md: {e}"));
            path.display().to_string()
        });
        conn.execute(
            "INSERT INTO papers (paper_id, year, title, authors, target_group, url, document_date,
                mailing_date, previous_version, markdown_path, citations_extracted_at)
             VALUES (?1, ?2, ?3, ?4, 'LEWG Library Evolution', ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                row.paper_id,
                row.year,
                row.title,
                row.authors,
                format!("https://wg21.link/{}", row.paper_id.to_lowercase()),
                row.document_date,
                row.mailing_date,
                row.previous_version,
                markdown_path,
                row.citations_extracted_at,
            ],
        )
        .unwrap_or_else(|e| panic!("insert: {e}"));
    }
    for (paper, cited, count) in CITATIONS {
        conn.execute(
            "INSERT INTO paper_citations (paper_id, cited_paper_id, count) VALUES (?1, ?2, ?3)",
            rusqlite::params![paper, cited, count],
        )
        .unwrap_or_else(|e| panic!("insert citation: {e}"));
    }
}
