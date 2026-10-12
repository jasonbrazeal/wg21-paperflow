//! Golden snapshots of the outline, sections, and chunk boundaries of real
//! papers chosen for their shape. The largest file records only its counts
//! and first sections, to keep its snapshot small. They read the markdown from the
//! paperstore under `$WG21_DATA_DIR`, which is not in the repository.

use std::fmt::Write;
use std::path::PathBuf;

use paper_mcp::text::{ChunkSettings, Document, PaperLabel, TokenCounter, chunk_document, outline};

/// Counts whitespace-separated words, so snapshots do not depend on the
/// model's tokenizer.
struct Words;

impl TokenCounter for Words {
    fn count(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}

/// Sections listed for a paper snapshotted in part.
const PARTIAL_SECTIONS: usize = 20;

const SETTINGS: ChunkSettings = ChunkSettings {
    target_tokens: 300,
    overlap_max_tokens: 150,
    max_input_tokens: 8192,
};

fn markdown(paper: &str) -> String {
    let dir = std::env::var("WG21_DATA_DIR").unwrap_or_else(|_| panic!("set WG21_DATA_DIR"));
    let path = PathBuf::from(dir)
        .join("paperstore")
        .join(format!("{paper}.md"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn render(paper: &str, full: bool) -> String {
    let text = markdown(paper);
    let doc = Document::parse(&text);
    let sections = doc.sections();
    let mut out = String::new();
    let _ = writeln!(out, "lines: {}", doc.line_count());
    let _ = writeln!(out, "sections: {}", sections.len());
    let shown = if full {
        sections.len()
    } else {
        PARTIAL_SECTIONS
    };
    for s in sections.iter().take(shown) {
        let _ = writeln!(out, "  {} {}-{} {:?}", s.index, s.first, s.last, s.heading);
    }
    if full {
        let _ = writeln!(out, "outline:");
        for h in outline(&doc, &sections) {
            let start = if h.starts_section { "*" } else { " " };
            let _ = writeln!(
                out,
                "  {} h{} s{}{start} {:?}",
                h.line, h.level, h.section_index, h.text
            );
        }
    }
    let label = PaperLabel {
        paper_id: paper,
        title: "",
    };
    let chunks = chunk_document(&doc, &sections, SETTINGS, &Words, label);
    let _ = writeln!(out, "chunks: {}", chunks.len());
    if !full {
        return out;
    }
    for c in &chunks {
        let repeated = c
            .repeated_line
            .map(|l| format!(" repeats {l}"))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "  s{} c{} {}-{}{repeated} {:?}",
            c.section,
            c.number,
            c.first,
            c.last,
            c.heading_path_text()
        );
    }
    out
}

macro_rules! golden {
    ($name:ident, $paper:literal, $full:literal) => {
        #[test]
        #[ignore = "needs the paperstore markdown under $WG21_DATA_DIR"]
        fn $name() {
            insta::assert_snapshot!($paper, render($paper, $full));
        }
    };
}

golden!(h2_headings, "p3045r9", true);
golden!(mixed_h2_and_bold_numbered, "p4021r2", true);
golden!(hackmd_export, "p3248r5", true);
golden!(stray_carriage_returns, "n3768", true);
golden!(largest_file, "n4485", false);
golden!(indented_fences, "p2079r10", true);
