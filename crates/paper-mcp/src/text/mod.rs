//! The paper text model (design §5.2 to §5.5): lines, headings, code
//! fences, sections, blocks, and retrieval chunks.
//!
//! Everything here is pure: no input or output, no model, no Qdrant. Ingest
//! and serve both build their views of a paper from [`Document`], so every
//! tool and every search hit agree on line numbers and section indexes.

mod blocks;
mod chunks;
mod headings;
#[cfg(test)]
mod proptests;
mod range;
mod sections;

use std::fmt;
use std::num::NonZeroU32;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use blocks::{Block, BlockKind};
pub use chunks::{
    Chunk, ChunkSettings, PaperLabel, TokenCounter, chunk_document, chunk_section, context_header,
};
pub use headings::{Heading, clean_heading_text};
pub use range::{InclusiveRange, resolve_range};
pub use sections::Section;

macro_rules! one_based {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
            JsonSchema,
        )]
        #[serde(transparent)]
        pub struct $name(NonZeroU32);

        impl $name {
            /// The first value, 1.
            pub const FIRST: Self = Self(NonZeroU32::MIN);

            /// Builds the value from a 1-based number, or `None` for 0 or a
            /// number that does not fit.
            #[must_use]
            pub fn new(value: usize) -> Option<Self> {
                u32::try_from(value).ok().and_then(NonZeroU32::new).map(Self)
            }

            /// Builds the value from a 0-based position, saturating at the
            /// largest representable value.
            #[must_use]
            pub fn from_zero_based(index: usize) -> Self {
                Self::new(index.saturating_add(1)).unwrap_or(Self(NonZeroU32::MAX))
            }

            /// The 1-based number.
            #[must_use]
            pub fn get(self) -> usize {
                self.0.get() as usize
            }

            /// The 0-based position.
            #[must_use]
            pub fn zero_based(self) -> usize {
                self.get() - 1
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

one_based!(
    /// A 1-based line number (§5.2).
    LineNo
);
one_based!(
    /// A 1-based section index (§5.4).
    SectionIndex
);
one_based!(
    /// A 1-based chunk number within one section (§5.5).
    ChunkNo
);

/// One paper's markdown, split into lines and analysed for code fences and
/// headings.
#[derive(Debug, Clone)]
pub struct Document<'a> {
    lines: Vec<&'a str>,
    fenced: Vec<bool>,
    fence_ends: Vec<Option<usize>>,
    headings: Vec<Heading>,
}

impl<'a> Document<'a> {
    /// Analyses `text`.
    ///
    /// Lines are the text between `\n` characters; a newline at the end of
    /// the text does not create an empty last line, and `\r` is an ordinary
    /// character.
    #[must_use]
    pub fn parse(text: &'a str) -> Self {
        let lines = split_lines(text);
        let (fenced, fence_ends) = headings::scan_fences(&lines);
        let headings = headings::find_headings(&lines, &fenced);
        Self {
            lines,
            fenced,
            fence_ends,
            headings,
        }
    }

    /// The number of lines.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// The lines, 0-based.
    #[must_use]
    pub fn lines(&self) -> &[&'a str] {
        &self.lines
    }

    /// One line by number, if it exists.
    #[must_use]
    pub fn line(&self, line: LineNo) -> Option<&'a str> {
        self.lines.get(line.zero_based()).copied()
    }

    /// Lines `first` through `last`, inclusive, joined with `\n`. Lines
    /// outside the paper are skipped.
    #[must_use]
    pub fn text(&self, first: LineNo, last: LineNo) -> String {
        let end = last.get().min(self.lines.len());
        let start = first.zero_based();
        if start >= end {
            return String::new();
        }
        self.lines[start..end].join("\n")
    }

    /// Every heading outside a code fence, in line order (§5.3).
    #[must_use]
    pub fn headings(&self) -> &[Heading] {
        &self.headings
    }

    /// Whether a 0-based line is inside a code fence, fence lines included.
    pub(crate) fn is_fenced(&self, index: usize) -> bool {
        self.fenced.get(index).copied().unwrap_or(false)
    }

    /// For a 0-based line that opens a code fence, the 0-based line that
    /// closes it, or the last line when the fence is left open.
    pub(crate) fn fence_end(&self, index: usize) -> Option<usize> {
        self.fence_ends.get(index).copied().flatten()
    }

    /// The paper's sections (§5.4). They tile the paper; a paper with no
    /// lines has none.
    #[must_use]
    pub fn sections(&self) -> Vec<Section> {
        sections::split(self)
    }
}

/// Splits text into lines by `\n` only (§5.2).
#[must_use]
pub fn split_lines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let body = text.strip_suffix('\n').unwrap_or(text);
    body.split('\n').collect()
}

/// One outline entry (§6.4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct OutlineEntry {
    /// The heading's line number.
    pub line: LineNo,
    /// 1 to 6 for markdown headings, 2 for bold-numbered lines.
    pub level: u8,
    /// The cleaned heading text.
    pub text: String,
    /// The section that contains the heading.
    pub section_index: SectionIndex,
    /// Whether this heading begins that section.
    pub starts_section: bool,
}

/// Builds the outline: every heading, with the section that holds it.
#[must_use]
pub fn outline(doc: &Document<'_>, sections: &[Section]) -> Vec<OutlineEntry> {
    doc.headings()
        .iter()
        .filter_map(|heading| {
            let section = sections
                .iter()
                .find(|s| s.first <= heading.line && heading.line <= s.last)?;
            Some(OutlineEntry {
                line: heading.line,
                level: heading.level,
                text: heading.text.clone(),
                section_index: section.index,
                starts_section: section.boundary == Some(heading.line),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_newline_does_not_add_a_line() {
        assert_eq!(split_lines("a\nb\n"), vec!["a", "b"]);
        assert_eq!(split_lines("a\nb"), vec!["a", "b"]);
        assert_eq!(split_lines("a\n\n"), vec!["a", ""]);
        assert_eq!(split_lines("\n"), vec![""]);
        assert!(split_lines("").is_empty());
    }

    #[test]
    fn carriage_returns_are_ordinary_characters() {
        let lines = split_lines("### 1.3 \r Approve \r minutes\r\nnext");
        assert_eq!(lines, vec!["### 1.3 \r Approve \r minutes\r", "next"]);
    }

    #[test]
    fn text_clamps_to_the_paper() {
        let doc = Document::parse("a\nb\nc\n");
        let line = |n| LineNo::new(n).unwrap_or(LineNo::FIRST);
        assert_eq!(doc.text(line(2), line(3)), "b\nc");
        assert_eq!(doc.text(line(2), line(9)), "b\nc");
        assert_eq!(doc.text(line(3), line(2)), "");
    }

    #[test]
    fn outline_marks_section_starts() {
        let md = "intro\n## A\ntext\n### A.1\n## B\n## C\n## D\n";
        let doc = Document::parse(md);
        let sections = doc.sections();
        let entries = outline(&doc, &sections);
        let summary: Vec<_> = entries
            .iter()
            .map(|e| (e.line.get(), e.section_index.get(), e.starts_section))
            .collect();
        assert_eq!(
            summary,
            vec![
                (2, 2, true),
                (4, 2, false),
                (5, 3, true),
                (6, 4, true),
                (7, 5, true)
            ]
        );
    }
}
