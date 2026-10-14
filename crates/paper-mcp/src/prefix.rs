// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Deterministic contextual prefix prepended to chunk text.
//!
//! The prefix stands in for LLM-written chunk context: it names the paper
//! and the section a chunk belongs to, so that both the embedding and the
//! BM25 index see that context. Text returned to clients is unprefixed.

use paperstore::PaperRow;

/// Version of the prefix layout, folded into the index-format version.
///
/// Bump it whenever a change to this module can change the prefix for any
/// row and heading path.
pub const PREFIX_VERSION: u32 = 1;

/// Separator between the joined author names.
const AUTHOR_SEPARATOR: &str = ", ";

/// Separator between the joined heading path parts.
const HEADING_SEPARATOR: &str = " > ";

/// Text between the prefix and the chunk text.
const PREFIX_BODY_SEPARATOR: &str = "\n\n";

/// Build the contextual prefix for a chunk of `row` under `heading_path`.
///
/// The prefix is one `Label: value` line per field, in this order: paper id
/// with revision (`Paper`), `Title`, `Authors`, `Target group`, document
/// `Date` and heading path (`Section`). Whitespace runs inside each value
/// collapse to one space, so every field stays on one line. Fields that are
/// empty after that are omitted, as are empty authors and heading parts.
/// Authors join with `", "` and heading parts with `" > "`. Lines join with
/// `\n` and there is no trailing newline; an all-empty input yields `""`.
#[must_use]
pub fn build_prefix(row: &PaperRow, heading_path: &[String]) -> String {
    let fields = [
        ("Paper", normalize(&row.paper_num)),
        ("Title", normalize(&row.title)),
        ("Authors", join_nonempty(&row.authors, AUTHOR_SEPARATOR)),
        ("Target group", normalize(&row.target_group)),
        ("Date", normalize(&row.document_date)),
        ("Section", join_nonempty(heading_path, HEADING_SEPARATOR)),
    ];
    fields
        .iter()
        .filter(|(_, value)| !value.is_empty())
        .map(|(label, value)| format!("{label}: {value}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Prepend `prefix` to `chunk`, separated by one blank line.
///
/// An empty `prefix` returns `chunk` unchanged.
#[must_use]
pub fn prefixed_text(prefix: &str, chunk: &str) -> String {
    if prefix.is_empty() {
        chunk.to_owned()
    } else {
        format!("{prefix}{PREFIX_BODY_SEPARATOR}{chunk}")
    }
}

/// Collapse every whitespace run in `value` to one space and trim the ends.
fn normalize(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Normalize each part, drop the empty ones and join the rest with `separator`.
fn join_nonempty(parts: &[String], separator: &str) -> String {
    parts
        .iter()
        .map(|part| normalize(part))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(separator)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row() -> PaperRow {
        PaperRow {
            paper_num: "P2169R4".to_owned(),
            title: "A nice placeholder with no name".to_owned(),
            authors: vec!["Corentin Jabot".to_owned(), "Michael Park".to_owned()],
            target_group: "EWG".to_owned(),
            document_date: "2023-06-16".to_owned(),
            ..PaperRow::default()
        }
    }

    fn path(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|p| (*p).to_owned()).collect()
    }

    #[test]
    fn prefix_has_fixed_field_order_and_layout() {
        let got = build_prefix(&row(), &path(&["Motivation", "Examples"]));
        assert_eq!(
            got,
            "Paper: P2169R4\n\
             Title: A nice placeholder with no name\n\
             Authors: Corentin Jabot, Michael Park\n\
             Target group: EWG\n\
             Date: 2023-06-16\n\
             Section: Motivation > Examples"
        );
    }

    #[test]
    fn prefix_is_byte_identical_across_calls() {
        let heading = path(&["Wording"]);
        let first = build_prefix(&row(), &heading);
        for _ in 0..8 {
            assert_eq!(build_prefix(&row().clone(), &heading.clone()), first);
        }
    }

    #[test]
    fn prefix_omits_empty_fields() {
        let sparse = PaperRow {
            paper_num: "N5034".to_owned(),
            title: "   ".to_owned(),
            authors: vec![String::new(), "Nina Ranns".to_owned(), " ".to_owned()],
            ..PaperRow::default()
        };
        assert_eq!(
            build_prefix(&sparse, &[]),
            "Paper: N5034\nAuthors: Nina Ranns"
        );
    }

    #[test]
    fn prefix_omits_empty_heading_parts() {
        let got = build_prefix(&row(), &path(&["", "Design", "  "]));
        assert!(got.ends_with("\nSection: Design"), "{got}");
    }

    #[test]
    fn prefix_collapses_whitespace_so_fields_stay_on_one_line() {
        let messy = PaperRow {
            paper_num: " P1000R1 ".to_owned(),
            title: "Line one\n  line\ttwo".to_owned(),
            ..PaperRow::default()
        };
        assert_eq!(
            build_prefix(&messy, &path(&["A\nB"])),
            "Paper: P1000R1\nTitle: Line one line two\nSection: A B"
        );
    }

    #[test]
    fn prefix_of_empty_row_and_path_is_empty() {
        assert_eq!(build_prefix(&PaperRow::default(), &[]), "");
    }

    #[test]
    fn prefixed_text_separates_prefix_with_blank_line() {
        assert_eq!(
            prefixed_text("Paper: P1", "body\ntext"),
            "Paper: P1\n\nbody\ntext"
        );
    }

    #[test]
    fn prefixed_text_with_empty_prefix_is_chunk_unchanged() {
        assert_eq!(prefixed_text("", "body"), "body");
    }
}
