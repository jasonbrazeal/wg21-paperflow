//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Validated search filters and their translation to index predicates.
//!
//! [`SearchFilters::from_input`] validates raw tool input: paper IDs through
//! [`PaperNum::parse`], revision scope and chunk kind as enums, mailings as
//! `YYYY-MM`, and free text for length and control characters.
//! [`SearchFilters::to_predicate`] renders the result as a LanceDB `only_if`
//! SQL predicate in which every string value is a quoted literal, with
//! `LIKE` wildcards escaped in substring matches.

use std::collections::BTreeSet;

use paperstore::PaperNum;

use crate::chunk::{ChunkKind, UnknownChunkKind};
use crate::revision::{RevisionError, RevisionScope};

/// Longest free-text filter value accepted, in characters.
pub const MAX_FILTER_TEXT_CHARS: usize = 200;

/// Escape character for `LIKE` patterns built from user text.
const LIKE_ESCAPE: char = '\\';

/// Separator between the year and month of a mailing.
const MAILING_SEPARATOR: char = '-';

/// Digits in a mailing year.
const MAILING_YEAR_DIGITS: usize = 4;

/// Digits in a mailing month.
const MAILING_MONTH_DIGITS: usize = 2;

/// Valid mailing months.
const MAILING_MONTHS: std::ops::RangeInclusive<u32> = 1..=12;

/// Errors from validating search filter input.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FilterError {
    /// A paper ID that `PaperNum::parse` rejects.
    #[error("paper_ids: {0:?} is not a valid paper ID")]
    PaperId(String),
    /// A revision scope other than `latest` or `all`.
    #[error("revision_scope: {0}")]
    RevisionScope(#[from] RevisionError),
    /// A chunk kind other than the stored names.
    #[error("chunk_kind: {0}")]
    ChunkKind(#[from] UnknownChunkKind),
    /// A year range whose lower bound exceeds its upper bound.
    #[error("year_from {from} is greater than year_to {to}")]
    YearRange {
        /// Lower bound given.
        from: i32,
        /// Upper bound given.
        to: i32,
    },
    /// A mailing not in `YYYY-MM` form with a month from 01 to 12.
    #[error("mailing: {0:?} is not a mailing; expected YYYY-MM")]
    Mailing(String),
    /// A free-text value that is empty after trimming.
    #[error("{field}: value is empty")]
    EmptyText {
        /// Input field name.
        field: &'static str,
    },
    /// A free-text value longer than [`MAX_FILTER_TEXT_CHARS`].
    #[error("{field}: value exceeds {MAX_FILTER_TEXT_CHARS} characters")]
    TextTooLong {
        /// Input field name.
        field: &'static str,
    },
    /// A free-text value containing a control character.
    #[error("{field}: value contains a control character")]
    ControlCharacter {
        /// Input field name.
        field: &'static str,
    },
}

/// Raw search filter input, as received from a tool call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilterInput {
    /// Exact paper IDs. An empty list is the same as no paper ID filter.
    pub paper_ids: Option<Vec<String>>,
    /// `latest` (the default) or `all`. Ignored when paper IDs are given.
    pub revision_scope: Option<String>,
    /// Earliest paper year, inclusive.
    pub year_from: Option<i32>,
    /// Latest paper year, inclusive.
    pub year_to: Option<i32>,
    /// Mailing as `YYYY-MM`.
    pub mailing: Option<String>,
    /// Case-insensitive substring of the target group.
    pub target_group: Option<String>,
    /// Case-insensitive substring of the authors.
    pub author: Option<String>,
    /// Paper intent, matched exactly after trimming and lowercasing.
    pub intent: Option<String>,
    /// Chunk kind name.
    pub chunk_kind: Option<String>,
}

/// Validated search filters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchFilters {
    paper_ids: BTreeSet<PaperNum>,
    revision_scope: RevisionScope,
    year_from: Option<i32>,
    year_to: Option<i32>,
    mailing: Option<String>,
    target_group: Option<String>,
    author: Option<String>,
    intent: Option<String>,
    chunk_kind: Option<ChunkKind>,
}

impl SearchFilters {
    /// Validate raw filter input.
    ///
    /// Paper IDs are normalized to uppercase and deduplicated. Free-text
    /// values are trimmed.
    ///
    /// # Errors
    ///
    /// Returns the [`FilterError`] for the first invalid field, checked in
    /// [`FilterInput`] field order.
    pub fn from_input(input: &FilterInput) -> Result<Self, FilterError> {
        let paper_ids = input
            .paper_ids
            .iter()
            .flatten()
            .map(|raw| PaperNum::parse(raw).map_err(|_| FilterError::PaperId(raw.clone())))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let revision_scope = input
            .revision_scope
            .as_deref()
            .map(str::parse)
            .transpose()?
            .unwrap_or_default();
        if let (Some(from), Some(to)) = (input.year_from, input.year_to)
            && from > to
        {
            return Err(FilterError::YearRange { from, to });
        }
        let mailing = input.mailing.as_deref().map(parse_mailing).transpose()?;
        let target_group = input
            .target_group
            .as_deref()
            .map(|v| validate_text("target_group", v))
            .transpose()?;
        let author = input
            .author
            .as_deref()
            .map(|v| validate_text("author", v))
            .transpose()?;
        let intent = input
            .intent
            .as_deref()
            .map(|v| validate_text("intent", v).map(|v| v.to_lowercase()))
            .transpose()?;
        let chunk_kind = input.chunk_kind.as_deref().map(str::parse).transpose()?;
        Ok(Self {
            paper_ids,
            revision_scope,
            year_from: input.year_from,
            year_to: input.year_to,
            mailing,
            target_group,
            author,
            intent,
            chunk_kind,
        })
    }

    /// Validated paper IDs, sorted. Empty when there is no paper ID filter.
    #[must_use]
    pub fn paper_ids(&self) -> &BTreeSet<PaperNum> {
        &self.paper_ids
    }

    /// Render the filters as a LanceDB `only_if` predicate for search.
    ///
    /// The predicate always requires `has_text = true`. A paper ID filter
    /// matches exactly those papers; without one, the `latest` scope adds
    /// `is_latest = true`.
    #[must_use]
    pub fn to_predicate(&self) -> String {
        let mut clauses = vec!["has_text = true".to_owned()];
        if !self.paper_ids.is_empty() {
            let ids: Vec<String> = self
                .paper_ids
                .iter()
                .map(|id| sql_string(id.as_str()))
                .collect();
            clauses.push(format!("paper_id IN ({})", ids.join(", ")));
        } else if self.revision_scope == RevisionScope::Latest {
            clauses.push("is_latest = true".to_owned());
        }
        if let Some(from) = self.year_from {
            clauses.push(format!("year >= {from}"));
        }
        if let Some(to) = self.year_to {
            clauses.push(format!("year <= {to}"));
        }
        if let Some(mailing) = &self.mailing {
            clauses.push(format!("mailing_date = {}", sql_string(mailing)));
        }
        if let Some(group) = &self.target_group {
            clauses.push(contains_predicate("target_group", group));
        }
        if let Some(author) = &self.author {
            clauses.push(contains_predicate("authors", author));
        }
        if let Some(intent) = &self.intent {
            clauses.push(format!("intent = {}", sql_string(intent)));
        }
        if let Some(kind) = self.chunk_kind {
            clauses.push(format!("chunk_kind = {}", sql_string(kind.as_str())));
        }
        clauses.join(" AND ")
    }
}

/// Validate a `YYYY-MM` mailing with a month from 01 to 12, after trimming.
fn parse_mailing(raw: &str) -> Result<String, FilterError> {
    let invalid = || FilterError::Mailing(raw.to_owned());
    let mailing = raw.trim();
    let (year, month) = mailing.split_once(MAILING_SEPARATOR).ok_or_else(invalid)?;
    let all_digits = |s: &str, len: usize| s.len() == len && s.bytes().all(|b| b.is_ascii_digit());
    if !all_digits(year, MAILING_YEAR_DIGITS) || !all_digits(month, MAILING_MONTH_DIGITS) {
        return Err(invalid());
    }
    let month: u32 = month.parse().map_err(|_| invalid())?;
    if !MAILING_MONTHS.contains(&month) {
        return Err(invalid());
    }
    Ok(mailing.to_owned())
}

/// Trim a free-text value and check it is non-empty, within
/// [`MAX_FILTER_TEXT_CHARS`], and free of control characters.
fn validate_text(field: &'static str, raw: &str) -> Result<String, FilterError> {
    let value = raw.trim();
    if value.is_empty() {
        return Err(FilterError::EmptyText { field });
    }
    if value.chars().count() > MAX_FILTER_TEXT_CHARS {
        return Err(FilterError::TextTooLong { field });
    }
    if value.chars().any(char::is_control) {
        return Err(FilterError::ControlCharacter { field });
    }
    Ok(value.to_owned())
}

/// Quote `value` as an SQL string literal.
fn sql_string(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// Case-insensitive match of `column` containing `value` literally.
fn contains_predicate(column: &str, value: &str) -> String {
    let mut pattern = String::from("%");
    for c in value.chars() {
        if matches!(c, '%' | '_') || c == LIKE_ESCAPE {
            pattern.push(LIKE_ESCAPE);
        }
        pattern.push(c);
    }
    pattern.push('%');
    format!(
        "{column} ILIKE {} ESCAPE {}",
        sql_string(&pattern),
        sql_string(&LIKE_ESCAPE.to_string())
    )
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::sync::Arc;

    use lancedb::arrow::arrow_array::{BooleanArray, Int32Array, RecordBatch, StringArray};
    use lancedb::arrow::arrow_schema::{DataType, Field, Schema};

    use super::*;

    const HAS_TEXT: &str = "has_text = true";
    const LATEST: &str = "is_latest = true";

    fn filters(input: &FilterInput) -> SearchFilters {
        SearchFilters::from_input(input).unwrap_or_else(|e| panic!("from_input: {e}"))
    }

    fn predicate(input: &FilterInput) -> String {
        filters(input).to_predicate()
    }

    fn ids(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|&id| id.to_owned()).collect()
    }

    fn text(value: &str) -> String {
        value.to_owned()
    }

    #[test]
    fn no_filters_restrict_to_searchable_latest_chunks() {
        assert_eq!(
            predicate(&FilterInput::default()),
            format!("{HAS_TEXT} AND {LATEST}")
        );
    }

    #[test]
    fn all_scope_without_paper_ids_drops_the_latest_clause() {
        let input = FilterInput {
            revision_scope: Some(text("all")),
            ..FilterInput::default()
        };
        assert_eq!(predicate(&input), HAS_TEXT);
        let input = FilterInput {
            revision_scope: Some(text("latest")),
            ..FilterInput::default()
        };
        assert_eq!(predicate(&input), format!("{HAS_TEXT} AND {LATEST}"));
    }

    #[test]
    fn paper_ids_match_exactly_and_override_either_scope() {
        for scope in [None, Some(text("latest")), Some(text("all"))] {
            let input = FilterInput {
                paper_ids: Some(ids(&["p2300r10", " P2300R4 ", "P2300R10"])),
                revision_scope: scope,
                ..FilterInput::default()
            };
            assert_eq!(
                predicate(&input),
                format!("{HAS_TEXT} AND paper_id IN ('P2300R10', 'P2300R4')")
            );
        }
    }

    #[test]
    fn empty_paper_id_list_is_no_paper_filter() {
        let input = FilterInput {
            paper_ids: Some(Vec::new()),
            ..FilterInput::default()
        };
        assert_eq!(predicate(&input), format!("{HAS_TEXT} AND {LATEST}"));
    }

    #[test]
    fn paper_ids_are_normalized_and_sorted() {
        let input = FilterInput {
            paper_ids: Some(ids(&["sd-5", "n4950", "P1000R0"])),
            ..FilterInput::default()
        };
        let validated = filters(&input);
        let got: Vec<&str> = validated.paper_ids().iter().map(PaperNum::as_str).collect();
        assert_eq!(got, ["N4950", "P1000R0", "SD-5"]);
    }

    #[test]
    fn each_filter_maps_to_its_clause() {
        let cases: [(FilterInput, &str); 8] = [
            (
                FilterInput {
                    year_from: Some(2024),
                    ..FilterInput::default()
                },
                "year >= 2024",
            ),
            (
                FilterInput {
                    year_to: Some(2025),
                    ..FilterInput::default()
                },
                "year <= 2025",
            ),
            (
                FilterInput {
                    mailing: Some(text(" 2025-03 ")),
                    ..FilterInput::default()
                },
                "mailing_date = '2025-03'",
            ),
            (
                FilterInput {
                    target_group: Some(text(" LEWG ")),
                    ..FilterInput::default()
                },
                r"target_group ILIKE '%LEWG%' ESCAPE '\'",
            ),
            (
                FilterInput {
                    author: Some(text("Niebler")),
                    ..FilterInput::default()
                },
                r"authors ILIKE '%Niebler%' ESCAPE '\'",
            ),
            (
                FilterInput {
                    intent: Some(text(" Ask ")),
                    ..FilterInput::default()
                },
                "intent = 'ask'",
            ),
            (
                FilterInput {
                    chunk_kind: Some(text("revision_history")),
                    ..FilterInput::default()
                },
                "chunk_kind = 'revision_history'",
            ),
            (
                FilterInput {
                    year_from: Some(2024),
                    year_to: Some(2024),
                    ..FilterInput::default()
                },
                "year >= 2024 AND year <= 2024",
            ),
        ];
        for (input, clause) in cases {
            assert_eq!(
                predicate(&input),
                format!("{HAS_TEXT} AND {LATEST} AND {clause}"),
                "{input:?}"
            );
        }
    }

    #[test]
    fn combined_filters_render_in_fixed_order() {
        let input = FilterInput {
            paper_ids: Some(ids(&["P2300R10"])),
            revision_scope: Some(text("all")),
            year_from: Some(2023),
            year_to: Some(2025),
            mailing: Some(text("2024-12")),
            target_group: Some(text("LEWG")),
            author: Some(text("Dominiak")),
            intent: Some(text("info")),
            chunk_kind: Some(text("wording")),
        };
        assert_eq!(
            predicate(&input),
            format!(
                "{HAS_TEXT} AND paper_id IN ('P2300R10') AND year >= 2023 AND year <= 2025 \
                 AND mailing_date = '2024-12' AND target_group ILIKE '%LEWG%' ESCAPE '\\' \
                 AND authors ILIKE '%Dominiak%' ESCAPE '\\' AND intent = 'info' \
                 AND chunk_kind = 'wording'"
            )
        );
    }

    #[test]
    fn hostile_substrings_are_quoted_and_wildcards_escaped() {
        let cases = [
            ("O'Brien", r"'%O''Brien%'"),
            ("x' OR '1'='1", r"'%x'' OR ''1''=''1%'"),
            ("a -- comment", r"'%a -- comment%'"),
            (
                "x'; DROP TABLE chunks; --",
                r"'%x''; DROP TABLE chunks; --%'",
            ),
            ("100%", r"'%100\%%'"),
            ("snake_case", r"'%snake\_case%'"),
            (r"back\slash", r"'%back\\slash%'"),
        ];
        for (hostile, pattern) in cases {
            let input = FilterInput {
                author: Some(text(hostile)),
                target_group: Some(text(hostile)),
                ..FilterInput::default()
            };
            assert_eq!(
                predicate(&input),
                format!(
                    "{HAS_TEXT} AND {LATEST} AND target_group ILIKE {pattern} ESCAPE '\\' \
                     AND authors ILIKE {pattern} ESCAPE '\\'"
                ),
                "{hostile}"
            );
        }
    }

    #[test]
    fn hostile_intent_is_quoted() {
        let input = FilterInput {
            intent: Some(text("ask' OR has_text = false --")),
            ..FilterInput::default()
        };
        assert_eq!(
            predicate(&input),
            format!("{HAS_TEXT} AND {LATEST} AND intent = 'ask'' or has_text = false --'")
        );
    }

    #[test]
    fn invalid_input_is_rejected() {
        let long = "a".repeat(MAX_FILTER_TEXT_CHARS + 1);
        let cases = [
            (
                FilterInput {
                    paper_ids: Some(ids(&["P2300R10", "P2300R10' OR '1'='1"])),
                    ..FilterInput::default()
                },
                FilterError::PaperId("P2300R10' OR '1'='1".to_owned()),
            ),
            (
                FilterInput {
                    revision_scope: Some(text("newest")),
                    ..FilterInput::default()
                },
                FilterError::RevisionScope(RevisionError::UnknownScope("newest".to_owned())),
            ),
            (
                FilterInput {
                    year_from: Some(2025),
                    year_to: Some(2024),
                    ..FilterInput::default()
                },
                FilterError::YearRange {
                    from: 2025,
                    to: 2024,
                },
            ),
            (
                FilterInput {
                    mailing: Some(text("2025-13")),
                    ..FilterInput::default()
                },
                FilterError::Mailing("2025-13".to_owned()),
            ),
            (
                FilterInput {
                    mailing: Some(text("2025-3")),
                    ..FilterInput::default()
                },
                FilterError::Mailing("2025-3".to_owned()),
            ),
            (
                FilterInput {
                    mailing: Some(text("2025-03' OR '1'='1")),
                    ..FilterInput::default()
                },
                FilterError::Mailing("2025-03' OR '1'='1".to_owned()),
            ),
            (
                FilterInput {
                    target_group: Some(text("   ")),
                    ..FilterInput::default()
                },
                FilterError::EmptyText {
                    field: "target_group",
                },
            ),
            (
                FilterInput {
                    author: Some(text(&long)),
                    ..FilterInput::default()
                },
                FilterError::TextTooLong { field: "author" },
            ),
            (
                FilterInput {
                    author: Some(text("a\0b")),
                    ..FilterInput::default()
                },
                FilterError::ControlCharacter { field: "author" },
            ),
            (
                FilterInput {
                    intent: Some(text("ask\nOR 1=1")),
                    ..FilterInput::default()
                },
                FilterError::ControlCharacter { field: "intent" },
            ),
            (
                FilterInput {
                    chunk_kind: Some(text("body' OR '1'='1")),
                    ..FilterInput::default()
                },
                FilterError::ChunkKind(UnknownChunkKind("body' OR '1'='1".to_owned())),
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(
                SearchFilters::from_input(&input),
                Err(expected),
                "{input:?}"
            );
        }
    }

    #[test]
    fn text_at_the_length_limit_is_accepted() {
        let input = FilterInput {
            author: Some(text(&"a".repeat(MAX_FILTER_TEXT_CHARS))),
            ..FilterInput::default()
        };
        assert!(SearchFilters::from_input(&input).is_ok());
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|e| panic!("build runtime: {e}"))
            .block_on(future)
    }

    /// Rows of `(paper_id, is_latest, year, authors)`, all with text.
    const ROWS: [(&str, bool, i32, &str); 4] = [
        ("P1R0", true, 2024, "O'Brien"),
        ("P2R1", true, 2025, "100% Effort"),
        ("P3R0", false, 2024, "snake_case; x'; DROP TABLE chunks; --"),
        ("P4R0", true, 2025, "Plain Name"),
    ];

    /// Paper IDs matching `input` in an in-memory LanceDB table of [`ROWS`].
    fn lance_matches(input: &FilterInput) -> Vec<String> {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let uri = dir
            .path()
            .to_str()
            .unwrap_or_else(|| panic!("non-UTF-8 tempdir"));
        let schema = Arc::new(Schema::new(vec![
            Field::new("paper_id", DataType::Utf8, false),
            Field::new("is_latest", DataType::Boolean, false),
            Field::new("has_text", DataType::Boolean, false),
            Field::new("year", DataType::Int32, true),
            Field::new("mailing_date", DataType::Utf8, false),
            Field::new("target_group", DataType::Utf8, false),
            Field::new("authors", DataType::Utf8, false),
            Field::new("intent", DataType::Utf8, false),
            Field::new("chunk_kind", DataType::Utf8, true),
        ]));
        let strings = |values: Vec<&str>| Arc::new(StringArray::from(values));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                strings(ROWS.iter().map(|r| r.0).collect()),
                Arc::new(BooleanArray::from(
                    ROWS.iter().map(|r| r.1).collect::<Vec<_>>(),
                )),
                Arc::new(BooleanArray::from(vec![true; ROWS.len()])),
                Arc::new(Int32Array::from(
                    ROWS.iter().map(|r| r.2).collect::<Vec<_>>(),
                )),
                strings(vec!["2025-03"; ROWS.len()]),
                strings(ROWS.iter().map(|r| r.3).collect()),
                strings(ROWS.iter().map(|r| r.3).collect()),
                strings(vec!["info"; ROWS.len()]),
                strings(vec!["body"; ROWS.len()]),
            ],
        )
        .unwrap_or_else(|e| panic!("batch: {e}"));
        let predicate = filters(input).to_predicate();
        block_on(async {
            let conn = lancedb::connect(uri)
                .execute()
                .await
                .unwrap_or_else(|e| panic!("connect: {e}"));
            let table = conn
                .create_table("rows", batch)
                .execute()
                .await
                .unwrap_or_else(|e| panic!("create table: {e}"));
            let mut matched = Vec::new();
            for (id, ..) in ROWS {
                let row = format!("paper_id = '{id}' AND ({predicate})");
                let count = table
                    .count_rows(Some(row))
                    .await
                    .unwrap_or_else(|e| panic!("predicate {predicate:?} rejected: {e}"));
                if count > 0 {
                    matched.push(id.to_owned());
                }
            }
            matched
        })
    }

    #[test]
    fn predicates_execute_in_lancedb_with_literal_matching() {
        let cases: [(&str, &[&str]); 6] = [
            ("O'Brien", &["P1R0"]),
            ("o'brien", &["P1R0"]),
            ("100%", &["P2R1"]),
            ("%", &["P2R1"]),
            ("snake_case", &[]),
            ("x'; DROP TABLE chunks; --", &[]),
        ];
        for (author, expected) in cases {
            let input = FilterInput {
                author: Some(text(author)),
                revision_scope: Some(text("latest")),
                ..FilterInput::default()
            };
            assert_eq!(lance_matches(&input), expected, "{author}");
        }
        let input = FilterInput {
            author: Some(text("x'; DROP TABLE chunks; --")),
            revision_scope: Some(text("all")),
            ..FilterInput::default()
        };
        assert_eq!(lance_matches(&input), ["P3R0"]);
        let input = FilterInput {
            author: Some(text("_")),
            revision_scope: Some(text("all")),
            ..FilterInput::default()
        };
        assert_eq!(lance_matches(&input), ["P3R0"]);
        let input = FilterInput {
            paper_ids: Some(ids(&["P3R0", "P4R0"])),
            year_from: Some(2025),
            ..FilterInput::default()
        };
        assert_eq!(lance_matches(&input), ["P4R0"]);
    }
}
