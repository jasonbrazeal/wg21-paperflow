// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Metadata filters shared by every search side.
//!
//! A [`SearchFilter`] renders to one SQL predicate over columns that both
//! index tables carry, so vector, keyword and metadata searches apply the
//! same restriction.

use paperstore::PaperNum;

/// Column holding the paper id with revision, as stored by paperstore.
pub(crate) const COL_PAPER_ID: &str = "paper_id";
/// Column holding the paper number without revision, shared by every revision.
pub(crate) const COL_PAPER_NUMBER: &str = "paper_number";
/// Column holding the revision, null for unversioned papers.
pub(crate) const COL_REVISION: &str = "revision";
/// Column flagging the latest revision of a paper number.
pub(crate) const COL_IS_LATEST: &str = "is_latest";
/// Column holding the target subgroup.
pub(crate) const COL_TARGET_GROUP: &str = "target_group";
/// Column holding the author names joined into one string.
pub(crate) const COL_AUTHORS_TEXT: &str = "authors_text";
/// Column holding the mailing year.
pub(crate) const COL_YEAR: &str = "year";
/// Column holding the mailing date (`YYYY-MM`).
pub(crate) const COL_MAILING_DATE: &str = "mailing_date";
/// Column holding the document date (`YYYY-MM-DD`).
pub(crate) const COL_DOCUMENT_DATE: &str = "document_date";
/// Column holding the intent.
pub(crate) const COL_INTENT: &str = "intent";

/// Restrictions on which index records a search may return.
///
/// Every set field must hold for a record to match; unset fields do not
/// restrict. Text comparisons on target group, author and intent ignore
/// case.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SearchFilter {
    /// A paper id. With a revision it matches only that revision; without
    /// one it matches every revision of the paper number.
    pub paper_id: Option<PaperNum>,
    /// Exact revision number.
    pub revision: Option<u32>,
    /// Substring of the target group (e.g. `LEWG` or `Library Evolution`).
    pub target_group: Option<String>,
    /// Substring of any author name.
    pub author: Option<String>,
    /// Exact mailing year.
    pub year: Option<u16>,
    /// Exact mailing date (`YYYY-MM`).
    pub mailing_date: Option<String>,
    /// Earliest document date, inclusive, compared as an ISO date string.
    pub document_date_from: Option<String>,
    /// Latest document date, inclusive, compared as an ISO date string.
    pub document_date_to: Option<String>,
    /// Exact intent.
    pub intent: Option<String>,
    /// Match only records flagged as the latest revision of their paper.
    pub latest_only: bool,
}

impl SearchFilter {
    /// Render the filter as a LanceDB SQL predicate.
    ///
    /// Clauses join with `AND` in a fixed order, and every string value is
    /// quoted with embedded single quotes doubled. A document date bound
    /// excludes records without a document date. Returns `None` when no
    /// field restricts.
    #[must_use]
    pub fn to_predicate(&self) -> Option<String> {
        let mut clauses = Vec::new();
        if let Some(id) = &self.paper_id {
            let (number, revision) = id.split_revision();
            clauses.push(match revision {
                Some(_) => eq(COL_PAPER_ID, id.as_str()),
                None => eq(COL_PAPER_NUMBER, number),
            });
        }
        if let Some(revision) = self.revision {
            clauses.push(format!("{COL_REVISION} = {revision}"));
        }
        if let Some(group) = &self.target_group {
            clauses.push(contains_ci(COL_TARGET_GROUP, group));
        }
        if let Some(author) = &self.author {
            clauses.push(contains_ci(COL_AUTHORS_TEXT, author));
        }
        if let Some(year) = self.year {
            clauses.push(eq(COL_YEAR, &year.to_string()));
        }
        if let Some(mailing) = &self.mailing_date {
            clauses.push(eq(COL_MAILING_DATE, mailing));
        }
        if self.document_date_from.is_some() || self.document_date_to.is_some() {
            clauses.push(format!("{COL_DOCUMENT_DATE} <> ''"));
        }
        if let Some(from) = &self.document_date_from {
            clauses.push(format!("{COL_DOCUMENT_DATE} >= {}", quote(from)));
        }
        if let Some(to) = &self.document_date_to {
            clauses.push(format!("{COL_DOCUMENT_DATE} <= {}", quote(to)));
        }
        if let Some(intent) = &self.intent {
            clauses.push(format!(
                "lower({COL_INTENT}) = {}",
                quote(&intent.to_lowercase())
            ));
        }
        if self.latest_only {
            clauses.push(format!("{COL_IS_LATEST} = true"));
        }
        (!clauses.is_empty()).then(|| clauses.join(" AND "))
    }
}

/// Quote `value` as a SQL string literal, doubling embedded single quotes.
#[must_use]
pub(crate) fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// Render `column = 'value'`.
pub(crate) fn eq(column: &str, value: &str) -> String {
    format!("{column} = {}", quote(value))
}

/// Render a case-insensitive substring test.
///
/// `strpos` takes the needle literally, so `%` and `_` need no escaping as
/// they would in a `LIKE` pattern.
fn contains_ci(column: &str, needle: &str) -> String {
    format!(
        "strpos(lower({column}), {}) > 0",
        quote(&needle.to_lowercase())
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(raw: &str) -> Option<PaperNum> {
        PaperNum::parse(raw).ok()
    }

    #[test]
    fn empty_filter_has_no_predicate() {
        assert_eq!(SearchFilter::default().to_predicate(), None);
    }

    #[test]
    fn versioned_id_matches_one_revision() {
        let f = SearchFilter {
            paper_id: num("p2300r4"),
            ..SearchFilter::default()
        };
        assert_eq!(f.to_predicate().as_deref(), Some("paper_id = 'P2300R4'"));
    }

    #[test]
    fn unversioned_id_matches_the_paper_number() {
        let f = SearchFilter {
            paper_id: num("P2300"),
            ..SearchFilter::default()
        };
        assert_eq!(f.to_predicate().as_deref(), Some("paper_number = 'P2300'"));
    }

    #[test]
    fn every_field_renders_in_order() {
        let f = SearchFilter {
            paper_id: num("N4950"),
            revision: Some(3),
            target_group: Some("LEWG".to_owned()),
            author: Some("Wakely".to_owned()),
            year: Some(2024),
            mailing_date: Some("2024-05".to_owned()),
            document_date_from: Some("2024-01-01".to_owned()),
            document_date_to: Some("2024-12-31".to_owned()),
            intent: Some("Inform".to_owned()),
            latest_only: true,
        };
        assert_eq!(
            f.to_predicate().as_deref(),
            Some(
                "paper_number = 'N4950' AND revision = 3 \
                 AND strpos(lower(target_group), 'lewg') > 0 \
                 AND strpos(lower(authors_text), 'wakely') > 0 \
                 AND year = '2024' AND mailing_date = '2024-05' \
                 AND document_date <> '' \
                 AND document_date >= '2024-01-01' \
                 AND document_date <= '2024-12-31' \
                 AND lower(intent) = 'inform' AND is_latest = true"
            )
        );
    }

    #[test]
    fn single_quotes_are_doubled() {
        let f = SearchFilter {
            author: Some("O'Neal' OR 1=1 --".to_owned()),
            mailing_date: Some("'".to_owned()),
            ..SearchFilter::default()
        };
        assert_eq!(
            f.to_predicate().as_deref(),
            Some(
                "strpos(lower(authors_text), 'o''neal'' or 1=1 --') > 0 \
                 AND mailing_date = ''''"
            )
        );
    }

    #[test]
    fn like_wildcards_stay_literal() {
        let f = SearchFilter {
            target_group: Some("50%_".to_owned()),
            ..SearchFilter::default()
        };
        assert_eq!(
            f.to_predicate().as_deref(),
            Some("strpos(lower(target_group), '50%_') > 0")
        );
    }

    #[test]
    fn one_date_bound_still_excludes_undated() {
        let f = SearchFilter {
            document_date_to: Some("2020-01-01".to_owned()),
            ..SearchFilter::default()
        };
        assert_eq!(
            f.to_predicate().as_deref(),
            Some("document_date <> '' AND document_date <= '2020-01-01'")
        );
    }
}
