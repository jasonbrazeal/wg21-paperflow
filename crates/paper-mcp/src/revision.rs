//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Paper revision lineage and latest-revision selection.
//!
//! The revisions of a paper are the paperstore rows whose paper IDs differ
//! only in their `R<n>` suffix. The latest revision is the one with the
//! highest revision number, compared numerically so `R10` follows `R9`.
//! Papers without a revision suffix (N papers and standing documents) are
//! always latest.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use paperstore::{PaperNum, PaperRow};

/// Marker introducing a paper's revision number.
const REVISION_MARKER: char = 'R';

/// Errors from parsing revision inputs.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RevisionError {
    /// A revision scope other than `latest` or `all`.
    #[error("{0:?} is not a revision scope; expected \"latest\" or \"all\"")]
    UnknownScope(String),
}

/// Which revisions a corpus-wide search covers.
///
/// Applies only when a search has no paper ID filter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum RevisionScope {
    /// Only the latest revision of each paper.
    #[default]
    Latest,
    /// Every revision.
    All,
}

impl RevisionScope {
    /// Name accepted from tool input.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Latest => "latest",
            Self::All => "all",
        }
    }
}

impl fmt::Display for RevisionScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for RevisionScope {
    type Err = RevisionError;

    /// Parse `latest` or `all`, exactly as named.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "latest" => Ok(Self::Latest),
            "all" => Ok(Self::All),
            _ => Err(RevisionError::UnknownScope(s.to_owned())),
        }
    }
}

/// Split a paper ID into its revisionless stem and revision digits, if any.
///
/// Standing document IDs (`SD-<n>`) never contain the marker, so they have no
/// revision.
fn split_revision(id: &str) -> (&str, Option<&str>) {
    match id.rfind(REVISION_MARKER) {
        Some(pos) => (&id[..pos], Some(&id[pos + REVISION_MARKER.len_utf8()..])),
        None => (id, None),
    }
}

/// Return the paper ID with any `R<n>` suffix removed.
///
/// IDs without a revision (N papers, standing documents) are returned
/// unchanged. All revisions of a paper share this value.
#[must_use]
pub fn revisionless_id(num: &PaperNum) -> String {
    split_revision(num.as_str()).0.to_owned()
}

/// Return the paper's revision number.
///
/// Returns `None` for IDs without an `R<n>` suffix (N papers, standing
/// documents) and for revision numbers that do not fit in a `u32`.
#[must_use]
pub fn revision_number(num: &PaperNum) -> Option<u32> {
    split_revision(num.as_str()).1?.parse().ok()
}

/// Return the paper IDs of the rows that are their paper's latest revision.
///
/// Rows are grouped by [`revisionless_id`]; in each group, the row with the
/// highest revision number is latest. When several rows tie on that number
/// (`P2300R7` and `P2300R007`), the one with the lowest paper ID is latest, so
/// the result does not depend on row order. Rows without an `R<n>` suffix are
/// always latest. Rows whose `paper_num` is not a valid paper number, or whose
/// revision number does not fit in a `u32`, are never included.
#[must_use]
pub fn latest_set(rows: &[PaperRow]) -> BTreeSet<String> {
    let mut latest = BTreeSet::new();
    let mut highest: BTreeMap<String, (u32, Reverse<PaperNum>)> = BTreeMap::new();
    for num in rows
        .iter()
        .filter_map(|r| PaperNum::parse(&r.paper_num).ok())
    {
        let (stem, digits) = split_revision(num.as_str());
        let Some(digits) = digits else {
            latest.insert(num.as_str().to_owned());
            continue;
        };
        let Ok(revision) = digits.parse::<u32>() else {
            continue;
        };
        let group = stem.to_owned();
        let candidate = (revision, Reverse(num));
        match highest.get(&group) {
            Some(best) if *best >= candidate => {}
            _ => {
                highest.insert(group, candidate);
            }
        }
    }
    latest.extend(
        highest
            .into_values()
            .map(|(_, Reverse(num))| num.as_str().to_owned()),
    );
    latest
}

/// Parse a `list_revisions` paper ID, with or without its `R<n>` suffix.
///
/// This is the only paper ID input that may omit the revision. Returns the
/// revisionless ID shared by every revision of the paper, so `P2300` and
/// `P2300R3` both yield `P2300`.
///
/// # Errors
///
/// Returns [`paperstore::Error::BadPaperNum`] when the input is not a paper
/// number.
pub fn parse_revision_query(raw: &str) -> paperstore::Result<String> {
    PaperNum::parse(raw).map(|num| revisionless_id(&num))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(raw: &str) -> PaperNum {
        PaperNum::parse(raw).unwrap_or_else(|e| panic!("{raw:?} should parse, got {e}"))
    }

    fn rows(ids: &[&str]) -> Vec<PaperRow> {
        ids.iter()
            .map(|id| PaperRow {
                paper_num: (*id).to_owned(),
                ..PaperRow::default()
            })
            .collect()
    }

    fn set(ids: &[&str]) -> BTreeSet<String> {
        ids.iter().map(|id| (*id).to_owned()).collect()
    }

    #[test]
    fn revisionless_id_strips_only_the_revision_suffix() {
        for (raw, want) in [
            ("P2300R10", "P2300"),
            ("P2300R0", "P2300"),
            ("D1234R2", "D1234"),
            ("P2300", "P2300"),
            ("N4950", "N4950"),
            ("SD-1", "SD-1"),
        ] {
            assert_eq!(revisionless_id(&num(raw)), want, "{raw}");
        }
    }

    #[test]
    fn revision_number_is_numeric_and_absent_for_n_and_sd() {
        assert_eq!(revision_number(&num("P2300R10")), Some(10));
        assert_eq!(revision_number(&num("P2300R9")), Some(9));
        assert_eq!(revision_number(&num("P2300R0")), Some(0));
        assert_eq!(revision_number(&num("P2300R007")), Some(7));
        assert_eq!(revision_number(&num("N4950")), None);
        assert_eq!(revision_number(&num("SD-5")), None);
        assert_eq!(revision_number(&num("P2300R99999999999")), None);
    }

    #[test]
    fn latest_set_orders_revisions_numerically() {
        let got = latest_set(&rows(&["P2300R9", "P2300R10", "P2300R2"]));
        assert_eq!(got, set(&["P2300R10"]));
    }

    #[test]
    fn latest_set_groups_each_paper_independently() {
        let got = latest_set(&rows(&[
            "P1000R0", "P1000R2", "P1000R1", "P2000R0", "D1000R5", "P10000R1",
        ]));
        assert_eq!(got, set(&["D1000R5", "P1000R2", "P10000R1", "P2000R0"]));
    }

    #[test]
    fn latest_set_keeps_every_n_and_sd_paper() {
        let got = latest_set(&rows(&["N4950", "N4951", "SD-1", "SD-5", "P3000R1"]));
        assert_eq!(got, set(&["N4950", "N4951", "P3000R1", "SD-1", "SD-5"]));
    }

    #[test]
    fn latest_set_is_independent_of_row_order() {
        let ids = ["P2300R10", "P2300R9", "N4950", "P2300R0"];
        let mut reversed = ids;
        reversed.reverse();
        assert_eq!(latest_set(&rows(&ids)), latest_set(&rows(&reversed)));
        assert_eq!(latest_set(&rows(&ids)), set(&["N4950", "P2300R10"]));
    }

    #[test]
    fn latest_set_breaks_revision_ties_by_lowest_paper_id_in_any_order() {
        let ids = ["P2300R7", "P2300R007", "P2300R2"];
        let mut reversed = ids;
        reversed.reverse();
        assert_eq!(latest_set(&rows(&ids)), set(&["P2300R007"]));
        assert_eq!(latest_set(&rows(&reversed)), set(&["P2300R007"]));
    }

    #[test]
    fn latest_set_skips_revisions_that_overflow() {
        let got = latest_set(&rows(&["P2300R99999999999", "P2300R3", "N4950"]));
        assert_eq!(got, set(&["N4950", "P2300R3"]));
        let got = latest_set(&rows(&["P2400R99999999999"]));
        assert!(got.is_empty());
    }

    #[test]
    fn latest_set_skips_invalid_paper_numbers() {
        let got = latest_set(&rows(&["P2300R1", "not a paper", ""]));
        assert_eq!(got, set(&["P2300R1"]));
    }

    #[test]
    fn latest_set_of_no_rows_is_empty() {
        assert!(latest_set(&[]).is_empty());
    }

    #[test]
    fn parse_revision_query_accepts_ids_with_and_without_revision() {
        for (raw, want) in [
            ("P2300", "P2300"),
            ("P2300R3", "P2300"),
            ("  p2300r10 ", "P2300"),
            ("N4950", "N4950"),
            ("sd-1", "SD-1"),
        ] {
            match parse_revision_query(raw) {
                Ok(got) => assert_eq!(got, want, "{raw:?}"),
                Err(e) => panic!("{raw:?} should parse, got {e}"),
            }
        }
    }

    #[test]
    fn parse_revision_query_rejects_non_paper_ids() {
        for raw in ["", "2300", "P2300R", "PR3", "P2300; DROP"] {
            match parse_revision_query(raw) {
                Err(paperstore::Error::BadPaperNum { 0: carried, .. }) => assert_eq!(carried, raw),
                other => panic!("{raw:?} should fail with BadPaperNum, got {other:?}"),
            }
        }
    }

    #[test]
    fn revision_scope_defaults_to_latest() {
        assert_eq!(RevisionScope::default(), RevisionScope::Latest);
    }

    #[test]
    fn revision_scope_round_trips_and_rejects_unknown_names() {
        for scope in [RevisionScope::Latest, RevisionScope::All] {
            assert_eq!(scope.as_str().parse::<RevisionScope>(), Ok(scope));
            assert_eq!(scope.to_string(), scope.as_str());
        }
        for raw in ["", "Latest", "ALL", "newest"] {
            assert_eq!(
                raw.parse::<RevisionScope>(),
                Err(RevisionError::UnknownScope(raw.to_owned()))
            );
        }
    }
}
