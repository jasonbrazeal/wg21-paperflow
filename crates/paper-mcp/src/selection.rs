// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Paper selection: operator selectors resolved to paper numbers.

use std::collections::BTreeSet;

use paperstore::{PaperNum, StorageBackend};

use crate::error::{Error, Result};

/// The selector that selects every stored paper, matched case-insensitively.
const ALL: &str = "all";

/// The number of ASCII digits in a year selector.
const YEAR_DIGITS: usize = 4;

/// One operator selector for the papers a command covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selector {
    /// Explicit paper ids, versioned or unversioned.
    Ids(Vec<PaperNum>),
    /// Every paper whose mailing year is the given year.
    Year(u16),
    /// Every stored paper.
    All,
}

/// Parses command-line selectors.
///
/// `all` (any case) selects every paper, four ASCII digits select a year,
/// and anything else must be a paper id. Consecutive ids are grouped into one
/// [`Selector::Ids`]; years and ids may be mixed.
///
/// # Errors
///
/// Returns [`Error::EmptySelection`] for empty input,
/// [`Error::AllWithOthers`] when `all` appears with any other selector, and
/// [`Error::BadSelector`] for an argument that is none of the three forms.
pub fn parse_selectors(args: &[String]) -> Result<Vec<Selector>> {
    if args.is_empty() {
        return Err(Error::EmptySelection);
    }
    if args.iter().any(|a| a.eq_ignore_ascii_case(ALL)) {
        return if args.len() == 1 {
            Ok(vec![Selector::All])
        } else {
            Err(Error::AllWithOthers)
        };
    }
    let mut selectors = Vec::new();
    for arg in args {
        if let Some(year) = parse_year(arg) {
            selectors.push(Selector::Year(year));
            continue;
        }
        let num = PaperNum::parse(arg).map_err(|_| Error::BadSelector(arg.clone()))?;
        if let Some(Selector::Ids(ids)) = selectors.last_mut() {
            ids.push(num);
        } else {
            selectors.push(Selector::Ids(vec![num]));
        }
    }
    Ok(selectors)
}

/// Resolves selectors to paper numbers, deduplicated and sorted ascending.
///
/// Each id resolves through [`resolve_latest`]. A year with no papers
/// contributes nothing.
///
/// # Errors
///
/// Returns [`Error::Store`] when the store cannot be read or an id names an
/// unknown paper.
pub fn resolve(store: &dyn StorageBackend, selectors: &[Selector]) -> Result<Vec<PaperNum>> {
    let mut papers = BTreeSet::new();
    for selector in selectors {
        match selector {
            Selector::Ids(ids) => {
                for id in ids {
                    papers.insert(resolve_latest(store, id)?);
                }
            }
            Selector::Year(year) => papers.extend(store.list_paper_ids_for_year(*year)?),
            Selector::All => papers.extend(store.list_all_paper_ids()?),
        }
    }
    Ok(papers.into_iter().collect())
}

/// Resolves a paper id to a stored revision.
///
/// An unversioned id resolves to the highest revision paperstore lists for
/// that paper number. A versioned id is returned unchanged once it is known
/// to exist.
///
/// # Errors
///
/// Returns [`Error::Store`] wrapping [`paperstore::Error::MissingPaper`]
/// when no matching paper is stored, [`paperstore::Error::BadPaperNum`] when
/// a listed revision carries a malformed paper number, or any other store
/// error.
pub fn resolve_latest(store: &dyn StorageBackend, num: &PaperNum) -> Result<PaperNum> {
    if num.split_revision().1.is_some() {
        store.meta(num)?;
        return Ok(num.clone());
    }
    let revisions = store
        .list_revisions(num)?
        .iter()
        .map(|row| PaperNum::parse(&row.paper_num))
        .collect::<paperstore::Result<Vec<_>>>()?;
    revisions
        .into_iter()
        .max_by_key(|n| n.split_revision().1)
        .ok_or_else(|| paperstore::Error::missing_paper(num.clone()).into())
}

fn parse_year(arg: &str) -> Option<u16> {
    if arg.len() == YEAR_DIGITS && arg.bytes().all(|b| b.is_ascii_digit()) {
        arg.parse().ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use paperstore::PaperRow;

    use super::*;

    /// In-memory store keyed by paper number, each row tagged with a year.
    ///
    /// `junk_revision`, when set, is listed as an extra revision row whose
    /// `paper_num` bypasses `PaperNum` validation.
    struct MemStore(BTreeMap<PaperNum, u16>, Option<String>);

    impl MemStore {
        fn new(papers: &[(&str, u16)]) -> Self {
            Self(
                papers.iter().map(|(id, year)| (num(id), *year)).collect(),
                None,
            )
        }

        fn row(n: &PaperNum, year: u16) -> PaperRow {
            PaperRow {
                paper_num: n.as_str().to_owned(),
                year: year.to_string(),
                ..PaperRow::default()
            }
        }
    }

    impl StorageBackend for MemStore {
        fn meta(&self, n: &PaperNum) -> paperstore::Result<PaperRow> {
            self.0
                .get(n)
                .map(|y| Self::row(n, *y))
                .ok_or_else(|| paperstore::Error::missing_paper(n.clone()))
        }

        fn paper_md(&self, n: &PaperNum) -> paperstore::Result<String> {
            Err(paperstore::Error::missing_markdown(n.clone()))
        }

        fn list_all_paper_ids(&self) -> paperstore::Result<Vec<PaperNum>> {
            Ok(self.0.keys().cloned().collect())
        }

        fn list_paper_ids_for_year(&self, year: u16) -> paperstore::Result<Vec<PaperNum>> {
            Ok(self
                .0
                .iter()
                .filter(|(_, y)| **y == year)
                .map(|(n, _)| n.clone())
                .collect())
        }

        fn list_revisions(&self, n: &PaperNum) -> paperstore::Result<Vec<PaperRow>> {
            let base = n.split_revision().0;
            let mut rows: Vec<(Option<u32>, PaperRow)> = self
                .0
                .iter()
                .filter(|(k, _)| k.split_revision().0 == base)
                .map(|(k, y)| (k.split_revision().1, Self::row(k, *y)))
                .collect();
            rows.sort_by_key(|r| std::cmp::Reverse(r.0));
            let mut out: Vec<PaperRow> = rows.into_iter().map(|(_, r)| r).collect();
            if let Some(junk) = &self.1 {
                out.push(PaperRow {
                    paper_num: junk.clone(),
                    ..PaperRow::default()
                });
            }
            Ok(out)
        }

        fn tool_dir(&self, tool: &str) -> PathBuf {
            PathBuf::from(tool)
        }
    }

    fn num(raw: &str) -> PaperNum {
        PaperNum::parse(raw).unwrap_or_else(|e| panic!("{raw:?}: {e}"))
    }

    fn nums(raws: &[&str]) -> Vec<PaperNum> {
        raws.iter().map(|r| num(r)).collect()
    }

    fn args(raws: &[&str]) -> Vec<String> {
        raws.iter().map(|s| (*s).to_owned()).collect()
    }

    fn store() -> MemStore {
        MemStore::new(&[
            ("P2300R4", 2022),
            ("P2300R9", 2024),
            ("P2300R10", 2024),
            ("P3000R0", 2024),
            ("N4950", 2023),
        ])
    }

    fn parse_ok(raws: &[&str]) -> Vec<Selector> {
        parse_selectors(&args(raws)).unwrap_or_else(|e| panic!("{raws:?}: {e}"))
    }

    fn resolve_ok(raws: &[&str]) -> Vec<PaperNum> {
        resolve(&store(), &parse_ok(raws)).unwrap_or_else(|e| panic!("{raws:?}: {e}"))
    }

    #[test]
    fn parse_ids_groups_and_normalizes() {
        assert_eq!(
            parse_ok(&["p2300r4", "N4950"]),
            vec![Selector::Ids(nums(&["P2300R4", "N4950"]))]
        );
    }

    #[test]
    fn parse_year_and_all() {
        assert_eq!(parse_ok(&["2024"]), vec![Selector::Year(2024)]);
        assert_eq!(parse_ok(&["all"]), vec![Selector::All]);
        assert_eq!(parse_ok(&["ALL"]), vec![Selector::All]);
    }

    #[test]
    fn parse_mixed_years_and_ids_keeps_order() {
        assert_eq!(
            parse_ok(&["P3000R0", "2022", "P2300", "N4950"]),
            vec![
                Selector::Ids(nums(&["P3000R0"])),
                Selector::Year(2022),
                Selector::Ids(nums(&["P2300", "N4950"])),
            ]
        );
    }

    #[test]
    fn parse_rejects_empty() {
        assert!(matches!(parse_selectors(&[]), Err(Error::EmptySelection)));
    }

    #[test]
    fn parse_rejects_all_with_others() {
        for raws in [&["all", "2024"][..], &["P2300R4", "all"], &["all", "all"]] {
            assert!(
                matches!(parse_selectors(&args(raws)), Err(Error::AllWithOthers)),
                "{raws:?}"
            );
        }
    }

    #[test]
    fn parse_rejects_malformed_with_original_string() {
        for raw in ["P4003R", "202", "20245", "foo bar", "", "-1"] {
            match parse_selectors(&args(&["2024", raw])) {
                Err(Error::BadSelector(carried)) => assert_eq!(carried, raw),
                other => panic!("{raw:?} should fail with BadSelector, got {other:?}"),
            }
        }
    }

    #[test]
    fn resolve_all_lists_every_paper_sorted() {
        assert_eq!(
            resolve_ok(&["all"]),
            nums(&["N4950", "P2300R10", "P2300R4", "P2300R9", "P3000R0"])
        );
    }

    #[test]
    fn resolve_year_lists_that_year_and_empty_year_gives_nothing() {
        assert_eq!(
            resolve_ok(&["2024"]),
            nums(&["P2300R10", "P2300R9", "P3000R0"])
        );
        assert_eq!(resolve_ok(&["1999"]), Vec::<PaperNum>::new());
    }

    #[test]
    fn resolve_ids_resolves_latest_and_deduplicates_across_selectors() {
        assert_eq!(
            resolve_ok(&["P3000R0", "2022", "p2300", "P2300R10", "N4950"]),
            nums(&["N4950", "P2300R10", "P2300R4", "P3000R0"])
        );
    }

    #[test]
    fn resolve_unknown_id_fails() {
        let err = resolve(&store(), &parse_ok(&["P9999"]));
        assert!(
            matches!(
                err,
                Err(Error::Store(paperstore::Error::MissingPaper { .. }))
            ),
            "{err:?}"
        );
    }

    #[test]
    fn resolve_latest_picks_highest_numeric_revision() {
        let got = resolve_latest(&store(), &num("P2300")).map_err(|e| e.to_string());
        assert_eq!(got, Ok(num("P2300R10")));
    }

    #[test]
    fn resolve_latest_unrevisioned_number_resolves_to_itself() {
        let got = resolve_latest(&store(), &num("N4950")).map_err(|e| e.to_string());
        assert_eq!(got, Ok(num("N4950")));
    }

    #[test]
    fn resolve_latest_versioned_passes_through_when_stored() {
        let got = resolve_latest(&store(), &num("P2300R4")).map_err(|e| e.to_string());
        assert_eq!(got, Ok(num("P2300R4")));
    }

    #[test]
    fn resolve_latest_malformed_listed_revision_fails() {
        let mut bad = store();
        bad.1 = Some("P2300Rx".to_owned());
        let err = resolve_latest(&bad, &num("P2300"));
        match err {
            Err(Error::Store(paperstore::Error::BadPaperNum { 0: carried, .. })) => {
                assert_eq!(carried, "P2300Rx");
            }
            other => panic!("should fail with BadPaperNum, got {other:?}"),
        }
    }

    #[test]
    fn resolve_latest_missing_paper_fails() {
        for raw in ["P2300R5", "P9999", "N1"] {
            let err = resolve_latest(&store(), &num(raw));
            match err {
                Err(Error::Store(paperstore::Error::MissingPaper { 0: carried, .. })) => {
                    assert_eq!(carried, num(raw));
                }
                other => panic!("{raw:?} should fail with MissingPaper, got {other:?}"),
            }
        }
    }
}
