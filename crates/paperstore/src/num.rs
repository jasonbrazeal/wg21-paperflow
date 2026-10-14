//! Validated WG21 paper numbers.

use std::fmt;
use std::str::FromStr;

use crate::error::{Error, Result};

/// A validated WG21 paper number, normalized to uppercase.
///
/// The shape is one or more ASCII letters, one or more digits, and an
/// optional `R` followed by one or more digits: `P4003R2`, `N4950`,
/// `D1234R0`. `R` is reserved as the revision marker, so it cannot appear
/// in the letter prefix. The revision must fit in a `u32`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PaperNum(String);

impl PaperNum {
    /// Parses and validates a paper number.
    ///
    /// Surrounding whitespace is trimmed and the number is ASCII-uppercased
    /// before validation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadPaperNum`] carrying the original string when the
    /// shape is invalid.
    pub fn parse(raw: &str) -> Result<Self> {
        let normalized = raw.trim().to_ascii_uppercase();
        if is_valid_shape(&normalized) {
            Ok(Self(normalized))
        } else {
            Err(Error::BadPaperNum(raw.to_owned()))
        }
    }

    /// Returns the uppercase paper number, as stored.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns the lowercase form, for filename stems.
    #[must_use]
    pub fn stem(&self) -> String {
        self.0.to_ascii_lowercase()
    }

    /// Splits into the paper number without revision and the revision.
    ///
    /// `P2300R10` gives `("P2300", Some(10))`; `N4950` gives
    /// `("N4950", None)`. Every revision of one paper number shares the
    /// first element, so it groups revisions. The letter prefix is kept, so
    /// `D2300R1` and `P2300R1` belong to different groups.
    #[must_use]
    pub fn split_revision(&self) -> (&str, Option<u32>) {
        match self.0.split_once('R') {
            Some((base, rev)) => (base, rev.parse().ok()),
            None => (&self.0, None),
        }
    }
}

impl fmt::Display for PaperNum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for PaperNum {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

fn is_valid_shape(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;
    // `R` is excluded from the prefix: it is reserved as the revision marker,
    // which is what rejects "PR2" (a revision with no base digits).
    if consume(bytes, &mut i, |b| b.is_ascii_alphabetic() && b != b'R') == 0 {
        return false;
    }
    if consume(bytes, &mut i, |b| b.is_ascii_digit()) == 0 {
        return false;
    }
    if i < bytes.len() && bytes[i] == b'R' {
        i += 1;
        let rev_start = i;
        if consume(bytes, &mut i, |b| b.is_ascii_digit()) == 0 {
            return false;
        }
        if s[rev_start..i].parse::<u32>().is_err() {
            return false;
        }
    }
    i == bytes.len()
}

fn consume(bytes: &[u8], i: &mut usize, pred: impl Fn(u8) -> bool) -> usize {
    let start = *i;
    while *i < bytes.len() && pred(bytes[*i]) {
        *i += 1;
    }
    *i - start
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_and_normalizes_valid_numbers() {
        for (raw, want) in [
            ("P4003R2", "P4003R2"),
            ("  p4003r2  ", "P4003R2"),
            ("n4950", "N4950"),
            ("D1234R0", "D1234R0"),
            ("\tp4003\n", "P4003"),
        ] {
            let num =
                PaperNum::parse(raw).unwrap_or_else(|e| panic!("{raw:?} should parse, got {e}"));
            assert_eq!(num.as_str(), want);
        }
    }

    #[test]
    fn parse_rejects_invalid_shapes_with_original_string() {
        for raw in [
            "", "   ", "4003", "P", "PR2", "P4003R", "P4003 R2", "P4003R2x", "R2",
        ] {
            match PaperNum::parse(raw) {
                Err(Error::BadPaperNum(carried)) => assert_eq!(carried, raw),
                other => panic!("{raw:?} should fail with BadPaperNum, got {other:?}"),
            }
        }
    }

    #[test]
    fn stem_lowercases() {
        let stem = PaperNum::parse("P4003R2").map(|n| n.stem()).ok();
        assert_eq!(stem.as_deref(), Some("p4003r2"));
    }

    #[test]
    fn split_revision_separates_number_and_revision() {
        for (raw, base, rev) in [
            ("P2300R10", "P2300", Some(10)),
            ("p2300r0", "P2300", Some(0)),
            ("N4950", "N4950", None),
            ("D1234R3", "D1234", Some(3)),
        ] {
            let num =
                PaperNum::parse(raw).unwrap_or_else(|e| panic!("{raw:?} should parse, got {e}"));
            assert_eq!(num.split_revision(), (base, rev), "{raw:?}");
        }
    }

    #[test]
    fn split_revision_groups_revisions_of_one_number() {
        let bases: Vec<String> = ["P2300R4", "P2300R10", "P2300"]
            .iter()
            .map(|raw| {
                PaperNum::parse(raw).map_or_else(
                    |e| panic!("{raw:?} should parse, got {e}"),
                    |n| n.split_revision().0.to_owned(),
                )
            })
            .collect();
        assert_eq!(bases, ["P2300", "P2300", "P2300"]);
    }

    #[test]
    fn parse_rejects_revision_overflowing_u32() {
        let raw = "P1R4294967296";
        match PaperNum::parse(raw) {
            Err(Error::BadPaperNum(carried)) => assert_eq!(carried, raw),
            other => panic!("{raw:?} should fail with BadPaperNum, got {other:?}"),
        }
        assert!(PaperNum::parse("P1R4294967295").is_ok());
    }

    #[test]
    fn from_str_parses_and_normalizes_via_str_parse() {
        let num = "  p4003r2 "
            .parse::<PaperNum>()
            .unwrap_or_else(|e| panic!("p4003r2 should parse, got {e}"));
        assert_eq!(num.as_str(), "P4003R2");
    }

    #[test]
    fn from_str_rejects_invalid_shapes_with_bad_paper_num() {
        for raw in ["", "4003", "PR2", "P4003R2x"] {
            match raw.parse::<PaperNum>() {
                Err(Error::BadPaperNum(carried)) => assert_eq!(carried, raw),
                other => panic!("{raw:?} should fail with BadPaperNum, got {other:?}"),
            }
        }
    }
}
