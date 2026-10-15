//! Validated WG21 paper numbers.

use std::fmt;
use std::str::FromStr;

use crate::error::{Error, Result};

/// A validated WG21 paper number, normalized to uppercase.
///
/// The shape is one or more ASCII letters, one or more digits, and an
/// optional `R` followed by one or more digits: `P4003R2`, `N4950`,
/// `D1234R0`. `R` is reserved as the revision marker, so it cannot appear
/// in the letter prefix. Standing documents use the separate shape `SD-`
/// followed by one or more digits, with no revision: `SD-1`, `SD-5`.
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

const SD_PREFIX: &str = "SD-";

fn is_valid_shape(s: &str) -> bool {
    if let Some(digits) = s.strip_prefix(SD_PREFIX) {
        return !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit());
    }
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
        if consume(bytes, &mut i, |b| b.is_ascii_digit()) == 0 {
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
    fn parse_accepts_and_normalizes_sd_numbers() {
        for (raw, want, stem) in [
            ("SD-1", "SD-1", "sd-1"),
            ("sd-5", "SD-5", "sd-5"),
            ("  Sd-12 ", "SD-12", "sd-12"),
        ] {
            let num =
                PaperNum::parse(raw).unwrap_or_else(|e| panic!("{raw:?} should parse, got {e}"));
            assert_eq!(num.as_str(), want);
            assert_eq!(num.to_string(), want);
            assert_eq!(num.stem(), stem);
        }
    }

    #[test]
    fn parse_rejects_invalid_sd_shapes_with_original_string() {
        for raw in [
            "SD-", "SD-R1", "SD-1R2", "SD--1", "SD-1x", "SD-1-2", "P-1", "-1", "SD -1",
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
