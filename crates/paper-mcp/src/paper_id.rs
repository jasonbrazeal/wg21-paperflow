//! Paper id parsing (design §5.1).
//!
//! A paper id is `P<number>R<revision>`, `D<number>R<revision>`,
//! `N<number>`, or `SD-<number>`, accepted in any letter case and always
//! stored and returned in lower case.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// The document series a paper id belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Series {
    /// Published proposals, `P<number>R<revision>`.
    P,
    /// Drafts, `D<number>R<revision>`.
    D,
    /// N papers, `N<number>`.
    N,
    /// Standing documents, `SD-<number>`.
    Sd,
}

impl Series {
    /// The lower-case series name stored in the `series` payload field.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Series::P => "p",
            Series::D => "d",
            Series::N => "n",
            Series::Sd => "sd",
        }
    }

    /// Whether papers in this series carry revisions.
    #[must_use]
    pub fn has_revisions(self) -> bool {
        matches!(self, Series::P | Series::D)
    }
}

/// Error for a string that is not a well-formed paper id.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("not a paper id: {0:?}")]
pub struct InvalidPaperId(pub String);

/// An exact, well-formed paper id, always lower case.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct PaperId {
    text: String,
    series: Series,
    number: u64,
    revision: Option<u64>,
}

impl PaperId {
    /// The lower-case id text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// The id's series.
    #[must_use]
    pub fn series(&self) -> Series {
        self.series
    }

    /// The paper number, without leading zeros.
    #[must_use]
    pub fn number(&self) -> u64 {
        self.number
    }

    /// The revision, present only for P and D papers.
    #[must_use]
    pub fn revision(&self) -> Option<u64> {
        self.revision
    }
}

impl fmt::Display for PaperId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

impl From<PaperId> for String {
    fn from(id: PaperId) -> Self {
        id.text
    }
}

impl TryFrom<String> for PaperId {
    type Error = InvalidPaperId;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl FromStr for PaperId {
    type Err = InvalidPaperId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match parse_any(s) {
            Some(Parsed::Exact(id)) => Ok(id),
            _ => Err(InvalidPaperId(s.to_owned())),
        }
    }
}

/// A `list_revisions` argument: an exact id, or a bare `P<number>` or
/// `D<number>` naming every revision of a paper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RevisionTarget {
    /// An exact paper id.
    Exact(PaperId),
    /// A P or D paper number with no revision.
    Bare {
        /// `Series::P` or `Series::D`.
        series: Series,
        /// The paper number.
        number: u64,
    },
}

impl FromStr for RevisionTarget {
    type Err = InvalidPaperId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match parse_any(s) {
            Some(Parsed::Exact(id)) => Ok(RevisionTarget::Exact(id)),
            Some(Parsed::Bare(series, number)) => Ok(RevisionTarget::Bare { series, number }),
            None => Err(InvalidPaperId(s.to_owned())),
        }
    }
}

enum Parsed {
    Exact(PaperId),
    Bare(Series, u64),
}

fn parse_digits(s: &str) -> Option<u64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

fn parse_any(input: &str) -> Option<Parsed> {
    if !input.is_ascii() {
        return None;
    }
    let text = input.to_ascii_lowercase();
    if let Some(rest) = text.strip_prefix("sd-") {
        let number = parse_digits(rest)?;
        return Some(Parsed::Exact(PaperId {
            text,
            series: Series::Sd,
            number,
            revision: None,
        }));
    }
    let (series, rest) = match text.as_bytes().first()? {
        b'p' => (Series::P, &text[1..]),
        b'd' => (Series::D, &text[1..]),
        b'n' => (Series::N, &text[1..]),
        _ => return None,
    };
    if series == Series::N {
        let number = parse_digits(rest)?;
        return Some(Parsed::Exact(PaperId {
            text,
            series,
            number,
            revision: None,
        }));
    }
    match rest.split_once('r') {
        Some((number, revision)) => {
            let number = parse_digits(number)?;
            let revision = parse_digits(revision)?;
            Some(Parsed::Exact(PaperId {
                text,
                series,
                number,
                revision: Some(revision),
            }))
        }
        None => Some(Parsed::Bare(series, parse_digits(rest)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact(s: &str) -> Result<PaperId, InvalidPaperId> {
        s.parse()
    }

    #[test]
    fn accepts_every_form_in_any_case() {
        for (input, text, series, number, revision) in [
            ("P3045R9", "p3045r9", Series::P, 3045, Some(9)),
            ("p0260r20", "p0260r20", Series::P, 260, Some(20)),
            ("D0263R1", "d0263r1", Series::D, 263, Some(1)),
            ("N5008", "n5008", Series::N, 5008, None),
            ("SD-1", "sd-1", Series::Sd, 1, None),
            ("sd-10", "sd-10", Series::Sd, 10, None),
        ] {
            let id = exact(input).unwrap_or_else(|e| panic!("{input}: {e}"));
            assert_eq!(id.as_str(), text);
            assert_eq!(id.series(), series);
            assert_eq!(id.number(), number);
            assert_eq!(id.revision(), revision);
        }
    }

    #[test]
    fn rejects_malformed_ids() {
        for input in [
            "",
            " p3045r9",
            "p3045r9 ",
            "[p3045r9]",
            "p3045r9.md",
            "https://wg21.link/p3045r9",
            "p3045",
            "d3045",
            "p3045r",
            "pr9",
            "n",
            "n12a",
            "sd1",
            "sd-",
            "x123",
            "p3045r9r1",
            "p３０４５r9",
        ] {
            assert!(exact(input).is_err(), "{input:?} should be rejected");
        }
    }

    #[test]
    fn bare_numbers_only_for_revision_targets() {
        assert_eq!(
            "P3045".parse::<RevisionTarget>(),
            Ok(RevisionTarget::Bare {
                series: Series::P,
                number: 3045
            })
        );
        assert_eq!(
            "d12".parse::<RevisionTarget>(),
            Ok(RevisionTarget::Bare {
                series: Series::D,
                number: 12
            })
        );
        assert!("n12".parse::<RevisionTarget>().is_ok());
        assert!("sd-1".parse::<RevisionTarget>().is_ok());
        assert!("p12 ".parse::<RevisionTarget>().is_err());
    }
}
