// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Line ranges of a paper's Markdown, addressed like Python slices.

use crate::error::{Error, Result};

/// A run of source lines cut from a Markdown document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineSlice {
    /// First line of the slice, 1-based and inclusive.
    pub start: usize,
    /// Line after the slice, 1-based and exclusive. Equal to `start` when
    /// the slice is empty.
    pub end: usize,
    /// Number of lines in the whole document.
    pub total_lines: usize,
    /// Lines `start..end` joined with `\n`, like chunk text.
    pub text: String,
}

/// Cuts lines out of `md` with 1-based Python slice semantics.
///
/// `start` is inclusive and `end` exclusive, both 1-based. A negative bound
/// counts from the end, so `-1` is the last line. Bounds past either end are
/// clamped, a missing `start` means the first line and a missing `end` means
/// past the last line, and an `end` of `0` or at or before `start` gives an
/// empty slice. Lines are split as by [`str::lines`].
///
/// # Errors
///
/// Returns [`Error::InvalidLineRange`] when `start` is `0`, which no 1-based
/// line has, or when both bounds are missing.
pub fn slice_lines(md: &str, start: Option<i64>, end: Option<i64>) -> Result<LineSlice> {
    if start == Some(0) {
        return Err(Error::InvalidLineRange(
            "lines are 1-based, so start cannot be 0",
        ));
    }
    if start.is_none() && end.is_none() {
        return Err(Error::InvalidLineRange(
            "give start, end or both; get_paper returns the whole paper",
        ));
    }
    let lines: Vec<&str> = md.lines().collect();
    let total = lines.len();
    let lo = start.map_or(0, |s| index(s, total));
    let hi = end.map_or(total, |e| index(e, total)).max(lo);
    Ok(LineSlice {
        start: lo + 1,
        end: hi + 1,
        total_lines: total,
        text: lines[lo..hi].join("\n"),
    })
}

/// Converts a nonzero 1-based or negative bound to a clamped 0-based index;
/// `0` maps to index 0.
fn index(bound: i64, total: usize) -> usize {
    let magnitude = usize::try_from(bound.unsigned_abs()).unwrap_or(usize::MAX);
    if bound < 0 {
        total.saturating_sub(magnitude)
    } else {
        magnitude.saturating_sub(1).min(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Applies Python's 0-based slice rules for `lines[start:end]`.
    fn python_slice(len: i64, start: Option<i64>, end: Option<i64>) -> (i64, i64) {
        let norm = |i: i64| if i < 0 { (i + len).max(0) } else { i.min(len) };
        let lo = start.map_or(0, norm);
        let hi = end.map_or(len, norm);
        (lo, hi.max(lo))
    }

    /// Translates a 1-based bound to the 0-based bound Python would take.
    /// An `end` of `0` lies before the first line; `Some(0)` stands for it.
    fn to_python(bound: Option<i64>) -> Option<i64> {
        bound.map(|b| if b > 0 { b - 1 } else { b })
    }

    fn doc(len: i64) -> String {
        (1..=len)
            .map(|i| format!("line {i}"))
            .map(|l| l + "\n")
            .collect()
    }

    fn bounds(len: i64) -> impl Iterator<Item = Option<i64>> + Clone {
        std::iter::once(None).chain((-len - 3..=len + 3).map(Some))
    }

    #[test]
    fn matches_python_slice_oracle() {
        for len in 0..=6_i64 {
            let md = doc(len);
            let lines: Vec<&str> = md.lines().collect();
            for start in bounds(len) {
                for end in bounds(len) {
                    let got = slice_lines(&md, start, end);
                    if start.is_none() && end.is_none() || start == Some(0) {
                        assert!(
                            matches!(got, Err(Error::InvalidLineRange(_))),
                            "len={len} start={start:?} end={end:?}: {got:?}"
                        );
                        continue;
                    }
                    let end_py = if end == Some(0) {
                        Some(0)
                    } else {
                        to_python(end)
                    };
                    let (lo, hi) = python_slice(len, to_python(start), end_py);
                    let lo_u = usize::try_from(lo).unwrap_or_default();
                    let hi_u = usize::try_from(hi).unwrap_or_default();
                    let want = LineSlice {
                        start: lo_u + 1,
                        end: hi_u + 1,
                        total_lines: lines.len(),
                        text: lines[lo_u..hi_u].join("\n"),
                    };
                    match got {
                        Ok(slice) => {
                            assert_eq!(slice, want, "len={len} start={start:?} end={end:?}");
                        }
                        Err(e) => panic!("len={len} start={start:?} end={end:?}: {e}"),
                    }
                }
            }
        }
    }

    #[test]
    fn last_line_without_trailing_newline_counts() {
        let got = slice_lines("a\nb\nc", Some(-1), None).map_err(|e| e.to_string());
        assert_eq!(
            got,
            Ok(LineSlice {
                start: 3,
                end: 4,
                total_lines: 3,
                text: "c".to_owned(),
            })
        );
    }

    #[test]
    fn start_zero_message_names_the_one_based_rule() {
        let err = slice_lines("a\n", Some(0), Some(2))
            .err()
            .map(|e| e.to_string());
        assert!(
            err.as_deref().is_some_and(|m| m.contains("1-based")),
            "{err:?}"
        );
    }
}
