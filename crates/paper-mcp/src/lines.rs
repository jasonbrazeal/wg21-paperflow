//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Line-range access to paper Markdown.
//!
//! Lines are those of [`str::lines`], the same split the chunker uses, so a
//! chunk's `start_line`/`end_line` select exactly its text.

/// Resolve optional slice bounds over `total` lines with Python slice
/// semantics.
///
/// Bounds are 0-based with `end` exclusive. An omitted `start` means the
/// first line and an omitted `end` means through the last line. Negative
/// bounds count from the end, and out-of-range bounds clamp to `0..=total`.
/// Returns `(start, end)` with `start <= end <= total`; bounds that cross
/// (Python's empty `lines[5:2]`) resolve to the empty range at `start`.
#[must_use]
pub fn resolve_slice(total: usize, start: Option<i64>, end: Option<i64>) -> (usize, usize) {
    let start = start.map_or(0, |bound| resolve_bound(total, bound));
    let end = end.map_or(total, |bound| resolve_bound(total, bound));
    (start, end.max(start))
}

/// Resolve one bound: negative values count back from `total`, and the
/// result is clamped to `0..=total`.
fn resolve_bound(total: usize, bound: i64) -> usize {
    let magnitude = usize::try_from(bound.unsigned_abs()).unwrap_or(usize::MAX);
    if bound < 0 {
        total.saturating_sub(magnitude)
    } else {
        magnitude.min(total)
    }
}

/// Return the text of lines `start..end` of `md`, joined with `\n`, and the
/// paper's total line count.
///
/// Bounds are resolved by [`resolve_slice`]; the returned tuple is
/// `(start, end, total_lines, text)`.
#[must_use]
pub fn slice_lines(
    md: &str,
    start: Option<i64>,
    end: Option<i64>,
) -> (usize, usize, usize, String) {
    let lines: Vec<&str> = md.lines().collect();
    let (start, end) = resolve_slice(lines.len(), start, end);
    (start, end, lines.len(), lines[start..end].join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOTAL: usize = 5;

    /// `(start, end, expected)`, with `expected` the indices Python's
    /// `list(range(5))[start:end]` yields.
    const CASES: [(Option<i64>, Option<i64>, &[usize]); 22] = [
        (None, None, &[0, 1, 2, 3, 4]),
        (Some(0), None, &[0, 1, 2, 3, 4]),
        (Some(2), None, &[2, 3, 4]),
        (None, Some(3), &[0, 1, 2]),
        (Some(1), Some(4), &[1, 2, 3]),
        (Some(-2), None, &[3, 4]),
        (None, Some(-1), &[0, 1, 2, 3]),
        (Some(-3), Some(-1), &[2, 3]),
        (Some(-5), None, &[0, 1, 2, 3, 4]),
        (Some(-100), None, &[0, 1, 2, 3, 4]),
        (None, Some(-100), &[]),
        (Some(5), None, &[]),
        (Some(100), None, &[]),
        (None, Some(100), &[0, 1, 2, 3, 4]),
        (Some(3), Some(100), &[3, 4]),
        (Some(-100), Some(100), &[0, 1, 2, 3, 4]),
        (Some(4), Some(2), &[]),
        (Some(-1), Some(-3), &[]),
        (Some(3), Some(-3), &[]),
        (Some(2), Some(2), &[]),
        (Some(i64::MIN), Some(i64::MAX), &[0, 1, 2, 3, 4]),
        (Some(i64::MAX), Some(i64::MIN), &[]),
    ];

    #[test]
    fn resolve_slice_matches_python_slicing() {
        let items: Vec<usize> = (0..TOTAL).collect();
        for (start, end, expected) in CASES {
            let (s, e) = resolve_slice(TOTAL, start, end);
            assert!(s <= e && e <= TOTAL, "{start:?}:{end:?} -> {s}..{e}");
            assert_eq!(&items[s..e], expected, "{start:?}:{end:?}");
        }
    }

    #[test]
    fn crossed_bounds_resolve_to_empty_range_at_start() {
        assert_eq!(resolve_slice(TOTAL, Some(4), Some(2)), (4, 4));
        assert_eq!(resolve_slice(TOTAL, Some(-1), Some(-3)), (4, 4));
    }

    #[test]
    fn empty_document_resolves_to_empty_range() {
        for (start, end, _) in CASES {
            assert_eq!(resolve_slice(0, start, end), (0, 0), "{start:?}:{end:?}");
        }
    }

    #[test]
    fn slice_lines_joins_selected_lines() {
        let md = "a\nb\r\nc\nd\n";
        assert_eq!(
            slice_lines(md, None, None),
            (0, 4, 4, "a\nb\nc\nd".to_owned())
        );
        assert_eq!(
            slice_lines(md, Some(1), Some(-1)),
            (1, 3, 4, "b\nc".to_owned())
        );
        assert_eq!(slice_lines(md, Some(3), Some(1)), (3, 3, 4, String::new()));
    }
}
