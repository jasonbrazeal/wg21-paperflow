//! Resolving `first` and `last` arguments (§6.5).

use super::LineNo;

/// An inclusive 1-based range after resolving negative and out-of-range
/// values. It is empty when `first > last`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InclusiveRange {
    /// The resolved first position.
    pub first: usize,
    /// The resolved last position.
    pub last: usize,
}

impl InclusiveRange {
    /// The bounds as line numbers, or `None` when the range is empty.
    #[must_use]
    pub fn bounds(self) -> Option<(LineNo, LineNo)> {
        if self.first > self.last {
            return None;
        }
        Some((LineNo::new(self.first)?, LineNo::new(self.last)?))
    }

    /// Whether the range selects nothing.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.bounds().is_none()
    }
}

/// Resolves `first` and `last` over `len` items the way Lua's
/// `string.sub(s, i, j)` does: both inclusive, negative values count from
/// the end, `first` defaults to 1 and `last` to -1, and out-of-range values
/// clamp.
#[must_use]
pub fn resolve_range(first: Option<i64>, last: Option<i64>, len: usize) -> InclusiveRange {
    let len = i64::try_from(len).unwrap_or(i64::MAX);
    let first = first.unwrap_or(1);
    let last = last.unwrap_or(-1);
    let first = if first < 0 {
        (len + first + 1).max(1)
    } else {
        first.max(1)
    };
    let last = if last < 0 {
        len + last + 1
    } else {
        last.min(len)
    };
    let first = usize::try_from(first).unwrap_or(usize::MAX);
    let last = usize::try_from(last.max(0)).unwrap_or(0);
    InclusiveRange { first, last }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolve(first: Option<i64>, last: Option<i64>) -> Option<(usize, usize)> {
        resolve_range(first, last, 100)
            .bounds()
            .map(|(a, b)| (a.get(), b.get()))
    }

    #[test]
    fn follows_the_get_lines_table() {
        assert_eq!(resolve(None, None), Some((1, 100)));
        assert_eq!(resolve(Some(100), None), Some((100, 100)));
        assert_eq!(resolve(Some(1), Some(50)), Some((1, 50)));
        assert_eq!(resolve(Some(-20), None), Some((81, 100)));
        assert_eq!(resolve(Some(11), Some(-11)), Some((11, 90)));
    }

    #[test]
    fn clamps_and_empties_like_string_sub() {
        assert_eq!(resolve(Some(0), Some(2)), Some((1, 2)));
        assert_eq!(resolve(Some(-500), Some(3)), Some((1, 3)));
        assert_eq!(resolve(Some(90), Some(500)), Some((90, 100)));
        assert_eq!(resolve(Some(101), None), None);
        assert_eq!(resolve(Some(5), Some(4)), None);
        assert_eq!(resolve(None, Some(-101)), None);
        assert_eq!(resolve(None, Some(0)), None);
        assert!(resolve_range(None, None, 0).is_empty());
    }
}
