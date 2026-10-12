//! Strict ISO 8601 calendar dates (`YYYY-MM-DD`) and mailing months
//! (`YYYY-MM`).

/// A valid calendar date.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Date {
    /// The year.
    pub year: i32,
    /// The month, 1 to 12.
    pub month: u8,
    /// The day of the month.
    pub day: u8,
}

impl Date {
    /// The date as `YYYY-MM-DD`.
    #[must_use]
    pub fn iso(self) -> String {
        format!("{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

fn digits<T: std::str::FromStr>(s: &str, len: usize) -> Option<T> {
    (s.len() == len && s.bytes().all(|b| b.is_ascii_digit()))
        .then(|| s.parse().ok())
        .flatten()
}

fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Parses a `YYYY-MM-DD` date, rejecting any other form and impossible
/// dates.
#[must_use]
pub fn parse_date(s: &str) -> Option<Date> {
    let mut parts = s.split('-');
    let (year, month, day) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    let year: i32 = digits(year, 4)?;
    let month: u8 = digits(month, 2)?;
    let day: u8 = digits(day, 2)?;
    if !(1..=12).contains(&month) || day == 0 || day > days_in_month(year, month) {
        return None;
    }
    Some(Date { year, month, day })
}

/// Parses a `YYYY-MM` mailing month as the first day of that month.
#[must_use]
pub fn parse_month(s: &str) -> Option<Date> {
    let (year, month) = s.split_once('-')?;
    let year: i32 = digits(year, 4)?;
    let month: u8 = digits(month, 2)?;
    (1..=12).contains(&month).then_some(Date {
        year,
        month,
        day: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_only_real_iso_dates() {
        assert_eq!(
            parse_date("2026-06-13").map(Date::iso),
            Some("2026-06-13".to_owned())
        );
        assert!(parse_date("2024-02-29").is_some());
        for bad in [
            "2023-02-29",
            "2013-4-25",
            "20130825",
            "2026-13-01",
            "2026-00-10",
            "2026-01-32",
            "2026‐01‐02",
            "Barry Revzin",
            "",
            "2026-01-02 ",
            "2026-01-02-03",
        ] {
            assert!(parse_date(bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn months_become_their_first_day() {
        assert_eq!(
            parse_month("2026-01").map(Date::iso),
            Some("2026-01-01".to_owned())
        );
        assert!(parse_month("2026-1").is_none());
        assert!(parse_month("2026-13").is_none());
    }
}
