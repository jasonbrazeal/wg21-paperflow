//! Sections (§5.4).

use super::headings::{clean_heading_text, is_bold_numbered, is_plain_numbered, markdown_heading};
use super::{Document, LineNo, SectionIndex};

/// The fewest sections a boundary rule must yield to be used.
const MIN_SECTIONS: usize = 4;

/// One section: a run of whole lines. Sections tile the paper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    /// The section's 1-based index.
    pub index: SectionIndex,
    /// The first line.
    pub first: LineNo,
    /// The last line.
    pub last: LineNo,
    /// The boundary line that starts the section, `None` for front matter
    /// and for a paper with no boundaries.
    pub boundary: Option<LineNo>,
    /// The cleaned boundary heading, empty when there is no boundary.
    pub heading: String,
}

#[derive(Clone, Copy)]
enum Rule {
    Primary,
    H3,
    H1,
    PlainNumbered,
}

impl Rule {
    fn matches(self, line: &str) -> bool {
        let level = markdown_heading(line).map(|(level, _)| level);
        match self {
            Rule::Primary => level == Some(2) || is_bold_numbered(line),
            Rule::H3 => level == Some(3),
            Rule::H1 => level == Some(1),
            Rule::PlainNumbered => is_plain_numbered(line),
        }
    }
}

const FALLBACKS: [Rule; 3] = [Rule::H3, Rule::H1, Rule::PlainNumbered];

fn boundaries(doc: &Document<'_>, rule: Rule) -> Vec<usize> {
    doc.lines()
        .iter()
        .enumerate()
        .filter(|&(index, line)| !doc.is_fenced(index) && rule.matches(line))
        .map(|(index, _)| index)
        .collect()
}

fn has_front_matter(doc: &Document<'_>, boundaries: &[usize]) -> bool {
    let end = boundaries.first().copied().unwrap_or(0);
    doc.lines()[..end]
        .iter()
        .any(|line| !line.trim().is_empty())
}

fn section_count(doc: &Document<'_>, boundaries: &[usize]) -> usize {
    if boundaries.is_empty() {
        return 1;
    }
    boundaries.len() + usize::from(has_front_matter(doc, boundaries))
}

pub(super) fn split(doc: &Document<'_>) -> Vec<Section> {
    let total = doc.line_count();
    if total == 0 {
        return Vec::new();
    }
    let primary = boundaries(doc, Rule::Primary);
    let chosen = if section_count(doc, &primary) >= MIN_SECTIONS {
        primary
    } else {
        FALLBACKS
            .iter()
            .map(|&rule| boundaries(doc, rule))
            .find(|found| section_count(doc, found) >= MIN_SECTIONS)
            .unwrap_or(primary)
    };

    let mut starts: Vec<(usize, Option<usize>)> = Vec::with_capacity(chosen.len() + 1);
    if chosen.is_empty() {
        starts.push((0, None));
    } else if has_front_matter(doc, &chosen) {
        starts.push((0, None));
        starts.extend(chosen.iter().map(|&b| (b, Some(b))));
    } else {
        starts.extend(chosen.iter().map(|&b| (b, Some(b))));
        starts[0].0 = 0;
    }

    starts
        .iter()
        .enumerate()
        .map(|(position, &(start, boundary))| {
            let end = starts
                .get(position + 1)
                .map_or(total - 1, |&(next, _)| next - 1);
            Section {
                index: SectionIndex::from_zero_based(position),
                first: LineNo::from_zero_based(start),
                last: LineNo::from_zero_based(end),
                boundary: boundary.map(LineNo::from_zero_based),
                heading: boundary
                    .map(|b| clean_heading_text(doc.lines()[b]))
                    .unwrap_or_default(),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::Document;

    fn summary(md: &str) -> Vec<(usize, usize, String)> {
        Document::parse(md)
            .sections()
            .into_iter()
            .map(|s| (s.first.get(), s.last.get(), s.heading))
            .collect()
    }

    fn owned(rows: &[(usize, usize, &str)]) -> Vec<(usize, usize, String)> {
        rows.iter().map(|&(a, b, h)| (a, b, h.to_owned())).collect()
    }

    #[test]
    fn h2_and_bold_numbered_are_primary() {
        let md = "front\n## A\na\n1. **B**\nb\n## C\n";
        assert_eq!(
            summary(md),
            owned(&[(1, 1, ""), (2, 3, "A"), (4, 5, "1. B"), (6, 6, "C")])
        );
    }

    #[test]
    fn blank_front_matter_joins_the_first_section() {
        let md = "\n\n## A\n## B\n## C\n## D\n";
        assert_eq!(
            summary(md),
            owned(&[(1, 3, "A"), (4, 4, "B"), (5, 5, "C"), (6, 6, "D")])
        );
    }

    #[test]
    fn falls_back_to_h3_then_h1_then_plain_numbers() {
        let h3 = "## Top\n### a\n### b\n### c\n### d\n";
        assert_eq!(summary(h3).len(), 5);
        assert_eq!(summary(h3)[1].2, "a");

        let h1 = "# a\n# b\n### x\n# c\n# d\n";
        assert_eq!(
            summary(h1),
            owned(&[(1, 1, "a"), (2, 3, "b"), (4, 4, "c"), (5, 5, "d")])
        );

        let plain = "Title\n1. One\n2. Two\n3. Three\n";
        assert_eq!(
            summary(plain),
            owned(&[
                (1, 1, ""),
                (2, 2, "1. One"),
                (3, 3, "2. Two"),
                (4, 4, "3. Three")
            ])
        );
    }

    #[test]
    fn keeps_primary_when_no_fallback_reaches_four() {
        let md = "x\n## A\n### a\n## B\n";
        assert_eq!(summary(md), owned(&[(1, 1, ""), (2, 3, "A"), (4, 4, "B")]));
    }

    #[test]
    fn front_matter_counts_toward_four() {
        let md = "front\n## A\n## B\n## C\n### x\n";
        assert_eq!(summary(md).len(), 4);
    }

    #[test]
    fn no_boundaries_is_one_section() {
        assert_eq!(summary("just\ntext\n"), owned(&[(1, 2, "")]));
        assert!(summary("").is_empty());
    }

    #[test]
    fn fenced_headings_are_not_boundaries() {
        let md = "## A\n```\n## X\n```\n## B\n## C\n## D\n";
        assert_eq!(
            summary(md),
            owned(&[(1, 4, "A"), (5, 5, "B"), (6, 6, "C"), (7, 7, "D")])
        );
    }
}
