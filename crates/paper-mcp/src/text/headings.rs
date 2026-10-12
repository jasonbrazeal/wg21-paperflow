//! Code fences and headings (§5.3).

use super::LineNo;

/// A heading line outside any code fence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Heading {
    /// The heading's line.
    pub line: LineNo,
    /// 1 to 6 for markdown headings, 2 for bold-numbered lines.
    pub level: u8,
    /// The cleaned heading text.
    pub text: String,
    /// The markdown heading level, `None` for a bold-numbered line.
    pub markdown_level: Option<u8>,
}

/// Level given to bold-numbered headings.
const BOLD_NUMBERED_LEVEL: u8 = 2;

/// Minimum run of backticks or tildes that opens a fence.
const MIN_FENCE_RUN: usize = 3;

/// Marks the lines inside code fences, fence lines included, and records
/// where each fence that opens on a line ends (0-based).
pub(super) fn scan_fences(lines: &[&str]) -> (Vec<bool>, Vec<Option<usize>>) {
    let mut fenced = vec![false; lines.len()];
    let mut ends = vec![None; lines.len()];
    let mut index = 0;
    while index < lines.len() {
        let Some((marker, run)) = fence_opener(lines[index]) else {
            index += 1;
            continue;
        };
        let close = (index + 1..lines.len())
            .find(|&i| closes_fence(lines[i], marker, run))
            .unwrap_or(lines.len() - 1);
        for flag in &mut fenced[index..=close] {
            *flag = true;
        }
        ends[index] = Some(close);
        index = close + 1;
    }
    (fenced, ends)
}

/// The fence character and run length when `line` opens a code fence.
pub(super) fn fence_opener(line: &str) -> Option<(char, usize)> {
    let trimmed = line.trim_start();
    let marker = trimmed.chars().next()?;
    if marker != '`' && marker != '~' {
        return None;
    }
    let run = trimmed.chars().take_while(|&c| c == marker).count();
    (run >= MIN_FENCE_RUN).then_some((marker, run))
}

fn closes_fence(line: &str, marker: char, run: usize) -> bool {
    let trimmed = line.trim_start();
    let count = trimmed.chars().take_while(|&c| c == marker).count();
    count >= run && trimmed[count * marker.len_utf8()..].trim().is_empty()
}

/// The level and the text after the marks, when `line` is a markdown
/// heading: one to six `#` followed by whitespace and text.
pub(super) fn markdown_heading(line: &str) -> Option<(u8, &str)> {
    let hashes = line.bytes().take_while(|&b| b == b'#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = &line[hashes..];
    if !rest.starts_with([' ', '\t']) || rest.trim().is_empty() {
        return None;
    }
    u8::try_from(hashes).ok().map(|level| (level, rest))
}

/// Whether `line` is a bold-numbered heading such as
/// `2. **Motivation and Scope**`: a number, an optional `.` or `)`, then
/// text wrapped in `**`, and nothing else.
pub(super) fn is_bold_numbered(line: &str) -> bool {
    let line = line.trim_end();
    let digits = line.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return false;
    }
    let mut rest = &line[digits..];
    rest = rest
        .strip_prefix('.')
        .or_else(|| rest.strip_prefix(')'))
        .unwrap_or(rest);
    let rest = rest.trim_start();
    let Some(inner) = rest.strip_prefix("**").and_then(|r| r.strip_suffix("**")) else {
        return false;
    };
    !inner.trim().is_empty() && !inner.contains("**")
}

/// Whether `line` is a plain numbered line such as `3. Title`, the last
/// section fallback (§5.4).
pub(super) fn is_plain_numbered(line: &str) -> bool {
    let digits = line.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 {
        return false;
    }
    let Some(rest) = line[digits..].strip_prefix('.') else {
        return false;
    };
    rest.starts_with([' ', '\t']) && !rest.trim().is_empty()
}

pub(super) fn find_headings(lines: &[&str], fenced: &[bool]) -> Vec<Heading> {
    lines
        .iter()
        .enumerate()
        .filter(|(index, _)| !fenced[*index])
        .filter_map(|(index, line)| {
            let line_no = LineNo::from_zero_based(index);
            if let Some((level, _)) = markdown_heading(line) {
                Some(Heading {
                    line: line_no,
                    level,
                    text: clean_heading_text(line),
                    markdown_level: Some(level),
                })
            } else if is_bold_numbered(line) {
                Some(Heading {
                    line: line_no,
                    level: BOLD_NUMBERED_LEVEL,
                    text: clean_heading_text(line),
                    markdown_level: None,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Cleans a heading line for display: removes the `#` marks, hackmd anchor
/// links, link targets, and emphasis and code marks, and collapses
/// whitespace.
#[must_use]
pub fn clean_heading_text(line: &str) -> String {
    let mut text = line.trim();
    if let Some((_, rest)) = markdown_heading(text) {
        text = strip_closing_hashes(rest.trim());
    }
    let text = strip_links(text);
    let text: String = text.chars().filter(|&c| c != '*' && c != '`').collect();
    let text = text.replace("~~", "");
    let text = strip_underscore_emphasis(&text);
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_closing_hashes(text: &str) -> &str {
    let without = text.trim_end_matches('#');
    if without.len() == text.len() {
        return text;
    }
    if without.is_empty() {
        return without;
    }
    if without.ends_with([' ', '\t']) {
        without.trim_end()
    } else {
        text
    }
}

/// Drops hackmd anchor links (`[#Changelog](#Changelog)`) and replaces
/// other links and images with their text.
fn strip_links(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find('[') {
        let Some((label, after)) = parse_link(&rest[open..]) else {
            out.push_str(&rest[..=open]);
            rest = &rest[open + 1..];
            continue;
        };
        let before = &rest[..open];
        let before = before.strip_suffix('!').unwrap_or(before);
        out.push_str(before);
        if !(label.starts_with('#') && after.0.starts_with('#')) {
            out.push_str(label);
        }
        rest = after.1;
    }
    out.push_str(rest);
    out
}

/// Parses `[label](target)` at the start of `text`, returning the label and
/// the target with the remaining text.
fn parse_link(text: &str) -> Option<(&str, (&str, &str))> {
    let close = text.find(']')?;
    let label = &text[1..close];
    if label.contains('[') {
        return None;
    }
    let after = text[close + 1..].strip_prefix('(')?;
    let end = after.find(')')?;
    Some((label, (&after[..end], &after[end + 1..])))
}

/// Removes `_` and `__` used as emphasis delimiters at word edges, leaving
/// underscores inside identifiers such as `is_same_v` alone.
fn strip_underscore_emphasis(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut index = 0;
    while index < chars.len() {
        if chars[index] != '_' {
            out.push(chars[index]);
            index += 1;
            continue;
        }
        let run_end = (index..chars.len())
            .find(|&i| chars[i] != '_')
            .unwrap_or(chars.len());
        let before_word = index > 0 && is_word(chars[index - 1]);
        let after_word = run_end < chars.len() && is_word(chars[run_end]);
        if before_word && after_word {
            out.extend(&chars[index..run_end]);
        }
        index = run_end;
    }
    out
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fences(md: &str) -> Vec<bool> {
        let lines: Vec<&str> = md.split('\n').collect();
        scan_fences(&lines).0
    }

    #[test]
    fn indented_fences_open_and_close() {
        let md = "- item\n    ```cpp\n    ## not a heading\n    ```\n## heading";
        assert_eq!(fences(md), vec![false, true, true, true, false]);
    }

    #[test]
    fn closing_fence_needs_the_same_character_and_enough_marks() {
        let md = "````\n```\n~~~~\n````  \nafter";
        assert_eq!(fences(md), vec![true, true, true, true, false]);
        let md = "~~~\n```\n~~~ x\n~~~~\nafter";
        assert_eq!(fences(md), vec![true, true, true, true, false]);
    }

    #[test]
    fn unclosed_fence_runs_to_the_end() {
        assert_eq!(fences("a\n```\n## b\nc"), vec![false, true, true, true]);
    }

    #[test]
    fn two_backticks_do_not_open_a_fence() {
        assert_eq!(fences("``x``\n## b"), vec![false, false]);
    }

    #[test]
    fn recognises_markdown_and_bold_numbered_headings() {
        assert_eq!(markdown_heading("## Motivation").map(|h| h.0), Some(2));
        assert_eq!(markdown_heading("###### Six").map(|h| h.0), Some(6));
        assert!(markdown_heading("####### Seven").is_none());
        assert!(markdown_heading("#hashtag").is_none());
        assert!(markdown_heading("## ").is_none());
        assert!(markdown_heading(" ## indented").is_none());
        assert!(is_bold_numbered("2. **Motivation and Scope**"));
        assert!(is_bold_numbered("3) **Design**  "));
        assert!(is_bold_numbered("4 **Wording**"));
        assert!(!is_bold_numbered("2. **Motivation** and scope"));
        assert!(!is_bold_numbered("2. **A** and **B**"));
        assert!(!is_bold_numbered("**Motivation**"));
        assert!(is_plain_numbered("3. Title"));
        assert!(!is_plain_numbered("3.Title"));
        assert!(!is_plain_numbered("2026 was a year"));
    }

    #[test]
    fn cleans_heading_text() {
        for (raw, clean) in [
            ("## Motivation ##", "Motivation"),
            ("## C# support", "C# support"),
            ("## [#Changelog](#Changelog) Changelog", "Changelog"),
            (
                "### [Design](#design) of `std::views`",
                "Design of std::views",
            ),
            (
                "## **Bold** and *italic* and __strong__",
                "Bold and italic and strong",
            ),
            ("## `is_same_v` and _emph_", "is_same_v and emph"),
            ("2. **Motivation and Scope**", "2. Motivation and Scope"),
            ("#  Spaced    out  ", "Spaced out"),
            ("## ~~Old~~ New", "Old New"),
            ("## ![logo](x.png) Title", "logo Title"),
        ] {
            assert_eq!(clean_heading_text(raw), clean, "{raw}");
        }
    }

    #[test]
    fn headings_inside_fences_are_ignored() {
        let lines = vec!["## A", "```", "## B", "```", "1. **C**"];
        let (fenced, _) = scan_fences(&lines);
        let found: Vec<_> = find_headings(&lines, &fenced)
            .into_iter()
            .map(|h| (h.line.get(), h.level, h.text))
            .collect();
        assert_eq!(
            found,
            vec![(1, 2, "A".to_owned()), (5, 2, "1. C".to_owned())]
        );
    }
}
