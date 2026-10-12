//! Blocks: the units chunks are built from (§5.5).

use super::headings::{is_bold_numbered, markdown_heading};
use super::{Document, LineNo, Section};

/// The kind of a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// A fenced code block, fence lines included.
    Code,
    /// A run of lines beginning with `|`.
    Table,
    /// A list item and its continuation lines.
    ListItem,
    /// One heading line.
    Heading,
    /// A run of other non-blank lines.
    Paragraph,
}

/// A run of whole lines of one kind. Blank lines belong to the block before
/// them, and blank lines at the start of a section to the first block, so a
/// section's blocks cover it with no gaps or overlaps. A section of only
/// blank lines is one paragraph block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    /// The block's kind.
    pub kind: BlockKind,
    /// The first line.
    pub first: LineNo,
    /// The last line, trailing blank lines included.
    pub last: LineNo,
}

fn is_blank(line: &str) -> bool {
    line.trim().is_empty()
}

fn is_table_line(line: &str) -> bool {
    line.trim_start().starts_with('|')
}

fn is_list_item(line: &str) -> bool {
    let trimmed = line.trim_start();
    let rest = if let Some(rest) = trimmed.strip_prefix(['-', '*', '+']) {
        rest
    } else {
        let digits = trimmed.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            return false;
        }
        match trimmed[digits..].strip_prefix(['.', ')']) {
            Some(rest) => rest,
            None => return false,
        }
    };
    rest.is_empty() || rest.starts_with([' ', '\t'])
}

fn is_heading(line: &str) -> bool {
    markdown_heading(line).is_some() || is_bold_numbered(line)
}

/// Whether line `index` starts a block other than a paragraph.
fn starts_other_block(doc: &Document<'_>, index: usize) -> bool {
    let line = doc.lines()[index];
    doc.fence_end(index).is_some() || is_heading(line) || is_table_line(line) || is_list_item(line)
}

/// Divides a section into blocks.
pub(super) fn split(doc: &Document<'_>, section: &Section) -> Vec<Block> {
    let start = section.first.zero_based();
    let end = section.last.zero_based();
    let lines = doc.lines();
    let mut blocks: Vec<Block> = Vec::new();
    let mut leading_blank: Option<usize> = None;
    let mut index = start;
    while index <= end {
        let line = lines[index];
        if is_blank(line) {
            if let Some(last) = blocks.last_mut() {
                last.last = LineNo::from_zero_based(index);
            } else if leading_blank.is_none() {
                leading_blank = Some(index);
            }
            index += 1;
            continue;
        }
        let continues =
            |next: usize| next <= end && !is_blank(lines[next]) && !starts_other_block(doc, next);
        let (kind, stop) = if let Some(close) = doc.fence_end(index) {
            (BlockKind::Code, close.min(end))
        } else if is_heading(line) {
            (BlockKind::Heading, index)
        } else if is_table_line(line) {
            let mut stop = index;
            while stop < end && is_table_line(lines[stop + 1]) {
                stop += 1;
            }
            (BlockKind::Table, stop)
        } else {
            let kind = if is_list_item(line) {
                BlockKind::ListItem
            } else {
                BlockKind::Paragraph
            };
            let mut stop = index;
            while continues(stop + 1) {
                stop += 1;
            }
            (kind, stop)
        };
        blocks.push(Block {
            kind,
            first: LineNo::from_zero_based(leading_blank.take().unwrap_or(index)),
            last: LineNo::from_zero_based(stop),
        });
        index = stop + 1;
    }
    if blocks.is_empty() && start <= end {
        blocks.push(Block {
            kind: BlockKind::Paragraph,
            first: section.first,
            last: section.last,
        });
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(md: &str) -> Vec<(BlockKind, usize, usize)> {
        let doc = Document::parse(md);
        let sections = doc.sections();
        sections
            .iter()
            .flat_map(|s| split(&doc, s))
            .map(|b| (b.kind, b.first.get(), b.last.get()))
            .collect()
    }

    #[test]
    fn classifies_every_kind() {
        use BlockKind::*;
        let md = "\n## Title\n\npara one\npara two\n\n- item\n  more\n- next\n\n| a |\n|---|\n\n```\nx\n\n## y\n```\n\ntail";
        assert_eq!(
            kinds(md),
            vec![
                (Heading, 1, 3),
                (Paragraph, 4, 6),
                (ListItem, 7, 8),
                (ListItem, 9, 10),
                (Table, 11, 13),
                (Code, 14, 19),
                (Paragraph, 20, 20),
            ]
        );
    }

    #[test]
    fn a_paragraph_stops_at_another_block() {
        use BlockKind::*;
        assert_eq!(
            kinds("text\n| t |\n### h\nmore\n```\nc\n```"),
            vec![
                (Paragraph, 1, 1),
                (Table, 2, 2),
                (Heading, 3, 3),
                (Paragraph, 4, 4),
                (Code, 5, 7)
            ]
        );
    }

    #[test]
    fn list_markers_need_whitespace() {
        assert!(is_list_item("- a"));
        assert!(is_list_item("  * a"));
        assert!(is_list_item("12) a"));
        assert!(is_list_item("3. a"));
        assert!(is_list_item("-"));
        assert!(!is_list_item("---"));
        assert!(!is_list_item("**bold**"));
        assert!(!is_list_item("3.14 is pi"));
    }
}
