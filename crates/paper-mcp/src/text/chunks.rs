//! Retrieval chunks (§5.5).
//!
//! Every chunk's embedded text fits [`ChunkSettings::max_tokens`]. Blocks
//! are kept whole when they fit; a block that does not is split at the most
//! meaningful boundary that makes its parts fit: blank-line groups of a code
//! block or rows of an HTML table, then lines, then points inside a line
//! (sentence ends, tag ends, whitespace, and as a last resort any character).

use super::blocks::{self, Block, BlockKind};
use super::{ChunkNo, Document, LineNo, Section, SectionIndex};

/// Counts tokens the way the embedding model does.
pub trait TokenCounter {
    /// The number of tokens in `text`, without special tokens.
    fn count(&self, text: &str) -> usize;
}

/// Tokens the model adds around every input (`<s>` and `</s>`).
pub const SPECIAL_TOKENS: usize = 2;

/// Head room kept below the size limit, because summing the token counts of
/// pieces can differ slightly from counting the joined text.
const SAFETY_MARGIN_TOKENS: usize = 32;

/// Tokens reserved in every chunk besides the header and the text.
const RESERVED_TOKENS: usize = SPECIAL_TOKENS + SAFETY_MARGIN_TOKENS;

/// The context header is shortened to at most `max_tokens / HEADER_SHARE`.
const HEADER_SHARE: usize = 4;

/// A fence or table line is repeated in later parts of a split block only
/// when it costs at most `max_tokens / REPEAT_SHARE`.
const REPEAT_SHARE: usize = 8;

/// When cutting a line, the longest piece searched is this many bytes per
/// token of budget. A shorter piece than possible is still a valid piece.
const SEARCH_BYTES_PER_TOKEN: usize = 16;

/// Separator between the parts of a heading path.
const HEADING_PATH_SEPARATOR: &str = " › ";

/// Marks a shortened context header.
const ELLIPSIS: &str = "…";

/// The smallest accepted [`ChunkSettings::max_tokens`], which leaves room
/// for the shortened header, a repeated line, and some text.
pub const MIN_MAX_TOKENS: usize = 128;

/// Chunk sizes, all in tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkSettings {
    /// The size a chunk grows to before a new one starts, and the size of
    /// the pieces an over-long line is cut into.
    pub target_tokens: usize,
    /// The largest block repeated as the overlap at the start of the next
    /// chunk.
    pub overlap_max_tokens: usize,
    /// The most tokens in a chunk's embedded text, special tokens included.
    /// At least [`MIN_MAX_TOKENS`] and at least twice `target_tokens`.
    pub max_tokens: usize,
}

/// One retrieval chunk: a contiguous span of one section. It begins and
/// ends at line boundaries unless it holds part of a line too long for one
/// chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// The section the chunk belongs to.
    pub section: SectionIndex,
    /// The chunk's number within its section.
    pub number: ChunkNo,
    /// The first line, the overlap included.
    pub first: LineNo,
    /// The byte offset in `first` where the chunk starts.
    pub start_byte: usize,
    /// The last line.
    pub last: LineNo,
    /// The byte offset in `last` where the chunk ends, exclusive, or `None`
    /// when it runs to the end of that line.
    pub end_byte: Option<usize>,
    /// The section heading followed by the headings in force where the
    /// chunk's own content starts.
    pub heading_path: Vec<String>,
    /// The context header embedded before the text, shortened when long.
    pub header: String,
    /// For the second and later parts of a split code block or table, the
    /// fence or opening line repeated at the top of the embedded text.
    pub repeated_line: Option<LineNo>,
}

impl Chunk {
    /// The heading path joined for display.
    #[must_use]
    pub fn heading_path_text(&self) -> String {
        self.heading_path.join(HEADING_PATH_SEPARATOR)
    }

    /// The chunk's stored text: exactly its span, lines joined with `\n`.
    #[must_use]
    pub fn text(&self, doc: &Document<'_>) -> String {
        let lines = doc.lines();
        let (first, last) = (self.first.zero_based(), self.last.zero_based());
        let mut out = String::new();
        for index in first..=last {
            let Some(&line) = lines.get(index) else {
                break;
            };
            let start = if index == first { self.start_byte } else { 0 };
            let end = if index == last {
                self.end_byte.unwrap_or(line.len())
            } else {
                line.len()
            };
            if index > first {
                out.push('\n');
            }
            out.push_str(line.get(start..end).unwrap_or_default());
        }
        out
    }

    /// The text that is embedded: the context header, a blank line, any
    /// repeated fence or table line, then the chunk text.
    #[must_use]
    pub fn embed_text(&self, doc: &Document<'_>) -> String {
        let mut out = self.header.clone();
        out.push_str("\n\n");
        if let Some(line) = self.repeated_line.and_then(|l| doc.line(l)) {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str(&self.text(doc));
        out
    }
}

/// The full context header placed before a chunk's text when it is
/// embedded.
#[must_use]
pub fn context_header(
    paper_id: &str,
    title: &str,
    section: SectionIndex,
    heading_path: &str,
) -> String {
    let section_line = if heading_path.is_empty() {
        format!("Section {section}")
    } else {
        format!("Section {section}: {heading_path}")
    };
    format!("Paper: {paper_id}\nTitle: {title}\n{section_line}")
}

/// What identifies the paper in every chunk's context header.
#[derive(Debug, Clone, Copy)]
pub struct PaperLabel<'a> {
    /// The lower-case paper id.
    pub paper_id: &'a str,
    /// The paper title.
    pub title: &'a str,
}

/// A position in the paper: a 0-based line and a byte offset into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pos {
    line: usize,
    byte: usize,
}

/// A span `start..end` that fits in a chunk, with its token cost.
#[derive(Debug, Clone, Copy)]
struct Unit {
    start: Pos,
    end: Pos,
    tokens: usize,
}

/// A context header and the tokens it and the reserve take.
#[derive(Debug, Clone)]
struct Header {
    path: Vec<String>,
    text: String,
    cost: usize,
}

struct Chunker<'d, 'a, C: TokenCounter> {
    doc: &'d Document<'a>,
    section: &'d Section,
    settings: ChunkSettings,
    counter: &'d C,
    label: PaperLabel<'d>,
    blocks: Vec<Block>,
    tokens: Vec<usize>,
    chunks: Vec<Chunk>,
}

struct Pending {
    overlap: Option<usize>,
    first: usize,
    last: usize,
    tokens: usize,
    header: Header,
}

impl<C: TokenCounter> Chunker<'_, '_, C> {
    fn heading_path(&self, line: LineNo) -> Vec<String> {
        let mut path = Vec::new();
        if !self.section.heading.is_empty() {
            path.push(self.section.heading.clone());
        }
        let mut stack: Vec<(u8, &str)> = Vec::new();
        for heading in self.doc.headings() {
            if heading.line < self.section.first || heading.line > line {
                continue;
            }
            if Some(heading.line) == self.section.boundary {
                continue;
            }
            while stack
                .last()
                .is_some_and(|&(level, _)| level >= heading.level)
            {
                stack.pop();
            }
            stack.push((heading.level, &heading.text));
        }
        path.extend(stack.into_iter().map(|(_, text)| text.to_owned()));
        path
    }

    /// The header for content starting at `line`, cut short with an
    /// ellipsis when it would take more than its share of a chunk.
    fn header(&self, line: LineNo) -> Header {
        let path = self.heading_path(line);
        let full = context_header(
            self.label.paper_id,
            self.label.title,
            self.section.index,
            &path.join(HEADING_PATH_SEPARATOR),
        );
        let budget = self.settings.max_tokens / HEADER_SHARE;
        let text = if self.counter.count(&full) <= budget {
            full
        } else {
            let room = budget.saturating_sub(self.counter.count(ELLIPSIS));
            let keep = self.fit_len(&full, room);
            format!("{}{ELLIPSIS}", &full[..keep])
        };
        // The blank line between header and text is counted as one token.
        let cost = self.counter.count(&text) + 1 + RESERVED_TOKENS;
        Header { path, text, cost }
    }

    fn line_len(&self, line: usize) -> usize {
        self.doc.lines()[line].len()
    }

    fn line_end(&self, line: usize) -> Pos {
        Pos {
            line,
            byte: self.line_len(line),
        }
    }

    fn push(&mut self, start: Pos, end: Pos, header: &Header, repeated: Option<LineNo>) {
        let number = ChunkNo::from_zero_based(self.chunks.len());
        let end_byte = (end.byte < self.line_len(end.line)).then_some(end.byte);
        self.chunks.push(Chunk {
            section: self.section.index,
            number,
            first: LineNo::from_zero_based(start.line),
            start_byte: start.byte,
            last: LineNo::from_zero_based(end.line),
            end_byte,
            heading_path: header.path.clone(),
            header: header.text.clone(),
            repeated_line: repeated,
        });
    }

    fn flush(&mut self, pending: &Pending) {
        let first = self.blocks[pending.overlap.unwrap_or(pending.first)].first;
        let start = Pos {
            line: first.zero_based(),
            byte: 0,
        };
        let end = self.line_end(self.blocks[pending.last].last.zero_based());
        self.push(start, end, &pending.header, None);
    }

    fn overlap_for(&self, tail: Option<usize>) -> Option<usize> {
        tail.filter(|&b| self.tokens[b] <= self.settings.overlap_max_tokens)
    }

    fn start(&self, block: usize, tail: Option<usize>, header: Header) -> Pending {
        let overlap = self.overlap_for(tail).filter(|&o| {
            self.tokens[o] + self.tokens[block] + header.cost <= self.settings.max_tokens
        });
        Pending {
            overlap,
            first: block,
            last: block,
            tokens: overlap.map_or(0, |o| self.tokens[o]) + self.tokens[block],
            header,
        }
    }

    /// The length in bytes of the longest prefix of `text` that counts at
    /// most `budget` tokens, at least one character when `text` is not
    /// empty.
    fn fit_len(&self, text: &str, budget: usize) -> usize {
        let window =
            floor_char_boundary(text, budget.saturating_mul(SEARCH_BYTES_PER_TOKEN).max(1));
        if window == text.len() && self.counter.count(text) <= budget {
            return text.len();
        }
        let ends: Vec<usize> = text[..window]
            .char_indices()
            .map(|(i, c)| i + c.len_utf8())
            .collect();
        // Only an end whose prefix was counted and fits moves `low`, so the
        // answer fits even where counts are not monotonic.
        let (mut low, mut high) = (0, ends.len());
        while low < high {
            let mid = low + (high - low) / 2;
            if self.counter.count(&text[..ends[mid]]) <= budget {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        match low {
            0 => ends.first().copied().unwrap_or(0),
            n => ends[n - 1],
        }
    }

    /// The length of the next piece of `text`: the longest fitting prefix,
    /// shortened to its last sentence end, tag end, or whitespace when one
    /// lies in its second half.
    fn piece_len(&self, text: &str, budget: usize) -> usize {
        let end = self.fit_len(text, budget);
        if end == text.len() {
            return end;
        }
        let head = &text.as_bytes()[..end];
        let floor = end / 2;
        let last_break =
            |at: &dyn Fn(usize) -> bool| (floor + 1..end).rev().find(|&i| at(i)).map(|i| i + 1);
        let sentence = |i: usize| {
            i > 0 && head[i].is_ascii_whitespace() && matches!(head[i - 1], b'.' | b'!' | b'?')
        };
        let tag = |i: usize| head[i] == b'>';
        let space = |i: usize| head[i].is_ascii_whitespace();
        last_break(&sentence)
            .or_else(|| last_break(&tag))
            .or_else(|| last_break(&space))
            .unwrap_or(end)
    }

    /// Cuts one line into pieces of at most `budget` tokens plus its
    /// newline.
    fn line_pieces(&self, line: usize, budget: usize, out: &mut Vec<Unit>) {
        let text = self.doc.lines()[line];
        let mut start = 0;
        loop {
            let len = self.piece_len(&text[start..], budget);
            let end = start + len;
            out.push(Unit {
                start: Pos { line, byte: start },
                end: Pos { line, byte: end },
                tokens: self.counter.count(&text[start..end]) + 1,
            });
            start = end;
            if start >= text.len() {
                break;
            }
        }
    }

    /// Lines `first..=last` as one unit when they fit `limit`, else line by
    /// line, cutting lines that are still too long.
    fn line_units(&self, first: usize, last: usize, limit: usize, out: &mut Vec<Unit>) {
        let tokens = self
            .counter
            .count(&self.doc.lines()[first..=last].join("\n"))
            + 1;
        if tokens <= limit {
            out.push(Unit {
                start: Pos {
                    line: first,
                    byte: 0,
                },
                end: self.line_end(last),
                tokens,
            });
        } else if first < last {
            for line in first..=last {
                self.line_units(line, line, limit, out);
            }
        } else {
            let budget = self
                .settings
                .target_tokens
                .min(limit.saturating_sub(1))
                .max(1);
            self.line_pieces(first, budget, out);
        }
    }

    /// Divides a block into units of at most `limit` tokens, in order.
    fn units(&self, block: Block, limit: usize) -> Vec<Unit> {
        let mut out = Vec::new();
        for (first, last) in groups(self.doc.lines(), block) {
            self.line_units(first, last, limit, &mut out);
        }
        out
    }

    /// Splits a block too large for one chunk into parts packed toward the
    /// target size.
    fn split_block(&mut self, block: usize, tail: Option<usize>, header: &Header) {
        let max = self.settings.max_tokens;
        let target = self.settings.target_tokens;
        let this = self.blocks[block];
        let lines = self.doc.lines();
        let repeat = matches!(this.kind, BlockKind::Code | BlockKind::Table)
            .then(|| {
                (this.first.zero_based()..=this.last.zero_based())
                    .find(|&i| !lines[i].trim().is_empty())
            })
            .flatten()
            .map(|i| (LineNo::from_zero_based(i), self.counter.count(lines[i]) + 1))
            .filter(|&(_, tokens)| tokens <= max / REPEAT_SHARE);
        let first_limit = max.saturating_sub(header.cost);
        let rest_limit = first_limit.saturating_sub(repeat.map_or(0, |(_, t)| t));
        let units = self.units(this, rest_limit);

        // The overlap and its newline join the first part when both fit.
        let overlap = self.overlap_for(tail).filter(|&o| {
            units
                .first()
                .is_some_and(|u| self.tokens[o] + 1 + u.tokens <= first_limit)
        });
        let mut part_start = overlap.map(|o| Pos {
            line: self.blocks[o].first.zero_based(),
            byte: 0,
        });
        let mut used = overlap.map_or(0, |o| self.tokens[o] + 1);
        let mut current: Option<(Pos, Pos)> = None;
        let mut repeated = None;
        for unit in units {
            let limit = if repeated.is_some() {
                rest_limit
            } else {
                first_limit
            };
            if let Some((start, end)) = current {
                if used + unit.tokens <= target.min(limit) {
                    current = Some((start, unit.end));
                    used += unit.tokens;
                    continue;
                }
                self.push(part_start.take().unwrap_or(start), end, header, repeated);
                repeated = repeat.map(|(line, _)| line);
                used = 0;
            }
            current = Some((unit.start, unit.end));
            used += unit.tokens;
        }
        if let Some((start, end)) = current {
            self.push(part_start.unwrap_or(start), end, header, repeated);
        }
    }

    fn run(mut self) -> Vec<Chunk> {
        let max = self.settings.max_tokens;
        let target = self.settings.target_tokens;
        let mut pending: Option<Pending> = None;
        let mut tail: Option<usize> = None;
        for index in 0..self.blocks.len() {
            let block = self.blocks[index];
            let tokens = self.tokens[index];
            let header = self.header(block.first);
            if tokens + header.cost > max {
                if let Some(p) = pending.take() {
                    tail = Some(p.last);
                    self.flush(&p);
                }
                self.split_block(index, tail, &header);
                tail = None;
                continue;
            }
            pending = Some(match pending.take() {
                None => self.start(index, tail, header),
                Some(mut p)
                    if p.tokens + tokens <= target && p.tokens + tokens + p.header.cost <= max =>
                {
                    p.last = index;
                    p.tokens += tokens;
                    p
                }
                Some(p) => {
                    tail = Some(p.last);
                    self.flush(&p);
                    self.start(index, tail, header)
                }
            });
        }
        if let Some(p) = pending {
            self.flush(&p);
        }
        self.chunks
    }
}

/// The largest char boundary of `text` at or below `index`.
fn floor_char_boundary(text: &str, index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    (0..=index)
        .rev()
        .find(|&i| text.is_char_boundary(i))
        .unwrap_or(0)
}

/// The 0-based line ranges a block divides into before lines: blank-line
/// separated groups of a code block, rows of an HTML table, and the whole
/// block otherwise.
fn groups(lines: &[&str], block: Block) -> Vec<(usize, usize)> {
    let (start, end) = (block.first.zero_based(), block.last.zero_based());
    let starts_group = |i: usize| match block.kind {
        BlockKind::Code => lines[i - 1].trim().is_empty() && !lines[i].trim().is_empty(),
        BlockKind::Table => blocks::opens_html_row(lines[i]),
        BlockKind::ListItem | BlockKind::Heading | BlockKind::Paragraph => false,
    };
    let mut out = Vec::new();
    let mut first = start;
    for i in start + 1..=end {
        if starts_group(i) {
            out.push((first, i - 1));
            first = i;
        }
    }
    out.push((first, end));
    out
}

/// Chunks one section.
pub fn chunk_section<C: TokenCounter>(
    doc: &Document<'_>,
    section: &Section,
    settings: ChunkSettings,
    counter: &C,
    label: PaperLabel<'_>,
) -> Vec<Chunk> {
    let blocks = blocks::split(doc, section);
    let tokens = blocks
        .iter()
        .map(|b| counter.count(&doc.text(b.first, b.last)))
        .collect();
    Chunker {
        doc,
        section,
        settings,
        counter,
        label,
        blocks,
        tokens,
        chunks: Vec::new(),
    }
    .run()
}

/// Chunks every section of a paper, in order.
pub fn chunk_document<C: TokenCounter>(
    doc: &Document<'_>,
    sections: &[Section],
    settings: ChunkSettings,
    counter: &C,
    label: PaperLabel<'_>,
) -> Vec<Chunk> {
    sections
        .iter()
        .flat_map(|section| chunk_section(doc, section, settings, counter, label))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Counts whitespace-separated words, so tests can reason about sizes.
    pub(crate) struct Words;

    impl TokenCounter for Words {
        fn count(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
    }

    const LABEL: PaperLabel<'static> = PaperLabel {
        paper_id: "p1r0",
        title: "T",
    };

    fn settings(target: usize, overlap: usize, max: usize) -> ChunkSettings {
        ChunkSettings {
            target_tokens: target,
            overlap_max_tokens: overlap,
            max_tokens: max,
        }
    }

    fn chunks(md: &str, s: ChunkSettings) -> Vec<Chunk> {
        let doc = Document::parse(md);
        chunk_document(&doc, &doc.sections(), s, &Words, LABEL)
    }

    fn ranges(md: &str, s: ChunkSettings) -> Vec<(usize, usize, usize, Option<usize>)> {
        chunks(md, s)
            .into_iter()
            .map(|c| {
                (
                    c.section.get(),
                    c.first.get(),
                    c.last.get(),
                    c.repeated_line.map(LineNo::get),
                )
            })
            .collect()
    }

    fn words(n: usize) -> String {
        vec!["w"; n].join(" ")
    }

    fn assert_fits(md: &str, found: &[Chunk], max: usize) {
        let doc = Document::parse(md);
        for chunk in found {
            let tokens = Words.count(&chunk.embed_text(&doc)) + SPECIAL_TOKENS;
            assert!(tokens <= max, "{tokens} > {max}: {chunk:?}");
        }
    }

    #[test]
    fn packs_blocks_up_to_the_target_with_overlap() {
        let md = format!(
            "{}\n\n{}\n\n{}\n\n{}\n",
            words(3),
            words(3),
            words(3),
            words(3)
        );
        // Blocks are lines 1-2, 3-4, 5-6, 7. Target 6 fits two blocks; the
        // second chunk repeats block two as its overlap.
        assert_eq!(
            ranges(&md, settings(6, 5, 1000)),
            vec![(1, 1, 4, None), (1, 3, 6, None), (1, 5, 7, None)]
        );
    }

    #[test]
    fn large_overlap_blocks_are_not_repeated() {
        let md = format!("{}\n\n{}\n", words(8), words(8));
        assert_eq!(
            ranges(&md, settings(10, 5, 1000)),
            vec![(1, 1, 2, None), (1, 3, 3, None)]
        );
    }

    #[test]
    fn a_block_larger_than_the_target_runs_long_while_it_fits() {
        let md = format!("{}\n", words(50));
        assert_eq!(ranges(&md, settings(10, 5, 1000)), vec![(1, 1, 1, None)]);
    }

    #[test]
    fn chunks_never_cross_sections() {
        let md = "## A\na\n## B\nb\n## C\nc\n## D\nd\n";
        let found = ranges(md, settings(100, 50, 1000));
        assert_eq!(
            found,
            vec![
                (1, 1, 2, None),
                (2, 3, 4, None),
                (3, 5, 6, None),
                (4, 7, 8, None)
            ]
        );
    }

    #[test]
    fn oversized_code_blocks_split_with_the_fence_repeated() {
        let body: Vec<String> = (0..30).map(|_| words(4)).collect();
        let md = format!("```cpp\n{}\n```\n", body.join("\n"));
        let found = chunks(&md, settings(10, 5, 128));
        assert!(found.len() > 1);
        assert_eq!(found[0].first.get(), 1);
        assert_eq!(found[0].repeated_line, None);
        for pair in found.windows(2) {
            assert_eq!(pair[1].first.get(), pair[0].last.get() + 1);
            assert_eq!(pair[1].repeated_line.map(LineNo::get), Some(1));
        }
        assert_eq!(found.last().map(|c| c.last.get()), Some(32));
        assert_fits(&md, &found, 128);
        let doc = Document::parse(&md);
        let embedded = found[1].embed_text(&doc);
        assert!(embedded.starts_with("Paper: p1r0\nTitle: T\nSection 1\n\n```cpp\nw w w w"));
    }

    #[test]
    fn oversized_code_blocks_split_at_blank_lines_first() {
        let group = vec![words(8); 3].join("\n");
        let md = format!("```\n{group}\n\n{group}\n\n{group}\n\n{group}\n```\n");
        // Each group is 24 words, a newline, and maybe a fence, so target 30
        // holds one group: the groups are lines 1-5, 6-9, 10-13, and 14-17.
        let found = ranges(&md, settings(30, 0, 128));
        assert_eq!(
            found,
            vec![
                (1, 1, 5, None),
                (1, 6, 9, Some(1)),
                (1, 10, 13, Some(1)),
                (1, 14, 17, Some(1))
            ]
        );
    }

    #[test]
    fn oversized_paragraphs_split_at_lines() {
        let md = vec![words(30); 6].join("\n");
        let found = chunks(&md, settings(40, 0, 128));
        let lines: Vec<_> = found
            .iter()
            .map(|c| (c.first.get(), c.last.get()))
            .collect();
        assert_eq!(lines, vec![(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]);
        assert!(
            found
                .iter()
                .all(|c| c.start_byte == 0 && c.end_byte.is_none())
        );
        assert_fits(&md, &found, 128);
    }

    #[test]
    fn over_long_lines_split_inside_the_line_at_sentence_ends() {
        let sentence = format!("{}.", words(7));
        let md = vec![sentence; 40].join(" ");
        let doc = Document::parse(&md);
        let found = chunks(&md, settings(20, 0, 128));
        assert_fits(&md, &found, 128);
        assert!(found.len() > 1);
        assert!(
            found
                .iter()
                .all(|c| c.first.get() == 1 && c.last.get() == 1)
        );
        let joined: String = found.iter().map(|c| c.text(&doc)).collect();
        assert_eq!(joined, md);
        for piece in &found[..found.len() - 1] {
            assert!(piece.text(&doc).ends_with(". "), "{:?}", piece.text(&doc));
        }
    }

    #[test]
    fn a_line_with_no_break_points_is_cut_anywhere() {
        struct Bytes;
        impl TokenCounter for Bytes {
            fn count(&self, text: &str) -> usize {
                text.len()
            }
        }
        let md = "x".repeat(10_000);
        let doc = Document::parse(&md);
        let found = chunk_document(&doc, &doc.sections(), settings(50, 0, 400), &Bytes, LABEL);
        assert!(found.len() > 1);
        let joined: String = found.iter().map(|c| c.text(&doc)).collect();
        assert_eq!(joined, md);
        for chunk in &found {
            assert!(Bytes.count(&chunk.embed_text(&doc)) + SPECIAL_TOKENS <= 400);
        }
    }

    #[test]
    fn html_tables_split_at_rows_with_the_opening_line_repeated() {
        let row = format!("<tr>\n<td>{}</td>\n</tr>", words(6));
        let md = format!(
            "<table border=\"1\">\n{}\n</table>\n",
            vec![row; 20].join("\n")
        );
        let found = chunks(&md, settings(16, 0, 128));
        assert!(found.len() > 1);
        assert_fits(&md, &found, 128);
        for chunk in &found[1..] {
            assert_eq!(chunk.repeated_line.map(LineNo::get), Some(1));
            let doc = Document::parse(&md);
            assert!(
                chunk.text(&doc).starts_with("<tr>"),
                "{:?}",
                chunk.text(&doc)
            );
        }
    }

    #[test]
    fn long_heading_paths_are_shortened_in_the_header() {
        let md = format!("## {}\n{}\n## B\n## C\n## D\n", words(200), words(5));
        let found = chunks(&md, settings(20, 0, 128));
        assert_fits(&md, &found, 128);
        assert!(found[0].header.ends_with(ELLIPSIS));
        assert!(Words.count(&found[0].header) <= 128 / HEADER_SHARE);
        assert_eq!(found[0].heading_path.len(), 1);
    }

    #[test]
    fn heading_path_follows_nested_headings() {
        let md = "## 4 Design\nintro\n### 4.1 Units\nunits\n#### 4.1.1 Base\nbase\n### 4.2 Systems\nsys\n## 5\n## 6\n## 7\n";
        let found = chunks(md, settings(1, 0, 1000));
        let paths: Vec<String> = found.iter().map(Chunk::heading_path_text).collect();
        assert_eq!(
            paths[..7],
            [
                "4 Design",
                "4 Design",
                "4 Design › 4.1 Units",
                "4 Design › 4.1 Units",
                "4 Design › 4.1 Units › 4.1.1 Base",
                "4 Design › 4.1 Units › 4.1.1 Base",
                "4 Design › 4.2 Systems",
            ]
        );
    }

    #[test]
    fn context_header_format() {
        let section = SectionIndex::new(5).unwrap_or(SectionIndex::FIRST);
        assert_eq!(
            context_header(
                "p3045r9",
                "Quantities and units library",
                section,
                "4 Design › 4.2 Units"
            ),
            "Paper: p3045r9\nTitle: Quantities and units library\nSection 5: 4 Design › 4.2 Units"
        );
    }
}
