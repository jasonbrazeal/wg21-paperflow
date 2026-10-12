//! Retrieval chunks (§5.5).

use super::blocks::{self, Block, BlockKind};
use super::{ChunkNo, Document, LineNo, Section, SectionIndex};

/// Counts tokens the way the embedding model does.
pub trait TokenCounter {
    /// The number of tokens in `text`, without special tokens.
    fn count(&self, text: &str) -> usize;
}

/// Tokens the model adds around every input (`<s>` and `</s>`).
const SPECIAL_TOKENS: usize = 2;

/// Head room kept below the model's input limit, because summing per-line
/// token counts can differ slightly from counting the joined text.
const SAFETY_MARGIN_TOKENS: usize = 32;

/// Separator between the parts of a heading path.
const HEADING_PATH_SEPARATOR: &str = " › ";

/// Chunk sizes, all in tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkSettings {
    /// The size a chunk grows to before a new one starts.
    pub target_tokens: usize,
    /// The largest block repeated as the overlap at the start of the next
    /// chunk.
    pub overlap_max_tokens: usize,
    /// The embedding model's input limit.
    pub max_input_tokens: usize,
}

/// One retrieval chunk: whole lines of one section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// The section the chunk belongs to.
    pub section: SectionIndex,
    /// The chunk's number within its section.
    pub number: ChunkNo,
    /// The first line, the overlap included.
    pub first: LineNo,
    /// The last line.
    pub last: LineNo,
    /// The section heading followed by the headings in force where the
    /// chunk's own content starts.
    pub heading_path: Vec<String>,
    /// For the second and later parts of a split code block or table, the
    /// fence or header line repeated at the top of the embedded text.
    pub repeated_line: Option<LineNo>,
    /// Whether the embedded text is still longer than the model accepts,
    /// which happens only for a single block or line that cannot be split.
    pub oversized: bool,
}

impl Chunk {
    /// The heading path joined for display.
    #[must_use]
    pub fn heading_path_text(&self) -> String {
        self.heading_path.join(HEADING_PATH_SEPARATOR)
    }

    /// The chunk's stored text: exactly its lines.
    #[must_use]
    pub fn text(&self, doc: &Document<'_>) -> String {
        doc.text(self.first, self.last)
    }

    /// The text that is embedded: the context header, a blank line, any
    /// repeated fence or header line, then the chunk text.
    #[must_use]
    pub fn embed_text(&self, doc: &Document<'_>, paper_id: &str, title: &str) -> String {
        let mut out = context_header(paper_id, title, self.section, &self.heading_path_text());
        out.push_str("\n\n");
        if let Some(line) = self.repeated_line.and_then(|l| doc.line(l)) {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str(&self.text(doc));
        out
    }
}

/// The context header placed before a chunk's text when it is embedded.
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

    fn header_tokens(&self, path: &[String]) -> usize {
        let header = context_header(
            self.label.paper_id,
            self.label.title,
            self.section.index,
            &path.join(HEADING_PATH_SEPARATOR),
        );
        self.counter.count(&header) + SPECIAL_TOKENS + SAFETY_MARGIN_TOKENS
    }

    fn push(
        &mut self,
        first: LineNo,
        last: LineNo,
        path: Vec<String>,
        repeated: Option<LineNo>,
        oversized: bool,
    ) {
        let number = ChunkNo::from_zero_based(self.chunks.len());
        self.chunks.push(Chunk {
            section: self.section.index,
            number,
            first,
            last,
            heading_path: path,
            repeated_line: repeated,
            oversized,
        });
    }

    fn flush(&mut self, pending: &Pending) {
        let content_first = self.blocks[pending.first].first;
        let path = self.heading_path(content_first);
        let oversized = pending.tokens + self.header_tokens(&path) > self.settings.max_input_tokens;
        let first = pending
            .overlap
            .map_or(content_first, |o| self.blocks[o].first);
        self.push(first, self.blocks[pending.last].last, path, None, oversized);
    }

    fn overlap_for(&self, tail: Option<usize>) -> Option<usize> {
        tail.filter(|&b| self.tokens[b] <= self.settings.overlap_max_tokens)
    }

    fn start(&self, block: usize, tail: Option<usize>, header: usize) -> Pending {
        let overlap = self.overlap_for(tail).filter(|&o| {
            self.tokens[o] + self.tokens[block] + header <= self.settings.max_input_tokens
        });
        Pending {
            overlap,
            first: block,
            last: block,
            tokens: overlap.map_or(0, |o| self.tokens[o]) + self.tokens[block],
        }
    }

    /// Splits an oversized code block or table at line boundaries.
    fn split_block(&mut self, block: usize, tail: Option<usize>) {
        let Block { first, last, .. } = self.blocks[block];
        let path = self.heading_path(first);
        let header = self.header_tokens(&path);
        let lines = self.doc.lines();
        let start = first.zero_based();
        let end = last.zero_based();
        let repeat = (start..=end)
            .find(|&i| !lines[i].trim().is_empty())
            .unwrap_or(start);
        let repeat_tokens = self.counter.count(lines[repeat]) + 1;
        let limit = self.settings.max_input_tokens;

        let mut overlap = self.overlap_for(tail);
        let mut part_start = start;
        let mut used = overlap.map_or(0, |o| self.tokens[o]);
        let mut extra = 0;
        let mut index = start;
        while index <= end {
            let cost = self.counter.count(lines[index]) + 1;
            let budget = limit.saturating_sub(header + extra);
            if index > part_start && used + cost > budget {
                self.emit_part(
                    part_start,
                    index - 1,
                    overlap.take(),
                    &path,
                    extra,
                    used,
                    header,
                    repeat,
                    start,
                );
                part_start = index;
                used = 0;
                extra = repeat_tokens;
                continue;
            }
            if index == part_start && used + cost > budget && overlap.is_some() {
                overlap = None;
                used = 0;
                continue;
            }
            used += cost;
            index += 1;
        }
        self.emit_part(
            part_start, end, overlap, &path, extra, used, header, repeat, start,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_part(
        &mut self,
        from: usize,
        to: usize,
        overlap: Option<usize>,
        path: &[String],
        extra: usize,
        used: usize,
        header: usize,
        repeat: usize,
        block_start: usize,
    ) {
        let first = overlap.map_or(LineNo::from_zero_based(from), |o| self.blocks[o].first);
        let repeated = (from != block_start).then(|| LineNo::from_zero_based(repeat));
        let oversized = used + extra + header > self.settings.max_input_tokens;
        self.push(
            first,
            LineNo::from_zero_based(to),
            path.to_vec(),
            repeated,
            oversized,
        );
    }

    fn run(mut self) -> Vec<Chunk> {
        let mut pending: Option<Pending> = None;
        let mut tail: Option<usize> = None;
        for index in 0..self.blocks.len() {
            let block = self.blocks[index];
            let tokens = self.tokens[index];
            let header = self.header_tokens(&self.heading_path(block.first));
            let splittable = matches!(block.kind, BlockKind::Code | BlockKind::Table);
            if splittable && tokens + header > self.settings.max_input_tokens {
                if let Some(p) = pending.take() {
                    tail = Some(p.last);
                    self.flush(&p);
                }
                self.split_block(index, tail);
                tail = None;
                continue;
            }
            pending = Some(match pending.take() {
                None => self.start(index, tail, header),
                Some(mut p) if p.tokens + tokens <= self.settings.target_tokens => {
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
            max_input_tokens: max,
        }
    }

    fn ranges(md: &str, s: ChunkSettings) -> Vec<(usize, usize, usize, Option<usize>)> {
        let doc = Document::parse(md);
        chunk_document(&doc, &doc.sections(), s, &Words, LABEL)
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
    fn a_block_larger_than_the_target_runs_long() {
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
        // Header "Paper: p1r0 / Title: T / Section 1" is 6 words, plus 34
        // reserved, so a 100-token limit leaves 60 for text.
        let doc = Document::parse(&md);
        let chunks = chunk_document(&doc, &doc.sections(), settings(10, 5, 100), &Words, LABEL);
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].first.get(), 1);
        assert_eq!(chunks[0].repeated_line, None);
        for pair in chunks.windows(2) {
            assert_eq!(pair[1].first.get(), pair[0].last.get() + 1);
            assert_eq!(pair[1].repeated_line.map(LineNo::get), Some(1));
        }
        assert_eq!(chunks.last().map(|c| c.last.get()), Some(32));
        assert!(chunks.iter().all(|c| !c.oversized));
        let embedded = chunks[1].embed_text(&doc, "p1r0", "T");
        assert!(embedded.starts_with("Paper: p1r0\nTitle: T\nSection 1\n\n```cpp\nw w w w"));
    }

    #[test]
    fn heading_path_follows_nested_headings() {
        let md = "## 4 Design\nintro\n### 4.1 Units\nunits\n#### 4.1.1 Base\nbase\n### 4.2 Systems\nsys\n## 5\n## 6\n## 7\n";
        let doc = Document::parse(md);
        let chunks = chunk_document(&doc, &doc.sections(), settings(1, 0, 1000), &Words, LABEL);
        let paths: Vec<String> = chunks.iter().map(Chunk::heading_path_text).collect();
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
