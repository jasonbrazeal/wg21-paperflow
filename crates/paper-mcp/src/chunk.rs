// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Structure-aware chunking of tomd Markdown.

/// Version of the chunking rules, folded into the index-format version.
///
/// Bump it whenever a change to this module can change chunk boundaries,
/// heading paths or chunk text for any input.
pub const CHUNKER_VERSION: u32 = 1;

/// Largest chunk, in embedder tokens, before the prefix is added, chosen by
/// the golden-set evaluation (`eval/RESULTS.md`).
///
/// Changing it changes every chunk, so it must ship with a
/// [`CHUNKER_VERSION`] bump.
pub const DEFAULT_CHUNK_BUDGET: usize = 512;

/// One chunk of a paper's Markdown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    /// Level 1 to 3 headings enclosing the chunk, outermost first.
    pub heading_path: Vec<String>,
    /// First source line of the chunk, 1-based and inclusive.
    pub start: usize,
    /// Source line after the chunk, 1-based and exclusive.
    pub end: usize,
    /// Source lines `start..end` joined with `\n`.
    pub text: String,
}

/// One ATX heading and the source lines it governs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Section {
    /// Heading level, 1 to 6.
    pub level: usize,
    /// Heading text without the `#` markers.
    pub heading: String,
    /// The heading's source line, 1-based and inclusive.
    pub start: usize,
    /// The line of the next heading at the same or a higher level, or one
    /// past the last line, 1-based and exclusive.
    pub end: usize,
}

/// Deepest heading level that starts a new chunk.
const SPLIT_LEVEL: usize = 3;

/// Deepest ATX heading level.
const MAX_HEADING_LEVEL: usize = 6;

/// Spaces of indentation an ATX heading may carry.
const MAX_HEADING_INDENT: usize = 3;

/// Shortest run of backticks or tildes that opens a code fence.
const MIN_FENCE_LEN: usize = 3;

/// Comment tomd places before an HTML table it could not render as a pipe table.
const MIXED_TABLE_MARKER: &str = "<!-- tomd:mixed-table -->";

/// One structural element of the source, as 0-based half-open line ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Item {
    /// Lines excluded from every chunk: front matter or a `<style>` block.
    Skip,
    /// An ATX heading on one line.
    Heading {
        line: usize,
        level: usize,
        text: String,
    },
    /// An atomic block: a paragraph, code fence, pipe table or HTML table.
    Block { start: usize, end: usize },
}

/// Lists every heading outside front matter, code fences and `<style>` blocks.
#[must_use]
pub fn outline(md: &str) -> Vec<Section> {
    let lines: Vec<&str> = md.lines().collect();
    let headings: Vec<(usize, usize, String)> = scan(&lines)
        .into_iter()
        .filter_map(|item| match item {
            Item::Heading { line, level, text } => Some((line, level, text)),
            Item::Skip | Item::Block { .. } => None,
        })
        .collect();
    headings
        .iter()
        .enumerate()
        .map(|(i, (line, level, text))| {
            let end = headings[i + 1..]
                .iter()
                .find(|(_, next_level, _)| next_level <= level)
                .map_or(lines.len(), |(next_line, _, _)| *next_line);
            Section {
                level: *level,
                heading: text.clone(),
                start: line + 1,
                end: end + 1,
            }
        })
        .collect()
}

/// Splits tomd Markdown into chunks of at most `budget` tokens.
///
/// YAML front matter and `<style>` blocks never appear in a chunk, but their
/// lines keep their numbers. A `<style>` with no closing `</style>` is
/// ordinary content. Every level 1 to 3 heading starts a new chunk,
/// and a section with no content below its heading yields no chunk. Fenced
/// code blocks, pipe tables and HTML tables are never split unless one alone
/// exceeds `budget`, in which case it becomes its own chunks split at line
/// boundaries. A heading that would push the block after it over `budget`
/// becomes its own chunk. A single line longer than `budget` is emitted whole.
#[must_use]
pub fn chunk_markdown(md: &str, budget: usize, count_tokens: &dyn Fn(&str) -> usize) -> Vec<Chunk> {
    let lines: Vec<&str> = md.lines().collect();
    let mut packer = Packer {
        lines: &lines,
        budget,
        count_tokens,
        chunks: Vec::new(),
    };
    let mut path: Vec<(usize, String)> = Vec::new();
    let mut group = Group::default();
    for item in scan(&lines) {
        match item {
            Item::Skip => packer.flush(&mut group, &path),
            Item::Heading { line, level, text } => {
                if level <= SPLIT_LEVEL {
                    packer.flush(&mut group, &path);
                    path.retain(|(outer, _)| *outer < level);
                    path.push((level, text));
                }
                group.push_heading(line, level);
            }
            Item::Block { start, end } => group.push_block(start, end),
        }
    }
    packer.flush(&mut group, &path);
    packer.chunks
}

/// One packing unit: an atomic line range, optionally led by a heading line.
#[derive(Clone, Copy, Debug)]
struct Unit {
    /// 0-based line of a heading that directly precedes the range.
    heading: Option<usize>,
    /// 0-based half-open line range, packed without being split.
    start: usize,
    end: usize,
}

impl Unit {
    fn line(line: usize) -> Self {
        Self {
            heading: None,
            start: line,
            end: line + 1,
        }
    }
}

/// The units of one chunk run: contiguous source between hard boundaries.
#[derive(Debug, Default)]
struct Group {
    units: Vec<Unit>,
    /// A heading waiting to join the unit that follows it.
    pending_heading: Option<usize>,
    /// Whether any unit holds more than a split-level heading.
    has_content: bool,
}

impl Group {
    fn push_heading(&mut self, line: usize, level: usize) {
        if let Some(prev) = self.pending_heading.take() {
            self.units.push(Unit::line(prev));
        }
        self.pending_heading = Some(line);
        if level > SPLIT_LEVEL {
            self.has_content = true;
        }
    }

    fn push_block(&mut self, start: usize, end: usize) {
        self.units.push(Unit {
            heading: self.pending_heading.take(),
            start,
            end,
        });
        self.has_content = true;
    }

    /// Returns the units and resets the group.
    fn take(&mut self) -> Option<Vec<Unit>> {
        if let Some(line) = self.pending_heading.take() {
            self.units.push(Unit::line(line));
        }
        let units = std::mem::take(&mut self.units);
        let has_content = std::mem::take(&mut self.has_content);
        has_content.then_some(units)
    }
}

/// Greedy packer of units into budget-bounded chunks.
struct Packer<'a> {
    lines: &'a [&'a str],
    budget: usize,
    count_tokens: &'a dyn Fn(&str) -> usize,
    chunks: Vec<Chunk>,
}

impl Packer<'_> {
    fn text(&self, start: usize, end: usize) -> String {
        self.lines[start..end].join("\n")
    }

    fn fits(&self, start: usize, end: usize) -> bool {
        (self.count_tokens)(&self.text(start, end)) <= self.budget
    }

    fn emit(&mut self, start: usize, end: usize, path: &[(usize, String)]) {
        let lines = self.lines;
        let blank = |i: usize| lines[i].trim().is_empty();
        let start = (start..end).find(|&i| !blank(i)).unwrap_or(end);
        let end = (start..end)
            .rev()
            .find(|&i| !blank(i))
            .map_or(start, |i| i + 1);
        if start == end {
            return;
        }
        self.chunks.push(Chunk {
            heading_path: path.iter().map(|(_, h)| h.clone()).collect(),
            start: start + 1,
            end: end + 1,
            text: self.text(start, end),
        });
    }

    fn flush(&mut self, group: &mut Group, path: &[(usize, String)]) {
        let Some(units) = group.take() else { return };
        let mut current: Option<(usize, usize)> = None;
        for unit in units {
            let start = unit.heading.unwrap_or(unit.start);
            let end = unit.end;
            if let Some((run_start, run_end)) = current {
                if self.fits(run_start, end) {
                    current = Some((run_start, end));
                    continue;
                }
                self.emit(run_start, run_end, path);
                current = None;
            }
            if self.fits(start, end) {
                current = Some((start, end));
            } else if let Some(line) = unit.heading
                && self.fits(unit.start, end)
            {
                // The heading alone is emitted so the block it leads stays whole.
                self.emit(line, unit.start, path);
                current = Some((unit.start, end));
            } else {
                self.split_lines(start, end, path);
            }
        }
        if let Some((start, end)) = current {
            self.emit(start, end, path);
        }
    }

    /// Emits an oversized unit as greedy runs of whole lines.
    fn split_lines(&mut self, start: usize, end: usize, path: &[(usize, String)]) {
        let mut piece_start = start;
        while piece_start < end {
            let mut piece_end = piece_start + 1;
            while piece_end < end && self.fits(piece_start, piece_end + 1) {
                piece_end += 1;
            }
            self.emit(piece_start, piece_end, path);
            piece_start = piece_end;
        }
    }
}

/// Splits source lines into structural items, dropping blank lines.
fn scan(lines: &[&str]) -> Vec<Item> {
    let mut items = Vec::new();
    let mut i = front_matter_end(lines).unwrap_or(0);
    if i > 0 {
        items.push(Item::Skip);
    }
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            i += 1;
        } else if let Some((level, text)) = heading(line) {
            items.push(Item::Heading {
                line: i,
                level,
                text,
            });
            i += 1;
        } else if let Some(close) = starts_style(trimmed)
            .then(|| (i..lines.len()).find(|&j| lines[j].contains("</style>")))
            .flatten()
        {
            i = close + 1;
            items.push(Item::Skip);
        } else {
            let end = block_end(lines, i);
            items.push(Item::Block { start: i, end });
            i = end;
        }
    }
    items
}

/// Returns the 0-based line after closed YAML front matter, if any.
fn front_matter_end(lines: &[&str]) -> Option<usize> {
    if lines.first()?.trim_end() != "---" {
        return None;
    }
    (1..lines.len())
        .find(|&j| matches!(lines[j].trim_end(), "---" | "..."))
        .map(|j| j + 1)
}

/// Returns the 0-based end of the block starting at non-blank line `start`.
fn block_end(lines: &[&str], start: usize) -> usize {
    let trimmed = lines[start].trim_start();
    if let Some((marker, len)) = fence_open(trimmed) {
        return (start + 1..lines.len())
            .find(|&j| closes_fence(lines[j], marker, len))
            .map_or(lines.len(), |j| j + 1);
    }
    if starts_html_table(trimmed) {
        return html_table_end(lines, start);
    }
    if trimmed.starts_with('|') {
        return (start + 1..lines.len())
            .find(|&j| !lines[j].trim_start().starts_with('|'))
            .unwrap_or(lines.len());
    }
    (start + 1..lines.len())
        .find(|&j| {
            let t = lines[j].trim_start();
            t.is_empty()
                || heading(lines[j]).is_some()
                || fence_open(t).is_some()
                || starts_style(t)
                || starts_html_table(t)
                || t.starts_with('|')
        })
        .unwrap_or(lines.len())
}

/// Returns the 0-based line after the `</table>` that balances the first `<table`.
fn html_table_end(lines: &[&str], start: usize) -> usize {
    let mut depth = 0usize;
    let mut opened = false;
    for (j, line) in lines.iter().enumerate().skip(start) {
        let lower = line.to_ascii_lowercase();
        let opens = lower.matches("<table").count();
        let closes = lower.matches("</table>").count();
        opened |= opens > 0;
        depth = (depth + opens).saturating_sub(closes);
        if opened && depth == 0 {
            return j + 1;
        }
    }
    lines.len()
}

fn starts_style(trimmed: &str) -> bool {
    trimmed.to_ascii_lowercase().starts_with("<style")
}

fn starts_html_table(trimmed: &str) -> bool {
    trimmed.starts_with(MIXED_TABLE_MARKER) || trimmed.to_ascii_lowercase().starts_with("<table")
}

/// Returns the fence character and run length when `trimmed` opens a code fence.
fn fence_open(trimmed: &str) -> Option<(char, usize)> {
    let marker = trimmed.chars().next().filter(|c| matches!(c, '`' | '~'))?;
    let len = trimmed.chars().take_while(|&c| c == marker).count();
    let info = &trimmed[len..];
    (len >= MIN_FENCE_LEN && !(marker == '`' && info.contains('`'))).then_some((marker, len))
}

fn closes_fence(line: &str, marker: char, len: usize) -> bool {
    let trimmed = line.trim();
    let run = trimmed.chars().take_while(|&c| c == marker).count();
    run >= len && run == trimmed.chars().count()
}

/// Returns the level and text when `line` is an ATX heading.
fn heading(line: &str) -> Option<(usize, String)> {
    let body = line.trim_start_matches(' ');
    if line.len() - body.len() > MAX_HEADING_INDENT {
        return None;
    }
    let level = body.chars().take_while(|&c| c == '#').count();
    if !(1..=MAX_HEADING_LEVEL).contains(&level) {
        return None;
    }
    let rest = &body[level..];
    if !(rest.is_empty() || rest.starts_with([' ', '\t'])) {
        return None;
    }
    let rest = rest.trim();
    let without_closing = rest.trim_end_matches('#');
    let text = if without_closing.is_empty() || without_closing.ends_with([' ', '\t']) {
        without_closing.trim_end()
    } else {
        rest
    };
    Some((level, text.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const P2169R4: &str = include_str!("../tests/fixtures/p2169r4.md");
    const P2322R4: &str = include_str!("../tests/fixtures/p2322r4.md");

    fn words(s: &str) -> usize {
        s.split_whitespace().count()
    }

    fn chunks(md: &str, budget: usize) -> Vec<Chunk> {
        chunk_markdown(md, budget, &words)
    }

    fn slice(md: &str, start: usize, end: usize) -> String {
        md.lines().collect::<Vec<_>>()[start - 1..end - 1].join("\n")
    }

    fn chunk(path: &[&str], start: usize, end: usize, md: &str) -> Chunk {
        Chunk {
            heading_path: path.iter().map(ToString::to_string).collect(),
            start,
            end,
            text: slice(md, start, end),
        }
    }

    fn section(level: usize, heading: &str, start: usize, end: usize) -> Section {
        Section {
            level,
            heading: heading.to_string(),
            start,
            end,
        }
    }

    /// Asserts one chunk holds lines `start..end` and no chunk cuts into them.
    fn assert_whole(cs: &[Chunk], start: usize, end: usize) {
        assert!(
            cs.iter().any(|c| c.start <= start && end <= c.end),
            "no chunk holds {start}..{end}: {:?}",
            cs.iter().map(|c| (c.start, c.end)).collect::<Vec<_>>()
        );
        for c in cs {
            let overlaps = c.start < end && start < c.end;
            let contains = c.start <= start && end <= c.end;
            assert!(
                !overlaps || contains,
                "chunk {}..{} cuts {start}..{end}",
                c.start,
                c.end
            );
        }
    }

    const INLINE: &str = "---\ntitle: x\n---\n\n<style>\np{}\n</style>\n\nIntro para.\n\n## A\n\n\
        alpha one.\n\nalpha two.\n\n### B\n\n```cpp\n# not heading\n```\n\n## C\n\n\
        | a | b |\n| - | - |\n| 1 | 2 |\n";

    #[test]
    fn splits_on_headings_with_exact_paths_and_ranges() {
        assert_eq!(
            chunks(INLINE, usize::MAX),
            vec![
                chunk(&[], 9, 10, INLINE),
                chunk(&["A"], 11, 16, INLINE),
                chunk(&["A", "B"], 17, 22, INLINE),
                chunk(&["C"], 23, 28, INLINE),
            ]
        );
    }

    #[test]
    fn outline_lists_headings_outside_fences() {
        assert_eq!(
            outline(INLINE),
            vec![
                section(2, "A", 11, 23),
                section(3, "B", 17, 23),
                section(2, "C", 23, 28)
            ]
        );
    }

    #[test]
    fn outline_includes_deep_headings_and_strips_closing_hashes() {
        let md = "## A ##\n\n#### Deep\n\nx\n\n### B\n";
        assert_eq!(
            outline(md),
            vec![
                section(2, "A", 1, 8),
                section(4, "Deep", 3, 7),
                section(3, "B", 7, 8)
            ]
        );
    }

    #[test]
    fn budget_splits_only_at_block_boundaries() {
        let md = "## A\n\none two three\n\nfour five six\n\nseven\n";
        assert_eq!(
            chunks(md, 7),
            vec![chunk(&["A"], 1, 4, md), chunk(&["A"], 5, 8, md)]
        );
    }

    #[test]
    fn code_fence_stays_whole_when_it_fits() {
        let md = "## A\n\nintro words here\n\n```\na b\nc d\n```\n";
        assert_eq!(
            chunks(md, 6),
            vec![chunk(&["A"], 1, 4, md), chunk(&["A"], 5, 9, md)]
        );
    }

    #[test]
    fn heading_splits_from_block_that_fits_only_alone() {
        let md = "## A\n\n```\na b\nc d\n```\n";
        assert_eq!(
            chunks(md, 6),
            vec![chunk(&["A"], 1, 2, md), chunk(&["A"], 3, 7, md)]
        );
    }

    #[test]
    fn unclosed_style_is_content() {
        let md = "## A\n\n<style>\nx{}\n\nafter\n\n## B\n\nbody\n";
        assert_eq!(
            chunks(md, usize::MAX),
            vec![chunk(&["A"], 1, 7, md), chunk(&["B"], 8, 11, md)]
        );
        assert_eq!(
            outline(md),
            vec![section(2, "A", 1, 8), section(2, "B", 8, 11)]
        );
    }

    #[test]
    fn lone_oversized_block_splits_at_line_boundaries() {
        let md = "## A\n\nintro words here\n\n```\na b\nc d\n```\n";
        assert_eq!(
            chunks(md, 3),
            vec![
                chunk(&["A"], 1, 2, md),
                chunk(&["A"], 3, 4, md),
                chunk(&["A"], 5, 7, md),
                chunk(&["A"], 7, 9, md),
            ]
        );
    }

    #[test]
    fn oversized_single_line_is_emitted_whole() {
        let md = "one two three four\n";
        assert_eq!(chunks(md, 2), vec![chunk(&[], 1, 2, md)]);
    }

    #[test]
    fn headingless_document_has_empty_path() {
        let md = "Just text.\n\nMore text.\n";
        assert_eq!(chunks(md, usize::MAX), vec![chunk(&[], 1, 4, md)]);
    }

    #[test]
    fn heading_only_sections_yield_no_chunk() {
        let md = "## A\n### B\n\nbody\n";
        assert_eq!(chunks(md, usize::MAX), vec![chunk(&["A", "B"], 2, 5, md)]);
    }

    #[test]
    fn style_block_inside_section_is_a_hard_boundary() {
        let md = "## A\n\nbefore\n\n<style>\nx{}\n</style>\n\nafter\n";
        assert_eq!(
            chunks(md, usize::MAX),
            vec![chunk(&["A"], 1, 4, md), chunk(&["A"], 9, 10, md)]
        );
    }

    #[test]
    fn unclosed_front_matter_is_content() {
        let md = "---\nnot closed\n";
        assert_eq!(chunks(md, usize::MAX), vec![chunk(&[], 1, 3, md)]);
    }

    #[test]
    fn crlf_line_endings_match_lf() {
        let lf = "## A\n\nbody\n";
        let crlf = lf.replace('\n', "\r\n");
        assert_eq!(chunks(&crlf, usize::MAX), chunks(lf, usize::MAX));
    }

    #[test]
    fn fixture_front_matter_and_style_excluded() {
        let cs = chunks(P2169R4, usize::MAX);
        assert_eq!(cs.first().map(|c| c.start), Some(16));
        assert!(
            cs.iter()
                .all(|c| !c.text.contains("<style>") && !c.text.contains("document:"))
        );
        let cs = chunks(P2322R4, usize::MAX);
        assert_eq!(cs.first().map(|c| c.start), Some(10));
    }

    #[test]
    fn fixture_ranges_reproduce_text_and_respect_budget() {
        for md in [P2169R4, P2322R4] {
            for budget in [usize::MAX, 400, 120, 40, 5] {
                let cs = chunks(md, budget);
                assert!(!cs.is_empty());
                let mut prev_end = 1;
                for c in &cs {
                    assert!(prev_end <= c.start && c.start < c.end, "{budget}: {c:?}");
                    assert_eq!(
                        c.text,
                        slice(md, c.start, c.end),
                        "{budget}: {}..{}",
                        c.start,
                        c.end
                    );
                    assert!(
                        words(&c.text) <= budget || c.end - c.start == 1,
                        "{budget}: {}..{} over budget",
                        c.start,
                        c.end
                    );
                    prev_end = c.end;
                }
            }
        }
    }

    #[test]
    fn fixture_heading_paths() {
        let cs = chunks(P2169R4, usize::MAX);
        let path_at = |cs: &[Chunk], line: usize| {
            cs.iter()
                .find(|c| c.start == line)
                .map(|c| c.heading_path.clone())
        };
        assert_eq!(
            path_at(&cs, 46),
            Some(vec!["Revisions".to_string(), "Revision 4".to_string()])
        );
        assert_eq!(
            path_at(&cs, 271),
            Some(vec![
                "Wording".to_string(),
                "? General [basic.scope.scope]".to_string()
            ])
        );
        assert_eq!(
            path_at(&cs, 44),
            None,
            "heading-only `## Revisions` must not be a chunk"
        );
        assert!(
            cs.iter()
                .all(|c| c.heading_path.iter().all(|h| !h.contains("define")))
        );

        let cs = chunks(P2322R4, usize::MAX);
        let deep = cs.iter().find(|c| c.start <= 181 && 181 < c.end);
        assert_eq!(
            deep.map(|c| (c.start, c.end, c.heading_path.clone())),
            Some((
                163,
                217,
                vec![
                    "4 Other `fold` algorithms".to_string(),
                    "4.1 `fold1`".to_string()
                ]
            ))
        );
    }

    #[test]
    fn fixture_outline() {
        let o = outline(P2322R4);
        assert!(o.contains(&section(4, "4.1.1 Distinct name?", 181, 203)));
        assert!(o.contains(&section(2, "4 Other `fold` algorithms", 159, 521)));
        assert_eq!(o.last(), Some(&section(2, "7 References", 868, 891)));
        assert!(
            outline(P2169R4)
                .iter()
                .all(|s| !s.heading.contains("define"))
        );
    }

    /// Chunks with a budget equal to the block's own size, so it fits exactly
    /// while its section does not.
    fn assert_block_whole(md: &str, start: usize, end: usize) {
        let budget = words(&slice(md, start, end));
        assert_whole(&chunks(md, budget), start, end);
    }

    #[test]
    fn fixture_code_fences_kept_whole() {
        assert_block_whole(P2169R4, 303, 342);
        assert_block_whole(P2169R4, 241, 244);
        assert_block_whole(P2322R4, 279, 285);
        assert_block_whole(P2322R4, 299, 317);
    }

    #[test]
    fn fixture_pipe_tables_kept_whole() {
        assert_block_whole(P2322R4, 230, 238);
        assert_block_whole(P2322R4, 260, 266);
    }

    #[test]
    fn fixture_mixed_html_tables_kept_whole() {
        assert_block_whole(P2169R4, 22, 43);
        assert_block_whole(P2322R4, 106, 145);
        assert_block_whole(P2322R4, 367, 400);
    }
}
