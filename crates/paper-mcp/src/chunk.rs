//
// Copyright (c) 2026 Jason Brazeal
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

//! Splitting tomd Markdown into indexable chunks.
//!
//! [`chunk_markdown`] splits a paper into sections on ATX headings, then packs
//! each section's blocks (paragraphs, fenced code, tables) into chunks of
//! about [`TARGET_TOKENS`]. A block is never split unless it alone exceeds
//! [`MAX_TOKENS`], in which case it is split on line boundaries. Every chunk
//! is an exact slice of the paper's lines, so its `start_line`/`end_line`
//! work directly with `get_paper_lines`.
//!
//! Each chunk carries a deterministic contextual prefix ([`build_prefix`]) in
//! its `indexed_text`, used for both embedding and keyword search.

/// Version of the chunking rules, recorded in the index manifest.
///
/// Bump whenever a change alters chunk boundaries, kinds, or prefixes, so the
/// indexer re-chunks every paper.
pub const CHUNKER_VERSION: u32 = 1;

/// Token count chunks are packed toward.
pub const TARGET_TOKENS: usize = 512;

/// Token count above which a single block is split on line boundaries.
pub const MAX_TOKENS: usize = 1024;

/// Characters per estimated token within a run of alphanumeric characters.
const CHARS_PER_WORD_PIECE: usize = 4;

/// Lowercased heading substrings marking a revision history section.
const REVISION_HISTORY_MARKERS: [&str; 6] = [
    "revision history",
    "changelog",
    "change log",
    "changes from",
    "changes since",
    "document history",
];

/// Lowercased heading substring marking a wording section.
const WORDING_MARKER: &str = "wording";

/// Separator between headings in a rendered section path.
const SECTION_PATH_SEPARATOR: &str = " > ";

/// Maximum leading spaces before an ATX heading marker.
const MAX_HEADING_INDENT: usize = 3;

/// Deepest ATX heading level.
const MAX_HEADING_LEVEL: usize = 6;

/// Minimum run of backticks or tildes opening a code fence.
const MIN_FENCE_LEN: usize = 3;

/// Category of a chunk, stored in the index and usable as a search filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkKind {
    /// Ordinary prose, or prose mixed with code and tables.
    Body,
    /// Proposed standard wording, by a heading containing "wording".
    Wording,
    /// Fenced code only.
    Code,
    /// A revision history or change log section.
    RevisionHistory,
    /// The YAML front matter block.
    FrontMatter,
}

impl ChunkKind {
    /// Name stored in the index and accepted by search filters.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Body => "body",
            Self::Wording => "wording",
            Self::Code => "code",
            Self::RevisionHistory => "revision_history",
            Self::FrontMatter => "front_matter",
        }
    }
}

/// Paper metadata rendered into each chunk's contextual prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMeta {
    /// Exact paper ID, for example `P2300R10`.
    pub paper_id: String,
    /// Paper title.
    pub title: String,
    /// Author names in paperstore order.
    pub authors: Vec<String>,
}

/// One indexable slice of a paper's Markdown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Heading titles from the outermost to the innermost enclosing heading.
    /// Empty for front matter and text before the first heading.
    pub section_path: Vec<String>,
    /// First line of the chunk, 0-based.
    pub start_line: usize,
    /// Line after the last line of the chunk (exclusive).
    pub end_line: usize,
    /// Category assigned by heading and content rules.
    pub chunk_kind: ChunkKind,
    /// The chunk's lines joined with `\n`, without the contextual prefix.
    pub text: String,
    /// [`build_prefix`] output followed by `text`; embedded and keyword
    /// indexed.
    pub indexed_text: String,
}

/// Render a section path for display and prefixes, for example `Design > API`.
#[must_use]
pub fn render_section_path(section_path: &[String]) -> String {
    section_path.join(SECTION_PATH_SEPARATOR)
}

/// Estimate the embedding model token count of `text`.
///
/// Each run of alphanumeric characters counts one token per started group of
/// four characters, and every other non-whitespace character counts one
/// token, approximating WordPiece tokenization without loading a tokenizer.
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    let mut tokens = 0;
    let mut run: usize = 0;
    for c in text.chars() {
        if c.is_alphanumeric() {
            run += 1;
            continue;
        }
        tokens += run.div_ceil(CHARS_PER_WORD_PIECE);
        run = 0;
        if !c.is_whitespace() {
            tokens += 1;
        }
    }
    tokens + run.div_ceil(CHARS_PER_WORD_PIECE)
}

/// Build the contextual prefix for a chunk in `section_path`.
///
/// The prefix is one `Label: value` line each for the paper ID, title,
/// authors, and section path, followed by a blank line. The authors and
/// section lines are omitted when empty. The same inputs always produce the
/// same prefix.
#[must_use]
pub fn build_prefix(meta: &PrefixMeta, section_path: &[String]) -> String {
    let mut prefix = format!("Paper: {}\nTitle: {}\n", meta.paper_id, meta.title);
    if !meta.authors.is_empty() {
        prefix.push_str("Authors: ");
        prefix.push_str(&meta.authors.join(", "));
        prefix.push('\n');
    }
    if !section_path.is_empty() {
        prefix.push_str("Section: ");
        prefix.push_str(&render_section_path(section_path));
        prefix.push('\n');
    }
    prefix.push('\n');
    prefix
}

/// Split tomd Markdown into chunks in document order.
///
/// Front matter becomes one `front_matter` chunk. Each heading starts a
/// section, and a heading is always kept with the content that follows it;
/// sections holding only a heading produce no chunk. Lines are split as by
/// [`str::lines`], so `\r\n` endings slice like `\n`.
#[must_use]
pub fn chunk_markdown(md: &str, meta: &PrefixMeta) -> Vec<Chunk> {
    let lines: Vec<&str> = md.lines().collect();
    let mut chunks = Vec::new();
    let body_start = match front_matter_end(&lines) {
        Some(end) => {
            let block = Block {
                start: 0,
                end,
                kind: BlockKind::FrontMatter,
                tokens: estimate_tokens(&lines[..end].join("\n")),
            };
            let section = Section {
                path: Vec::new(),
                blocks: vec![block],
            };
            push_section_chunks(&lines, &section, meta, &mut chunks);
            end
        }
        None => 0,
    };
    for section in parse_sections(&lines, body_start) {
        push_section_chunks(&lines, &section, meta, &mut chunks);
    }
    chunks
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Heading,
    Text,
    Code,
    FrontMatter,
}

#[derive(Debug, Clone, Copy)]
struct Block {
    start: usize,
    end: usize,
    kind: BlockKind,
    tokens: usize,
}

#[derive(Debug)]
struct Section {
    path: Vec<String>,
    blocks: Vec<Block>,
}

#[derive(Debug, Clone, Copy)]
struct Fence {
    marker: char,
    len: usize,
    start: usize,
}

#[derive(Debug, Clone, Copy)]
struct Piece {
    start: usize,
    end: usize,
    tokens: usize,
    has_content: bool,
    all_code: bool,
    front_matter: bool,
}

fn front_matter_end(lines: &[&str]) -> Option<usize> {
    if lines.first().map(|l| l.trim_end()) != Some("---") {
        return None;
    }
    lines
        .iter()
        .skip(1)
        .position(|l| matches!(l.trim_end(), "---" | "..."))
        .map(|i| i + 2)
}

fn parse_sections(lines: &[&str], from: usize) -> Vec<Section> {
    let mut sections = vec![Section {
        path: Vec::new(),
        blocks: Vec::new(),
    }];
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut paragraph: Option<usize> = None;
    let mut fence: Option<Fence> = None;
    for (i, line) in lines.iter().enumerate().skip(from) {
        if let Some(open) = fence {
            if closes_fence(line, open) {
                push_block(lines, &mut sections, open.start, i + 1, BlockKind::Code);
                fence = None;
            }
            continue;
        }
        if let Some((marker, len)) = opens_fence(line) {
            flush_paragraph(lines, &mut sections, &mut paragraph, i);
            fence = Some(Fence {
                marker,
                len,
                start: i,
            });
        } else if let Some((level, title)) = parse_heading(line) {
            flush_paragraph(lines, &mut sections, &mut paragraph, i);
            while stack.last().is_some_and(|(l, _)| *l >= level) {
                stack.pop();
            }
            stack.push((level, title));
            sections.push(Section {
                path: stack.iter().map(|(_, t)| t.clone()).collect(),
                blocks: Vec::new(),
            });
            push_block(lines, &mut sections, i, i + 1, BlockKind::Heading);
        } else if line.trim().is_empty() {
            flush_paragraph(lines, &mut sections, &mut paragraph, i);
        } else if paragraph.is_none() {
            paragraph = Some(i);
        }
    }
    flush_paragraph(lines, &mut sections, &mut paragraph, lines.len());
    if let Some(open) = fence {
        push_block(
            lines,
            &mut sections,
            open.start,
            lines.len(),
            BlockKind::Code,
        );
    }
    sections
}

fn flush_paragraph(
    lines: &[&str],
    sections: &mut [Section],
    paragraph: &mut Option<usize>,
    end: usize,
) {
    if let Some(start) = paragraph.take() {
        push_block(lines, sections, start, end, BlockKind::Text);
    }
}

fn push_block(lines: &[&str], sections: &mut [Section], start: usize, end: usize, kind: BlockKind) {
    if let Some(section) = sections.last_mut() {
        section.blocks.push(Block {
            start,
            end,
            kind,
            tokens: estimate_tokens(&lines[start..end].join("\n")),
        });
    }
}

fn opens_fence(line: &str) -> Option<(char, usize)> {
    let trimmed = line.trim_start();
    let marker = trimmed.chars().next().filter(|c| matches!(c, '`' | '~'))?;
    let len = trimmed.chars().take_while(|c| *c == marker).count();
    if len < MIN_FENCE_LEN {
        return None;
    }
    let info = &trimmed[len * marker.len_utf8()..];
    if marker == '`' && info.contains('`') {
        return None;
    }
    Some((marker, len))
}

fn closes_fence(line: &str, open: Fence) -> bool {
    let trimmed = line.trim();
    let len = trimmed.chars().take_while(|c| *c == open.marker).count();
    len >= open.len && len == trimmed.chars().count()
}

fn parse_heading(line: &str) -> Option<(usize, String)> {
    let indent = line.chars().take_while(|c| *c == ' ').count();
    if indent > MAX_HEADING_INDENT {
        return None;
    }
    let rest = &line[indent..];
    let level = rest.chars().take_while(|c| *c == '#').count();
    if level == 0 || level > MAX_HEADING_LEVEL {
        return None;
    }
    let after = &rest[level..];
    if !(after.is_empty() || after.starts_with([' ', '\t'])) {
        return None;
    }
    let mut title = after.trim();
    let without_closing = title.trim_end_matches('#');
    if without_closing.is_empty() || without_closing.ends_with([' ', '\t']) {
        title = without_closing.trim_end();
    }
    Some((level, title.to_owned()))
}

/// Split a block into line runs of at most [`TARGET_TOKENS`] each.
///
/// A single line over the target becomes its own run.
fn split_block(lines: &[&str], block: Block) -> Vec<Block> {
    let mut runs = Vec::new();
    let mut start = block.start;
    let mut tokens = 0;
    for (i, line) in lines.iter().enumerate().take(block.end).skip(block.start) {
        let line_tokens = estimate_tokens(line);
        if i > start && tokens + line_tokens > TARGET_TOKENS {
            runs.push(Block {
                start,
                end: i,
                kind: block.kind,
                tokens,
            });
            start = i;
            tokens = 0;
        }
        tokens += line_tokens;
    }
    runs.push(Block {
        start,
        end: block.end,
        kind: block.kind,
        tokens,
    });
    runs
}

fn push_section_chunks(
    lines: &[&str],
    section: &Section,
    meta: &PrefixMeta,
    chunks: &mut Vec<Chunk>,
) {
    let heading_kind = heading_kind(&section.path);
    let prefix = build_prefix(meta, &section.path);
    let mut pieces = Vec::new();
    let mut current: Option<Piece> = None;
    for block in &section.blocks {
        let pending = current.filter(|p| !p.has_content).map_or(0, |p| p.tokens);
        let units = if block.tokens + pending > MAX_TOKENS {
            split_block(lines, *block)
        } else {
            vec![*block]
        };
        for unit in units {
            if let Some(piece) = current
                && piece.has_content
                && piece.tokens + unit.tokens > TARGET_TOKENS
            {
                pieces.push(piece);
                current = None;
            }
            let is_content = unit.kind != BlockKind::Heading;
            let piece = current.get_or_insert(Piece {
                start: unit.start,
                end: unit.end,
                tokens: 0,
                has_content: false,
                all_code: true,
                front_matter: false,
            });
            piece.end = unit.end;
            piece.tokens += unit.tokens;
            piece.has_content |= is_content;
            if is_content {
                piece.all_code &= unit.kind == BlockKind::Code;
            }
            piece.front_matter |= unit.kind == BlockKind::FrontMatter;
        }
    }
    pieces.extend(current);
    for piece in pieces.into_iter().filter(|p| p.has_content) {
        let chunk_kind = if piece.front_matter {
            ChunkKind::FrontMatter
        } else if let Some(kind) = heading_kind {
            kind
        } else if piece.all_code {
            ChunkKind::Code
        } else {
            ChunkKind::Body
        };
        let text = lines[piece.start..piece.end].join("\n");
        chunks.push(Chunk {
            section_path: section.path.clone(),
            start_line: piece.start,
            end_line: piece.end,
            chunk_kind,
            indexed_text: format!("{prefix}{text}"),
            text,
        });
    }
}

/// Kind implied by the innermost heading in `path` that matches a rule.
fn heading_kind(path: &[String]) -> Option<ChunkKind> {
    path.iter().rev().find_map(|heading| {
        let lower = heading.to_lowercase();
        if REVISION_HISTORY_MARKERS.iter().any(|m| lower.contains(m)) {
            Some(ChunkKind::RevisionHistory)
        } else if lower.contains(WORDING_MARKER) {
            Some(ChunkKind::Wording)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> PrefixMeta {
        PrefixMeta {
            paper_id: "P1000R2".to_owned(),
            title: "std::ring_span, a fixed-capacity circular view".to_owned(),
            authors: vec!["Ada Lovelace".to_owned(), "Charles Babbage".to_owned()],
        }
    }

    fn path(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    fn assert_line_ranges(md: &str, chunks: &[Chunk]) {
        let lines: Vec<&str> = md.lines().collect();
        for chunk in chunks {
            assert!(chunk.start_line < chunk.end_line, "empty range: {chunk:?}");
            assert!(chunk.end_line <= lines.len(), "range past end: {chunk:?}");
            assert_eq!(
                chunk.text,
                lines[chunk.start_line..chunk.end_line].join("\n"),
                "text differs from its line slice"
            );
        }
        for pair in chunks.windows(2) {
            assert!(pair[0].end_line <= pair[1].start_line, "overlap: {pair:?}");
        }
    }

    fn paragraph(words: usize, seed: usize) -> String {
        (0..words)
            .map(|i| format!("word{}", (i + seed) % 97))
            .collect::<Vec<_>>()
            .join(" ")
    }

    const FIXTURE_P1000R2: &str = include_str!("../tests/fixtures/paperstore/p1000r2.md");

    #[test]
    fn splits_on_headings_with_nested_section_paths() {
        let md =
            "## Intro\n\nHello there.\n\n### Detail\n\nNested text.\n\n## Design\n\nDesign text.\n";
        let chunks = chunk_markdown(md, &meta());
        let paths: Vec<Vec<String>> = chunks.iter().map(|c| c.section_path.clone()).collect();
        assert_eq!(
            paths,
            vec![
                path(&["Intro"]),
                path(&["Intro", "Detail"]),
                path(&["Design"])
            ]
        );
        assert_eq!(chunks[0].text, "## Intro\n\nHello there.");
        assert_eq!(chunks[1].text, "### Detail\n\nNested text.");
        assert_eq!(chunks[2].text, "## Design\n\nDesign text.");
        assert_line_ranges(md, &chunks);
    }

    #[test]
    fn sibling_heading_replaces_deeper_path_entries() {
        let md = "## A\n\na\n\n### B\n\nb\n\n#### C\n\nc\n\n### D\n\nd\n";
        let chunks = chunk_markdown(md, &meta());
        let paths: Vec<Vec<String>> = chunks.iter().map(|c| c.section_path.clone()).collect();
        assert_eq!(
            paths,
            vec![
                path(&["A"]),
                path(&["A", "B"]),
                path(&["A", "B", "C"]),
                path(&["A", "D"]),
            ]
        );
    }

    #[test]
    fn heading_only_sections_produce_no_chunk() {
        let md = "## Parent\n\n### Child\n\nChild text.\n";
        let chunks = chunk_markdown(md, &meta());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].section_path, path(&["Parent", "Child"]));
    }

    #[test]
    fn packs_paragraphs_up_to_target() {
        let paragraphs: Vec<String> = (0..40).map(|i| paragraph(60, i)).collect();
        let md = format!("## Body\n\n{}\n", paragraphs.join("\n\n"));
        let chunks = chunk_markdown(&md, &meta());
        assert!(
            chunks.len() > 1,
            "expected packing to produce several chunks"
        );
        for chunk in &chunks {
            let tokens = estimate_tokens(&chunk.text);
            assert!(
                tokens <= TARGET_TOKENS,
                "chunk of {tokens} tokens exceeds target"
            );
            assert!(
                tokens > TARGET_TOKENS / 2,
                "chunk of {tokens} tokens is underpacked"
            );
            assert_eq!(chunk.section_path, path(&["Body"]));
        }
        assert_line_ranges(&md, &chunks);
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(chunks.last().map(|c| c.end_line), Some(lines.len()));
    }

    #[test]
    fn paragraph_between_target_and_max_stays_whole() {
        let big = paragraph(400, 0);
        let tokens = estimate_tokens(&big);
        assert!(tokens > TARGET_TOKENS && tokens <= MAX_TOKENS, "{tokens}");
        let md = format!("## Body\n\nShort lead.\n\n{big}\n\nShort tail.\n");
        let chunks = chunk_markdown(&md, &meta());
        assert!(chunks.iter().any(|c| c.text == big));
        assert_line_ranges(&md, &chunks);
    }

    #[test]
    fn code_fence_with_blank_lines_and_hash_lines_stays_whole() {
        let fence = "```cpp\n// first\n\n# define X 1\n## not a heading\n\nint main() {}\n```";
        let filler = paragraph(450, 3);
        let md = format!("## Code\n\n{filler}\n\n{fence}\n");
        let chunks = chunk_markdown(&md, &meta());
        let holders: Vec<&Chunk> = chunks
            .iter()
            .filter(|c| c.text.contains("// first"))
            .collect();
        assert_eq!(holders.len(), 1);
        assert!(holders[0].text.ends_with(fence));
        assert!(chunks.iter().all(|c| c.section_path == path(&["Code"])));
        assert_line_ranges(&md, &chunks);
    }

    #[test]
    fn table_stays_whole() {
        let rows: Vec<String> = (0..30)
            .map(|i| format!("| row{i} | {} |", paragraph(6, i)))
            .collect();
        let table = format!("| a | b |\n| --- | --- |\n{}", rows.join("\n"));
        let filler = paragraph(420, 5);
        let md = format!("## Table\n\n{filler}\n\n{table}\n");
        let chunks = chunk_markdown(&md, &meta());
        let holders: Vec<&Chunk> = chunks
            .iter()
            .filter(|c| c.text.contains("| a | b |"))
            .collect();
        assert_eq!(holders.len(), 1);
        assert!(holders[0].text.ends_with(&table));
        assert_line_ranges(&md, &chunks);
    }

    #[test]
    fn oversized_code_fence_splits_on_line_boundaries() {
        let body: Vec<String> = (0..400)
            .map(|i| format!("    value_{i} = compute(value_{i}, {i});"))
            .collect();
        let fence = format!("```cpp\n{}\n```", body.join("\n"));
        assert!(estimate_tokens(&fence) > MAX_TOKENS);
        let md = format!("## Big\n\n{fence}\n");
        let chunks = chunk_markdown(&md, &meta());
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(estimate_tokens(&chunk.text) <= MAX_TOKENS);
            assert_eq!(chunk.chunk_kind, ChunkKind::Code);
        }
        assert_line_ranges(&md, &chunks);
        for pair in chunks.windows(2) {
            assert_eq!(
                pair[0].end_line, pair[1].start_line,
                "gap inside split fence"
            );
        }
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(chunks.last().map(|c| c.end_line), Some(lines.len()));
    }

    #[test]
    fn assigns_each_chunk_kind() {
        let chunks = chunk_markdown(FIXTURE_P1000R2, &meta());
        let kind_of = |heading: &str| {
            chunks
                .iter()
                .find(|c| c.section_path.first().map(String::as_str) == Some(heading))
                .map(|c| c.chunk_kind)
        };
        assert_eq!(chunks[0].chunk_kind, ChunkKind::FrontMatter);
        assert!(chunks[0].section_path.is_empty());
        assert!(chunks[0].text.starts_with("---\ntitle:"));
        assert!(chunks[0].text.ends_with("---"));
        assert_eq!(
            kind_of("Revision History"),
            Some(ChunkKind::RevisionHistory)
        );
        assert_eq!(kind_of("Motivation"), Some(ChunkKind::Body));
        assert_eq!(kind_of("Wording"), Some(ChunkKind::Wording));

        let code_only = "## Example\n\n```cpp\nint x;\n```\n";
        assert_eq!(
            chunk_markdown(code_only, &meta())[0].chunk_kind,
            ChunkKind::Code
        );
        let mixed = "## Example\n\nProse first.\n\n```cpp\nint x;\n```\n";
        assert_eq!(
            chunk_markdown(mixed, &meta())[0].chunk_kind,
            ChunkKind::Body
        );
    }

    #[test]
    fn heading_rules_match_variants_and_ancestors() {
        let md = "## Proposed Wording\n\n### Header synopsis\n\nAdd it.\n\n## Changes from R1\n\nFixed.\n\n## Changelog\n\nMore.\n";
        let kinds: Vec<ChunkKind> = chunk_markdown(md, &meta())
            .iter()
            .map(|c| c.chunk_kind)
            .collect();
        assert_eq!(
            kinds,
            vec![
                ChunkKind::Wording,
                ChunkKind::RevisionHistory,
                ChunkKind::RevisionHistory
            ]
        );
        let code_in_wording = "## Wording\n\n```cpp\nint x;\n```\n";
        assert_eq!(
            chunk_markdown(code_in_wording, &meta())[0].chunk_kind,
            ChunkKind::Wording
        );
    }

    #[test]
    fn chunk_kind_names_match_index_values() {
        let names: Vec<&str> = [
            ChunkKind::Body,
            ChunkKind::Wording,
            ChunkKind::Code,
            ChunkKind::RevisionHistory,
            ChunkKind::FrontMatter,
        ]
        .iter()
        .map(|k| k.as_str())
        .collect();
        assert_eq!(
            names,
            [
                "body",
                "wording",
                "code",
                "revision_history",
                "front_matter"
            ]
        );
    }

    #[test]
    fn line_ranges_match_fixture_slices() {
        let chunks = chunk_markdown(FIXTURE_P1000R2, &meta());
        assert_line_ranges(FIXTURE_P1000R2, &chunks);
        let lines: Vec<&str> = FIXTURE_P1000R2.lines().collect();
        assert_eq!(chunks[0].start_line, 0);
        assert_eq!(chunks.last().map(|c| c.end_line), Some(lines.len()));
        let design = chunks
            .iter()
            .find(|c| c.section_path == path(&["Design"]))
            .map(|c| (c.start_line, lines[c.start_line]));
        let design_line = lines.iter().position(|l| *l == "## Design");
        assert_eq!(design.map(|d| d.0), design_line);
        assert_eq!(design.map(|d| d.1), Some("## Design"));
    }

    #[test]
    fn handles_missing_front_matter_and_preamble() {
        let md = "Loose preamble.\n\n## Section\n\nText.";
        let chunks = chunk_markdown(md, &meta());
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].section_path.is_empty());
        assert_eq!(chunks[0].chunk_kind, ChunkKind::Body);
        assert_eq!(chunks[0].text, "Loose preamble.");
        assert_line_ranges(md, &chunks);
        assert!(chunk_markdown("", &meta()).is_empty());
    }

    #[test]
    fn prefix_carries_id_title_authors_and_section_path() {
        let prefix = build_prefix(&meta(), &path(&["Design", "Overflow"]));
        assert_eq!(
            prefix,
            "Paper: P1000R2\n\
             Title: std::ring_span, a fixed-capacity circular view\n\
             Authors: Ada Lovelace, Charles Babbage\n\
             Section: Design > Overflow\n\n"
        );
        let bare = PrefixMeta {
            paper_id: "N5001".to_owned(),
            title: "Draft".to_owned(),
            authors: Vec::new(),
        };
        assert_eq!(build_prefix(&bare, &[]), "Paper: N5001\nTitle: Draft\n\n");
    }

    #[test]
    fn indexed_text_is_prefix_plus_text() {
        let chunks = chunk_markdown(FIXTURE_P1000R2, &meta());
        for chunk in &chunks {
            let prefix = build_prefix(&meta(), &chunk.section_path);
            assert_eq!(chunk.indexed_text, format!("{prefix}{}", chunk.text));
            assert!(!chunk.text.contains("Paper: P1000R2"));
        }
    }

    #[test]
    fn chunking_is_deterministic() {
        assert_eq!(
            chunk_markdown(FIXTURE_P1000R2, &meta()),
            chunk_markdown(FIXTURE_P1000R2, &meta())
        );
    }

    #[test]
    fn crlf_lines_slice_like_lf() {
        let md = "## A\r\n\r\nText one.\r\n\r\n## B\r\n\r\nText two.\r\n";
        let chunks = chunk_markdown(md, &meta());
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].text, "## B\n\nText two.");
        assert_line_ranges(md, &chunks);
    }

    #[test]
    fn token_estimate_counts_words_and_punctuation() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a b"), 2);
        assert_eq!(estimate_tokens("std::vector"), 5);
        assert_eq!(estimate_tokens("internationalization"), 5);
    }
}
