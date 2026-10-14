//! Property tests for the text invariants of design §13.

use proptest::prelude::*;

use super::chunks::tests::Words;
use super::{
    Chunk, ChunkSettings, Document, MIN_MAX_TOKENS, PaperLabel, SPECIAL_TOKENS, TokenCounter,
    chunk_document, outline,
};

fn line() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::new()),
        Just("   ".to_owned()),
        "[a-z ]{1,40}".prop_map(|s| format!("w {s}")),
        (1usize..=6, "[a-z]{1,10}").prop_map(|(n, s)| format!("{} {s}", "#".repeat(n))),
        (1u32..20, "[a-z]{1,10}").prop_map(|(n, s)| format!("{n}. **{s}**")),
        (1u32..20, "[a-z]{1,10}").prop_map(|(n, s)| format!("{n}. {s}")),
        Just("```cpp".to_owned()),
        Just("```".to_owned()),
        Just("    ~~~".to_owned()),
        Just("~~~~".to_owned()),
        "[a-z]{1,8}".prop_map(|s| format!("| {s} | x |")),
        "[a-z ]{1,20}".prop_map(|s| format!("- {s}")),
        "[a-z ]{1,20}".prop_map(|s| format!("  * {s}")),
        Just("a \r b".to_owned()),
        Just("<table border=\"1\">".to_owned()),
        Just("<tr>".to_owned()),
        "[a-z ]{1,30}".prop_map(|s| format!("<td>w {s}</td>")),
        Just("</tr>".to_owned()),
        Just("</table>".to_owned()),
        "[a-z .]{200,1500}".prop_map(|s| format!("w {s}")),
        "[a-z]{1,8}".prop_map(|s| format!("{s}é{s}— {s}")),
    ]
}

fn markdown() -> impl Strategy<Value = String> {
    (prop::collection::vec(line(), 0..120), any::<bool>()).prop_map(|(lines, trailing)| {
        let mut text = lines.join("\n");
        if trailing {
            text.push('\n');
        }
        text
    })
}

fn settings() -> impl Strategy<Value = ChunkSettings> {
    (MIN_MAX_TOKENS..400)
        .prop_flat_map(|max| (1..=max / 2, Just(max)))
        .prop_flat_map(|(target, max)| (Just(target), 0..target, Just(max)))
        .prop_map(|(target, overlap, max)| ChunkSettings {
            target_tokens: target,
            overlap_max_tokens: overlap,
            max_tokens: max,
        })
}

/// The byte offsets in the paper text of a chunk's start and end.
fn span(doc: &Document<'_>, chunk: &Chunk) -> (usize, usize) {
    let offset = |line: usize| -> usize { doc.lines()[..line].iter().map(|l| l.len() + 1).sum() };
    let last = chunk.last.zero_based();
    let end = chunk.end_byte.unwrap_or(doc.lines()[last].len());
    (
        offset(chunk.first.zero_based()) + chunk.start_byte,
        offset(last) + end,
    )
}

const LABEL: PaperLabel<'static> = PaperLabel {
    paper_id: "p1r0",
    title: "Title",
};

proptest! {
    #[test]
    fn sections_tile_the_paper(md in markdown()) {
        let doc = Document::parse(&md);
        let sections = doc.sections();
        if doc.line_count() == 0 {
            prop_assert!(sections.is_empty());
            return Ok(());
        }
        prop_assert_eq!(sections[0].first.get(), 1);
        prop_assert_eq!(sections.last().map(|s| s.last.get()), Some(doc.line_count()));
        for (position, section) in sections.iter().enumerate() {
            prop_assert_eq!(section.index.get(), position + 1);
            prop_assert!(section.first <= section.last);
        }
        for pair in sections.windows(2) {
            prop_assert_eq!(pair[1].first.get(), pair[0].last.get() + 1);
        }
        let joined: Vec<String> = sections.iter().map(|s| doc.text(s.first, s.last)).collect();
        prop_assert_eq!(joined.join("\n"), doc.lines().join("\n"));
    }

    #[test]
    fn chunks_cover_each_section_and_stay_inside_it(md in markdown(), s in settings()) {
        let doc = Document::parse(&md);
        let text = doc.lines().join("\n");
        let sections = doc.sections();
        let chunks = chunk_document(&doc, &sections, s, &Words, LABEL);
        for section in &sections {
            let mine: Vec<_> = chunks.iter().filter(|c| c.section == section.index).collect();
            prop_assert!(!mine.is_empty());
            prop_assert_eq!(mine[0].first, section.first);
            prop_assert_eq!(mine[0].start_byte, 0);
            prop_assert_eq!(mine.last().map(|c| (c.last, c.end_byte)), Some((section.last, None)));
            for (position, chunk) in mine.iter().enumerate() {
                prop_assert_eq!(chunk.number.get(), position + 1);
                prop_assert!(section.first <= chunk.first && chunk.last <= section.last);
                let (start, end) = span(&doc, chunk);
                prop_assert!(start < end || (start == end && chunk.first == chunk.last));
                prop_assert_eq!(chunk.text(&doc), &text[start..end]);
            }
            for pair in mine.windows(2) {
                let (_, previous_end) = span(&doc, pair[0]);
                let (start, end) = span(&doc, pair[1]);
                // A chunk starting a line may follow the previous line's
                // newline; one starting inside a line must leave no gap.
                let gap = usize::from(pair[1].start_byte == 0);
                prop_assert!(start <= previous_end + gap);
                prop_assert!(end > previous_end);
            }
        }
    }

    #[test]
    fn every_chunk_fits_the_size_limit(md in markdown(), s in settings()) {
        let doc = Document::parse(&md);
        for chunk in chunk_document(&doc, &doc.sections(), s, &Words, LABEL) {
            let tokens = Words.count(&chunk.embed_text(&doc)) + SPECIAL_TOKENS;
            prop_assert!(tokens <= s.max_tokens, "{} > {}: {:?}", tokens, s.max_tokens, chunk);
        }
    }

    #[test]
    fn analysis_is_deterministic(md in markdown(), s in settings()) {
        let first = Document::parse(&md);
        let second = Document::parse(&md);
        prop_assert_eq!(first.sections(), second.sections());
        prop_assert_eq!(
            chunk_document(&first, &first.sections(), s, &Words, LABEL),
            chunk_document(&second, &second.sections(), s, &Words, LABEL)
        );
    }

    #[test]
    fn every_heading_lies_in_its_section(md in markdown()) {
        let doc = Document::parse(&md);
        let sections = doc.sections();
        let entries = outline(&doc, &sections);
        prop_assert_eq!(entries.len(), doc.headings().len());
        for entry in entries {
            let section = &sections[entry.section_index.zero_based()];
            prop_assert!(section.first <= entry.line && entry.line <= section.last);
            prop_assert_eq!(entry.starts_section, section.boundary == Some(entry.line));
        }
    }
}
