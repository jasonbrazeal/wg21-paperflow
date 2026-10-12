//! Property tests for the text invariants of design §13.

use proptest::prelude::*;

use super::chunks::tests::Words;
use super::{ChunkSettings, Document, PaperLabel, chunk_document, outline};

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
    (1usize..60, 0usize..30, 40usize..200).prop_map(|(target, overlap, max)| ChunkSettings {
        target_tokens: target,
        overlap_max_tokens: overlap,
        max_input_tokens: max,
    })
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
        let sections = doc.sections();
        let chunks = chunk_document(&doc, &sections, s, &Words, LABEL);
        for section in &sections {
            let mine: Vec<_> = chunks.iter().filter(|c| c.section == section.index).collect();
            prop_assert!(!mine.is_empty());
            prop_assert_eq!(mine[0].first, section.first);
            prop_assert_eq!(mine.last().map(|c| c.last), Some(section.last));
            for (position, chunk) in mine.iter().enumerate() {
                prop_assert_eq!(chunk.number.get(), position + 1);
                prop_assert!(section.first <= chunk.first && chunk.last <= section.last);
                prop_assert!(chunk.first <= chunk.last);
                prop_assert_eq!(chunk.text(&doc), doc.lines()[chunk.first.zero_based()..chunk.last.get()].join("\n"));
            }
            for pair in mine.windows(2) {
                prop_assert!(pair[1].first.get() <= pair[0].last.get() + 1);
                prop_assert!(pair[1].last > pair[0].last);
            }
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
