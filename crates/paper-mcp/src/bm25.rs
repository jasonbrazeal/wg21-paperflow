//! BM25 settings (§7.2), the tokenizer `match_lines` uses (§6.8), and the
//! case and accent folding shared with the author filter (§6.8).
//!
//! Qdrant generates the BM25 vectors itself from the options built here.
//! The tokenizer reproduces Qdrant's `word` tokenizer with these options:
//! split on every character that is not alphanumeric, fold to ASCII, then
//! lower-case, with no stemming and no stopwords.

use std::collections::HashSet;

use serde_json::{Value, json};
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

use crate::text::LineNo;

/// The Qdrant model name for local BM25 inference.
pub const BM25_MODEL: &str = "qdrant/bm25";

/// The BM25 options sent with every document and query, for a vector whose
/// texts average `avg_len` tokens.
#[must_use]
pub fn options(avg_len: f64) -> Value {
    json!({
        "tokenizer": "word",
        "lowercase": true,
        "ascii_folding": true,
        "stemmer": { "type": "none" },
        "stopwords": { "custom": [] },
        "avg_len": avg_len,
    })
}

/// Letters that canonical decomposition does not reduce to ASCII.
fn fold_special(c: char) -> Option<&'static str> {
    Some(match c {
        'ß' => "ss",
        'ẞ' => "SS",
        'æ' => "ae",
        'Æ' => "AE",
        'œ' => "oe",
        'Œ' => "OE",
        'ø' => "o",
        'Ø' => "O",
        'đ' | 'ð' => "d",
        'Đ' | 'Ð' => "D",
        'ł' | 'ŀ' => "l",
        'Ł' | 'Ŀ' => "L",
        'ı' => "i",
        'þ' => "th",
        'Þ' => "TH",
        'ħ' => "h",
        'Ħ' => "H",
        'ŧ' => "t",
        'Ŧ' => "T",
        'ŋ' => "n",
        'Ŋ' => "N",
        _ => return None,
    })
}

/// Folds accented Latin letters to ASCII, leaving other characters alone.
#[must_use]
pub fn fold_to_ascii(text: &str) -> String {
    if text.is_ascii() {
        return text.to_owned();
    }
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii() {
            out.push(c);
        } else if let Some(folded) = fold_special(c) {
            out.push_str(folded);
        } else {
            let decomposed: String = c.nfkd().filter(|&d| !is_combining_mark(d)).collect();
            if !decomposed.is_empty() && decomposed.is_ascii() {
                out.push_str(&decomposed);
            } else {
                out.push(c);
            }
        }
    }
    out
}

/// Folds case and accents, for matching author names.
#[must_use]
pub fn fold(text: &str) -> String {
    fold_to_ascii(text).to_lowercase()
}

/// Splits text into BM25 tokens.
pub fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(fold)
}

/// The lines of `text` that contain a query token. `text` starts at line
/// `first`.
#[must_use]
pub fn match_lines(text: &str, first: LineNo, query: &str) -> Vec<LineNo> {
    let wanted: HashSet<String> = tokens(query).collect();
    if wanted.is_empty() {
        return Vec::new();
    }
    text.split('\n')
        .enumerate()
        .filter(|(_, line)| tokens(line).any(|token| wanted.contains(&token)))
        .filter_map(|(offset, _)| LineNo::new(first.get() + offset))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_split_on_non_alphanumerics_and_fold() {
        let found: Vec<String> =
            tokens("std::views::filter foo_bar don't Çağrı ﬁx Straße 日本語").collect();
        assert_eq!(
            found,
            vec![
                "std",
                "views",
                "filter",
                "foo",
                "bar",
                "don",
                "t",
                "cagri",
                "fix",
                "strasse",
                "日本語"
            ]
        );
    }

    #[test]
    fn folds_author_names() {
        assert!(fold("Eric Niebler").contains(&fold("iebler")));
        assert!(fold("Murat Can Çağrı").contains(&fold("cagri")));
        assert_eq!(fold("Gašper Ažman"), "gasper azman");
        assert_eq!(
            fold("Johel Ernesto Guerrero Peña"),
            "johel ernesto guerrero pena"
        );
    }

    #[test]
    fn match_lines_reports_lines_with_any_query_token() {
        let text = "intro\nfor (auto x : views::iota)\nnothing\nRanges and VIEWS";
        let start = LineNo::new(10).unwrap_or(LineNo::FIRST);
        let lines: Vec<usize> = match_lines(text, start, "views for")
            .into_iter()
            .map(LineNo::get)
            .collect();
        assert_eq!(lines, vec![11, 13]);
        assert!(match_lines(text, start, "::").is_empty());
    }
}
