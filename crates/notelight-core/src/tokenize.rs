//! Text tokenization for BM25.
//!
//! The tokenizer is deliberately simple and language-agnostic:
//!
//! - Split on any character that is not alphanumeric (Unicode-aware), so
//!   `snake_case`, `kebab-case` and `path/to/file` break into words.
//! - Lowercase every token.
//! - For identifiers written in `camelCase` or `PascalCase`, also emit the
//!   whole lowercased identifier, so both "getUserName" and "user name" match.
//! - Drop a short list of English stopwords and single-letter tokens.
//!
//! There is no stemming: "index" and "indexing" are different terms. The
//! embedding side of hybrid search covers morphology; keeping BM25 exact keeps
//! it predictable for names, error codes and identifiers.

/// Common English words that carry little signal for keyword search.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

fn is_stopword(token: &str) -> bool {
    STOPWORDS.binary_search(&token).is_ok()
}

/// Split `text` into lowercase BM25 terms.
#[must_use]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        let parts = split_camel_case(word);
        if parts.len() > 1 {
            // Emit the parts and the joined identifier: getUserName -> get, user, name, getusername.
            for part in &parts {
                push_term(&mut out, part);
            }
        }
        push_term(&mut out, word);
    }
    out
}

fn push_term(out: &mut Vec<String>, raw: &str) {
    // Lowercasing can introduce combining marks ("İ" -> "i\u{307}"); keep only
    // letters and digits so every term is a clean alphanumeric string.
    let term: String = raw
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect();
    if term.is_empty() {
        return;
    }
    if term.chars().count() < 2 && !term.chars().all(|c| c.is_ascii_digit()) {
        return; // single letters are noise, but keep single digits ("v2" splits to "v", "2")
    }
    if is_stopword(&term) {
        return;
    }
    out.push(term);
}

/// Split `camelCase`, `PascalCase` and `HTTPServer`-style words at case changes.
/// Words without internal case changes are returned whole.
fn split_camel_case(word: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = word.char_indices().collect();
    let mut parts = Vec::new();
    let mut start = 0;
    for w in 1..chars.len() {
        let (idx, cur) = chars[w];
        let prev = chars[w - 1].1;
        let next = chars.get(w + 1).map(|&(_, c)| c);
        let lower_to_upper = prev.is_lowercase() && cur.is_uppercase();
        // "HTTPServer": split before the 'S' that starts a capitalised word.
        let acronym_end =
            prev.is_uppercase() && cur.is_uppercase() && next.is_some_and(char::is_lowercase);
        let letter_digit = prev.is_alphabetic() != cur.is_alphabetic();
        if lower_to_upper || acronym_end || letter_digit {
            parts.push(&word[start..idx]);
            start = idx;
        }
    }
    parts.push(&word[start..]);
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopwords_are_sorted_for_binary_search() {
        assert!(STOPWORDS.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn lowercases_and_splits_on_punctuation() {
        assert_eq!(
            tokenize("Fix the auth-timeout in src/server.rs!"),
            ["fix", "auth", "timeout", "src", "server", "rs"]
        );
    }

    #[test]
    fn camel_case_emits_parts_and_whole() {
        assert_eq!(
            tokenize("getUserName"),
            ["get", "user", "name", "getusername"]
        );
        assert_eq!(tokenize("HTTPServer"), ["http", "server", "httpserver"]);
        assert_eq!(tokenize("snake_case"), ["snake", "case"]);
    }

    #[test]
    fn digits_split_from_letters() {
        assert_eq!(
            tokenize("utf8 v2 E1104"),
            ["utf", "8", "utf8", "2", "v2", "1104", "e1104"]
        );
    }

    #[test]
    fn lowercasing_never_leaves_combining_marks() {
        assert_eq!(tokenize("İstanbul"), ["istanbul"]);
    }

    #[test]
    fn unicode_words_are_kept() {
        assert_eq!(
            tokenize("Привет, мир — naïve café"),
            ["привет", "мир", "naïve", "café"]
        );
    }
}
