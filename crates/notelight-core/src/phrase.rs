//! Phrase matching: quoted phrases and in-order query terms.
//!
//! BM25 treats a query as a bag of words and vector search ranks by meaning,
//! so neither rewards a chunk for containing the query's words *in order*.
//! This module adds that signal on top of fusion (see
//! [`crate::index::IndexReader::search`]):
//!
//! - A query may quote phrases: `"lock file" windows`. Chunks that contain
//!   every quoted phrase rank above all other chunks.
//! - A query of two or more terms whose terms appear in a chunk consecutively
//!   and in order gets a score bonus, quoted or not.
//!
//! Phrases are compared as sequences of [`tokenize`] terms, the same terms
//! BM25 uses, so case, punctuation and stopwords are ignored: `"Lock-File"`
//! matches "lock file", and `"lock the file"` matches "lock a file".

use crate::tokenize::tokenize;

/// A query split into its quoted phrases and all of its terms.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PhraseQuery {
    /// Terms of each quoted phrase that has at least one term.
    pub quoted: Vec<Vec<String>>,
    /// Terms of the whole query, quoted parts included, in order.
    pub terms: Vec<String>,
}

impl PhraseQuery {
    /// Parse `query`. Phrases are delimited by straight (`"`) or curly
    /// (`“ ”`) double quotes; an unclosed quote runs to the end of the query.
    #[must_use]
    pub fn parse(query: &str) -> Self {
        let mut quoted = Vec::new();
        let mut inside = false;
        let mut current = String::new();
        for c in query.chars() {
            if matches!(c, '"' | '\u{201C}' | '\u{201D}') {
                if inside {
                    push_phrase(&mut quoted, &current);
                    current.clear();
                }
                inside = !inside;
            } else if inside {
                current.push(c);
            }
        }
        if inside {
            push_phrase(&mut quoted, &current);
        }
        Self {
            quoted,
            terms: tokenize(query),
        }
    }

    /// Whether the query quotes at least one phrase.
    #[must_use]
    pub fn has_quoted(&self) -> bool {
        !self.quoted.is_empty()
    }

    /// Whether the in-order bonus can apply: the query has two or more terms.
    #[must_use]
    pub fn has_sequence(&self) -> bool {
        self.terms.len() >= 2
    }
}

fn push_phrase(out: &mut Vec<Vec<String>>, text: &str) {
    let terms = tokenize(text);
    if !terms.is_empty() {
        out.push(terms);
    }
}

/// Whether `needle` occurs in `haystack` as a contiguous run of terms.
/// An empty needle occurs everywhere.
#[must_use]
pub fn contains_sequence(haystack: &[String], needle: &[String]) -> bool {
    needle.is_empty() || haystack.windows(needle.len()).any(|w| w == needle)
}

/// How one chunk matches a [`PhraseQuery`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PhraseMatch {
    /// The chunk contains every quoted phrase (false if none was quoted).
    pub all_quoted: bool,
    /// The chunk contains all query terms in order (false for one term).
    pub sequence: bool,
}

/// Match the terms of one chunk against `query`.
#[must_use]
pub fn match_terms(query: &PhraseQuery, chunk_terms: &[String]) -> PhraseMatch {
    PhraseMatch {
        all_quoted: query.has_quoted()
            && query
                .quoted
                .iter()
                .all(|p| contains_sequence(chunk_terms, p)),
        sequence: query.has_sequence() && contains_sequence(chunk_terms, &query.terms),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(words: &[&str]) -> Vec<String> {
        words.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn parses_quoted_phrases_and_all_terms() {
        let q = PhraseQuery::parse(r#""lock file" windows "Commit Protocol""#);
        assert_eq!(
            q.quoted,
            [terms(&["lock", "file"]), terms(&["commit", "protocol"])]
        );
        assert_eq!(
            q.terms,
            terms(&["lock", "file", "windows", "commit", "protocol"])
        );
    }

    #[test]
    fn curly_quotes_and_unclosed_quote() {
        let q = PhraseQuery::parse("“exact match” and \"open ended");
        assert_eq!(
            q.quoted,
            [terms(&["exact", "match"]), terms(&["open", "ended"])]
        );
    }

    #[test]
    fn empty_or_stopword_phrases_are_dropped() {
        assert!(!PhraseQuery::parse(r#"a "" b "the of""#).has_quoted());
        assert!(!PhraseQuery::parse("no quotes here").has_quoted());
    }

    #[test]
    fn sequence_needs_two_terms() {
        assert!(!PhraseQuery::parse("bread").has_sequence());
        assert!(!PhraseQuery::parse("the bread").has_sequence());
        assert!(PhraseQuery::parse("sourdough bread").has_sequence());
    }

    #[test]
    fn contains_sequence_finds_contiguous_runs_only() {
        let hay = terms(&["one", "writer", "lock", "file", "here"]);
        assert!(contains_sequence(&hay, &terms(&["lock", "file"])));
        assert!(contains_sequence(&hay, &[]));
        assert!(!contains_sequence(&hay, &terms(&["writer", "file"])));
        assert!(!contains_sequence(&hay, &terms(&["file", "lock"])));
        assert!(!contains_sequence(
            &terms(&["lock"]),
            &terms(&["lock", "file"])
        ));
    }

    #[test]
    fn match_ignores_case_punctuation_and_stopwords() {
        let chunk = tokenize("One writer, enforced with a Lock-File.");
        let m = match_terms(&PhraseQuery::parse(r#""lock the file""#), &chunk);
        assert!(m.all_quoted);
        assert!(m.sequence);
        let m = match_terms(&PhraseQuery::parse("file lock"), &chunk);
        assert_eq!(m, PhraseMatch::default());
    }

    #[test]
    fn every_quoted_phrase_must_match() {
        let chunk = tokenize("the lock file is removed after commit");
        let both = PhraseQuery::parse(r#""lock file" "after commit""#);
        assert!(match_terms(&both, &chunk).all_quoted);
        let one_missing = PhraseQuery::parse(r#""lock file" "before commit""#);
        assert!(!match_terms(&one_missing, &chunk).all_quoted);
    }
}
