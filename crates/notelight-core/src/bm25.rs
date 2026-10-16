//! Okapi BM25 keyword search over an immutable, serialized inverted index.
//!
//! [`Bm25Builder`] collects tokenized documents and serializes them to bytes.
//! [`Bm25`] answers queries directly from those bytes, whether they live in a
//! `Vec<u8>` or a memory-mapped file, without deserializing the index first.
//!
//! File layout (little-endian, header described in `docs/index-format.md`):
//!
//! ```text
//! header        magic "NLBM", version 1, a = n_docs, b = n_terms
//! params        avgdl f32, k1 f32, b f32, reserved u32
//! doc_lens      [u32; n_docs]          tokens per document
//! term_offsets  [u32; n_terms + 1]     byte ranges into term_bytes
//! post_offsets  [u32; n_terms + 1]     entry ranges into postings
//! term_bytes    UTF-8, terms sorted, padded to 4 bytes
//! postings      [(doc u32, tf u32); P] grouped by term, sorted by doc
//! ```

use std::collections::HashMap;

use crate::bin_format::{self, HEADER_LEN, f32_at, to_u32, u32_at};
use crate::error::{Error, Result};
use crate::tokenize::tokenize;
use crate::topk::top_k;

const MAGIC: [u8; 4] = *b"NLBM";
const VERSION: u16 = 1;

/// BM25 parameters. The defaults (`k1 = 1.2`, `b = 0.75`) are the usual ones.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Params {
    /// Term-frequency saturation. Higher values reward repeated terms more.
    pub k1: f32,
    /// Length normalization, from 0 (none) to 1 (full).
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

/// Accumulates documents for a BM25 index.
#[derive(Debug, Default)]
pub struct Bm25Builder {
    doc_lens: Vec<u32>,
    postings: HashMap<String, Vec<(u32, u32)>>,
}

impl Bm25Builder {
    /// Create an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Tokenize `text` and add it as the next document. Returns its id.
    pub fn add_text(&mut self, text: &str) -> u32 {
        self.add_tokens(&tokenize(text))
    }

    /// Add an already tokenized document. Returns its id.
    pub fn add_tokens(&mut self, tokens: &[String]) -> u32 {
        let doc = self.doc_lens.len() as u32;
        let mut tf: HashMap<&str, u32> = HashMap::new();
        for t in tokens {
            *tf.entry(t.as_str()).or_insert(0) += 1;
        }
        for (term, count) in tf {
            self.postings
                .entry(term.to_string())
                .or_default()
                .push((doc, count));
        }
        self.doc_lens.push(tokens.len() as u32);
        doc
    }

    /// Number of documents added so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.doc_lens.len()
    }

    /// Whether no documents have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.doc_lens.is_empty()
    }

    /// Serialize the index.
    ///
    /// # Errors
    /// Fails if the index exceeds the format's 32-bit limits.
    pub fn finish(self, params: Bm25Params) -> Result<Vec<u8>> {
        let n_docs = to_u32(self.doc_lens.len(), "document count")?;
        let mut terms: Vec<(String, Vec<(u32, u32)>)> = self.postings.into_iter().collect();
        terms.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let n_terms = to_u32(terms.len(), "term count")?;

        let total: u64 = self.doc_lens.iter().map(|&l| u64::from(l)).sum();
        let avgdl = if n_docs == 0 {
            0.0
        } else {
            (total as f64 / f64::from(n_docs)) as f32
        };

        let mut w = bin_format::Writer::with_header(MAGIC, VERSION, n_docs, n_terms);
        w.f32(avgdl);
        w.f32(params.k1);
        w.f32(params.b);
        w.u32(0);
        for &l in &self.doc_lens {
            w.u32(l);
        }

        let mut byte_pos = 0usize;
        w.u32(0);
        for (term, _) in &terms {
            byte_pos += term.len();
            w.u32(to_u32(byte_pos, "term dictionary size")?);
        }
        let mut entry_pos = 0usize;
        w.u32(0);
        for (_, posts) in &terms {
            entry_pos += posts.len();
            w.u32(to_u32(entry_pos, "posting count")?);
        }
        for (term, _) in &terms {
            w.bytes(term.as_bytes());
        }
        w.pad4();
        for (_, posts) in &mut terms {
            posts.sort_unstable_by_key(|&(doc, _)| doc);
            for &(doc, tf) in posts.iter() {
                w.u32(doc);
                w.u32(tf);
            }
        }
        Ok(w.into_inner())
    }
}

/// A read-only BM25 index backed by its serialized bytes.
#[derive(Debug)]
pub struct Bm25<B> {
    bytes: B,
    n_docs: usize,
    n_terms: usize,
    avgdl: f32,
    params: Bm25Params,
    doc_lens_at: usize,
    term_offsets_at: usize,
    post_offsets_at: usize,
    term_bytes_at: usize,
    postings_at: usize,
}

impl<B: AsRef<[u8]>> Bm25<B> {
    /// Open an index from its bytes, validating every section's bounds.
    ///
    /// # Errors
    /// Fails if the bytes are not a well-formed BM25 file.
    pub fn from_bytes(bytes: B) -> Result<Self> {
        let data = bytes.as_ref();
        let header = bin_format::read_header(data, MAGIC, VERSION, "bm25")?;
        let (n_docs, n_terms) = (header.a as usize, header.b as usize);
        // Layout math in u64 so hostile headers cannot overflow `usize`
        // (32 bits on WebAssembly); offsets are converted back only after
        // they are known to lie inside the file.
        let len = data.len() as u64;
        let params_at = HEADER_LEN;
        let doc_lens_at = params_at + 16;
        let term_offsets_at = doc_lens_at as u64 + 4 * u64::from(header.a);
        let post_offsets_at = term_offsets_at + 4 * (u64::from(header.b) + 1);
        let term_bytes_at = post_offsets_at + 4 * (u64::from(header.b) + 1);
        if len < term_bytes_at {
            return Err(Error::format("bm25: truncated offset tables"));
        }
        let (term_offsets_at, post_offsets_at, term_bytes_at) = (
            term_offsets_at as usize,
            post_offsets_at as usize,
            term_bytes_at as usize,
        );
        let term_bytes_len = u64::from(u32_at(data, term_offsets_at + 4 * n_terms));
        let postings_at = term_bytes_at as u64 + term_bytes_len.div_ceil(4) * 4;
        let n_postings = u64::from(u32_at(data, post_offsets_at + 4 * n_terms));
        if len != postings_at + 8 * n_postings {
            return Err(Error::format("bm25: size does not match its tables"));
        }
        let postings_at = postings_at as usize;
        let index = Self {
            n_docs,
            n_terms,
            avgdl: f32_at(data, params_at),
            params: Bm25Params {
                k1: f32_at(data, params_at + 4),
                b: f32_at(data, params_at + 8),
            },
            doc_lens_at,
            term_offsets_at,
            post_offsets_at,
            term_bytes_at,
            postings_at,
            bytes,
        };
        index.validate()?;
        Ok(index)
    }

    /// Check that offset tables are monotonic, terms are valid UTF-8 in
    /// sorted order, and each term's postings reference existing documents
    /// in strictly increasing order. Done once at open time so queries can
    /// index (and binary-search postings) without further checks.
    fn validate(&self) -> Result<()> {
        let data = self.bytes.as_ref();
        let mut prev_term: Option<&str> = None;
        for t in 0..self.n_terms {
            let (s, e) = (self.term_offset(t), self.term_offset(t + 1));
            let (ps, pe) = (self.post_offset(t), self.post_offset(t + 1));
            if s > e || ps > pe {
                return Err(Error::format("bm25: offsets are not monotonic"));
            }
            let term = std::str::from_utf8(&data[self.term_bytes_at + s..self.term_bytes_at + e])
                .map_err(|_| Error::format("bm25: term is not valid UTF-8"))?;
            if prev_term.is_some_and(|p| p >= term) {
                return Err(Error::format("bm25: terms are not strictly sorted"));
            }
            prev_term = Some(term);
        }
        for t in 0..self.n_terms {
            let mut prev_doc: Option<u32> = None;
            for i in self.post_offset(t)..self.post_offset(t + 1) {
                let doc = u32_at(data, self.postings_at + 8 * i);
                if doc as usize >= self.n_docs {
                    return Err(Error::format("bm25: posting references a missing document"));
                }
                if prev_doc.is_some_and(|p| p >= doc) {
                    return Err(Error::format("bm25: postings are not sorted by document"));
                }
                prev_doc = Some(doc);
            }
        }
        Ok(())
    }

    fn term_offset(&self, t: usize) -> usize {
        u32_at(self.bytes.as_ref(), self.term_offsets_at + 4 * t) as usize
    }

    fn post_offset(&self, t: usize) -> usize {
        u32_at(self.bytes.as_ref(), self.post_offsets_at + 4 * t) as usize
    }

    fn term(&self, t: usize) -> &[u8] {
        let start = self.term_bytes_at + self.term_offset(t);
        let end = self.term_bytes_at + self.term_offset(t + 1);
        &self.bytes.as_ref()[start..end]
    }

    /// Binary-search the sorted term dictionary.
    fn find_term(&self, term: &str) -> Option<usize> {
        let (mut lo, mut hi) = (0, self.n_terms);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.term(mid).cmp(term.as_bytes()) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// Number of documents in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_docs
    }

    /// Whether the index has no documents.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_docs == 0
    }

    /// Number of documents that contain `term` (after tokenization rules).
    #[must_use]
    pub fn doc_freq(&self, term: &str) -> usize {
        self.find_term(term)
            .map_or(0, |t| self.post_offset(t + 1) - self.post_offset(t))
    }

    /// Whether document `doc` contains every one of `terms` (already
    /// tokenized). A binary search per term over its sorted postings.
    #[must_use]
    pub fn contains_all(&self, doc: u32, terms: &[String]) -> bool {
        let data = self.bytes.as_ref();
        terms.iter().all(|term| {
            self.find_term(term).is_some_and(|t| {
                let (mut lo, mut hi) = (self.post_offset(t), self.post_offset(t + 1));
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    match u32_at(data, self.postings_at + 8 * mid).cmp(&doc) {
                        std::cmp::Ordering::Less => lo = mid + 1,
                        std::cmp::Ordering::Greater => hi = mid,
                        std::cmp::Ordering::Equal => return true,
                    }
                }
                false
            })
        })
    }

    /// Tokenize `query` and return the `k` best `(doc, score)` pairs.
    #[must_use]
    pub fn search(&self, query: &str, k: usize) -> Vec<(u32, f32)> {
        self.search_tokens(&tokenize(query), k)
    }

    /// Score pre-tokenized query terms. Repeated terms count once.
    #[must_use]
    pub fn search_tokens(&self, terms: &[String], k: usize) -> Vec<(u32, f32)> {
        let data = self.bytes.as_ref();
        let mut unique: Vec<&str> = terms.iter().map(String::as_str).collect();
        unique.sort_unstable();
        unique.dedup();

        let mut scores = vec![0.0_f32; self.n_docs];
        let mut touched: Vec<u32> = Vec::new();
        let n = self.n_docs as f32;
        let Bm25Params { k1, b } = self.params;
        let avgdl = self.avgdl.max(f32::MIN_POSITIVE);

        for term in unique {
            let Some(t) = self.find_term(term) else {
                continue;
            };
            let (start, end) = (self.post_offset(t), self.post_offset(t + 1));
            let df = (end - start) as f32;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for i in start..end {
                let pos = self.postings_at + 8 * i;
                let doc = u32_at(data, pos);
                let tf = u32_at(data, pos + 4) as f32;
                let dl = u32_at(data, self.doc_lens_at + 4 * doc as usize) as f32;
                let norm = k1 * (1.0 - b + b * dl / avgdl);
                let slot = &mut scores[doc as usize];
                if *slot == 0.0 {
                    touched.push(doc);
                }
                *slot += idf * tf * (k1 + 1.0) / (tf + norm);
            }
        }
        top_k(touched.into_iter().map(|d| (d, scores[d as usize])), k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(docs: &[&str]) -> Bm25<Vec<u8>> {
        let mut b = Bm25Builder::new();
        for d in docs {
            b.add_text(d);
        }
        Bm25::from_bytes(b.finish(Bm25Params::default()).unwrap()).unwrap()
    }

    #[test]
    fn ranks_matching_documents() {
        let idx = index(&[
            "the quick brown fox",
            "fox fox fox jumps",
            "lazy dog sleeps",
        ]);
        let hits = idx.search("fox", 10);
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), [1, 0]);
        assert!(hits[0].1 > hits[1].1);
        assert_eq!(idx.search("cat", 10), []);
    }

    #[test]
    fn rare_terms_weigh_more_than_common_ones() {
        let idx = index(&["alpha common", "beta common", "gamma common", "alpha"]);
        // "beta" is rare, "common" is everywhere: doc 1 should win.
        assert_eq!(idx.search("beta common", 1)[0].0, 1);
    }

    #[test]
    fn repeated_query_terms_count_once() {
        let idx = index(&["rust code", "python code"]);
        assert_eq!(idx.search("rust", 2), idx.search("rust rust rust", 2));
    }

    #[test]
    fn contains_all_checks_every_term() {
        let idx = index(&["lock file here", "lock only", "file only"]);
        let terms = |ws: &[&str]| ws.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert!(idx.contains_all(0, &terms(&["lock", "file"])));
        assert!(!idx.contains_all(1, &terms(&["lock", "file"])));
        assert!(idx.contains_all(2, &terms(&["file"])));
        assert!(!idx.contains_all(2, &terms(&["missing"])));
        assert!(idx.contains_all(1, &[]));
    }

    #[test]
    fn doc_freq_and_len() {
        let idx = index(&["a1 b2", "b2 c3", "c3"]);
        assert_eq!(idx.len(), 3);
        assert_eq!(idx.doc_freq("b2"), 2);
        assert_eq!(idx.doc_freq("zz"), 0);
    }

    #[test]
    fn empty_index_answers_nothing() {
        let idx = index(&[]);
        assert!(idx.is_empty());
        assert_eq!(idx.search("anything", 5), []);
    }

    #[test]
    fn rejects_corrupt_bytes() {
        let mut b = Bm25Builder::new();
        b.add_text("hello world");
        let mut bytes = b.finish(Bm25Params::default()).unwrap();
        assert!(Bm25::from_bytes(&bytes[..10]).is_err());
        let last = bytes.len() - 8;
        bytes[last..last + 4].copy_from_slice(&99u32.to_le_bytes()); // doc id out of range
        assert!(Bm25::from_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_postings_out_of_document_order() {
        let mut b = Bm25Builder::new();
        b.add_text("shared");
        b.add_text("shared");
        let mut bytes = b.finish(Bm25Params::default()).unwrap();
        // The only term's two postings are the last 16 bytes: (0, tf), (1, tf).
        let first = bytes.len() - 16;
        bytes[first..first + 4].copy_from_slice(&1u32.to_le_bytes());
        bytes[first + 8..first + 12].copy_from_slice(&0u32.to_le_bytes());
        let err = Bm25::from_bytes(bytes).unwrap_err();
        assert!(err.to_string().contains("not sorted"), "{err}");
    }
}
