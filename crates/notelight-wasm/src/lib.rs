//! WebAssembly bindings for `notelight-core`, loaded from JavaScript.
//!
//! The browser has no filesystem, so an index is opened from the bytes of one
//! generation's files (`manifest.json`, `docs.bin`, `bm25.bin` and, when the
//! index has embeddings, `vectors.bin`), however the page obtained them: a
//! directory picker, `fetch`, `IndexedDB`.
//!
//! ```js
//! import init, { Index } from "./pkg/notelight_wasm.js";
//! await init();
//! const index = new Index(manifestBytes, docsBytes, bm25Bytes, null);
//! for (const hit of index.search("phrase ranking", null, 10)) {
//!   console.log(hit.score, index.chunk(hit.id).path);
//! }
//! ```
//!
//! Results cross into JavaScript as plain objects with `camelCase` fields
//! (see the `TypeScript` section below). Keep this layer thin, like
//! `notelight-py`: convert types and delegate to the core.
//!
//! Every `#[wasm_bindgen]` method has a plain-Rust twin (`hits_bm25`,
//! `hits`, `record`, ...) that does the work; the exported method only
//! converts the result. The twins are what the native unit tests call, since
//! creating a `JsValue` outside a JavaScript host panics.

use serde::Serialize;
use wasm_bindgen::prelude::*;

use notelight_core::index::{ChunkRecord, IndexReader, SearchOpts};

/// Version of the core crate, for the page to display.
#[wasm_bindgen]
#[must_use]
pub fn version() -> String {
    notelight_core::VERSION.to_string()
}

#[wasm_bindgen(typescript_custom_section)]
const TS_TYPES: &str = r#"
/** One keyword-search result. */
export interface Bm25Hit {
  id: number;
  /** BM25 score (higher is better). */
  score: number;
}

/** One hybrid-search result. */
export interface Hit {
  id: number;
  /** Fused RRF score plus any phrase bonus (higher is better). */
  score: number;
  /** 1-based rank in the BM25 list, or null if BM25 did not return it. */
  bm25Rank: number | null;
  /** 1-based rank in the vector list, or null. */
  vectorRank: number | null;
}

/** A stored chunk: where it came from and its text. */
export interface Chunk {
  path: string;
  headingPath: string[];
  lineStart: number;
  lineEnd: number;
  text: string;
}
"#;

/// A keyword-search result as JavaScript sees it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Bm25Hit {
    /// Chunk id.
    pub id: u32,
    /// BM25 score.
    pub score: f32,
}

/// A hybrid-search result as JavaScript sees it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Hit {
    /// Chunk id.
    pub id: u32,
    /// Fused score.
    pub score: f32,
    /// 1-based rank in the BM25 list.
    pub bm25_rank: Option<u32>,
    /// 1-based rank in the vector list.
    pub vector_rank: Option<u32>,
}

/// A chunk record as JavaScript sees it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Chunk {
    /// Source path, relative to the indexed folder, `/`-separated.
    pub path: String,
    /// Enclosing headings, outermost first.
    pub heading_path: Vec<String>,
    /// 1-based first line.
    pub line_start: u32,
    /// 1-based last line (inclusive).
    pub line_end: u32,
    /// The chunk text.
    pub text: String,
}

impl From<ChunkRecord> for Chunk {
    fn from(r: ChunkRecord) -> Self {
        Self {
            path: r.path,
            heading_path: r.heading_path,
            line_start: r.line_start,
            line_end: r.line_end,
            text: r.text,
        }
    }
}

/// An opened index generation, held in WebAssembly memory.
///
/// JavaScript owns a handle to it; call `free()` when done (or rely on the
/// `FinalizationRegistry` that wasm-bindgen registers).
#[wasm_bindgen]
#[derive(Debug)]
pub struct Index {
    reader: IndexReader,
}

#[wasm_bindgen]
impl Index {
    /// Open an index from the bytes of one generation's files. Pass `null`
    /// (or omit) `vectors` for a keyword-only index.
    ///
    /// # Errors
    /// Throws if the files are invalid, disagree with each other or with the
    /// manifest, or were written by another schema version.
    #[wasm_bindgen(constructor)]
    pub fn new(
        manifest: &[u8],
        docs: Vec<u8>,
        bm25: Vec<u8>,
        vectors: Option<Vec<u8>>,
    ) -> Result<Index, JsError> {
        Self::open(manifest, docs, bm25, vectors).map_err(js_error)
    }

    /// Number of chunks.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn length(&self) -> usize {
        self.reader.len()
    }

    /// Embedding dimension, or 0 for a keyword-only index.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn dim(&self) -> u32 {
        self.reader.manifest().dim
    }

    /// Identifier of the embedding model the index was built with, if any.
    /// A query vector passed to [`Self::search`] must come from this model.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn embedder(&self) -> Option<String> {
        self.reader.manifest().embedder.clone()
    }

    /// Whether the index stores embeddings.
    #[wasm_bindgen(getter, js_name = hasVectors)]
    #[must_use]
    pub fn has_vectors(&self) -> bool {
        self.reader.has_vectors()
    }

    /// The generation number from the manifest.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.reader.manifest().generation
    }

    /// Keyword search: the `k` best chunks by BM25.
    ///
    /// # Errors
    /// Throws only if the results cannot be converted to JavaScript.
    #[wasm_bindgen(js_name = searchBm25, unchecked_return_type = "Bm25Hit[]")]
    pub fn search_bm25(&self, text: &str, k: usize) -> Result<JsValue, JsError> {
        to_js(&self.hits_bm25(text, k))
    }

    /// Hybrid search. With `vector` null it ranks by BM25 alone (plus
    /// phrase ranking); with a query embedding from the index's embedder it
    /// fuses BM25 and vector results. `vector` may be a `Float32Array` or a
    /// plain array of numbers.
    ///
    /// # Errors
    /// Throws if `vector` is given for a keyword-only index or has the wrong
    /// dimension.
    #[wasm_bindgen(unchecked_return_type = "Hit[]")]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "wasm-bindgen copies an optional array into an owned Vec"
    )]
    pub fn search(
        &self,
        text: &str,
        vector: Option<Vec<f32>>,
        k: usize,
    ) -> Result<JsValue, JsError> {
        let hits = self.hits(text, vector.as_deref(), k).map_err(js_error)?;
        to_js(&hits)
    }

    /// The stored record for chunk `id`.
    ///
    /// # Errors
    /// Throws if `id` is out of range.
    #[wasm_bindgen(unchecked_return_type = "Chunk")]
    pub fn chunk(&self, id: u32) -> Result<JsValue, JsError> {
        to_js(&self.record(id).map_err(js_error)?)
    }
}

/// The plain-Rust side of [`Index`], callable without a JavaScript host.
impl Index {
    /// Open an index from in-memory files.
    ///
    /// # Errors
    /// See [`IndexReader::from_parts`].
    pub fn open(
        manifest: &[u8],
        docs: Vec<u8>,
        bm25: Vec<u8>,
        vectors: Option<Vec<u8>>,
    ) -> notelight_core::Result<Self> {
        Ok(Self {
            reader: IndexReader::from_parts(manifest, docs, bm25, vectors)?,
        })
    }

    /// See [`Self::search_bm25`].
    #[must_use]
    pub fn hits_bm25(&self, text: &str, k: usize) -> Vec<Bm25Hit> {
        self.reader
            .search_bm25(text, k)
            .into_iter()
            .map(|(id, score)| Bm25Hit { id, score })
            .collect()
    }

    /// See [`Self::search`].
    ///
    /// # Errors
    /// See [`IndexReader::search`].
    pub fn hits(
        &self,
        text: &str,
        vector: Option<&[f32]>,
        k: usize,
    ) -> notelight_core::Result<Vec<Hit>> {
        let opts = SearchOpts {
            k,
            ..SearchOpts::default()
        };
        Ok(self
            .reader
            .search(text, vector, &opts)?
            .into_iter()
            .map(|h| Hit {
                id: h.id,
                score: h.score,
                bm25_rank: h.bm25_rank,
                vector_rank: h.vector_rank,
            })
            .collect())
    }

    /// See [`Self::chunk`].
    ///
    /// # Errors
    /// See [`IndexReader::chunk`].
    pub fn record(&self, id: u32) -> notelight_core::Result<Chunk> {
        self.reader.chunk(id).map(Chunk::from)
    }
}

/// Convert a core error into a JavaScript `Error` carrying its message.
#[allow(
    clippy::needless_pass_by_value,
    reason = "used as `map_err(js_error)`, which passes the error by value"
)]
fn js_error(e: notelight_core::Error) -> JsError {
    JsError::new(&e.to_string())
}

/// Serialize a result as plain JavaScript objects and arrays: `None` becomes
/// `null` (not `undefined`) and maps become objects (not `Map`s), as
/// `JSON.parse` would produce.
fn to_js<T: Serialize + ?Sized>(value: &T) -> Result<JsValue, JsError> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| JsError::new(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use notelight_core::index::{IndexWriter, WriterOptions};

    fn record(path: &str, heading: &str, text: &str) -> ChunkRecord {
        ChunkRecord {
            path: path.into(),
            heading_path: vec![heading.into()],
            line_start: 1,
            line_end: 3,
            text: text.into(),
        }
    }

    /// A two-chunk index built in memory, with 4-dimensional one-hot vectors
    /// when `with_vectors`.
    fn index(with_vectors: bool) -> Index {
        let dim = if with_vectors { 4 } else { 0 };
        let mut w = IndexWriter::new(WriterOptions {
            dim,
            embedder: with_vectors.then(|| "test".to_string()),
            ..WriterOptions::default()
        });
        let docs = [
            record("a.md", "Install", "Install with winget on Windows."),
            record("b.md", "Search", "Phrase ranking puts exact phrases first."),
        ];
        for (i, r) in docs.iter().enumerate() {
            let v = with_vectors.then(|| {
                let mut v = vec![0.0; 4];
                v[i] = 1.0;
                v
            });
            w.add(r, v.as_deref()).unwrap();
        }
        let g = w.build(1).unwrap();
        Index::open(&g.manifest.to_json().unwrap(), g.docs, g.bm25, g.vectors).unwrap()
    }

    #[test]
    fn keyword_search_and_chunk() {
        let idx = index(false);
        assert_eq!(idx.length(), 2);
        assert!(!idx.has_vectors());
        let hits = idx.hits_bm25("winget", 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 0);
        let c = idx.record(0).unwrap();
        assert_eq!(c.path, "a.md");
        assert_eq!(c.heading_path, ["Install"]);
        assert!(idx.record(2).is_err());
    }

    #[test]
    fn hybrid_search_uses_the_vector() {
        let idx = index(true);
        let hits = idx.hits("winget", Some(&[0.0, 1.0, 0.0, 0.0]), 2).unwrap();
        let ids: Vec<u32> = hits.iter().map(|h| h.id).collect();
        assert_eq!(ids.len(), 2);
        let b = hits.iter().find(|h| h.id == 1).unwrap();
        assert_eq!(b.vector_rank, Some(1));
        assert_eq!(b.bm25_rank, None);
        assert!(idx.hits("winget", Some(&[1.0]), 2).is_err());
    }

    #[test]
    fn keyword_only_index_rejects_a_vector() {
        let idx = index(false);
        assert!(idx.hits("winget", None, 2).is_ok());
        assert!(idx.hits("winget", Some(&[1.0; 4]), 2).is_err());
    }

    #[test]
    fn results_serialize_with_camel_case_and_nulls() {
        let hit = Hit {
            id: 3,
            score: 0.5,
            bm25_rank: Some(1),
            vector_rank: None,
        };
        assert_eq!(
            serde_json::to_value(hit).unwrap(),
            serde_json::json!({"id": 3, "score": 0.5, "bm25Rank": 1, "vectorRank": null})
        );
        let chunk = index(false).record(1).unwrap();
        let json = serde_json::to_value(chunk).unwrap();
        assert_eq!(json["headingPath"], serde_json::json!(["Search"]));
        assert_eq!(json["lineStart"], 1);
    }
}
