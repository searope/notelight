//! The exported JavaScript API, run as real WebAssembly under Node by
//! `wasm-bindgen-test-runner`:
//!
//! ```powershell
//! cargo test --target wasm32-unknown-unknown -p notelight-wasm --test node
//! ```
//!
//! (`.cargo/config.toml` makes the runner the default for that target.) The
//! index is the committed fixture the browser demo loads, rebuilt by
//! `scripts/make_sample_index.py`. On native targets this file is empty.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;

use notelight_wasm::Index;

const MANIFEST: &[u8] = include_bytes!("../../../web-demo/sample-index/manifest.json");
const DOCS: &[u8] = include_bytes!("../../../web-demo/sample-index/docs.bin");
const BM25: &[u8] = include_bytes!("../../../web-demo/sample-index/bm25.bin");
const VECTORS: &[u8] = include_bytes!("../../../web-demo/sample-index/vectors.bin");

fn sample() -> Index {
    Index::new(
        MANIFEST,
        DOCS.to_vec(),
        BM25.to_vec(),
        Some(VECTORS.to_vec()),
    )
    .unwrap()
}

/// `obj[key]`, as JavaScript would read it.
fn get(obj: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(obj, &JsValue::from_str(key)).unwrap()
}

fn array(v: &JsValue) -> js_sys::Array {
    assert!(js_sys::Array::is_array(v), "expected an array, got {v:?}");
    js_sys::Array::from(v)
}

#[wasm_bindgen_test]
fn opens_the_sample_index() {
    let idx = sample();
    assert!(idx.length() > 0);
    assert_eq!(idx.dim(), 384);
    assert!(idx.has_vectors());
    // The browser embedder (web-demo/embed.js) must report this same id.
    assert_eq!(
        idx.embedder().as_deref(),
        Some("BAAI/bge-small-en-v1.5@5c38ec7c405e")
    );
}

#[wasm_bindgen_test]
fn bm25_results_are_plain_objects() {
    let idx = sample();
    let hits = array(&idx.search_bm25("generation", 5).unwrap());
    assert_eq!(hits.length(), 5);
    let first = hits.get(0);
    assert!(get(&first, "id").as_f64().is_some());
    let scores: Vec<f64> = hits
        .iter()
        .map(|h| get(&h, "score").as_f64().unwrap())
        .collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1]), "{scores:?}");
}

#[wasm_bindgen_test]
fn hybrid_search_without_a_vector_reports_null_ranks() {
    let idx = sample();
    let hits = array(&idx.search("memory mapped", None, 3).unwrap());
    assert_eq!(hits.length(), 3);
    let first = hits.get(0);
    assert_eq!(get(&first, "bm25Rank").as_f64(), Some(1.0));
    assert!(get(&first, "vectorRank").is_null());
}

#[wasm_bindgen_test]
fn hybrid_search_with_a_vector_reports_vector_ranks() {
    let idx = sample();
    let mut vector = vec![0.0_f32; 384];
    vector[0] = 1.0;
    let hits = array(&idx.search("memory mapped", Some(vector), 10).unwrap());
    assert_eq!(hits.length(), 10);
    assert!(
        hits.iter()
            .any(|h| get(&h, "vectorRank").as_f64().is_some())
    );
}

#[wasm_bindgen_test]
fn same_ranking_as_the_plain_rust_api() {
    // The JS objects carry exactly what the core returned.
    let idx = sample();
    let js = array(&idx.search_bm25("phrase ranking", 10).unwrap());
    let rust = idx.hits_bm25("phrase ranking", 10);
    assert_eq!(js.length() as usize, rust.len());
    for (h, r) in js.iter().zip(&rust) {
        assert_eq!(get(&h, "id").as_f64(), Some(f64::from(r.id)));
    }
}

#[wasm_bindgen_test]
fn chunk_returns_a_record_with_camel_case_fields() {
    let idx = sample();
    let id = idx.hits_bm25("wasm32", 1)[0].id;
    let chunk = idx.chunk(id).unwrap();
    let path = get(&chunk, "path").as_string().unwrap();
    assert_eq!(std::path::Path::new(&path).extension(), Some("md".as_ref()));
    assert!(js_sys::Array::is_array(&get(&chunk, "headingPath")));
    assert!(get(&chunk, "lineStart").as_f64().unwrap() >= 1.0);
    assert!(get(&chunk, "text").as_string().is_some());
}

#[wasm_bindgen_test]
fn errors_become_exceptions() {
    let idx = sample();
    assert!(idx.chunk(u32::MAX).is_err());
    assert!(idx.search("x", Some(vec![1.0; 383]), 3).is_err());
    let mut short = DOCS.to_vec();
    short.pop();
    assert!(Index::new(MANIFEST, short, BM25.to_vec(), Some(VECTORS.to_vec())).is_err());
    // The manifest declares vectors.bin, so it cannot be left out.
    assert!(Index::new(MANIFEST, DOCS.to_vec(), BM25.to_vec(), None).is_err());
}

#[wasm_bindgen_test]
fn chunk_text_returns_plain_objects() {
    let text = "# Title\n\nIntro.\n\n## Install\n\nRun it.\n";
    let chunks = array(&notelight_wasm::chunk_text(text, 1200, true).unwrap());
    assert!(chunks.length() > 0);
    let last = chunks.get(chunks.length() - 1);
    let headings = array(&get(&last, "headingPath"));
    assert_eq!(headings.get(1).as_string().as_deref(), Some("Install"));
    assert!(get(&last, "lineStart").as_f64().unwrap() >= 1.0);
    assert!(get(&last, "path").is_undefined());
    assert!(notelight_wasm::chunk_text(text, 0, true).is_err());
}

#[wasm_bindgen_test]
fn writer_rebuilds_the_sample_byte_for_byte() {
    // The sample's chunks, added in id order without vectors, give exactly
    // the docs.bin and bm25.bin the Python CLI wrote.
    let idx = sample();
    let mut w = notelight_wasm::Writer::new(0, None, Some(r#"{"source": "docs"}"#.into())).unwrap();
    for id in 0..u32::try_from(idx.length()).unwrap() {
        let c = idx.record(id).unwrap();
        let added = w
            .add(
                &c.path,
                c.heading_path,
                c.line_start,
                c.line_end,
                &c.text,
                None,
            )
            .unwrap();
        assert_eq!(added, id);
    }
    let g = w.build(3).unwrap();
    assert_eq!(g.docs(), DOCS);
    assert_eq!(g.bm25(), BM25);
    assert!(g.vectors().is_none());
    let rebuilt = Index::new(&g.manifest(), g.docs(), g.bm25(), None).unwrap();
    assert_eq!(rebuilt.generation(), 3);
    assert_eq!(
        rebuilt.hits_bm25("generation", 10),
        idx.hits_bm25("generation", 10)
    );
}

#[wasm_bindgen_test]
fn writer_errors_become_exceptions() {
    assert!(notelight_wasm::Writer::new(384, None, None).is_err());
    assert!(notelight_wasm::Writer::new(0, None, Some("not json".into())).is_err());
    let mut w = notelight_wasm::Writer::new(2, Some("m".into()), None).unwrap();
    assert!(w.add("a.md", vec![], 1, 1, "text", None).is_err());
    assert!(
        w.add("a.md", vec![], 1, 1, "text", Some(vec![1.0]))
            .is_err()
    );
    assert_eq!(
        w.add("a.md", vec![], 1, 1, "text", Some(vec![1.0, 0.0]))
            .unwrap(),
        0
    );
}
