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

fn sample() -> Index {
    Index::new(MANIFEST, DOCS.to_vec(), BM25.to_vec(), None).unwrap()
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
    assert_eq!(idx.dim(), 0);
    assert!(!idx.has_vectors());
    assert_eq!(idx.embedder(), None);
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
    assert!(get(&chunk, "path").as_string().unwrap().ends_with(".md"));
    assert!(js_sys::Array::is_array(&get(&chunk, "headingPath")));
    assert!(get(&chunk, "lineStart").as_f64().unwrap() >= 1.0);
    assert!(get(&chunk, "text").as_string().is_some());
}

#[wasm_bindgen_test]
fn errors_become_exceptions() {
    let idx = sample();
    assert!(idx.chunk(u32::MAX).is_err());
    assert!(idx.search("x", Some(vec![1.0; 384]), 3).is_err());
    let mut short = DOCS.to_vec();
    short.pop();
    assert!(Index::new(MANIFEST, short, BM25.to_vec(), None).is_err());
}
