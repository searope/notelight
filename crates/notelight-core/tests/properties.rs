//! Property-based tests: invariants checked against simple reference
//! implementations on thousands of generated inputs.

use std::collections::HashMap;

use notelight_core::bm25::{Bm25, Bm25Builder, Bm25Params};
use notelight_core::chunk::{ChunkOpts, chunk_text};
use notelight_core::quant::{
    TERNARY_THRESHOLD, hamming, l2_normalized, quantize_binary, quantize_int8, ternary_dot,
    ternary_mask,
};
use notelight_core::tokenize::tokenize;
use notelight_core::topk::{smallest_k_by_histogram, top_k};
use notelight_core::vectors::{VectorBuilder, VectorMode, Vectors};
use proptest::prelude::*;

/// Markdown-ish documents: headings, fences, paragraphs, blank lines, long
/// words, CRLF and non-ASCII text.
fn markdown_doc() -> impl Strategy<Value = String> {
    let line = prop_oneof![
        3 => "[a-zA-Zé ]{0,40}",
        1 => "#{1,7} [a-z ]{0,20}",
        1 => Just("```".to_string()),
        1 => Just(String::new()),
        1 => "[a-z]{30,90}",
        1 => "  [а-я ]{0,30}",
    ];
    (prop::collection::vec(line, 0..60), any::<bool>())
        .prop_map(|(lines, crlf)| lines.join(if crlf { "\r\n" } else { "\n" }))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn chunks_are_exact_ordered_bounded_and_complete(
        doc in markdown_doc(),
        max_chars in 5usize..200,
        markdown in any::<bool>(),
    ) {
        let chunks = chunk_text(&doc, &ChunkOpts { max_chars, markdown });
        let mut covered = vec![false; doc.len()];
        let mut prev_end = 0;
        for c in &chunks {
            // Exact slice of the source, within bounds, in order, non-overlapping.
            prop_assert!(c.byte_start >= prev_end);
            prop_assert_eq!(&doc[c.byte_start..c.byte_end], c.text.as_str());
            prop_assert!(c.text.chars().count() <= max_chars);
            prop_assert!(!c.text.trim().is_empty());
            prev_end = c.byte_end;
            // Line numbers agree with byte offsets.
            let line_of = |pos: usize| doc.as_bytes()[..pos].split(|&b| b == b'\n').count() as u32;
            prop_assert_eq!(c.line_start, line_of(c.byte_start));
            prop_assert_eq!(c.line_end, line_of(c.byte_end.max(c.byte_start + 1) - 1));
            covered[c.byte_start..c.byte_end].iter_mut().for_each(|b| *b = true);
        }
        // Every non-whitespace character lands in some chunk.
        for (i, ch) in doc.char_indices() {
            if !ch.is_whitespace() {
                prop_assert!(covered[i], "char {:?} at byte {} not covered", ch, i);
            }
        }
    }

    #[test]
    fn tokens_are_lowercase_nonempty_terms(text in "\\PC{0,200}") {
        for t in tokenize(&text) {
            prop_assert!(!t.is_empty());
            prop_assert_eq!(t.to_lowercase(), t.clone());
            prop_assert!(t.chars().all(char::is_alphanumeric));
        }
    }

    #[test]
    fn binary_codes_match_reference_and_hamming_is_a_metric(
        a in prop::collection::vec(-1.0f32..1.0, 1..200),
        seed in any::<u64>(),
    ) {
        let dim = a.len();
        let b: Vec<f32> = (0..dim).map(|i| ((seed.rotate_left(i as u32) & 0xff) as f32) - 127.5).collect();
        let c: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
        let (ca, cb, cc) = (quantize_binary(&a), quantize_binary(&b), quantize_binary(&c));

        let reference = a.iter().zip(&b).filter(|(x, y)| (**x > 0.0) != (**y > 0.0)).count() as u32;
        prop_assert_eq!(hamming(&ca, &cb), reference);
        prop_assert_eq!(hamming(&ca, &ca), 0);
        prop_assert_eq!(hamming(&ca, &cb), hamming(&cb, &ca));
        prop_assert!(hamming(&ca, &cc) <= hamming(&ca, &cb) + hamming(&cb, &cc));
    }

    #[test]
    fn int8_error_is_at_most_half_a_step(v in prop::collection::vec(-10.0f32..10.0, 1..300)) {
        let (codes, scale) = quantize_int8(&v);
        for (x, c) in v.iter().zip(&codes) {
            prop_assert!((x - f32::from(*c) * scale).abs() <= scale * 0.5 + 1e-6);
        }
    }

    #[test]
    fn top_k_matches_full_sort(
        scores in prop::collection::vec(-100i32..100, 0..300),
        k in 0usize..50,
    ) {
        let items: Vec<(u32, f32)> = scores.iter().enumerate().map(|(i, &s)| (i as u32, s as f32)).collect();
        let mut reference = items.clone();
        reference.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        reference.truncate(k);
        prop_assert_eq!(top_k(items, k), reference);
    }

    #[test]
    fn histogram_select_matches_full_sort(
        dists in prop::collection::vec(0u16..400, 0..500),
        k in 0usize..80,
    ) {
        let mut reference: Vec<u32> = (0..dists.len() as u32).collect();
        reference.sort_by_key(|&i| (dists[i as usize], i));
        reference.truncate(k);
        prop_assert_eq!(smallest_k_by_histogram(&dists, k), reference);
    }

    #[test]
    fn bm25_matches_a_naive_reference(
        docs in prop::collection::vec(prop::collection::vec("[a-e]{2}", 0..12), 1..30),
        query in prop::collection::vec("[a-f]{2}", 1..4),
    ) {
        let mut builder = Bm25Builder::new();
        for d in &docs {
            builder.add_tokens(d);
        }
        let params = Bm25Params::default();
        let index = Bm25::from_bytes(builder.finish(params).unwrap()).unwrap();
        let got = index.search_tokens(&query, docs.len());

        // Reference: textbook BM25 over the same token lists.
        let n = docs.len() as f32;
        let avgdl = docs.iter().map(Vec::len).sum::<usize>() as f32 / n;
        let mut unique = query.clone();
        unique.sort();
        unique.dedup();
        let mut expected: Vec<(u32, f32)> = Vec::new();
        for (i, d) in docs.iter().enumerate() {
            let mut tf: HashMap<&str, f32> = HashMap::new();
            for t in d {
                *tf.entry(t.as_str()).or_default() += 1.0;
            }
            let mut score = 0.0;
            let mut matched = false;
            for q in &unique {
                let Some(&f) = tf.get(q.as_str()) else { continue };
                matched = true;
                let df = docs.iter().filter(|d| d.contains(q)).count() as f32;
                let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                let norm = params.k1 * (1.0 - params.b + params.b * d.len() as f32 / avgdl.max(f32::MIN_POSITIVE));
                score += idf * f * (params.k1 + 1.0) / (f + norm);
            }
            if matched {
                expected.push((i as u32, score));
            }
        }
        expected.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        prop_assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(&expected) {
            prop_assert!((g.1 - e.1).abs() < 1e-4, "score {} vs {}", g.1, e.1);
        }
        // Same documents (order may differ only between near-equal scores).
        let mut gi: Vec<u32> = got.iter().map(|g| g.0).collect();
        let mut ei: Vec<u32> = expected.iter().map(|e| e.0).collect();
        gi.sort_unstable();
        ei.sort_unstable();
        prop_assert_eq!(gi, ei);
    }

    #[test]
    fn two_stage_search_with_all_candidates_equals_exhaustive(
        vs in prop::collection::vec(prop::collection::vec(-1.0f32..1.0, 24), 1..60),
        q in prop::collection::vec(-1.0f32..1.0, 24),
        k in 1usize..10,
    ) {
        prop_assume!(q.iter().any(|x| *x != 0.0));
        let mut b = VectorBuilder::new(24);
        for v in &vs {
            b.add(v).unwrap();
        }
        let idx = Vectors::from_bytes(b.finish().unwrap()).unwrap();
        prop_assert_eq!(
            idx.search(&q, k, vs.len()).unwrap(),
            idx.search_int8_exhaustive(&q, k).unwrap()
        );
    }

    #[test]
    fn ternary_dot_matches_reference(
        a in prop::collection::vec(-1.0f32..1.0, 1..200),
        seed in any::<u64>(),
    ) {
        let b: Vec<f32> = (0..a.len()).map(|i| ((seed.rotate_left(i as u32) & 0xff) as f32) - 127.5).collect();
        // Reference: each dimension becomes -1, 0 or +1, then a plain dot product.
        let ternary = |v: &[f32]| -> Vec<i64> {
            let threshold = TERNARY_THRESHOLD * (v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32);
            v.iter().map(|&x| if x.abs() > threshold { if x > 0.0 { 1 } else { -1 } } else { 0 }).collect()
        };
        let reference: i64 = ternary(&a).iter().zip(ternary(&b)).map(|(x, y)| x * y).sum();
        let (sa, ma) = (quantize_binary(&a), ternary_mask(&a));
        let (sb, mb) = (quantize_binary(&b), ternary_mask(&b));
        prop_assert_eq!(ternary_dot(&sa, &ma, &sb, &mb), reference);
        prop_assert_eq!(ternary_dot(&sa, &ma, &sb, &mb), ternary_dot(&sb, &mb, &sa, &ma));
        // Padding bits past `dim` are never set.
        if a.len() % 8 != 0 {
            prop_assert_eq!(ma.last().unwrap() >> (a.len() % 8), 0);
        }
    }

    #[test]
    fn binary_mode_matches_a_hamming_sort(
        vs in prop::collection::vec(prop::collection::vec(-1.0f32..1.0, 24), 1..60),
        q in prop::collection::vec(-1.0f32..1.0, 24),
        k in 1usize..10,
    ) {
        prop_assume!(q.iter().any(|x| *x != 0.0));
        let idx = build(&vs);
        let qc = quantize_binary(&q);
        let mut reference: Vec<u32> = (0..vs.len() as u32).collect();
        reference.sort_by_key(|&i| (hamming(&qc, &quantize_binary(&vs[i as usize])), i));
        reference.truncate(k);
        let got: Vec<u32> = idx.search_with(&q, k, 0, VectorMode::Binary).unwrap().iter().map(|h| h.0).collect();
        prop_assert_eq!(got, reference);
    }

    #[test]
    fn ternary_search_rescores_the_best_ternary_candidates(
        vs in prop::collection::vec(prop::collection::vec(-1.0f32..1.0, 24), 1..60),
        q in prop::collection::vec(-1.0f32..1.0, 24),
        k in 1usize..10,
        candidates in 1usize..40,
    ) {
        prop_assume!(q.iter().any(|x| *x != 0.0));
        let idx = build(&vs);
        let n = vs.len();
        // Stored ternary codes come from the int8 codes of the normalized vector.
        let stored: Vec<(Vec<u8>, Vec<u8>)> = vs.iter().map(|v| {
            let v = l2_normalized(v);
            let codes: Vec<f32> = quantize_int8(&v).0.iter().map(|&c| f32::from(c)).collect();
            (quantize_binary(&v), ternary_mask(&codes))
        }).collect();
        let qn = l2_normalized(&q);
        let (qs, qm) = (quantize_binary(&qn), ternary_mask(&qn));
        let mut cands: Vec<u32> = (0..n as u32).collect();
        cands.sort_by_key(|&i| {
            let (s, m) = &stored[i as usize];
            (-ternary_dot(&qs, &qm, s, m), i)
        });
        cands.truncate(candidates.max(k));
        // Rescore with the index's own int8 scores, best first, ties by id.
        let int8: HashMap<u32, f32> = idx.search_with(&q, n, 0, VectorMode::Int8).unwrap().into_iter().collect();
        let mut reference: Vec<(u32, f32)> = cands.iter().map(|&i| (i, int8[&i])).collect();
        reference.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        reference.truncate(k);
        prop_assert_eq!(idx.search_with(&q, k, candidates, VectorMode::TernaryInt8).unwrap(), reference);
        // With every vector as a candidate it is exactly the int8 scan.
        prop_assert_eq!(
            idx.search_with(&q, k, n, VectorMode::TernaryInt8).unwrap(),
            idx.search_with(&q, k, 0, VectorMode::Int8).unwrap()
        );
    }
}

fn build(vs: &[Vec<f32>]) -> Vectors<Vec<u8>> {
    let mut b = VectorBuilder::new(vs[0].len());
    for v in vs {
        b.add(v).unwrap();
    }
    Vectors::from_bytes(b.finish().unwrap()).unwrap()
}
