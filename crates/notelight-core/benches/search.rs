//! Benchmarks for the hot paths: vector search, BM25 and chunking.
//!
//! Run with `cargo bench -p notelight-core`. Data is synthetic and generated
//! deterministically, so numbers are comparable across runs and machines.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use notelight_core::bm25::{Bm25, Bm25Builder, Bm25Params};
use notelight_core::chunk::{ChunkOpts, chunk_text};
use notelight_core::vectors::{VectorBuilder, Vectors};

const DIM: usize = 384;

/// Small deterministic PRNG (`SplitMix64`) so benches need no extra dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn vectors(n: usize) -> (Vectors<Vec<u8>>, Vec<f32>) {
    let mut rng = Rng(42);
    let mut b = VectorBuilder::new(DIM);
    for _ in 0..n {
        let v: Vec<f32> = (0..DIM).map(|_| rng.unit()).collect();
        b.add(&v).unwrap();
    }
    let q: Vec<f32> = (0..DIM).map(|_| rng.unit()).collect();
    (Vectors::from_bytes(b.finish().unwrap()).unwrap(), q)
}

fn bench_vectors(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    group.sample_size(20);
    for n in [10_000, 100_000] {
        let (idx, q) = vectors(n);
        group.bench_with_input(BenchmarkId::new("binary+int8_rescore", n), &n, |b, _| {
            b.iter(|| idx.search(black_box(&q), 10, 200).unwrap());
        });
        group.bench_with_input(BenchmarkId::new("int8_exhaustive", n), &n, |b, _| {
            b.iter(|| idx.search_int8_exhaustive(black_box(&q), 10).unwrap());
        });
    }
    group.finish();
}

fn bench_bm25(c: &mut Criterion) {
    // Zipf-like vocabulary: low term ids are common, high ones rare.
    let mut rng = Rng(7);
    let vocab: Vec<String> = (0..20_000).map(|i| format!("w{i}")).collect();
    let mut builder = Bm25Builder::new();
    for _ in 0..100_000 {
        let doc: Vec<String> = (0..60)
            .map(|_| {
                let r = (rng.next() % 1_000_000) as f64 / 1_000_000.0;
                vocab[((r * r * r) * vocab.len() as f64) as usize].clone()
            })
            .collect();
        builder.add_tokens(&doc);
    }
    let idx = Bm25::from_bytes(builder.finish(Bm25Params::default()).unwrap()).unwrap();
    let query: Vec<String> = ["w5", "w300", "w4000"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    c.bench_function("bm25_search/100000_docs_3_terms", |b| {
        b.iter(|| idx.search_tokens(black_box(&query), 10));
    });
}

fn bench_chunking(c: &mut Criterion) {
    let section = "## Section title\n\nSome paragraph text with several words in it. ".repeat(4)
        + "\n\n```rust\nfn main() { println!(\"hi\"); }\n```\n\n";
    let doc = format!("# Doc\n\n{}", section.repeat(3_400)); // ~1 MB
    let mut group = c.benchmark_group("chunking");
    group.throughput(criterion::Throughput::Bytes(doc.len() as u64));
    group.bench_function("markdown_1MB", |b| {
        b.iter(|| chunk_text(black_box(&doc), &ChunkOpts::default()));
    });
    group.finish();
}

criterion_group!(benches, bench_vectors, bench_bm25, bench_chunking);
criterion_main!(benches);
