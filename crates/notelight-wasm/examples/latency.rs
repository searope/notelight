//! Native search latency on one index generation, measured the same way as
//! the browser (`web-demo/bench.js`) so the numbers compare directly:
//!
//! ```powershell
//! cargo run --release -p notelight-wasm --example latency -- web-demo\sample-index web-demo\bench-queries.json
//! ```
//!
//! The queries file is `{"queries": [{"text": "...", "vector": [..] | null}]}`.
//! Each query runs `reps` times in a row and its time is the mean of those
//! runs (browsers coarsen `performance.now()`, so single calls cannot be
//! timed there); the report gives the median and 95th percentile over
//! queries, in microseconds. It goes through the same `Index` the
//! WebAssembly build exports, minus the conversion to JavaScript objects.

use std::path::Path;
use std::time::Instant;

use notelight_wasm::Index;

const K: usize = 10;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, queries] = &args[..2] else {
        eprintln!("usage: latency <generation-dir> <queries.json> [reps]");
        std::process::exit(2);
    };
    let reps: usize = args.get(2).map_or(20, |r| r.parse().expect("reps"));
    let dir = Path::new(dir);
    let read = |name: &str| std::fs::read(dir.join(name)).expect(name);

    let start = Instant::now();
    let manifest = read("manifest.json");
    let docs = read("docs.bin");
    let bm25 = read("bm25.bin");
    let vectors = dir
        .join("vectors.bin")
        .is_file()
        .then(|| read("vectors.bin"));
    let read_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let index = Index::open(&manifest, docs, bm25, vectors).expect("open index");
    let open_ms = start.elapsed().as_secs_f64() * 1000.0;

    let file: serde_json::Value =
        serde_json::from_slice(&std::fs::read(queries).expect("queries file")).expect("json");
    let queries: Vec<(String, Option<Vec<f32>>)> = file["queries"]
        .as_array()
        .expect("queries array")
        .iter()
        .map(|q| {
            let text = q["text"].as_str().expect("text").to_string();
            let vector = q["vector"].as_array().map(|v| {
                v.iter()
                    .map(|x| x.as_f64().expect("number") as f32)
                    .collect()
            });
            (text, vector)
        })
        .collect();

    println!(
        "native: {} chunks, {} queries, k={K}, {reps} reps; read {read_ms:.1} ms, open {open_ms:.2} ms",
        index.length(),
        queries.len()
    );
    report("bm25", &queries, reps, |(text, _)| {
        std::hint::black_box(index.hits_bm25(text, K));
    });
    report("hybrid, text only", &queries, reps, |(text, _)| {
        std::hint::black_box(index.hits(text, None, K).expect("search"));
    });
    if index.has_vectors() && queries.iter().all(|(_, v)| v.is_some()) {
        report("hybrid, text + vector", &queries, reps, |(text, v)| {
            std::hint::black_box(index.hits(text, v.as_deref(), K).expect("search"));
        });
    }
}

/// Time `run` on every query and print the median and p95 of the per-query means.
fn report<Q>(label: &str, queries: &[Q], reps: usize, mut run: impl FnMut(&Q)) {
    // One untimed pass warms caches, as the browser bench does.
    queries.iter().for_each(&mut run);
    let mut micros: Vec<f64> = queries
        .iter()
        .map(|q| {
            let start = Instant::now();
            for _ in 0..reps {
                run(q);
            }
            start.elapsed().as_secs_f64() * 1e6 / reps as f64
        })
        .collect();
    micros.sort_by(f64::total_cmp);
    let pick = |p: f64| micros[((micros.len() - 1) as f64 * p).round() as usize];
    println!(
        "  {label:<22} median {:>8.1} us   p95 {:>8.1} us",
        pick(0.5),
        pick(0.95)
    );
}
