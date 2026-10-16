// Search latency of the WebAssembly build, measured exactly like the native
// `latency` example (crates/notelight-wasm/examples/latency.rs) so the
// numbers compare directly. Used by the demo page and by bench-node.mjs.
//
// Each query runs `reps` times in a row and its time is the mean of those
// runs: browsers coarsen performance.now() (to 0.1 ms in Chrome without
// cross-origin isolation), so single microsecond-scale calls cannot be timed.
// The report gives the median and 95th percentile over queries.

const K = 10;

/** Time `run` on every query; returns per-query means in microseconds, sorted. */
function timeQueries(queries, reps, run) {
  queries.forEach(run); // one untimed pass, as natively
  const micros = queries.map((q) => {
    const start = performance.now();
    for (let i = 0; i < reps; i++) run(q);
    return ((performance.now() - start) * 1000) / reps;
  });
  return micros.sort((a, b) => a - b);
}

function summary(label, micros) {
  const pick = (p) => micros[Math.round((micros.length - 1) * p)];
  return { label, median: pick(0.5), p95: pick(0.95) };
}

/**
 * Benchmark `index` (an opened notelight Index) on `queries`
 * ([{text, vector | null}]). Returns [{label, median, p95}] in microseconds.
 */
export function bench(index, queries, reps = 20) {
  const rows = [
    summary("bm25", timeQueries(queries, reps, (q) => index.searchBm25(q.text, K))),
    summary("hybrid, text only", timeQueries(queries, reps, (q) => index.search(q.text, null, K))),
  ];
  if (index.hasVectors && queries.every((q) => q.vector)) {
    // Convert once, outside the timed loop, as the native bench parses once.
    const withVectors = queries.map((q) => ({ text: q.text, vector: Float32Array.from(q.vector) }));
    rows.push(
      summary(
        "hybrid, text + vector",
        timeQueries(withVectors, reps, (q) => index.search(q.text, q.vector, K)),
      ),
    );
  }
  return rows;
}

/** The rows as text, aligned like the native example's output. */
export function formatBench(rows) {
  const us = (x) => x.toFixed(1).padStart(8);
  return rows
    .map((r) => `  ${r.label.padEnd(22)} median ${us(r.median)} us   p95 ${us(r.p95)} us`)
    .join("\n");
}
