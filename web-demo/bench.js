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

/**
 * Warm query latency with the browser embedder: embedding alone, and
 * embedding plus hybrid search, which is what one query costs the page.
 * Each query is embedded `reps` times after one untimed pass; milliseconds
 * per call are long enough to time singly. Returns rows in microseconds.
 */
export async function benchEmbedded(index, embedder, queries, reps = 5) {
  const embed = [];
  const total = [];
  for (const q of queries) await embedder.embedQuery(q.text);
  for (const q of queries) {
    let embedSum = 0;
    let totalSum = 0;
    for (let i = 0; i < reps; i++) {
      const start = performance.now();
      const vector = await embedder.embedQuery(q.text);
      const embedded = performance.now();
      index.search(q.text, vector, K);
      const done = performance.now();
      embedSum += embedded - start;
      totalSum += done - start;
    }
    embed.push((embedSum * 1000) / reps);
    total.push((totalSum * 1000) / reps);
  }
  const sorted = (xs) => xs.sort((a, b) => a - b);
  return [summary("embed query", sorted(embed)), summary("embed + hybrid search", sorted(total))];
}

/** The rows as text, aligned like the native example's output. */
export function formatBench(rows) {
  const us = (x) => x.toFixed(1).padStart(8);
  return rows
    .map((r) => `  ${r.label.padEnd(22)} median ${us(r.median)} us   p95 ${us(r.p95)} us`)
    .join("\n");
}
