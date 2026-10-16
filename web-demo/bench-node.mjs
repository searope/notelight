// The browser benchmark, run under Node on the same .wasm:
//
//   python scripts/build_wasm.py
//   node web-demo/bench-node.mjs web-demo/sample-index web-demo/bench-queries.json [reps]
//
// Prints the same table as the native `latency` example.
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { argv } from "node:process";

import { initSync, Index } from "./pkg/notelight_wasm.js";
import { bench, formatBench } from "./bench.js";

const [dir, queriesPath, reps = "20"] = argv.slice(2);
if (!dir || !queriesPath) {
  console.error("usage: node bench-node.mjs <generation-dir> <queries.json> [reps]");
  process.exit(2);
}

const wasmUrl = new URL("./pkg/notelight_wasm_bg.wasm", import.meta.url);
initSync({ module: await readFile(wasmUrl) });

let start = performance.now();
const read = (name) => readFile(join(dir, name));
const vectorsPath = join(dir, "vectors.bin");
const [manifest, docs, bm25, vectors] = await Promise.all([
  read("manifest.json"),
  read("docs.bin"),
  read("bm25.bin"),
  existsSync(vectorsPath) ? readFile(vectorsPath) : null,
]);
const readMs = performance.now() - start;
start = performance.now();
const index = new Index(manifest, docs, bm25, vectors);
const openMs = performance.now() - start;

const { queries } = JSON.parse(await readFile(queriesPath, "utf8"));
console.log(
  `node wasm: ${index.length} chunks, ${queries.length} queries, k=10, ${reps} reps; ` +
    `read ${readMs.toFixed(1)} ms, open ${openMs.toFixed(2)} ms`,
);
console.log(formatBench(bench(index, queries, Number(reps))));
