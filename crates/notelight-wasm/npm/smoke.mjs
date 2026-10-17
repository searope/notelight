// Smoke test for the packed npm package, run by scripts/smoke_npm.py from a
// scratch project that installed the tarball (so "notelight-wasm" resolves to
// node_modules, not to this repository):
//
//   node smoke.mjs <sample-index dir> <queries.json fixture>
//
// It opens the committed sample index, checks BM25 search, checks hybrid
// search against the Python CLI's top 10 for the fixture's query vectors,
// and builds and searches an index in memory.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { argv } from "node:process";
import { fileURLToPath } from "node:url";

import init, { chunkText, Index, IndexWriter, initSync, version } from "notelight-wasm";

const [dir, queriesPath] = argv.slice(2);
assert.ok(dir && queriesPath, "usage: node smoke.mjs <sample-index dir> <queries.json>");

// The Node entry has already instantiated the module; init() is a no-op.
assert.equal(typeof initSync, "function");
await init();
assert.match(fileURLToPath(import.meta.resolve("notelight-wasm")), /[\\/]node\.js$/);
const pkg = JSON.parse(
  await readFile(fileURLToPath(import.meta.resolve("notelight-wasm/package.json")), "utf8"),
);
assert.equal(version(), pkg.version, "the core's version is the package version");
await readFile(fileURLToPath(import.meta.resolve("notelight-wasm/notelight_wasm_bg.wasm")));

const read = (name) => readFile(join(dir, name));
const index = new Index(
  await read("manifest.json"),
  await read("docs.bin"),
  await read("bm25.bin"),
  await read("vectors.bin"),
);
assert.ok(index.length > 0);
assert.equal(index.dim, 384);
assert.ok(index.hasVectors);

const bm25 = index.searchBm25("generation", 5);
assert.equal(bm25.length, 5);
for (const { id, score } of bm25) {
  assert.ok(score > 0);
  assert.match(index.chunk(id).text, /generation/i);
}
const top = index.chunk(bm25[0].id);
assert.equal(typeof top.path, "string");
assert.ok(Array.isArray(top.headingPath));
assert.ok(top.lineStart >= 1 && top.lineEnd >= top.lineStart);

// Hybrid search with the Python embedder's query vectors gives the CLI's top 10.
const fixture = JSON.parse(await readFile(queriesPath, "utf8"));
assert.equal(index.embedder, fixture.model_id);
for (const q of fixture.queries) {
  const bytes = Buffer.from(q.vector, "base64");
  const vector = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
  const ids = index.search(q.text, vector, fixture.k).map((h) => h.id);
  assert.deepEqual(ids, q.hits, `top ${fixture.k} for ${JSON.stringify(q.text)}`);
}
assert.throws(() => index.search("generation", new Float32Array(3), 5), /dimension|length/i);
const keywordOnly = index.search("generation", null, 3);
assert.ok(keywordOnly.every((h) => h.vectorRank === null && h.bm25Rank !== null));
index.free();

// Chunk and index text in memory, then reopen the generation and search it.
const text = "# Release\n\n## Hotfixes\n\nTag the commit, build the wheels, publish.\n";
const chunks = chunkText(text, 1200, true);
assert.deepEqual(chunks.at(-1).headingPath, ["Release", "Hotfixes"]);
const writer = new IndexWriter(0, null, JSON.stringify({ chunker: { max_chars: 1200 } }));
for (const c of chunks) {
  writer.add("runbooks/release.md", c.headingPath, c.lineStart, c.lineEnd, c.text, null);
}
assert.equal(writer.length, chunks.length);
const gen = writer.build(1);
assert.equal(gen.vectors, undefined);
const built = new Index(gen.manifest, gen.docs, gen.bm25, gen.vectors ?? null);
const [hit] = built.searchBm25("hotfixes", 1);
assert.equal(built.chunk(hit.id).path, "runbooks/release.md");
assert.throws(() => chunkText(text, 0, true), /maxChars/);

console.log(
  `notelight-wasm ${pkg.version}: sample index (${fixture.queries.length} hybrid queries ` +
    "match the CLI's top 10), BM25, chunkText and IndexWriter ok",
);
