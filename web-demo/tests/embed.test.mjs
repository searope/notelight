// The browser embedder matches the Python one, and its model files are
// verified before use.
//
//   python scripts/vendor_ort.py      # onnxruntime-web into web-demo/vendor/
//   python scripts/build_wasm.py      # notelight-wasm into web-demo/pkg/
//   node --test web-demo/tests/
//
// The parity tests need the bge-small files in the Python model cache (or
// NOTELIGHT_MODEL_DOWNLOAD=1 to fetch them); they are skipped without them,
// except in CI, where they must run. The fixture comes from
// scripts/make_sample_index.py, which embeds the same queries with
// `OnnxEmbedder` and records their hybrid top-10 on the sample index.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { test } from "node:test";

import {
  BGE_SMALL_EN_V15 as SPEC,
  checkIndexEmbedder,
  EmbedderMismatchError,
  fileUrl,
  l2Normalize,
  loadVerified,
  modelIdOf,
  QueryEmbedder,
  sha256Hex,
  VerificationError,
} from "../embed.js";
import {
  cosine,
  DEMO,
  FIXTURES,
  fileCache,
  float32FromBase64,
  modelFile,
  ORT,
  PKG,
  requireOrSkip,
} from "./helpers.mjs";

const fixture = JSON.parse(await readFile(join(FIXTURES, "queries.json"), "utf8"));

// --------------------------------------------------------------------------
// The model contract, without the model
// --------------------------------------------------------------------------

test("model id and URLs match embed.py", () => {
  assert.equal(modelIdOf(SPEC), "BAAI/bge-small-en-v1.5@5c38ec7c405e");
  assert.equal(modelIdOf(SPEC), fixture.model_id);
  assert.equal(
    fileUrl(SPEC, SPEC.onnx),
    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/onnx/model.onnx",
  );
  assert.equal(fileUrl(SPEC, SPEC.tokenizer, "http://127.0.0.1:8000/m/"), "http://127.0.0.1:8000/m/tokenizer.json");
});

test("a mismatched index is refused with the Python message", () => {
  checkIndexEmbedder(fixture.model_id, modelIdOf(SPEC), "docs");
  assert.throws(
    () => checkIndexEmbedder("hashing-256", modelIdOf(SPEC), "docs"),
    (e) =>
      e instanceof EmbedderMismatchError &&
      e.message ===
        "the index for docs was built with hashing-256, but the query embedder is " +
          "BAAI/bge-small-en-v1.5@5c38ec7c405e; re-index with this embedder or search with the original one",
  );
  assert.throws(() => checkIndexEmbedder(null, modelIdOf(SPEC), "docs"), /built with no embedder/);
});

test("sha256Hex is Web Crypto SHA-256", async () => {
  const abc = new TextEncoder().encode("abc");
  assert.equal(await sha256Hex(abc), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
});

test("l2Normalize gives a unit vector", () => {
  const v = l2Normalize(new Float32Array([3, 4]));
  assert.deepEqual([...v], [0.6000000238418579, 0.800000011920929]);
});

/** A pinned file for `content`, plus a fake fetch and an in-memory cache. */
async function fakes(content) {
  const bytes = new TextEncoder().encode(content);
  const file = { path: "f.bin", size: bytes.length, sha256: await sha256Hex(bytes) };
  const fetched = [];
  const served = { body: bytes };
  const fetchFn = async (url) => {
    fetched.push(url);
    return new Response(served.body);
  };
  const store = new Map();
  const cache = {
    get: async (url) => store.get(url) ?? null,
    put: async (url, b) => void store.set(url, b),
    delete: async (url) => void store.delete(url),
  };
  return { bytes, file, fetchFn, fetched, served, store, cache };
}

test("loadVerified downloads, verifies, caches, then reads the cache", async () => {
  const f = await fakes("model bytes");
  const progress = [];
  const first = await loadVerified("u", f.file, { ...f, onProgress: (p) => progress.push(p) });
  assert.deepEqual(first.bytes, f.bytes);
  assert.equal(first.cached, false);
  assert.equal(progress.at(-1).received, f.bytes.length);
  assert.ok(f.store.has("u"));
  const second = await loadVerified("u", f.file, f);
  assert.equal(second.cached, true);
  assert.equal(f.fetched.length, 1);
});

test("a download with the wrong hash is refused and not cached", async () => {
  const f = await fakes("model bytes");
  f.served.body = new TextEncoder().encode("MODEL BYTES"); // same size, other content
  await assert.rejects(loadVerified("u", f.file, f), (e) => {
    assert.ok(e instanceof VerificationError);
    assert.match(e.message, /f\.bin failed verification: expected SHA-256 [0-9a-f]{64}, got/);
    return true;
  });
  assert.equal(f.store.size, 0);
  f.served.body = new TextEncoder().encode("model bytes and more");
  await assert.rejects(loadVerified("u", f.file, f), /more than 11 bytes/);
});

test("a corrupt cached copy is replaced by a fresh download", async () => {
  const f = await fakes("model bytes");
  f.store.set("u", new TextEncoder().encode("tampered!!!"));
  const got = await loadVerified("u", f.file, f);
  assert.equal(got.cached, false);
  assert.deepEqual(f.store.get("u"), f.bytes);
});

test("HTTP errors name the file", async () => {
  const f = await fakes("x");
  const fetchFn = async () => new Response("nope", { status: 404 });
  await assert.rejects(loadVerified("u", f.file, { fetchFn }), /downloading f\.bin failed: HTTP 404/);
});

// --------------------------------------------------------------------------
// Parity with the Python embedder (needs the model, the runtime and the wasm)
// --------------------------------------------------------------------------

/** The embedder and the opened sample index, or a reason they are unavailable. */
async function setUp() {
  if (!existsSync(ORT)) return { reason: "onnxruntime-web is not vendored (python scripts/vendor_ort.py)" };
  if (!existsSync(PKG)) return { reason: "web-demo/pkg is not built (python scripts/build_wasm.py)" };
  const [onnx, tok] = [await modelFile(SPEC.onnx), await modelFile(SPEC.tokenizer)];
  if (!onnx || !tok) return { reason: "bge-small is not in the model cache (set NOTELIGHT_MODEL_DOWNLOAD=1)" };

  const ort = await import(pathToFileURL(ORT).href);
  ort.env.wasm.numThreads = 1;
  // The files are verified on the way in, from the model cache folder.
  const embedder = await QueryEmbedder.load(ort, { cache: fileCache });

  const wasm = await import(pathToFileURL(PKG).href);
  wasm.initSync({ module: await readFile(join(DEMO, "pkg", "notelight_wasm_bg.wasm")) });
  const read = (name) => readFile(join(DEMO, "sample-index", name));
  const index = new wasm.Index(
    await read("manifest.json"),
    await read("docs.bin"),
    await read("bm25.bin"),
    await read("vectors.bin"),
  );
  return { embedder, index };
}

const env = await setUp();

test("query vectors match OnnxEmbedder (cosine >= 0.999)", async (t) => {
  if (!requireOrSkip(t, env.embedder, env.reason)) return;
  let worst = 1;
  for (const q of fixture.queries) {
    const expected = float32FromBase64(q.vector);
    const got = await env.embedder.embedQuery(q.text);
    assert.equal(got.length, 384);
    const cos = cosine(got, expected);
    worst = Math.min(worst, cos);
    assert.ok(cos >= 0.999, `${q.text}: cosine ${cos}`);
  }
  t.diagnostic(`worst over ${fixture.queries.length} queries: 1 - cosine = ${(1 - worst).toExponential(1)}`);
});

test("hybrid top-10 on the sample index matches Python", async (t) => {
  if (!requireOrSkip(t, env.embedder, env.reason)) return;
  checkIndexEmbedder(env.index.embedder, env.embedder.modelId, "the sample index");
  for (const q of fixture.queries) {
    const vector = await env.embedder.embedQuery(q.text);
    const ids = env.index.search(q.text, vector, fixture.k).map((h) => h.id);
    assert.deepEqual(ids, q.hits, q.text);
  }
});
