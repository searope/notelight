// Query embeddings in the browser: the same model contract as
// python/notelight/embed.py, so a query embedded here searches an index the
// Python CLI built with its default embedder.
//
// - bge-small-en-v1.5 pinned to one revision; onnx/model.onnx and
//   tokenizer.json are checked against their size and SHA-256 (Web Crypto)
//   before use, on every load, cached or not;
// - the bge query prefix, CLS pooling, L2 normalization;
// - `modelId` equals the Python one, and an index built with another
//   embedder is refused (`checkIndexEmbedder`).
//
// The ONNX runtime is passed in (onnxruntime-web, vendored by
// scripts/vendor_ort.py), and so are `fetch` and the cache, so this module
// runs unchanged in the browser and under Node (tests/embed.test.mjs).

import { WordPieceTokenizer } from "./tokenizer.js";

/** bge-small-en-v1.5 as `embed.py` pins it (`BGE_SMALL_EN_V15`). */
export const BGE_SMALL_EN_V15 = Object.freeze({
  repo: "BAAI/bge-small-en-v1.5",
  revision: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
  onnx: Object.freeze({
    path: "onnx/model.onnx",
    sha256: "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
    size: 133_093_490,
  }),
  tokenizer: Object.freeze({
    path: "tokenizer.json",
    sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    size: 711_396,
  }),
  dim: 384,
  maxTokens: 512,
  queryPrefix: "Represent this sentence for searching relevant passages: ",
  baseUrl: "https://huggingface.co",
});

/** `ModelSpec.model_id`: repository plus short revision. */
export const modelIdOf = (spec) => `${spec.repo}@${spec.revision.slice(0, 12)}`;

/**
 * Where to download `file` from: the Hub at the pinned revision, or, with
 * `mirror`, a folder laid out like the Python model cache's revision folder
 * (`<mirror>/onnx/model.onnx`, `<mirror>/tokenizer.json`).
 */
export function fileUrl(spec, file, mirror = null) {
  if (mirror) return `${mirror.replace(/\/+$/, "")}/${file.path}`;
  return `${spec.baseUrl}/${spec.repo}/resolve/${spec.revision}/${file.path}`;
}

/** A model file failed its size or SHA-256 check. */
export class VerificationError extends Error {
  name = "VerificationError";
}

/** The index was built with a different embedder (as in `search.py`). */
export class EmbedderMismatchError extends Error {
  name = "EmbedderMismatchError";
}

/**
 * Refuse an index whose manifest names another embedder, with the message
 * `Searcher` gives in Python.
 * @param {string | null} indexedWith  the index's `embedder` (manifest)
 * @param {string} modelId  the query embedder's id
 * @param {string} label  how to name the index in the message
 */
export function checkIndexEmbedder(indexedWith, modelId, label) {
  if (indexedWith !== modelId) {
    throw new EmbedderMismatchError(
      `the index for ${label} was built with ${indexedWith ?? "no embedder"}, ` +
        `but the query embedder is ${modelId}; ` +
        "re-index with this embedder or search with the original one",
    );
  }
}

/** Lowercase hex SHA-256 of `bytes`, by Web Crypto. */
export async function sha256Hex(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Throw unless `bytes` has `file`'s pinned size and SHA-256. */
export async function verify(file, bytes) {
  if (bytes.length !== file.size) {
    throw new VerificationError(
      `${file.path} failed verification: expected ${file.size} bytes, got ${bytes.length}`,
    );
  }
  const actual = await sha256Hex(bytes);
  if (actual !== file.sha256) {
    throw new VerificationError(
      `${file.path} failed verification: expected SHA-256 ${file.sha256}, got ${actual}`,
    );
  }
}

/**
 * A model cache over the browser's Cache API, or null where there is none
 * (Node, or a page that is not a secure context). Only verified bytes are
 * stored, and they are verified again when read.
 */
export async function browserCache(name = "notelight-models") {
  if (!globalThis.caches) return null;
  const cache = await caches.open(name);
  return {
    async get(url) {
      const hit = await cache.match(url);
      return hit ? new Uint8Array(await hit.arrayBuffer()) : null;
    },
    put: (url, bytes) => cache.put(url, new Response(bytes)),
    delete: (url) => cache.delete(url),
  };
}

/** Read a response body into one buffer of `size` bytes, reporting progress. */
async function readBody(response, file, onProgress) {
  const out = new Uint8Array(file.size);
  let received = 0;
  const reader = response.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (received + value.length > file.size) {
      await reader.cancel();
      throw new VerificationError(`${file.path} failed verification: more than ${file.size} bytes`);
    }
    out.set(value, received);
    received += value.length;
    onProgress?.({ file: file.path, received, total: file.size });
  }
  return out.subarray(0, received);
}

/**
 * The bytes of one pinned file, verified: from `cache` if it holds a copy
 * that verifies, else downloaded with `fetchFn` (and then cached).
 * @returns {Promise<{bytes: Uint8Array, cached: boolean, fetchMs: number, verifyMs: number}>}
 */
export async function loadVerified(url, file, { fetchFn = fetch, cache = null, onProgress } = {}) {
  let start = performance.now();
  const cachedBytes = cache ? await cache.get(url) : null;
  if (cachedBytes) {
    const fetchMs = performance.now() - start;
    start = performance.now();
    try {
      await verify(file, cachedBytes);
      return { bytes: cachedBytes, cached: true, fetchMs, verifyMs: performance.now() - start };
    } catch (e) {
      if (!(e instanceof VerificationError)) throw e;
      await cache.delete(url); // a corrupt or stale copy: download it again
    }
  }
  start = performance.now();
  let response;
  try {
    response = await fetchFn(url);
  } catch (e) {
    throw new Error(`downloading ${file.path} failed (${e.message}); check the network`);
  }
  if (!response.ok) throw new Error(`downloading ${file.path} failed: HTTP ${response.status}`);
  const bytes = await readBody(response, file, onProgress);
  const fetchMs = performance.now() - start;
  start = performance.now();
  await verify(file, bytes);
  const verifyMs = performance.now() - start;
  await cache?.put(url, bytes);
  return { bytes, cached: false, fetchMs, verifyMs };
}

/** Scale `v` to unit length (float64 accumulation), as `_l2_normalize` does. */
export function l2Normalize(v) {
  let sum = 0;
  for (const x of v) sum += x * x;
  const norm = Math.max(Math.sqrt(sum), 1e-12);
  return Float32Array.from(v, (x) => x / norm);
}

/** bge-small query embeddings with onnxruntime-web. */
export class QueryEmbedder {
  #session;
  #tokenizer;
  #ort;

  constructor(ort, session, tokenizer, spec, timings) {
    this.#ort = ort;
    this.#session = session;
    this.#tokenizer = tokenizer;
    this.spec = spec;
    /** How long each load step took, in ms, and whether files came from the cache. */
    this.timings = timings;
  }

  /**
   * Fetch (or read from the cache) and verify the model files, then start an
   * ONNX Runtime session.
   * @param {object} ort  the onnxruntime-web module
   * @param {object} [options]
   * @param {object} [options.spec]  model spec (default bge-small)
   * @param {string} [options.mirror]  folder URL to fetch the files from instead of the Hub
   * @param {object} [options.cache]  {get, put, delete} over verified bytes, e.g. `browserCache()`
   * @param {Function} [options.fetchFn]  `fetch` replacement
   * @param {Function} [options.onProgress]  called with {file, received, total} while downloading
   */
  static async load(ort, { spec = BGE_SMALL_EN_V15, mirror = null, cache = null, fetchFn, onProgress } = {}) {
    const opts = { cache, fetchFn, onProgress };
    const start = performance.now();
    const [tok, onnx] = await Promise.all([
      loadVerified(fileUrl(spec, spec.tokenizer, mirror), spec.tokenizer, opts),
      loadVerified(fileUrl(spec, spec.onnx, mirror), spec.onnx, opts),
    ]);
    const filesMs = performance.now() - start;
    const tokenizer = new WordPieceTokenizer(
      JSON.parse(new TextDecoder().decode(tok.bytes)),
      spec.maxTokens,
    );
    const sessionStart = performance.now();
    const session = await ort.InferenceSession.create(onnx.bytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    const timings = {
      cached: tok.cached && onnx.cached,
      filesMs, // fetch or cache read plus verification, both files in parallel
      fetchMs: onnx.fetchMs,
      verifyMs: onnx.verifyMs,
      sessionMs: performance.now() - sessionStart,
      totalMs: performance.now() - start,
    };
    return new QueryEmbedder(ort, session, tokenizer, spec, timings);
  }

  /** The id an index's manifest must name for its vectors to match ours. */
  get modelId() {
    return modelIdOf(this.spec);
  }

  /** Vector dimension. */
  get dim() {
    return this.spec.dim;
  }

  /** Token ids for `text`, without the query prefix. */
  tokenize(text) {
    return this.#tokenizer.encode(text);
  }

  /** Embed a search query: prefix, tokenize, run, CLS-pool, normalize. */
  async embedQuery(text) {
    const enc = this.#tokenizer.encode(this.spec.queryPrefix + text);
    const n = enc.ids.length;
    const tensor = (values) =>
      new this.#ort.Tensor("int64", BigInt64Array.from(values, BigInt), [1, n]);
    const all = {
      input_ids: tensor(enc.ids),
      attention_mask: tensor(enc.attentionMask),
      token_type_ids: tensor(enc.typeIds),
    };
    const feeds = Object.fromEntries(this.#session.inputNames.map((name) => [name, all[name]]));
    const outputs = await this.#session.run(feeds);
    const hidden = outputs[this.#session.outputNames[0]]; // (1, n, dim)
    const cls = l2Normalize(hidden.data.subarray(0, this.dim));
    for (const t of Object.values(outputs)) t.dispose?.();
    return cls;
  }

  /** Release the ONNX Runtime session. */
  async release() {
    await this.#session.release();
  }
}
