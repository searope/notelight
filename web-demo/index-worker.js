// The indexing worker: everything that touches stored indexes runs here, off
// the page's main thread, so reading, embedding and writing never freeze the
// page. The page talks to it with request/response messages (app.js `call`):
//
//   {op: "init", threads}           -> {store, threads} the store in use, embedder threads
//   {op: "list"}                    -> meta[]          stored indexes
//   {op: "load", id}                -> {meta, files}   a generation's bytes (transferred)
//   {op: "delete", id}              -> null
//   {op: "index", id, name, source, semantic, mirror}
//                                   -> {stats, meta, files}, with
//                                      {progress} messages on the way
//   {op: "cancel"}                  stops a running "index" between batches
//
// `source` is {handle} (a FileSystemDirectoryHandle) or {files: [{path, file}]}
// from <input webkitdirectory>. Files are read here and never leave the
// browser; the only network request is the model download (embed.js).
// pdf.js (documents.js) is loaded from vendor/pdfjs/ the first time a folder
// has a PDF to read, and runs in this worker.
import init, * as wasm from "./pkg/notelight_wasm.js";
import { BGE_SMALL_EN_V15, browserCache, modelIdOf, QueryEmbedder } from "./embed.js";
import { lazyPdfExtractor } from "./documents.js";
import { fromDirectoryHandle, fromFiles } from "./ingest.js";
import { CancelledError, deleteIndex, listIndexes, loadGeneration, newIndexId, readMeta, syncFolder } from "./indexer.js";
import { openStore } from "./store.js";

const ORT_URL = "./vendor/onnxruntime-web/ort.wasm.min.mjs";
const CACHE_NAME = "notelight-models";

let store = null;
let threads = 1;
let running = null; // the AbortController of the sync in progress

const transferables = (files) => Object.values(files ?? {}).filter(Boolean).map((b) => b.buffer);

async function loadEmbedder(mirror, report) {
  let ort;
  try {
    ort = await import(ORT_URL);
  } catch {
    throw new Error("onnxruntime-web is not in web-demo/vendor/; run: python scripts/vendor_ort.py");
  }
  ort.env.wasm.numThreads = threads;
  const cache = await browserCache(CACHE_NAME);
  report({ stage: "model", done: 0, total: 0 });
  const embedder = await QueryEmbedder.load(ort, {
    mirror,
    cache,
    onProgress: ({ received, total }) => report({ stage: "model", done: received, total }),
  });
  embedder.timings.threads = threads;
  return embedder;
}

/** Run `fn` holding the writer lock for index `id` (one writer per index, across tabs). */
function withLock(id, fn) {
  return navigator.locks ? navigator.locks.request(`notelight-index-${id}`, fn) : fn();
}

const ops = {
  async init({ threads: requested }) {
    await init();
    store = await openStore();
    // Threads need a cross-origin isolated page (see coi-sw.js).
    const cores = navigator.hardwareConcurrency || 1;
    threads = crossOriginIsolated ? Math.max(1, Math.min(requested ?? 4, cores)) : 1;
    return { store: store.kind, threads };
  },

  list: () => listIndexes(store),

  async load({ id }) {
    const { meta, files } = await loadGeneration(store, id);
    return { result: { meta, files }, transfer: transferables(files) };
  },

  async delete({ id }) {
    await withLock(id, () => deleteIndex(store, id));
    return null;
  },

  async index({ id, name, source, semantic, mirror }, report) {
    if (running) throw new Error("an index is already being built in this tab");
    id ??= newIndexId(name);
    const controller = new AbortController();
    running = controller;
    let model = null; // loaded on the first batch that needs it
    let modelMs = 0;
    // Like OnnxEmbedder, the model loads only when there is something to
    // embed: an update that finds every vector in the cache never loads it.
    const lazy = {
      modelId: modelIdOf(BGE_SMALL_EN_V15),
      dim: BGE_SMALL_EN_V15.dim,
      async embedDocuments(texts, options) {
        if (!model) {
          const start = performance.now();
          model = await loadEmbedder(mirror, report);
          modelMs = performance.now() - start;
          if (controller.signal.aborted) throw new CancelledError();
        }
        return model.embedDocuments(texts, options);
      },
    };
    const pdf = lazyPdfExtractor();
    try {
      return await withLock(id, async () => {
        const root = source.handle ? fromDirectoryHandle(source.handle) : fromFiles(source.files);
        const { stats, files } = await syncFolder({
          wasm,
          store,
          root,
          id,
          name,
          embedder: semantic ? lazy : null,
          pdf,
          onProgress: report,
          signal: controller.signal,
        });
        // Loading the model and pdf.js is reported on its own, not as indexing time.
        stats.ms.model = modelMs;
        if (stats.ms.embed) stats.ms.embed -= modelMs;
        stats.ms.pdfjs = pdf.loadMs;
        stats.ms.extract -= pdf.loadMs;
        stats.ms.total -= modelMs + pdf.loadMs;
        stats.threads = model ? threads : null;
        const meta = await readMeta(store, id);
        return { result: { stats, meta, files }, transfer: transferables(files) };
      });
    } finally {
      running = null;
      // Releasing a threaded session can take seconds; do not hold up the reply.
      model?.release().catch(() => {});
    }
  },
};

self.onmessage = async ({ data }) => {
  const { reqId, op } = data;
  if (op === "cancel") {
    running?.abort();
    return;
  }
  const report = (progress) => self.postMessage({ reqId, progress });
  try {
    if (!ops[op]) throw new Error(`unknown operation ${op}`);
    if (op !== "init" && !store) throw new Error("the worker is not initialized");
    const out = await ops[op](data, report);
    const { result, transfer } = out && "transfer" in out ? out : { result: out, transfer: [] };
    self.postMessage({ reqId, result }, transfer);
  } catch (e) {
    self.postMessage({ reqId, error: { name: e.name, message: e.message ?? String(e) } });
  }
};
