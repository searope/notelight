// The demo page: load an index generation into WebAssembly memory, then
// search it with the Rust core. When the index has bge-small embeddings, the
// query is embedded here too (embed.js, onnxruntime-web) and the search is
// hybrid. The page, the .wasm, the runtime and the sample index are served
// from this folder; the only other request is the one-time model download,
// which is verified by SHA-256 and kept in the browser's Cache API.
//
// URL parameters, for measuring other indexes without the folder picker:
//   ?index=<url of a generation folder>  load it on start
//   ?bench=<url of a queries file>       queries for the Benchmark button
//   ?model=<url of a folder>             fetch the model files from there instead
//                                        of the Hub (laid out like the Python
//                                        model cache's revision folder)
//   ?threads=1                           embed on one thread (no service worker)
import init, { Index, version } from "./pkg/notelight_wasm.js";
import { bench, benchEmbedded, formatBench } from "./bench.js";
import {
  BGE_SMALL_EN_V15,
  browserCache,
  checkIndexEmbedder,
  modelIdOf,
  QueryEmbedder,
} from "./embed.js";

const ORT_URL = "./vendor/onnxruntime-web/ort.wasm.min.mjs";
const CACHE_NAME = "notelight-models";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
let index = null;
let embedder = null; // a QueryEmbedder once loaded, reused across indexes
let embedderLoading = null; // the pending load, so two indexes do not load it twice
let useVectors = false; // whether the open index takes our query vectors

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

function setModelStatus(text, isError = false) {
  $("model-status").hidden = !text;
  $("model-status").textContent = text;
  $("model-status").classList.toggle("error", isError);
}

const kib = (n) => (n / 1024 >= 1024 ? `${(n / 1048576).toFixed(1)} MiB` : `${Math.round(n / 1024)} KiB`);
const ms = (x) => (x >= 1000 ? `${(x / 1000).toFixed(1)} s` : `${x.toFixed(0)} ms`);

/** Load bge-small once: the runtime, the verified files (cache or download), a session. */
function loadEmbedder() {
  embedderLoading ??= (async () => {
    let ort;
    try {
      ort = await import(ORT_URL);
    } catch {
      throw new Error("onnxruntime-web is not in web-demo/vendor/; run: python scripts/vendor_ort.py");
    }
    // Threads need a cross-origin isolated page: COOP and COEP headers from
    // the server (`serve_demo.py --isolated`) or from coi-sw.js (see isolate()).
    const threaded = crossOriginIsolated && params.get("threads") !== "1";
    ort.env.wasm.numThreads = threaded ? Math.min(4, navigator.hardwareConcurrency) : 1;
    const cache = await browserCache(CACHE_NAME);
    const onProgress = ({ file, received, total }) => {
      if (file === BGE_SMALL_EN_V15.onnx.path) {
        setModelStatus(`Downloading bge-small (one time): ${kib(received)} of ${kib(total)}…`);
      }
    };
    setModelStatus("Loading the bge-small query embedder…");
    const loaded = await QueryEmbedder.load(ort, { mirror: params.get("model"), cache, onProgress });
    loaded.timings.threads = ort.env.wasm.numThreads;
    window.embedderTimings = loaded.timings; // for measuring from DevTools
    return loaded;
  })();
  return embedderLoading;
}

function describeLoad(t) {
  const where = t.cached ? `read from the browser cache in ${ms(t.fetchMs)}` : `downloaded in ${ms(t.fetchMs)}`;
  return (
    `Query embedder ready: bge-small ${where}, SHA-256 verified in ${ms(t.verifyMs)}, ` +
    `ONNX session started in ${ms(t.sessionMs)} (${ms(t.totalMs)} in all), ` +
    `${t.threads} thread${t.threads === 1 ? "" : "s"}. Search is hybrid.`
  );
}

/** Decide how the open index is searched, loading the embedder if the index takes it. */
async function prepareVectors(source) {
  useVectors = false;
  if (!index.hasVectors) {
    setModelStatus("This index has no embeddings: keyword search (BM25 and phrases).");
    return;
  }
  try {
    checkIndexEmbedder(index.embedder, modelIdOf(BGE_SMALL_EN_V15), source);
  } catch (e) {
    setModelStatus(`${e.message}. Searching by keywords only.`, true);
    return;
  }
  const opened = index;
  try {
    embedder = await loadEmbedder();
  } catch (e) {
    embedderLoading = null; // let the next index retry
    setModelStatus(`${e.message}. Searching by keywords only.`, true);
    return;
  }
  if (index !== opened) return; // another index was opened meanwhile
  useVectors = true;
  setModelStatus(describeLoad(embedder.timings));
  if ($("q").value) await search();
}

/** Open an index from the four files' bytes, show what was loaded, set up search. */
async function open(files, source, readMs) {
  index?.free();
  index = null;
  useVectors = false;
  const start = performance.now();
  index = new Index(files.manifest, files.docs, files.bm25, files.vectors ?? null);
  const openMs = performance.now() - start;
  const bytes = [files.docs, files.bm25, files.vectors].reduce((n, b) => n + (b?.length ?? 0), 0);
  const semantic = index.hasVectors ? ` ${index.dim}-d embeddings from ${index.embedder}.` : "";
  setStatus(
    `${source}: ${index.length.toLocaleString()} chunks, ${kib(bytes)}, read in ${readMs.toFixed(0)} ms, ` +
      `opened in ${openMs.toFixed(1)} ms.${semantic}`,
  );
  $("search").hidden = false;
  $("results").replaceChildren();
  $("timing").textContent = "";
  $("q").focus();
  if ($("q").value) await search();
  await prepareVectors(source);
}

/** Fetch a generation's files from a URL folder (the sample, or ?index=). */
async function loadFromUrl(base, label) {
  const dir = base.endsWith("/") ? base : `${base}/`;
  const get = async (name) => {
    const r = await fetch(dir + name);
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  };
  const start = performance.now();
  const manifest = await get("manifest.json");
  const { dim } = JSON.parse(new TextDecoder().decode(manifest));
  const [docs, bm25, vectors] = await Promise.all([
    get("docs.bin"),
    get("bm25.bin"),
    dim > 0 ? get("vectors.bin") : null,
  ]);
  await open({ manifest, docs, bm25, vectors }, label, performance.now() - start);
}

/**
 * Find the generation folder in a directory pick. Accepts a generation
 * folder itself, or any folder above one that has a CURRENT pointer (the
 * index root, or the per-folder directory that contains it).
 */
async function generationFiles(fileList) {
  const files = [...fileList];
  const dirOf = (f) => f.webkitRelativePath.split("/").slice(0, -1).join("/");
  const depth = (f) => f.webkitRelativePath.split("/").length;
  const shallowest = (name) =>
    files.filter((f) => f.name === name).sort((a, b) => depth(a) - depth(b))[0];

  let dir;
  const pointer = shallowest("CURRENT");
  if (pointer) {
    dir = `${dirOf(pointer)}/${(await pointer.text()).trim()}`;
  } else {
    const manifest = shallowest("manifest.json");
    if (!manifest) throw new Error("no notelight index here (no CURRENT or manifest.json)");
    dir = dirOf(manifest);
  }
  const inDir = (name) => files.find((f) => f.webkitRelativePath === `${dir}/${name}`);
  const bytes = async (name, required = true) => {
    const f = inDir(name);
    if (!f && required) throw new Error(`${dir}/${name} is missing`);
    return f ? new Uint8Array(await f.arrayBuffer()) : null;
  };
  const [manifest, docs, bm25, vectors] = await Promise.all([
    bytes("manifest.json"),
    bytes("docs.bin"),
    bytes("bm25.bin"),
    bytes("vectors.bin", false),
  ]);
  return { dir, files: { manifest, docs, bm25, vectors } };
}

/** Wrap each query word in `text` in <mark>, building DOM nodes (no innerHTML). */
function highlighted(text, query) {
  const words = [...new Set(query.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [])];
  const frag = document.createDocumentFragment();
  if (words.length === 0) {
    frag.append(text);
    return frag;
  }
  const escaped = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(?<![\\p{L}\\p{N}])(${escaped.join("|")})(?![\\p{L}\\p{N}])`, "giu");
  let last = 0;
  for (const m of text.matchAll(re)) {
    frag.append(text.slice(last, m.index));
    const mark = document.createElement("mark");
    mark.textContent = m[0];
    frag.append(mark);
    last = m.index + m[0].length;
  }
  frag.append(text.slice(last));
  return frag;
}

async function search() {
  const query = $("q").value.trim();
  if (!index || !query) return;
  const k = Number($("k").value);
  const searched = index; // another index may open while the query is embedded
  let start = performance.now();
  const vector = useVectors ? await embedder.embedQuery(query) : null;
  const embedMs = performance.now() - start;
  if (index !== searched) return;
  start = performance.now();
  const hits = index.search(query, vector, k);
  const searchMs = performance.now() - start;
  const chunks = hits.map((h) => index.chunk(h.id));
  const readMs = performance.now() - start - searchMs;

  const items = hits.map((hit, i) => {
    const c = chunks[i];
    const li = $("result").content.firstElementChild.cloneNode(true);
    li.querySelector(".path").textContent = c.path;
    li.querySelector(".lines").textContent = `lines ${c.lineStart}–${c.lineEnd}`;
    li.querySelector(".headings").textContent = c.headingPath.join(" › ");
    li.querySelector(".text").append(highlighted(c.text, query));
    li.querySelector(".score").textContent =
      `score ${hit.score.toFixed(4)} · BM25 rank ${hit.bm25Rank ?? "–"}` +
      (vector ? ` · vector rank ${hit.vectorRank ?? "–"}` : "");
    return li;
  });
  $("results").replaceChildren(...items);
  const how = vector ? `embed ${embedMs.toFixed(1)} ms, hybrid search` : "keyword search";
  $("timing").textContent = hits.length
    ? `${hits.length} results: ${how} ${searchMs.toFixed(2)} ms, ${readMs.toFixed(2)} ms to read the chunks`
    : "No results.";
}

async function runBench() {
  const button = $("bench");
  button.disabled = true;
  $("bench-out").hidden = false;
  $("bench-out").textContent = "Running…";
  try {
    const url = params.get("bench") ?? "bench-queries.json";
    const { queries } = await (await fetch(url)).json();
    // Let the "Running…" text paint before the synchronous loop starts.
    await new Promise((r) => setTimeout(r, 30));
    const reps = Number(params.get("reps") ?? 20);
    const rows = bench(index, queries, reps);
    if (useVectors) rows.push(...(await benchEmbedded(index, embedder, queries)));
    const head = `wasm in ${navigator.userAgent.match(/(Firefox|Edg|Chrome|Safari)\/[\d.]+/)?.[0] ?? "this browser"}: ` +
      `${index.length} chunks, ${queries.length} queries, k=10, ${reps} reps` +
      (useVectors ? `; embedder on ${embedder.timings.threads} thread(s), 5 reps` : "");
    $("bench-out").textContent = `${head}\n${formatBench(rows)}`;
    window.benchResult = { head, rows };
  } catch (e) {
    $("bench-out").textContent = `Benchmark failed: ${e.message}`;
  } finally {
    button.disabled = false;
  }
}

/**
 * Make the page cross-origin isolated where the server does not send the
 * headers (GitHub Pages, `python -m http.server`): register coi-sw.js, which
 * adds them to this origin's responses, and reload once so the page itself
 * is served with them. Returns true when a reload is under way. Skipped with
 * `?threads=1`, and tried at most once per tab so it can never loop.
 */
async function isolate() {
  if (crossOriginIsolated || params.get("threads") === "1") return false;
  if (!isSecureContext || !("serviceWorker" in navigator)) return false;
  const key = "notelight-isolation-tried";
  try {
    if (sessionStorage.getItem(key)) return false;
    sessionStorage.setItem(key, "1");
    await navigator.serviceWorker.register("./coi-sw.js");
    await navigator.serviceWorker.ready;
  } catch {
    return false; // storage or service workers blocked: run on one thread
  }
  location.reload();
  return true;
}

async function main() {
  if (await isolate()) return;
  try {
    await init();
  } catch (e) {
    setStatus(`Could not load the WebAssembly module (${e.message}). Build it with: python scripts/build_wasm.py`, true);
    return;
  }
  setStatus(`notelight-core ${version()} is loaded. Choose an index.`);

  const guard = (fn) => async (...args) => {
    try {
      await fn(...args);
    } catch (e) {
      setStatus(e.message ?? String(e), true);
    }
  };
  $("load-sample").addEventListener("click", guard(() => loadFromUrl("sample-index", "Sample index")));
  $("pick").addEventListener(
    "change",
    guard(async (ev) => {
      if (!ev.target.files.length) return;
      setStatus("Reading files…");
      const start = performance.now();
      const { dir, files } = await generationFiles(ev.target.files);
      await open(files, dir, performance.now() - start);
    }),
  );
  $("search").addEventListener("submit", (ev) => {
    ev.preventDefault();
    guard(search)();
  });
  $("bench").addEventListener("click", runBench);
  $("forget-model").addEventListener(
    "click",
    guard(async () => {
      const removed = globalThis.caches ? await caches.delete(CACHE_NAME) : false;
      setModelStatus(
        removed
          ? "Deleted the cached model; the next page load downloads it again."
          : "No model is cached in this browser.",
      );
    }),
  );

  const preset = params.get("index");
  if (preset) await guard(() => loadFromUrl(preset, preset))();
}

main();
