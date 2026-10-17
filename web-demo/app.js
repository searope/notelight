// The demo page: index a folder of the user's notes in the browser, or load
// an existing index generation, into WebAssembly memory, then search it with
// the Rust core. When the index has bge-small embeddings, the query is
// embedded here too (embed.js, onnxruntime-web) and the search is hybrid.
// Indexing and storage run in a worker (index-worker.js): files are read
// there, chunked by the core, embedded, and kept in the Origin Private File
// System (or IndexedDB). The page, the .wasm, the runtime and the sample
// index are served from this folder; the only other request is the one-time
// model download, which is verified by SHA-256 and kept in the Cache API.
//
// URL parameters, for measuring other indexes without the folder picker:
//   ?index=<url of a generation folder>  load it on start
//   ?bench=<url of a queries file>       queries for the Benchmark button
//   ?model=<url of a folder>             fetch the model files from there instead
//                                        of the Hub (laid out like the Python
//                                        model cache's revision folder)
//   ?threads=1                           embed on one thread (no service worker)
//   ?index-threads=N                     threads for embedding while indexing
//                                        (default: up to 8, as many as there are cores)
import init, { Index, version } from "./pkg/notelight_wasm.js";
import { bench, benchEmbedded, formatBench } from "./bench.js";
import {
  BGE_SMALL_EN_V15,
  browserCache,
  checkIndexEmbedder,
  modelIdOf,
  QueryEmbedder,
} from "./embed.js";
import { formatPages, pageRange, pagesByPath } from "./documents.js";
import { deleteHandle, loadHandle, saveHandle } from "./store.js";

const ORT_URL = "./vendor/onnxruntime-web/ort.wasm.min.mjs";
const CACHE_NAME = "notelight-models";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
let index = null;
let embedder = null; // a QueryEmbedder once loaded, reused across indexes
let embedderLoading = null; // the pending load, so two indexes do not load it twice
let useVectors = false; // whether the open index takes our query vectors
let openId = null; // the stored index that is open, if any
let openPages = new Map(); // path -> line each page starts on, for the open index's PDFs

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

function setModelStatus(text, isError = false) {
  $("model-status").hidden = !text;
  $("model-status").textContent = text;
  $("model-status").classList.toggle("error", isError);
}

/**
 * The activity bar at the top of the window: shows what the page is doing
 * while it works. `begin(text)` returns a job with `update(text, done, total)`
 * (a known `total` shows a progress bar) and `end()`; the newest unfinished job
 * is shown. `delay` ms hides jobs that finish quickly, so a 10 ms search does
 * not flash the bar.
 */
const activity = (() => {
  const jobs = [];
  function render() {
    const top = [...jobs].reverse().find((j) => j.visible);
    $("activity").hidden = !top;
    document.body.setAttribute("aria-busy", String(jobs.length > 0));
    if (!top) return;
    $("activity-text").textContent = top.text;
    const bar = $("activity-bar");
    bar.hidden = !(top.total > 0);
    if (top.total > 0) {
      bar.max = top.total;
      bar.value = Math.min(top.done, top.total);
    }
  }
  return {
    begin(text, { delay = 0 } = {}) {
      const job = { text, done: 0, total: 0, visible: delay === 0 };
      jobs.push(job);
      if (delay > 0) {
        job.timer = setTimeout(() => {
          job.visible = true;
          render();
        }, delay);
      }
      render();
      return {
        update(newText, done = 0, total = 0) {
          Object.assign(job, { text: newText ?? job.text, done, total });
          render();
        },
        end() {
          clearTimeout(job.timer);
          const i = jobs.indexOf(job);
          if (i >= 0) jobs.splice(i, 1);
          render();
        },
      };
    },
  };
})();

/** Run `fn` with the activity bar showing `text`; the job is passed to `fn`. */
async function busy(text, fn, options) {
  const job = activity.begin(text, options);
  try {
    return await fn(job);
  } finally {
    job.end();
  }
}

/**
 * Controls that start a load or an indexing run are disabled while one is in
 * progress, so a second click cannot start another on top of it.
 */
let uiLocks = 0;
let indexingReady = false;
function setLocked(locked) {
  uiLocks += locked ? 1 : -1;
  const off = uiLocks > 0;
  $("load-sample").disabled = off;
  $("pick").disabled = off;
  $("index-folder").disabled = off || !indexingReady;
  for (const b of document.querySelectorAll("#stored button")) b.disabled = off;
}
async function locked(fn) {
  setLocked(true);
  try {
    return await fn();
  } finally {
    setLocked(false);
  }
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
    const job = activity.begin("Loading the bge-small model…");
    const onProgress = ({ file, received, total }) => {
      if (file === BGE_SMALL_EN_V15.onnx.path) {
        const text = `Downloading bge-small (one time): ${kib(received)} of ${kib(total)}`;
        setModelStatus(`${text}…`);
        job.update(text, received, total);
      }
    };
    const onStage = ({ stage, file }) => {
      if (stage === "verify" && file === BGE_SMALL_EN_V15.onnx.path) {
        job.update("Checking the model's SHA-256…");
      } else if (stage === "session") {
        job.update("Starting the model…");
      }
    };
    setModelStatus("Loading the bge-small query embedder…");
    try {
      const loaded = await QueryEmbedder.load(ort, { mirror: params.get("model"), cache, onProgress, onStage });
      loaded.timings.threads = ort.env.wasm.numThreads;
      window.embedderTimings = loaded.timings; // for measuring from DevTools
      return loaded;
    } finally {
      job.end();
    }
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

/**
 * Open an index from the four files' bytes, show what was loaded, set up
 * search. `meta` is a stored index's meta.json, for its PDFs' pages.
 */
async function open(files, source, readMs, storedId = null, meta = null) {
  index?.free();
  index = null;
  useVectors = false;
  openId = storedId;
  openPages = pagesByPath(meta);
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
  // The search box sits below both ways of getting an index; bring it into
  // view so it is obvious the index is ready to search.
  $("status").textContent += " Search it below.";
  $("search").scrollIntoView({ block: "center", behavior: "smooth" });
  $("search").classList.remove("ready");
  void $("search").offsetWidth; // restart the highlight animation
  $("search").classList.add("ready");
  $("q").focus({ preventScroll: true });
  if ($("q").value) await search();
  // The index is searchable by keywords now; the embedder (a 127 MiB download
  // the first time) loads in the background, shown in the activity bar, and
  // search becomes hybrid when it is ready. Not awaited, so the controls are
  // not locked for the whole download. prepareVectors reports its own errors.
  prepareVectors(source).catch((e) => setModelStatus(e.message ?? String(e), true));
}

/** Fetch a generation's files from a URL folder (the sample, or ?index=). */
async function loadFromUrl(base, label) {
  await locked(() => busy(`Loading ${label}…`, () => fetchIndex(base, label)));
}

async function fetchIndex(base, label) {
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
  const submit = $("search").querySelector('button[type="submit"]');
  submit.disabled = true;
  $("results").setAttribute("aria-busy", "true");
  let start = performance.now();
  let vector;
  try {
    vector = useVectors
      ? await busy("Embedding the query…", () => embedder.embedQuery(query), { delay: 150 })
      : null;
  } finally {
    submit.disabled = false;
    $("results").removeAttribute("aria-busy");
  }
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
    const starts = openPages.get(c.path);
    const pages = starts ? ` (${formatPages(...pageRange(starts, c.lineStart, c.lineEnd))})` : "";
    li.querySelector(".lines").textContent = `lines ${c.lineStart}–${c.lineEnd}${pages}`;
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
  const job = activity.begin("Running the benchmark…");
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
    job.end();
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

// --------------------------------------------------------------------------
// Your notes: indexing in the worker, stored indexes
// --------------------------------------------------------------------------

let worker = null;
let nextReq = 0;
const pending = new Map();

/** Ask the worker to run `op`; `onProgress` gets its progress messages. */
function call(op, payload = {}, onProgress = null) {
  const reqId = ++nextReq;
  return new Promise((resolve, reject) => {
    pending.set(reqId, { resolve, reject, onProgress });
    worker.postMessage({ reqId, op, ...payload });
  });
}

function startWorker() {
  worker = new Worker(new URL("./index-worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    const req = pending.get(data.reqId);
    if (!req) return;
    if (data.progress) {
      req.onProgress?.(data.progress);
      return;
    }
    pending.delete(data.reqId);
    if (data.error) {
      const e = new Error(data.error.message);
      e.name = data.error.name;
      req.reject(e);
    } else {
      req.resolve(data.result);
    }
  };
  worker.onerror = (e) => {
    // A worker that fails to load (or crashes) answers nothing: fail every request.
    const error = new Error(`the indexing worker failed (${e.message || "it could not be loaded"})`);
    for (const req of pending.values()) req.reject(error);
    pending.clear();
  };
}

function setIndexStatus(text, isError = false) {
  $("index-status").textContent = text;
  $("index-status").classList.toggle("error", isError);
}

const STAGES = {
  scan: "Looking through the folder…",
  model: "Loading the bge-small embedder…",
  read: "Reading and chunking files",
  embed: "Embedding chunks",
  write: "Writing the index…",
  done: "Done",
};

let indexJob = null; // the activity-bar job of the running indexing run

function showProgress({ stage, done, total }) {
  const bar = $("progress-bar");
  if (total > 0) {
    bar.max = total;
    bar.value = done;
  } else {
    bar.removeAttribute("value"); // indeterminate
  }
  let text = STAGES[stage] ?? stage;
  if (stage === "model" && total > 0) text = `Downloading bge-small (one time): ${kib(done)} of ${kib(total)}`;
  else if (total > 0) text = `${text.replace(/…$/, "")}: ${done.toLocaleString()} of ${total.toLocaleString()}`;
  $("progress-text").textContent = text;
  indexJob?.update(text, done, total);
}

const rate = (n, msTaken) => (msTaken > 0 ? n / (msTaken / 1000) : 0);
const fmtRate = (x) => (x >= 100 ? x.toFixed(0) : x.toFixed(1));

/** One line on what a sync did and how fast, from the indexer's stats. */
function describeSync(stats, name) {
  const t = stats.ms;
  if (stats.upToDate) {
    return `${name} is up to date: ${stats.files} files checked in ${ms(t.total)}, nothing to re-index.`;
  }
  const mib = stats.bytesIndexed / 1048576;
  const parts = [
    `Indexed ${name}: ${stats.files.toLocaleString()} files (${kib(stats.bytesIndexed)}), ` +
      `${stats.chunks.toLocaleString()} chunks in ${ms(t.total)}: ` +
      `${fmtRate(rate(stats.chunks, t.total))} chunks/s, ${fmtRate(mib / (t.total / 1000))} MiB/s.`,
  ];
  if (stats.pdfs) {
    parts.push(
      `Extracted ${stats.pdfs.toLocaleString()} PDF${stats.pdfs === 1 ? "" : "s"} ` +
        `(${stats.pdfPages.toLocaleString()} pages, ${kib(stats.pdfBytes)}) in ${ms(t.extract)}: ` +
        `${fmtRate(rate(stats.pdfPages, t.extract))} pages/s` +
        (t.pdfjs ? `; pdf.js loaded in ${ms(t.pdfjs)} (not counted above).` : "."),
    );
  }
  if (stats.embedder) {
    parts.push(
      stats.embedded
        ? `Embedded ${stats.embedded.toLocaleString()} new chunks in ${ms(t.embed)} ` +
            `(${fmtRate(rate(stats.embedded, t.embed))} chunks/s on ${stats.threads} thread${stats.threads === 1 ? "" : "s"}); ` +
            `the rest came from the cache.`
        : "Every embedding came from the cache.",
    );
    if (t.model) parts.push(`Model loaded in ${ms(t.model)} (not counted above).`);
  } else {
    parts.push("Keyword search only.");
  }
  if (stats.added || stats.changed || stats.removed) {
    parts.push(`${stats.added} added, ${stats.changed} changed, ${stats.removed} removed.`);
  }
  return parts.join(" ");
}

function showSkipped(skipped) {
  const box = $("skipped");
  box.hidden = skipped.length === 0;
  if (box.hidden) return;
  box.querySelector("summary").textContent = `${skipped.length} file${skipped.length === 1 ? "" : "s"} skipped`;
  box.querySelector("ul").replaceChildren(
    ...skipped.map(({ path, reason }) => {
      const li = document.createElement("li");
      li.textContent = `${path}: ${reason}`;
      return li;
    }),
  );
}

async function refreshStored() {
  const metas = await call("list");
  const items = metas.map((meta) => {
    const li = $("stored-item").content.firstElementChild.cloneNode(true);
    li.querySelector(".stored-name").textContent = meta.name;
    const kind = meta.embedder ? "semantic + keywords" : "keywords only";
    li.querySelector(".stored-info").textContent =
      `${meta.files.length.toLocaleString()} files, ${meta.stats.chunks.toLocaleString()} chunks, ${kind}, ` +
      `updated ${new Date(meta.updated).toLocaleString()}`;
    for (const button of li.querySelectorAll("button")) {
      button.addEventListener("click", () => storedAction(button.dataset.action, meta));
    }
    return li;
  });
  $("stored").replaceChildren(...items);
  const usage = (await navigator.storage?.estimate?.())?.usage;
  const persisted = await navigator.storage?.persisted?.();
  const where = storeKind === "opfs" ? "the browser's private file system for this site (OPFS)" : "IndexedDB";
  $("storage-note").textContent = metas.length
    ? `Indexes are kept in ${where}` +
      (usage ? `; this site uses ${kib(usage)} in all, the model included` : "") +
      (persisted ? "." : ". The browser may clear them if the disk runs low on space.")
    : "";
}

let storeKind = null;
let indexing = false;

/** Index (or update) a folder in the worker, then open the result. */
async function runIndex({ id = null, name, source, semantic }) {
  if (indexing) return;
  indexing = true;
  setLocked(true);
  indexJob = activity.begin(`Indexing ${name}…`);
  $("progress").hidden = false;
  $("cancel").disabled = false;
  setIndexStatus("");
  showSkipped([]);
  showProgress({ stage: "scan", done: 0, total: 0 });
  // Ask the browser to keep the indexes when space runs low (Chrome may
  // grant it silently; Firefox asks).
  navigator.storage?.persist?.().catch(() => {});
  try {
    const { stats, meta, files } = await call(
      "index",
      { id, name, source, semantic, mirror: params.get("model") },
      showProgress,
    );
    window.indexStats = stats; // for measuring from DevTools
    if (source.handle) await saveHandle(meta.id, source.handle).catch(() => {});
    setIndexStatus(describeSync(stats, meta.name));
    showSkipped(stats.skipped);
    await refreshStored();
    if (files) {
      await open(files, meta.name, 0, meta.id, meta);
    } else if (openId !== meta.id) {
      await loadStored(meta);
    }
  } catch (e) {
    const cancelled = e.name === "AbortError";
    setIndexStatus(
      cancelled ? "Cancelled. Embeddings finished so far are kept, so indexing again resumes from there." : e.message,
      !cancelled,
    );
  } finally {
    indexing = false;
    indexJob.end();
    indexJob = null;
    setLocked(false);
    $("progress").hidden = true;
  }
}

async function openStored(meta) {
  await locked(() => loadStored(meta));
}

async function loadStored(meta) {
  await busy(`Opening ${meta.name}…`, async () => {
    setStatus(`Opening ${meta.name}…`);
    const start = performance.now();
    const loaded = await call("load", { id: meta.id });
    await open(loaded.files, meta.name, performance.now() - start, meta.id, loaded.meta);
  });
}

/** Pick a folder: the File System Access API where there is one, else the input. */
async function pickFolder(updateId = null, semantic = $("semantic").checked) {
  if (window.showDirectoryPicker) {
    let handle;
    try {
      handle = await window.showDirectoryPicker({ id: "notelight", mode: "read" });
    } catch (e) {
      if (e.name !== "AbortError") setIndexStatus(e.message, true);
      return;
    }
    await runIndex({ id: updateId, name: handle.name, source: { handle }, semantic });
  } else {
    pendingPick = { id: updateId, semantic };
    $("index-pick").value = "";
    $("index-pick").click();
  }
}
let pendingPick = null;

async function storedAction(action, meta) {
  if (action === "open") {
    await openStored(meta);
  } else if (action === "update") {
    const semantic = meta.embedder !== null;
    const handle = window.showDirectoryPicker ? await loadHandle(meta.id).catch(() => null) : null;
    if (handle && (await handle.requestPermission({ mode: "read" })) === "granted") {
      await runIndex({ id: meta.id, name: meta.name, source: { handle }, semantic });
    } else {
      setIndexStatus(`Pick the folder ${meta.name} again to update its index.`);
      await pickFolder(meta.id, semantic);
    }
  } else if (action === "delete") {
    if (!confirm(`Delete the index of ${meta.name} from this browser? Your files are not touched.`)) return;
    await locked(() => busy(`Deleting ${meta.name}…`, () => call("delete", { id: meta.id })));
    await deleteHandle(meta.id).catch(() => {});
    if (openId === meta.id) {
      index?.free();
      index = null;
      openId = null;
      $("search").hidden = true;
      $("results").replaceChildren();
      $("timing").textContent = "";
      setStatus("The open index was deleted. Choose another.");
    }
    setIndexStatus(`Deleted the index of ${meta.name}.`);
    await refreshStored();
  }
}

async function setUpIndexing() {
  try {
    startWorker();
    // Indexing embeds hundreds of chunks, where 8 threads are 1.5x faster than
    // 4 (design notes, "v2: indexing in the browser"); queries stay on 4.
    const threads = params.get("threads") === "1" ? 1 : Number(params.get("index-threads")) || 8;
    ({ store: storeKind } = await busy("Starting the indexer…", () => call("init", { threads })));
  } catch (e) {
    setIndexStatus(`Indexing is not available in this browser (${e.message}).`, true);
    return;
  }
  const guard = (fn) => async (...args) => {
    try {
      await fn(...args);
    } catch (e) {
      setIndexStatus(e.message ?? String(e), true);
    }
  };
  indexingReady = true;
  $("index-folder").disabled = uiLocks > 0;
  $("index-folder").addEventListener("click", guard(() => pickFolder()));
  $("index-pick").addEventListener(
    "change",
    guard(async (ev) => {
      const picked = [...ev.target.files];
      const { id, semantic } = pendingPick ?? { id: null, semantic: $("semantic").checked };
      pendingPick = null;
      if (!picked.length) return;
      // webkitRelativePath does not survive postMessage, so send it alongside.
      const files = picked.map((file) => ({ path: file.webkitRelativePath, file }));
      const name = files[0].path.split("/")[0];
      await runIndex({ id, name, source: { files }, semantic });
    }),
  );
  $("cancel").addEventListener("click", () => {
    $("cancel").disabled = true;
    $("progress-text").textContent = "Cancelling after the current batch…";
    worker.postMessage({ op: "cancel" });
  });
  await guard(() => busy("Listing your stored indexes…", refreshStored))();
}

async function main() {
  if (await isolate()) return;
  try {
    await busy("Loading the WebAssembly module…", () => init());
  } catch (e) {
    setStatus(`Could not load the WebAssembly module (${e.message}). Build it with: python scripts/build_wasm.py`, true);
    return;
  }
  setStatus(`notelight-core ${version()} is loaded. Index a folder or choose an index.`);
  await setUpIndexing();

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
      await locked(() =>
        busy("Reading the index files…", async () => {
          const start = performance.now();
          const { dir, files } = await generationFiles(ev.target.files);
          await open(files, dir, performance.now() - start);
        }),
      );
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
