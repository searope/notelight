// The demo page: load an index generation into WebAssembly memory, then
// search it with the Rust core. No framework and no network beyond this
// folder: the sample index and the .wasm are served next to this file.
//
// URL parameters, for measuring other indexes without the folder picker:
//   ?index=<url of a generation folder>  load it on start
//   ?bench=<url of a queries file>       queries for the Benchmark button
import init, { Index, version } from "./pkg/notelight_wasm.js";
import { bench, formatBench } from "./bench.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
let index = null;

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

const kib = (n) => (n / 1024 >= 1024 ? `${(n / 1048576).toFixed(1)} MiB` : `${Math.round(n / 1024)} KiB`);

/** Open an index from the four files' bytes and show what was loaded. */
function open(files, source, readMs) {
  index?.free();
  index = null;
  const start = performance.now();
  index = new Index(files.manifest, files.docs, files.bm25, files.vectors ?? null);
  const openMs = performance.now() - start;
  const bytes = [files.docs, files.bm25, files.vectors].reduce((n, b) => n + (b?.length ?? 0), 0);
  const semantic = index.hasVectors
    ? ` It has ${index.dim}-d embeddings from ${index.embedder}; this spike cannot embed queries in the browser, so it ranks by BM25 and phrases only.`
    : "";
  setStatus(
    `${source}: ${index.length.toLocaleString()} chunks, ${kib(bytes)}, read in ${readMs.toFixed(0)} ms, ` +
      `opened in ${openMs.toFixed(1)} ms.${semantic}`,
  );
  $("search").hidden = false;
  $("results").replaceChildren();
  $("timing").textContent = "";
  $("q").focus();
  if ($("q").value) search();
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
  open({ manifest, docs, bm25, vectors }, label, performance.now() - start);
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

function search() {
  const query = $("q").value.trim();
  if (!index || !query) return;
  const k = Number($("k").value);
  const start = performance.now();
  const hits = index.search(query, null, k);
  const searchMs = performance.now() - start;
  const chunks = hits.map((h) => index.chunk(h.id));
  const totalMs = performance.now() - start;

  const items = hits.map((hit, i) => {
    const c = chunks[i];
    const li = $("result").content.firstElementChild.cloneNode(true);
    li.querySelector(".path").textContent = c.path;
    li.querySelector(".lines").textContent = `lines ${c.lineStart}–${c.lineEnd}`;
    li.querySelector(".headings").textContent = c.headingPath.join(" › ");
    li.querySelector(".text").append(highlighted(c.text, query));
    li.querySelector(".score").textContent =
      `score ${hit.score.toFixed(4)} · BM25 rank ${hit.bm25Rank ?? "–"}`;
    return li;
  });
  $("results").replaceChildren(...items);
  $("timing").textContent = hits.length
    ? `${hits.length} results: search ${searchMs.toFixed(2)} ms, ${(totalMs - searchMs).toFixed(2)} ms to read the chunks`
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
    const head = `wasm in ${navigator.userAgent.match(/(Firefox|Edg|Chrome|Safari)\/[\d.]+/)?.[0] ?? "this browser"}: ` +
      `${index.length} chunks, ${queries.length} queries, k=10, ${reps} reps`;
    $("bench-out").textContent = `${head}\n${formatBench(rows)}`;
    window.benchResult = { head, rows };
  } catch (e) {
    $("bench-out").textContent = `Benchmark failed: ${e.message}`;
  } finally {
    button.disabled = false;
  }
}

async function main() {
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
      open(files, dir, performance.now() - start);
    }),
  );
  $("search").addEventListener("submit", (ev) => {
    ev.preventDefault();
    guard(search)();
  });
  $("bench").addEventListener("click", runBench);

  const preset = params.get("index");
  if (preset) await guard(() => loadFromUrl(preset, preset))();
}

main();
