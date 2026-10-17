// Index a folder in the browser, incrementally, the way
// python/notelight/indexer.py does on disk:
//
// 1. Walk the folder (ingest.js). A file whose size and modification time
//    match the last sync is not read again: its chunks come from the live
//    generation. Otherwise it is read and hashed (SHA-256); if the hash
//    matches, its chunks are reused too.
// 2. Decode and chunk new and changed files with the Rust core (`chunkText`,
//    1200 characters, Markdown structure for .md and .markdown). PDFs are
//    first turned into text by pdf.js (documents.js), and the line each page
//    starts on is kept with the file, as the CLI's `pages` table keeps it.
//    How a file becomes text (`contentMode`: md, txt or the PDF extractor's
//    id) is part of the cache key, so a new pdf.js re-extracts every PDF.
// 3. Embed only chunks whose text is not in the embedding cache, keyed by
//    sha256(model_id + "\n" + embedded text) as in Python, so unchanged,
//    renamed and copied notes are never embedded twice. The embedded text is
//    the heading path, then the chunk (`embeddingText`).
// 4. Build one new generation with the Rust core's writer, store its files,
//    then switch to it by rewriting meta.json (the pointer), and delete the
//    older generation. If nothing changed, nothing is written.
//
// Storage layout (store.js), per index:
//   indexes/<id>/meta.json        name, live generation, per-file cache (with
//                                 page starts for PDFs), last stats
//   indexes/<id>/gen-NNNNNN/      manifest.json, docs.bin, bm25.bin, vectors.bin
//   indexes/<id>/embeddings.bin   the embedding cache (format below)
//
// The module takes the wasm module, the store and the embedder as arguments,
// so it runs unchanged in the indexing worker and in the Node parity test.

import { contentMode, DOCX_SKIP_REASON, pageStarts } from "./documents.js";
import { decode, isMarkdown, suffixOf, walk } from "./ingest.js";

export const DEFAULT_MAX_CHARS = 1200;
/** Missing embeddings are computed this many at a time, as `indexer.EMBED_BATCH`. */
export const EMBED_BATCH = 64;
export const META_FORMAT = "notelight-browser-index";
export const META_VERSION = 1;
export const PDF_UNAVAILABLE_REASON = "PDF text extraction is not available here";

const encoder = new TextEncoder();
const GEN = /^gen-(\d{6})$/;
const genName = (n) => `gen-${String(n).padStart(6, "0")}`;
const indexDir = (id) => `indexes/${id}`;

/** Lowercase hex SHA-256, by Web Crypto. */
export async function sha256Hex(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  let hex = "";
  for (const b of digest) hex += b.toString(16).padStart(2, "0");
  return hex;
}

/** The string embedded for a chunk: heading path, then text (`indexer.embedding_text`). */
export function embeddingText(headingPath, text) {
  return headingPath.length ? `${headingPath.join(" > ")}\n${text}` : text;
}

/** The embedding cache key (`indexer._embedding_key`). */
export const embeddingKey = (modelId, text) => sha256Hex(encoder.encode(`${modelId}\n${text}`));

/**
 * How a cached file was turned into text (`documents.contentMode`). Indexes
 * written before PDFs were read here held only text files and no `mode`.
 */
const modeOf = (file) => file.mode ?? contentMode(suffixOf(file.path));

/** Thrown when indexing is cancelled; the embeddings finished so far are kept. */
export class CancelledError extends Error {
  name = "AbortError";
  constructor() {
    super("Indexing was cancelled");
  }
}

// --------------------------------------------------------------------------
// Embedding cache file
// --------------------------------------------------------------------------
//
//   magic "NLEMB001" (8 bytes) | dim u32 | count u32 | id_len u32 | model id (UTF-8)
//   | zero padding to a multiple of 4
//   | count x (32-byte SHA-256 key | dim x f32), all little-endian

const MAGIC = "NLEMB001";

/** Serialize `vectors` (Map of hex key to Float32Array) for model `modelId`. */
export function encodeEmbeddings(modelId, dim, vectors) {
  const id = encoder.encode(modelId);
  const head = Math.ceil((20 + id.length) / 4) * 4;
  const record = 32 + dim * 4;
  const out = new Uint8Array(head + vectors.size * record);
  const view = new DataView(out.buffer);
  out.set(encoder.encode(MAGIC), 0);
  view.setUint32(8, dim, true);
  view.setUint32(12, vectors.size, true);
  view.setUint32(16, id.length, true);
  out.set(id, 20);
  let at = head;
  for (const [key, v] of vectors) {
    for (let i = 0; i < 32; i++) out[at + i] = parseInt(key.slice(2 * i, 2 * i + 2), 16);
    at += 32;
    for (let i = 0; i < dim; i++, at += 4) view.setFloat32(at, v[i], true);
  }
  return out;
}

/**
 * Read an embedding cache, or an empty Map if there is none, or it belongs to
 * another model or dimension, or it does not parse: a cache can always be
 * rebuilt, so a bad one is dropped rather than trusted.
 */
export function decodeEmbeddings(bytes, modelId, dim) {
  const empty = new Map();
  if (!bytes || bytes.length < 20) return empty;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (new TextDecoder().decode(bytes.subarray(0, 8)) !== MAGIC) return empty;
  const fileDim = view.getUint32(8, true);
  const count = view.getUint32(12, true);
  const idLen = view.getUint32(16, true);
  if (fileDim !== dim || 20 + idLen > bytes.length) return empty;
  if (new TextDecoder().decode(bytes.subarray(20, 20 + idLen)) !== modelId) return empty;
  const head = Math.ceil((20 + idLen) / 4) * 4;
  const record = 32 + dim * 4;
  if (bytes.length !== head + count * record) return empty;
  const out = new Map();
  for (let r = 0, at = head; r < count; r++) {
    let key = "";
    for (let i = 0; i < 32; i++) key += bytes[at + i].toString(16).padStart(2, "0");
    at += 32;
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++, at += 4) v[i] = view.getFloat32(at, true);
    out.set(key, v);
  }
  return out;
}

// --------------------------------------------------------------------------
// Stored indexes
// --------------------------------------------------------------------------

/** A new index id: a slug of the folder name plus a random suffix. */
export function newIndexId(name) {
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "folder";
  const rand = Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) => b.toString(16).padStart(2, "0")).join("");
  return `${slug}-${rand}`;
}

/** The meta.json of index `id`, or null. */
export async function readMeta(store, id) {
  const bytes = await store.read(`${indexDir(id)}/meta.json`);
  if (!bytes) return null;
  const meta = JSON.parse(new TextDecoder().decode(bytes));
  return meta.format === META_FORMAT && meta.version === META_VERSION ? meta : null;
}

/** Every stored index's meta.json, most recently updated first. */
export async function listIndexes(store) {
  const metas = [];
  for (const id of await store.list("indexes")) {
    try {
      const meta = await readMeta(store, id);
      if (meta && meta.current !== null) metas.push(meta);
    } catch {
      // unreadable meta.json: not listed; deleteIndex can still remove it
    }
  }
  return metas.sort((a, b) => (a.updated < b.updated ? 1 : -1));
}

/** The live generation's files of index `id`, ready for `new Index(...)`. */
export async function loadGeneration(store, id, meta = null) {
  meta ??= await readMeta(store, id);
  if (!meta || meta.current === null) throw new Error(`no stored index ${id}`);
  const dir = `${indexDir(id)}/${genName(meta.current)}`;
  const need = async (name) => {
    const b = await store.read(`${dir}/${name}`);
    if (!b) throw new Error(`${dir}/${name} is missing`);
    return b;
  };
  const [manifest, docs, bm25, vectors] = await Promise.all([
    need("manifest.json"),
    need("docs.bin"),
    need("bm25.bin"),
    meta.dim > 0 ? need("vectors.bin") : null,
  ]);
  return { meta, files: { manifest, docs, bm25, vectors } };
}

/** Delete index `id`: its generations, embedding cache and meta.json. */
export async function deleteIndex(store, id) {
  await store.remove(indexDir(id));
}

// --------------------------------------------------------------------------
// Sync
// --------------------------------------------------------------------------

const now = () => performance.now();

/**
 * Create or update index `id` from folder `root` (an ingest.js folder tree).
 *
 * @param {object} o
 * @param {object} o.wasm  the notelight-wasm module: { Index, IndexWriter, chunkText }
 * @param {object} o.store  a store.js store
 * @param {object} o.root  the folder
 * @param {string} o.id  index id (`newIndexId`)
 * @param {string} [o.name]  display name, recorded as the manifest's `source`
 * @param {object|null} o.embedder  {modelId, dim, embedDocuments(texts, {onBatch})} or null
 *   for keyword-only; `onBatch(done)` reports progress within a call and may throw to stop
 * @param {object|null} [o.pdf]  {id, extract(bytes)} turns a PDF into
 *   {text, skipReason, pages} (documents.js `lazyPdfExtractor`); `id` is part
 *   of the cache key (`PDF_EXTRACTOR_ID`). Without it, PDFs are skipped
 * @param {number} [o.maxChars]
 * @param {Function} [o.onProgress]  called with {stage, done, total}
 * @param {AbortSignal} [o.signal]  cancels between files and between embedding batches
 * @returns {Promise<{stats: object, files: object|null}>}  the new generation's
 *   files, or null when the index was already up to date
 */
export async function syncFolder({
  wasm,
  store,
  root,
  id,
  name = root.name,
  embedder,
  pdf = null,
  maxChars = DEFAULT_MAX_CHARS,
  onProgress = () => {},
  signal,
}) {
  const t0 = now();
  const checkCancel = () => {
    if (signal?.aborted) throw new CancelledError();
  };
  const modelId = embedder?.modelId ?? null;
  const dim = embedder?.dim ?? 0;
  const chunker = { max_chars: maxChars };
  const stats = {
    files: 0,
    added: 0,
    changed: 0,
    removed: 0,
    skipped: [],
    chunks: 0,
    embedded: 0,
    reusedFiles: 0,
    pdfs: 0,
    pdfPages: 0,
    pdfBytes: 0,
    bytesRead: 0,
    bytesIndexed: 0,
    generation: null,
    upToDate: false,
    embedder: modelId,
    ms: {},
  };

  onProgress({ stage: "scan", done: 0, total: 0 });
  const prev = await readMeta(store, id);
  const { files } = await walk(root);
  stats.ms.scan = now() - t0;
  checkCancel();

  // The previous generation's chunks, by path: the chunk cache. Only valid if
  // it was chunked with the same settings.
  const known = new Map((prev?.files ?? []).map((f) => [f.path, f]));
  const knownSkipped = new Map((prev?.skipped ?? []).map((f) => [f.path, f]));
  const sameChunker = prev && JSON.stringify(prev.chunker) === JSON.stringify(chunker);
  let prevChunks = null;
  const previousChunks = async () => {
    if (prevChunks === null) {
      prevChunks = new Map();
      if (sameChunker && prev.current !== null) {
        const { files: gen } = await loadGeneration(store, id, prev);
        const index = new wasm.Index(gen.manifest, gen.docs, gen.bm25, gen.vectors);
        try {
          for (let i = 0; i < index.length; i++) {
            const c = index.chunk(i);
            if (!prevChunks.has(c.path)) prevChunks.set(c.path, []);
            prevChunks.get(c.path).push({ headingPath: c.headingPath, lineStart: c.lineStart, lineEnd: c.lineEnd, text: c.text });
          }
        } finally {
          index.free();
        }
      }
    }
    return prevChunks;
  };
  const canReuse = sameChunker && prev.current !== null;

  const docs = []; // {path, size, lastModified, sha256, mode, pages?, chunks}
  const skipped = [];
  const skip = (f, mode, reason) => {
    skipped.push({ path: f.path, size: f.size, lastModified: f.lastModified, mode, reason });
    stats.skipped.push({ path: f.path, reason });
  };
  let readMs = 0;
  let chunkMs = 0;
  let extractMs = 0;
  for (const [i, f] of files.entries()) {
    const isPdf = f.suffix === ".pdf";
    checkCancel();
    if (i % 16 === 0 || isPdf) onProgress({ stage: "read", done: i, total: files.length });
    // A PDF's mode is its extractor's id; with no extractor it is skipped.
    const mode = isPdf ? (pdf?.id ?? null) : contentMode(f.suffix);
    const prior = known.get(f.path);
    const sameMode = prior && modeOf(prior) === mode;
    if (canReuse && sameMode && prior.size === f.size && prior.lastModified === f.lastModified) {
      docs.push({ ...prior, mode, chunks: (await previousChunks()).get(f.path) ?? [] });
      stats.reusedFiles++;
      continue;
    }
    // A file rejected before stays rejected until it changes, unless it is
    // now read differently (a PDF skipped before PDFs were read here).
    const priorSkip = knownSkipped.get(f.path);
    if (
      priorSkip &&
      (priorSkip.mode ?? null) === mode &&
      priorSkip.size === f.size &&
      priorSkip.lastModified === f.lastModified
    ) {
      skip(f, mode, priorSkip.reason);
      continue;
    }
    if (mode === null) {
      skip(f, mode, isPdf ? PDF_UNAVAILABLE_REASON : DOCX_SKIP_REASON);
      continue;
    }
    let t = now();
    let bytes;
    try {
      bytes = await f.read();
    } catch (e) {
      skip(f, mode, `unreadable: ${e.message ?? e}`);
      continue;
    }
    stats.bytesRead += bytes.length;
    const sha256 = await sha256Hex(bytes);
    readMs += now() - t;
    const entry = { path: f.path, size: f.size, lastModified: f.lastModified, sha256, mode };
    if (canReuse && sameMode && prior.sha256 === sha256) {
      // Touched but identical: no need to decode or chunk again.
      const pages = prior.pages ? { pages: prior.pages } : {};
      docs.push({ ...entry, ...pages, chunks: (await previousChunks()).get(f.path) ?? [] });
      stats.reusedFiles++;
      continue;
    }
    t = now();
    let decoded;
    if (isPdf) {
      decoded = await pdf.extract(bytes); // may detach `bytes`
      extractMs += now() - t;
      stats.pdfs++;
      stats.pdfBytes += f.size;
      stats.pdfPages += decoded.pages ?? 0;
      t = now();
    } else {
      decoded = decode(bytes);
    }
    if (decoded.text === null) {
      chunkMs += now() - t;
      skip(f, mode, decoded.skipReason);
      continue;
    }
    if (!prior) stats.added++;
    else if (!sameMode || prior.sha256 !== sha256) stats.changed++;
    const chunks = wasm.chunkText(decoded.text, maxChars, isMarkdown(f.path));
    chunkMs += now() - t;
    docs.push({ ...entry, ...(isPdf ? { pages: pageStarts(decoded.text) } : {}), chunks });
  }
  const seen = new Set(docs.map((d) => d.path));
  stats.removed = [...known.keys()].filter((p) => !seen.has(p)).length;
  stats.files = docs.length;
  stats.chunks = docs.reduce((n, d) => n + d.chunks.length, 0);
  stats.bytesIndexed = docs.reduce((n, d) => n + d.size, 0);
  stats.ms.read = readMs;
  stats.ms.chunk = chunkMs;
  stats.ms.extract = extractMs;
  onProgress({ stage: "read", done: files.length, total: files.length });

  const metaOf = (current) => ({
    format: META_FORMAT,
    version: META_VERSION,
    id,
    name,
    created: prev?.created ?? new Date().toISOString(),
    updated: new Date().toISOString(),
    current,
    embedder: modelId,
    dim,
    chunker,
    store: store.kind,
    files: docs.map(({ path, size, lastModified, sha256, mode, pages }) =>
      pages ? { path, size, lastModified, sha256, mode, pages } : { path, size, lastModified, sha256, mode },
    ),
    skipped,
    stats,
  });
  const writeMeta = (meta) => store.write(`${indexDir(id)}/meta.json`, encoder.encode(JSON.stringify(meta)));

  const unchanged = !(stats.added || stats.changed || stats.removed);
  if (unchanged && canReuse && prev.embedder === modelId) {
    // Refresh the per-file cache (touched files) and keep the generation.
    stats.upToDate = true;
    stats.generation = prev.current;
    stats.ms.total = now() - t0;
    await writeMeta({ ...metaOf(prev.current), stats });
    return { stats, files: null };
  }

  // Embeddings: only the texts the cache does not have.
  const keys = [];
  let vectors = null;
  const cachePath = `${indexDir(id)}/embeddings.bin`;
  if (embedder) {
    let t = now();
    const texts = new Map();
    for (const d of docs) {
      for (const c of d.chunks) {
        const text = embeddingText(c.headingPath, c.text);
        const key = await embeddingKey(modelId, text);
        texts.set(key, text);
        keys.push(key);
      }
    }
    vectors = decodeEmbeddings(await store.read(cachePath), modelId, dim);
    const missing = [...texts.keys()].filter((k) => !vectors.has(k)).sort();
    stats.ms.keys = now() - t;
    t = now();
    try {
      for (let start = 0; start < missing.length; start += EMBED_BATCH) {
        checkCancel();
        onProgress({ stage: "embed", done: start, total: missing.length });
        const batch = missing.slice(start, start + EMBED_BATCH);
        const out = await embedder.embedDocuments(
          batch.map((k) => texts.get(k)),
          {
            onBatch: (n) => {
              onProgress({ stage: "embed", done: start + n, total: missing.length });
              checkCancel(); // stop within the batch; finished batches are kept
            },
          },
        );
        if (out.length !== batch.length || out.some((v) => v.length !== dim)) {
          throw new Error(`the embedder returned ${out.length} vectors for ${batch.length} texts`);
        }
        batch.forEach((k, j) => vectors.set(k, out[j]));
        stats.embedded += batch.length;
      }
    } catch (e) {
      // Keep the finished batches, as Python commits after each one: the
      // next sync starts where this one stopped.
      if (stats.embedded > 0) await store.write(cachePath, encodeEmbeddings(modelId, dim, vectors));
      throw e;
    }
    stats.ms.embed = now() - t;
    onProgress({ stage: "embed", done: missing.length, total: missing.length });
  }

  // One new generation.
  onProgress({ stage: "write", done: 0, total: stats.chunks });
  let t = now();
  const generation = (prev?.current ?? 0) + 1;
  const writer = new wasm.IndexWriter(dim, modelId, JSON.stringify({ source: name, chunker }));
  let k = 0;
  for (const d of docs) {
    for (const c of d.chunks) {
      const vector = embedder ? vectors.get(keys[k++]) : null;
      writer.add(d.path, c.headingPath, c.lineStart, c.lineEnd, c.text, vector);
    }
  }
  const built = writer.build(generation); // consumes the writer
  const genFiles = { manifest: built.manifest, docs: built.docs, bm25: built.bm25, vectors: built.vectors ?? null };
  built.free();
  stats.ms.build = now() - t;

  t = now();
  const dir = `${indexDir(id)}/${genName(generation)}`;
  await store.remove(dir); // leftovers of an interrupted sync
  for (const [file, bytes] of Object.entries({
    "manifest.json": genFiles.manifest,
    "docs.bin": genFiles.docs,
    "bm25.bin": genFiles.bm25,
    "vectors.bin": genFiles.vectors,
  })) {
    if (bytes) await store.write(`${dir}/${file}`, bytes);
  }
  if (embedder) {
    // Keep only the vectors this generation uses, as `_prune` does. A
    // keyword-only sync leaves the cache alone, so switching back re-embeds nothing.
    const live = new Map(keys.map((key) => [key, vectors.get(key)]));
    await store.write(cachePath, encodeEmbeddings(modelId, dim, live));
  }
  stats.generation = generation;
  stats.ms.store = now() - t;
  stats.ms.total = now() - t0;
  await writeMeta(metaOf(generation)); // the switch: readers follow meta.json
  for (const entry of await store.list(indexDir(id))) {
    if (GEN.test(entry) && entry !== genName(generation)) await store.remove(`${indexDir(id)}/${entry}`);
  }
  onProgress({ stage: "done", done: stats.chunks, total: stats.chunks });
  return { stats, files: genFiles };
}
