// The browser indexer matches the Python one, and updates incrementally.
//
//   python scripts/build_wasm.py      # notelight-wasm into web-demo/pkg/
//   python scripts/vendor_ort.py      # onnxruntime-web (for the semantic parity test)
//   node --test web-demo/tests/
//
// The fixture comes from scripts/make_indexing_fixture.py: a notes folder,
// what `notelight.indexer.sync` made of it with bge-small, and query vectors
// with their top 10. Here the same folder is written to disk, read through
// ingest.js as the browser reads a picked folder, and indexed by indexer.js
// with the wasm core. The keyword tests need only the wasm package; the
// semantic test also needs onnxruntime-web and the model (skipped without
// them, except in CI).
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { after, before, describe, test } from "node:test";
import { pathToFileURL } from "node:url";

import { DOCX_SKIP_REASON } from "../documents.js";
import { BGE_SMALL_EN_V15 as SPEC, modelIdOf, QueryEmbedder } from "../embed.js";
import {
  CancelledError,
  decodeEmbeddings,
  deleteIndex,
  PDF_UNAVAILABLE_REASON,
  embeddingKey,
  embeddingText,
  encodeEmbeddings,
  listIndexes,
  loadGeneration,
  sha256Hex,
  syncFolder,
} from "../indexer.js";
import {
  compareCodePoints,
  compileIgnore,
  decode,
  fromFiles,
  IgnorePatternError,
  splitLines,
  suffixOf,
  walk,
} from "../ingest.js";
import { memoryStore } from "../store.js";
import {
  cosine,
  DEMO,
  FIXTURES,
  fileCache,
  float32FromBase64,
  fromDisk,
  modelFile,
  ORT,
  PKG,
  requireOrSkip,
} from "./helpers.mjs";

const fixture = JSON.parse(await readFile(join(FIXTURES, "indexing.json"), "utf8"));
const bytesOf = (b64) => new Uint8Array(Buffer.from(b64, "base64"));

let wasm = null;
if (existsSync(PKG)) {
  wasm = await import(pathToFileURL(PKG).href);
  wasm.initSync({ module: await readFile(join(DEMO, "pkg", "notelight_wasm_bg.wasm")) });
}

/** Write the fixture folder under a fresh temporary directory. */
async function writeFolder() {
  const root = join(await mkdtemp(join(tmpdir(), "notelight-indexer-")), "notes");
  for (const [rel, b64] of Object.entries(fixture.files)) {
    const path = join(root, ...rel.split("/"));
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, bytesOf(b64));
  }
  return root;
}

/** All chunks of an index generation, in id order. */
function chunksOf(files) {
  const index = new wasm.Index(files.manifest, files.docs, files.bm25, files.vectors);
  try {
    return Array.from({ length: index.length }, (_, i) => index.chunk(i));
  } finally {
    index.free();
  }
}

/**
 * A deterministic stand-in for the model: a vector from the text's SHA-256,
 * and a log of every text it was asked to embed.
 */
function fakeEmbedder(dim = 8) {
  const seen = [];
  return {
    modelId: "fake-embedder",
    dim,
    seen,
    async embedDocuments(texts) {
      seen.push(...texts);
      return Promise.all(
        texts.map(async (t) => {
          const h = await sha256Hex(new TextEncoder().encode(t));
          return Float32Array.from({ length: dim }, (_, i) => parseInt(h.slice(2 * i, 2 * i + 2), 16) / 255);
        }),
      );
    },
  };
}

// --------------------------------------------------------------------------
// Walking and decoding, without the wasm package
// --------------------------------------------------------------------------

describe("ingest.js matches python/notelight/ingest.py", () => {
  test(".notelightignore follows pathspec's GitIgnoreSpec", () => {
    for (const { patterns, cases } of fixture.ignore) {
      const spec = compileIgnore(patterns.join("\n"));
      for (const { path, ignored } of cases) {
        assert.equal(spec.match(path), ignored, `${JSON.stringify(patterns)} on ${path}`);
      }
    }
  });

  test("an unfinished escape is an invalid pattern, as in pathspec", () => {
    assert.throws(() => compileIgnore("ok.md\nbad\\"), IgnorePatternError);
  });

  test("decoding: UTF-8 with or without a BOM, then Windows-1252", () => {
    const enc = (s) => new TextEncoder().encode(s);
    assert.deepEqual(decode(enc("\ufeffa\r\nb\rc")), { text: "a\nb\nc", encoding: "utf-8-sig" });
    assert.deepEqual(decode(Uint8Array.of(0x43, 0x61, 0x66, 0xe9, 0x20, 0x80)), { text: "Café €", encoding: "cp1252" });
    for (const b of [0x81, 0x8d, 0x8f, 0x90, 0x9d]) {
      assert.equal(decode(Uint8Array.of(0xe9, b)).skipReason, "not UTF-8 or Windows-1252 text");
    }
    assert.equal(decode(Uint8Array.of(0x61, 0, 0x62)).skipReason, "binary");
    // Only the first 8 KiB are sniffed for NUL, as in Python.
    const late = new Uint8Array(9000).fill(0x61);
    late[8500] = 0;
    assert.equal(decode(late).encoding, "utf-8-sig");
  });

  test("suffixes, sort order and line splitting follow Python", () => {
    assert.equal(suffixOf("a.MD"), ".md");
    assert.equal(suffixOf("a.tar.Markdown"), ".markdown");
    assert.equal(suffixOf("trailing."), "");
    assert.equal(suffixOf("noext"), "");
    const names = ["apple.md", "Zebra.md", "Ärger.md", "\u{1F600}.md", "\ufffd.md"];
    assert.deepEqual(names.sort(compareCodePoints), ["Zebra.md", "apple.md", "Ärger.md", "\ufffd.md", "\u{1F600}.md"]);
    assert.deepEqual(splitLines("a\r\nb\rc\x1cd\u2028e\n"), ["a", "b", "c", "d", "e"]);
  });

  test("the walk finds the files Python found, in the same order", async () => {
    const root = await writeFolder();
    try {
      const { files } = await walk(fromDisk(root));
      const skipped = new Set(fixture.skipped.map(([path]) => path));
      const walked = files.map((f) => f.path);
      assert.deepEqual(
        walked.filter((p) => !skipped.has(p)),
        fixture.indexed_files,
      );
      assert.deepEqual(walked.filter((p) => skipped.has(p)).sort(), [...skipped].sort());
    } finally {
      await rm(dirname(root), { recursive: true, force: true });
    }
  });

  test("the webkitdirectory file list reads like the folder", async () => {
    const items = Object.entries(fixture.files).map(([rel, b64]) => {
      const bytes = bytesOf(b64);
      return {
        path: `notes/${rel}`,
        file: { size: bytes.length, lastModified: 1, arrayBuffer: async () => bytes.buffer },
      };
    });
    const root = fromFiles(items);
    assert.equal(root.name, "notes");
    const { files } = await walk(root);
    const skipped = new Set(fixture.skipped.map(([path]) => path));
    assert.deepEqual(
      files.map((f) => f.path).filter((p) => !skipped.has(p)),
      fixture.indexed_files,
    );
  });

  test("size limits: text over 10 MiB is left out, PDF up to 50 MiB is found", async () => {
    const big = (size) => ({ size, lastModified: 1, arrayBuffer: async () => new ArrayBuffer(0) });
    const root = fromFiles([
      { path: "r/huge.md", file: big(10 * 1024 * 1024 + 1) },
      { path: "r/ok.md", file: big(10 * 1024 * 1024) },
      { path: "r/paper.pdf", file: big(50 * 1024 * 1024) },
      { path: "r/huge.pdf", file: big(50 * 1024 * 1024 + 1) },
    ]);
    const { files } = await walk(root);
    assert.deepEqual(files.map((f) => f.path), ["ok.md", "paper.pdf"]);
  });
});

// --------------------------------------------------------------------------
// The embedding cache file
// --------------------------------------------------------------------------

test("the embedding cache round-trips, and refuses another model or dimension", () => {
  const vectors = new Map([
    ["ab".repeat(32), Float32Array.of(1, -2.5, 0.125)],
    ["0f".repeat(32), Float32Array.of(0, 3, -1e-7)],
  ]);
  const bytes = encodeEmbeddings("model@x", 3, vectors);
  assert.deepEqual(decodeEmbeddings(bytes, "model@x", 3), vectors);
  assert.equal(decodeEmbeddings(bytes, "model@y", 3).size, 0);
  assert.equal(decodeEmbeddings(bytes, "model@x", 4).size, 0);
  assert.equal(decodeEmbeddings(bytes.subarray(0, bytes.length - 1), "model@x", 3).size, 0);
  assert.equal(decodeEmbeddings(null, "model@x", 3).size, 0);
});

test("embedding keys are sha256(model_id + newline + heading path and text)", async () => {
  const text = embeddingText(["Rust", "Tooling"], "Install");
  assert.equal(text, "Rust > Tooling\nInstall");
  assert.equal(embeddingText([], "plain"), "plain");
  // hashlib.sha256("m\nRust > Tooling\nInstall".encode()).hexdigest()
  assert.equal(
    await embeddingKey("m", text),
    await sha256Hex(new TextEncoder().encode("m\nRust > Tooling\nInstall")),
  );
});

// --------------------------------------------------------------------------
// Keyword-only and incremental indexing (wasm, no model)
// --------------------------------------------------------------------------

describe("indexing with the wasm core", () => {
  let root;
  before(async () => {
    root = await writeFolder();
  });
  after(async () => {
    if (root) await rm(dirname(root), { recursive: true, force: true });
  });

  test("keyword-only: the same chunks, index bytes and BM25 top 10 as Python", async (t) => {
    if (!requireOrSkip(t, wasm, "web-demo/pkg is not built (python scripts/build_wasm.py)")) return;
    const store = memoryStore();
    const { stats, files } = await syncFolder({ wasm, store, root: fromDisk(root), id: "notes", embedder: null });
    assert.deepEqual(chunksOf(files), fixture.chunks);
    assert.equal(await sha256Hex(files.docs), fixture.sha256["docs.bin"]);
    assert.equal(await sha256Hex(files.bm25), fixture.sha256["bm25.bin"]);
    assert.equal(files.vectors, null);
    assert.deepEqual(stats.skipped.map((s) => [s.path, s.reason]).sort(), fixture.skipped);
    assert.equal(stats.files, fixture.indexed_files.length);
    assert.equal(stats.chunks, fixture.chunks.length);
    const index = new wasm.Index(files.manifest, files.docs, files.bm25, null);
    for (const q of fixture.queries) {
      assert.deepEqual(index.searchBm25(q.text, fixture.k).map((h) => h.id), q.bm25, q.text);
    }
    const manifest = JSON.parse(new TextDecoder().decode(files.manifest));
    assert.deepEqual(manifest.metadata, { source: "notes", chunker: { max_chars: 1200 } });
    index.free();
  });

  test("updates embed only new text, and reuse the stored chunks", async (t) => {
    if (!requireOrSkip(t, wasm, "web-demo/pkg is not built (python scripts/build_wasm.py)")) return;
    const store = memoryStore();
    const embedder = fakeEmbedder();
    const sync = () => syncFolder({ wasm, store, root: fromDisk(root), id: "n", name: "notes", embedder });

    const first = await sync();
    const unique = new Set(fixture.chunks.map((c) => embeddingText(c.headingPath, c.text)));
    assert.equal(first.stats.embedded, unique.size);
    assert.equal(first.stats.generation, 1);
    assert.deepEqual(chunksOf(first.files), fixture.chunks);

    // Nothing changed: no new generation, nothing read or embedded.
    const again = await sync();
    assert.ok(again.stats.upToDate);
    assert.equal(again.files, null);
    assert.equal(again.stats.bytesRead, 0);
    assert.equal(embedder.seen.length, unique.size);

    // Touched but identical: read and hashed, but not re-chunked or re-embedded.
    const garden = join(root, "projects", "garden.md");
    await utimes(garden, new Date(), new Date(Date.now() + 5000));
    const touched = await sync();
    assert.ok(touched.stats.upToDate);
    assert.ok(touched.stats.bytesRead > 0);

    // One section edited, one file added, one removed, one renamed.
    const before = await readFile(garden, "utf8");
    await writeFile(garden, before.replace("Turn the compost every two weeks.", "Turn the compost weekly."));
    await writeFile(join(root, "notes", "new.md"), "# New\n\nA brand new note about kayaks.\n");
    await rm(join(root, "notes", "UPPER.MD"));
    const old = join(root, "archive", "2019", "summary.md");
    const moved = join(root, "archive", "2019", "renamed.md");
    await writeFile(moved, await readFile(old));
    await rm(old);
    embedder.seen.length = 0;
    const edited = await sync();
    assert.equal(edited.stats.generation, 2);
    assert.deepEqual(
      { added: edited.stats.added, changed: edited.stats.changed, removed: edited.stats.removed },
      { added: 2, changed: 1, removed: 2 },
    );
    // Only the edited Compost section and the new note: the renamed file's
    // text was embedded before, and the garden note's first section did not change.
    assert.deepEqual(embedder.seen.sort(), [
      "Garden plan > Compost\n## Compost\n\nTurn the compost weekly. Brown material (leaves, cardboard) balances the green (kitchen scraps, grass); a heap that smells is too wet or too green.",
      "New\n# New\n\nA brand new note about kayaks.",
    ]);
    const chunks = chunksOf(edited.files);
    assert.ok(chunks.some((c) => c.path === "archive/2019/renamed.md"));
    assert.ok(!chunks.some((c) => c.path === "notes/UPPER.MD"));

    // Stored: one index, one generation, which reopens without the folder.
    const metas = await listIndexes(store);
    assert.equal(metas.length, 1);
    assert.equal(metas[0].current, 2);
    assert.deepEqual(await store.list("indexes/n"), ["embeddings.bin", "gen-000002", "meta.json"]);
    const { files } = await loadGeneration(store, "n");
    assert.deepEqual(chunksOf(files), chunks);
    // The cache holds exactly the vectors the live generation uses.
    const cache = decodeEmbeddings(await store.read("indexes/n/embeddings.bin"), "fake-embedder", 8);
    assert.equal(cache.size, new Set(chunks.map((c) => embeddingText(c.headingPath, c.text))).size);

    // Keyword-only leaves the embedding cache alone; switching back embeds nothing.
    const keywords = await syncFolder({ wasm, store, root: fromDisk(root), id: "n", embedder: null });
    assert.equal(keywords.files.vectors, null);
    assert.equal(decodeEmbeddings(await store.read("indexes/n/embeddings.bin"), "fake-embedder", 8).size, cache.size);
    embedder.seen.length = 0;
    const back = await sync();
    assert.equal(back.stats.embedded, 0);
    assert.ok(back.files.vectors);

    await deleteIndex(store, "n");
    assert.deepEqual(await listIndexes(store), []);
    assert.equal(store.files.size, 0);
  });

  test("cancelling keeps the finished batches, and the next sync resumes", async (t) => {
    if (!requireOrSkip(t, wasm, "web-demo/pkg is not built (python scripts/build_wasm.py)")) return;
    // 100 notes: two embedding batches of 64 and 36.
    const items = Array.from({ length: 100 }, (_, i) => {
      const bytes = new TextEncoder().encode(`# Note ${i}\n\nBody of note number ${i}.\n`);
      return { path: `many/n${String(i).padStart(3, "0")}.md`, file: { size: bytes.length, lastModified: 1, arrayBuffer: async () => bytes.buffer } };
    });
    const store = memoryStore();
    const embedder = fakeEmbedder();
    const controller = new AbortController();
    const cancelling = { ...embedder, embedDocuments: async (texts) => {
      const out = await embedder.embedDocuments(texts);
      controller.abort(); // the user clicks Cancel during the first batch
      return out;
    } };
    await assert.rejects(
      syncFolder({ wasm, store, root: fromFiles(items), id: "m", embedder: cancelling, signal: controller.signal }),
      CancelledError,
    );
    assert.deepEqual(await listIndexes(store), []); // no generation was switched to
    assert.equal(decodeEmbeddings(await store.read("indexes/m/embeddings.bin"), "fake-embedder", 8).size, 64);
    const { stats } = await syncFolder({ wasm, store, root: fromFiles(items), id: "m", embedder });
    assert.equal(stats.embedded, 36);
    assert.equal(stats.chunks, 100);
  });

  // PDFs with an extractor: documents.test.mjs.
  test("DOCX is skipped with a note, and PDFs when there is no extractor", async (t) => {
    if (!requireOrSkip(t, wasm, "web-demo/pkg is not built (python scripts/build_wasm.py)")) return;
    const file = (text) => {
      const bytes = new TextEncoder().encode(text);
      return { size: bytes.length, lastModified: 1, arrayBuffer: async () => bytes.buffer };
    };
    const root = fromFiles([
      { path: "d/a.md", file: file("# A\n\ntext\n") },
      { path: "d/paper.pdf", file: file("%PDF-1.7") },
      { path: "d/letter.DOCX", file: file("PK") },
    ]);
    const { stats } = await syncFolder({ wasm, store: memoryStore(), root, id: "d", embedder: null });
    assert.deepEqual(stats.skipped, [
      { path: "letter.DOCX", reason: DOCX_SKIP_REASON },
      { path: "paper.pdf", reason: PDF_UNAVAILABLE_REASON },
    ]);
    assert.equal(stats.files, 1);
  });
});

// --------------------------------------------------------------------------
// Semantic parity: bge-small in the browser path vs the Python index
// --------------------------------------------------------------------------

test("with bge-small: the same vectors and hybrid top 10 as Python", async (t) => {
  if (!requireOrSkip(t, wasm, "web-demo/pkg is not built (python scripts/build_wasm.py)")) return;
  if (!requireOrSkip(t, existsSync(ORT), "onnxruntime-web is not vendored (python scripts/vendor_ort.py)")) return;
  const [onnx, tok] = [await modelFile(SPEC.onnx), await modelFile(SPEC.tokenizer)];
  if (!requireOrSkip(t, onnx && tok, "bge-small is not in the model cache (set NOTELIGHT_MODEL_DOWNLOAD=1)")) return;
  const ort = await import(pathToFileURL(ORT).href);
  ort.env.wasm.numThreads = 1;
  const embedder = await QueryEmbedder.load(ort, { cache: fileCache });
  assert.equal(embedder.modelId, fixture.model_id);
  assert.equal(modelIdOf(SPEC), fixture.model_id);
  const root = await writeFolder();
  try {
    const { files, stats } = await syncFolder({ wasm, store: memoryStore(), root: fromDisk(root), id: "notes", embedder });
    assert.deepEqual(chunksOf(files), fixture.chunks);
    assert.equal(await sha256Hex(files.docs), fixture.sha256["docs.bin"]);
    assert.equal(await sha256Hex(files.bm25), fixture.sha256["bm25.bin"]);
    assert.equal(stats.embedded, new Set(fixture.chunks.map((c) => embeddingText(c.headingPath, c.text))).size);

    // Each chunk's vector, before quantization, against the one Python stored.
    const vectors = await embedder.embedDocuments(fixture.chunks.map((c) => embeddingText(c.headingPath, c.text)));
    let worst = 1;
    fixture.chunk_vectors.forEach((b64, i) => {
      worst = Math.min(worst, cosine(vectors[i], float32FromBase64(b64)));
    });
    assert.ok(worst >= 0.999, `worst cosine ${worst}`);
    t.diagnostic(`document vectors: worst 1 - cosine = ${(1 - worst).toExponential(1)}`);

    const index = new wasm.Index(files.manifest, files.docs, files.bm25, files.vectors);
    assert.equal(index.embedder, fixture.model_id);
    for (const q of fixture.queries) {
      const hits = index.search(q.text, float32FromBase64(q.vector), fixture.k).map((h) => h.id);
      assert.deepEqual(hits, q.hybrid, q.text);
    }
    index.free();
  } finally {
    await embedder.release();
    await rm(dirname(root), { recursive: true, force: true });
  }
});
