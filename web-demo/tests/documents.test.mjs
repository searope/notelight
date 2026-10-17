// PDF text in the browser: documents.js with the vendored pdf.js, and PDFs
// indexed and searched through indexer.js.
//
//   python scripts/vendor_pdfjs.py    # pdf.js into web-demo/vendor/pdfjs/
//   python scripts/build_wasm.py      # notelight-wasm, for the indexing tests
//   node --test web-demo/tests/
//
// The PDFs come from scripts/make_pdf_fixtures.py, which writes them object
// by object so their text is known, and records in pdf.json the text the CLI's
// page layout gives them and what pypdf extracts. pdf.js and pypdf are
// different extractors; on these simple files they agree, and the tests pin
// down what must hold on any file: the page joining, the line-to-page
// mapping, and the skip reasons.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, test } from "node:test";
import { pathToFileURL } from "node:url";

import {
  cleanText,
  contentMode,
  DOCX_SKIP_REASON,
  extractPdf,
  formatPages,
  joinPages,
  lazyPdfExtractor,
  PAGE_BREAK,
  PDF_EXTRACTOR_ID,
  PDFJS_VERSION,
  pageRange,
  pageStarts,
  pagesByPath,
  pageText,
} from "../documents.js";
import { PDF_UNAVAILABLE_REASON, readMeta, syncFolder } from "../indexer.js";
import { fromFiles } from "../ingest.js";
import { memoryStore } from "../store.js";
import { DEMO, FIXTURES, PKG, requireOrSkip } from "./helpers.mjs";

const PDFJS = join(DEMO, "vendor", "pdfjs");
const PDF_DIR = join(FIXTURES, "pdf");
const fixtures = JSON.parse(await readFile(join(PDF_DIR, "pdf.json"), "utf8"));
const pdfBytes = async (name) => new Uint8Array(await readFile(join(PDF_DIR, name)));

let pdfjs = null;
if (existsSync(join(PDFJS, "pdf.min.mjs"))) {
  // The order matters: with the worker module loaded, pdf.js parses in this
  // thread, as it does in the indexing worker.
  await import(pathToFileURL(join(PDFJS, "pdf.worker.min.mjs")).href);
  pdfjs = await import(pathToFileURL(join(PDFJS, "pdf.min.mjs")).href);
}
let wasm = null;
if (existsSync(PKG)) {
  wasm = await import(pathToFileURL(PKG).href);
  wasm.initSync({ module: await readFile(join(DEMO, "pkg", "notelight_wasm_bg.wasm")) });
}

/** A CMap loader over the vendored folder that records what it was asked for. */
function diskCMaps() {
  const asked = [];
  const load = async (name) => {
    asked.push(name);
    return new Uint8Array(await readFile(join(PDFJS, "cmaps", name)));
  };
  return { asked, load };
}

const needPdfjs = (t) => requireOrSkip(t, pdfjs, "pdf.js is not vendored (python scripts/vendor_pdfjs.py)");
const needWasm = (t) => requireOrSkip(t, wasm, "web-demo/pkg is not built (python scripts/build_wasm.py)");

// --------------------------------------------------------------------------
// The page layout, without pdf.js (documents.py's functions, ported)
// --------------------------------------------------------------------------

describe("page joining and mapping match python/notelight/documents.py", () => {
  test("page_range and page_starts", () => {
    const starts = [1, 4, 9];
    assert.deepEqual(pageRange(starts, 1, 3), [1, 1]);
    assert.deepEqual(pageRange(starts, 3, 4), [1, 2]);
    assert.deepEqual(pageRange(starts, 5, 20), [2, 3]);
    assert.deepEqual(pageStarts("one page\nonly\n"), [1]);
    assert.deepEqual(pageStarts("a\n\fb\n\fc\n"), [1, 2, 3]);
    // A form feed on line 1 is not a page break; one inside a line is not either.
    assert.deepEqual(pageStarts("\fa\nb\fc\n"), [1]);
  });

  test("pages end in a newline and are joined by form feeds; blank pages keep a line", () => {
    const text = joinPages(["one\ntwo\n\n", "", "three"]);
    assert.equal(text, "one\ntwo\n\f\n\fthree\n");
    assert.deepEqual(pageStarts(text), [1, 3, 4]);
    assert.ok(text.split(PAGE_BREAK).every((p) => p.endsWith("\n")));
  });

  test("_clean: NUL, form feed and CR go; a vertical tab is a newline", () => {
    assert.equal(cleanText("a\r\nb\0c\fd\re\vf"), "a\nbcde\nf");
    assert.equal(pageText([{ str: "x\f", hasEOL: true }, { str: "y" }, { hasEOL: true }]), "x\ny\n");
  });

  test("labels and modes", () => {
    assert.equal(formatPages(3, 3), "p. 3");
    assert.equal(formatPages(3, 4), "pp. 3-4");
    assert.equal(contentMode(".md"), "md");
    assert.equal(contentMode(".markdown"), "md");
    assert.equal(contentMode(".txt"), "txt");
    assert.equal(contentMode(".pdf"), `pdf1:pdfjs-${PDFJS_VERSION}`);
    assert.equal(contentMode(".docx"), null);
    assert.deepEqual([...pagesByPath({ files: [{ path: "a.md" }, { path: "b.pdf", pages: [1, 5] }] })], [["b.pdf", [1, 5]]]);
    assert.deepEqual([...pagesByPath(null)], []);
  });

  test("the extractor id names the version that scripts/vendor_pdfjs.py pins", async () => {
    const script = await readFile(join(DEMO, "..", "scripts", "vendor_pdfjs.py"), "utf8");
    assert.equal(script.match(/^VERSION = "([^"]+)"$/m)?.[1], PDFJS_VERSION);
    assert.equal(PDF_EXTRACTOR_ID, `pdf1:pdfjs-${PDFJS_VERSION}`);
  });
});

// --------------------------------------------------------------------------
// Extraction with the vendored pdf.js
// --------------------------------------------------------------------------

describe("pdf.js extraction", () => {
  test("the vendored pdf.js is the pinned version", (t) => {
    if (!needPdfjs(t)) return;
    assert.equal(pdfjs.version, PDFJS_VERSION);
  });

  for (const name of ["report.pdf", "field-guide.pdf", "owner-only.pdf", "cjk.pdf"]) {
    test(`${name}: the text and pages the CLI's layout gives`, async (t) => {
      if (!needPdfjs(t)) return;
      const cmaps = diskCMaps();
      const out = await extractPdf(pdfjs, await pdfBytes(name), cmaps.load);
      const { expected, pypdf } = fixtures[name];
      assert.equal(out.skipReason, undefined);
      assert.equal(out.text, expected);
      assert.equal(out.pages, expected.split(PAGE_BREAK).length);
      // On these files pypdf agrees; on real ones the two differ (see the design notes).
      assert.equal(pypdf.text, expected);
      if (name === "cjk.pdf") assert.deepEqual(cmaps.asked.sort(), ["Adobe-Japan1-UCS2.bcmap", "UniJIS-UCS2-H.bcmap"]);
      else assert.deepEqual(cmaps.asked, []);
    });
  }

  test("every line maps to the page it was drawn on", async (t) => {
    if (!needPdfjs(t)) return;
    const { text } = await extractPdf(pdfjs, await pdfBytes("field-guide.pdf"), diskCMaps().load);
    const starts = pageStarts(text);
    assert.deepEqual(starts, [1, 16, 31, 46, 61]);
    text
      .split("\n")
      .slice(0, -1)
      .forEach((line, i) => {
        const chapter = Number(line.match(/(?:Chapter |Observation )(\d)/)[1]);
        assert.deepEqual(pageRange(starts, i + 1, i + 1), [chapter, chapter], `line ${i + 1}: ${line}`);
      });
    // The blank third page of report.pdf is a line of its own.
    const report = (await extractPdf(pdfjs, await pdfBytes("report.pdf"), diskCMaps().load)).text;
    assert.deepEqual(pageStarts(report), [1, 3, 5, 6]);
    assert.deepEqual(pageRange(pageStarts(report), 6, 6), [4, 4]);
  });

  test("encrypted, image-only and broken PDFs are skipped with the CLI's reasons", async (t) => {
    if (!needPdfjs(t)) return;
    const reason = async (bytes) => (await extractPdf(pdfjs, bytes, diskCMaps().load)).skipReason;
    for (const name of ["locked.pdf", "scanned.pdf"]) {
      assert.equal(await reason(await pdfBytes(name)), fixtures[name].pypdf.skipReason, name);
    }
    assert.match(await reason(await pdfBytes("broken.pdf")), /^unreadable PDF: InvalidPDFException: /);
    assert.match(fixtures["broken.pdf"].pypdf.skipReason, /^unreadable PDF: /);
    for (const bytes of [new Uint8Array(0), new TextEncoder().encode("not a pdf at all")]) {
      assert.match(await reason(bytes), /^unreadable PDF: /);
    }
    // A truncated file either yields text or a reason; it never throws.
    const half = (await pdfBytes("report.pdf")).subarray(0, 900);
    const out = await extractPdf(pdfjs, half.slice(), diskCMaps().load);
    assert.ok(out.text !== null || out.skipReason, JSON.stringify(out));
  });

  test("the lazy extractor loads pdf.js on its first PDF only", async (t) => {
    if (!needPdfjs(t)) return;
    let loads = 0;
    const extractor = lazyPdfExtractor({
      load: async () => {
        loads++;
        return pdfjs;
      },
      loadCMap: diskCMaps().load,
    });
    assert.equal(extractor.id, PDF_EXTRACTOR_ID);
    assert.equal(loads, 0);
    await extractor.extract(await pdfBytes("report.pdf"));
    await extractor.extract(await pdfBytes("cjk.pdf"));
    assert.equal(loads, 1);
    assert.ok(extractor.loadMs >= 0);
  });
});

// --------------------------------------------------------------------------
// A folder of notes and PDFs, indexed and searched in the browser path
// --------------------------------------------------------------------------

/** The fixture folder as <input webkitdirectory> files, with fixed dates. */
async function mixedFolder(overrides = {}) {
  const file = (bytes, lastModified = 1) => ({
    size: bytes.length,
    lastModified,
    arrayBuffer: async () => bytes.slice().buffer,
  });
  const text = (s) => new TextEncoder().encode(s);
  const items = [
    { path: "lib/notes.md", file: file(text("# Bread\n\nSourdough needs a starter fed every week.\n")) },
    { path: "lib/letter.docx", file: file(text("PK")) },
  ];
  for (const name of ["report.pdf", "field-guide.pdf", "locked.pdf", "scanned.pdf", "broken.pdf"]) {
    items.push({ path: `lib/papers/${name}`, file: file(await pdfBytes(name)) });
  }
  return fromFiles(items.map((it) => (overrides[it.path] ? { ...it, file: overrides[it.path](file) } : it)));
}

/** An extractor over the vendored pdf.js that counts the PDFs it read. */
function countingExtractor(id = PDF_EXTRACTOR_ID) {
  const inner = lazyPdfExtractor({ load: async () => pdfjs, loadCMap: diskCMaps().load });
  const read = [];
  return {
    id,
    read,
    async extract(bytes) {
      read.push(bytes.length);
      return inner.extract(bytes);
    },
  };
}

describe("indexing a folder with PDFs", () => {
  test("PDF chunks cite lines of the extracted text, with pages, and search finds them", async (t) => {
    if (!needPdfjs(t) || !needWasm(t)) return;
    const store = memoryStore();
    const pdf = countingExtractor();
    const { stats, files } = await syncFolder({ wasm, store, root: await mixedFolder(), id: "lib", embedder: null, pdf });

    assert.deepEqual(
      stats.skipped.map((s) => [s.path, s.reason]),
      [
        ["letter.docx", DOCX_SKIP_REASON],
        ["papers/broken.pdf", "unreadable PDF: InvalidPDFException: Invalid PDF structure."],
        ["papers/locked.pdf", "encrypted PDF (password required)"],
        ["papers/scanned.pdf", "PDF has no text layer (scanned or image-only?)"],
      ],
    );
    assert.equal(stats.files, 3);
    assert.equal(stats.pdfs, 5);
    assert.equal(stats.pdfPages, 4 + 5 + 2); // report, field guide, scanned
    assert.equal(pdf.read.length, 5);

    // The page starts are kept per file in meta.json, as the CLI's pages table.
    const meta = await readMeta(store, "lib");
    const pages = pagesByPath(meta);
    const expected = (name) => fixtures[name].expected;
    assert.deepEqual([...pages.keys()], ["papers/field-guide.pdf", "papers/report.pdf"]);
    assert.deepEqual(pages.get("papers/report.pdf"), pageStarts(expected("report.pdf")));
    assert.deepEqual(pages.get("papers/field-guide.pdf"), pageStarts(expected("field-guide.pdf")));
    for (const f of meta.files) assert.equal(f.mode, f.path.endsWith(".pdf") ? PDF_EXTRACTOR_ID : "md");

    // Each chunk is the cited lines of the extracted text.
    const index = new wasm.Index(files.manifest, files.docs, files.bm25, null);
    try {
      const guideLines = expected("field-guide.pdf").split("\n");
      let guideChunks = 0;
      for (let i = 0; i < index.length; i++) {
        const c = index.chunk(i);
        if (c.path !== "papers/field-guide.pdf") continue;
        guideChunks++;
        const cited = guideLines.slice(c.lineStart - 1, c.lineEnd).join("\n");
        assert.ok(cited.includes(c.text.trim().split("\n")[0].trim()), `${c.lineStart}-${c.lineEnd}`);
      }
      assert.ok(guideChunks >= 5, `${guideChunks} chunks`);

      // The page of each cited line, counted from the form feeds before it:
      // a second, simpler way to get what pageRange gives.
      const pageOfLine = (name, line) =>
        expected(name).split("\n").slice(0, line).join("\n").split(PAGE_BREAK).length;
      const top = (query) => {
        const [hit] = index.search(query, null, 3);
        const c = index.chunk(hit.id);
        const starts = pages.get(c.path);
        if (!starts) return { path: c.path, pages: null, text: c.text };
        const name = c.path.split("/").pop();
        const counted = formatPages(pageOfLine(name, c.lineStart), pageOfLine(name, c.lineEnd));
        const label = formatPages(...pageRange(starts, c.lineStart, c.lineEnd));
        assert.equal(label, counted, `${c.path}:${c.lineStart}-${c.lineEnd}`);
        return { path: c.path, pages: pageRange(starts, c.lineStart, c.lineEnd), text: c.text };
      };
      const horsetails = top("horsetails rhizomes sandpaper");
      assert.equal(horsetails.path, "papers/field-guide.pdf");
      assert.equal(horsetails.pages[1], 5);
      assert.match(horsetails.text, /Horsetails|rhizomes/);
      const lichen = top("lichen fungus alga");
      assert.equal(lichen.path, "papers/field-guide.pdf");
      assert.ok(lichen.pages[0] <= 3 && 3 <= lichen.pages[1], `${lichen.pages}`);
      assert.deepEqual(top("glaciers meltwater").pages, [1, 4]); // report.pdf is one chunk
      assert.deepEqual(top("weekly starter fed").path, "notes.md");
    } finally {
      index.free();
    }
  });

  test("an update re-reads no PDF unless it changed or pdf.js did", async (t) => {
    if (!needPdfjs(t) || !needWasm(t)) return;
    const store = memoryStore();
    await syncFolder({ wasm, store, root: await mixedFolder(), id: "lib", embedder: null, pdf: countingExtractor() });

    // Nothing changed: no PDF is read, and skipped ones stay skipped.
    let pdf = countingExtractor();
    let { stats } = await syncFolder({ wasm, store, root: await mixedFolder(), id: "lib", embedder: null, pdf });
    assert.equal(stats.upToDate, true);
    assert.deepEqual(pdf.read, []);
    assert.equal(stats.skipped.length, 4);

    // Touched but identical: hashed, not extracted again, pages kept.
    const reportBytes = await pdfBytes("report.pdf");
    const touched = { "lib/papers/report.pdf": (file) => file(reportBytes, 2) };
    pdf = countingExtractor();
    ({ stats } = await syncFolder({ wasm, store, root: await mixedFolder(touched), id: "lib", embedder: null, pdf }));
    assert.equal(stats.upToDate, true);
    assert.deepEqual(pdf.read, []);
    assert.deepEqual(pagesByPath(await readMeta(store, "lib")).get("papers/report.pdf"), [1, 3, 5, 6]);

    // A new pdf.js (another extractor id) re-extracts every PDF.
    pdf = countingExtractor("pdf1:pdfjs-next");
    ({ stats } = await syncFolder({ wasm, store, root: await mixedFolder(touched), id: "lib", embedder: null, pdf }));
    assert.equal(pdf.read.length, 5);
    assert.equal(stats.changed, 2);
    assert.equal(stats.upToDate, false);
  });

  test("PDFs skipped while they could not be read are read once they can", async (t) => {
    if (!needPdfjs(t) || !needWasm(t)) return;
    const store = memoryStore();
    let { stats } = await syncFolder({ wasm, store, root: await mixedFolder(), id: "lib", embedder: null, pdf: null });
    assert.equal(stats.files, 1);
    assert.equal(stats.skipped.filter((s) => s.reason === PDF_UNAVAILABLE_REASON).length, 5);

    // As an index made before PDFs were read here: skipped entries without a mode.
    const meta = await readMeta(store, "lib");
    for (const s of meta.skipped) delete s.mode;
    await store.write("indexes/lib/meta.json", new TextEncoder().encode(JSON.stringify(meta)));

    const pdf = countingExtractor();
    ({ stats } = await syncFolder({ wasm, store, root: await mixedFolder(), id: "lib", embedder: null, pdf }));
    assert.equal(pdf.read.length, 5);
    assert.equal(stats.added, 2);
    assert.equal(stats.files, 3);
  });
});
