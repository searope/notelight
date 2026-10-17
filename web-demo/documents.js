// Text extraction for PDFs in the browser, the counterpart of
// python/notelight/documents.py. The extracted text is what gets chunked and
// cited, with the same layout as the CLI's:
//
// - each page's text ends in a newline, and pages are joined with a form feed
//   (`\f`), so every page after the first starts with `\f` on its first line;
//   pages never share a line, so a line maps to exactly one page
//   (`pageStarts`, `pageRange`);
// - NUL, form feed and carriage return are removed from the page text and a
//   vertical tab becomes a newline (`cleanText`, Python's `_clean`).
//
// The text itself comes from pdf.js, not pypdf, so it is not byte-identical
// to the CLI's: the two libraries decide differently where a line ends,
// where to put spaces between text runs and how to order them. Citations are
// lines of the text extracted here, and a browser index only ever shows its
// own text, so they stay consistent; they are not interchangeable with a
// CLI index of the same folder. The extractor id names pdf.js and its
// version and is part of the indexer's chunk cache key, so a new pdf.js
// re-extracts every PDF.
//
// pdf.js normally parses in a worker of its own. The indexing worker is
// already off the main thread, so it imports pdf.js's worker module too and
// pdf.js runs it in the same thread (its "fake worker"): no nested worker, no
// extra script URL, nothing that the page's COOP/COEP service worker has to
// allow. pdf.js is loaded only when a folder has a PDF to read
// (`lazyPdfExtractor`).
//
// DOCX is not read in the browser; such files are skipped with
// DOCX_SKIP_REASON.

/** Bump when the extracted text for the same file and pdf.js could change. */
export const EXTRACT_VERSION = 1;
/** Must equal scripts/vendor_pdfjs.py's VERSION; checked when pdf.js loads. */
export const PDFJS_VERSION = "6.3.289";
/** Identifies how a PDF becomes text, for cache keys (`documents.extractor_id`). */
export const PDF_EXTRACTOR_ID = `pdf${EXTRACT_VERSION}:pdfjs-${PDFJS_VERSION}`;
/** Separates PDF pages in extracted text; starts each page after the first. */
export const PAGE_BREAK = "\f";
export const DOCX_SKIP_REASON =
  "DOCX is not read in the browser yet; index the folder with the notelight CLI to include it";
export const PDFJS_DIR = "./vendor/pdfjs/";

const MARKDOWN = new Set([".md", ".markdown"]);

/**
 * How a file with `suffix` becomes chunks, as part of the chunk cache key
 * (`documents.content_mode`): `md`, `txt`, the PDF extractor's id, or null
 * for a file that is not read here (DOCX).
 */
export function contentMode(suffix) {
  if (suffix === ".pdf") return PDF_EXTRACTOR_ID;
  if (suffix === ".docx") return null;
  return MARKDOWN.has(suffix) ? "md" : "txt";
}

/** Python's `_clean`: characters that would shift line numbers or page breaks. */
export function cleanText(text) {
  return text.replace(/\r\n/g, "\n").replace(/[\0\f\r]/g, "").replace(/\v/g, "\n");
}

/** Join cleaned page texts as `_extract_pdf` does: each ends in "\n", joined by "\f". */
export function joinPages(pages) {
  return pages.map((p) => `${p.replace(/\n+$/, "")}\n`).join(PAGE_BREAK);
}

/**
 * The 1-based line on which each page of extracted PDF text starts
 * (`documents.page_starts`). Text without page breaks is one page.
 */
export function pageStarts(text) {
  const starts = [1];
  const lines = text.split("\n");
  for (let i = 1; i < lines.length; i++) if (lines[i].startsWith(PAGE_BREAK)) starts.push(i + 1);
  return starts;
}

/** `bisect.bisect_right`: how many of the sorted `values` are <= `x`. */
function bisectRight(values, x) {
  let lo = 0;
  let hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (x < values[mid]) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

/** The first and last page (1-based) that lines `lineStart..=lineEnd` touch. */
export function pageRange(starts, lineStart, lineEnd) {
  return [Math.max(bisectRight(starts, lineStart), 1), Math.max(bisectRight(starts, lineEnd), 1)];
}

/**
 * The line each page starts on, per indexed PDF, from a browser index's meta.json: what a
 * search hit needs for `pageRange`. Other files have no pages.
 * @returns {Map<string, number[]>}
 */
export function pagesByPath(meta) {
  return new Map((meta?.files ?? []).filter((f) => f.pages).map((f) => [f.path, f.pages]));
}

/** `p. 3` or `pp. 3-4` (`search.format_pages`). */
export function formatPages(first, last) {
  return first === last ? `p. ${first}` : `pp. ${first}-${last}`;
}

/** One page's text from pdf.js text content: runs in order, a newline at each end of line. */
export function pageText(items) {
  let text = "";
  for (const item of items) {
    text += item.str ?? "";
    if (item.hasEOL) text += "\n";
  }
  return cleanText(text);
}

/**
 * pdf.js asks its host for CMap files (CJK fonts) by name. Its own loader
 * resolves URLs against `document.baseURI`, which a worker does not have, so
 * the host passes this one: `load(name)` returns the bytes of
 * `vendor/pdfjs/cmaps/<name>`. Standard font data and image decoders are
 * not vendored (text extraction does not draw); pdf.js carries on without them.
 */
function binaryDataFactory(load) {
  return class {
    async fetch({ kind, filename }) {
      if (kind !== "cMapUrl" || !/^[\w.-]+$/.test(filename)) throw new Error(`no ${kind} data for ${filename}`);
      return load(filename);
    }
  };
}

/**
 * Extract the text of a PDF from its bytes with a loaded pdf.js
 * (`documents.extract`). Never throws for bad input: encrypted (unless the
 * empty user password opens it, which pdf.js tries first), image-only and
 * malformed files come back with `text: null` and a `skipReason`.
 *
 * `bytes` may be detached by pdf.js; hash it first.
 *
 * @param {object} pdfjs  the pdf.js module (`loadPdfjs`)
 * @param {Uint8Array} bytes
 * @param {(name: string) => Promise<Uint8Array>} loadCMap  bytes of a vendored CMap file
 * @returns {Promise<{text: string|null, skipReason?: string, pages?: number}>}  `pages`: how many were read
 */
export async function extractPdf(pdfjs, bytes, loadCMap) {
  const task = pdfjs.getDocument({
    data: bytes,
    // Text only: no fonts installed, no eval for glyph programs, no fetching
    // by pdf.js itself (it would need `document`), no XFA forms.
    BinaryDataFactory: binaryDataFactory(loadCMap),
    cMapUrl: "cmaps/",
    cMapPacked: true,
    useWorkerFetch: false,
    useSystemFonts: false,
    disableFontFace: true,
    isEvalSupported: false,
    isOffscreenCanvasSupported: false,
    isImageDecoderSupported: false,
    enableXfa: false,
    stopAtErrors: false,
    verbosity: 0, // errors only
  });
  task.onPassword = null; // a password prompt rejects with PasswordException
  try {
    const doc = await task.promise;
    const pages = [];
    for (let n = 1; n <= doc.numPages; n++) {
      const page = await doc.getPage(n);
      const { items } = await page.getTextContent();
      pages.push(pageText(items));
      page.cleanup();
    }
    if (!pages.some((p) => p.trim())) {
      return { text: null, skipReason: "PDF has no text layer (scanned or image-only?)", pages: pages.length };
    }
    return { text: joinPages(pages), pages: pages.length };
  } catch (e) {
    if (e?.name === "PasswordException") return { text: null, skipReason: "encrypted PDF (password required)" };
    return { text: null, skipReason: `unreadable PDF: ${e?.name ?? "Error"}: ${e?.message ?? e}` };
  } finally {
    await task.destroy();
  }
}

/**
 * Load the vendored pdf.js from `dir` (a URL folder): its worker module
 * first, so pdf.js finds `globalThis.pdfjsWorker` and parses in this thread.
 */
export async function loadPdfjs(dir = PDFJS_DIR) {
  let pdfjs;
  try {
    await import(new URL("pdf.worker.min.mjs", new URL(dir, import.meta.url)).href);
    pdfjs = await import(new URL("pdf.min.mjs", new URL(dir, import.meta.url)).href);
  } catch (e) {
    throw new Error(`pdf.js is not in web-demo/vendor/pdfjs/ (${e.message}); run: python scripts/vendor_pdfjs.py`);
  }
  if (pdfjs.version !== PDFJS_VERSION) {
    throw new Error(`vendor/pdfjs has pdf.js ${pdfjs.version}, expected ${PDFJS_VERSION}`);
  }
  return pdfjs;
}

/**
 * A PDF extractor for indexer.js that loads pdf.js on its first PDF, so a
 * folder without PDFs never downloads it.
 *
 * @param {object} o
 * @param {() => Promise<object>} [o.load]  loads the pdf.js module
 * @param {(name: string) => Promise<Uint8Array>} [o.loadCMap]
 * @returns {{id: string, loadMs: number, extract: (bytes: Uint8Array) => Promise<object>}}
 */
export function lazyPdfExtractor({ load = () => loadPdfjs(), loadCMap = fetchCMap } = {}) {
  let pdfjs = null;
  return {
    id: PDF_EXTRACTOR_ID,
    /** Milliseconds spent loading pdf.js (0 until a PDF needed it). */
    loadMs: 0,
    async extract(bytes) {
      if (!pdfjs) {
        const start = performance.now();
        pdfjs = await load();
        this.loadMs = performance.now() - start;
      }
      return extractPdf(pdfjs, bytes, loadCMap);
    },
  };
}

/** A vendored CMap file by name, fetched from the same origin. */
async function fetchCMap(name) {
  const response = await fetch(new URL(`cmaps/${name}`, new URL(PDFJS_DIR, import.meta.url)));
  if (!response.ok) throw new Error(`cmaps/${name}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}
