# notelight in the browser (v2 demo)

A static page that indexes a folder of your notes, or loads an existing
notelight index, into WebAssembly and searches it with the Rust core,
entirely client-side. When the index was built with the
default embedder, the query is embedded in the page too (bge-small through
onnxruntime-web) and the search is hybrid, as in the CLI. No framework,
bundler or CDN. Design and measurements: `docs/design-notes.md`, "v2 spike:
the core in the browser", "v2: query embeddings in the browser", "v2:
indexing in the browser" and "v2: PDFs in the browser".

Live: <https://searope.github.io/notelight/>, deployed from `main` by
`.github/workflows/pages.yml` (`scripts/build_site.py` assembles the site).

## Run it (PowerShell)

```powershell
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked   # the version in Cargo.lock
python scripts/build_wasm.py                                # writes web-demo/pkg/
python scripts/vendor_ort.py                                # writes web-demo/vendor/onnxruntime-web/ (pinned, verified)
python scripts/vendor_pdfjs.py                              # writes web-demo/vendor/pdfjs/ (pinned, verified)
python scripts/serve_demo.py                                # http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000>. **Index a folder…** picks a folder of notes
(the File System Access API in Chrome and Edge, `<input webkitdirectory>` in
Firefox and Safari) and indexes it in a Web Worker with the same rules as
`notelight index`: `.md`, `.markdown` and `.txt`, hidden and build folders
skipped, `.notelightignore` honoured, UTF-8 then Windows-1252. PDFs are read
with pdf.js in the worker (loaded only when the folder has one: 1.75 MiB,
529 KiB gzipped) and cited like the CLI's, by lines of the extracted text
with the pages beside them (`lines 541–560 (p. 18)`); encrypted, image-only
and broken PDFs are listed as skipped with the CLI's reasons. pdf.js does
not extract exactly the text pypdf does, so a PDF's line numbers differ
from a CLI index of the same folder. DOCX is listed as skipped. With **Semantic search** checked, chunks are embedded
with bge-small (progress and **Cancel** are shown; a cancelled run keeps the
batches it finished). Indexes are stored in the Origin Private File System
(IndexedDB where OPFS is missing) and listed on every visit with **Open**,
**Update** and **Delete**. **Update** re-reads the folder (without asking
again in Chrome, which remembers the folder handle), re-chunks only files
whose size, date or content changed, and embeds only text the embedding
cache does not have. The files never leave the browser.

Or click **Load the sample index**, or
**Open an index folder…** and pick one of your own: the folder that
`notelight status` prints as `index:`, or a `gen-NNNNNN` folder inside it. The
files are read by the page and never uploaded.

An index built with bge-small (the CLI's default) loads the query embedder:
the first time, the page downloads `onnx/model.onnx` (127 MiB) and
`tokenizer.json` from Hugging Face at the pinned revision, checks both
against their SHA-256, and keeps them in the browser's Cache API; later loads
read and re-verify the cached copy and need no network. **Forget the model**
deletes it. An index built with another embedder is refused with the CLI's
message and searched by keywords only; a keyword-only index never loads the
model.

The embedder runs on 4 threads (about 10 ms per query instead of 29 ms) when
the page is cross-origin isolated. On a host that cannot send the COOP and
COEP headers, such as GitHub Pages, the page registers `coi-sw.js`, a service
worker that adds them, and reloads once on the first visit; `?threads=1`
skips that. `python scripts/serve_demo.py --isolated` sends the headers
itself, for browsers without service workers. Use `serve_demo.py` rather than
`python -m http.server`, which on Windows often serves `.mjs` as
`text/plain`, so the browser refuses to load onnxruntime-web.

## The npm package

`pkg/` is also what the npm package `notelight-wasm` ships: the same
`wasm-bindgen --target web` output, plus a Node entry, its own README and a
package.json, assembled in `dist/npm/` by `python scripts/build_npm.py --pack`
(`python scripts/smoke_npm.py dist\npm\notelight-wasm-<version>.tgz`
installs and tests the tarball). The page imports `./pkg/notelight_wasm.js`
directly, so it needs no npm install or bundler. The package holds only the
core (`Index`, `chunkText`, `IndexWriter`); `embed.js`, `tokenizer.js`,
`ingest.js`, `indexer.js` and `store.js` stay in this folder. See
`crates/notelight-wasm/npm/README.md` and the design notes, "v2: the npm
package".

## Files

| File | What it is |
|---|---|
| `index.html`, `app.js`, `style.css` | the page |
| `documents.js` | PDF text with pdf.js: pages joined as in `documents.py`, page ranges, skip reasons, lazy loading |
| `embed.js` | the embedder (queries and documents): pinned model, SHA-256 checks, cache, CLS pooling |
| `ingest.js` | the folder walk: skip rules, `.notelightignore` (a port of pathspec's gitignore matching), decoding |
| `indexer.js` | incremental indexing: per-file cache, chunk reuse, embedding cache, one new generation per sync |
| `store.js` | storage: OPFS, with IndexedDB as the fallback; remembered folder handles |
| `index-worker.js` | the worker that indexes, stores, lists, loads and deletes indexes |
| `tokenizer.js` | bge-small's WordPiece tokenizer, ported from Hugging Face `tokenizers` |
| `coi-sw.js` | service worker that adds the COOP/COEP headers, so threads work on static hosts |
| `bench.js` | the latency benchmark behind the **Benchmark** button |
| `bench-node.mjs` | the same benchmark under Node |
| `bench-queries.json` | queries for the sample index |
| `sample-index/` | this repository's `docs/`, indexed with bge-small; rebuild with `python scripts/make_sample_index.py` |
| `tests/` | Node tests: tokenizer, embedder and indexer parity with Python, verification, incremental updates, PDF extraction and indexing (fixture PDFs from `scripts/make_pdf_fixtures.py`) |
| `THIRD-PARTY-NOTICES.md` | licences of what the page loads but this repository does not contain |
| `pkg/` | built by `scripts/build_wasm.py`, not committed |
| `vendor/` | onnxruntime-web and pdf.js, fetched by `scripts/vendor_ort.py` and `scripts/vendor_pdfjs.py`, not committed |

## Tests

```powershell
node --test "web-demo/tests/*.test.mjs"
```

They need `pkg/`, both `vendor/` folders and the bge-small files in the Python model cache
(any `notelight index` with the default embedder downloads them); without
them the parity tests are skipped, or set `$env:NOTELIGHT_MODEL_DOWNLOAD=1`
to let the tests download the model. In CI nothing is skipped. The fixtures
come from `scripts/make_sample_index.py` (query vectors and top-10 hits from
the Python embedder), `scripts/make_indexing_fixture.py` (a notes folder and
what `notelight index` made of it: chunks, vectors, index hashes, top 10,
and pathspec's verdicts on `.notelightignore` patterns),
`scripts/make_tokenizer_fixture.py` (token ids from `tokenizers`) and
`scripts/make_pdf_fixtures.py` (small PDFs with known text, and what pypdf
extracts from them); `python scripts/make_tokenizer_fixture.py --sweep` compares the
tokenizers on every Unicode code point.

## Measuring

```powershell
cargo run --release -p notelight-wasm --example latency -- web-demo\sample-index web-demo\bench-queries.json 200
node web-demo\bench-node.mjs web-demo\sample-index web-demo\bench-queries.json 200
```

In the page, **Benchmark** runs the same queries, and with the embedder loaded
it also times embedding plus hybrid search. Load times are in the status line
and in `window.embedderTimings`. For another index, serve a folder that
contains both the demo and the index, and pass
`?index=<url of a generation folder>&bench=<url of a queries file>&reps=5`.
A queries file is `{"queries": [{"text": "...", "vector": [...] | null}]}`;
vectors enable the hybrid text + vector row. `?model=<url of a folder>` loads
the model files from a mirror laid out like the Python model cache's revision
folder instead of from the Hub. `?index-threads=N` sets the embedder's
threads while indexing (default: up to 8). Every index run leaves its
numbers (`stats.ms` per stage, with `extract` and `pdfjs` for PDFs;
chunks, bytes, PDF pages, threads) in `window.indexStats`.
