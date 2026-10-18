# Changelog

All notable changes to notelight are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version may
change the CLI, the Python API or the index format; an index from an older
format is detected and rebuilt by `notelight index`.

## [Unreleased]

### Fixed

- **Browser app after a deploy.** GitHub Pages lets browsers reuse each file
  for 10 minutes (`Cache-Control: max-age=600`), so a returning visitor could
  get the new `index.html` with an old `app.js`. The page's service worker
  (`coi-sw.js`) now fetches same-origin files with `cache: "no-cache"`, so
  every load revalidates them by ETag: unchanged files are a 304 with no
  body, changed ones are fetched. Cross-origin requests (the model) are
  untouched.

### Changed

- **Hybrid search weighs each keyword hit by the query terms it contains.**
  Fusion used to give a BM25 hit on one incidental word as much say as a
  vector hit at the same rank, so queries that share no word with their
  answer lost the vector results (paraphrases on the notes benchmark: 0.082
  nDCG@10 against 0.257 for vectors alone). Each BM25 hit now adds
  `(matched / total)² / (60 + rank)`, where it contains `matched` of the
  query's `total` distinct terms. Held-out notes benchmark nDCG@10 0.715 ->
  0.759 (95% CI of the difference +0.012 to +0.083); BEIR hybrid SciFact
  0.712 -> 0.719, NFCorpus 0.350 -> 0.358, FiQA 0.351 -> 0.376. It costs
  0.02 to 0.06 ms per query. Keyword-only search is unchanged. It is in the
  Rust core, so the CLI, MCP server, web UI, browser app and npm package all
  get it. New `SearchOpts::coverage_power` (Rust; adding a field breaks
  struct literals that list every field), `coverage_power=` on
  `IndexReader.search` (Python) and `notelight eval --coverage-power` (0 is
  plain RRF); new `fuse::rrf_weighted`, `fuse::coverage_weight` and
  `Bm25::count_contained`. See `docs/results.md`, "Fusion: weighting keyword
  hits by coverage", for the candidates that lost.

- **The browser app shows what it is doing.** An activity bar pinned to the
  top of the window, with a spinner and a progress bar when the size is
  known, covers every slow step: loading the WebAssembly module, the indexer
  and an index, downloading bge-small (in MiB), checking its SHA-256,
  starting it, embedding a query that takes over 150 ms, indexing (each
  stage with counts) and the benchmark. Buttons that start another load or
  indexing run are disabled meanwhile. After an index is built or opened the
  page scrolls to the search box, focuses it and says "Search it below". The
  model now loads in the background: an index can be searched by keywords at
  once and switches to hybrid when the model is ready, instead of locking the
  page for the whole first download.

### Added

- **"No close match".** When nothing indexed is closely related to a query
  (the best cosine between the query and any chunk is below 0.60 with
  bge-small), the CLI, the localhost UI and the browser app say so above the
  results, which are still shown. `notelight search --json` and the MCP
  `search` tool return `close_match` (true, false, or null when unknown:
  keyword-only indexes, the hashing embedder) and `best_similarity`, and the
  MCP instructions tell agents not to answer from such results.
  `Searcher.query` returns both in Python (`SearchResults`), and the wasm
  `Index` gains `searchVector(vector, k)`. The floor was measured on the notes
  benchmark, off-topic queries and BEIR (`scripts/measure_match_floor.py`;
  design notes, "No close match").

- **PDFs in the browser app.** A folder indexed in the page now includes the
  text of its PDFs, extracted in the indexing worker with pdf.js 6.3.289
  (vendored by `scripts/vendor_pdfjs.py`: pinned, SHA-512 tarball and
  SHA-256 per file, Apache-2.0; no CDN). pdf.js is loaded only when a folder
  has a PDF to read: 1.75 MiB, 529 KiB gzipped, plus CMap files fetched only
  by PDFs with CJK fonts. Same contract as the CLI: pages joined with form
  feeds, citations are lines of the extracted text, and results show
  `p. 3` / `pp. 3-4` beside them, from page starts kept per file in the
  index's `meta.json`. Encrypted PDFs are tried with the empty password,
  then skipped like image-only and unreadable ones, with the CLI's reasons.
  The extractor id (`pdf1:pdfjs-6.3.289`) is part of the chunk cache key,
  so a new pdf.js re-extracts. pdf.js and pypdf extract text differently
  (mostly whitespace: 99.0 to 99.96% of non-blank lines agree on three real
  documents), so a PDF's chunks and line numbers in the browser are not the
  CLI's. Measured in Chrome 155 on one thread: 52 to 168 pages/s,
  0.6 to 2.0 MiB/s (pypdf natively on the same files: 12 to 21 pages/s).
  Tests: generated fixture PDFs (multi-page with a blank page, page-to-line
  mapping, empty-password and password encryption, Japanese through a
  predefined CMap, image-only, broken) and a Markdown + PDF folder indexed
  and searched, under Node in CI. DOCX is still skipped in the browser.

- **Notes benchmark query set 2** (`notelight eval notes2`): 80 new queries
  (24 keyword, 16 phrase, 20 question, 20 paraphrase) over the same pinned
  corpus, written and labelled by reading before any search with them, with
  their own tune and held-out halves. Set 1's halves had informed three
  decisions. `notelight eval notes` is still set 1; `load_notes` gains
  `query_set=`. First use: weighting vector hits down when BM25 looks right
  was tested on it and not shipped (held-out +0.002, interval 0.000 to
  +0.004; up to −0.034 on BEIR FiQA for the stronger variants). See
  `docs/results.md`, "Fusion: weighting vector hits (tested, not shipped)".

## [0.2.0] - 2026-10-16

The Python package gains the notes benchmark (`notelight eval notes`) and
richer eval output. The browser app (indexing, query embeddings, the Pages
demo at https://searope.github.io/notelight/) lives in this repository and
the demo; it is not part of the PyPI package.

### Added

- **npm package `notelight-wasm`**, published as 0.2.0.
  The core as WebAssembly for browsers, bundlers and Node 22+: `Index`
  (BM25 and hybrid search with a caller-supplied query vector), `chunkText`
  and `IndexWriter`, with TypeScript declarations. Same build as the browser
  demo (`wasm-bindgen --target web`), plus a Node entry that loads the
  `.wasm` from disk; no embedding model. Version single-sourced from
  Cargo.toml. `scripts/build_npm.py` builds it and checks `npm pack
  --dry-run`; `scripts/smoke_npm.py` installs the tarball and checks hybrid
  search against the CLI's top 10 on the sample index and type-checks the
  declarations, in CI and on 3 OSes x Node 22/24 in the new `npm` workflow,
  which publishes `npm-vX.Y.Z` tags with trusted publishing and provenance
  behind an owner-approved `npm` environment (0.2.0 itself was published
  by hand from the CI-built tarball, so it has no provenance).
- **Indexing in the browser.** The demo page indexes a folder you pick
  (File System Access API, or `<input webkitdirectory>` in Firefox and
  Safari) and keeps the index in the Origin Private File System (IndexedDB
  where OPFS is missing), so it is listed and reopened on the next visit
  without re-embedding; each index can be updated or deleted. Files never
  leave the browser; the only network request is still the verified model
  download. The walk follows `notelight index` (skip rules,
  `.notelightignore` with pathspec's gitignore semantics, UTF-8 then
  Windows-1252; PDF and DOCX are listed as skipped for now), and updates
  re-chunk only changed files and embed only text the embedding cache
  (keyed `sha256(model_id + "\n" + text)`, as in Python) does not have.
  Indexing runs in a Web Worker with progress and Cancel; a cancelled run
  keeps its finished batches. Keyword-only indexing needs no model. A Node
  test in CI indexes a fixture folder through the browser path and gets the
  CLI's chunks, byte-identical `docs.bin` and `bm25.bin`, the same document
  vectors (1 − cosine ≈ 2·10⁻¹³) and the same top 10. In Chrome on an
  i9-10900K, this repository's `docs/` (207 chunks) indexes in 41 ms
  without the model and in 27 s with bge-small on 8 threads (about 8
  chunks/s; the native CLI does 14); an update with nothing changed takes
  8 ms.
- `notelight-wasm` exports `chunkText` and `IndexWriter`, which builds a
  generation in memory with the core's writer.

- **Query embeddings in the browser demo.** `web-demo/` embeds the query with
  bge-small through onnxruntime-web (vendored at a pinned version, no CDN) and
  runs hybrid search against an index the CLI built with its default embedder.
  Same model contract as the Python embedder: pinned revision, SHA-256 checks
  on every load, CLS pooling, L2 normalization, query prefix, and the same
  `model_id`, so an index built with another embedder is refused. The model
  is downloaded once and kept in the browser's Cache API. A Node test in CI
  shows the vectors match `OnnxEmbedder` (1 − cosine ≈ 2·10⁻¹³) with the same
  top-10 hits; a WordPiece port matches Hugging Face `tokenizers` on every
  Unicode code point. About 10 ms per query in Chromium on 4 threads, 29 ms
  on 1.
- `scripts/serve_demo.py` serves the demo with correct `.mjs`/`.wasm` types
  (and, with `--isolated`, the headers WebAssembly threads need).
- `web-demo/coi-sw.js`: a service worker that adds those headers on hosts
  that cannot send them (GitHub Pages), so the browser embedder uses 4
  threads there too (about 10 ms per query instead of 29 ms). The first visit
  reloads once; `?threads=1` opts out.
- **Notes benchmark**: `notelight eval notes` scores every method on 82
  short, note-style queries (keywords, identifiers and phrases, questions,
  paraphrases) over this repository's docs, bundled with the package and
  labelled by line range, so `--max-chars` can be swept. `--split tune`
  (the default) is for choosing settings, `--split heldout` for reporting
  them. On the held-out half hybrid search scores 0.715 nDCG@10, BM25
  0.667 and vector search 0.639. A sweep of chunk size and phrase weight
  changed no default ([results](docs/results.md#notes-benchmark)).
- `notelight eval` reports MRR@10 next to nDCG@10 and recall@10, a
  per-style table for datasets whose queries have styles, and per-query
  scores in `--json`.

## [0.1.1] - 2026-10-16

A bug-fix release of the Python package. The WebAssembly crate below lives in
the repository; it is not published to PyPI or npm yet.

### Added

- **v2 spike: the core in the browser.** `notelight-wasm` (wasm-bindgen
  bindings: open an index from in-memory files, `searchBm25`, `search`,
  `chunk`), a static demo in `web-demo/` that searches an index entirely
  client-side, and a CI job that runs the core's unit tests as wasm32 and the
  JavaScript API under Node. Search latency in Chrome matches native; the
  module is 245 KiB (102 KiB gzipped).
- `IndexWriter::build` serializes a generation in memory (`GenerationFiles`);
  `commit` writes exactly those bytes.

### Changed

- `IndexReader::from_parts` checks file sizes against the manifest, as opening
  from disk always did.

### Fixed

- The CLI writes UTF-8 when its output is piped or redirected, on every OS.
  On Windows it used the ANSI code page, so `notelight search > hits.txt` (or
  `--json`, or a pipe from an MCP or script caller) crashed on a note with a
  character outside it, such as CJK text, and `--json` was not valid UTF-8.

## [0.1.0] - 2026-10-16

The first release on PyPI: `pip install notelight`. It covers milestones 0 to 5
of the [roadmap](README.md#roadmap).

### Added

- **Rust retrieval core** (`notelight-core`, no Python, ML or async
  dependencies):
  - Markdown-aware chunking (sections, paragraphs, code fences) with heading
    paths and exact line ranges for citations.
  - BM25 with Unicode-aware terms and camelCase splitting, served directly
    from the serialized index.
  - Quantized vector search: binary or ternary codes pick candidates with
    `popcount`, and int8 vectors re-score them. Ternary candidates with int8
    rescoring is the default; on BEIR it matches exhaustive int8 nDCG@10 to
    three decimals. The int8 dot product keeps 16 partial sums so the
    compiler can use SIMD.
  - Reciprocal rank fusion for hybrid (BM25 + vector) search.
  - A versioned, memory-mapped index format
    ([spec](docs/index-format.md), schema version 2): immutable generations
    with an atomic `CURRENT` swap, full validation on open, and recovery from
    interrupted commits, including on Windows.
  - Unit, integration and property tests against naive references, and
    criterion benchmarks.
- **Python package** (abi3 wheels for Python 3.10+, built with PyO3 and
  maturin):
  - `sync(folder, embedder)` and `Searcher(folder, embedder)`, with typed
    stubs for the Rust extension.
  - `OnnxEmbedder`: BAAI/bge-small-en-v1.5 on ONNX Runtime, pinned to one
    revision, verified by SHA-256 and downloaded once. `HashingEmbedder` for
    tests and model-free use.
  - An incremental indexer: only new or changed chunks are embedded, an
    unchanged folder never loads the model, and a lock file allows one writer
    per index. Indexes live in the user data directory (`NOTELIGHT_HOME`),
    never in the indexed folder.
  - Ingest of `.md`, `.markdown` and `.txt` in UTF-8 or Windows-1252, skipping
    hidden and build folders and anything in `.notelightignore`.
- **CLI**: `notelight index`, `search` (with `--json`), `status`, `mcp`,
  `serve`, `eval` and `version`; `--embedder bge-small|hashing|none` and
  `--offline`. On a terminal, `search` highlights the matched words and
  starts each snippet near the first match (`NO_COLOR` turns colour off).
- **Phrase ranking** in the core, for every interface: quoted phrases
  (`'"lock file" windows'`) rank chunks containing them first, and a chunk
  with all query terms in order gets a bonus worth one extra first place in
  fusion. Neutral on BEIR (within 0.002 nDCG@10); aimed at short note queries.
- **PDF and DOCX** as optional extras (`notelight[pdf]`, `[docx]`,
  `[documents]`; pypdf and python-docx). Citations stay `path:lines` of the
  extracted text, with PDF pages shown beside them; DOCX headings become
  heading paths. Without the extra, these files are skipped with a hint.
- **Localhost web UI** as an optional extra (`notelight[web]`,
  `notelight serve`): one page over the same service as the MCP server,
  bound to 127.0.0.1 with a Host-header check against DNS rebinding, no
  build step and nothing loaded from the network.
- **MCP server** (`notelight mcp`, official Python SDK, stdio) with four
  read-only tools: `search`, `read_context`, `list_sources` and
  `index_status`. `read_context` reads only files that are in the index.
- **Evaluation**: `notelight eval` reports recall@10, nDCG@10 and latency for
  every retrieval method on BEIR SciFact, NFCorpus and FiQA (datasets pinned
  by SHA-256), and checks the Rust searches against a NumPy implementation
  ([results](docs/results.md)).
- Docs: design notes, the index format spec, and a Rust walkthrough per
  milestone for readers coming from C# or Python.
- Prebuilt wheels for Linux x86_64 and aarch64, Windows x64 and macOS (Apple
  silicon and Intel), each installed and smoke-tested in CI before publishing.

### Fixed

Findings from an independent review of milestones 2 to 4, each with a
regression test:

- `notelight index <folder>` without `--embedder` keeps the embedder the
  index was built with, instead of switching a keyword-only or hashing index
  to bge-small (a 133 MB download and a full re-embed).
- Files the indexer skips (binary, undecodable, unreadable) are remembered, so
  `index_status` no longer reports them as new forever.
- An all-folder MCP `search` skips a folder whose index cannot be searched and
  lists it in a `skipped` field, instead of failing for every folder.
- MCP results from several folders are merged by per-folder rank, so
  keyword-only folders are no longer buried by raw fusion scores.
- Network errors while downloading the model raise `ModelNotAvailableError`
  with a hint instead of a traceback.
- A keyword-only sync no longer wipes the embedding cache.
- A query whose vector is all zeros (hashing embedder, punctuation-only
  query) falls back to BM25 instead of raising `ValueError`.
- `status` and the MCP server no longer crash when index files vanish during a
  concurrent sync.
- Changing the chunk size no longer empties the file cache before the new
  generation commits.

[Unreleased]: https://github.com/searope/notelight/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/searope/notelight/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/searope/notelight/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/searope/notelight/releases/tag/v0.1.0
