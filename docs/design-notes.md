# Design notes

This document explains why notelight is built the way it is. It grows with each
milestone.

## Rust engine, Python interface

The core (chunking, BM25, quantized vector search, index format) is Rust. Everything
around it (embeddings, CLI, MCP server, evaluation, UI) is Python. This is the
pattern used by Polars, Ruff, pydantic-core and Hugging Face tokenizers.

**Why not pure Python?** At personal-folder scale, NumPy alone is fast enough for
brute-force search, so speed is not the main reason. The reasons are:

1. **Portability.** The same core compiles to WebAssembly for the planned browser
   version (v2). A NumPy implementation cannot follow it there.
2. **Memory-mapped indexes** that open instantly and are not copied into Python.
3. **Tight loops without interpreter overhead**: BM25 tokenization and scoring,
   int8 rescoring.

Milestone 3 benchmarks the Rust search against a pure-NumPy baseline. If Rust
does not win on some workload, `docs/results.md` says so.

## No ML in the core

Embeddings are computed in Python with ONNX Runtime. Keeping ML runtimes out of
`notelight-core` keeps it small, fast to compile, and straightforward to build
for `wasm32-unknown-unknown`. The core only ever sees `&[f32]` vectors.

## One wheel per platform (abi3)

The extension is built against CPython's stable ABI (`abi3-py310`), so one wheel
per platform works on every Python from 3.10 up. That cuts the release matrix
from about 20 wheels to 5.

## Binding layer stays thin

`notelight-py` only converts types, releases the GIL around heavy work
(`py.detach`), and maps errors to Python exceptions. Logic lives in the core so
it is tested once, in Rust, and reused by the WASM build.

## Binary code layout

Binary codes pack bits little-endian within each byte: dimension 0 is bit 0 of
byte 0. This matches `numpy.packbits(..., bitorder="little")`, so the Python
tests can check the Rust output against a NumPy reference.

## Milestone 1 decisions

### Exhaustive search, not an ANN index

Vector search scans every stored vector. A personal folder holds thousands to
low hundreds of thousands of chunks, and at 100k chunks the full ternary scan
plus int8 rescoring takes about 1.5 ms. An approximate index (HNSW, IVF) would
add memory, build time and tuning knobs, and it can silently miss results.
Exhaustive search is simpler and exact up to quantization.

### Two-stage vector search

1. **Ternary stage.** Each dimension is `-1`, `0` or `+1`: the sign bit plus a
   mask bit for dimensions above 0.7 × the vector's mean absolute value. The
   ternary dot product is two popcounts over `u64` words (agreements minus
   disagreements among dimensions significant in both): 96 bytes per 384-dim
   vector. Ignoring the near-zero dimensions makes the 200 candidates almost
   exactly the ones exhaustive int8 would rank first. (Milestone 1 used sign
   bits alone; see "Ternary is the default" below.)
2. **Int8 stage.** Only the best 200 candidates are re-scored with int8 codes.
   The query stays `f32` ("asymmetric" scoring), which is more accurate than
   quantizing both sides.

Candidate selection uses a histogram instead of a sort. Ternary distances
(`dim` minus the ternary dot product) are integers between 0 and `2 * dim`, so
counting them finds the cut-off in one pass with no comparisons.

Full-precision `f32` vectors are not stored. The int8 codes keep ranking
quality close to `f32` at a quarter of the size. Milestone 3 measures exactly
how close.

### Hybrid ranking with Reciprocal Rank Fusion

BM25 scores and cosine similarities are on different scales, and the right
weighting differs per corpus. RRF combines ranks instead of scores, needs no
tuning, and rewards chunks that both retrievers agree on.

### Structure-aware chunking

Chunks follow Markdown structure: a heading starts a section, paragraphs and
code fences stay whole when they fit, and nothing crosses a section boundary.
Each chunk keeps its heading path (`Setup > Windows`), which is indexed for
BM25 and shown in results, and its exact line range, so agents can cite
`path:line`. Property tests check, over thousands of generated documents, that
chunks are exact slices of the source, ordered, within the size limit, and
cover every non-whitespace character.

### Tokenizer: simple and predictable

BM25 tokens are lowercase alphanumeric runs. `camelCase` identifiers emit both
their parts and the whole word, so `getUserName` matches "user name" and
"getusername". There is no stemming: exact terms keep keyword search
predictable for names, error codes and identifiers, and the embedding side
handles paraphrase.

### Immutable generations instead of in-place updates

Each commit writes a complete new generation and atomically switches the
`CURRENT` pointer (see [index-format.md](index-format.md)). This:

- makes memory-mapping sound, because mapped files are never modified;
- works on Windows, where a mapped file cannot be replaced or deleted;
- gives crash safety: the old generation stays live until the new one is
  complete, and the next commit sweeps up anything a crash left behind.

Rebuilding everything sounds expensive, but the expensive part is computing
embeddings, not writing files. The Python layer (milestone 2) caches embeddings
by chunk content hash, so re-indexing after editing one file only embeds the
changed chunks, then rewrites files that take milliseconds to build.

### Reading binary files without `unsafe`

Readers decode integers with `from_le_bytes` at computed offsets instead of
reinterpreting byte slices as `&[u32]`. That avoids alignment requirements and
`unsafe` code, and it is endian-independent. The only `unsafe` block in the
crate is the memory-map call itself, with its safety argument documented next
to it. Each file is fully validated when it is opened, so query paths can index
without repeating checks.

## Milestone 2 decisions

### Full rebuild per sync, with caches in front

Every sync writes a complete new generation (see above) instead of patching
the index. The costly step, embedding, is cached, so a full rebuild only costs
the time to write files. The indexer keeps its caches in `cache.sqlite` next
to the index:

| Table | Key | Saves |
|---|---|---|
| `files` | relative path → size, mtime, content key | reading and hashing files that did not change |
| `chunks` | `sha256(file bytes)` + chunking mode (or document extractor) | re-chunking known content |
| `pages` | same key as `chunks` | the line each PDF page starts on (see Milestone 5) |
| `embeddings` | `sha256(model_id + "\n" + embedded text)` | re-embedding known chunks |
| `skipped` | relative path → size, mtime, reason | re-reading binary or undecodable files; reporting them as new in `index_status` |

The embedding key comes from the exact text sent to the model (heading path
plus chunk text) and the model id. A renamed, moved or copied file reuses its
vectors, a file edited in one section re-embeds only the chunks that changed,
and switching models can never return stale vectors. After a successful commit,
rows no current file refers to are deleted, so the cache cannot grow without
bound. A keyword-only sync (`--embedder none`) uses no vectors and leaves the
embedding cache alone, so switching back to the model re-embeds nothing.
Switching between two models keeps only the current one's vectors.

`files` and `skipped` describe the live generation, and readers use them
without the writer lock: staleness checks, `notelight status` and MCP
`read_context`. They are rewritten only after a commit, so a sync that fails
partway, even one that changed the chunk size, leaves them matching the index
that is still live.

A file whose size and mtime match is trusted without being read. When they
differ but the content hash matches (a `touch`, or a checkout that rewrites
files), nothing is re-chunked or re-embedded, and the stored stat is refreshed.
When nothing changed at all, no generation is written.

SQLite was chosen over a directory of `.npy` files because the cache needs
atomic batch updates, key lookups, and garbage collection with a `NOT IN`
query. All three are one statement each, from the standard library.

### One writer, enforced with a lock file

The core assumes a single writer: `commit` sweeps up leftovers from earlier
failed commits, which would be unsafe while another commit is running. The
Python layer enforces this with `filelock` on `<index dir>/lock`. The lock
uses OS file locks (`fcntl`/`msvcrt`) that are released when a process dies,
so a crash cannot leave a stale lock. A second `notelight index` waits up to 10
seconds, then fails with `IndexBusyError`. Readers never take the lock: they
open whichever generation is live and keep using it.

### Indexes live outside the indexed folder

Indexes are stored in the user data directory, keyed by a hash of the folder's
absolute path (`os.path.normcase` first, so `C:\Notes` and `c:\notes` match on
Windows). This keeps notelight out of synced folders (Dropbox, OneDrive, git),
and it lets you index read-only folders. `notelight status` lists every index
and its source folder. `NOTELIGHT_HOME` moves everything elsewhere.

### The embedder is part of the index's identity

The manifest records the embedder's id, for example
`BAAI/bge-small-en-v1.5@5c38ec7c405e`. Searching with a different embedder
raises `EmbedderMismatchError` instead of returning plausible-looking nonsense.
The CLI reads the id from the manifest and loads the matching embedder, so
`notelight search` needs no flags. BM25-only search (`--embedder none`) is
always allowed.

### Pinned model files

bge-small is downloaded from one fixed Hugging Face revision. Each file's
SHA-256 is checked before the file is moved into the cache, and a mismatch
deletes the download. Cached files are checked by size on load, so startup
does not hash 133 MB. `--offline` (and `NOTELIGHT_MODEL_CACHE` for a shared
cache) cover machines without network access. The model follows its
reference recipe: CLS pooling, L2 normalization, and the bge query instruction
prefix on queries only.

### A fake embedder for tests

The pipeline depends on an `Embedder` protocol, not on ONNX. Tests use
`HashingEmbedder`, a deterministic feature-hashing bag of words, which
needs no download, network or randomness. A test for the real model runs when
`NOTELIGHT_MODEL_TESTS=1` is set.

## Milestone 3 decisions

### What is measured, and against what

`notelight eval` scores seven methods on BEIR SciFact, NFCorpus and FiQA
with recall@10 and nDCG@10, using the same bge-small embedder and the same
index code as `notelight search`:

| Method | What it is |
|---|---|
| `f32` | exact cosine similarity over float32 vectors, in NumPy: the ceiling for the quantized methods |
| `int8` | int8 scores for every vector (Rust) |
| `binary` | Hamming distance alone (Rust) |
| `binary+int8` | 200 binary candidates re-scored with int8 |
| `ternary+int8` | 200 ternary candidates re-scored with int8: the default |
| `bm25` | keyword search alone |
| `hybrid` | BM25 and the default vector search fused with RRF: what `notelight search` returns |

Each BEIR document is indexed as one chunk, with its title as the heading
path, so the index, the BM25 text and the embedded text are built exactly as
for a note. Metrics follow the BEIR reference (`pytrec_eval`): linear gain
for nDCG, judged documents missing from the corpus still count as relevant,
and queries without a positive judgment are skipped.

### Ternary codes live in the core, but not on disk

A ternary code is the stored binary code (the sign) plus a mask of
"significant" dimensions, those above 0.7 × the vector's mean absolute value.
The mask is derived from the stored int8 codes the first time a reader runs a
ternary search, and cached in a `OnceLock` for the reader's lifetime. That
keeps it in Rust next to the binary stage (the browser build will want the
same loop) without changing the index format or bumping `SCHEMA_VERSION`.
Whether it should become the default is a question for the numbers; see
[results.md](results.md). It did, and the masks moved to disk; see the next
section.

### Ternary is the default, and its masks are stored (schema 2)

On all three BEIR datasets `ternary+int8` matches exhaustive int8 nDCG@10 to
three decimals, where `binary+int8` loses up to 0.010, for about 0.3 ms more
per query at 57k documents. So it became the default for vector search and
for the vector half of hybrid search.

Deriving the masks on first use would have kept the format, but every
`notelight search` opens a fresh reader, so each CLI query would pay one pass
over all int8 codes (117 ms at 57k chunks). The masks are now written next to
the sign bits instead: 48 more bytes per 384-dim vector (484 instead of 436,
about 11% more vector data), no warm-up, and the masks are memory-mapped like
everything else. The writer computes each mask from the int8 codes rather
than the floats, so a stored mask is exactly what a reader could recompute,
and the property tests check search against that recomputation. This changed
the `vectors.bin` layout, so `SCHEMA_VERSION` went to 2; old indexes are
rejected and rebuilt, which is always safe for a cache of your files.

`binary+int8` stays available as a `VectorMode` for the evaluation.

### Datasets are pinned like the model

The three zips are downloaded once from the BEIR authors' server, checked
against a pinned SHA-256 before they are unpacked, and unpacked under a
temporary name that is renamed into place, so a folder in the cache is
always complete. Members that would unpack outside the cache are refused.
`--offline` works once the cache is filled. Any folder in the BEIR layout can
be evaluated by passing its path, which is how the tests run without a
network.

### Embeddings are cached per dataset and embedder

Embedding FiQA's 57,638 documents takes several minutes on a laptop CPU, and
searching takes seconds, so corpus and query vectors are saved as `.npy`
files next to an index built from them, under
`<cache>/eval/runs/<dataset>-<content hash>/<embedder>/`. A second run only
searches. The cache key includes a hash of the dataset files and the
embedder id, so a changed dataset or model can never reuse stale vectors.

### A NumPy baseline, implemented the way a NumPy user would

`NumpyVectors` re-implements `int8`, `binary` and `binary+int8` in NumPy with
the same codes and tie-breaking, and the harness reports latency and top-10
agreement for each pair. It answers two questions: is the Rust correct
end to end (agreement), and is it worth having (latency)? Two differences are
deliberate. NumPy has no int8 × float32 matrix-vector kernel, so its int8
codes are kept dequantized as float32 (four times the memory). And NumPy's
matrix product may use several threads, while the Rust search is one thread
per query.

## Milestone 4 decisions

### The MCP server only reads

`notelight mcp` serves indexes; it never builds them. Indexing a large folder
takes minutes and downloads a model the first time, which does not fit inside
an agent's tool call, and a server that can write would need the writer lock
and could block `notelight index`. So the server never takes the lock, and
an agent learns that an index is out of date from `index_status` (which
compares file sizes and mtimes with the indexer's cache) and tells the user to
run `notelight index`.

### One searcher per folder, reopened when the generation changes

The server keeps a `Searcher` per folder and an embedder per model id for its
whole life, so bge-small loads once, not per query. Before each use it
compares the reader's generation with `CURRENT` (one small file read); after a
re-index it opens the new generation and drops the old reader, which on
Windows also lets the next commit delete the old generation's mapped files.
Tools run in worker threads; the Rust reader is shared between them (it is a
`frozen`, `Sync` pyclass and releases the GIL while searching), and the
embedder is behind a lock because ONNX Runtime is loaded lazily.

### `read_context` reads only indexed files

The agent passes back a citation (`path:start-end`) and gets those lines plus
some context, read from disk. The path is accepted only if it is in the
index's file list (the `files` table of the indexer's cache), compared after
normalizing separators. It is never resolved on disk first. That one check
refuses `..`, absolute paths outside the folder, and hidden or
ignored files such as `.env`, because none of them was indexed. The agent can
read nothing that `search` could not already have shown it. Output is capped
at 400 lines per call.

### Searching several folders merges by rank

With no `folder` argument, `search` queries every served folder and merges
the hits by their rank within their own folder, then by score among hits of
equal rank. Fused scores alone are not comparable across indexes: a
keyword-only index contributes one rank list per hit and a hybrid index two,
so a keyword-only folder's best hit (at most 1/61) could rank below a
mediocre hybrid hit (up to 2/61) and be cut off. Merging by rank puts every
folder's best hits near the top. A folder whose index cannot be searched
(corrupt, unknown embedder, model unavailable offline) is listed in
`skipped` instead of failing the whole call. Passing `folder` searches one
folder and reports its error directly. `--folder` (repeatable) limits what
the server exposes at all.

### mcp 2.x, structured output

The server uses the official SDK's `MCPServer` (mcp 2.x, the successor of
`FastMCP`). Each tool returns a pydantic model, so clients get an output JSON
schema and `structuredContent` alongside the text. Anticipated failures (not
indexed, embedder mismatch, refused path) raise `ToolError`, which reaches
the model as an `is_error` result with a message it can act on, rather than a
protocol error. Tests drive the server through the SDK's in-process `Client`,
plus one test that launches `notelight mcp` as a subprocess over stdio to
prove nothing but protocol messages is written to stdout.

## Milestone 5 decisions: PDF and DOCX

### Optional extras, pure-Python readers

PDF and DOCX support ships as extras (`notelight[pdf]`, `[docx]`,
`[documents]`), so the base install keeps its small dependency list. The
readers are pypdf (BSD-3-Clause) and python-docx (MIT). Both are pure Python
and installed from wheels on every OS, so they add no build step and no
native library to the Windows, macOS and Linux matrix. The alternatives
were rejected: PyMuPDF is faster and better at layout, but it is AGPL, which
does not fit an MIT/Apache project; pdfminer.six is MIT and accurate, but
slower and much larger than pypdf for plain text; docx2txt drops the styles
needed for headings. Extraction quality is pypdf's: reading order follows the
content stream, and there is no OCR.

Without the extra, `walk` still finds `.pdf` and `.docx` files and the indexer
records them in `skipped` with the reason `install notelight[pdf]` (or
`[docx]`). That reason is special in two places: the indexer reads such a file
again once the extra is importable, even though its size and mtime did not
change, and `index_status` counts it as added. Other skip reasons stick until
the file changes. Encrypted PDFs are tried with the empty user password
first (many are encrypted only to restrict printing); otherwise they, image-only
PDFs (no text on any page) and anything pypdf or python-docx fails to parse are
skipped with a reason. Extraction catches every exception from the readers:
a malformed file must never stop a sync. Documents may be up to 50 MB
(`MAX_DOCUMENT_BYTES`); text files keep the 10 MB limit.

### Citations: lines of the extracted text, pages beside them

The core's `ChunkRecord` (path, heading path, line range, text) and the
on-disk format are unchanged. A PDF or DOCX is chunked as the text extracted
from it, and its citation is `path:line_start-line_end` in that text, like
any other file. `read_context` extracts the file again the same way, so the
cited lines are the ones it returns. The cache key of a document's chunks
includes the extractor id (an extraction version plus the library and its
version, for example `pdf1:pypdf-6.20.1`), so upgrading pypdf re-extracts
and re-chunks rather than mixing old chunks with new line numbers. A
`read_context` served while the installed library differs from the one used
at index time could still shift lines; `index_status` does not detect that,
and `notelight index` fixes it.

PDF pages are joined with a form feed: every page's text ends in a newline,
and every page after the first starts with `\f`. A page therefore always
starts on a new line, and a line belongs to exactly one page. The line on
which each page starts is stored per document in a `pages` table of
`cache.sqlite` (next to `chunks`, keyed and pruned the same way). A search
hit's pages come from that table and its line range (`page_start`,
`page_end`, shown as `p. 3` or `pp. 3-4`); `read_context` computes them from
the text it extracted. Chunks may cross a page boundary: chunking per page
would leave short pages as tiny chunks, and a page range is still a precise
citation. Keeping lines as the primary citation, rather than switching PDFs
to `report.pdf:p3`, means one parser and one `read_context` for every file
type, and lines stay finer-grained than pages.

Putting the page into the core's `ChunkRecord` was the alternative. It would
be self-contained in the index, but it is a schema change (a new field in
`chunks.bin` and `SCHEMA_VERSION` 3) for something only one file type has,
and the page table is derived data that the indexer's cache already holds.
The cost of the cache is that a searcher reading an old generation sees the
newest page table, so for one query after a PDF changes its pages can be off;
the line citation is not affected.

DOCX has no stable lines or pages (pagination depends on the renderer).
Body paragraphs and tables are written in document order, one paragraph per
line and a blank line after each, which is the paragraph structure the
chunker already understands. Paragraphs styled `Heading 1` to `Heading 6`
become `#` headings and the text is chunked as Markdown, so they become
heading paths exactly as in a `.md` note. A body line that Markdown would read
as a heading or a fence (`# ...`, ```` ``` ````) is escaped with a backslash.
Table rows become `cell | cell` lines; headers, footers, comments and text
boxes are left out.

## Web UI decisions

### A thin layer over the MCP service

`notelight serve` (the `web` extra: FastAPI and uvicorn) puts four GET
endpoints over the same `NotesService` that backs `notelight mcp`:
`/api/search`, `/api/context`, `/api/sources` and `/api/status`. Searcher
caching, rank-merging across folders, per-folder skips and the
indexed-files check for reading are therefore shared, not reimplemented, and
the browser can see exactly what an agent can. A `ToolError` becomes a 400
with the same message. Like the MCP server it never takes the writer lock.
FastAPI runs the plain `def` endpoints in a thread pool, which the service
already allows (see "One searcher per folder").

### One page, no build step

The page is `index.html`, `app.js` and `style.css` in package data, served
from three fixed routes. There is no static-directory mount, so no URL maps
to a path on disk. There is no framework, bundler or CDN: nothing is fetched
from the network, and the page keeps working offline. All server text is
inserted with `textContent`, and query terms are highlighted by building
`<mark>` nodes rather than HTML strings, so indexed content cannot inject
markup. A strict Content-Security-Policy (`script-src 'self'`, no inline
script) backs that up.

### Localhost only, and checking `Host`

uvicorn binds to 127.0.0.1, and there is no option to change that: exposing
private notes to a network would be a different product. Binding alone does
not stop DNS rebinding. A page on `evil.example` can re-resolve its own name
to 127.0.0.1 and then make same-origin requests to the server. Those requests
still carry `Host: evil.example`, so a middleware refuses any request whose
`Host` is not `127.0.0.1`, `localhost` or `[::1]` (with an optional port).
The API is GET-only and sends no CORS headers, so other origins cannot read
its responses.

## Phrase search decisions

### Quoted phrases first, and a bonus for terms in order

BM25 scores a query as a bag of words, and the embeddings rank by meaning, so
neither notices that a chunk contains the query's words side by side. A chunk
that repeats "lock" and "file" in separate sentences could outrank the one
that says "lock file". Two rules now run after fusion, in the core
(`phrase.rs`, applied in `IndexReader::search`):

- **Quoted phrases are a hard tier.** `"lock file" windows` puts every chunk
  containing all quoted phrases above every chunk that does not, keeping the
  fused order within each tier. Straight and curly quotes both count, and an
  unclosed quote runs to the end of the query, because phones insert curly
  quotes and people forget the closing one.
- **Terms in order earn a bonus, quoted or not.** If a query has two or more
  terms and a chunk contains all of them consecutively and in order, its
  fused score gains `phrase_boost / (rrf_k + 1)`: as much as one more first
  place in RRF. The default weight is 1.

Phrases are compared as sequences of BM25 terms (the same tokenizer, over
the heading path and the text, as indexed), so case, punctuation and
stopwords do not break a match: `"lock-file"` matches "Lock file", and
`"lock the file"` matches "lock a file". Only fused candidates (at most 50
per retriever) are re-ranked. A phrase match that neither retriever found
stays missing, but every chunk containing the phrase also contains its terms,
so BM25 nearly always finds it.

### Cheap enough to always run

Re-reading and re-tokenizing every candidate made hybrid search 40x slower
on BEIR (about 7 ms per query instead of 0.2 ms). Most candidates cannot
match, because they lack at least one query term, and the BM25 postings
already know that: each term's postings are sorted by document id, so
`Bm25::contains_all` is one binary search per term. Only chunks that contain
every term are read and tokenized. That brings the cost back to noise on
BEIR, and to 0.03 to 0.3 ms per query on this repository's docs, against
about 5 ms to embed the query. The sort order was always part of the format;
opening an index now also checks it.

### Measured, and neutral on BEIR

`notelight eval --phrase-boost` sweeps the weight. On SciFact, NFCorpus and
FiQA, nDCG@10 moves by at most 0.002 at any weight from 0.5 to 4, because
their queries are long questions whose terms almost never appear in order.
The bonus is aimed at the short, phrase-like queries people type into notes,
which BEIR does not measure, so the rule is: it may not hurt BEIR. Quotes
change BEIR slightly even at weight 0, because 33 of FiQA's 648 test queries
contain quotes (`Why is “cheque cashing” a legitimate business?`), and those
now act as phrases. See `docs/results.md`.


## v2 spike: the core in the browser

The spike answers one question before the browser app is built: does
`notelight-core` run in WebAssembly, unchanged, at native speed? It does. The
pieces are `crates/notelight-wasm` (the bindings), `web-demo/` (a static page
that loads an index and searches it client-side), and a `wasm` CI job.

### The core needed no changes to compile

`cargo build -p notelight-core --target wasm32-unknown-unknown` worked
before the spike started: the M1 rule of no Python, ML or async dependencies,
and memmap2 behind `cfg(not(target_arch = "wasm32"))`, had already done the
work. Two small additions make the browser path first-class rather than
incidental:

- `IndexWriter::build(generation)` returns a generation's files as bytes
  (`GenerationFiles`); `commit` now calls it and writes them. A browser has no
  filesystem, so the future app will build indexes in memory and keep the
  bytes in IndexedDB or OPFS. A test checks that `build` produces exactly the
  bytes `commit` writes.
- `IndexReader::from_parts` checks each file's size against the manifest, as
  `open_generation` always did. Bytes from a directory picker or `fetch` are
  as untrusted as bytes on disk.

The std filesystem calls still compile on `wasm32-unknown-unknown` (they
return errors at run time), so no `cfg` was added to the core: the reader
and writer that touch disk are simply not called from the browser.

### Tests run as WebAssembly, two ways

- **The JS API, under Node.** `crates/notelight-wasm/tests/node.rs` uses
  `wasm-bindgen-test`; `.cargo/config.toml` makes `wasm-bindgen-test-runner`
  the runner for `wasm32-unknown-unknown`, so `cargo test --target
  wasm32-unknown-unknown -p notelight-wasm --test node` runs the real module in
  Node against the committed sample index. It checks what JavaScript sees:
  plain arrays and objects, `camelCase` fields, `null` (not `undefined`) for a
  missing rank, and exceptions for errors.
- **The core's unit tests, as wasm32.** `wasm-bindgen-test-runner` only runs
  `#[wasm_bindgen_test]` functions, and annotating the core's 60 `#[test]`s
  for it would put browser concerns into the core. Instead they run on
  `wasm32-wasip1` under Node's built-in WASI (`scripts/wasi_run.mjs` is the
  runner). That target shares what matters with the browser one: a 32-bit
  `usize`, which is where layout math and `u64` offset conversions could
  break. Proptest, tempfile and criterion are native-only dev-dependencies, so
  the integration and property tests stay native.

### `cargo build` plus `wasm-bindgen`, not wasm-pack

wasm-pack wraps the same two steps and adds npm packaging this project does not
need yet; it also downloads tools at build time, and its maintenance has been
uncertain since the rustwasm GitHub organization was wound down in 2025.
Running them directly from `scripts/build_wasm.py` keeps the build visible: compile to
`.wasm`, then `wasm-bindgen --target web` writes the ES module glue and
TypeScript types. The one sharp edge is that the CLI must match the
`wasm-bindgen` crate version exactly; the script reads the version from
`Cargo.lock` and refuses a mismatch with the install command, and CI installs
that same version. `--target web` produces a plain ES module, so the demo
loads it with `<script type="module">` and no bundler.

### Size

| Build (`wasm-release` profile) | .wasm | gzipped |
|---|---|---|
| opt-level 3, thin LTO | 298 KiB | 110 KiB |
| opt-level 3, fat LTO, stripped (**chosen**) | 245 KiB | 102 KiB |
| opt-level "s", fat LTO, stripped | 216 KiB | 84 KiB |
| opt-level "z", fat LTO, stripped | 203 KiB | 80 KiB |
| opt-level 3 with `+simd128` | 245 KiB | 102 KiB |

"s" was 25% slower on vector search in Node, for 18 KiB gzipped. Enabling
`simd128` changed nothing measurable: the hot loops (`int8_dot`, popcount over
`u64` words) are not auto-vectorized into WASM SIMD, so it would take explicit
`core::arch::wasm32` intrinsics, which is the same open question as the M3
"vectorized `int8_dot`" follow-up. `wasm-opt` (Binaryen) is not used yet.

### Latency: native and WebAssembly are within noise

Measured on an i9-10900K (Windows 11), k = 10, the same index files and
queries everywhere, through the same `Index` type: natively by
`cargo run --release -p notelight-wasm --example latency`, in Node by
`web-demo/bench-node.mjs`, and in Chrome 152 by the demo's Benchmark button
(`web-demo/bench.js`). Each query runs 5 to 200 times in a row and its time is
the mean (browsers coarsen `performance.now()` to 0.1 ms); the table gives the
median and p95 over queries, best of three runs, all measured in one sitting
because background load moved single runs by up to 2x.

**FiQA** (57,638 chunks, 104 MiB with 384-d bge-small vectors; 200 test
queries with their cached embeddings):

| | native | wasm, Node 25 | wasm, Chrome 152 |
|---|---|---|---|
| open from bytes in memory | 5.5 ms | 33 ms | 70 ms |
| BM25 | 236 / 591 µs | 285 / 638 µs | 280 / 620 µs |
| hybrid, text only | 343 / 1032 µs | 349 / 873 µs | 360 / 900 µs |
| hybrid, text + vector | 1458 / 2348 µs | 1344 / 2156 µs | 1440 / 2400 µs |

**This repository's docs** (the committed sample, 132 chunks; 20 queries in
`web-demo/bench-queries.json`):

| | native | wasm, Node 25 | wasm, Chrome 152 |
|---|---|---|---|
| BM25 | 2.8 / 4.0 µs | 9.6 / 48.6 µs | 3.5 / 4.5 µs |
| hybrid, text only | 32 / 94 µs | 48 / 177 µs | 26 / 81 µs |

Search is as fast in the browser as natively, including vector search
without SIMD. On a tiny index each call pays a few microseconds to cross
into WebAssembly and build the result objects, which is invisible next to
the ~5 ms it takes to embed a query. Opening is slower because the browser
copies the files into WebAssembly memory (the native reader here also owns
its bytes; with mmap it would not copy at all); 70 ms for 104 MiB is paid
once per page load.

### What the browser app needs next

- **Query embeddings in the browser.** bge-small through onnxruntime-web
  (WASM or WebGPU backend) or transformers.js, with the same pinned revision
  and SHA-256 checks as `embed.py`, and a test that its vectors match the
  Python embedder's within int8 quantization error. `model_id` must match, or
  the index refuses the query as it does today.
- **Indexing in the browser.** `chunk_text`, `IndexWriter::build` and the
  embedder are enough to build a generation from files the user picks
  (File System Access API where available, `<input webkitdirectory>`
  elsewhere). PDF and DOCX need JS-side text extraction (pdf.js) or are left
  out at first.
- **Storage.** Keep generations in OPFS (or IndexedDB), with the same
  immutable-generation idea: write a new one, then switch a pointer. An
  incremental rebuild needs the per-file cache the Python indexer keeps in
  SQLite, as a small JSON or IndexedDB table.
- **A worker.** Opening 100 MiB and embedding block the main thread; run the
  core and the model in a Web Worker and pass results by `postMessage`.
- **Memory.** wasm32 linear memory is limited to 4 GiB and only grows. Large
  indexes may want docs.bin kept outside WebAssembly memory (in JS, fetched per
  hit), which needs a `Docs` that reads through a callback.
- **The TypeScript app itself**, with the types `wasm-bindgen` generates
  (`notelight_wasm.d.ts`), and a deploy (GitHub Pages) built in CI by
  `scripts/build_wasm.py`.
