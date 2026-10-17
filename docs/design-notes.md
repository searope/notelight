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
now act as phrases. See `docs/results.md`. The notes benchmark (next section)
later measured the short queries: there the bonus is worth 0.02 to 0.03
nDCG@10. It lifts phrase and identifier queries and is mixed on keywords.

## Notes benchmark decisions

### A small benchmark of our own, labelled by line range

BEIR cannot say whether the defaults suit notes: its queries are long
questions, and its documents are already split. `notelight eval notes` adds
82 short queries (keywords, identifiers and phrases, questions, paraphrases)
over this repository's own Markdown, pinned to one commit. It ships inside
the package (61 KB), so it runs offline, needs no checkout, and the tests can
check its rules in CI. The corpus is the project's docs because they could
be published and nobody's private notes were available; the cost is a
technical, repetitive corpus that is not a personal notebook.

Labels are `[path, line_start, line_end, grade]`, not chunk ids, so the same
labels work at any `max_chars`. Scoring has to handle chunks that overlap
several labelled passages, and passages split across chunks. Each ranked chunk
is replaced by the labelled passages it shows for the first time, best grade
first, and a chunk with none stays one non-relevant position
(`Dataset.judged_ranking`). The standard nDCG, recall and MRR then apply
unchanged. A passage counts once, so a big chunk cannot score above 1 by
covering three passages, and a passage is found as soon as any part of it
is shown, which is how a person reads a hit.

### Honest by construction

The queries and labels were written from the documents before any search
ran, committed before the first one, and never edited after results were
seen. Each query records the section it came from. Paraphrase queries must
share no BM25 term with their answers, heading paths included, and a test
enforces it (four were reworded before the commit to pass it). The split into tune and held-out halves alternates within each
style and was fixed in the same commit. The decision rule was written down
before the held-out run: change a default only if the held-out paired 95%
bootstrap interval excludes zero and agrees with tune. The author was an LLM
(Claude), which `docs/results.md` says next to the numbers.

### The defaults held

On the tune half, 600-character chunks with phrase weight 4 looked best
(+0.022 hybrid nDCG@10 over 1200 and 1, interval −0.017 to +0.064). On the
held-out half they scored 0.024 below the defaults (0.690 vs 0.715; 18
queries worse, 11 better), and weight 4 matched weight 1 on every held-out
query. So `max_chars` stays 1200 and `phrase_boost` stays 1. Two findings
did hold on both halves: hybrid beats vector search alone (+0.076 held-out,
interval 0.025 to 0.126), and turning the phrase bonus off loses 0.018 to
0.032, on phrase and identifier queries (keyword queries are mixed). Fusion's weak spot is
paraphrase, where BM25 contributes nothing and halves what the vectors find
(5 queries per half, too few to act on). A tuned RRF weight or a
query-style switch would be the thing to test if more paraphrase queries
show the same.


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

- **Query embeddings in the browser.** Done: see "v2: query embeddings in
  the browser" below.
- **Indexing in the browser.** Done, except PDF and DOCX: see "v2:
  indexing in the browser" below.
- **Storage.** Done: generations in OPFS (or IndexedDB), switched by a
  pointer file, with the per-file and embedding caches beside them.
- **A worker.** Done for indexing, embedding documents and storage.
  Opening a large index and embedding queries still run on the main
  thread.
- **Memory.** wasm32 linear memory is limited to 4 GiB and only grows. Large
  indexes may want docs.bin kept outside WebAssembly memory (in JS, fetched per
  hit), which needs a `Docs` that reads through a callback.
- **The app itself.** The page stays plain JavaScript with no build step
  beyond `wasm-bindgen`; the generated `notelight_wasm.d.ts` documents the
  API for an npm package. The GitHub Pages deploy is done.

## v2: query embeddings in the browser

The demo now embeds the query in the page with bge-small and runs the same
hybrid search as the CLI (BM25 + vectors, RRF, phrase ranking) against an
index the Python CLI built with its default embedder. After the model's
one-time download nothing leaves the page. The pieces are
`web-demo/tokenizer.js` (a WordPiece port), `web-demo/embed.js` (the model
contract), onnxruntime-web vendored by `scripts/vendor_ort.py`, and Node tests
in `web-demo/tests/` that run in the `wasm` CI job.

### onnxruntime-web, not transformers.js

Both run the same ONNX file on the same engine: transformers.js 4.3.1 depends
on onnxruntime-web (a 1.31 dev build) and adds its own JavaScript, so it
cannot be smaller. What decided it:

| | onnxruntime-web 1.30.0 | transformers.js 4.3.1 |
|---|---|---|
| Licence | MIT | Apache-2.0 (plus onnxruntime-web, MIT) |
| What the page loads | 50 KiB module + 24 KiB glue + 13.6 MiB `.wasm` (3.5 MiB gzipped) | the same runtime (its WebGPU build, 27 MiB `.wasm`) + a 444 KiB library |
| Pin and verify the model files | we fetch the bytes, check SHA-256, hand them to `InferenceSession.create` | it fetches by repository and revision into its own cache; no hook to check a hash before use |
| Tokenizer | ours, tested against `tokenizers` on every code point (below) | its own JavaScript port, which would need the same test |
| Pooling, prefix, normalization | ours, a few lines that mirror `embed.py` | configurable, but one more layer to match |

So the runtime is a dependency, and everything with a parity requirement
(fetching, verifying, tokenizing, pooling) is code in this repository that the
tests pin down. WebGPU is not used: for a 20-token query on a 33M-parameter
model the WASM backend is fast enough (below), and it is one fewer code path.

Nothing loads from a CDN. `scripts/vendor_ort.py` downloads the npm tarball
of one pinned release, checks it against the SHA-512 integrity npm publishes,
and extracts the three files the WASM backend needs, each checked against a
pinned SHA-256, into `web-demo/vendor/`. That folder is git-ignored like
`pkg/`, since the `.wasm` is 14 MB; the licence is committed in
`web-demo/THIRD-PARTY-NOTICES.md`.

### The same model contract as `embed.py`

- The spec in `embed.js` copies `BGE_SMALL_EN_V15`: repository, revision
  `5c38ec7c405e...`, and each file's size and SHA-256. `modelId` is built the
  same way, so it is `BAAI/bge-small-en-v1.5@5c38ec7c405e`, the id the index
  manifest records; a test checks it against the fixture Python wrote.
- `onnx/model.onnx` and `tokenizer.json` are checked (size, then SHA-256 with
  Web Crypto) before either is used, on every load and not only after a
  download: a Cache API entry is as untrusted as any other bytes. A cached
  copy that fails is deleted and downloaded again; a download that fails is
  never cached. Verifying 127 MiB costs about 300 ms, almost half of a cached load,
  and is worth it.
- The query gets the bge prefix and runs through the model, and the CLS
  vector is L2-normalized, as `OnnxEmbedder.embed_query` does.
- An index whose manifest names another embedder is refused with the message
  `Searcher` gives ("the index for X was built with hashing-256, but the
  query embedder is ..."), and the page falls back to keyword search. A
  keyword-only index never loads the model.

### The tokenizer is a port, checked exhaustively

bge-small's `tokenizer.json` is plain BERT: `BertNormalizer` (drop control
characters, map whitespace, space out CJK ideographs, strip accents,
lowercase), `BertPreTokenizer`, `WordPiece`, `[CLS] ... [SEP]`, and five
special tokens matched in the raw text. That is under 250 lines of JavaScript. The
constructor refuses a `tokenizer.json` that uses anything else, rather than
tokenizing it differently.

"Close enough" is not the bar: one different token id changes the vector.
`scripts/make_tokenizer_fixture.py --sweep` tokenizes every Unicode code
point in context, plus every line and paragraph of this repository's
Markdown, with both libraries: 1,112,064 code points and about 3,800 lines
and paragraphs, all identical. Getting there found three things that reading
the Rust source would not have:

- `tokenizers` treats unassigned code points as ordinary characters, not as
  control characters (JavaScript's `\p{C}` includes them).
- Its Unicode category tables are older than V8's. 654 code points assigned
  since then are plain characters to it, and four changed category (U+166D
  and U+111C9 are still punctuation to it, U+1734 and U+1171E still accents).
  `tokenizer.js` lists them. A newer browser could add characters to the
  first list, which only matters for scripts added to Unicode recently.
- Lowercasing is per code point, so there is no final-sigma rule.

The committed fixture (`web-demo/tests/fixtures/tokenizer-cases.json`, 34
cases) keeps the interesting cases in CI; the sweep stays a script because
its input is 1.1 million texts.

### Parity with the Python embedder

`scripts/make_sample_index.py` now builds the sample with bge-small
(`vectors.bin` adds 90 KiB; the sample is 398 KiB for 191 chunks) and, in the
same run, embeds 14 queries with
`OnnxEmbedder` and records each vector and its hybrid top 10 on that index in
`web-demo/tests/fixtures/queries.json` (31 KiB). `embed.test.mjs` embeds the
same queries through the browser path under Node and requires cosine
≥ 0.999 and identical top-10 ids. The measured worst case is
1 − cosine = 2·10⁻¹³, and all 14 top-10 lists are identical: the tokens are
the same, and ONNX Runtime's WASM kernels reproduce the native floats to
rounding. The vectors are compared before quantization, so int8 does not
hide differences.

In CI the test downloads the model into the cache the model-check workflow
also uses, and a missing model, runtime or wasm package fails the job
instead of skipping it. Locally it uses the CLI's model cache and skips
without the model, like `pytest`.

### Measured

On an i9-10900K (Windows 11) in Chromium 152 (the browser pane of the Claude
desktop app; `Chrome/152.0.7977.149`), with the sample index and the 20
queries in `web-demo/bench-queries.json` (13 to 21 tokens with the prefix).
Query latency is the median / p95 over queries of the mean of 5 runs, best of
three runs. A first load was measured once per setting (each origin has its
own Cache API); cached loads give the range of two page loads.

| Load | 1 thread | 4 threads (`--isolated`) |
|---|---|---|
| First (empty cache): download from huggingface.co | 2.8 s | 2.5 s |
| First: SHA-256 of both files | 301 ms | 300 ms |
| First: ONNX session | 456 ms | 444 ms |
| First, in all (includes writing the cache) | 3.9 s | 3.6 s |
| Cached: read from the Cache API | 65 to 70 ms | 69 to 85 ms |
| Cached: SHA-256 | 287 to 301 ms | 295 to 297 ms |
| Cached: ONNX session | 268 to 443 ms | 300 to 477 ms |
| Cached, in all | 0.64 to 0.83 s | 0.70 to 0.86 s |

The download time reflects this machine's connection (about 45 MiB/s); the
file is 127 MiB. A cached load made no request outside 127.0.0.1.

| Warm query (median / p95) | embed | embed + hybrid search |
|---|---|---|
| Chromium 152, 1 thread | 29.0 / 32.9 ms | 29.1 / 33.1 ms |
| Chromium 152, 4 threads | 10.3 / 11.8 ms | 10.5 / 12.0 ms |
| Node 25, 1 thread | 29.5 / 34.3 ms | 29.6 / 34.6 ms |
| Python onnxruntime 1.31, 1 thread (native) | 10.5 / 12.4 ms | |
| Python onnxruntime 1.31, default threads (native) | 5.0 / 5.3 ms | |

Embedding is the whole cost: the search adds about 0.2 ms on this index.
Single-threaded WebAssembly is 2.8x slower than native ONNX Runtime on one
thread (WebAssembly SIMD is 128 bits wide; native ONNX Runtime picks wider
x86 kernels), and four threads bring it to native single-thread speed.
Threads need a cross-origin isolated page (COOP and COEP headers). The page
uses 4 threads when it is isolated and one otherwise; see "Threads without
server headers" below for how it gets isolated on any host.

| What the page loads | Raw | gzipped |
|---|---|---|
| onnxruntime-web (module, glue, `.wasm`) | 13.7 MiB | 3.5 MiB |
| notelight-wasm (`.wasm` + glue) | 262 KiB | 106 KiB |
| The demo's own HTML, CSS and JavaScript | 40 KiB | 16 KiB |
| bge-small (`model.onnx` + `tokenizer.json`), once, then cached | 127.6 MiB | |

### Serving the demo

`python -m http.server` takes MIME types from the OS, and on Windows the
registry often maps `.mjs` to `text/plain`, which browsers refuse to run as a
module; onnxruntime-web ships as `.mjs`. `scripts/serve_demo.py` sets the
types explicitly and can send the isolation headers. Like the CLI's web UI, it
binds to 127.0.0.1 only.

### Threads without server headers

Static hosts such as GitHub Pages cannot send COOP and COEP, which would leave
a deployed demo on one thread (29 ms per query). `web-demo/coi-sw.js` is a
service worker that adds the two headers to this origin's responses and does
nothing else: no caching, no body rewriting, and cross-origin requests pass
through untouched (the model download is a CORS request, which COEP
`require-corp` allows). A page becomes isolated only when its own navigation
response carries the headers, so on the first visit `app.js` registers the
worker and reloads once. A per-tab flag means it is tried at most once and can
never loop, and where service workers or storage are unavailable the page
simply runs on one thread. `?threads=1` skips it, for measuring.

It is about 30 lines of our own rather than the common `coi-serviceworker`
package, which also patches `window` and handles credentialless COEP; this
page needs neither. Verified with headless Chrome 155 and a fresh profile
against `serve_demo.py` without `--isolated`: the first load reloads once and
is then isolated, the model downloads from the Hub, and the embedder runs on
4 threads at 10.0 / 12.5 ms per query (median / p95, best of three), against
28 ms with `?threads=1`. The Claude desktop app's browser pane, where the
other numbers above were measured, does not run service workers, so there
`--isolated` is still the way to get threads.

## v2: indexing in the browser

The demo now indexes a folder the user picks, in the page, and keeps the
index across reloads. A folder indexed here and the same folder indexed by
`notelight index` give the same chunks, byte-identical `docs.bin` and
`bm25.bin`, document vectors equal to rounding, and the same top 10. The
pieces are `chunkText` and `IndexWriter` in `crates/notelight-wasm`,
`web-demo/ingest.js` (the walk), `web-demo/indexer.js` (incremental sync),
`web-demo/store.js` (OPFS or IndexedDB), and `web-demo/index-worker.js`.

### The core writes the index; JavaScript only walks and caches

`notelight-wasm` gained two thin bindings: `chunkText(text, maxChars,
markdown)` over `chunk::chunk_text`, and an `IndexWriter` class over the
core's writer whose `build(generation)` returns the four files' bytes
(`IndexWriter::build`, added in the spike). Nothing in the core changed and
the on-disk format is the same: a generation built in the browser is
byte-for-byte what `commit` writes (a Node test rebuilds the committed
sample from its own chunks and gets the same `docs.bin` and `bm25.bin`).
The `.wasm` grew from 245 KiB to 322 KiB (128 KiB gzipped) because the BM25
builder, the chunker and `serde_json` now ship to the browser.

### The same files, in the same order

Chunk ids follow the walk order, and BM25 breaks ties by id, so "the same
top 10" needs the same files in the same order, not only the same text.
`ingest.js` follows `ingest.py` line by line: hidden names and `SKIP_DIRS`
pruned, files sorted by code point (Python's string order, not JavaScript's
UTF-16 order), `Path.suffix` semantics, the 10 MiB and 50 MiB limits, a NUL
in the first 8 KiB means binary, then UTF-8 (BOM dropped) and Windows-1252.
WHATWG's `windows-1252` decoder maps the five bytes Python's `cp1252`
rejects (0x81, 0x8D, 0x8F, 0x90, 0x9D) to control characters, so those
bytes are checked first and the file is skipped with Python's reason.

`.notelightignore` needed the most care. "gitignore syntax" is not one
thing: Git, the `ignore` npm package and pathspec disagree at the edges
(re-including a file under an excluded directory, `**/` alone, invalid
ranges). The walk must agree with the CLI, so `ingest.js` ports pathspec
1.1's `GitIgnoreSpecPattern` and its matching order (patterns scanned from
the last, a match through a directory marker ranking below a direct one)
instead of using a library. `scripts/make_indexing_fixture.py` records
pathspec's verdict on 51 paths against two sets of tricky patterns, and the
Node test requires the same verdicts. Even Python's quirks are kept: a
UTF-8 BOM at the start of `.notelightignore` stays part of the first
pattern, as `read_text(encoding="utf-8")` leaves it.

PDF and DOCX are found by the walk. PDFs are now read with pdf.js (see
"v2: PDFs in the browser"); DOCX is skipped with a note that says to use the
CLI.

### Incremental, with Python's caches in a browser's shape

| Python (`cache.sqlite`) | Browser (per index) |
|---|---|
| `files`: path → size, mtime, content key | `meta.json`, `files`: path, size, `lastModified`, SHA-256 |
| `chunks`: content key → chunks | the live generation itself: unchanged files take their chunks from it |
| `embeddings`: `sha256(model_id + "\n" + text)` → vector | `embeddings.bin`, same keys (format below) |
| `skipped`: path → size, mtime, reason | `meta.json`, `skipped` |

A file whose size and modification time match is not read. One that was
touched but hashes the same is not re-chunked. Only texts missing from the
embedding cache are embedded, in Python's order (missing keys sorted, 64 per
call, sorted by length into batches of 32), so a renamed or copied note costs
nothing and an edited one re-embeds only the sections that changed. After a
successful sync the cache keeps only the vectors the live generation uses; a
keyword-only sync leaves it alone. A sync with nothing to embed never loads
the model, as `OnnxEmbedder` loads lazily. If nothing changed, nothing is
written. The Node test checks each of these with a counting fake embedder.

`embeddings.bin` is `NLEMB001`, then dim, count and model id length (u32
little-endian), the model id, padding to 4 bytes, then per vector a 32-byte
SHA-256 key and `dim` float32s, read with `DataView` at checked offsets. A
cache for another model or dimension, or one whose length does not add up,
is ignored rather than trusted: it can always be rebuilt.

### Storage: OPFS, IndexedDB as the fallback

`store.js` is four calls (`read`, `write`, `list`, `remove`) over the Origin
Private File System, with an IndexedDB store of path → bytes where OPFS is
missing or cannot write (the worker writes, reads back and removes a probe
file to decide). An index is `indexes/<id>/meta.json`, one
`gen-NNNNNN/` folder and `embeddings.bin`. As on disk, generations are
immutable: a sync writes the new generation, then `embeddings.bin`, then
`meta.json`, which is the pointer, and only then deletes the old
generation. A sync interrupted before `meta.json` leaves the old index
live; its leftovers are removed by the next sync. `navigator.locks` plays
the writer lock's part, so two tabs cannot update one index at once. The
page asks for persistent storage (`navigator.storage.persist()`), and says
when the browser may still evict it.

With the File System Access API the folder handle is kept in IndexedDB, so
**Update** re-reads the folder after one permission prompt; elsewhere the
user picks it again.

### A worker

Everything that touches files and storage runs in `index-worker.js`: the
walk, hashing, chunking, embedding (its own onnxruntime-web session, loaded
only when needed and released after the sync), building and storing. The page
sends requests and receives progress messages and the result, with the
generation's bytes transferred rather than copied. **Cancel** is checked
between batches of 32 chunks; finished batches are saved first, so the next
sync resumes where the cancelled one stopped. Searching stays on the main
thread: it takes microseconds to milliseconds.

### Privacy

The promise is that files never leave the browser, and the code keeps it
simple to check: files are read only in the worker, nothing is sent with
them, the only cross-origin request is the pinned model download
(`embed.js`, verified by SHA-256), and storage is private to the origin.
The page says so above the folder button.

### Measured

This repository's `docs/` (11 files, 141 KiB, 207 chunks), on an i9-10900K
(10 cores, Windows 11), in headless Chrome 155 driven over the DevTools
protocol through the page's `<input webkitdirectory>`, with
`serve_demo.py --isolated`. Each run indexes from an empty store; the time
is from the walk to the stored generation, without loading the model.
Native is `notelight.indexer.sync` on the same folder in a fresh
`NOTELIGHT_HOME`, with onnxruntime 1.31.

| Fresh index of docs/ | Browser | Native (Python) |
|---|---|---|
| Keywords only | 41 to 48 ms: 4,300 to 5,100 chunks/s, 2.9 to 3.4 MiB/s | 0.27 to 0.38 s: 550 to 760 chunks/s |
| bge-small, 1 thread | 149 s: 1.4 chunks/s | 43.4 to 43.7 s: 4.7 chunks/s |
| bge-small, 4 threads | 41.9 s: 4.9 chunks/s | 16.3 s: 12.7 chunks/s |
| bge-small, 8 threads (the page's default) | 26.7 to 27.4 s: 7.6 to 7.8 chunks/s, 5 KiB/s | |
| bge-small, 10 threads | 27.9 s: 7.4 chunks/s | |
| bge-small, onnxruntime's default threads | | 14.3 to 14.7 s: 14.1 to 14.5 chunks/s |
| Update, nothing changed | 7.6 to 8.5 ms, the model not loaded | |

Without the model, indexing is quick: of the best keyword run's 41 ms, the
walk took 0.7 ms, reading and hashing 5.4 ms, decoding and chunking 3.2 ms,
building the generation 13 ms and writing it to OPFS 18 ms. The Python
figure covers the same steps plus its SQLite cache writes.

With the model, embedding is over 99% of the time, and WebAssembly is about
3x slower than native per thread, as for queries (128-bit SIMD against
native's wider x86 kernels). Four threads in the browser match one native
thread; eight are 1.5x faster than four, and ten add nothing, so indexing
uses up to 8 (`?index-threads=N` overrides it) while queries keep 4. At
about 8 chunks/s, a first index of a 10 MiB notes folder (roughly 15,000
chunks) would take about half an hour; after that, updates embed only what
changed. Keyword-only indexing is offered for exactly this reason.

Loading the model in the worker took 3.1 to 4.3 s per run here because
headless Chrome could not open the Cache API in its profile, so every run
downloaded it again (the page now carries on without a cache instead of
failing). In the desktop app's Chromium the worker reads it from the cache in
0.5 to 0.8 s, as the page does for queries.

Correctness is measured in CI, not here: the fixture folder indexed by the
browser path under Node gives the same 23 chunks as the CLI, byte-identical
`docs.bin` and `bm25.bin`, document vectors within 1 − cosine = 2.4·10⁻¹³ of
Python's, and the same hybrid and BM25 top 10 for all 10 queries.

| What the page loads | Raw | gzipped |
|---|---|---|
| notelight-wasm (`.wasm` + glue), now with the writer and chunker | 349 KiB | 134 KiB |
| The demo's own HTML, CSS and JavaScript | 106 KiB | 37 KiB |

### Next

- **Faster embedding.** WebGPU, or a smaller model, would be the lever; the
  vectors would then need the same parity test against the CLI.
- **PDF and DOCX** with JavaScript extractors that match the CLI's text, or
  a clear statement that they differ. PDF is done, with pdf.js, and its text
  differs (see "v2: PDFs in the browser"); DOCX is still to do.
- **Memory.** The writer holds every chunk in WebAssembly memory before
  `build`; very large folders would want a streaming writer.
- **npm.** Publish `notelight-wasm` with `chunkText`, `IndexWriter` and
  `Index`: packaged, see "v2: the npm package".

## v2: PDFs in the browser

A folder indexed in the page now includes its PDFs, with the CLI's citation
contract: the text of each page ends in a newline, pages are joined with a
form feed, chunks cite `path:line_start-line_end` of that text, and results
show the pages those lines are on (`p. 3`, `pp. 3-4`). The pieces are
`web-demo/documents.js` (the counterpart of `documents.py`), a few lines in
`indexer.js` and the worker, and `scripts/vendor_pdfjs.py`.

### pdf.js, vendored and pinned

pdf.js (Mozilla, Apache-2.0) is the PDF reader Firefox's viewer is built on,
and the only mature one in plain JavaScript; the alternatives were compiling
a native library to WebAssembly (MuPDF is AGPL, as for the CLI; PDFium is
large and needs its own toolchain) or porting pypdf. `scripts/vendor_pdfjs.py`
pins `pdfjs-dist` 6.3.289 as `vendor_ort.py` pins onnxruntime-web: the npm
tarball is checked against npm's SHA-512 integrity, the two scripts against
their own SHA-256, and the 169 CMap files (character maps for CJK fonts)
against one SHA-256 over their names and hashes. Nothing comes from a CDN.
It ships the legacy build: the modern one calls
`Map.prototype.getOrInsertComputed`, which Node 24 and browsers a year or
two old do not have, and the legacy build is only 0.1 MiB larger. Standard
font data and the image decoders (`standard_fonts/`, `wasm/`) are left out:
they draw pages and decode images, and only text is read.

| What a folder with PDFs adds | Raw | gzipped |
|---|---|---|
| `pdf.min.mjs` + `pdf.worker.min.mjs` (loaded on the first PDF) | 1.75 MiB | 529 KiB |
| `cmaps/` (one file at a time, only for PDFs that use a predefined CMap) | 1.11 MiB in 169 files | 0.84 MiB |
| `documents.js` (always) | 9 KiB | 4 KiB |

A folder without PDFs loads none of it: the extractor (`lazyPdfExtractor`)
imports pdf.js on its first PDF. Loading it took 0.6 s on a fresh profile
from `serve_demo.py` and 0.14 to 0.2 s when the browser had it cached; the
worker reports it on its own (`stats.ms.pdfjs`), like the model.

### pdf.js runs in the indexing worker, without a worker of its own

pdf.js normally parses in a Web Worker of its own. The indexing worker is
already off the main thread, so it imports `pdf.worker.min.mjs` first, and
pdf.js finds `globalThis.pdfjsWorker` and runs the parser in the same thread
(its "fake worker"). That avoids a nested worker, a second script URL for
the page's COOP/COEP service worker to cover, and copying each file to
another thread. pdf.js's own loader for CMap files resolves URLs against
`document.baseURI`, which a worker does not have, so `documents.js` passes
a `BinaryDataFactory` that fetches `vendor/pdfjs/cmaps/<name>` (only names
made of word characters, dots and dashes). Fonts are not installed
(`disableFontFace`), glyph programs are not compiled with `eval`
(`isEvalSupported: false`) and XFA forms are off. Verified in headless
Chrome 155 against the built site served without isolation headers, so
through `coi-sw.js` as on GitHub Pages: the page was cross-origin isolated,
indexing read the PDFs, and the console showed no errors.

### The same contract as the CLI, not the same text

The text of each page is pdf.js's text runs in order with a newline where
pdf.js marks the end of a line (`hasEOL`); then exactly what `documents.py`
does: NUL, form feed and CR removed, a vertical tab made a newline, trailing
newlines stripped, one newline added, pages joined with `\f`. A page with no
text is a line of its own, so a line belongs to exactly one page. Page
numbers are the page's position in the file (1-based), not its printed
label, as in the CLI.

The page starts are stored per file in `meta.json` (`files[].pages`), the
browser's counterpart of the CLI's `pages` table, and the page computes
`pageRange` from them and a hit's lines. An index opened from a CLI index
folder has no page table in the page, so its PDF hits show lines only.

pdf.js and pypdf decide differently where a line ends, where spaces go and
in which order text runs come, so the extracted text is not byte-identical
and a PDF's chunks and line numbers in a browser index are not the CLI's.
That is accepted: each index cites its own text consistently, and an index
built in the browser is only read in the browser. On the generated test PDFs
the two agree exactly. On three real documents (below) both give the same
pages, and comparing word sequences page by page gives a similarity of 1.000
on average (lowest page 0.974); the differences are mostly whitespace:
pypdf keeps trailing spaces and lines that hold only a space, pdf.js does
not. With whitespace collapsed and blank lines dropped, 99.0% (NIST SP
800-63B), 99.2% (SP 800-53r5) and 99.96% (a novel printed by Chrome) of
pypdf's distinct lines appear in pdf.js's text, and the line counts differ
by 0.05 to 4%.

Skips follow the CLI, with the same reasons: pdf.js tries the empty user
password itself, so a PDF encrypted only to restrict printing or copying is
read, and one that needs a password is skipped as `encrypted PDF (password
required)`; a PDF with no text on any page is `PDF has no text layer
(scanned or image-only?)`; anything pdf.js fails to parse is `unreadable
PDF: <error>`, and extraction never throws. The 50 MiB limit is the walk's,
as before.

### Cache keys

How a file becomes text is now part of the per-file cache in `meta.json`
(`mode`: `md`, `txt`, or the PDF extractor's id, `pdf1:pdfjs-6.3.289`), as
`content_mode` is part of the CLI's `doc_key`. A file is reused only when its
mode is unchanged, so a new pdf.js re-extracts every PDF (counted as
changed), and a PDF skipped before PDFs were read here is read on the next
update. Indexes written before this change have no `mode`; their files were
all text, so their mode is derived from the suffix and nothing is re-read.
`meta.json` keeps its version: the new fields are optional.

### Tests

`scripts/make_pdf_fixtures.py` writes seven small PDFs object by object, so
their text is known exactly (18 KiB in all, committed), and records what
pypdf makes of them. `web-demo/tests/documents.test.mjs`, in CI's wasm job,
checks with the vendored pdf.js: the text and page count of a four-page PDF
with a blank page, of a five-page one and of the empty-password encrypted
copy; that every line of the five-page one maps to the page it was drawn
on; Japanese text in a non-embedded CID font, read through the two CMap
files it needs and no others; the skip reasons for a password-protected,
an image-only, a broken, an empty and a non-PDF file; and that the vendored
pdf.js is the version the extractor id names. Then a folder of a Markdown
note, two readable PDFs, three unreadable ones and a DOCX is indexed through
`indexer.js` with the wasm core and searched: the skip reasons, the page
starts in `meta.json`, every chunk citing lines of the extracted text,
search hits on the right pages (checked against a second count of form
feeds), no PDF re-read on an unchanged update or a touched file, every PDF
re-read under a new extractor id, and PDFs skipped by an older index read
on the next update.

### Measured

Extraction in headless Chrome 155 on an i9-10900K (Windows 11), driven over
the DevTools protocol through the page's `<input webkitdirectory>`, with the
built site served by `serve_demo.py` without isolation headers, keyword-only,
each run a fresh index of a folder holding one PDF. The time is
`stats.ms.extract`: pdf.js parsing and extracting text in the worker, on one
thread, without loading pdf.js. Two unrelated jobs kept the machine's cores
busy during most runs, so the ranges are wide; each row covers all 8 runs.

| Document | Pages | Size | Extraction | Pages/s | MiB/s | pypdf, native (best of 3) |
|---|---|---|---|---|---|---|
| NIST SP 800-63B (2017, Adobe PDF Library) | 80 | 1.41 MiB | 0.72 to 0.90 s | 89 to 112 | 1.6 to 2.0 | 3.8 s: 21 pages/s |
| NIST SP 800-53 Rev. 5 (2020, Adobe PDF Library) | 492 | 5.79 MiB | 3.8 to 9.5 s | 52 to 130 | 0.6 to 1.5 | 26.9 s: 18 pages/s |
| *Pride and Prejudice*, Project Gutenberg HTML printed by Chrome (Skia) | 192 | 2.26 MiB | 1.14 to 1.99 s | 96 to 168 | 1.1 to 2.0 | 16.4 s: 12 pages/s |

The three in one folder: 764 pages, 9.46 MiB, extracted in 5.5 s (138
pages/s, 1.7 MiB/s), 2,152 chunks indexed in 5.9 s in all. The NIST
publications are works of the US government and the novel is in the public
domain; they were downloaded for the measurement and are not in the
repository. Extraction is cheap next to embedding: at about 8 chunks/s with
bge-small, the 171 chunks of SP 800-63B take about 20 s to embed and under
a second to extract. pypdf, the CLI's reader, took 5 to 10 times as long on
the same files and the same busy machine (pure Python, natively).

## v2: the npm package

`notelight-wasm` on npm is the crate the demo loads, packaged for other
JavaScript projects: `scripts/build_npm.py` builds it into `dist/npm/`,
`scripts/smoke_npm.py` installs the tarball into a scratch project and uses
it, and `.github/workflows/npm.yml` publishes it. The README the package
ships is `crates/notelight-wasm/npm/README.md`.

### Name: `notelight-wasm`

Both `notelight` and `notelight-wasm` were free. The package is only the
retrieval core: no embedder, no folder walk, no CLI. Under the bare name
`notelight`, people would expect what `pip install notelight` gives them, and
the name would be taken if a fuller JavaScript package (with the demo's
embedder and indexer) ever exists. `notelight-wasm` matches the crate and the
file names wasm-bindgen generates (`notelight_wasm.js`), and says what it is.
A scope (`@searope/notelight`) would tie the name to a personal account and
needs an npm user or organisation called `searope`; the unscoped name is
easier to find and to type.

### One build: `--target web`, plus a Node entry

The package ships the demo's exact files (`wasm-bindgen --target web`): ES
modules whose default export, `init()`, fetches the `.wasm` from
`new URL("notelight_wasm_bg.wasm", import.meta.url)`. Browsers run that
without a bundler, and Vite, webpack 5 and Rollup recognise the pattern and
emit the `.wasm` as an asset. `--target bundler` would instead import the
`.wasm` as an ES module, which needs bundler plugins or experimental flags,
and would mean testing two builds; `--target nodejs` is CommonJS.

Node's `fetch` does not read `file:` URLs, so the `node` export condition
points to `node.js`, ten lines that read the `.wasm` from disk, call
`initSync` and re-export everything. `init()` then returns at once, so code
shared between the browser and Node is the same. Deno and Bun use the `node`
condition too. `sideEffects` lists only `node.js`: a bundler may drop any
other unused module, but not the one whose import instantiates the module.
`engines` says Node 22+, the versions CI tests (22 and 24, on Linux, Windows
and macOS); Node 20 is past its end of life.

### What it checks

`build_npm.py` fails unless `npm pack --dry-run` lists exactly the nine
expected files (the four wasm-bindgen outputs, `node.js`, the README, both
licences and `package.json`), so a stray file cannot ship. The smoke test
runs from outside the checkout, so `notelight-wasm` resolves to
`node_modules`: it opens the committed sample index, checks BM25 results,
checks that hybrid search with the Python embedder's query vectors returns
the CLI's top 10 for all 14 fixture queries, builds and searches an index in
memory, and type-checks a TypeScript file against the shipped declarations
with a pinned TypeScript (5.9.3). That check found that wasm-bindgen's
declarations use `Symbol.dispose`, so TypeScript users need
`esnext.disposable` in `lib` (or `skipLibCheck`); the README says so.

The package is 145 KiB packed (382 KiB unpacked; the `.wasm` is 322 KiB, 128
KiB gzipped).

### Versions and publishing

The version is Cargo's `[workspace.package]` version, the same as the
Python package's, so `version()` in JavaScript, `notelight version` and the
index's `created_by` agree. npm releases have their own tags, `npm-vX.Y.Z`,
because PyPI releases use `vX.Y.Z` and the two need not ship together: the
npm workflow checks that the tag equals the Cargo version, builds, runs the
smoke test on every OS, and publishes the tarball it tested from the `npm`
GitHub environment, which needs the owner's approval, like `pypi`. A
prerelease version is published under the `next` dist-tag.

Publishing uses npm trusted publishing (OIDC; npm CLI 11.5.1+, pinned to
11.19.1 in the workflow): no token is stored, and provenance is attached
automatically for a public repository (`--provenance` is passed anyway, to be
explicit). A trusted publisher can be configured only for a package that
already exists on the registry (npm's docs, checked 2026-10-16). In
practice the package was created with a placeholder version,
`0.0.0-stage` ("Temporary package placeholder for staged publishing"), and
the trusted publisher could then be attached before any real version
existed, so a workflow publish with provenance was possible for the first
version too. The placeholder stays in the version list; deprecate it
(`npm deprecate`) rather than unpublish it.

What happened for 0.2.0 (2026-10-17): it was published by hand from the
tarball the `workflow_dispatch` run built, seconds before the tag's
workflow tried; the workflow then failed with "cannot publish over the
previously published versions", which is harmless. The registry's shasum
equals the CI-tested tarball's, so the bytes are the tested ones, but 0.2.0
has no provenance statement. An earlier tag run failed at the version
check because the tag pointed at a branch commit with the old version; the
check did its job. Lessons, now in `CLAUDE.md`, "Releasing to npm": tag
`origin/main` by name, and publish only through the workflow.

A trusted publisher that has not completed a publish expires after 2
days, so the one attached for 0.2.0 has to be re-created, or checked with
`npm trust list notelight-wasm`, before the next release.

### API stability

The package is 0.x and follows the project's rule: a minor version may
change the API; a patch version does not. What is likely to change before
1.0:

- **Argument names in the TypeScript declarations** are still the Rust
  names (`max_chars`, `heading_path`); positional calls are not affected.
- **Search options.** `search` takes only `k`; the core's other options
  (fusion depth, candidate count, RRF constant, phrase bonus) may be exposed,
  probably as an options object, which would change the signature.
- **Result shapes** may gain fields (for example phrase-match details);
  fields are not removed or renamed in a patch release.
- **`IndexWriter` metadata** is a JSON string today; it may become an
  object.
- **The index schema version.** A change to the on-disk layout bumps
  `SCHEMA_VERSION`; a package then refuses older indexes (`new Index`
  throws) until they are rebuilt by a matching `notelight index`. This is
  the same rule as for the Python package, whose version this package
  shares.

What should stay: the `Index`, `IndexWriter`, `chunkText` and `version`
names, the bytes-in constructor (the package never touches a filesystem or
the network), and results as plain objects with camelCase fields.
