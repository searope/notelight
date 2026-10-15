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
the time to write files. The indexer keeps three caches in `cache.sqlite` next
to the index:

| Table | Key | Saves |
|---|---|---|
| `files` | relative path → size, mtime, content key | reading and hashing files that did not change |
| `chunks` | `sha256(file bytes)` + chunking mode | re-chunking known content |
| `embeddings` | `sha256(model_id + "\n" + embedded text)` | re-embedding known chunks |

The embedding key comes from the exact text sent to the model (heading path
plus chunk text) and the model id. A renamed, moved or copied file reuses its
vectors, a file edited in one section re-embeds only the chunks that changed,
and switching models can never return stale vectors. After a successful commit,
rows no current file refers to are deleted, so the cache cannot grow without
bound.

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
