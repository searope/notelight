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
low hundreds of thousands of chunks, and at 100k chunks the full binary scan
plus int8 rescoring takes about 1.3 ms. An approximate index (HNSW, IVF) would
add memory, build time and tuning knobs, and it can silently miss results.
Exhaustive search is simpler and exact up to quantization.

### Two-stage vector search

1. **Binary stage.** One bit per dimension, compared with XOR + popcount over
   `u64` words: 48 bytes per 384-dim vector.
2. **Int8 stage.** Only the best 200 candidates are re-scored with int8 codes.
   The query stays `f32` ("asymmetric" scoring), which is more accurate than
   quantizing both sides.

Candidate selection uses a histogram instead of a sort. Hamming distances are
integers between 0 and `dim`, so counting them finds the cut-off in one pass
with no comparisons.

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
