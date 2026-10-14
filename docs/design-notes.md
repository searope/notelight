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
