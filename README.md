# notelight

**Private semantic search over your own files, for you and your AI agents.**
A Rust retrieval engine with a Python interface, a CLI, and an MCP server, with
retrieval quality you can measure.

> **Status: milestone 1 of 5.** The Rust retrieval core (chunking, BM25,
> quantized vector search, fusion and the on-disk index) is done and tested.
> The Python CLI that indexes your folders lands in milestone 2. See the
> [roadmap](#roadmap).

## Why

Tools that let you "chat with your files" usually need a server, an account, or
an upload. notelight keeps everything on your machine:

- **Hybrid retrieval.** BM25 plus quantized embeddings, fused with Reciprocal
  Rank Fusion, so both exact names and paraphrased questions work.
- **Agent-ready.** An MCP server lets Claude, Copilot or Cursor search your notes
  and cite `path:line`.
- **Measured.** `notelight eval` reports recall@10 and nDCG@10 on BEIR datasets
  for every quantization setting, not just anecdotes.
- **Private.** No telemetry, and no network after a one-time model download.

## Architecture

```
            ┌────────────── notelight-core (Rust) ─────────────┐
            │ chunking · BM25 · binary + int8 search · RRF     │
            │ memory-mapped, versioned index format            │
            └───────┬──────────────────────────────┬───────────┘
              PyO3 / maturin                 wasm-bindgen (v2)
                    │                              │
   notelight (Python package)              browser app (v2)
   embeddings via ONNX Runtime
   CLI · MCP server · eval harness · localhost UI
```

The core crate has **no Python, ML or async dependencies**. That keeps it small,
easy to test, and portable to WebAssembly. Embeddings run in Python, where the
ML tooling is.

## Performance (Rust core)

Measured with `cargo bench -p notelight-core` on synthetic data, using a
single core of a cloud VM. These are indicative numbers, not a tuned benchmark:

| Operation | Size | Time |
|---|---|---|
| Vector search, binary candidates + int8 rescore (k=10, 200 candidates) | 100k × 384-dim | 1.3 ms |
| Same, exhaustive int8 scan (reference) | 100k × 384-dim | 45 ms |
| BM25 query, 3 terms | 100k chunks | 0.2 ms |
| Markdown chunking | 1 MB document | 6 ms (≈165 MiB/s) |

The two-stage search is about 33× faster than scanning every int8 vector,
because the binary pass compares 48 bytes per vector with `popcount`, and only
200 candidates are re-scored. Retrieval quality is measured in milestone 3.

## Development

Requirements: [Rust](https://rustup.rs) (stable) and Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"        # builds the Rust extension via maturin
pytest                         # Python tests
cargo test -p notelight-core   # Rust unit, integration and property tests
cargo bench -p notelight-core  # Rust benchmarks
notelight version
```

After changing Rust code, rebuild with `maturin develop --release`.

## Roadmap

| # | Milestone | Status |
|---|---|---|
| 0 | Scaffold: Rust workspace, PyO3 bindings, CI on 3 OSes | ✅ |
| 1 | Rust core: chunking, BM25, quantized search, index format | ✅ |
| 2 | Python package: embedder, incremental indexer, CLI | ⏳ |
| 3 | Evaluation: BEIR results table, Rust vs NumPy baseline | |
| 4 | MCP server for AI agents | |
| 5 | Web UI, PDF/DOCX, wheels on PyPI | |
| v2 | Same core compiled to WebAssembly, in-browser app | |

## Documentation

- [Design notes](docs/design-notes.md): why it is built this way
- [Index format](docs/index-format.md): the on-disk layout, byte by byte
- Rust walkthroughs for readers coming from C# or Python:
  [milestone 0](docs/learning/m0-rust-walkthrough.md),
  [milestone 1](docs/learning/m1-rust-walkthrough.md)

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
