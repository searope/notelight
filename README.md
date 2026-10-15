# notelight

[![CI](https://github.com/searope/notelight/actions/workflows/ci.yml/badge.svg)](https://github.com/searope/notelight/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](Cargo.toml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**Private semantic search over your own files, for you and your AI agents.**
A Rust retrieval engine with a Python interface, a CLI, and an MCP server, with
retrieval quality you can measure.

> **Status: milestone 2 of 5.** You can index a folder of notes and search it
> from the command line. Retrieval-quality numbers (milestone 3) and the MCP
> server for AI agents (milestone 4) are next. See the [roadmap](#roadmap).

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

## Usage

notelight is not on PyPI yet (milestone 5), so install it from source. You
need [Rust](https://rustup.rs) and Python 3.10+:

```bash
pip install git+https://github.com/searope/notelight
```

Index a folder of Markdown and text files, then search it. This is a real
run over this repository's `docs/` folder on a laptop CPU:

```console
$ notelight index docs
Downloading onnx/model.onnx (133 MB, one time)...
Embedding 62 new chunks...
D:\Projects\notelight\docs: 5 files (5 added, 0 changed, 0 removed), 62 chunks, 62 embedded, generation 1, 4.9s

$ notelight search "what happens when two processes index at once" -k 2
 1. design-notes.md:153-161  Design notes > Milestone 2 decisions > One writer, enforced with a lock file
    ### One writer, enforced with a lock file The core assumes a single writer: `commit` sweeps up leftovers from earlier failed commits, which would be…
 2. learning/m1-rust-walkthrough.md:82-88  Milestone 1: Rust walkthrough > 5. Validate once, then index freely (`Bm25::validate`)
    ## 5. Validate once, then index freely (`Bm25::validate`) `from_bytes` checks every offset table, every term and every posting…
```

Run `notelight index` again after editing your notes. Only new or changed
chunks are embedded. On an unchanged folder the model is never loaded, so the
re-run takes about as long as starting the CLI (0.5 s for the folder above).

| Command | What it does |
|---|---|
| `notelight index <folder>` | create or update the index for a folder |
| `notelight search "<query>" [-k 10] [--json] [-f <folder>]` | hybrid search; `--json` returns path, line range, heading path and text for scripts and agents |
| `notelight status [<folder>] [--json]` | list indexed folders, sizes, embedders and generations |
| `notelight version` | versions and the data directory |

Options:

- `--embedder bge-small|hashing|none`: `bge-small` (default) is
  [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5),
  pinned to one revision and verified by SHA-256. `none` builds a keyword-only
  (BM25) index with no model. `search` picks the index's own embedder
  automatically and refuses a different one.
- `--offline`: never download; fail if the model is not cached.
- Files: `.md`, `.markdown` and `.txt`, in UTF-8 or Windows-1252. Hidden
  folders (`.git`, `.obsidian`, ...), `node_modules`, `target`, `build` and
  similar are skipped, as is anything matched by a `.notelightignore` file
  (`.gitignore` syntax) in the folder root.
- Indexes are stored in your user data directory, never in the indexed folder.
  Set `NOTELIGHT_HOME` to keep them elsewhere, or `NOTELIGHT_MODEL_CACHE` to
  share one model download.

From Python:

```python
from notelight import OnnxEmbedder, Searcher, sync

embedder = OnnxEmbedder()
sync("~/notes", embedder)
for hit in Searcher("~/notes", embedder).search("release hotfixes", k=5):
    print(hit.location, hit.heading_path)
```

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
pytest                         # Python tests (no network; uses a fake embedder)
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
| 2 | Python package: embedder, incremental indexer, CLI | ✅ |
| 3 | Evaluation: BEIR results table, Rust vs NumPy baseline | ⏳ |
| 4 | MCP server for AI agents | |
| 5 | Web UI, PDF/DOCX, wheels on PyPI | |
| v2 | Same core compiled to WebAssembly, in-browser app | |

## Documentation

- [Design notes](docs/design-notes.md): why it is built this way
- [Index format](docs/index-format.md): the on-disk layout, byte by byte
- Rust walkthroughs for readers coming from C# or Python:
  [milestone 0](docs/learning/m0-rust-walkthrough.md),
  [milestone 1](docs/learning/m1-rust-walkthrough.md),
  [milestone 2](docs/learning/m2-rust-walkthrough.md)

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
