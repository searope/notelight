# CLAUDE.md: notelight

Private semantic search over a user's own files, for people and AI agents. It
has a Rust retrieval core with a Python interface, a CLI, an MCP server and an
eval harness. It is a **portfolio project**: code quality, tests, docs and
honest benchmarks matter as much as features.

## Owner and working style

- The owner is Serge, a senior engineer (C#/.NET, Python). He knows **some Rust
  and is learning it** through this project.
- Every milestone ends with a Rust walkthrough in `docs/learning/mN-rust-walkthrough.md`
  that explains new Rust ideas in C#/Python terms. Keep the Rust in the core
  plain: structs, slices, `Result`. No async, no clever generics unless they
  earn their place.
- He develops on **Windows**: give PowerShell commands and keep paths and
  scripts Windows-friendly. CI also covers Linux and macOS.
- When offering choices, give a short list of clear options.

## Layout

```
crates/notelight-core/   pure Rust engine, no Python/ML/async deps (WASM-portable)
  src/chunk.rs           Markdown-aware chunking with heading paths + line ranges
  src/tokenize.rs        BM25 terms (Unicode, camelCase split, no stemming)
  src/bm25.rs            BM25 served from serialized bytes
  src/quant.rs, vectors.rs  binary codes (candidates) + int8 rescoring
  src/fuse.rs            reciprocal rank fusion
  src/index/             generations, manifest, writer (commit protocol), reader, mmap storage
  tests/                 integration (index.rs) and proptest suites (properties.rs)
  benches/search.rs      criterion benchmarks (numbers quoted in README)
crates/notelight-py/     thin PyO3 bindings -> python module notelight._core
crates/notelight-py/src/ lib.rs, errors.rs (core Error -> Python exceptions), chunking.rs, index.rs
python/notelight/        Python package: embed.py, ingest.py, documents.py (PDF/DOCX), indexer.py, search.py, paths.py, cli.py, mcp_server.py,
                         download.py, eval/ (datasets, metrics, NumPy baseline, runner, report)
scripts/                 make_demo_gif.py (README demo GIF from a real MCP session)
tests/                   pytest
docs/                    design-notes.md, index-format.md (spec), results.md (BEIR numbers), learning/
```

## Commands (PowerShell)

```powershell
.venv\Scripts\activate
maturin develop --release             # rebuild the Rust extension after Rust changes
pytest; ruff check .; ruff format --check .; mypy
cargo test -p notelight-core          # unit + integration + property tests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
cargo bench -p notelight-core --bench search
$env:PROPTEST_CASES=20000; cargo test --release -p notelight-core --test properties
```

The `_core` extension needs the MSVC C++ build tools (Desktop development with
C++, x64/x86 build tools, Windows SDK).

## Conventions (enforced by CI)

- Clippy `pedantic` with `-D warnings`. The CI lint job is pinned to Rust
  1.99.0, so bump it deliberately and fix new lints in the same change.
- `unsafe_code = "deny"`. The only exception is `index/storage.rs`
  (`Mmap::map`), which carries a `// SAFETY:` argument. Keep it that way.
- Binary formats are decoded with `from_le_bytes` at validated offsets, never
  pointer casts. Validate fully on open, using overflow-safe (u64/u128) layout
  math, so query paths can index freely.
- Any change to an on-disk layout bumps `SCHEMA_VERSION` and updates
  `docs/index-format.md` in the same commit.
- New logic gets unit tests. Invariants get proptests against a naive reference.
- Public items need docs (`missing_docs`), plus `# Errors` and `# Panics`
  sections. Docs build with `RUSTDOCFLAGS=-D warnings`.
- Python: strict mypy, ruff, type stubs in `python/notelight/_core.pyi` for
  everything exported from Rust.
- README claims must be backed by tests or benchmarks. If a number changes,
  update the README.
- Commit messages: a summary line plus a bullet body describing what changed and why.

## Status

| # | Milestone | State |
|---|---|---|
| 0 | Scaffold, PyO3, CI on 3 OSes | done |
| 1 | Rust core: chunking, BM25, quantized search, RRF, index format | done (independently reviewed; 8 findings fixed) |
| 2 | Python package: bindings, embedder, incremental indexer, CLI | done |
| 3 | Eval: BEIR (SciFact, NFCorpus, FiQA) recall@10/nDCG@10 for f32, int8, binary, binary+rescore, ternary+rescore, BM25, hybrid; Rust vs NumPy baseline | done |
| 4 | MCP server (official Python SDK, stdio): search, read_context, list_sources, index_status; demo GIF | done |
| 5a | Wheels via maturin-action (smoke-tested per OS), sdist, CHANGELOG, PyPI 0.1.0 | release prep done; tag + publish pending the owner |
| 5b | Localhost web UI (FastAPI + one page), PDF/DOCX extras (0.2) | **next** |
| v2 | Same core compiled to wasm32 + TypeScript browser app | |

## Milestone 2 notes (for whoever picks up M3)

- Design decisions (caches, lock, embedder identity, pinned model) are in
  `docs/design-notes.md`, "Milestone 2 decisions".
- `NOTELIGHT_HOME` relocates indexes and models; `NOTELIGHT_MODEL_CACHE`
  relocates only models. Tests set `NOTELIGHT_HOME` per test (autouse fixture).
- `pytest` never downloads. `NOTELIGHT_MODEL_TESTS=1` enables the real
  bge-small test (set `NOTELIGHT_MODEL_CACHE` to reuse a downloaded model).
- bge-small is pinned to revision `5c38ec7c405e...` with SHA-256 per file in
  `embed.py`. Changing the model or revision changes `model_id`, which forces a
  re-embed and makes old indexes refuse queries from the new embedder.
- `IndexReader.search_bm25` / `search_vector` are exposed separately for the
  M3 eval harness, which needs each retriever on its own.

## Milestone 3 notes (for whoever picks up M4)

- `notelight eval` writes the tables in `docs/results.md` and the README.
  Re-run it (cached embeddings make it ~1 minute) and update both whenever
  search code changes. Datasets and embeddings live in
  `<cache>/eval/`; tests use synthetic BEIR folders and file:// URLs.
- `IndexReader.search_vector(..., mode=...)` takes one of `VECTOR_MODES`
  (`int8`, `binary`, `binary+int8`, `ternary+int8`). `ternary+int8` is the
  default (it matches exhaustive int8 quality); its masks are stored in
  `vectors.bin` (schema version 2), computed from the int8 codes.
- Open follow-up from the results: a vectorized `int8_dot` (NumPy BLAS wins
  the exhaustive int8 scan 5x).

## Milestone 4 notes (for whoever picks up M5)

- `notelight mcp` lives in `mcp_server.py`: `NotesService` is plain Python
  (testable without MCP); `create_server` wraps it in mcp 2.x `MCPServer`.
  Decisions are in `docs/design-notes.md`, "Milestone 4 decisions".
- The server is read-only and never takes the writer lock. It caches one
  `Searcher` per folder and reopens it when `Searcher.is_live()` is false.
- `read_context` accepts only paths listed in the index's `files` cache table
  (`search.indexed_files`); keep that as the single security check.
- Tests use `mcp.Client(server)` in-process with `@pytest.mark.anyio`; one
  test runs `python -m notelight.cli mcp` over real stdio.
- The README demo GIF (`docs/assets/mcp-demo.gif`) is rendered by
  `scripts/make_demo_gif.py` from a real stdio session over `docs/` (needs
  Pillow and the cached bge-small). Re-run it when search results change.

## Releasing (milestone 5a)

- The version lives only in `Cargo.toml` (`workspace.package.version`);
  maturin copies it into the Python metadata, and `notelight.__version__`
  reads it from the extension. Bump it there, add a `CHANGELOG.md` section,
  then tag `vX.Y.Z`.
- `.github/workflows/wheels.yml` builds abi3 wheels (Linux x86_64/aarch64,
  Windows x64, macOS arm64/x86_64) and an sdist, installs every wheel into a
  clean venv without Rust on Python 3.10 and 3.14 (3.13 on Intel macOS, where
  onnxruntime stops) and runs `notelight index/search --embedder hashing`,
  and builds the sdist from source. On a `v*` tag it checks the tag matches
  the version and publishes via PyPI trusted publishing (environment `pypi`).
  It also runs on `workflow_dispatch` and on PRs touching packaging files.
- The README is the PyPI long description, so its links must be absolute
  GitHub URLs (in-page `#anchors` are fine). `twine check` the dist files.
