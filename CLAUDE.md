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
python/notelight/        Python package (CLI via Typer; M2 adds embedder/indexer)
tests/                   pytest
docs/                    design-notes.md, index-format.md (spec), learning/
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
| 2 | Python package: bindings, embedder, incremental indexer, CLI | **next** |
| 3 | Eval: BEIR (SciFact, NFCorpus, FiQA) recall@10/nDCG@10 for f32, int8, binary, binary+rescore, ternary+rescore, BM25, hybrid; Rust vs NumPy baseline | |
| 4 | MCP server (official Python SDK, stdio): search, read_context, list_sources, index_status; demo GIF | |
| 5 | Localhost web UI (FastAPI + one page), PDF/DOCX extras, wheels via maturin-action, PyPI | |
| v2 | Same core compiled to wasm32 + TypeScript browser app | |

## Milestone 2 plan

1. **Bindings** (`crates/notelight-py`): expose `chunk_text(text, max_chars, markdown)`,
   `IndexWriter(dim, embedder, metadata).add(record, embedding: np.ndarray | None)`
   and `.commit(root)`, and `IndexReader.open(root)` with `.search(text, vec, k, ...)`,
   `.chunk(id)`, `.manifest`. Map `notelight_core::Error` variants to Python
   exceptions (`SchemaError` means "rebuild the index"). Release the GIL in
   commit and search. Update `_core.pyi`.
2. **Embedder** (`python/notelight/embed.py`): ONNX Runtime plus HF `tokenizers`,
   default `BAAI/bge-small-en-v1.5` (384-dim, MIT). Pin the revision and
   SHA-256, download once into a platformdirs cache, and add an `--offline`
   flag. bge uses CLS pooling plus L2-normalize, and queries get the bge query
   instruction prefix. Use a `Protocol` so tests use a deterministic fake
   embedder (no network in tests).
3. **Ingest**: walk a folder, skip hidden, VCS and build dirs plus anything in
   `.notelightignore`. Read `.md` and `.txt` as UTF-8, with a fallback and
   skip-on-binary. Use relative `/` paths.
4. **Indexer**: per-file content hash, and an embedding cache keyed by
   `sha256(model_id + chunk_text)` (SQLite in the index dir), so re-indexing
   embeds only changed chunks. Each sync writes one new generation through the
   core writer. Use a lock file (`filelock`) for the single-writer rule. Store
   indexes in the user data dir, keyed by a hash of the folder path, and refuse
   to search with a different embedder than the manifest's.
5. **CLI**: `notelight index <folder>`, `search "<query>" [--k] [--json]`,
   `status`. Show `path:line_start-line_end`, the heading path and a snippet.
6. **Tests**: pytest end-to-end on a fixture folder with the fake embedder,
   covering incremental re-index, the embedder-mismatch error, and Windows
   path handling.
7. **Docs**: README usage section, design notes for the cache and lock, and
   `docs/learning/m2-rust-walkthrough.md` (PyO3 classes, error mapping, GIL).
