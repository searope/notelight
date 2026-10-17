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
crates/notelight-wasm/   wasm-bindgen bindings (v2): Index over in-memory files, chunkText, IndexWriter; tests/node.rs runs under Node
                         npm/ (npm README, Node entry node.js, smoke.mjs + smoke-types.ts)
python/notelight/        Python package: embed.py, ingest.py, documents.py (PDF/DOCX), indexer.py, search.py, paths.py, cli.py, mcp_server.py,
                         download.py, eval/ (datasets, notes benchmark, metrics, NumPy baseline, runner, report)
                         web/ (`notelight serve`: FastAPI over NotesService + static/ page)
scripts/                 make_demo_gif.py (README demo GIF from a real MCP session),
                         build_wasm.py (-> web-demo/pkg), vendor_ort.py + vendor_pdfjs.py (-> web-demo/vendor),
                         make_pdf_fixtures.py (PDFs for web-demo/tests),
                         build_npm.py (-> dist/npm, npm package), smoke_npm.py (installs the tarball, searches),
                         serve_demo.py, make_sample_index.py (+ parity fixture),
                         make_tokenizer_fixture.py, make_indexing_fixture.py, wasi_run.mjs
web-demo/                static browser app: wasm build, embed.js + tokenizer.js (bge-small via
                         onnxruntime-web), ingest.js + indexer.js + store.js + index-worker.js
                         (index a folder in the browser, OPFS/IndexedDB), documents.js (PDF text
                         via pdf.js), sample index, tests/
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

# WebAssembly (v2); runners are set in .cargo/config.toml
rustup target add wasm32-unknown-unknown wasm32-wasip1
cargo install wasm-bindgen-cli --version 0.2.129 --locked   # must equal Cargo.lock's wasm-bindgen
cargo test --target wasm32-unknown-unknown -p notelight-wasm --test node   # JS API under Node
cargo test --target wasm32-wasip1 -p notelight-core --lib                  # core unit tests as wasm32
python scripts/build_wasm.py                                              # web-demo/pkg + size report
python scripts/vendor_ort.py                                              # onnxruntime-web -> web-demo/vendor
python scripts/vendor_pdfjs.py                                            # pdf.js -> web-demo/vendor/pdfjs
node --test "web-demo/tests/*.test.mjs"                                   # browser embedder parity (Node)
python scripts/serve_demo.py [--isolated]                                 # demo on 127.0.0.1:8000
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
| 5a | Wheels via maturin-action (smoke-tested per OS), sdist, CHANGELOG, PyPI 0.1.0 | done; **0.1.0, 0.1.1 and 0.2.0 published to PyPI (2026-10-16)** (tags `v0.1.0`, `v0.1.1`, `v0.2.0`; version lives in Cargo.toml `[workspace.package]`; trusted publishing via the `pypi` environment, which needs the owner's approval per release) |
| 5b | Localhost web UI, PDF/DOCX extras, phrase ranking, CLI highlighting | done (PRs #5-#8); ships in 0.1.0 |
| v2 | Same core compiled to wasm32 + browser app | spike done; query embeddings in the browser done; indexing a folder in the browser done (worker, OPFS/IndexedDB, incremental, parity with the CLI); npm package `notelight-wasm` **0.2.0 published to npm (2026-10-17)**, by hand, no provenance (see "Releasing to npm") |

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
- The notes benchmark (`notelight eval notes`, data in
  `python/notelight/eval/notes/`) is the check for note-style queries. Tune on
  `--split tune`; run `--split heldout` once per decision, for settings chosen
  beforehand. Never edit its labels after seeing results; add new queries in
  a new version instead. Its corpus is pinned to commit `6a9d410`
  (`scripts/make_notes_corpus.py --check`), so later doc edits do not move
  the labels. Decisions and numbers are in `docs/design-notes.md`, "Notes
  benchmark decisions".

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

## v2 spike notes (for whoever builds the browser app)

- The core is unchanged for wasm32; keep it that way: no `cfg` in the core
  beyond `storage.rs`, and no JS types. Browser concerns go in
  `crates/notelight-wasm`. Decisions, measurements and the list of what the
  app needs next are in `docs/design-notes.md`, "v2 spike: the core in the
  browser".
- Browsers build and open generations from bytes:
  `IndexWriter::build` -> `GenerationFiles`, `IndexReader::from_parts`.
- Every `#[wasm_bindgen]` method has a plain-Rust twin (`hits_bm25`, `hits`,
  `record`) for native tests; creating a `JsValue` natively panics.
- `web-demo/sample-index/` is a committed fixture read by `tests/node.rs`.
  Rebuild it with `scripts/make_sample_index.py` after a schema change.
- The wasm-bindgen crate and CLI versions must match exactly; bump both together
  (CI installs the CLI version from `Cargo.lock`).
- Query embeddings: `web-demo/embed.js` mirrors `embed.py` (spec, SHA-256 on
  every load, prefix, CLS, L2, `modelId`); change both together, then re-run
  `scripts/make_sample_index.py` (rebuilds the sample with bge-small and the
  parity fixture). `tokenizer.js` must stay token-identical to HF
  `tokenizers`: run `scripts/make_tokenizer_fixture.py --sweep` after any
  change to it. onnxruntime-web is pinned in `scripts/vendor_ort.py` (tarball
  SHA-512 + per-file SHA-256); bumping it means new hashes and new
  measurements. Decisions and numbers: design notes, "v2: query embeddings in
  the browser".
- Indexing in the browser: `ingest.js` mirrors `ingest.py` (and ports
  pathspec's gitignore matching) and `indexer.js` mirrors `indexer.py`'s
  caches; change them together with the Python side, then re-run
  `scripts/make_indexing_fixture.py` (the parity fixture in
  `web-demo/tests/fixtures/indexing.json`). Browser storage layout
  (`meta.json`, `embeddings.bin`) is in the design notes, "v2: indexing in
  the browser"; generation files are the normal index format.
- PDFs in the browser: `documents.js` mirrors `documents.py`'s page layout
  (form feeds, `page_starts`, `page_range`, skip reasons) with pdf.js, which
  is pinned in `scripts/vendor_pdfjs.py` and runs inside the indexing
  worker. Its text differs from pypdf's by design; don't try to make
  browser PDF chunks match the CLI's. Bumping pdf.js means new hashes in the
  script and `PDFJS_VERSION` in `documents.js` (a test checks they match),
  which re-extracts PDFs in existing browser indexes. Decisions and numbers:
  design notes, "v2: PDFs in the browser".

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

## Releasing to npm

- The npm package `notelight-wasm` is the demo's `--target web` build plus
  `crates/notelight-wasm/npm/` (README, `node.js` Node entry). Its version is
  the Cargo version; `scripts/build_npm.py` generates package.json and fails
  if `npm pack --dry-run` lists anything but the nine expected files.
  Decisions: design notes, "v2: the npm package".
- `.github/workflows/npm.yml` builds, smoke-tests (3 OSes x Node 22/24) and,
  on an `npm-vX.Y.Z` tag (not `vX.Y.Z`, which is PyPI), publishes the tested
  tarball from the `npm` environment via trusted publishing (OIDC, no token).
- State: 0.2.0 is on npm (2026-10-17), published by hand from the CI-built
  tarball (same shasum), so without provenance. The package also lists a
  `0.0.0-stage` placeholder from its creation; deprecate it with
  `npm deprecate notelight-wasm@0.0.0-stage "placeholder; use 0.2.0 or later"`.
  The `npm` environment exists with the tag rule `npm-v*`; give it a
  required reviewer (the owner) so a tag waits for approval. What happened
  and why: design notes, "v2: the npm package", "Versions and publishing".
- Publish only through the workflow; never `npm publish` by hand, or the
  tag's run fails ("cannot publish over") and the version has no
  provenance. Owner steps for each release:
  1. Check the trusted publisher: `npm trust list notelight-wasm`. One that
     has never completed a publish expires after 2 days, so the one made for
     0.2.0 is likely gone. Re-create it no more than 2 days before tagging:
     `npm trust github notelight-wasm --repo searope/notelight --file npm.yml --env npm --allow-publish --yes`
     (npm 11.15+, 2FA), or on npmjs.com: package > Settings > Trusted
     publishing > GitHub Actions, user `searope`, repository `notelight`,
     workflow `npm.yml`, environment `npm`.
  2. Bump the Cargo version and CHANGELOG as for PyPI and merge. Then tag
     `origin/main` by name, which works from any worktree (`git switch main`
     fails in a worktree while main is checked out elsewhere, and a tag on a
     branch commit fails the workflow's version check):
     ```powershell
     git fetch origin
     git tag -a npm-vX.Y.Z -m "notelight-wasm X.Y.Z" origin/main
     git push origin npm-vX.Y.Z
     ```
     and approve the `npm` deployment. A tag that failed before publishing
     can be moved: `git push origin :refs/tags/npm-vX.Y.Z; git tag -d npm-vX.Y.Z`,
     then tag again.
  3. After the first trusted publish works: package > Settings > Publishing
     access > "Require two-factor authentication and disallow tokens", and
     `npm logout`.
