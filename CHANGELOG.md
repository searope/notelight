# Changelog

All notable changes to notelight are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version may
change the CLI, the Python API or the index format; an index from an older
format is detected and rebuilt by `notelight index`.

## [Unreleased]

## [0.1.0] - unreleased

The first release on PyPI: `pip install notelight`. It covers milestones 0 to 5
of the [roadmap](README.md#roadmap).

### Added

- **Rust retrieval core** (`notelight-core`, no Python, ML or async
  dependencies):
  - Markdown-aware chunking (sections, paragraphs, code fences) with heading
    paths and exact line ranges for citations.
  - BM25 with Unicode-aware terms and camelCase splitting, served directly
    from the serialized index.
  - Quantized vector search: binary or ternary codes pick candidates with
    `popcount`, and int8 vectors re-score them. Ternary candidates with int8
    rescoring is the default; on BEIR it matches exhaustive int8 nDCG@10 to
    three decimals. The int8 dot product keeps 16 partial sums so the
    compiler can use SIMD.
  - Reciprocal rank fusion for hybrid (BM25 + vector) search.
  - A versioned, memory-mapped index format
    ([spec](docs/index-format.md), schema version 2): immutable generations
    with an atomic `CURRENT` swap, full validation on open, and recovery from
    interrupted commits, including on Windows.
  - Unit, integration and property tests against naive references, and
    criterion benchmarks.
- **Python package** (abi3 wheels for Python 3.10+, built with PyO3 and
  maturin):
  - `sync(folder, embedder)` and `Searcher(folder, embedder)`, with typed
    stubs for the Rust extension.
  - `OnnxEmbedder`: BAAI/bge-small-en-v1.5 on ONNX Runtime, pinned to one
    revision, verified by SHA-256 and downloaded once. `HashingEmbedder` for
    tests and model-free use.
  - An incremental indexer: only new or changed chunks are embedded, an
    unchanged folder never loads the model, and a lock file allows one writer
    per index. Indexes live in the user data directory (`NOTELIGHT_HOME`),
    never in the indexed folder.
  - Ingest of `.md`, `.markdown` and `.txt` in UTF-8 or Windows-1252, skipping
    hidden and build folders and anything in `.notelightignore`.
- **CLI**: `notelight index`, `search` (with `--json`), `status`, `mcp`,
  `serve`, `eval` and `version`; `--embedder bge-small|hashing|none` and
  `--offline`. On a terminal, `search` highlights the matched words and
  starts each snippet near the first match (`NO_COLOR` turns colour off).
- **Phrase ranking** in the core, for every interface: quoted phrases
  (`'"lock file" windows'`) rank chunks containing them first, and a chunk
  with all query terms in order gets a bonus worth one extra first place in
  fusion. Neutral on BEIR (within 0.002 nDCG@10); aimed at short note queries.
- **PDF and DOCX** as optional extras (`notelight[pdf]`, `[docx]`,
  `[documents]`; pypdf and python-docx). Citations stay `path:lines` of the
  extracted text, with PDF pages shown beside them; DOCX headings become
  heading paths. Without the extra, these files are skipped with a hint.
- **Localhost web UI** as an optional extra (`notelight[web]`,
  `notelight serve`): one page over the same service as the MCP server,
  bound to 127.0.0.1 with a Host-header check against DNS rebinding, no
  build step and nothing loaded from the network.
- **MCP server** (`notelight mcp`, official Python SDK, stdio) with four
  read-only tools: `search`, `read_context`, `list_sources` and
  `index_status`. `read_context` reads only files that are in the index.
- **Evaluation**: `notelight eval` reports recall@10, nDCG@10 and latency for
  every retrieval method on BEIR SciFact, NFCorpus and FiQA (datasets pinned
  by SHA-256), and checks the Rust searches against a NumPy implementation
  ([results](docs/results.md)).
- Docs: design notes, the index format spec, and a Rust walkthrough per
  milestone for readers coming from C# or Python.
- Prebuilt wheels for Linux x86_64 and aarch64, Windows x64 and macOS (Apple
  silicon and Intel), each installed and smoke-tested in CI before publishing.

### Fixed

Findings from an independent review of milestones 2 to 4, each with a
regression test:

- `notelight index <folder>` without `--embedder` keeps the embedder the
  index was built with, instead of switching a keyword-only or hashing index
  to bge-small (a 133 MB download and a full re-embed).
- Files the indexer skips (binary, undecodable, unreadable) are remembered, so
  `index_status` no longer reports them as new forever.
- An all-folder MCP `search` skips a folder whose index cannot be searched and
  lists it in a `skipped` field, instead of failing for every folder.
- MCP results from several folders are merged by per-folder rank, so
  keyword-only folders are no longer buried by raw fusion scores.
- Network errors while downloading the model raise `ModelNotAvailableError`
  with a hint instead of a traceback.
- A keyword-only sync no longer wipes the embedding cache.
- A query whose vector is all zeros (hashing embedder, punctuation-only
  query) falls back to BM25 instead of raising `ValueError`.
- `status` and the MCP server no longer crash when index files vanish during a
  concurrent sync.
- Changing the chunk size no longer empties the file cache before the new
  generation commits.

[Unreleased]: https://github.com/searope/notelight/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/searope/notelight/releases/tag/v0.1.0
