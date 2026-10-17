"""Write the fixture that checks the browser indexer against the Python one.

    python scripts/make_indexing_fixture.py

The fixture, web-demo/tests/fixtures/indexing.json, holds:

- a small notes folder (every file as base64, so line endings, BOMs and
  legacy encodings survive git), with the cases the walk must agree on:
  hidden and build folders, a .notelightignore with negation, binary and
  undecodable files, CRLF, a UTF-8 BOM, Windows-1252, upper-case suffixes,
  non-ASCII names and an empty file;
- what `notelight.indexer.sync` made of it with the default embedder and
  chunker: the chunks in id order with the vector each was given (from the
  embedding cache), the skipped files, and the SHA-256 of docs.bin and
  bm25.bin;
- queries embedded by `OnnxEmbedder`, with their hybrid and BM25 top 10;
- .notelightignore patterns and paths with pathspec's verdict, for the
  port in web-demo/ingest.js.

web-demo/tests/indexer.test.mjs writes the folder to disk, indexes it with
web-demo/indexer.js (the wasm core and the browser embedder, under Node) and
requires the same chunks, the same index bytes and the same top 10. Re-run
this script when the chunker, the walk or the index format changes.

The index is built in a temporary NOTELIGHT_HOME; the model comes from the
usual model cache (downloaded if missing).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web-demo" / "tests" / "fixtures" / "indexing.json"
K = 10

LONG_NOTE = "\n\n".join(
    [
        "# Sourdough log",
        "Notes from a year of baking bread at home, kept so the next loaf is better than the last.",
        "## Starter",
        "The starter lives in the fridge and gets fed once a week: equal weights of flour and "
        "water, about 50 grams each. Before baking it needs two feeds at room temperature, "
        "twelve hours apart, until it doubles in four to six hours. A starter that smells of "
        "acetone is hungry, not dead.",
        "## Hydration",
        "Seventy percent hydration is forgiving: 700 grams of water for a kilogram of flour. "
        "Above 78 percent the dough spreads unless the flour is strong. Whole wheat drinks "
        "more water, so add five percent for every fifth of the flour that is whole grain.",
        "## Bulk fermentation",
        "Bulk fermentation took five hours at 24 degrees with four sets of stretch and folds "
        "in the first two hours. The dough is ready when it has grown by half, jiggles, and "
        "shows bubbles on the sides of the container. Under-proofed loaves have a tight, "
        "gummy crumb with a few large holes; over-proofed ones spread flat and taste sour.",
        "## Baking",
        "Bake in a preheated Dutch oven at 250 degrees, lid on for 20 minutes, then lid off "
        "at 230 degrees for another 25 until the crust is deep brown. Let it cool for an "
        "hour before cutting, or the crumb turns gummy.",
        "## Troubleshooting",
        "Flat loaf: weak shaping or over-proofing. Pale crust: oven not hot enough, or no "
        "steam. Dense crumb: the starter was not active, or bulk fermentation was cut short. "
        "Large tunnels under the crust: under-proofed, or the scoring was too shallow.",
    ]
)

# Path -> bytes. Order does not matter; the walk sorts.
FILES: dict[str, bytes] = {
    "README.md": (
        b"# My notes\n\n"
        b"A folder of plain text notes, indexed with notelight.\n\n"
        b"## Layout\n\n"
        b"Journal entries live in `journal/`, project notes in `projects/`, and old "
        b"notes from the previous laptop in `legacy/`.\n\n"
        b'```sh\nnotelight index .\nnotelight search "how to back up"\n```\n'
    ),
    "baking/sourdough.md": LONG_NOTE.encode(),
    "journal/2026-10-01.md": (
        b"# Wednesday\n\n"
        b"Moved the backups to a second disk. The backup script copies the notes folder "
        b"every night at two, keeps seven daily and four weekly snapshots, and emails me "
        b"if a run fails.\n\n"
        b"## Reading\n\n"
        b"Finished the chapter on ownership and borrowing in the Rust book. A value has one "
        b"owner; references borrow it, and the borrow checker makes sure no reference "
        b"outlives the value.\n"
    ),
    # A UTF-8 BOM and Windows line endings.
    "journal/2026-10-02.markdown": (
        "\ufeff# Thursday\r\n\r\n"
        "Team meeting about the release: we agreed to ship the browser demo first and the "
        "npm package after it.\r\n\r\n"
        "## Action items\r\n\r\n"
        "- Write the changelog entry.\r\n"
        "- Measure indexing speed in Chrome.\r\n"
        "- Ask for a review before merging.\r\n"
    ).encode(),
    "projects/notelight/ideas.txt": (
        b"Ideas for the browser app.\n\n"
        b"Keep the index in the origin private file system so it survives a reload, and "
        b"fall back to IndexedDB where that is missing.\n\n"
        b"Embed only the chunks that changed: key the cache by the hash of the model id and "
        b"the text, like the command line tool does.\n\n"
        b"Never upload a file. The only request is the model download, checked by SHA-256.\n"
    ),
    "projects/garden.md": (
        b"# Garden plan\n\n"
        b"Tomatoes go in the sunny bed by the fence after the last frost, about mid May. "
        b"Basil next to them; it likes the same warmth. Water deeply twice a week rather "
        b"than a little every day.\n\n"
        b"## Compost\n\n"
        b"Turn the compost every two weeks. Brown material (leaves, cardboard) balances the "
        b"green (kitchen scraps, grass); a heap that smells is too wet or too green.\n"
    ),
    # Windows-1252: e acute (0xE9), euro sign (0x80), curly quotes (0x93, 0x94).
    "legacy/old-note.txt": (
        b"Caf\xe9 receipts from the trip to Lyon: 12\x80 for lunch, 4\x80 for coffee.\n\n"
        b"Remember the \x93bouchon\x94 near the old town; best quenelles we had.\n"
    ),
    # Not UTF-8, and 0x81 is undefined in Windows-1252: skipped.
    "legacy/broken.txt": b"bad bytes \x81\x8d here\n",
    # A NUL byte: binary, skipped.
    "legacy/blob.md": b"# Not text\x00\x01\x02\n",
    "notes/UPPER.MD": b"# Shouting\n\nUpper-case suffixes are still Markdown.\n",
    "notes/empty.md": b"",
    "notes/keep.tmp.md": b"# Kept\n\nRe-included by a negated pattern in .notelightignore.\n",
    "notes/scratch.tmp.md": b"# Scratch\n\nIgnored by the *.tmp.md pattern.\n",
    "notes/data.csv": b"a,b\n1,2\n",
    "über/Ärger.md": "# Ärger\n\nNon-ASCII names sort by code point, as in Python.\n".encode(),
    "über/Zebra.md": b"# Zebra\n\nUpper-case Z sorts before lower-case letters.\n",
    "über/apple.md": b"# Apple\n\nLower-case a sorts after upper-case Z.\n",
    "drafts/wip.md": b"# Draft\n\nIgnored: drafts/ is in .notelightignore.\n",
    "archive/2019/drafts/old.md": b"# Old draft\n\nAlso ignored: drafts/ matches at any depth.\n",
    "archive/2019/summary.md": b"# 2019\n\nThe year we planted the first garden.\n",
    "root-only.md": b"# Root only\n\nIgnored by the anchored pattern /root-only.md.\n",
    "nested/root-only.md": b"# Nested\n\nNot ignored: an anchored pattern matches at the root.\n",
    ".hidden/secret.md": b"# Hidden\n\nHidden folders are skipped.\n",
    ".dotfile.md": b"# Dotfile\n\nHidden files are skipped.\n",
    "node_modules/pkg/readme.md": b"# Package\n\nDependency folders are skipped.\n",
    "build/out.md": b"# Build output\n\nBuild folders are skipped.\n",
    ".notelightignore": (
        b"# Not part of the notes\ndrafts/\n*.tmp.md\n!keep.tmp.md\n/root-only.md\n\n"
    ),
}

QUERIES = (
    "how long should bulk fermentation take",
    "why is my bread dense",
    "backup schedule",
    "rust borrow checker",
    "what did we decide about the release",
    "where is the index stored in the browser",
    "when to plant tomatoes",
    "compost smells",
    "coffee in Lyon",
    "Dutch oven temperature",
)

# .notelightignore pattern sets and paths for the gitignore port. Paths ending
# in "/" are directories, as the walk asks about them.
IGNORE_PATTERNS = [
    "# comment",
    "",
    "*.log",
    "!important.log",
    "build/",
    "/root.txt",
    "docs/**/*.tmp",
    "**/cache",
    "a/**/b",
    "lit\\#eral",
    "\\!bang",
    "trailing   ",
    "escaped\\ ",
    "[abc]x.md",
    "[!d-f]y.md",
    "[]]z.md",
    "q?.md",
    "dir/*",
    "!dir/keep.md",
    "excluded/",
    "!excluded/inner.md",
    "bad[range",
]
# "*/" and "**/" match everything below the root, so they get their own set.
IGNORE_PATTERNS_DEEP = ["*/", "!keep/", "!**/keep.md", "top.md"]
IGNORE_PATHS = [
    "a.log",
    "sub/a.log",
    "important.log",
    "sub/important.log",
    "build/",
    "src/build/",
    "build",
    "build/x.md",
    "root.txt",
    "sub/root.txt",
    "docs/x.tmp",
    "docs/a/b/x.tmp",
    "other/docs/x.tmp",
    "cache",
    "x/y/cache",
    "x/y/cache/",
    "x/cache/z.md",
    "a/b",
    "a/x/y/b",
    "a/x/b/",
    "c/a/b",
    "lit#eral",
    "!bang",
    "trailing",
    "trailing   ",
    "escaped ",
    "escaped",
    "ax.md",
    "dx.md",
    "ay.md",
    "ey.md",
    "]z.md",
    "q1.md",
    "q12.md",
    "dir/one.md",
    "dir/keep.md",
    "dir/sub/",
    "dir/sub/two.md",
    "excluded/inner.md",
    "excluded/other.md",
    "excluded/",
    "plain.md",
    "folder/",
    "folder/file.md",
    "bad[range",
    "top.md",
    "x/top.md",
    "keep/",
    "keep/a.md",
    "x/keep.md",
    "x/other.md",
]


def build_folder(root: Path) -> None:
    for rel, data in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def ignore_sets() -> list[dict[str, object]]:
    import pathspec

    sets = []
    for patterns in (IGNORE_PATTERNS, IGNORE_PATTERNS_DEEP):
        spec = pathspec.GitIgnoreSpec.from_lines(patterns)
        cases = [{"path": p, "ignored": spec.match_file(p)} for p in IGNORE_PATHS]
        sets.append({"patterns": patterns, "cases": cases})
    return sets


def indexed_files(index_dir: Path) -> list[str]:
    """The files the indexer recorded, in walk order (its `files` cache table)."""
    from notelight.indexer import CACHE_FILE

    db = sqlite3.connect(index_dir / CACHE_FILE)
    try:
        return [row[0] for row in db.execute("SELECT path FROM files ORDER BY rowid")]
    finally:
        db.close()


def chunk_vectors(index_dir: Path, model_id: str, chunks: list[dict[str, object]]) -> list[str]:
    """Each chunk's vector as the indexer stored it, base64 little-endian float32."""
    from notelight.indexer import CACHE_FILE, embedding_text

    db = sqlite3.connect(index_dir / CACHE_FILE)
    try:
        out = []
        for c in chunks:
            text = embedding_text(c["headingPath"], c["text"])  # type: ignore[arg-type]
            key = hashlib.sha256(f"{model_id}\n{text}".encode()).hexdigest()
            (blob,) = db.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
            out.append(base64.b64encode(blob).decode("ascii"))
        return out
    finally:
        db.close()


def main() -> None:
    from notelight import _core, paths
    from notelight.embed import OnnxEmbedder
    from notelight.indexer import CORE_DIR, sync
    from notelight.search import Searcher

    models = str(paths.model_cache_dir())
    with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as tmp:
        os.environ.update({"NOTELIGHT_HOME": home, "NOTELIGHT_MODEL_CACHE": models})
        folder = Path(tmp) / "notes"
        build_folder(folder)
        embedder = OnnxEmbedder()
        stats = sync(folder, embedder)
        core = paths.index_dir_for(folder) / CORE_DIR
        reader = _core.IndexReader.open(core)
        gen = reader.dir
        chunks = []
        for i in range(len(reader)):
            c = reader.chunk(i)
            chunks.append(
                {
                    "path": c.path,
                    "headingPath": list(c.heading_path),
                    "lineStart": c.line_start,
                    "lineEnd": c.line_end,
                    "text": c.text,
                }
            )
        searcher = Searcher(folder, embedder)
        queries = []
        for text in QUERIES:
            vector = embedder.embed_query(text).astype("<f4")
            queries.append(
                {
                    "text": text,
                    "vector": base64.b64encode(vector.tobytes()).decode("ascii"),
                    "hybrid": [h.chunk_id for h in searcher.search(text, k=K)],
                    "bm25": [i for i, _ in reader.search_bm25(text, K)],
                }
            )
        sha = {
            name: hashlib.sha256((gen / name).read_bytes()).hexdigest()
            for name in ("docs.bin", "bm25.bin")
        }
        fixture = {
            "comment": "Written by scripts/make_indexing_fixture.py: a notes folder (base64 "
            "bytes), what notelight.indexer.sync made of it with bge-small, query vectors "
            "(base64 little-endian float32) with their top-k chunk ids, and pathspec "
            "verdicts for .notelightignore patterns.",
            "model_id": embedder.model_id,
            "max_chars": 1200,
            "k": K,
            "files": {
                rel: base64.b64encode(data).decode("ascii") for rel, data in sorted(FILES.items())
            },
            "skipped": sorted([s.path, s.reason] for s in stats.skipped),
            "indexed_files": indexed_files(paths.index_dir_for(folder)),
            "chunks": chunks,
            "chunk_vectors": chunk_vectors(paths.index_dir_for(folder), embedder.model_id, chunks),
            "sha256": sha,
            "queries": queries,
            "ignore": ignore_sets(),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(
        f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KiB): "
        f"{len(FILES)} files, {stats.files} indexed, {len(chunks)} chunks"
    )


if __name__ == "__main__":
    main()
