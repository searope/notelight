"""Incremental indexing of a folder.

`sync(folder, embedder)` brings the folder's index up to date:

1. Take the single-writer lock (`lock` file, via `filelock`).
2. Walk the folder. A file whose size and modification time match the cache
   is not read again; otherwise it is read and hashed (SHA-256).
3. Chunk each file once per content hash (chunk cache).
4. Embed only chunks whose text is not in the embedding cache, keyed by
   `sha256(model_id + text)`. Unchanged notes are never embedded twice, and a
   file that is renamed or copied reuses its vectors.
5. Write every chunk as one new generation through the Rust core's writer,
   which swaps it in atomically, then prune cache entries nothing refers to.

If nothing changed, no generation is written. All caches live in
`cache.sqlite` next to the index, see `notelight.paths`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import filelock
import numpy as np
import numpy.typing as npt

from notelight import _core, paths
from notelight.embed import Embedder
from notelight.errors import IndexBusyError
from notelight.ingest import SourceFile, decode, walk

__all__ = [
    "CACHE_FILE",
    "CORE_DIR",
    "DEFAULT_MAX_CHARS",
    "LOCK_FILE",
    "SOURCE_FILE",
    "SkippedFile",
    "SyncStats",
    "embedding_text",
    "sync",
    "writer_lock",
]

CORE_DIR = "index"
CACHE_FILE = "cache.sqlite"
LOCK_FILE = "lock"
SOURCE_FILE = "source.json"
DEFAULT_MAX_CHARS = 1200
CACHE_SCHEMA = "1"
EMBED_BATCH = 64

#: Called as `progress(stage, done, total)` while syncing.
ProgressCallback = Callable[[str, int, int], None]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (
    path     TEXT PRIMARY KEY,
    size     INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    doc_key  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    doc_key      TEXT NOT NULL,
    ord          INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    line_start   INTEGER NOT NULL,
    line_end     INTEGER NOT NULL,
    text         TEXT NOT NULL,
    PRIMARY KEY (doc_key, ord)
);
CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL);
"""


@dataclass(frozen=True)
class SkippedFile:
    """A file that was found but not indexed."""

    path: str
    reason: str


@dataclass
class SyncStats:
    """What a call to `sync` did."""

    folder: Path
    files: int = 0
    added: int = 0
    changed: int = 0
    removed: int = 0
    skipped: list[SkippedFile] = field(default_factory=list)
    chunks: int = 0
    embedded: int = 0
    generation: int | None = None
    up_to_date: bool = False
    seconds: float = 0.0


@dataclass(frozen=True)
class _Doc:
    """One indexed file in this sync."""

    file: SourceFile
    doc_key: str


@dataclass(frozen=True)
class _CachedChunk:
    heading_path: list[str]
    line_start: int
    line_end: int
    text: str


def embedding_text(heading_path: Sequence[str], text: str) -> str:
    """The string that is embedded for a chunk: its heading path, then its text.

    Headings give a short chunk the context it lacks on its own ("Install"
    under "Rust > Tooling").
    """
    if not heading_path:
        return text
    return " > ".join(heading_path) + "\n" + text


def _embedding_key(model_id: str, text: str) -> str:
    return hashlib.sha256(f"{model_id}\n{text}".encode()).hexdigest()


@contextlib.contextmanager
def writer_lock(index_dir: Path, timeout: float) -> Iterator[None]:
    """Hold the single-writer lock for `index_dir`.

    Raises:
        IndexBusyError: another process holds the lock for longer than `timeout`.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(index_dir / LOCK_FILE, timeout=timeout)
    try:
        lock.acquire()
    except filelock.Timeout as e:
        raise IndexBusyError(
            f"another notelight process is updating {index_dir}; try again when it finishes"
        ) from e
    try:
        yield
    finally:
        lock.release()


def _open_cache(index_dir: Path, chunker: str) -> sqlite3.Connection:
    db = sqlite3.connect(index_dir / CACHE_FILE)
    db.executescript(_SCHEMA)
    meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
    if meta.get("schema", CACHE_SCHEMA) != CACHE_SCHEMA or meta.get("chunker") != chunker:
        # Chunking settings changed: cached chunks no longer apply. Embeddings
        # stay valid because they are keyed by the exact text embedded.
        with db:
            db.execute("DELETE FROM chunks")
            db.execute("DELETE FROM files")
    with db:
        db.executemany(
            "INSERT OR REPLACE INTO meta VALUES (?, ?)",
            [("schema", CACHE_SCHEMA), ("chunker", chunker)],
        )
    return db


def _doc_key(content_sha: str, markdown: bool) -> str:
    # The same bytes chunk differently as .md and .txt, so the mode is part of the key.
    return f"{content_sha}:{'md' if markdown else 'txt'}"


def _scan(
    root: Path,
    db: sqlite3.Connection,
    max_chars: int,
    stats: SyncStats,
) -> list[_Doc]:
    """Walk the folder, refresh the chunk cache, and return the files to index."""
    known = {
        row[0]: (row[1], row[2], row[3])
        for row in db.execute("SELECT path, size, mtime_ns, doc_key FROM files")
    }
    cached_docs = {row[0] for row in db.execute("SELECT DISTINCT doc_key FROM chunks")}
    docs: list[_Doc] = []
    for f in walk(root):
        prior = known.get(f.rel_path)
        if prior is not None and prior[:2] == (f.size, f.mtime_ns) and prior[2] in cached_docs:
            docs.append(_Doc(f, prior[2]))
            continue
        try:
            data = f.abs_path.read_bytes()
        except OSError as e:
            stats.skipped.append(SkippedFile(f.rel_path, f"unreadable: {e.strerror or e}"))
            continue
        decoded = decode(data)
        if decoded.text is None:
            stats.skipped.append(SkippedFile(f.rel_path, decoded.skip_reason or "unreadable"))
            continue
        doc_key = _doc_key(hashlib.sha256(data).hexdigest(), f.is_markdown)
        if prior is None:
            stats.added += 1
        elif prior[2] != doc_key:
            stats.changed += 1
        if doc_key not in cached_docs:
            chunks = _core.chunk_text(decoded.text, max_chars=max_chars, markdown=f.is_markdown)
            with db:
                db.executemany(
                    "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            doc_key,
                            i,
                            json.dumps(c.heading_path),
                            c.line_start,
                            c.line_end,
                            c.text,
                        )
                        for i, c in enumerate(chunks)
                    ],
                )
            cached_docs.add(doc_key)
        docs.append(_Doc(f, doc_key))
    seen = {d.file.rel_path for d in docs}
    stats.removed = sum(1 for p in known if p not in seen)
    stats.files = len(docs)
    return docs


def _load_chunks(db: sqlite3.Connection, doc_key: str) -> list[_CachedChunk]:
    return [
        _CachedChunk(json.loads(r[0]), r[1], r[2], r[3])
        for r in db.execute(
            "SELECT heading_path, line_start, line_end, text FROM chunks "
            "WHERE doc_key = ? ORDER BY ord",
            (doc_key,),
        )
    ]


def _batched(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _cached_keys(db: sqlite3.Connection, keys: Iterable[str]) -> set[str]:
    found: set[str] = set()
    key_list = list(keys)
    for batch in _batched(key_list, 500):
        marks = ",".join("?" * len(batch))
        found.update(
            r[0] for r in db.execute(f"SELECT key FROM embeddings WHERE key IN ({marks})", batch)
        )
    return found


def _embed_missing(
    db: sqlite3.Connection,
    embedder: Embedder,
    texts: dict[str, str],
    progress: ProgressCallback | None,
) -> int:
    """Embed texts (by key) not yet in the cache. Commits after each batch, so
    an interrupted sync keeps the work it finished."""
    missing = sorted(set(texts) - _cached_keys(db, texts))
    for done, batch in enumerate(_batched(missing, EMBED_BATCH)):
        if progress is not None:
            progress("embed", done * EMBED_BATCH, len(missing))
        vectors = embedder.embed_documents([texts[k] for k in batch])
        if vectors.shape != (len(batch), embedder.dim):
            raise ValueError(
                f"embedder returned shape {vectors.shape}, expected {(len(batch), embedder.dim)}"
            )
        with db:
            db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                [
                    (k, np.asarray(v, dtype="<f4").tobytes())
                    for k, v in zip(batch, vectors, strict=True)
                ],
            )
    if progress is not None and missing:
        progress("embed", len(missing), len(missing))
    return len(missing)


def _load_vectors(
    db: sqlite3.Connection, keys: Sequence[str], dim: int
) -> dict[str, npt.NDArray[np.float32]]:
    out: dict[str, npt.NDArray[np.float32]] = {}
    for batch in _batched(list(dict.fromkeys(keys)), 500):
        marks = ",".join("?" * len(batch))
        for key, blob in db.execute(
            f"SELECT key, vector FROM embeddings WHERE key IN ({marks})", batch
        ):
            vec = np.frombuffer(blob, dtype="<f4")
            if vec.shape != (dim,):
                raise ValueError(f"cached embedding {key} has {vec.size} dims, expected {dim}")
            out[key] = vec.astype(np.float32)
    return out


def _prune(db: sqlite3.Connection, docs: list[_Doc], live_keys: set[str]) -> None:
    """Drop cache rows for files and chunks that are no longer in the folder."""
    with db:
        db.execute("CREATE TEMP TABLE IF NOT EXISTS live_docs (doc_key TEXT PRIMARY KEY)")
        db.execute("CREATE TEMP TABLE IF NOT EXISTS live_keys (key TEXT PRIMARY KEY)")
        db.execute("DELETE FROM live_docs")
        db.execute("DELETE FROM live_keys")
        db.executemany("INSERT OR IGNORE INTO live_docs VALUES (?)", [(d.doc_key,) for d in docs])
        db.executemany("INSERT OR IGNORE INTO live_keys VALUES (?)", [(k,) for k in live_keys])
        db.execute("DELETE FROM chunks WHERE doc_key NOT IN (SELECT doc_key FROM live_docs)")
        db.execute("DELETE FROM embeddings WHERE key NOT IN (SELECT key FROM live_keys)")


def _record_files(db: sqlite3.Connection, docs: list[_Doc]) -> None:
    """Remember each file's size, mtime and content key for the next scan."""
    with db:
        db.execute("DELETE FROM files")
        db.executemany(
            "INSERT INTO files VALUES (?, ?, ?, ?)",
            [(d.file.rel_path, d.file.size, d.file.mtime_ns, d.doc_key) for d in docs],
        )


def _index_is_current(core_root: Path, embedder: Embedder | None, chunker: str) -> bool:
    if not _core.index_exists(core_root):
        return False
    try:
        manifest = _core.IndexReader.open(core_root).manifest
    except (OSError, _core.NotelightError):
        return False  # unreadable or old schema: rebuild
    model_id = embedder.model_id if embedder is not None else None
    return bool(
        manifest.get("embedder") == model_id
        and manifest.get("metadata", {}).get("chunker") == json.loads(chunker)
    )


def sync(
    folder: str | Path,
    embedder: Embedder | None,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    lock_timeout: float = 10.0,
    progress: ProgressCallback | None = None,
) -> SyncStats:
    """Create or update the index for `folder`.

    `embedder=None` builds a keyword-only (BM25) index.

    Raises:
        NotADirectoryError: `folder` is not a directory.
        IndexBusyError: another process is updating the same index.
    """
    started = time.perf_counter()
    root = paths.normalize_folder(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"not a folder: {root}")
    index_dir = paths.index_dir_for(root)
    core_root = index_dir / CORE_DIR
    chunker = json.dumps({"max_chars": max_chars}, sort_keys=True)
    stats = SyncStats(folder=root)

    with writer_lock(index_dir, lock_timeout):
        (index_dir / SOURCE_FILE).write_text(
            json.dumps({"folder": str(root)}, indent=2) + "\n", encoding="utf-8"
        )
        db = _open_cache(index_dir, chunker)
        try:
            if progress is not None:
                progress("scan", 0, 0)
            docs = _scan(root, db, max_chars, stats)
            chunks = {d.doc_key: _load_chunks(db, d.doc_key) for d in docs}
            stats.chunks = sum(len(c) for c in chunks.values())

            unchanged = not (stats.added or stats.changed or stats.removed)
            if unchanged and _index_is_current(core_root, embedder, chunker):
                _record_files(db, docs)  # touched-but-identical files: skip re-hashing next time
                stats.up_to_date = True
                stats.seconds = time.perf_counter() - started
                return stats

            texts: dict[str, str] = {}
            chunk_keys: dict[tuple[str, int], str] = {}
            if embedder is not None:
                for doc_key, doc_chunks in chunks.items():
                    for i, c in enumerate(doc_chunks):
                        text = embedding_text(c.heading_path, c.text)
                        key = _embedding_key(embedder.model_id, text)
                        texts[key] = text
                        chunk_keys[(doc_key, i)] = key
                stats.embedded = _embed_missing(db, embedder, texts, progress)
            vectors = _load_vectors(db, list(texts), embedder.dim) if embedder is not None else {}

            writer = _core.IndexWriter(
                embedder.dim if embedder is not None else 0,
                embedder.model_id if embedder is not None else None,
                {"source": str(root), "chunker": json.loads(chunker)},
            )
            if progress is not None:
                progress("write", 0, stats.chunks)
            for d in docs:
                for i, c in enumerate(chunks[d.doc_key]):
                    record = _core.ChunkRecord(
                        d.file.rel_path, c.heading_path, c.line_start, c.line_end, c.text
                    )
                    vec = vectors[chunk_keys[(d.doc_key, i)]] if embedder is not None else None
                    writer.add(record, vec)
            info = writer.commit(core_root)
            stats.generation = info.generation
            _record_files(db, docs)
            _prune(db, docs, set(texts))
        finally:
            db.close()
    stats.seconds = time.perf_counter() - started
    return stats
