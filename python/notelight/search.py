"""Search an indexed folder."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from notelight import _core, paths
from notelight.embed import Embedder
from notelight.errors import EmbedderMismatchError, IndexNotFoundError
from notelight.indexer import CACHE_FILE, CORE_DIR, SOURCE_FILE
from notelight.ingest import walk

__all__ = [
    "CURRENT_FILE",
    "IndexInfo",
    "SearchResult",
    "Searcher",
    "Staleness",
    "check_staleness",
    "describe_index",
    "indexed_files",
    "indexed_folders",
    "list_indexes",
    "snippet",
]

#: The file in an index root that names the live generation (see docs/index-format.md).
CURRENT_FILE = "CURRENT"


@dataclass(frozen=True)
class SearchResult:
    """One search hit, ready to show or to hand to an agent."""

    path: str
    line_start: int
    line_end: int
    heading_path: list[str]
    text: str
    score: float
    bm25_rank: int | None
    vector_rank: int | None
    chunk_id: int

    @property
    def location(self) -> str:
        """`path:start-end`, the citation format used throughout notelight."""
        return f"{self.path}:{self.line_start}-{self.line_end}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, including `location`."""
        return {**asdict(self), "location": self.location}


def snippet(text: str, width: int = 240) -> str:
    """The first `width` characters of `text` on one line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1].rstrip() + "…"


class Searcher:
    """An open index for one folder. Reuse it for many queries.

    The embedder must be the one the index was built with; pass `None` to
    search a keyword-only index, or to run BM25 alone on any index.

    Raises:
        IndexNotFoundError: the folder has not been indexed.
        EmbedderMismatchError: `embedder` differs from the index's embedder.
        IndexFormatError: the index is corrupt or from an incompatible version.
    """

    def __init__(self, folder: str | Path, embedder: Embedder | None) -> None:
        self.folder = paths.normalize_folder(folder)
        root = paths.index_dir_for(self.folder) / CORE_DIR
        if not _core.index_exists(root):
            raise IndexNotFoundError(
                f"{self.folder} is not indexed; run `notelight index {self.folder}` first"
            )
        self._reader = _core.IndexReader.open(root)
        self.manifest: dict[str, Any] = self._reader.manifest
        indexed_with = self.manifest.get("embedder")
        if embedder is not None and embedder.model_id != indexed_with:
            raise EmbedderMismatchError(
                f"the index for {self.folder} was built with {indexed_with or 'no embedder'}, "
                f"but the query embedder is {embedder.model_id}; "
                "re-index with this embedder or search with the original one"
            )
        self._embedder = embedder

    def is_live(self) -> bool:
        """Whether this searcher still reads the live generation.

        A sync that commits a new generation leaves open searchers on the old
        one; long-lived callers (the MCP server) check this and reopen.
        """
        try:
            current = (self._reader.dir.parent / CURRENT_FILE).read_text(encoding="utf-8")
        except OSError:
            return False
        return current.strip() == self._reader.dir.name

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        """Hybrid BM25 + vector search (BM25 only without an embedder)."""
        vector = self._embedder.embed_query(query) if self._embedder is not None else None
        hits = self._reader.search(query, vector, k=k)
        results = []
        for hit in hits:
            rec = self._reader.chunk(hit.id)
            results.append(
                SearchResult(
                    path=rec.path,
                    line_start=rec.line_start,
                    line_end=rec.line_end,
                    heading_path=rec.heading_path,
                    text=rec.text,
                    score=hit.score,
                    bm25_rank=hit.bm25_rank,
                    vector_rank=hit.vector_rank,
                    chunk_id=hit.id,
                )
            )
        return results


@dataclass(frozen=True)
class IndexInfo:
    """Summary of one index, for `notelight status`."""

    folder: Path
    index_dir: Path
    generation: int | None
    chunks: int
    files: int
    embedder: str | None
    size_bytes: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        d = asdict(self)
        d["folder"], d["index_dir"] = str(self.folder), str(self.index_dir)
        return d


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def describe_index(index_dir: Path) -> IndexInfo | None:
    """Summarize the index in `index_dir`, or None if it has no source record."""
    try:
        source = json.loads((index_dir / SOURCE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    folder = Path(source["folder"])
    root = index_dir / CORE_DIR
    size = _dir_size(index_dir)
    if not _core.index_exists(root):
        return IndexInfo(folder, index_dir, None, 0, 0, None, size, error="no committed index")
    try:
        reader = _core.IndexReader.open(root)
    except (OSError, _core.NotelightError) as e:
        return IndexInfo(folder, index_dir, None, 0, 0, None, size, error=str(e))
    manifest = reader.manifest
    return IndexInfo(
        folder=folder,
        index_dir=index_dir,
        generation=manifest["generation"],
        chunks=manifest["n_chunks"],
        files=_count_files(index_dir / CACHE_FILE),
        embedder=manifest.get("embedder"),
        size_bytes=size,
    )


def _count_files(cache: Path) -> int:
    """Number of indexed files, from the indexer's cache (read-only)."""
    try:
        with contextlib.closing(sqlite3.connect(f"{cache.as_uri()}?mode=ro", uri=True)) as db:
            return int(db.execute("SELECT COUNT(*) FROM files").fetchone()[0])
    except sqlite3.Error:
        return 0


def indexed_files(folder: str | Path) -> dict[str, tuple[int, int]]:
    """Relative path -> (size, mtime_ns) of every file in the last sync.

    Read from the indexer's cache without taking the writer lock. Empty if the
    folder was never indexed.
    """
    cache = paths.index_dir_for(folder) / CACHE_FILE
    if not cache.is_file():
        return {}
    try:
        with contextlib.closing(sqlite3.connect(f"{cache.as_uri()}?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT path, size, mtime_ns FROM files").fetchall()
    except sqlite3.Error:
        return {}
    return {str(p): (int(s), int(m)) for p, s, m in rows}


@dataclass(frozen=True)
class Staleness:
    """How a folder differs from its index, judged by file size and mtime.

    `modified` counts files whose size or mtime changed; a file that was only
    touched counts too, although `notelight index` would find nothing to do.
    """

    added: int
    modified: int
    removed: int

    @property
    def stale(self) -> bool:
        """Whether `notelight index` may have something to update."""
        return bool(self.added or self.modified or self.removed)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, including `stale`."""
        return {**asdict(self), "stale": self.stale}


def check_staleness(folder: str | Path) -> Staleness:
    """Compare `folder` on disk with its last sync. Reads only file metadata."""
    root = paths.normalize_folder(folder)
    known = indexed_files(root)
    added = modified = 0
    seen: set[str] = set()
    for f in walk(root):
        seen.add(f.rel_path)
        prior = known.get(f.rel_path)
        if prior is None:
            added += 1
        elif prior != (f.size, f.mtime_ns):
            modified += 1
    return Staleness(added, modified, sum(1 for p in known if p not in seen))


def indexed_folders() -> list[Path]:
    """Folders with a committed index, sorted. Cheaper than `list_indexes`:
    no index is opened."""
    base = paths.indexes_dir()
    if not base.is_dir():
        return []
    folders = []
    for d in sorted(base.iterdir()):
        try:
            source = json.loads((d / SOURCE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if _core.index_exists(d / CORE_DIR):
            folders.append(Path(source["folder"]))
    return sorted(folders, key=lambda f: str(f).lower())


def list_indexes() -> list[IndexInfo]:
    """All indexes under the data directory, sorted by folder."""
    base = paths.indexes_dir()
    if not base.is_dir():
        return []
    infos = [describe_index(d) for d in sorted(base.iterdir()) if d.is_dir()]
    return sorted((i for i in infos if i is not None), key=lambda i: str(i.folder).lower())
