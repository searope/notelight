"""Search an indexed folder."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from notelight import _core, documents, paths
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
    "format_pages",
    "indexed_files",
    "indexed_folders",
    "list_indexes",
    "match_spans",
    "page_starts",
    "query_terms",
    "skipped_files",
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
    page_start: int | None = None
    """First PDF page the chunk is on (1-based); None for other files."""
    page_end: int | None = None
    """Last PDF page the chunk is on; None for other files."""

    @property
    def location(self) -> str:
        """`path:start-end`, the citation format used throughout notelight.

        For a PDF the lines are those of its extracted text (see
        `notelight.documents`); `pages` names the pages they are on.
        """
        return f"{self.path}:{self.line_start}-{self.line_end}"

    @property
    def pages(self) -> str | None:
        """`p. 3` or `pp. 3-4` for a PDF hit, else None."""
        return format_pages(self.page_start, self.page_end)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, including `location` and `pages`."""
        return {**asdict(self), "location": self.location, "pages": self.pages}


def format_pages(first: int | None, last: int | None) -> str | None:
    """`p. 3` or `pp. 3-4`, or None without a page."""
    if first is None or last is None:
        return None
    return f"p. {first}" if first == last else f"pp. {first}-{last}"


#: A run of letters and digits: the unit `match_spans` highlights.
_WORD = re.compile(r"[^\W_]+")


def query_terms(query: str) -> frozenset[str]:
    """The BM25 terms of `query`: what search matched, and what to highlight."""
    return frozenset(_core.tokenize(query))


def match_spans(text: str, terms: AbstractSet[str]) -> list[tuple[int, int]]:
    """`(start, end)` spans of the words in `text` that match one of `terms`.

    Words are compared as BM25 terms, as search compares them, so case does
    not matter and `getUserName` matches the query "user".
    """
    if not terms:
        return []
    return [
        m.span() for m in _WORD.finditer(text) if not terms.isdisjoint(_core.tokenize(m.group()))
    ]


def snippet(text: str, width: int = 240, terms: AbstractSet[str] = frozenset()) -> str:
    """At most `width` characters of `text`, on one line.

    Normally the start of the text. Given query `terms` (see `query_terms`)
    and no match near the start, the window starts a little before the first
    matching word instead, so the match is visible; `…` marks cut ends.
    """
    flat = " ".join(text.split())
    if len(flat) <= width:
        return flat
    spans = match_spans(flat, terms)
    if not spans or spans[0][1] <= width - 1:
        return flat[: width - 1].rstrip() + "…"
    # Show some lead-in, keep the window full near the end, start on a word.
    first = spans[0][0]
    start = max(0, min(first - width // 4, len(flat) - (width - 1)))
    space = flat.find(" ", start, first)
    if space != -1:
        start = space + 1
    if start == 0:  # a single huge word: nothing better to show than the start
        return flat[: width - 1].rstrip() + "…"
    body = flat[start : start + width - 2]
    if start + width - 2 >= len(flat):
        return "…" + body
    return "…" + body.rstrip() + "…"


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
        if vector is not None and not np.any(vector):
            # The query has no features the embedder knows (for example only
            # punctuation with the hashing embedder): search keywords alone.
            vector = None
        hits = self._reader.search(query, vector, k=k)
        records = [(hit, self._reader.chunk(hit.id)) for hit in hits]
        pdfs = {rec.path for _, rec in records if rec.path.lower().endswith(".pdf")}
        starts = page_starts(self.folder, pdfs)
        results = []
        for hit, rec in records:
            pages: tuple[int | None, int | None] = (None, None)
            if rec.path in starts:
                pages = documents.page_range(starts[rec.path], rec.line_start, rec.line_end)
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
                    page_start=pages[0],
                    page_end=pages[1],
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
    """Total size of the files under `path`.

    A sync may run at the same time (SQLite journals, old generations and
    temporary files come and go), so files that vanish mid-scan are skipped.
    """
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total


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


def page_starts(folder: str | Path, rel_paths: set[str]) -> dict[str, list[int]]:
    """Relative path -> the line each page starts on, for the indexed PDFs
    among `rel_paths` (see `documents.page_starts`).

    Read from the indexer's cache without the writer lock. Paths with no
    recorded pages are left out.
    """
    cache = paths.index_dir_for(folder) / CACHE_FILE
    if not rel_paths or not cache.is_file():
        return {}
    wanted = sorted(rel_paths)
    marks = ",".join("?" * len(wanted))
    try:
        with contextlib.closing(sqlite3.connect(f"{cache.as_uri()}?mode=ro", uri=True)) as db:
            rows = db.execute(
                "SELECT files.path, pages.starts FROM files "
                f"JOIN pages ON pages.doc_key = files.doc_key WHERE files.path IN ({marks})",
                wanted,
            ).fetchall()
    except sqlite3.Error:
        return {}  # a cache from before pages were recorded
    return {str(p): [int(n) for n in json.loads(s)] for p, s in rows}


def _skipped_rows(folder: str | Path) -> dict[str, tuple[int, int, str]]:
    cache = paths.index_dir_for(folder) / CACHE_FILE
    if not cache.is_file():
        return {}
    try:
        with contextlib.closing(sqlite3.connect(f"{cache.as_uri()}?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT path, size, mtime_ns, reason FROM skipped").fetchall()
    except sqlite3.Error:
        return {}
    return {str(p): (int(s), int(m), str(r)) for p, s, m, r in rows}


def skipped_files(folder: str | Path) -> dict[str, tuple[int, int]]:
    """Relative path -> (size, mtime_ns) of every file the last sync found but
    could not index (binary, undecodable, unreadable, missing extra). Empty
    for a cache written before this was recorded."""
    return {p: (s, m) for p, (s, m, _) in _skipped_rows(folder).items()}


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
    skipped = _skipped_rows(root)
    added = modified = 0
    seen: set[str] = set()
    for f in walk(root):
        seen.add(f.rel_path)
        prior = known.get(f.rel_path)
        if prior is None:
            # A file the last sync skipped is not new unless it changed since,
            # or it was skipped for want of an extra that is now installed.
            skip = skipped.get(f.rel_path)
            if (
                skip is None
                or skip[:2] != (f.size, f.mtime_ns)
                or (
                    documents.is_missing_extra_reason(skip[2])
                    and documents.missing_extra(f.suffix) is None
                )
            ):
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
