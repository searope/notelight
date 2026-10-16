"""An MCP server that lets AI agents search and read the user's indexed folders.

`notelight mcp` runs it over stdio. It exposes four read-only tools:

- `search`: hybrid search; every hit carries a `path:start-end` citation.
- `read_context`: the lines around a citation, read from disk.
- `list_sources`: the folders this server can search.
- `index_status`: generation, size, embedder and staleness of one index.

The server never writes an index and never takes the writer lock. It keeps
one `Searcher` per folder (and one embedder per model) loaded across calls,
and reopens a searcher when `notelight index` has published a new
generation. Tools run in worker threads; the Rust search releases the GIL.

`read_context` reads only files that the index already covers: the path
must appear in the index's file list, so `..`, absolute paths elsewhere,
hidden and ignored files are all refused. PDF and DOCX files are served as
the text the indexer extracted from them (the same extraction, so citation
line numbers match), with the PDF pages the returned lines are on.
"""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

import numpy as np
import numpy.typing as npt
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

import notelight
from notelight import _core, documents, paths
from notelight.embed import Embedder, make_embedder, name_for_model_id
from notelight.errors import NotelightError
from notelight.indexer import CORE_DIR
from notelight.ingest import DOCUMENT_SUFFIXES, max_bytes
from notelight.search import (
    IndexInfo,
    Searcher,
    check_staleness,
    describe_index,
    format_pages,
    indexed_files,
    indexed_folders,
    list_indexes,
    snippet,
)

__all__ = [
    "MAX_CONTEXT_LINES",
    "EmbedderFactory",
    "FileContext",
    "IndexStatus",
    "NotesService",
    "SearchHit",
    "SearchResponse",
    "Source",
    "create_server",
    "default_embedder_factory",
    "serve",
]

#: Most lines `read_context` returns in one call.
MAX_CONTEXT_LINES = 400
#: Most results `search` returns in one call.
MAX_K = 50
#: Extracted PDF/DOCX texts kept between `read_context` calls.
EXTRACT_CACHE_SIZE = 8

#: Builds the query embedder for an index's embedder id (None: keyword-only).
EmbedderFactory = Callable[[str | None], Embedder | None]

INSTRUCTIONS = """\
notelight searches the user's own indexed notes and documents (Markdown, text,
and PDF/DOCX when installed). Use `search` first; every result has a `location`
like `notes/rust.md:12-30` (path relative to its folder, then the line range).
Cite locations in answers. For PDF and DOCX files the lines are those of the
extracted text, and PDF hits also carry `pages` (for example `p. 3`): cite both,
as in `report.pdf:40-52 (p. 3)`.
Use `read_context` with a location to read the surrounding lines from disk.
`list_sources` shows which folders are searchable; `index_status` shows whether an
index is out of date (the user refreshes it with `notelight index <folder>`).
"""

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_LOCATION = re.compile(r"^(?P<path>.+):(?P<start>\d+)(?:-(?P<end>\d+))?$")


# ---------------------------------------------------------------------------
# Tool results (pydantic models give clients a JSON schema for each tool)
# ---------------------------------------------------------------------------
class SearchHit(BaseModel):
    """One search result."""

    location: str = Field(description="Citation: `path:line_start-line_end`.")
    folder: str = Field(description="The indexed folder the file belongs to.")
    path: str = Field(description="File path relative to `folder`, `/`-separated.")
    line_start: int = Field(
        description="First line of the chunk (1-based); for PDF and DOCX, a line of the "
        "extracted text."
    )
    line_end: int = Field(description="Last line of the chunk (inclusive).")
    heading_path: list[str] = Field(description="Markdown headings above the chunk.")
    snippet: str = Field(description="The start of the chunk text, on one line.")
    pages: str | None = Field(
        default=None, description="PDF pages of the chunk (`p. 3`, `pp. 3-4`); null otherwise."
    )
    page_start: int | None = Field(default=None, description="First PDF page (1-based).")
    page_end: int | None = Field(default=None, description="Last PDF page (inclusive).")
    score: float = Field(
        description="Fused (RRF) score within its folder's index; higher is better. "
        "Not comparable across folders."
    )


class SkippedFolder(BaseModel):
    """An indexed folder that could not be searched."""

    folder: str
    error: str = Field(description="Why, e.g. a corrupt index; `notelight index` may fix it.")


class SearchResponse(BaseModel):
    """Results of `search`, best first."""

    query: str
    results: list[SearchHit]
    skipped: list[SkippedFolder] = Field(
        default_factory=list,
        description="Folders left out of an all-folder search because they failed.",
    )


class FileContext(BaseModel):
    """Lines read from an indexed file."""

    location: str = Field(description="Citation for exactly the returned lines.")
    folder: str
    path: str
    line_start: int = Field(description="First returned line (1-based).")
    line_end: int = Field(description="Last returned line (inclusive).")
    total_lines: int = Field(description="Number of lines in the file.")
    truncated: bool = Field(description=f"True if the range was cut to {MAX_CONTEXT_LINES} lines.")
    pages: str | None = Field(
        default=None, description="PDF pages of the returned lines; null for other files."
    )
    page_start: int | None = None
    page_end: int | None = None
    text: str


class Source(BaseModel):
    """One indexed folder."""

    folder: str
    files: int
    chunks: int
    embedder: str | None = Field(description="Embedding model id; null for keyword-only.")
    error: str | None = Field(default=None, description="Set if the index cannot be read.")


class IndexStatus(BaseModel):
    """State of one folder's index."""

    folder: str
    generation: int | None
    files: int
    chunks: int
    embedder: str | None
    size_bytes: int
    stale: bool = Field(description="Files changed on disk since the last `notelight index`.")
    added: int = Field(description="Files on disk that are not indexed yet.")
    modified: int = Field(description="Indexed files whose size or mtime changed.")
    removed: int = Field(description="Indexed files that no longer exist.")
    hint: str | None = Field(default=None, description="What to do about it, if anything.")
    error: str | None = None


# ---------------------------------------------------------------------------
# The service: plain Python, independent of MCP
# ---------------------------------------------------------------------------
class _LockedEmbedder:
    """Serializes queries to one embedder.

    Tool calls run in parallel worker threads, and `OnnxEmbedder` loads its
    session lazily on first use, which is not thread-safe.
    """

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed_documents(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        with self._lock:
            return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        with self._lock:
            return self._inner.embed_query(text)


def default_embedder_factory(offline: bool = False) -> EmbedderFactory:
    """The embedder registered for an index's embedder id, as `notelight search` picks it."""

    def factory(model_id: str | None) -> Embedder | None:
        name = name_for_model_id(model_id)
        if name is None:
            raise ToolError(
                f"the index uses embedder {model_id!r}, which this version of notelight "
                "does not know; re-index the folder"
            )
        return make_embedder(name, offline=offline)

    return factory


class NotesService:
    """Read-only access to indexed folders for one server process.

    Args:
        folders: the folders to serve. None serves every indexed folder,
            including ones indexed after the server started.
        embedder_factory: builds the query embedder for an index's embedder
            id. The default loads the registered model (bge-small).

    Every method raises `ToolError` with a message meant for the agent.
    """

    def __init__(
        self,
        folders: Sequence[str | Path] | None = None,
        *,
        embedder_factory: EmbedderFactory | None = None,
    ) -> None:
        self._folders = None if folders is None else [paths.normalize_folder(f) for f in folders]
        self._embedder_factory = embedder_factory or default_embedder_factory()
        self._lock = threading.Lock()
        self._searchers: dict[Path, Searcher] = {}
        self._embedders: dict[str | None, Embedder | None] = {}
        self._extracted: OrderedDict[tuple[str, int, int], str] = OrderedDict()

    # -- folders -------------------------------------------------------------
    def sources(self) -> list[IndexInfo]:
        """The served folders' index summaries."""
        if self._folders is None:
            return list_indexes()
        infos = []
        for folder in self._folders:
            info = describe_index(paths.index_dir_for(folder))
            infos.append(
                info
                or IndexInfo(
                    folder, paths.index_dir_for(folder), None, 0, 0, None, 0, error="not indexed"
                )
            )
        return infos

    def _resolve_folder(self, folder: str | None) -> Path:
        """The served folder named by `folder`, or the only one if it is None."""
        if folder is None:
            usable = self._usable_folders()
            if len(usable) == 1:
                return usable[0]
            if not usable:
                raise ToolError("no folder is indexed yet; run `notelight index <folder>` first")
            listing = ", ".join(str(f) for f in usable)
            raise ToolError(f"several folders are indexed; pass `folder`, one of: {listing}")
        target = paths.normalize_folder(folder)
        key = paths.folder_key(target)
        if self._folders is not None:
            for served in self._folders:
                if paths.folder_key(served) == key:
                    return served
            raise ToolError(f"{target} is not served by this server; see `list_sources`")
        info = describe_index(paths.index_dir_for(target))
        if info is None:
            raise ToolError(f"{target} is not indexed; run `notelight index {target}` first")
        return info.folder

    def _usable_folders(self) -> list[Path]:
        """Served folders that have a committed index (no index is opened)."""
        if self._folders is None:
            return indexed_folders()
        return [f for f in self._folders if _core.index_exists(paths.index_dir_for(f) / CORE_DIR)]

    # -- search --------------------------------------------------------------
    def _embedder_for(self, model_id: str | None) -> Embedder | None:
        with self._lock:
            if model_id not in self._embedders:
                emb = self._embedder_factory(model_id)
                self._embedders[model_id] = None if emb is None else _LockedEmbedder(emb)
            return self._embedders[model_id]

    def searcher(self, folder: Path) -> Searcher:
        """An open searcher for `folder`, reopened if the index has changed."""
        with self._lock:
            cached = self._searchers.get(folder)
        if cached is not None and cached.is_live():
            return cached
        try:
            probe = Searcher(folder, None)
            embedder = self._embedder_for(probe.manifest.get("embedder"))
            searcher = probe if embedder is None else Searcher(folder, embedder)
        except NotelightError as e:
            raise ToolError(str(e)) from e
        with self._lock:
            self._searchers[folder] = searcher
        return searcher

    def search(self, query: str, k: int = 10, folder: str | None = None) -> SearchResponse:
        """Search one folder, or every served folder when `folder` is None.

        Folders are merged by each hit's rank within its own folder, then by
        score among hits of equal rank. Ranking by score alone would bury
        keyword-only folders: their fused scores top out at half of what a
        hybrid index can reach. With all
        folders, a folder that cannot be searched is reported in `skipped`
        instead of failing the whole call.
        """
        if not query.strip():
            raise ToolError("query is empty")
        if not 1 <= k <= MAX_K:
            raise ToolError(f"k must be between 1 and {MAX_K}")
        if folder is not None:
            searcher = self.searcher(self._resolve_folder(folder))
            return SearchResponse(query=query, results=self._search_one(searcher, query, k))
        targets = self._usable_folders()
        if not targets:
            raise ToolError("no folder is indexed yet; run `notelight index <folder>` first")
        ranked: list[tuple[int, int, SearchHit]] = []
        skipped: list[SkippedFolder] = []
        for order, target in enumerate(targets):
            try:
                hits = self._search_one(self.searcher(target), query, k)
            except (ToolError, NotelightError, OSError, ValueError) as e:
                skipped.append(SkippedFolder(folder=str(target), error=str(e)))
                continue
            ranked.extend((rank, order, hit) for rank, hit in enumerate(hits))
        if not ranked and skipped and len(skipped) == len(targets):
            raise ToolError(
                "no indexed folder could be searched: "
                + "; ".join(f"{s.folder}: {s.error}" for s in skipped)
            )
        # Interleave by per-folder rank, so every folder's best hits reach the
        # top whatever its score scale; within one rank, the stronger hit first.
        ranked.sort(key=lambda t: (t[0], -t[2].score, t[1]))
        return SearchResponse(
            query=query, results=[hit for _, _, hit in ranked[:k]], skipped=skipped
        )

    def _search_one(self, searcher: Searcher, query: str, k: int) -> list[SearchHit]:
        try:
            results = searcher.search(query, k=k)
        except NotelightError as e:
            raise ToolError(str(e)) from e
        return [
            SearchHit(
                location=r.location,
                folder=str(searcher.folder),
                path=r.path,
                line_start=r.line_start,
                line_end=r.line_end,
                heading_path=r.heading_path,
                snippet=snippet(r.text),
                pages=r.pages,
                page_start=r.page_start,
                page_end=r.page_end,
                score=r.score,
            )
            for r in results
        ]

    # -- reading files -------------------------------------------------------
    def read_context(
        self,
        location: str,
        context: int = 5,
        folder: str | None = None,
    ) -> FileContext:
        """Lines `start-end` of an indexed file, plus `context` lines around them.

        `location` is a citation from `search` (`path:start-end` or
        `path:line`), or a bare path for the start of the file.
        """
        if not 0 <= context <= 100:
            raise ToolError("context must be between 0 and 100")
        rel, start, end = _parse_location(location)
        root, rel = self._indexed_file(rel, folder)
        text = self._file_text(root / rel, rel)
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()  # a final newline does not start another line
        total = len(lines)
        if start is None:
            first, last = 1, min(total, MAX_CONTEXT_LINES)
        else:
            if start > total:
                raise ToolError(f"{rel} has {total} lines; line {start} does not exist")
            first = max(1, start - context)
            last = min(total, (end if end is not None else start) + context)
        truncated = last - first + 1 > MAX_CONTEXT_LINES or (start is None and total > last)
        last = min(last, first + MAX_CONTEXT_LINES - 1)
        page_start = page_end = None
        if rel.lower().endswith(".pdf"):
            page_start, page_end = documents.page_range(documents.page_starts(text), first, last)
        return FileContext(
            location=f"{rel}:{first}-{last}",
            folder=str(root),
            path=rel,
            line_start=first,
            line_end=last,
            total_lines=total,
            truncated=truncated,
            pages=format_pages(page_start, page_end),
            page_start=page_start,
            page_end=page_end,
            # Page breaks are markers, not content.
            text="\n".join(lines[first - 1 : last]).replace(documents.PAGE_BREAK, ""),
        )

    def _file_text(self, path: Path, rel: str) -> str:
        """The indexed text of a file: decoded, or extracted for PDF and DOCX.

        Extractions are cached by path, size and mtime, so paging through a
        long PDF does not parse it once per call.
        """
        suffix = path.suffix.lower()
        missing = documents.missing_extra(suffix)
        if missing is not None:
            raise ToolError(f"cannot read {rel}: {missing}")
        try:
            st = path.stat()
            if st.st_size > max_bytes(suffix):
                raise ToolError(f"{rel} is larger than {max_bytes(suffix)} bytes")
            key = (str(path), st.st_size, st.st_mtime_ns)
            with self._lock:
                cached = self._extracted.get(key)
            if cached is not None:
                return cached
            decoded = documents.load_text(suffix, path.read_bytes())
        except OSError as e:
            raise ToolError(
                f"cannot read {rel}: {e.strerror or e}; it may have been moved or deleted "
                "since it was indexed"
            ) from e
        if decoded.text is None:
            raise ToolError(f"cannot read {rel}: {decoded.skip_reason}")
        if suffix in DOCUMENT_SUFFIXES:
            with self._lock:
                self._extracted[key] = decoded.text
                while len(self._extracted) > EXTRACT_CACHE_SIZE:
                    self._extracted.popitem(last=False)
        return decoded.text

    def _indexed_file(self, rel: str, folder: str | None) -> tuple[Path, str]:
        """The served folder and indexed relative path that `rel` names.

        Only paths in an index's file list are accepted, so nothing outside
        what was indexed (parent directories, hidden or ignored files) can be
        read, however the path is spelled.
        """
        candidates = [self._resolve_folder(folder)] if folder else self._usable_folders()
        matches = []
        for root in candidates:
            normalized = _relative_to(rel, root)
            if normalized is not None and normalized in indexed_files(root):
                matches.append((root, normalized))
        if len(matches) == 1:
            return matches[0]
        if matches:
            listing = ", ".join(str(root) for root, _ in matches)
            raise ToolError(f"{rel} exists in several folders; pass `folder`, one of: {listing}")
        raise ToolError(
            f"{rel} is not an indexed file in "
            + (str(candidates[0]) if len(candidates) == 1 else "any served folder")
            + "; use a `location` returned by `search`"
        )

    # -- status --------------------------------------------------------------
    def index_status(self, folder: str | None = None) -> IndexStatus:
        """Generation, size and staleness of one folder's index."""
        root = self._resolve_folder(folder)
        info = describe_index(paths.index_dir_for(root))
        if info is None:
            raise ToolError(f"{root} is not indexed; run `notelight index {root}` first")
        st = check_staleness(root) if root.is_dir() else None
        hint = None
        if st is None:
            hint = f"the folder {root} no longer exists"
        elif st.stale:
            hint = f"run `notelight index {root}` to bring the index up to date"
        return IndexStatus(
            folder=str(root),
            generation=info.generation,
            files=info.files,
            chunks=info.chunks,
            embedder=info.embedder,
            size_bytes=info.size_bytes,
            stale=st is None or st.stale,
            added=st.added if st else 0,
            modified=st.modified if st else 0,
            removed=st.removed if st else info.files,
            hint=hint,
            error=info.error,
        )


def _parse_location(location: str) -> tuple[str, int | None, int | None]:
    """Split `path:start-end` / `path:line` / `path` into its parts."""
    location = location.strip()
    if not location:
        raise ToolError("location is empty")
    m = _LOCATION.match(location)
    if m is None:
        return location, None, None
    start = int(m["start"])
    end = int(m["end"]) if m["end"] else None
    if start < 1 or (end is not None and end < start):
        raise ToolError(f"bad line range in {location!r}")
    return m["path"], start, end


def _relative_to(rel: str, root: Path) -> str | None:
    """`rel` as a `/`-separated path relative to `root`, or None if it is outside.

    Accepts an absolute path inside `root`. Nothing is resolved on disk: the
    result is only ever compared with the index's file list.
    """
    candidate = Path(rel)
    if candidate.is_absolute():
        try:
            candidate = Path(os.path.normpath(candidate)).relative_to(root)
        except ValueError:
            return None
    posix = PurePosixPath(candidate.as_posix())
    if posix.is_absolute() or ".." in posix.parts:
        return None
    return posix.as_posix()


# ---------------------------------------------------------------------------
# The MCP server
# ---------------------------------------------------------------------------
def create_server(service: NotesService) -> MCPServer:
    """An MCP server exposing `service` as four read-only tools."""
    server = MCPServer(
        name="notelight",
        title="notelight",
        version=notelight.__version__,
        instructions=INSTRUCTIONS,
    )

    @server.tool(annotations=_READ_ONLY)
    def search(query: str, k: int = 10, folder: str | None = None) -> SearchResponse:
        """Search the user's indexed notes (hybrid keyword + semantic search).

        Args:
            query: what to look for, in natural language or keywords.
            k: number of results (1-50).
            folder: an indexed folder from `list_sources`; omit to search all of them.
        """
        return service.search(query, k=k, folder=folder)

    @server.tool(annotations=_READ_ONLY)
    def read_context(location: str, context: int = 5, folder: str | None = None) -> FileContext:
        """Read lines of an indexed file from disk, with surrounding lines.

        Args:
            location: a `location` from `search` (`path:start-end`), `path:line`,
                or a bare path for the start of the file.
            context: extra lines before and after the range (0-100).
            folder: the folder the path is relative to; needed only if several
                indexed folders contain the same path.
        """
        return service.read_context(location, context=context, folder=folder)

    @server.tool(annotations=_READ_ONLY)
    def list_sources() -> list[Source]:
        """List the indexed folders this server can search, with file and chunk counts."""
        return [
            Source(
                folder=str(i.folder),
                files=i.files,
                chunks=i.chunks,
                embedder=i.embedder,
                error=i.error,
            )
            for i in service.sources()
        ]

    @server.tool(annotations=_READ_ONLY)
    def index_status(folder: str | None = None) -> IndexStatus:
        """Show an index's generation, size and embedder, and whether files changed since.

        Args:
            folder: an indexed folder; may be omitted when only one is indexed.
        """
        return service.index_status(folder)

    return server


def serve(folders: Sequence[str | Path] | None, *, offline: bool = False) -> None:
    """Run the server over stdio until the client disconnects."""
    service = NotesService(folders, embedder_factory=default_embedder_factory(offline))
    create_server(service).run("stdio")
