"""The MCP server, driven through the SDK's in-process client.

Every test indexes the `notes` fixture with the fake HashingEmbedder, so
nothing is downloaded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters
from mcp_types import TextContent

from notelight import paths
from notelight.embed import HashingEmbedder
from notelight.indexer import CORE_DIR, sync
from notelight.mcp_server import MAX_CONTEXT_LINES, NotesService, create_server
from notelight.paths import HOME_ENV
from notelight.search import Searcher, check_staleness

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def indexed(notes: Path) -> Path:
    sync(notes, HashingEmbedder())
    return notes


async def call(service: NotesService, tool: str, args: dict[str, Any] | None = None) -> Any:
    """Call `tool` and return its structured result; fail if it reports an error."""
    async with Client(create_server(service)) as client:
        result = await client.call_tool(tool, args or {})
    assert not result.is_error, result.content
    return result.structured_content


async def call_error(service: NotesService, tool: str, args: dict[str, Any]) -> str:
    """Call `tool`, expect an error result, and return its message."""
    async with Client(create_server(service)) as client:
        result = await client.call_tool(tool, args)
    assert result.is_error
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


async def test_lists_four_read_only_tools() -> None:
    async with Client(create_server(NotesService())) as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == {"search", "read_context", "list_sources", "index_status"}
    for tool in tools:
        assert tool.annotations is not None and tool.annotations.read_only_hint
        assert tool.output_schema is not None


async def test_stdio_subprocess(indexed: Path) -> None:
    """`notelight mcp` speaks MCP on stdout and logs nothing else there."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "notelight.cli", "mcp", "--folder", str(indexed)],
        env={HOME_ENV: os.environ[HOME_ENV]},
    )
    async with Client(params, read_timeout_seconds=60) as client:
        result = await client.call_tool("search", {"query": "sourdough", "k": 1})
    assert not result.is_error, result.content
    assert result.structured_content is not None
    assert result.structured_content["results"][0]["path"] == "recipes/bread.txt"


# -- search --------------------------------------------------------------------
async def test_search_returns_citations(indexed: Path) -> None:
    out = await call(NotesService(), "search", {"query": "single owner", "k": 3})
    top = out["results"][0]
    assert top["path"] == "rust/ownership.md"
    assert top["heading_path"] == ["Rust", "Ownership"]
    assert top["location"] == f"rust/ownership.md:{top['line_start']}-{top['line_end']}"
    assert top["folder"] == str(indexed.resolve())
    assert "single owner" in top["snippet"]
    assert len(out["results"]) <= 3


async def test_search_named_folder(indexed: Path) -> None:
    out = await call(NotesService(), "search", {"query": "sourdough", "folder": str(indexed)})
    assert out["results"][0]["path"] == "recipes/bread.txt"


async def test_search_merges_several_folders(indexed: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "garden.md").write_text("# Garden\n\nTomatoes need sun.\n", encoding="utf-8")
    sync(other, HashingEmbedder())
    service = NotesService()
    out = await call(service, "search", {"query": "tomatoes sun"})
    assert out["results"][0]["folder"] == str(other.resolve())
    out = await call(service, "search", {"query": "sourdough"})
    assert out["results"][0]["folder"] == str(indexed.resolve())


async def test_search_not_indexed(tmp_path: Path) -> None:
    msg = await call_error(NotesService(), "search", {"query": "x", "folder": str(tmp_path)})
    assert "is not indexed" in msg and "notelight index" in msg
    msg = await call_error(NotesService(), "search", {"query": "x"})
    assert "no folder is indexed yet" in msg


async def test_search_embedder_mismatch(indexed: Path) -> None:
    service = NotesService(embedder_factory=lambda _model_id: HashingEmbedder(dim=32))
    msg = await call_error(service, "search", {"query": "owner"})
    assert "was built with hashing-256" in msg
    assert "hashing-32" in msg


async def test_search_unknown_embedder(notes: Path) -> None:
    sync(notes, HashingEmbedder(dim=64))  # not a registered embedder
    msg = await call_error(NotesService(), "search", {"query": "owner"})
    assert "does not know" in msg


async def test_search_keyword_only_index(notes: Path) -> None:
    sync(notes, None)
    out = await call(NotesService(), "search", {"query": "mypy"})
    assert out["results"][0]["path"] == "python/typing.md"


async def test_search_rejects_bad_arguments(indexed: Path) -> None:
    assert "query is empty" in await call_error(NotesService(), "search", {"query": "  "})
    assert "k must be" in await call_error(NotesService(), "search", {"query": "x", "k": 0})


async def test_server_restricted_to_folders(indexed: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "a.md").write_text("sourdough elsewhere\n", encoding="utf-8")
    sync(other, HashingEmbedder())
    service = NotesService([indexed])
    out = await call(service, "search", {"query": "sourdough"})
    assert {r["folder"] for r in out["results"]} == {str(indexed.resolve())}
    msg = await call_error(service, "search", {"query": "x", "folder": str(other)})
    assert "not served" in msg
    sources = await call(service, "list_sources")
    assert [s["folder"] for s in sources["result"]] == [str(indexed.resolve())]


async def test_searcher_reopens_after_reindex(indexed: Path) -> None:
    service = NotesService()
    first = service.searcher(indexed.resolve())
    assert service.searcher(indexed.resolve()) is first  # kept across calls
    (indexed / "new.md").write_text("# Kites\n\nKites fly in wind.\n", encoding="utf-8")
    sync(indexed, HashingEmbedder())
    assert not first.is_live()
    out = await call(service, "search", {"query": "kites wind"})
    assert out["results"][0]["path"] == "new.md"


async def test_search_does_not_take_the_writer_lock(indexed: Path) -> None:
    import filelock

    from notelight.indexer import LOCK_FILE

    lock = filelock.FileLock(paths.index_dir_for(indexed) / LOCK_FILE, timeout=0)
    with lock:  # a writer is busy; searching and status still work
        out = await call(NotesService(), "search", {"query": "owner"})
        assert out["results"]
        assert (await call(NotesService(), "index_status", {}))["generation"] == 1


# -- read_context --------------------------------------------------------------
async def test_read_context_from_a_citation(indexed: Path) -> None:
    service = NotesService()
    hit = (await call(service, "search", {"query": "lifetimes valid", "k": 1}))["results"][0]
    out = await call(service, "read_context", {"location": hit["location"], "context": 1})
    assert out["path"] == hit["path"]
    assert out["line_start"] == max(1, hit["line_start"] - 1)
    lines = (indexed / hit["path"]).read_text(encoding="utf-8").splitlines()
    assert out["total_lines"] == len(lines)
    assert out["text"] == "\n".join(lines[out["line_start"] - 1 : out["line_end"]])
    assert out["location"] == f"{hit['path']}:{out['line_start']}-{out['line_end']}"


async def test_read_context_single_line_and_whole_file(indexed: Path) -> None:
    service = NotesService()
    out = await call(service, "read_context", {"location": "recipes/bread.txt:3", "context": 0})
    assert out["text"].startswith("Mix flour")
    assert (out["line_start"], out["line_end"]) == (3, 3)
    out = await call(service, "read_context", {"location": "python/typing.md"})
    assert out["line_start"] == 1 and out["line_end"] == out["total_lines"] == 4
    assert not out["truncated"]


async def test_read_context_absolute_and_backslash_paths(indexed: Path) -> None:
    service = NotesService()
    absolute = str((indexed / "rust" / "ownership.md").resolve())
    out = await call(service, "read_context", {"location": f"{absolute}:3-3", "context": 0})
    assert out["path"] == "rust/ownership.md"
    if os.sep == "\\":
        out = await call(service, "read_context", {"location": "rust\\ownership.md:1"})
        assert out["path"] == "rust/ownership.md"


async def test_read_context_is_bounded(indexed: Path) -> None:
    (indexed / "long.md").write_text("".join(f"line {i}\n" for i in range(1, 1001)))
    sync(indexed, HashingEmbedder())
    service = NotesService()
    out = await call(service, "read_context", {"location": "long.md:1-1000"})
    assert out["truncated"] and out["line_end"] - out["line_start"] + 1 == MAX_CONTEXT_LINES
    msg = await call_error(service, "read_context", {"location": "long.md:5000"})
    assert "has 1000 lines" in msg


@pytest.mark.parametrize(
    "location",
    [
        "../secret.md",
        "rust/../../secret.md",
        "SECRET_ABS",
        "/etc/passwd",
        "C:/Windows/win.ini",
        ".hidden.md",
        "missing.md",
    ],
)
async def test_read_context_rejects_paths_outside_the_index(indexed: Path, location: str) -> None:
    secret = indexed.parent / "secret.md"
    secret.write_text("password\n", encoding="utf-8")
    (indexed / ".hidden.md").write_text("hidden\n", encoding="utf-8")
    location = location.replace("SECRET_ABS", str(secret))
    msg = await call_error(NotesService(), "read_context", {"location": location})
    assert "is not an indexed file" in msg
    assert "password" not in msg


async def test_read_context_file_deleted_since_indexing(indexed: Path) -> None:
    (indexed / "recipes" / "bread.txt").unlink()
    msg = await call_error(NotesService(), "read_context", {"location": "recipes/bread.txt:1"})
    assert "cannot read recipes/bread.txt" in msg


async def test_read_context_bad_range(indexed: Path) -> None:
    msg = await call_error(NotesService(), "read_context", {"location": "python/typing.md:4-2"})
    assert "bad line range" in msg


# -- list_sources and index_status --------------------------------------------
async def test_list_sources(indexed: Path) -> None:
    out = await call(NotesService(), "list_sources")
    assert out["result"] == [
        {
            "folder": str(indexed.resolve()),
            "files": 3,
            "chunks": out["result"][0]["chunks"],
            "embedder": "hashing-256",
            "error": None,
        }
    ]
    assert out["result"][0]["chunks"] >= 3


async def test_list_sources_empty() -> None:
    assert (await call(NotesService(), "list_sources"))["result"] == []


async def test_index_status_fresh_then_stale(indexed: Path) -> None:
    service = NotesService()
    out = await call(service, "index_status", {"folder": str(indexed)})
    assert out["generation"] == 1 and out["files"] == 3 and out["embedder"] == "hashing-256"
    assert not out["stale"] and out["hint"] is None
    (indexed / "new.md").write_text("new\n", encoding="utf-8")
    (indexed / "recipes" / "bread.txt").unlink()
    (indexed / "python" / "typing.md").write_text("changed text\n", encoding="utf-8")
    out = await call(service, "index_status", {})
    assert out["stale"]
    assert (out["added"], out["modified"], out["removed"]) == (1, 1, 1)
    assert "notelight index" in out["hint"]


async def test_index_status_not_indexed(tmp_path: Path) -> None:
    msg = await call_error(NotesService(), "index_status", {"folder": str(tmp_path)})
    assert "is not indexed" in msg


# -- helpers -------------------------------------------------------------------
def test_check_staleness_matches_sync(indexed: Path) -> None:
    assert not check_staleness(indexed).stale
    (indexed / "x.md").write_text("x\n", encoding="utf-8")
    assert check_staleness(indexed).to_dict() == {
        "added": 1,
        "modified": 0,
        "removed": 0,
        "stale": True,
    }
    sync(indexed, HashingEmbedder())
    assert not check_staleness(indexed).stale


def test_searcher_is_live(indexed: Path) -> None:
    searcher = Searcher(indexed, None)
    assert searcher.is_live()
    assert (paths.index_dir_for(indexed) / CORE_DIR / "CURRENT").is_file()


# -- regressions from the milestone 2-4 review -----------------------------------


@pytest.fixture
def other(tmp_path: Path) -> Path:
    """A second notes folder, indexed keyword-only."""
    root = tmp_path / "other"
    root.mkdir()
    (root / "baking.md").write_text("# Baking\n\nSourdough bread needs a starter.\n", "utf-8")
    sync(root, None)
    return root


async def test_all_folder_search_skips_a_broken_index(indexed: Path, other: Path) -> None:
    live = Searcher(other, None).manifest["generation"]
    gen_dir = paths.index_dir_for(other) / CORE_DIR / f"gen-{live:06d}"
    (gen_dir / "docs.bin").write_bytes(b"corrupt")
    out = await call(NotesService(), "search", {"query": "sourdough"})
    assert {r["folder"] for r in out["results"]} == {str(indexed)}
    assert [s["folder"] for s in out["skipped"]] == [str(other.resolve())]
    # Asking for the broken folder by name still reports its error.
    msg = await call_error(NotesService(), "search", {"query": "x", "folder": str(other)})
    assert "invalid index data" in msg or "corrupt" in msg.lower()


async def test_all_folder_search_interleaves_folders_by_rank(indexed: Path, other: Path) -> None:
    # `other` is keyword-only, so its scores are at most half a hybrid index's.
    out = await call(NotesService(), "search", {"query": "sourdough bread starter", "k": 2})
    assert {r["folder"] for r in out["results"]} == {str(indexed), str(other.resolve())}
    assert out["skipped"] == []
