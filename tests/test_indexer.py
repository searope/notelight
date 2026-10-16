"""End-to-end indexing and search with the fake embedder: incremental updates,
embedder checks, locking and path handling."""

from __future__ import annotations

import contextlib
import os
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import filelock
import numpy as np
import numpy.typing as npt
import pytest
from conftest import CountingEmbedder

from notelight import indexer, paths
from notelight.embed import HashingEmbedder
from notelight.errors import EmbedderMismatchError, IndexBusyError, IndexNotFoundError
from notelight.indexer import LOCK_FILE, sync
from notelight.search import Searcher, check_staleness, indexed_files, list_indexes


def test_first_sync_indexes_everything(notes: Path, embedder: CountingEmbedder) -> None:
    stats = sync(notes, embedder)
    assert (stats.files, stats.added, stats.changed, stats.removed) == (3, 3, 0, 0)
    assert stats.generation == 1 and not stats.up_to_date
    assert stats.embedded == stats.chunks == len(embedder.embedded) > 0


def test_search_returns_citable_results(notes: Path, embedder: CountingEmbedder) -> None:
    sync(notes, embedder)
    results = Searcher(notes, embedder).search("who owns a value", k=3)
    top = results[0]
    assert top.path == "rust/ownership.md"
    assert top.heading_path == ["Rust", "Ownership"]
    assert top.location == f"rust/ownership.md:{top.line_start}-{top.line_end}"
    assert "single owner" in top.text
    assert top.to_dict()["location"] == top.location


def test_heading_text_is_searchable(notes: Path, embedder: CountingEmbedder) -> None:
    # "Lifetimes" appears in the heading and the body; headings are embedded too.
    sync(notes, embedder)
    (top, *_) = Searcher(notes, embedder).search("lifetimes", k=1)
    assert top.heading_path == ["Rust", "Lifetimes"]


def test_unchanged_folder_is_up_to_date(notes: Path, embedder: CountingEmbedder) -> None:
    sync(notes, embedder)
    embedder.embedded.clear()
    stats = sync(notes, embedder)
    assert stats.up_to_date and stats.generation is None
    assert embedder.embedded == []


def test_touched_but_identical_file_is_not_reembedded(
    notes: Path, embedder: CountingEmbedder
) -> None:
    sync(notes, embedder)
    target = notes / "python" / "typing.md"
    st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    embedder.embedded.clear()
    assert sync(notes, embedder).up_to_date
    assert embedder.embedded == []


def test_incremental_reindex_embeds_only_changed_chunks(
    notes: Path, embedder: CountingEmbedder
) -> None:
    first = sync(notes, embedder)
    embedder.embedded.clear()

    (notes / "recipes" / "bread.txt").write_text("Rye bread needs a long proof.\n", "utf-8")
    (notes / "new.md").write_text("# Fresh\n\nA brand new note about tea.\n", "utf-8")
    (notes / "python" / "typing.md").unlink()

    stats = sync(notes, embedder)
    assert (stats.files, stats.added, stats.changed, stats.removed) == (3, 1, 1, 1)
    assert stats.generation == (first.generation or 0) + 1
    # Only the two new/changed files' chunks were embedded; Rust notes reused the cache.
    assert stats.embedded == len(embedder.embedded) == 2
    assert all("Rust" not in t for t in embedder.embedded)

    searcher = Searcher(notes, embedder)
    paths_found = {r.path for r in searcher.search("tea rye bread mypy", k=10)}
    assert "new.md" in paths_found and "recipes/bread.txt" in paths_found
    assert "python/typing.md" not in paths_found


def test_renamed_file_reuses_embeddings(notes: Path, embedder: CountingEmbedder) -> None:
    sync(notes, embedder)
    embedder.embedded.clear()
    (notes / "rust" / "ownership.md").rename(notes / "rust" / "borrowing.md")
    stats = sync(notes, embedder)
    assert (stats.added, stats.removed, stats.embedded) == (1, 1, 0)
    (top, *_) = Searcher(notes, embedder).search("single owner", k=1)
    assert top.path == "rust/borrowing.md"


def test_changing_chunk_size_rechunks(notes: Path, embedder: CountingEmbedder) -> None:
    before = sync(notes, embedder).chunks
    stats = sync(notes, embedder, max_chars=60)
    assert not stats.up_to_date and stats.chunks > before


def test_switching_embedder_rebuilds(notes: Path, embedder: CountingEmbedder) -> None:
    sync(notes, embedder)
    other = HashingEmbedder(dim=32)
    stats = sync(notes, other)
    assert not stats.up_to_date and stats.embedded == stats.chunks
    assert Searcher(notes, other).manifest["embedder"] == other.model_id


def test_quoted_phrase_ranks_first(notes: Path, embedder: CountingEmbedder) -> None:
    sync(notes, embedder)
    (top, *_) = Searcher(notes, embedder).search('"single owner"', k=3)
    assert top.path == "rust/ownership.md"
    assert "single owner" in top.text


def test_search_with_different_embedder_is_refused(notes: Path, embedder: CountingEmbedder) -> None:
    sync(notes, embedder)
    with pytest.raises(EmbedderMismatchError, match=embedder.model_id):
        Searcher(notes, HashingEmbedder(dim=32))


def test_keyword_only_index(notes: Path) -> None:
    stats = sync(notes, None)
    assert stats.embedded == 0 and stats.chunks > 0
    s = Searcher(notes, None)
    assert s.manifest["embedder"] is None
    assert s.search("sourdough")[0].path == "recipes/bread.txt"
    with pytest.raises(EmbedderMismatchError):
        Searcher(notes, HashingEmbedder())


def test_searching_unindexed_folder(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError, match="notelight index"):
        Searcher(tmp_path, None)


def test_lock_blocks_a_second_writer(notes: Path, embedder: CountingEmbedder) -> None:
    index_dir = paths.index_dir_for(notes)
    index_dir.mkdir(parents=True)
    with (
        filelock.FileLock(index_dir / LOCK_FILE),
        pytest.raises(IndexBusyError, match="another notelight process"),
    ):
        sync(notes, embedder, lock_timeout=0)
    assert sync(notes, embedder).generation == 1  # works once the lock is free


def test_skipped_files_are_reported(notes: Path, embedder: CountingEmbedder) -> None:
    (notes / "blob.txt").write_bytes(b"\x00\x01binary")
    stats = sync(notes, embedder)
    assert [(s.path, s.reason) for s in stats.skipped] == [("blob.txt", "binary")]
    assert stats.files == 3


def test_empty_folder(tmp_path: Path, embedder: CountingEmbedder) -> None:
    stats = sync(tmp_path, embedder)
    assert stats.chunks == 0 and stats.generation == 1
    assert Searcher(tmp_path, embedder).search("anything") == []


def test_not_a_folder(tmp_path: Path, embedder: CountingEmbedder) -> None:
    with pytest.raises(NotADirectoryError):
        sync(tmp_path / "missing", embedder)


def test_status_lists_indexes(notes: Path, embedder: CountingEmbedder) -> None:
    sync(notes, embedder)
    (info,) = list_indexes()
    assert info.folder == notes.resolve()
    assert (info.files, info.generation, info.embedder) == (3, 1, embedder.model_id)
    assert info.chunks > 0 and info.size_bytes > 0 and info.error is None


def test_relative_and_absolute_paths_share_an_index(
    notes: Path, embedder: CountingEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync(notes, embedder)
    monkeypatch.chdir(notes.parent)
    assert sync(Path("notes"), embedder).up_to_date


@pytest.mark.skipif(sys.platform != "win32", reason="Windows paths are case-insensitive")
def test_windows_folder_key_ignores_case_and_separators(notes: Path) -> None:
    upper = Path(str(notes).upper())
    assert paths.folder_key(upper) == paths.folder_key(notes)
    assert paths.folder_key(str(notes).replace("\\", "/")) == paths.folder_key(notes)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific path handling")
def test_windows_paths_are_stored_with_forward_slashes(
    notes: Path, embedder: CountingEmbedder
) -> None:
    sync(notes, embedder)
    paths_found = {r.path for r in Searcher(notes, embedder).search("rust python bread", k=10)}
    assert paths_found and all("\\" not in p for p in paths_found)
    assert "rust/ownership.md" in paths_found


# -- regressions from the milestone 2-4 review -----------------------------------


def _cache_rows(folder: Path, table: str) -> int:
    with contextlib.closing(
        sqlite3.connect(paths.index_dir_for(folder) / indexer.CACHE_FILE)
    ) as db:
        return int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_skipped_file_does_not_make_the_index_stale(
    notes: Path, embedder: CountingEmbedder
) -> None:
    blob = notes / "blob.txt"
    blob.write_bytes(b"\x00\x01binary")
    sync(notes, embedder)
    assert not check_staleness(notes).stale
    # Still reported as skipped on the next sync, without being read again.
    assert [s.path for s in sync(notes, embedder).skipped] == ["blob.txt"]
    # Once it changes it may have become indexable, so it counts as new.
    blob.write_bytes(b"now plain text about sourdough\n")
    assert check_staleness(notes).added == 1
    stats = sync(notes, embedder)
    assert (stats.files, stats.added, stats.skipped) == (4, 1, [])


def test_keyword_only_sync_keeps_the_embedding_cache(
    notes: Path, embedder: CountingEmbedder
) -> None:
    sync(notes, embedder)
    cached = _cache_rows(notes, "embeddings")
    assert cached > 0
    sync(notes, None)
    assert _cache_rows(notes, "embeddings") == cached
    embedder.embedded.clear()
    assert sync(notes, embedder).embedded == 0  # back to the model: nothing re-embedded


class _FailingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__(dim=64)

    def embed_documents(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        raise RuntimeError("embedding interrupted")


def test_failed_rechunk_keeps_the_live_file_list(notes: Path, embedder: CountingEmbedder) -> None:
    sync(notes, embedder)
    before = indexed_files(notes)
    assert len(before) == 3
    # A new chunk size forces a re-chunk and re-embed; make the sync fail midway.
    with pytest.raises(RuntimeError, match="interrupted"):
        sync(notes, _FailingEmbedder(), max_chars=60)
    assert indexed_files(notes) == before  # the old generation is still live and described
    assert not check_staleness(notes).stale
    stats = sync(notes, embedder, max_chars=60)  # and a later sync completes the change
    assert stats.generation == 2 and len(indexed_files(notes)) == 3


def test_query_without_known_features_falls_back_to_keywords(
    notes: Path, embedder: CountingEmbedder
) -> None:
    sync(notes, embedder)
    searcher = Searcher(notes, embedder)
    assert not embedder.embed_query("???").any()
    assert searcher.search("???") == []  # BM25 finds nothing; no ValueError from the core


def test_status_tolerates_files_vanishing_during_a_sync(
    notes: Path, embedder: CountingEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync(notes, embedder)
    index_dir = paths.index_dir_for(notes)
    journal = index_dir / "cache.sqlite-journal"
    journal.write_bytes(b"x")
    real_stat = os.stat

    def stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if str(path).endswith("-journal"):
            raise FileNotFoundError(path)  # deleted between listing and stat
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    (info,) = list_indexes()
    assert info.error is None and info.size_bytes > 0
