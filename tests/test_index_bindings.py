"""The PyO3 index API: chunking, writer/reader round trip, and error mapping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from notelight import _core


def test_chunk_text_reports_headings_and_lines() -> None:
    chunks = _core.chunk_text("# A\n\nintro\n\n## B\n\nbody text\n", max_chars=200)
    assert [c.heading_path for c in chunks] == [["A"], ["A", "B"]]
    assert (chunks[1].line_start, chunks[1].line_end) == (5, 7)
    assert "body text" in chunks[1].text


def test_chunk_text_plain_mode_ignores_headings() -> None:
    (chunk,) = _core.chunk_text("# not a heading\nline two\n", markdown=False)
    assert chunk.heading_path == []


def test_chunk_text_byte_offsets_are_utf8() -> None:
    text = "# Café\n\nnaïve text\n"
    (chunk,) = _core.chunk_text(text)
    assert text.encode()[chunk.byte_start : chunk.byte_end].decode() == chunk.text


def test_chunk_text_rejects_zero_max_chars() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        _core.chunk_text("x", max_chars=0)


def _vec(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float32)


def _two_chunk_index(root: Path) -> _core.CommitInfo:
    w = _core.IndexWriter(3, "test-model", {"source": "unit", "n": [1, 2]})
    w.add(_core.ChunkRecord("a.md", ["A"], 1, 2, "alpha apples"), _vec(1, 0, 0))
    w.add(_core.ChunkRecord("b/c.md", [], 3, 9, "beta bananas"), _vec(0, 1, 0))
    assert len(w) == 2
    return w.commit(root)


def test_round_trip(tmp_path: Path) -> None:
    info = _two_chunk_index(tmp_path)
    assert (info.generation, info.n_chunks) == (1, 2)
    assert _core.index_exists(tmp_path)

    r = _core.IndexReader.open(tmp_path)
    assert len(r) == 2 and r.has_vectors
    assert r.manifest["embedder"] == "test-model"
    assert r.manifest["metadata"] == {"source": "unit", "n": [1, 2]}
    assert r.chunk(1) == _core.ChunkRecord("b/c.md", [], 3, 9, "beta bananas")

    hits = r.search("apples", _vec(0, 1, 0), k=5)
    assert {h.id for h in hits} == {0, 1}
    by_id = {h.id: h for h in hits}
    assert (by_id[0].bm25_rank, by_id[1].vector_rank) == (1, 1)
    assert r.search_bm25("bananas", 5)[0][0] == 1
    assert r.search_vector(_vec(1, 0, 0), 1)[0][0] == 0


def test_second_commit_is_a_new_generation(tmp_path: Path) -> None:
    _two_chunk_index(tmp_path)
    assert _two_chunk_index(tmp_path).generation == 2


def test_keyword_only_index(tmp_path: Path) -> None:
    w = _core.IndexWriter(0)
    w.add(_core.ChunkRecord("a.txt", [], 1, 1, "plain words"))
    w.commit(tmp_path)
    r = _core.IndexReader.open(tmp_path, mmap=False)
    assert not r.has_vectors and r.manifest["embedder"] is None
    assert [h.id for h in r.search("words")] == [0]
    with pytest.raises(ValueError, match="keyword-only"):
        r.search("words", _vec(1, 0, 0))


def test_writer_is_single_use(tmp_path: Path) -> None:
    w = _core.IndexWriter(0)
    w.commit(tmp_path)
    with pytest.raises(RuntimeError, match="already committed"):
        w.add(_core.ChunkRecord("a", [], 1, 1, "t"))
    with pytest.raises(RuntimeError, match="already committed"):
        w.commit(tmp_path)


def test_writer_rejects_bad_embeddings() -> None:
    w = _core.IndexWriter(3)
    rec = _core.ChunkRecord("a", [], 1, 1, "t")
    with pytest.raises(ValueError, match="needs one"):
        w.add(rec)
    with pytest.raises(ValueError, match="3-dimensional"):
        w.add(rec, _vec(1, 0))
    with pytest.raises(TypeError):
        w.add(rec, np.zeros(3, dtype=np.float64))  # type: ignore[arg-type]


def test_metadata_must_be_json(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        _core.IndexWriter(0, None, {"bad": object()})


def test_query_dimension_mismatch_is_value_error(tmp_path: Path) -> None:
    _two_chunk_index(tmp_path)
    r = _core.IndexReader.open(tmp_path)
    with pytest.raises(ValueError, match="3-dimensional"):
        r.search("x", _vec(1, 0))


def test_missing_index_is_oserror(tmp_path: Path) -> None:
    assert not _core.index_exists(tmp_path)
    with pytest.raises(OSError):
        _core.IndexReader.open(tmp_path)


def test_schema_mismatch_raises_schema_error(tmp_path: Path) -> None:
    info = _two_chunk_index(tmp_path)
    manifest = info.dir / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["schema_version"] = _core.SCHEMA_VERSION + 1
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(_core.SchemaError, match="rebuild"):
        _core.IndexReader.open(tmp_path)
    assert issubclass(_core.SchemaError, _core.IndexFormatError)
    assert issubclass(_core.IndexFormatError, _core.NotelightError)


def test_corrupt_file_raises_index_format_error(tmp_path: Path) -> None:
    info = _two_chunk_index(tmp_path)
    (info.dir / "docs.bin").write_bytes(b"garbage")
    with pytest.raises(_core.IndexFormatError):
        _core.IndexReader.open(tmp_path)
