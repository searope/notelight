"""The `notelight` command line, driven through Typer's test runner."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import notelight
from notelight.cli import app

runner = CliRunner()


def _index(folder: Path, *extra: str) -> str:
    result = runner.invoke(app, ["index", str(folder), "--embedder", "hashing", *extra])
    assert result.exit_code == 0, result.output
    return result.output


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"notelight {notelight.__version__}" in result.output


def test_index_then_up_to_date(notes: Path) -> None:
    assert "3 files (3 added" in _index(notes)
    assert "up to date" in _index(notes)


def test_search_human_output(notes: Path) -> None:
    _index(notes)
    result = runner.invoke(app, ["search", "sourdough starter", "--folder", str(notes), "-k", "2"])
    assert result.exit_code == 0, result.output
    first_line = result.output.splitlines()[0]
    assert "recipes/bread.txt:" in first_line


def test_search_json_output(notes: Path) -> None:
    _index(notes)
    result = runner.invoke(app, ["search", "single owner", "-f", str(notes), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    top = payload["results"][0]
    assert top["path"] == "rust/ownership.md"
    assert top["heading_path"] == ["Rust", "Ownership"]
    assert top["location"].startswith("rust/ownership.md:")


def test_search_uses_the_only_index_by_default(notes: Path) -> None:
    _index(notes)
    result = runner.invoke(app, ["search", "mypy", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["results"][0]["path"] == "python/typing.md"


def test_search_with_wrong_embedder_fails(notes: Path) -> None:
    _index(notes)
    result = runner.invoke(app, ["search", "x", "-f", str(notes), "--embedder", "none"])
    assert result.exit_code == 0  # BM25 only is always allowed
    result = runner.invoke(app, ["index", str(notes), "--embedder", "none"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["search", "x", "-f", str(notes), "--embedder", "hashing"])
    assert result.exit_code == 1
    assert "was built with no embedder" in result.output


def test_search_before_indexing(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", "x", "-f", str(tmp_path)])
    assert result.exit_code == 2
    assert "not indexed" in result.output


def test_nothing_indexed(monkeypatch: object) -> None:
    result = runner.invoke(app, ["search", "x"])
    assert result.exit_code == 1
    assert "nothing is indexed" in result.output


def test_index_missing_folder(tmp_path: Path) -> None:
    result = runner.invoke(app, ["index", str(tmp_path / "nope"), "--embedder", "hashing"])
    assert result.exit_code == 2


def test_unknown_embedder(notes: Path) -> None:
    result = runner.invoke(app, ["index", str(notes), "--embedder", "gpt"])
    assert result.exit_code == 2
    assert "unknown embedder" in result.output


def test_status(notes: Path) -> None:
    result = runner.invoke(app, ["status"])
    assert "No indexes yet" in result.output
    _index(notes)
    result = runner.invoke(app, ["status", "--json"])
    (info,) = json.loads(result.output)
    assert info["files"] == 3 and info["embedder"] == "hashing-256"
    result = runner.invoke(app, ["status", str(notes)])
    assert "3 files" in result.output
