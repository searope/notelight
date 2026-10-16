"""The `notelight` command line, driven through Typer's test runner."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import typer
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


def test_search_highlights_query_words(notes: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _index(notes)
    args = ["search", "sourdough starter", "-f", str(notes), "-k", "1"]
    monkeypatch.delenv("NO_COLOR", raising=False)
    colored = runner.invoke(app, args, color=True).output
    mark = typer.style("Sourdough", fg=typer.colors.YELLOW, bold=True)
    assert mark in colored
    assert typer.style("starter", fg=typer.colors.YELLOW, bold=True) in colored
    assert typer.style("bread", fg=typer.colors.YELLOW, bold=True) not in colored
    # Piped output (no terminal) and NO_COLOR get plain text.
    plain = runner.invoke(app, args).output
    assert "\x1b[" not in plain and "Sourdough bread" in plain
    monkeypatch.setenv("NO_COLOR", "1")
    assert "\x1b[" not in runner.invoke(app, args, color=True).output


def test_search_json_output(notes: Path) -> None:
    _index(notes)
    result = runner.invoke(app, ["search", "single owner", "-f", str(notes), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    top = payload["results"][0]
    assert top["path"] == "rust/ownership.md"
    assert top["heading_path"] == ["Rust", "Ownership"]
    assert top["location"].startswith("rust/ownership.md:")


@pytest.mark.parametrize("fmt", ["--json", "--text"])
def test_search_writes_utf8_to_a_pipe_whatever_the_locale(tmp_path: Path, fmt: str) -> None:
    # Regression: on Windows, piped output used the ANSI code page, so a note
    # outside it crashed `notelight search > hits.txt`. PYTHONIOENCODING
    # stands in for that code page on every OS.
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "kyoto.md").write_text("# 京都\n\nKyoto trip, bake at 250 °C.\n", encoding="utf-8")
    _index(notes)
    args = ["search", "kyoto", "-f", str(notes)] + (["--json"] if fmt == "--json" else [])
    done = subprocess.run(
        [sys.executable, "-m", "notelight.cli", *args],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    out = done.stdout.decode("utf-8")  # strict: fails on cp1252 bytes
    assert "京都" in out
    assert "250 °C" in out
    if fmt == "--json":
        assert json.loads(out)["results"][0]["path"] == "kyoto.md"


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


def test_mcp_refuses_a_folder_that_is_not_indexed(tmp_path: Path) -> None:
    result = runner.invoke(app, ["mcp", "--folder", str(tmp_path)])
    assert result.exit_code == 2
    assert "is not indexed" in result.output


def test_reindex_without_embedder_keeps_the_index_embedder(notes: Path) -> None:
    for name, model_id in (("none", None), ("hashing", "hashing-256")):
        result = runner.invoke(app, ["index", str(notes), "--embedder", name])
        assert result.exit_code == 0, result.output
        # A plain re-index must not switch to bge-small (download + full re-embed).
        result = runner.invoke(app, ["index", str(notes), "--offline"])
        assert result.exit_code == 0, result.output
        assert "up to date" in result.output
        (info,) = json.loads(runner.invoke(app, ["status", "--json"]).output)
        assert info["embedder"] == model_id
