"""Data and model directory resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from notelight import paths


def test_home_holds_indexes_and_models(
    notelight_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(paths.MODEL_CACHE_ENV, raising=False)
    assert paths.data_dir() == notelight_home
    assert paths.indexes_dir() == notelight_home / "indexes"
    assert paths.model_cache_dir() == notelight_home / "models"


def test_model_cache_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.MODEL_CACHE_ENV, str(tmp_path / "shared"))
    assert paths.model_cache_dir() == tmp_path / "shared"


def test_folder_key_is_stable_and_path_based(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    assert paths.folder_key(a) == paths.folder_key(tmp_path / "x" / ".." / "a")
    assert paths.folder_key(a) != paths.folder_key(b)
    assert len(paths.folder_key(a)) == 16
    assert paths.index_dir_for(a).parent == paths.indexes_dir()
