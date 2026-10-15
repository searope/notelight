"""Where notelight keeps indexes and downloaded models.

By default indexes live in the user data directory and models in the user
cache directory (via `platformdirs`). Setting `NOTELIGHT_HOME` puts both under
one folder instead, which is what the tests do. `NOTELIGHT_MODEL_CACHE`, if
set, overrides the model directory alone (for example, a shared or
pre-populated cache on an offline machine).

Each indexed folder gets its own directory, named by a hash of its absolute
path::

    <data>/indexes/<key>/
      source.json    the folder this index covers
      lock           single-writer lock file
      cache.sqlite   file hashes, chunk cache, embedding cache
      index/         the Rust core's index (CURRENT, gen-NNNNNN/)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import platformdirs

APP_NAME = "notelight"
HOME_ENV = "NOTELIGHT_HOME"
MODEL_CACHE_ENV = "NOTELIGHT_MODEL_CACHE"


def data_dir() -> Path:
    """Root directory for indexes."""
    home = os.environ.get(HOME_ENV)
    if home:
        return Path(home)
    return platformdirs.user_data_path(APP_NAME, appauthor=False)


def model_cache_dir() -> Path:
    """Directory for downloaded embedding models."""
    override = os.environ.get(MODEL_CACHE_ENV)
    if override:
        return Path(override)
    home = os.environ.get(HOME_ENV)
    if home:
        return Path(home) / "models"
    return platformdirs.user_cache_path(APP_NAME, appauthor=False) / "models"


def indexes_dir() -> Path:
    """Parent directory of all per-folder index directories."""
    return data_dir() / "indexes"


def normalize_folder(folder: str | os.PathLike[str]) -> Path:
    """Absolute, resolved form of `folder` (symlinks followed)."""
    return Path(folder).expanduser().resolve()


def folder_key(folder: str | os.PathLike[str]) -> str:
    """Stable directory name for an indexed folder.

    Uses `os.path.normcase`, so on Windows `C:\\Notes` and `c:\\notes` map to
    the same index, as they name the same folder.
    """
    canonical = os.path.normcase(str(normalize_folder(folder)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def index_dir_for(folder: str | os.PathLike[str]) -> Path:
    """The index directory for `folder` (it may not exist yet)."""
    return indexes_dir() / folder_key(folder)
