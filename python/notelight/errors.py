"""Exceptions raised by notelight.

Everything derives from `NotelightError`, so callers can catch one class. The
index errors come from the Rust core; the rest are raised by the Python layer.
"""

from __future__ import annotations

from notelight._core import IndexFormatError, NotelightError, SchemaError

__all__ = [
    "DatasetNotAvailableError",
    "EmbedderMismatchError",
    "IndexBusyError",
    "IndexFormatError",
    "IndexNotFoundError",
    "ModelNotAvailableError",
    "NotelightError",
    "SchemaError",
]


class IndexNotFoundError(NotelightError):
    """No index exists for the requested folder. Run `notelight index <folder>` first."""


class IndexBusyError(NotelightError):
    """Another process is updating this index (the single-writer lock is held)."""


class EmbedderMismatchError(NotelightError):
    """The query embedder differs from the one the index was built with."""


class ModelNotAvailableError(NotelightError):
    """The embedding model is not in the cache and downloading is not allowed."""


class DatasetNotAvailableError(NotelightError):
    """An evaluation dataset could not be downloaded, verified or read."""
