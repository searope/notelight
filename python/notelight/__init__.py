"""notelight: private semantic search over your own files, for you and your AI agents.

The main entry points:

- `sync(folder, embedder)` creates or updates a folder's index;
- `Searcher(folder, embedder).search(query)` queries it;
- `OnnxEmbedder` (bge-small by default) and `HashingEmbedder` produce vectors.

The Rust core is available as `notelight._core` for lower-level use.
"""

from notelight._core import __version__ as core_version
from notelight._core import chunk_text, hamming, quantize_binary
from notelight.embed import Embedder, HashingEmbedder, OnnxEmbedder
from notelight.errors import (
    EmbedderMismatchError,
    IndexBusyError,
    IndexFormatError,
    IndexNotFoundError,
    ModelNotAvailableError,
    NotelightError,
    SchemaError,
)
from notelight.indexer import SyncStats, sync
from notelight.search import Searcher, SearchResult

__version__ = core_version

__all__ = [
    "Embedder",
    "EmbedderMismatchError",
    "HashingEmbedder",
    "IndexBusyError",
    "IndexFormatError",
    "IndexNotFoundError",
    "ModelNotAvailableError",
    "NotelightError",
    "OnnxEmbedder",
    "SchemaError",
    "SearchResult",
    "Searcher",
    "SyncStats",
    "__version__",
    "chunk_text",
    "core_version",
    "hamming",
    "quantize_binary",
    "sync",
]
