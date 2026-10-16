"""Type stubs for the compiled Rust extension (crates/notelight-py)."""

from os import PathLike
from pathlib import Path
from typing import Any, Literal, final

import numpy as np
import numpy.typing as npt

__version__: str
SCHEMA_VERSION: int
#: Names accepted by `IndexReader.search_vector(mode=...)`, in report order.
VECTOR_MODES: list[str]

VectorMode = Literal["int8", "binary", "binary+int8", "ternary+int8"]

def quantize_binary(vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Quantize float32 vectors of shape (n, dim) to packed bits, shape (n, ceil(dim / 8))."""

def hamming(a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8]) -> int:
    """Hamming distance between two packed binary codes of equal length."""

def chunk_text(text: str, max_chars: int = 1200, markdown: bool = True) -> list[Chunk]:
    """Split text into chunks of at most max_chars characters.

    Raises ValueError if max_chars is 0.
    """

def index_exists(root: str | PathLike[str]) -> bool:
    """Whether root holds a committed index."""

class NotelightError(Exception):
    """Base class for notelight errors raised by the Rust core."""

class IndexFormatError(NotelightError):
    """The index on disk is corrupt or not a notelight index. Rebuild it."""

class SchemaError(IndexFormatError):
    """The index was written by an incompatible version of notelight. Rebuild it."""

@final
class Chunk:
    """A contiguous piece of a document.

    byte_start/byte_end are UTF-8 byte offsets into the source (not str indices);
    line_start/line_end are 1-based and inclusive.
    """

    @property
    def text(self) -> str: ...
    @property
    def heading_path(self) -> list[str]: ...
    @property
    def byte_start(self) -> int: ...
    @property
    def byte_end(self) -> int: ...
    @property
    def line_start(self) -> int: ...
    @property
    def line_end(self) -> int: ...

@final
class ChunkRecord:
    """One stored chunk: where it came from and its text."""

    def __init__(
        self, path: str, heading_path: list[str], line_start: int, line_end: int, text: str
    ) -> None: ...
    @property
    def path(self) -> str: ...
    @property
    def heading_path(self) -> list[str]: ...
    @property
    def line_start(self) -> int: ...
    @property
    def line_end(self) -> int: ...
    @property
    def text(self) -> str: ...

@final
class Hit:
    """One hybrid search result. Ranks are 1-based, None if absent from that list."""

    @property
    def id(self) -> int: ...
    @property
    def score(self) -> float: ...
    @property
    def bm25_rank(self) -> int | None: ...
    @property
    def vector_rank(self) -> int | None: ...

@final
class CommitInfo:
    """Result of a successful commit."""

    @property
    def generation(self) -> int: ...
    @property
    def dir(self) -> Path: ...
    @property
    def n_chunks(self) -> int: ...

@final
class IndexWriter:
    """Collects chunks in memory, then writes them as one index generation.

    Single-use: after commit(), further calls raise RuntimeError.
    """

    def __init__(
        self, dim: int, embedder: str | None = None, metadata: dict[str, Any] | None = None
    ) -> None:
        """dim=0 builds a keyword-only index; metadata must be JSON-serializable."""

    def add(self, record: ChunkRecord, embedding: npt.NDArray[np.float32] | None = None) -> int:
        """Add one chunk and return its id. Raises ValueError on a missing or bad embedding."""

    def commit(self, root: str | PathLike[str]) -> CommitInfo:
        """Write a new generation under root and make it live (releases the GIL)."""

    def __len__(self) -> int: ...

@final
class IndexReader:
    """A read-only view of the live index generation under a root directory."""

    @staticmethod
    def open(root: str | PathLike[str], mmap: bool = True) -> IndexReader:
        """Open the live generation. Raises OSError, IndexFormatError or SchemaError."""

    def search(
        self,
        text: str,
        vector: npt.NDArray[np.float32] | None = None,
        k: int = 10,
        depth: int = 50,
        candidates: int = 200,
        rrf_k: float = 60.0,
        phrase_boost: float = 1.0,
    ) -> list[Hit]:
        """Hybrid BM25 + vector search fused with RRF (releases the GIL).

        Chunks containing every `"quoted phrase"` in `text` rank first; a
        chunk with all query terms in order gains `phrase_boost / (rrf_k + 1)`.
        """

    def search_bm25(self, text: str, k: int) -> list[tuple[int, float]]:
        """Keyword search only: (id, bm25_score) pairs."""

    def search_vector(
        self,
        vector: npt.NDArray[np.float32],
        k: int,
        candidates: int = 200,
        mode: VectorMode = "ternary+int8",
    ) -> list[tuple[int, float]]:
        """Vector search only: (id, cosine) pairs.

        mode: "ternary+int8" (default, as in hybrid search), "binary+int8"
        (binary candidates instead of ternary ones), "binary" (Hamming
        distance alone; scores are 1 - 2 * distance / dim) or "int8" (every
        vector, no candidate stage). Raises ValueError on an unknown mode.
        """

    def chunk(self, id: int) -> ChunkRecord:
        """The stored chunk with this id."""

    @property
    def manifest(self) -> dict[str, Any]: ...
    @property
    def dir(self) -> Path: ...
    @property
    def has_vectors(self) -> bool: ...
    def __len__(self) -> int: ...
