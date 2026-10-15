"""Text embedders.

The indexer and searcher depend only on the `Embedder` protocol, so tests can
use the deterministic `HashingEmbedder` with no model download or network.

`OnnxEmbedder` runs a sentence-embedding model with ONNX Runtime and Hugging
Face `tokenizers`. The default model is `BAAI/bge-small-en-v1.5` (384
dimensions, MIT license). Its files are pinned to one revision and verified by
SHA-256 when downloaded, so the vectors in an index never silently change.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from notelight import paths
from notelight.errors import ModelNotAvailableError

__all__ = [
    "BGE_SMALL_EN_V15",
    "EMBEDDER_NAMES",
    "Embedder",
    "HashingEmbedder",
    "ModelFile",
    "ModelSpec",
    "OnnxEmbedder",
    "ensure_model_files",
    "make_embedder",
    "name_for_model_id",
]

Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    """Turns text into L2-normalized float32 vectors."""

    @property
    def model_id(self) -> str:
        """Identifier recorded in the index manifest; two embedders with the
        same id must produce the same vectors."""
        ...

    @property
    def dim(self) -> int:
        """Vector dimension."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """Embed passages for indexing. Returns shape `(len(texts), dim)`."""
        ...

    def embed_query(self, text: str) -> Vectors:
        """Embed a search query. Returns shape `(dim,)`."""
        ...


def _l2_normalize(m: Vectors) -> Vectors:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return (m / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


# --------------------------------------------------------------------------
# Hashing embedder (tests, offline baseline)
# --------------------------------------------------------------------------

_WORD = re.compile(r"\w+")


class HashingEmbedder:
    """Deterministic bag-of-words embedder based on feature hashing.

    Each lowercase word adds +1 or -1 to one of `dim` buckets, chosen by a
    hash. Texts sharing words get similar vectors, which is enough to test
    the pipeline end to end. It is not a semantic model.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def model_id(self) -> str:
        return f"hashing-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, text: str) -> Vectors:
        v = np.zeros(self._dim, dtype=np.float32)
        for word in _WORD.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
            v[h % self._dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _l2_normalize(v)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = self._embed(text)
        return out

    def embed_query(self, text: str) -> Vectors:
        return self._embed(text)


# --------------------------------------------------------------------------
# Pinned model files
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFile:
    """One file of a model repository, pinned by size and SHA-256."""

    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelSpec:
    """A sentence-embedding model on the Hugging Face Hub, pinned to a revision."""

    repo: str
    revision: str
    onnx: ModelFile
    tokenizer: ModelFile
    dim: int
    max_tokens: int = 512
    query_prefix: str = ""
    base_url: str = "https://huggingface.co"

    @property
    def model_id(self) -> str:
        """Repository plus short revision, e.g. `BAAI/bge-small-en-v1.5@5c38ec7c405e`."""
        return f"{self.repo}@{self.revision[:12]}"

    def url(self, file: ModelFile) -> str:
        """Download URL of one file at the pinned revision."""
        return f"{self.base_url}/{self.repo}/resolve/{self.revision}/{file.path}"


BGE_SMALL_EN_V15 = ModelSpec(
    repo="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    onnx=ModelFile(
        path="onnx/model.onnx",
        sha256="828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
        size=133_093_490,
    ),
    tokenizer=ModelFile(
        path="tokenizer.json",
        sha256="d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        size=711_396,
    ),
    dim=384,
    # bge v1.5 recommends this instruction for short queries against passages.
    query_prefix="Represent this sentence for searching relevant passages: ",
)

DownloadCallback = Callable[[ModelFile], None]


def _local_path(spec: ModelSpec, file: ModelFile, cache_dir: Path) -> Path:
    return cache_dir / spec.repo.replace("/", "--") / spec.revision / file.path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path, expected: ModelFile) -> None:
    """Stream `url` to a temporary file next to `dest`, verify, then rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".download-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
            for block in iter(lambda: resp.read(1 << 20), b""):
                h.update(block)
                out.write(block)
        digest = h.hexdigest()
        if digest != expected.sha256:
            raise ModelNotAvailableError(
                f"{expected.path}: SHA-256 mismatch (expected {expected.sha256}, got {digest}); "
                "the download was corrupted or the file changed upstream"
            )
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def ensure_model_files(
    spec: ModelSpec,
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
    on_download: DownloadCallback | None = None,
) -> tuple[Path, Path]:
    """Return local paths of the model and tokenizer, downloading them if needed.

    A cached file is trusted if its size matches; a freshly downloaded file is
    checked against its SHA-256 before it is moved into the cache.

    Raises:
        ModelNotAvailableError: a file is missing and `offline` is set, or a
            download failed verification.
    """
    cache = cache_dir if cache_dir is not None else paths.model_cache_dir()
    local: list[Path] = []
    for file in (spec.onnx, spec.tokenizer):
        dest = _local_path(spec, file, cache)
        if not (dest.is_file() and dest.stat().st_size == file.size):
            if offline:
                raise ModelNotAvailableError(
                    f"{spec.repo} is not in the model cache ({cache}) and --offline is set; "
                    "run once without --offline to download it"
                )
            if on_download is not None:
                on_download(file)
            _download(spec.url(file), dest, file)
        local.append(dest)
    return local[0], local[1]


# --------------------------------------------------------------------------
# ONNX embedder
# --------------------------------------------------------------------------


class OnnxEmbedder:
    """Sentence embeddings with ONNX Runtime: CLS pooling, then L2 normalization.

    The model is located (and downloaded if needed) on the first call that
    embeds text, not in the constructor. A sync that finds nothing to embed
    never loads ONNX Runtime; call `load()` to fail early instead.
    """

    def __init__(
        self,
        spec: ModelSpec = BGE_SMALL_EN_V15,
        *,
        cache_dir: Path | None = None,
        offline: bool = False,
        batch_size: int = 32,
        on_download: DownloadCallback | None = None,
    ) -> None:
        self._spec = spec
        self._cache_dir = cache_dir
        self._offline = offline
        self._batch_size = batch_size
        self._on_download = on_download
        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: set[str] = set()

    def load(self) -> None:
        """Locate or download the model files and start the ONNX session.

        Raises:
            ModelNotAvailableError: the model is not cached and `offline` is
                set, or a download failed verification.
        """
        if self._session is not None:
            return
        # Imported here: both are slow to import and unused by keyword-only
        # indexes, up-to-date syncs and the test suite.
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path, tokenizer_path = ensure_model_files(
            self._spec,
            cache_dir=self._cache_dir,
            offline=self._offline,
            on_download=self._on_download,
        )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=self._spec.max_tokens)
        tokenizer.enable_padding()
        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in session.get_inputs()}
        self._tokenizer = tokenizer
        self._session = session

    @property
    def model_id(self) -> str:
        return self._spec.model_id

    @property
    def dim(self) -> int:
        return self._spec.dim

    def _run(self, texts: Sequence[str]) -> Vectors:
        self.load()
        encodings = self._tokenizer.encode_batch(list(texts))
        feeds = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        hidden = self._session.run(None, feeds)[0]  # (batch, seq, dim)
        return _l2_normalize(np.asarray(hidden[:, 0, :], dtype=np.float32))  # CLS token

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        # Batch texts of similar length together so little time goes to padding.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), self._batch_size):
            idx = order[start : start + self._batch_size]
            out[idx] = self._run([texts[i] for i in idx])
        return out

    def embed_query(self, text: str) -> Vectors:
        vector: Vectors = self._run([self._spec.query_prefix + text])[0]
        return vector


# --------------------------------------------------------------------------
# Registry used by the CLI
# --------------------------------------------------------------------------

#: Names accepted by `--embedder`. "none" builds a keyword-only (BM25) index.
EMBEDDER_NAMES = ("bge-small", "hashing", "none")


def make_embedder(
    name: str, *, offline: bool = False, on_download: DownloadCallback | None = None
) -> Embedder | None:
    """Build the embedder registered under `name` (None for "none")."""
    if name == "bge-small":
        return OnnxEmbedder(BGE_SMALL_EN_V15, offline=offline, on_download=on_download)
    if name == "hashing":
        return HashingEmbedder()
    if name == "none":
        return None
    raise ValueError(f"unknown embedder {name!r}; choose one of {', '.join(EMBEDDER_NAMES)}")


def name_for_model_id(model_id: str | None) -> str | None:
    """The registry name whose embedder produces `model_id`, if any."""
    if model_id is None:
        return "none"
    if model_id == BGE_SMALL_EN_V15.model_id:
        return "bge-small"
    if model_id == HashingEmbedder().model_id:
        return "hashing"
    return None
