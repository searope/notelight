"""A pure-NumPy implementation of notelight's vector search, as a baseline.

`NumpyVectors` mirrors the Rust core's `Vectors` (see `vectors.rs`): the same
binary codes, the same int8 quantization and the same tie-breaking, written
the way a NumPy user would. The evaluation runs both on the same queries and
reports their latency and how often they return the same top 10.

Two differences are deliberate and documented in `docs/results.md`:

- NumPy has no int8 x float32 matrix-vector kernel, so the int8 codes are
  kept *dequantized* in float32 (4x the memory of the Rust index).
- NumPy may run its matrix-vector product on several threads; the Rust core
  is single-threaded per query.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["NumpyVectors"]

F32 = npt.NDArray[np.float32]
Hits = list[tuple[int, float]]

#: Ternary threshold, as a fraction of a vector's mean |value| (see `quant.rs`).
_TERNARY_THRESHOLD = np.float32(0.7)

_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_rows(x: npt.NDArray[np.uint8]) -> npt.NDArray[np.int64]:
    """Set bits per row of a packed bit matrix."""
    # numpy >= 2.0 has a native popcount; older versions use a lookup table.
    bitwise_count: Callable[[Any], Any] | None = getattr(np, "bitwise_count", None)
    counts = bitwise_count(x) if bitwise_count is not None else _POPCOUNT_TABLE[x]
    return np.asarray(counts.sum(axis=1, dtype=np.int64))


def _ternary_masks(m: F32) -> npt.NDArray[np.uint8]:
    """Packed ternary masks: bit set where `|x| > 0.7 * mean(|x|)` of its row."""
    a = np.abs(np.atleast_2d(m))
    threshold = _TERNARY_THRESHOLD * a.mean(axis=1, dtype=np.float32, keepdims=True)
    return np.packbits(a > threshold, axis=1, bitorder="little")


def _l2_normalize(m: F32) -> F32:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return np.asarray(np.where(norms > 0, m / np.where(norms > 0, norms, 1), m), dtype=np.float32)


def _quantize_int8(m: F32) -> tuple[F32, F32]:
    """Per-row symmetric int8 codes (as float32) and scales, like `quantize_int8`.

    Rust's `f32::round` rounds halves away from zero; `np.round` rounds them
    to even, so this rounds explicitly.
    """
    max_abs = np.abs(m).max(axis=1)
    scales = (max_abs / np.float32(127.0)).astype(np.float32)
    safe = np.where(scales > 0, scales, np.float32(1.0))[:, None]
    y = m / safe
    codes = np.clip(np.sign(y) * np.floor(np.abs(y) + 0.5), -127, 127)
    codes[scales == 0] = 0
    return codes.astype(np.float32), scales


def _top_k_by_score(ids: npt.NDArray[Any], scores: npt.NDArray[Any], k: int) -> Hits:
    """Best `k` of `(ids, scores)`: score descending, then id ascending."""
    if len(ids) > k:
        keep = np.argpartition(-scores, k - 1)[:k]
        # Partitioning splits ties at the cut-off arbitrarily; take every
        # item tied with the k-th score so the id tie-break below is exact.
        cutoff = np.min(scores[keep])
        keep = np.flatnonzero(scores >= cutoff)
        ids, scores = ids[keep], scores[keep]
    order = np.lexsort((ids, -scores))[:k]
    return [(int(ids[i]), float(scores[i])) for i in order]


class NumpyVectors:
    """Exhaustive vector search in NumPy, mirroring the Rust core.

    `vectors` has shape `(n, dim)`; rows are L2-normalized here, as the Rust
    core does on insert.
    """

    def __init__(self, vectors: F32) -> None:
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("vectors must be a non-empty (n, dim) array")
        self.f32 = _l2_normalize(np.ascontiguousarray(vectors, dtype=np.float32))
        self.dim = self.f32.shape[1]
        self.bits = np.packbits(self.f32 > 0, axis=1, bitorder="little")
        self.codes, self.scales = _quantize_int8(self.f32)
        # Like the Rust index, masks come from the int8 codes, not the floats.
        self.masks = _ternary_masks(self.codes)
        self._ids = np.arange(len(self.f32), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.f32)

    def _hamming(self, q: F32) -> npt.NDArray[np.int64]:
        q_bits = np.packbits(q > 0, bitorder="little")
        return _popcount_rows(np.bitwise_xor(self.bits, q_bits))

    def search_f32(self, query: F32, k: int) -> Hits:
        """Exact cosine similarity over every vector."""
        q = _l2_normalize(query)
        return _top_k_by_score(self._ids, self.f32 @ q, k)

    def search_int8(self, query: F32, k: int) -> Hits:
        """Int8 scores over every vector (codes held as float32)."""
        q = _l2_normalize(query)
        return _top_k_by_score(self._ids, (self.codes @ q) * self.scales, k)

    def search_binary(self, query: F32, k: int) -> Hits:
        """Rank by Hamming distance, ties by id; score `1 - 2 * d / dim`."""
        dist = self._hamming(_l2_normalize(query))
        ids = self._smallest(dist, k)
        return [(int(i), 1.0 - 2.0 * float(dist[i]) / self.dim) for i in ids]

    def search_binary_int8(self, query: F32, k: int, candidates: int = 200) -> Hits:
        """Binary candidates, then int8 rescoring: the Rust default."""
        q = _l2_normalize(query)
        cands = self._smallest(self._hamming(q), max(candidates, k))
        scores = (self.codes[cands] @ q) * self.scales[cands]
        return _top_k_by_score(cands, scores, k)

    def search_ternary_int8(self, query: F32, k: int, candidates: int = 200) -> Hits:
        """Ternary candidates, then int8 rescoring: the Rust default.

        The ternary dot is `popcount(m) - 2 * popcount(m & (sign_q ^ sign_d))`
        with `m = mask_q & mask_d`; candidates are the smallest `dim - dot`.
        """
        q = _l2_normalize(query)
        q_bits = np.packbits(q > 0, bitorder="little")
        both = np.bitwise_and(self.masks, _ternary_masks(q))
        differ = np.bitwise_and(both, np.bitwise_xor(self.bits, q_bits))
        dot = _popcount_rows(both) - 2 * _popcount_rows(differ)
        cands = self._smallest(self.dim - dot, max(candidates, k))
        scores = (self.codes[cands] @ q) * self.scales[cands]
        return _top_k_by_score(cands, scores, k)

    def _smallest(self, dist: npt.NDArray[np.int64], k: int) -> npt.NDArray[np.int64]:
        """Ids of the `k` smallest distances, ordered by (distance, id)."""
        # One unique integer key per row gives the Rust tie-break for free.
        key = dist * len(self) + self._ids
        if len(key) > k:
            part = np.argpartition(key, k - 1)[:k]
            return part[np.argsort(key[part])]
        return np.argsort(key)
