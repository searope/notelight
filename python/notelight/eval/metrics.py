"""Retrieval metrics, matching the BEIR reference evaluation (`pytrec_eval`).

All functions take one query's ranked document ids (best first) and its
relevance judgments (doc id -> grade, grades above zero only).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = ["mean", "ndcg_at_k", "overlap_at_k", "recall_at_k"]


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Fraction of the relevant documents that appear in the top `k`.

    Raises:
        ValueError: `relevant` is empty (recall is undefined) or `k` < 1.
    """
    if not relevant:
        raise ValueError("recall is undefined for a query with no relevant documents")
    if k < 1:
        raise ValueError("k must be at least 1")
    found = sum(1 for doc in ranked[:k] if doc in relevant)
    return found / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Normalized discounted cumulative gain at `k`.

    The gain of a document is its grade (linear gain, as in `trec_eval` and
    BEIR), discounted by `log2(rank + 1)` for 1-based ranks. The result is
    divided by the gain of the ideal ranking, so it is 1.0 when the top `k`
    holds the most relevant documents in order.

    Raises:
        ValueError: `relevant` is empty or `k` < 1.
    """
    if not relevant:
        raise ValueError("nDCG is undefined for a query with no relevant documents")
    if k < 1:
        raise ValueError("k must be at least 1")
    dcg = sum(
        relevant.get(doc, 0) / math.log2(rank + 1) for rank, doc in enumerate(ranked[:k], start=1)
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg


def overlap_at_k(a: Sequence[object], b: Sequence[object], k: int) -> float:
    """Fraction of the top `k` of `a` that is also in the top `k` of `b`.

    Used to check that two implementations of the same search agree. Two
    empty lists agree fully.
    """
    top_a, top_b = list(a[:k]), set(b[:k])
    if not top_a:
        return 1.0 if not top_b else 0.0
    return sum(1 for x in top_a if x in top_b) / len(top_a)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0
