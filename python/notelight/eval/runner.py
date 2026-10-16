"""Run every retrieval method on a dataset and score it.

The corpus is embedded once per (dataset, embedder) and cached under
`paths.eval_cache_dir()/runs/`, together with the query embeddings and a
notelight index built from them, so a second run only searches.

Each BEIR document becomes one indexed chunk: its title is the heading path
and its text is the chunk text, exactly as `notelight index` would store a
section of a note. Documents longer than the model's 512 tokens are
truncated by the embedder, as in the BEIR reference setup; BM25 sees the
whole text. The notes benchmark (`notes.py`) is already chunked by the real
chunker, and keeps each chunk's heading path and line range.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from notelight import _core, paths
from notelight.embed import Embedder
from notelight.errors import IndexFormatError
from notelight.eval.baseline import NumpyVectors
from notelight.eval.datasets import Dataset
from notelight.eval.metrics import mean, mrr_at_k, ndcg_at_k, overlap_at_k, recall_at_k
from notelight.indexer import embedding_text

__all__ = [
    "BASELINE_SEARCHES",
    "METHODS",
    "BaselineRow",
    "DatasetResult",
    "MethodResult",
    "Progress",
    "StyleResult",
    "evaluate",
]

F32 = npt.NDArray[np.float32]

#: `IndexReader.search`'s default phrase bonus (the Rust core's `DEFAULT_PHRASE_BOOST`).
DEFAULT_PHRASE_BOOST = 1.0

#: Every method `evaluate` can run, in report order. `f32` is exact cosine
#: similarity in NumPy (the quality ceiling for the vector methods); the four
#: quantized modes are the Rust core's `VectorMode`s; `bm25` is keyword search
#: alone; `hybrid` is what `notelight search` does: BM25 and ternary+int8,
#: fused with reciprocal rank fusion.
METHODS: tuple[str, ...] = ("f32", *_core.VECTOR_MODES, "bm25", "hybrid")

#: Searches implemented both in Rust and in NumPy (`NumpyVectors`).
BASELINE_SEARCHES: tuple[str, ...] = ("int8", "binary", "binary+int8", "ternary+int8")

#: `progress(stage, done, total)`, with stage "embed-docs" or "embed-queries".
Progress = Callable[[str, int, int], None]

_EMBED_BATCH = 256


@dataclass(frozen=True)
class StyleResult:
    """Quality of one method on the queries of one style."""

    style: str
    queries: int
    ndcg: float
    recall: float
    mrr: float


@dataclass(frozen=True)
class MethodResult:
    """Quality and speed of one method on one dataset."""

    method: str
    ndcg: float
    recall: float
    #: Median time per query in milliseconds, search only (no embedding).
    latency_ms: float
    #: Mean reciprocal rank of the first relevant result in the top k.
    mrr: float = 0.0
    #: Per query style, for datasets that label styles, in query order.
    by_style: list[StyleResult] = field(default_factory=list)


@dataclass(frozen=True)
class BaselineRow:
    """The same search run by the Rust core and by `NumpyVectors`."""

    search: str
    rust_ms: float
    numpy_ms: float
    #: Mean fraction of Rust's top k that NumPy also returns.
    agreement: float


@dataclass
class DatasetResult:
    """Everything measured on one dataset."""

    dataset: str
    docs: int
    queries: int
    embedder: str
    k: int
    candidates: int
    #: Hybrid bonus for query terms in order, as passed to `evaluate`.
    phrase_boost: float = DEFAULT_PHRASE_BOOST
    methods: list[MethodResult] = field(default_factory=list)
    baseline: list[BaselineRow] = field(default_factory=list)

    def method(self, name: str) -> MethodResult:
        """The result for method `name`.

        Raises:
            KeyError: the method was not run.
        """
        for m in self.methods:
            if m.method == name:
                return m
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _save_npy(path: Path, array: F32) -> None:
    """Write `array` atomically: a crash never leaves a truncated cache file."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def _load_npy(path: Path, shape: tuple[int, int]) -> F32 | None:
    if not path.is_file():
        return None
    try:
        array = np.load(path)
    except (OSError, ValueError):
        return None
    if array.shape != shape or array.dtype != np.float32:
        return None
    return np.asarray(array, dtype=np.float32)


def _embed_documents(texts: Sequence[str], embedder: Embedder, progress: Progress | None) -> F32:
    out = np.zeros((len(texts), embedder.dim), dtype=np.float32)
    # Similar lengths in one batch waste little time on padding.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for start in range(0, len(order), _EMBED_BATCH):
        idx = order[start : start + _EMBED_BATCH]
        out[idx] = embedder.embed_documents([texts[i] for i in idx])
        if progress is not None:
            progress("embed-docs", min(start + _EMBED_BATCH, len(order)), len(order))
    return out


def _prepare(
    dataset: Dataset,
    query_ids: list[str],
    embedder: Embedder,
    cache_dir: Path | None,
    progress: Progress | None,
) -> tuple[_core.IndexReader, F32, F32]:
    """Embeddings for the corpus and queries, and an index over the corpus,
    all cached per (dataset, embedder)."""
    base = cache_dir if cache_dir is not None else paths.eval_cache_dir()
    run_dir = base / "runs" / f"{dataset.name}-{dataset.fingerprint}" / _slug(embedder.model_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    n, dim = len(dataset.doc_ids), embedder.dim

    doc_path = run_dir / "docs.npy"
    doc_vectors = _load_npy(doc_path, (n, dim))
    if doc_vectors is None:
        texts = [
            embedding_text(heading_path, text)
            for heading_path, text in zip(_heading_paths(dataset), dataset.texts, strict=True)
        ]
        doc_vectors = _embed_documents(texts, embedder, progress)
        _save_npy(doc_path, doc_vectors)

    # Query vectors are cached with the ids they belong to, so a run with
    # --max-queries does not poison a later full run.
    ids_path, query_path = run_dir / "queries.json", run_dir / "queries.npy"
    cached_ids: list[str] = (
        json.loads(ids_path.read_text(encoding="utf-8")) if ids_path.is_file() else []
    )
    all_vectors = _load_npy(query_path, (len(cached_ids), dim)) if cached_ids else None
    by_id = dict(zip(cached_ids, all_vectors, strict=True)) if all_vectors is not None else {}
    missing = [qid for qid in query_ids if qid not in by_id]
    for i, qid in enumerate(missing):
        by_id[qid] = embedder.embed_query(dataset.queries[qid])
        if progress is not None:
            progress("embed-queries", i + 1, len(missing))
    if missing:
        ids = list(by_id)
        _save_npy(query_path, np.stack([by_id[q] for q in ids]).astype(np.float32))
        ids_path.write_text(json.dumps(ids), encoding="utf-8")
    query_vectors = np.stack([by_id[q] for q in query_ids]).astype(np.float32)

    reader = _open_or_build_index(run_dir / "index", dataset, doc_vectors, embedder)
    return reader, doc_vectors, query_vectors


def _heading_paths(dataset: Dataset) -> list[list[str]]:
    if dataset.heading_paths is not None:
        return dataset.heading_paths
    return [[title] if title else [] for title in dataset.titles]


def _open_or_build_index(
    root: Path, dataset: Dataset, doc_vectors: F32, embedder: Embedder
) -> _core.IndexReader:
    if _core.index_exists(root):
        try:
            reader = _core.IndexReader.open(root)
            if len(reader) == len(dataset.doc_ids) and (
                reader.manifest.get("embedder") == embedder.model_id
            ):
                return reader
            del reader
        except IndexFormatError:
            pass  # written by another schema version: rebuild below
        shutil.rmtree(root, ignore_errors=True)
    writer = _core.IndexWriter(
        embedder.dim,
        embedder.model_id,
        {"dataset": dataset.name, "fingerprint": dataset.fingerprint},
    )
    locations = dataset.locations or [(doc_id, 1, 1) for doc_id in dataset.doc_ids]
    for (path, start, end), heading_path, text, vector in zip(
        locations, _heading_paths(dataset), dataset.texts, doc_vectors, strict=True
    ):
        writer.add(_core.ChunkRecord(path, heading_path, start, end, text), vector)
    writer.commit(root)
    return _core.IndexReader.open(root)


def _means(scores: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Mean nDCG, recall and MRR over per-query (nDCG, recall, MRR) triples."""
    return (
        mean([s[0] for s in scores]),
        mean([s[1] for s in scores]),
        mean([s[2] for s in scores]),
    )


def _timed(fn: Callable[..., list[int]], *args: Any) -> tuple[list[int], float]:
    """`fn(*args)` and its wall time in milliseconds."""
    start = time.perf_counter()
    ids = fn(*args)
    return ids, (time.perf_counter() - start) * 1000.0


def evaluate(
    dataset: Dataset,
    embedder: Embedder,
    *,
    methods: Sequence[str] = METHODS,
    k: int = 10,
    candidates: int = 200,
    depth: int = 50,
    phrase_boost: float = DEFAULT_PHRASE_BOOST,
    max_queries: int | None = None,
    baseline: bool = True,
    cache_dir: Path | None = None,
    progress: Progress | None = None,
) -> DatasetResult:
    """Score `methods` on `dataset` with recall@k, nDCG@k and MRR@k.

    `candidates` is the number of binary or ternary candidates re-scored with
    int8 codes, `depth` the number of results each retriever contributes
    to hybrid fusion, and `phrase_boost` the hybrid bonus for query terms in
    order (0 turns it off); the defaults are the ones `notelight search` uses.
    `max_queries` evaluates only the first queries, for quick runs. With
    `baseline`, the searches in `BASELINE_SEARCHES` also run in NumPy.

    Raises:
        ValueError: an unknown method name or `k` < 1.
    """
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown method(s) {', '.join(unknown)}; choose from {METHODS}")
    if k < 1:
        raise ValueError("k must be at least 1")
    query_ids = list(dataset.queries)[:max_queries]
    reader, doc_vectors, query_vectors = _prepare(dataset, query_ids, embedder, cache_dir, progress)
    doc_ids = dataset.doc_ids
    numpy_vectors = NumpyVectors(doc_vectors)

    def run(method: str, qi: int) -> list[int]:
        text, vec = dataset.queries[query_ids[qi]], query_vectors[qi]
        has_vec = bool(np.any(vec))  # an all-zero query has no direction
        if method == "bm25":
            return [i for i, _ in reader.search_bm25(text, k)]
        if method == "hybrid":
            hits = reader.search(
                text, vec if has_vec else None, k=k, depth=depth, phrase_boost=phrase_boost
            )
            return [h.id for h in hits]
        if not has_vec:
            return []
        if method == "f32":
            return [i for i, _ in numpy_vectors.search_f32(vec, k)]
        mode = cast("_core.VectorMode", method)
        return [i for i, _ in reader.search_vector(vec, k, candidates, mode)]

    result = DatasetResult(
        dataset=dataset.name,
        docs=len(doc_ids),
        queries=len(query_ids),
        embedder=embedder.model_id,
        k=k,
        candidates=candidates,
        phrase_boost=phrase_boost,
    )
    if not query_ids:
        return result
    for method in methods:
        run(method, 0)  # warm-up: page in the index
        scores: dict[str, tuple[float, float, float]] = {}
        times = []
        for qi, qid in enumerate(query_ids):
            ids, ms = _timed(run, method, qi)
            relevant = dataset.qrels[qid]
            ranked = dataset.judged_ranking([doc_ids[i] for i in ids], relevant)
            scores[qid] = (
                ndcg_at_k(ranked, relevant, k),
                recall_at_k(ranked, relevant, k),
                mrr_at_k(ranked, relevant, k),
            )
            times.append(ms)
        by_style = []
        styles = dataset.query_styles
        for style in dict.fromkeys(styles[q] for q in query_ids if q in styles):
            subset = [s for q, s in scores.items() if styles.get(q) == style]
            by_style.append(StyleResult(style, len(subset), *_means(subset)))
        ndcg, recall, mrr = _means(list(scores.values()))
        result.methods.append(
            MethodResult(method, ndcg, recall, statistics.median(times), mrr, by_style)
        )

    if baseline:

        def run_numpy(search: str, qi: int) -> list[int]:
            vec = query_vectors[qi]
            if search == "int8":
                hits = numpy_vectors.search_int8(vec, k)
            elif search == "binary":
                hits = numpy_vectors.search_binary(vec, k)
            elif search == "binary+int8":
                hits = numpy_vectors.search_binary_int8(vec, k, candidates)
            else:
                hits = numpy_vectors.search_ternary_int8(vec, k, candidates)
            return [i for i, _ in hits]

        for search in BASELINE_SEARCHES:
            rust_times, numpy_times, agreements = [], [], []
            run_numpy(search, 0)  # warm-up
            for qi in range(len(query_ids)):
                if not np.any(query_vectors[qi]):
                    continue
                rust_ids, rust_ms = _timed(run, search, qi)
                numpy_ids, numpy_ms = _timed(run_numpy, search, qi)
                rust_times.append(rust_ms)
                numpy_times.append(numpy_ms)
                agreements.append(overlap_at_k(rust_ids, numpy_ids, k))
            if rust_times:
                result.baseline.append(
                    BaselineRow(
                        search,
                        statistics.median(rust_times),
                        statistics.median(numpy_times),
                        mean(agreements),
                    )
                )
    return result
