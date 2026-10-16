"""The evaluation harness: metrics, datasets, the NumPy baseline and the runner.

Nothing here touches the network: datasets are synthetic, "downloads" are
served from file:// URLs, and the embedder is the HashingEmbedder.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import CountingEmbedder
from typer.testing import CliRunner

from notelight import _core
from notelight.cli import app
from notelight.embed import HashingEmbedder
from notelight.errors import DatasetNotAvailableError
from notelight.eval import (
    BASELINE_SEARCHES,
    METHODS,
    DatasetSpec,
    NumpyVectors,
    ensure_dataset,
    evaluate,
    format_markdown,
    load_beir,
    load_dataset,
    ndcg_at_k,
    overlap_at_k,
    recall_at_k,
    runner,
)

# --------------------------------------------------------------------------
# Metrics, against hand-computed values
# --------------------------------------------------------------------------


def test_recall_counts_relevant_documents_in_the_top_k() -> None:
    relevant = {"a": 1, "b": 2, "c": 1, "d": 1}
    assert recall_at_k(["a", "x", "b", "y"], relevant, 10) == 0.5
    assert recall_at_k(["x", "a", "b"], relevant, 1) == 0.0
    assert recall_at_k(["a", "b", "c", "d"], relevant, 4) == 1.0
    assert recall_at_k([], relevant, 10) == 0.0


def test_ndcg_matches_a_hand_computed_example() -> None:
    # Ranking: b (grade 2) at rank 1, x at 2, a (grade 1) at rank 3.
    # DCG  = 2/log2(2) + 0 + 1/log2(4) = 2 + 0.5 = 2.5
    # IDCG = 2/log2(2) + 1/log2(3) = 2 + 0.63093 = 2.63093
    relevant = {"a": 1, "b": 2}
    expected = 2.5 / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["b", "x", "a"], relevant, 10) == pytest.approx(expected)
    assert ndcg_at_k(["b", "a"], relevant, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], relevant, 10) == 0.0


def test_ndcg_uses_linear_gain_and_cuts_off_at_k() -> None:
    # Linear gain (trec_eval): a grade-2 document counts twice a grade-1 one.
    relevant = {"a": 1, "b": 2}
    worse_first = ndcg_at_k(["a", "b"], relevant, 2)
    assert worse_first == pytest.approx((1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3)))
    # At k=1 the ideal list is just "b", and "a" alone earns 1/2 of it.
    assert ndcg_at_k(["a", "b"], relevant, 1) == pytest.approx(0.5)


def test_metrics_reject_undefined_cases() -> None:
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k(["a"], {}, 10)
    with pytest.raises(ValueError, match="undefined"):
        ndcg_at_k(["a"], {}, 10)
    with pytest.raises(ValueError, match="k must be"):
        ndcg_at_k(["a"], {"a": 1}, 0)


def test_overlap() -> None:
    assert overlap_at_k([1, 2, 3, 4], [3, 1, 9, 2], 3) == pytest.approx(2 / 3)
    assert overlap_at_k([], [], 10) == 1.0
    assert overlap_at_k([1], [], 10) == 0.0


# --------------------------------------------------------------------------
# A tiny synthetic dataset in the BEIR layout
# --------------------------------------------------------------------------

N_DOCS = 40
N_QUERIES = 10
FILLER = ["the", "of", "and", "data", "result", "method", "study", "analysis", "model", "value"]


def _write_beir(folder: Path) -> Path:
    """Doc i mentions its own two rare words; doc i + 20 shares the first one.

    Query i asks for doc i's two words, so doc i is the best answer (grade 2)
    and doc i + 20 a partial one (grade 1).
    """
    folder.mkdir(parents=True)
    with (folder / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for i in range(N_DOCS):
            topic = i % 20
            words = [f"alpha{topic}"] + ([f"beta{topic}"] if i < 20 else [])
            filler = [FILLER[(i + j) % len(FILLER)] for j in range(6)]
            doc = {"_id": f"d{i}", "title": f"Note {i}", "text": " ".join(words + filler)}
            f.write(json.dumps(doc) + "\n")
    with (folder / "queries.jsonl").open("w", encoding="utf-8") as f:
        for i in range(N_QUERIES + 2):
            f.write(json.dumps({"_id": f"q{i}", "text": f"alpha{i} beta{i}"}) + "\n")
    (folder / "qrels").mkdir()
    lines = ["query-id\tcorpus-id\tscore"]
    for i in range(N_QUERIES):
        lines += [f"q{i}\td{i}\t2", f"q{i}\td{i + 20}\t1"]
    lines.append(f"q{N_QUERIES}\td0\t0")  # only a zero grade: dropped
    (folder / "qrels" / "test.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


@pytest.fixture
def beir_folder(tmp_path: Path) -> Path:
    return _write_beir(tmp_path / "toy")


def test_load_beir(beir_folder: Path) -> None:
    ds = load_beir(beir_folder)
    assert ds.name == "toy"
    assert len(ds.doc_ids) == N_DOCS and ds.doc_ids[3] == "d3"
    assert ds.titles[3] == "Note 3" and "alpha3" in ds.texts[3]
    # q10 has only a zero grade and q11 no judgments at all: both are dropped.
    assert list(ds.queries) == [f"q{i}" for i in range(N_QUERIES)]
    assert ds.qrels["q0"] == {"d0": 2, "d20": 1}
    assert len(ds.fingerprint) == 16
    assert load_beir(beir_folder).fingerprint == ds.fingerprint


def test_load_beir_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotAvailableError, match=r"corpus\.jsonl is missing"):
        load_beir(tmp_path)
    with pytest.raises(DatasetNotAvailableError, match="neither a known dataset"):
        load_dataset(str(tmp_path / "nope"))


# --------------------------------------------------------------------------
# Pinned downloads (served from file:// URLs)
# --------------------------------------------------------------------------


def _zip_dataset(tmp_path: Path, members: dict[str, bytes]) -> DatasetSpec:
    """Zip `members` as `<remote>/toy.zip` and return a spec pinned to it."""
    remote = tmp_path / "remote"
    remote.mkdir()
    archive = remote / "toy.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    data = archive.read_bytes()
    return DatasetSpec("toy", hashlib.sha256(data).hexdigest(), len(data), base_url=remote.as_uri())


def _beir_members(folder: Path) -> dict[str, bytes]:
    return {
        f"toy/{p.relative_to(folder).as_posix()}": p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def test_dataset_downloads_once_and_unpacks(tmp_path: Path, beir_folder: Path) -> None:
    spec = _zip_dataset(tmp_path, _beir_members(beir_folder))
    cache = tmp_path / "cache"
    seen: list[str] = []
    folder = ensure_dataset(spec, cache_dir=cache, on_download=lambda s: seen.append(s.name))
    assert seen == ["toy"]
    assert folder == cache / "datasets" / "toy"
    assert load_beir(folder).fingerprint == load_beir(beir_folder).fingerprint
    # Second call: from the cache, offline, no download, no leftovers.
    assert ensure_dataset(spec, cache_dir=cache, offline=True) == folder
    assert seen == ["toy"]
    assert not [p for p in (cache / "datasets").iterdir() if p.name.startswith(".")]


def test_dataset_offline_without_cache_fails(tmp_path: Path, beir_folder: Path) -> None:
    spec = _zip_dataset(tmp_path, _beir_members(beir_folder))
    with pytest.raises(DatasetNotAvailableError, match="--offline"):
        ensure_dataset(spec, cache_dir=tmp_path / "cache", offline=True)


def test_dataset_hash_mismatch_is_rejected(tmp_path: Path, beir_folder: Path) -> None:
    spec = dataclasses.replace(_zip_dataset(tmp_path, _beir_members(beir_folder)), sha256="0" * 64)
    cache = tmp_path / "cache"
    with pytest.raises(DatasetNotAvailableError, match="SHA-256 mismatch"):
        ensure_dataset(spec, cache_dir=cache)
    assert [p for p in cache.rglob("*") if p.is_file()] == []


def test_dataset_archive_cannot_escape_the_cache(tmp_path: Path) -> None:
    spec = _zip_dataset(tmp_path, {"../evil.txt": b"x", "toy/corpus.jsonl": b""})
    with pytest.raises(DatasetNotAvailableError, match="unsafe path"):
        ensure_dataset(spec, cache_dir=tmp_path / "cache")
    assert not (tmp_path / "evil.txt").exists()


def test_pinned_beir_specs() -> None:
    from notelight.eval import BEIR_DATASETS

    assert list(BEIR_DATASETS) == ["scifact", "nfcorpus", "fiqa"]
    for spec in BEIR_DATASETS.values():
        assert len(spec.sha256) == 64 and spec.size > 0
        assert spec.url.endswith(f"/{spec.name}.zip")


# --------------------------------------------------------------------------
# NumPy baseline agrees with the Rust core
# --------------------------------------------------------------------------


def _random_index(
    tmp_path: Path, n: int = 300, dim: int = 48
) -> tuple[_core.IndexReader, np.ndarray]:
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    writer = _core.IndexWriter(dim, "random")
    for i, v in enumerate(vectors):
        writer.add(_core.ChunkRecord(f"d{i}", [], 1, 1, "x"), v)
    writer.commit(tmp_path / "index")
    return _core.IndexReader.open(tmp_path / "index"), vectors


def test_numpy_binary_codes_match_rust() -> None:
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((20, 37)).astype(np.float32)
    np.testing.assert_array_equal(NumpyVectors(vectors).bits, _core.quantize_binary(vectors))


@pytest.mark.parametrize("mode", ["int8", "binary", "binary+int8", "ternary+int8"])
def test_numpy_baseline_agrees_with_rust(tmp_path: Path, mode: str) -> None:
    reader, vectors = _random_index(tmp_path)
    npv = NumpyVectors(vectors)
    rng = np.random.default_rng(11)
    for _ in range(20):
        q = rng.standard_normal(vectors.shape[1]).astype(np.float32)
        rust = reader.search_vector(q, 10, 50, mode)  # type: ignore[arg-type]
        if mode == "int8":
            ours = npv.search_int8(q, 10)
        elif mode == "binary":
            ours = npv.search_binary(q, 10)
        elif mode == "binary+int8":
            ours = npv.search_binary_int8(q, 10, 50)
        else:
            ours = npv.search_ternary_int8(q, 10, 50)
        # Same ids in the same order; scores equal up to float summation order.
        assert [i for i, _ in ours] == [i for i, _ in rust]
        np.testing.assert_allclose([s for _, s in ours], [s for _, s in rust], atol=1e-5)


def test_numpy_f32_is_exact_cosine(tmp_path: Path) -> None:
    _, vectors = _random_index(tmp_path, n=50)
    npv = NumpyVectors(vectors)
    hits = npv.search_f32(vectors[5] * 3.0, 3)
    assert hits[0][0] == 5 and hits[0][1] == pytest.approx(1.0, abs=1e-6)


def test_search_vector_defaults_to_ternary(tmp_path: Path) -> None:
    reader, vectors = _random_index(tmp_path)
    q = vectors[3] + 0.5 * vectors[4]
    default = reader.search_vector(q, 10)
    assert default == reader.search_vector(q, 10, 200, "ternary+int8")


def test_search_vector_rejects_unknown_mode(tmp_path: Path) -> None:
    reader, vectors = _random_index(tmp_path, n=5)
    with pytest.raises(ValueError, match="unknown vector mode"):
        reader.search_vector(vectors[0], 3, 10, "hnsw")  # type: ignore[arg-type]
    assert _core.VECTOR_MODES == ["int8", "binary", "binary+int8", "ternary+int8"]


# --------------------------------------------------------------------------
# Runner and report
# --------------------------------------------------------------------------


def test_evaluate_scores_every_method(beir_folder: Path, tmp_path: Path) -> None:
    ds = load_beir(beir_folder)
    result = evaluate(ds, HashingEmbedder(64), cache_dir=tmp_path / "cache", candidates=20)
    assert [m.method for m in result.methods] == list(METHODS)
    assert (result.docs, result.queries, result.k) == (N_DOCS, N_QUERIES, 10)
    for m in result.methods:
        assert 0.0 <= m.ndcg <= 1.0 and 0.0 <= m.recall <= 1.0 and m.latency_ms >= 0
    # Every relevant document contains a query word, so keyword search
    # retrieves both of them, best one first.
    assert result.method("bm25").recall == 1.0
    assert result.method("bm25").ndcg == 1.0
    # Exact search over a 64-dim bag of words finds the grade-2 document.
    assert result.method("f32").ndcg > 0.8
    assert [b.search for b in result.baseline] == list(BASELINE_SEARCHES)
    assert all(b.agreement >= 0.9 for b in result.baseline)

    table = format_markdown([result])
    assert "| `hybrid` |" in table and "toy nDCG@10" in table and "Top-10 agreement" in table


def test_evaluate_caches_embeddings_and_index(beir_folder: Path, tmp_path: Path) -> None:
    ds = load_beir(beir_folder)
    cache = tmp_path / "cache"
    first = CountingEmbedder()
    evaluate(ds, first, methods=["bm25"], baseline=False, cache_dir=cache, max_queries=3)
    assert len(first.embedded) == N_DOCS
    # A later run reuses the corpus vectors and embeds only the new queries.
    second = CountingEmbedder()
    result = evaluate(ds, second, methods=["f32", "hybrid"], baseline=False, cache_dir=cache)
    assert second.embedded == []
    assert result.queries == N_QUERIES


def test_evaluate_passes_phrase_boost_to_hybrid(
    beir_folder: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[float] = []

    class SpyReader:
        def __init__(self, inner: _core.IndexReader) -> None:
            self._inner = inner

        def search(self, *args: Any, **kwargs: Any) -> list[_core.Hit]:
            seen.append(kwargs["phrase_boost"])
            return self._inner.search(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    real_open = runner._open_or_build_index
    monkeypatch.setattr(
        runner, "_open_or_build_index", lambda *a, **kw: SpyReader(real_open(*a, **kw))
    )
    ds = load_beir(beir_folder)
    for boost in (0.0, 2.5):
        result = evaluate(
            ds,
            HashingEmbedder(64),
            methods=["hybrid"],
            baseline=False,
            cache_dir=tmp_path / "cache",
            phrase_boost=boost,
        )
        assert 0.0 <= result.method("hybrid").ndcg <= 1.0
    assert set(seen) == {0.0, 2.5}


def test_evaluate_rejects_unknown_methods(beir_folder: Path) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        evaluate(load_beir(beir_folder), HashingEmbedder(), methods=["colbert"])


def test_cli_eval_on_a_folder(beir_folder: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["eval", str(beir_folder), "--embedder", "hashing", "--methods", "bm25,f32"]
    )
    assert result.exit_code == 0, result.output
    assert "| `bm25` | 1.000 | 1.000 |" in result.output

    result = runner.invoke(
        app, ["eval", str(beir_folder), "-e", "hashing", "-m", "bm25", "--no-baseline", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("[") :])
    assert payload[0]["methods"][0]["method"] == "bm25" and payload[0]["baseline"] == []


def test_cli_eval_rejects_bad_input(beir_folder: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["eval", str(beir_folder), "-e", "hashing", "-m", "nope"])
    assert result.exit_code == 2 and "unknown method" in result.output
    result = runner.invoke(app, ["eval", str(beir_folder), "-e", "none"])
    assert result.exit_code == 2
