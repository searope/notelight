"""The notes benchmark: loader, line-range judgments, MRR, per-style results.

Most tests use a tiny synthetic benchmark in the bundled layout. The tests on
the bundled benchmark check its structure and labelling rules only; they never
look at search results, so nothing here can leak the held-out queries.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from notelight import _core
from notelight.cli import app
from notelight.embed import HashingEmbedder
from notelight.errors import DatasetNotAvailableError
from notelight.eval import (
    NOTES_SPLITS,
    NOTES_STYLES,
    Dataset,
    evaluate,
    format_markdown,
    load_dataset,
    load_notes,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from notelight.eval.notes import NOTES_DIR, Passage

# --------------------------------------------------------------------------
# A tiny benchmark in the bundled layout
# --------------------------------------------------------------------------

ALPHA = """# Alpha

## Lock

The writer takes a lock file so only one process commits.

## Cache

Embeddings are cached by content hash.
"""

BETA = """# Beta

Ternary codes pick candidates; int8 rescores them.

Quoted phrases rank first.
"""

QUERIES: list[dict[str, Any]] = [
    {
        "id": "k1",
        "style": "keyword",
        "split": "tune",
        "text": "lock file",
        "origin": "test",
        "relevant": [["alpha.md", 3, 5, 2]],
    },
    {
        "id": "k2",
        "style": "keyword",
        "split": "heldout",
        "text": "ternary",
        "origin": "test",
        "relevant": [["beta.md", 3, 3, 2]],
    },
    {
        "id": "q1",
        "style": "question",
        "split": "tune",
        "text": "how are embeddings cached?",
        "origin": "test",
        "relevant": [["alpha.md", 7, 9, 2], ["alpha.md", 1, 1, 1]],
    },
    {
        "id": "p1",
        "style": "phrase",
        "split": "heldout",
        "text": '"quoted phrases"',
        "origin": "test",
        "relevant": [["beta.md", 5, 5, 2]],
    },
]


def _write_notes(folder: Path, queries: list[dict[str, Any]] = QUERIES) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    files = {"alpha.md": ALPHA, "beta.md": BETA}
    spec = {
        "name": "notes",
        "version": 1,
        "corpus_commit": "0" * 40,
        "corpus_files": {p: hashlib.sha256(t.encode()).hexdigest() for p, t in files.items()},
        "queries": queries,
    }
    (folder / "queries.json").write_text(json.dumps(spec), encoding="utf-8")
    (folder / "corpus.json.gz").write_bytes(gzip.compress(json.dumps(files).encode()))
    return folder


@pytest.fixture
def notes_folder(tmp_path: Path) -> Path:
    return _write_notes(tmp_path / "notes")


def test_load_notes_chunks_the_corpus_and_keeps_line_ranges(notes_folder: Path) -> None:
    ds = load_notes(split="all", folder=notes_folder)
    assert ds.name == "notes-all"
    assert ds.locations is not None and ds.heading_paths is not None
    # Every chunk is a real chunk: its id is its location.
    for doc_id, (path, start, end) in zip(ds.doc_ids, ds.locations, strict=True):
        assert doc_id == f"{path}:{start}-{end}"
    lock = next(i for i, text in enumerate(ds.texts) if "lock file" in text)
    assert ds.heading_paths[lock] == ["Alpha", "Lock"]
    assert list(ds.queries) == ["k1", "k2", "q1", "p1"]
    assert ds.qrels["q1"] == {"alpha.md:7-9": 2, "alpha.md:1-1": 1}
    assert ds.query_styles == {"k1": "keyword", "k2": "keyword", "q1": "question", "p1": "phrase"}


def test_load_notes_splits(notes_folder: Path) -> None:
    tune = load_notes(split="tune", folder=notes_folder)
    heldout = load_notes(split="heldout", folder=notes_folder)
    assert list(tune.queries) == ["k1", "q1"] and list(heldout.queries) == ["k2", "p1"]
    assert tune.doc_ids == heldout.doc_ids
    # The chunk size is part of the name and the cache key; the split is in the name.
    small = load_notes(split="tune", max_chars=20, folder=notes_folder)
    assert small.name == "notes-tune@20" and len(small.doc_ids) > len(tune.doc_ids)
    assert small.fingerprint != tune.fingerprint
    with pytest.raises(DatasetNotAvailableError, match="unknown split"):
        load_notes(split="test", folder=notes_folder)
    with pytest.raises(ValueError, match="positive"):
        load_notes(max_chars=0, folder=notes_folder)


def test_chunks_cover_the_passages_they_overlap(notes_folder: Path) -> None:
    ds = load_notes(split="all", max_chars=30, folder=notes_folder)
    assert ds.covers is not None and ds.locations is not None
    for doc_id, (path, start, end) in zip(ds.doc_ids, ds.locations, strict=True):
        for pid in ds.covers.get(doc_id, ()):
            p_path, lines = pid.rsplit(":", 1)
            p_start, p_end = map(int, lines.split("-"))
            assert p_path == path and start <= p_end and p_start <= end
    assert Passage("a.md", 3, 5).overlaps("a.md", 5, 9)
    assert not Passage("a.md", 3, 5).overlaps("a.md", 6, 9)
    assert not Passage("a.md", 3, 5).overlaps("b.md", 3, 5)


def test_corpus_must_match_its_hashes(notes_folder: Path) -> None:
    files = {"alpha.md": ALPHA + "edited\n", "beta.md": BETA}
    (notes_folder / "corpus.json.gz").write_bytes(gzip.compress(json.dumps(files).encode()))
    with pytest.raises(DatasetNotAvailableError, match=r"alpha.md does not match"):
        load_notes(folder=notes_folder)


def test_labels_must_lie_inside_their_file(tmp_path: Path) -> None:
    bad = [dict(QUERIES[0], relevant=[["alpha.md", 8, 99, 2]])]
    with pytest.raises(DatasetNotAvailableError, match="outside the file"):
        load_notes(folder=_write_notes(tmp_path / "a", bad))
    bad = [dict(QUERIES[0], relevant=[["gamma.md", 1, 1, 2]])]
    with pytest.raises(DatasetNotAvailableError, match="outside the file"):
        load_notes(folder=_write_notes(tmp_path / "b", bad))
    bad = [dict(QUERIES[0], style="essay")]
    with pytest.raises(DatasetNotAvailableError, match="unknown style"):
        load_notes(folder=_write_notes(tmp_path / "c", bad))


# --------------------------------------------------------------------------
# Judging a ranked list of chunks against line-range labels
# --------------------------------------------------------------------------


def _covering(covers: dict[str, tuple[str, ...]]) -> Dataset:
    return Dataset("t", list(covers), [], [], {}, {}, covers=covers)


def test_judged_ranking_expands_chunks_into_the_passages_they_show() -> None:
    ds = _covering({"c1": ("A", "B"), "c2": ("A",), "c3": (), "c4": ("C",)})
    relevant = {"A": 1, "B": 2}
    # c1 shows both passages, best grade first; c2 adds nothing new; c4's
    # passage is not relevant to this query.
    assert ds.judged_ranking(["c1", "c2", "c3", "c4"], relevant) == [
        "B",
        "A",
        "doc:c2",
        "doc:c3",
        "doc:c4",
    ]
    ranked = ds.judged_ranking(["c1", "c2"], relevant)
    assert ndcg_at_k(ranked, relevant, 10) == pytest.approx(1.0)
    assert recall_at_k(ranked, relevant, 10) == 1.0


def test_judged_ranking_never_counts_a_passage_twice() -> None:
    # A chunk whose id equals a passage id must not count that passage again.
    ds = _covering({"a.md:1-5": ("a.md:1-5",), "a.md:1-9": ("a.md:1-5",)})
    relevant = {"a.md:1-5": 2}
    ranked = ds.judged_ranking(["a.md:1-9", "a.md:1-5"], relevant)
    assert ranked == ["a.md:1-5", "doc:a.md:1-5"]
    assert ndcg_at_k(ranked, relevant, 10) == pytest.approx(1.0)


def test_judged_ranking_is_the_identity_without_covers() -> None:
    ds = Dataset("t", ["d1", "d2"], [], [], {}, {})
    assert ds.judged_ranking(["d2", "d1"], {"d1": 1}) == ["d2", "d1"]


def test_mrr() -> None:
    relevant = {"a": 1, "b": 2}
    assert mrr_at_k(["a", "b"], relevant, 10) == 1.0
    assert mrr_at_k(["x", "y", "b"], relevant, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "b"], relevant, 2) == 0.0
    with pytest.raises(ValueError, match="undefined"):
        mrr_at_k(["a"], {}, 10)
    with pytest.raises(ValueError, match="k must be"):
        mrr_at_k(["a"], relevant, 0)


# --------------------------------------------------------------------------
# Running the harness on it
# --------------------------------------------------------------------------


def test_evaluate_on_notes_reports_mrr_and_styles(notes_folder: Path, tmp_path: Path) -> None:
    ds = load_notes(split="all", folder=notes_folder)
    result = evaluate(
        ds, HashingEmbedder(64), cache_dir=tmp_path / "cache", candidates=10, phrase_boost=4.0
    )
    bm25 = result.method("bm25")
    # Every relevant passage contains a query word, so keyword search finds it first.
    assert bm25.mrr == 1.0
    assert [s.style for s in bm25.by_style] == ["keyword", "question", "phrase"]
    assert [s.queries for s in bm25.by_style] == [2, 1, 1]
    for m in result.methods:
        assert 0.0 <= m.mrr <= 1.0 and 0.0 <= m.ndcg <= 1.0
        for s in m.by_style:
            assert 0.0 <= s.ndcg <= 1.0 and 0.0 <= s.recall <= 1.0
    assert result.phrase_boost == 4.0
    # Per-query scores are kept for paired comparisons, and average to the totals.
    assert list(bm25.per_query) == list(ds.queries)
    assert sum(s[0] for s in bm25.per_query.values()) / 4 == pytest.approx(bm25.ndcg)

    table = format_markdown([result])
    assert "notes-all MRR@10" in table and "notes-all by query style:" in table
    assert "keyword (2) nDCG@10" in table and "phrase_boost 4" in table


def test_evaluate_indexes_real_paths_and_line_ranges(notes_folder: Path, tmp_path: Path) -> None:
    ds = load_notes(split="all", folder=notes_folder)
    cache = tmp_path / "cache"
    evaluate(ds, HashingEmbedder(64), methods=["bm25"], baseline=False, cache_dir=cache)
    (root,) = [p for p in (cache / "runs").rglob("index") if _core.index_exists(p)]
    reader = _core.IndexReader.open(root)
    assert len(reader) == len(ds.doc_ids)
    for i, (path, start, end) in enumerate(ds.locations or []):
        record = reader.chunk(i)
        assert (record.path, record.line_start, record.line_end) == (path, start, end)
        assert record.heading_path == (ds.heading_paths or [])[i]


def test_load_dataset_routes_notes() -> None:
    ds = load_dataset("notes", notes_split="heldout", max_chars=600)
    assert ds.name == "notes-heldout@600"
    assert load_dataset("notes").name == "notes-tune"
    with pytest.raises(DatasetNotAvailableError, match="notes benchmark only"):
        load_dataset("scifact", max_chars=600)


def test_cli_eval_notes() -> None:
    runner = CliRunner()
    args = ["eval", "notes", "-e", "hashing", "-m", "bm25,hybrid", "--no-baseline"]
    result = runner.invoke(app, [*args, "--split", "heldout", "--max-chars", "600"])
    assert result.exit_code == 0, result.output
    assert "notes-heldout@600 MRR@10" in result.output
    assert "notes-heldout@600 by query style:" in result.output
    result = runner.invoke(app, [*args, "--split", "test"])
    assert result.exit_code == 2 and "unknown split" in result.output


# --------------------------------------------------------------------------
# The bundled benchmark: structure and labelling rules (no search)
# --------------------------------------------------------------------------


def _bundled() -> dict[str, Any]:
    spec: dict[str, Any] = json.loads((NOTES_DIR / "queries.json").read_text(encoding="utf-8"))
    return spec


def test_bundled_benchmark_loads_and_its_splits_partition_it() -> None:
    tune, heldout = load_notes(split="tune"), load_notes(split="heldout")
    every = load_notes(split="all")
    assert set(tune.queries).isdisjoint(heldout.queries)
    assert set(tune.queries) | set(heldout.queries) == set(every.queries)
    assert 60 <= len(every.queries) <= 100
    assert NOTES_SPLITS == ("tune", "heldout", "all")


def test_bundled_splits_alternate_within_each_style() -> None:
    queries = _bundled()["queries"]
    for style in NOTES_STYLES:
        of_style = [q for q in queries if q["style"] == style]
        assert len(of_style) >= 10
        assert [q["split"] for q in of_style] == [
            "tune" if i % 2 == 0 else "heldout" for i in range(len(of_style))
        ]
    counts = Counter(q["split"] for q in queries)
    assert abs(counts["tune"] - counts["heldout"]) <= len(NOTES_STYLES)


def test_bundled_queries_are_well_formed() -> None:
    queries = _bundled()["queries"]
    assert len({q["id"] for q in queries}) == len(queries)
    assert len({q["text"] for q in queries}) == len(queries)
    for q in queries:
        assert q["origin"], q["id"]
        assert any(grade == 2 for *_, grade in q["relevant"]), q["id"]
        assert all(grade in (1, 2) for *_, grade in q["relevant"]), q["id"]
        if q["style"] == "keyword":
            assert 1 <= len(q["text"].split()) <= 4, q["id"]


def test_bundled_paraphrases_share_no_term_with_their_answers() -> None:
    """A paraphrase shares no BM25 term with any grade-2 passage, including
    the heading path BM25 indexes with each chunk that overlaps it."""
    spec = _bundled()
    files: dict[str, str] = json.loads(gzip.decompress((NOTES_DIR / "corpus.json.gz").read_bytes()))
    chunks = {
        path: _core.chunk_text(text, max_chars=1200, markdown=True) for path, text in files.items()
    }
    paraphrases = [q for q in spec["queries"] if q["style"] == "paraphrase"]
    assert paraphrases
    for q in paraphrases:
        terms = set(_core.tokenize(q["text"]))
        for path, start, end, grade in q["relevant"]:
            if grade != 2:
                continue
            lines = files[path].splitlines()[start - 1 : end]
            headings = [
                " ".join(c.heading_path)
                for c in chunks[path]
                if c.line_start <= end and start <= c.line_end
            ]
            shared = terms & set(_core.tokenize("\n".join([*lines, *headings])))
            assert not shared, f"{q['id']} shares {shared} with {path}:{start}-{end}"


def test_bundled_corpus_is_pinned() -> None:
    spec = _bundled()
    assert len(spec["corpus_commit"]) == 40
    assert len(spec["corpus_files"]) == 14
    # Every labelled file is in the corpus, and the loader checked the hashes.
    labelled = {path for q in spec["queries"] for path, *_ in q["relevant"]}
    assert labelled <= set(spec["corpus_files"])
    assert math.isclose(len(load_notes(split="all").queries), len(spec["queries"]))
