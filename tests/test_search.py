"""Snippets, query-term highlighting and close-match reporting (`notelight.search`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from notelight import _core
from notelight.embed import BGE_SMALL_EN_V15, HashingEmbedder, match_floor_for
from notelight.indexer import sync
from notelight.search import Searcher, SearchResults, match_spans, query_terms, snippet


def test_tokenize_binding_matches_bm25_rules() -> None:
    assert _core.tokenize("Fix getUserName in the auth-timeout") == [
        "fix",
        "get",
        "user",
        "name",
        "getusername",
        "auth",
        "timeout",
    ]


def test_query_terms_drop_quotes_stopwords_and_case() -> None:
    assert query_terms('"Lock the File" windows') == {"lock", "file", "windows"}
    assert query_terms("the a of") == frozenset()


def test_match_spans_use_bm25_terms() -> None:
    text = "Call getUserName; LOCK-file is not a lockfile."
    spans = match_spans(text, query_terms("user lock file"))
    assert [text[s:e] for s, e in spans] == ["getUserName", "LOCK", "file"]
    assert match_spans(text, frozenset()) == []


def test_short_text_is_returned_whole() -> None:
    assert snippet("one\n  two   three", terms=query_terms("three")) == "one two three"


def test_snippet_keeps_the_start_when_the_match_is_visible() -> None:
    text = "lock file " + "filler " * 60
    assert snippet(text, width=80, terms=query_terms("lock")).startswith("lock file filler")


def test_snippet_moves_to_a_late_match() -> None:
    text = "filler " * 60 + "the sourdough starter rises overnight " + "more " * 60
    plain = snippet(text, width=80)
    assert "sourdough" not in plain and plain.endswith("…")
    moved = snippet(text, width=80, terms=query_terms("sourdough"))
    assert moved.startswith("…") and moved.endswith("…")
    assert "sourdough starter" in moved
    assert len(moved) <= 80
    assert moved[1:].split(" ")[0] in {"filler", "the"}  # starts on a whole word


def test_snippet_near_the_end_stays_full_width() -> None:
    text = "filler " * 60 + "final word sourdough"
    moved = snippet(text, width=80, terms=query_terms("sourdough"))
    assert moved.startswith("…") and moved.endswith("sourdough")
    assert 70 <= len(moved) <= 80


def test_snippet_without_a_match_is_the_start() -> None:
    text = "filler " * 60
    assert snippet(text, width=80, terms=query_terms("absent")) == snippet(text, width=80)


# -- close match ----------------------------------------------------------------


def _set_floor(monkeypatch: pytest.MonkeyPatch, floor: float) -> None:
    """Give the hashing embedder a match floor (only bge-small has a real one)."""
    hashing = HashingEmbedder().model_id
    monkeypatch.setattr(
        "notelight.search.match_floor_for", lambda model_id: floor if model_id == hashing else None
    )


def test_close_match_needs_both_numbers() -> None:
    assert SearchResults([], best_similarity=0.7, match_floor=0.6).close_match is True
    assert SearchResults([], best_similarity=0.6, match_floor=0.6).close_match is True
    assert SearchResults([], best_similarity=0.5, match_floor=0.6).close_match is False
    assert SearchResults([], best_similarity=None, match_floor=0.6).close_match is None
    assert SearchResults([], best_similarity=0.5, match_floor=None).close_match is None


def test_only_bge_small_has_a_floor() -> None:
    assert match_floor_for(BGE_SMALL_EN_V15.model_id) == BGE_SMALL_EN_V15.match_floor == 0.60
    assert match_floor_for(HashingEmbedder().model_id) is None
    assert match_floor_for(None) is None


def test_query_reports_the_best_cosine(notes: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    embedder = HashingEmbedder()
    sync(notes, embedder)
    searcher = Searcher(notes, embedder)
    assert searcher.query("sourdough starter").close_match is None  # no floor: unknown

    _set_floor(monkeypatch, 0.0)
    found = searcher.query("sourdough starter", k=3)
    assert found.results == searcher.search("sourdough starter", k=3)
    vector = embedder.embed_query("sourdough starter")
    assert found.best_similarity == pytest.approx(searcher._reader.search_vector(vector, k=1)[0][1])
    assert found.close_match is True

    _set_floor(monkeypatch, 1.01)
    found = searcher.query("sourdough starter", k=3)
    assert found.close_match is False
    assert found.results  # the nearest chunks are still returned


def test_keyword_search_has_no_verdict(notes: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_floor(monkeypatch, 1.01)
    sync(notes, HashingEmbedder())
    found = Searcher(notes, None).query("sourdough")
    assert found.results and found.best_similarity is None and found.close_match is None
