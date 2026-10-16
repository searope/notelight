"""Snippets and query-term highlighting (`notelight.search`)."""

from __future__ import annotations

from notelight import _core
from notelight.search import match_spans, query_terms, snippet


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
