"""The fixture behind `scripts/model_check.py` and the model-check workflow.

The workflow claims that a pass proves the embeddings work, because keyword
search cannot find any of the expected files. These tests keep that claim true
as the fixture changes. The real-model run is opt-in (NOTELIGHT_MODEL_TESTS=1).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from notelight import _core

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "model-check"
NOTES = FIXTURE / "notes"
QUERIES = json.loads((FIXTURE / "queries.json").read_text(encoding="utf-8"))


def test_expected_files_exist() -> None:
    assert len(QUERIES) >= 3
    for q in QUERIES:
        assert (NOTES / q["expected"]).is_file(), q


def test_no_query_shares_a_bm25_term_with_any_note() -> None:
    vocabulary: set[str] = set()
    for path in NOTES.rglob("*"):
        if path.is_file():
            vocabulary |= set(_core.tokenize(path.read_text(encoding="utf-8")))
            vocabulary |= set(_core.tokenize(path.relative_to(NOTES).as_posix()))
    for q in QUERIES:
        assert not set(_core.tokenize(q["query"])) & vocabulary, q["query"]


@pytest.mark.skipif(
    os.environ.get("NOTELIGHT_MODEL_TESTS") != "1",
    reason="set NOTELIGHT_MODEL_TESTS=1 to download and run bge-small",
)
def test_model_check_script_passes() -> None:
    script = ROOT / "scripts" / "model_check.py"
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
