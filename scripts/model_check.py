"""Check that an installed notelight finds notes by meaning, not just by words.

Indexes `tests/fixtures/model-check/notes` with the default embedder
(bge-small), then runs each query in `queries.json` twice through the
installed `notelight` CLI:

- hybrid search (the default) must rank the expected file in the top k;
- keyword-only search (`--embedder none`) must not find it at all.

No query shares a BM25 term with any fixture note (`tests/test_model_check.py`
keeps it that way), so a pass shows the embeddings did the work. The model
workflow runs this against wheels installed from PyPI; locally:

    python scripts/model_check.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "model-check"


def notelight(*args: str) -> str:
    # The console script next to this interpreter first, so a venv's Python
    # checks that venv's notelight even when the venv is not activated.
    exe = shutil.which("notelight", path=str(Path(sys.executable).parent)) or shutil.which(
        "notelight"
    )
    if exe is None:
        sys.exit("notelight is not installed next to this Python or on PATH")
    # UTF-8 pipes on every OS, so this checks the model and not the console
    # encoding (0.1.0 on Windows writes piped output in the ANSI code page).
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    done = subprocess.run([exe, *args], capture_output=True, text=True, encoding="utf-8", env=env)
    if done.returncode != 0:
        sys.exit(f"notelight {' '.join(args)} failed ({done.returncode}):\n{done.stderr}")
    return done.stdout


def ranked_paths(query: str, notes: Path, k: int, *extra: str) -> list[str]:
    out = notelight("search", query, "--folder", str(notes), "-k", str(k), "--json", *extra)
    return [hit["path"] for hit in json.loads(out)["results"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixture", type=Path, default=FIXTURE, help="Fixture folder.")
    parser.add_argument("-k", type=int, default=3, help="Expected file must rank in the top k.")
    args = parser.parse_args()

    notes = args.fixture / "notes"
    queries = json.loads((args.fixture / "queries.json").read_text(encoding="utf-8"))
    print(notelight("version"))
    print(notelight("index", str(notes)))

    failures = 0
    for q in queries:
        hybrid = ranked_paths(q["query"], notes, args.k)
        keyword = ranked_paths(q["query"], notes, args.k, "--embedder", "none")
        ok = q["expected"] in hybrid and q["expected"] not in keyword
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {q['query']!r} -> expected {q['expected']}")
        print(f"     hybrid top {args.k}: {hybrid}")
        print(f"     keyword only: {keyword}")
    print(f"{len(queries) - failures}/{len(queries)} queries passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
