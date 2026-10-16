"""Snapshot the notes benchmark corpus from git (`notelight eval notes`).

The corpus is this repository's own Markdown at the commit named by
`corpus_commit` in `python/notelight/eval/notes/queries.json`. This script
reads every file at that commit with `git show` (so the working tree does not
matter), writes them to `corpus.json.gz` next to the queries, and records each
file's SHA-256 in the queries file. The loader checks those hashes, so the
labels' line ranges always refer to exactly these bytes.

    python scripts/make_notes_corpus.py              # re-create from the pinned commit
    python scripts/make_notes_corpus.py --check      # verify the committed snapshot

Changing the commit means re-checking every label's line range by hand.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "python" / "notelight" / "eval" / "notes"

#: The corpus: user-facing docs, design notes, walkthroughs and project notes.
#: Contributor boilerplate (CONTRIBUTING, SECURITY, templates) is left out.
CORPUS = [
    "README.md",
    "CHANGELOG.md",
    "CLAUDE.md",
    "docs/design-notes.md",
    "docs/how-it-works.md",
    "docs/index-format.md",
    "docs/results.md",
    "docs/learning/m0-rust-walkthrough.md",
    "docs/learning/m1-rust-walkthrough.md",
    "docs/learning/m2-rust-walkthrough.md",
    "docs/learning/m3-int8-dot-vectorization.md",
    "docs/learning/m3-rust-walkthrough.md",
    "docs/learning/m4-rust-walkthrough.md",
    "docs/learning/v2-rust-walkthrough.md",
]


def snapshot(commit: str) -> dict[str, str]:
    """Each corpus file's text at `commit`, as stored by git (LF line endings)."""
    files = {}
    for path in CORPUS:
        data = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=True
        ).stdout
        files[path] = data.decode("utf-8")
    return files


def corpus_bytes(files: dict[str, str]) -> bytes:
    """Deterministic gzip (no timestamp), so the same corpus gives the same bytes."""
    payload = json.dumps(files, ensure_ascii=False, indent=0, sort_keys=True).encode("utf-8")
    return gzip.compress(payload, compresslevel=9, mtime=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    args = parser.parse_args()

    queries_path = NOTES / "queries.json"
    spec = json.loads(queries_path.read_text(encoding="utf-8"))
    files = snapshot(spec["corpus_commit"])
    hashes = {p: hashlib.sha256(t.encode("utf-8")).hexdigest() for p, t in files.items()}
    data = corpus_bytes(files)

    if args.check:
        ok = spec.get("corpus_files") == hashes and (NOTES / "corpus.json.gz").read_bytes() == data
        print("corpus snapshot matches" if ok else "corpus snapshot differs; re-run to update")
        return 0 if ok else 1

    (NOTES / "corpus.json.gz").write_bytes(data)
    spec["corpus_files"] = hashes
    text = queries_path.read_text(encoding="utf-8")
    # Rewrite only the corpus_files block, so the hand-formatted queries stay as they are.
    block = "  " + json.dumps({"corpus_files": hashes}, indent=2)[4:-2]
    start = text.find('  "corpus_files"')
    if start == -1:
        anchor = text.index('  "method"')
        text = text[:anchor] + block + ",\n" + text[anchor:]
    else:
        end = text.index("},\n", start) + 3
        text = text[:start] + block + ",\n" + text[end:]
    queries_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(files)} files ({len(data):,} bytes gzipped) at {spec['corpus_commit'][:7]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
