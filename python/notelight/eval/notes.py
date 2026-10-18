"""The notes benchmark: short, note-style queries over this repository's docs.

BEIR measures long questions over papers and forum posts. Notes search is
different: two keywords, an identifier, a quoted phrase, or a short question
over a few hundred Markdown sections. This dataset measures that, at a small
scale and with the limits stated in `docs/results.md`.

The data lives in `notes/` next to this module:

    queries.json     queries, their set, style and split (tune or heldout), how
                     each set was written, and relevant passages as
                     [path, line_start, line_end, grade]
    corpus.json.gz   the corpus files at `corpus_commit`, {path: text}, checked
                     against the SHA-256 per file recorded in queries.json

The corpus is chunked with the real chunker, so `max_chars` can be swept.
Labels are line ranges, not chunk ids, so they hold for any chunking: a chunk
is relevant to a query when its lines overlap a labelled passage (see
`Dataset.judged_ranking`).

Queries come in sets over the same corpus. Set 1 is the original benchmark;
set 2 was added once set 1's halves had informed several decisions. A set is
never edited after its first run; new queries go into a new set.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from notelight import _core
from notelight.errors import DatasetNotAvailableError
from notelight.eval.datasets import Dataset
from notelight.indexer import DEFAULT_MAX_CHARS

__all__ = ["NOTES_DIR", "NOTES_SETS", "NOTES_SPLITS", "NOTES_STYLES", "Passage", "load_notes"]

#: The bundled benchmark.
NOTES_DIR = Path(__file__).with_name("notes")

#: `tune` is for choosing settings, `heldout` is reported once for the chosen
#: settings, `all` is both.
NOTES_SPLITS: tuple[str, ...] = ("tune", "heldout", "all")

#: Query sets. A query without a `set` field belongs to set 1.
NOTES_SETS: tuple[int, ...] = (1, 2)

#: Query styles, in report order.
NOTES_STYLES: tuple[str, ...] = ("keyword", "phrase", "question", "paraphrase")


@dataclass(frozen=True)
class Passage:
    """A labelled range of lines in one corpus file (1-based, inclusive)."""

    path: str
    line_start: int
    line_end: int

    @property
    def id(self) -> str:
        """`path:start-end`, the id used in qrels."""
        return f"{self.path}:{self.line_start}-{self.line_end}"

    def overlaps(self, path: str, line_start: int, line_end: int) -> bool:
        """Whether a chunk at `path:line_start-line_end` shares a line with this passage."""
        return path == self.path and line_start <= self.line_end and self.line_start <= line_end


def _bad(folder: Path, why: str) -> DatasetNotAvailableError:
    return DatasetNotAvailableError(f"{folder}: invalid notes benchmark ({why})")


def _read_corpus(folder: Path, spec: dict[str, Any]) -> dict[str, str]:
    path = folder / "corpus.json.gz"
    try:
        files: dict[str, str] = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    except (OSError, ValueError) as e:
        raise _bad(folder, f"cannot read {path.name}: {e}") from e
    expected: dict[str, str] = spec.get("corpus_files", {})
    if set(files) != set(expected):
        raise _bad(folder, "corpus files differ from corpus_files")
    for name, text in files.items():
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected[name]:
            raise _bad(folder, f"{name} does not match its pinned SHA-256")
    return files


def load_notes(
    *,
    split: str = "tune",
    max_chars: int = DEFAULT_MAX_CHARS,
    query_set: int = 1,
    folder: Path | None = None,
) -> Dataset:
    """Chunk the notes corpus and load the queries of one set and split.

    `folder` defaults to the bundled benchmark; tests pass a small one in the
    same layout. The dataset is named `notes-<split>` for set 1 and
    `notes<set>-<split>` for later sets, plus `@<max_chars>` when the chunk
    size is not the default.

    Raises:
        DatasetNotAvailableError: an unknown set or split, a missing or malformed
            file, a corpus file that does not match its hash, or a label
            outside its file.
        ValueError: `max_chars` < 1.
    """
    if split not in NOTES_SPLITS:
        raise DatasetNotAvailableError(f"unknown split {split!r}; choose from {NOTES_SPLITS}")
    if query_set not in NOTES_SETS:
        raise DatasetNotAvailableError(f"unknown query set {query_set!r}; choose from {NOTES_SETS}")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    folder = folder if folder is not None else NOTES_DIR
    queries_path = folder / "queries.json"
    try:
        raw = queries_path.read_bytes()
        spec = json.loads(raw)
    except (OSError, ValueError) as e:
        raise _bad(folder, f"cannot read queries.json: {e}") from e
    files = _read_corpus(folder, spec)

    doc_ids: list[str] = []
    texts: list[str] = []
    heading_paths: list[list[str]] = []
    locations: list[tuple[str, int, int]] = []
    for path in sorted(files):
        for c in _core.chunk_text(files[path], max_chars=max_chars, markdown=True):
            doc_ids.append(f"{path}:{c.line_start}-{c.line_end}")
            texts.append(c.text)
            heading_paths.append(c.heading_path)
            locations.append((path, c.line_start, c.line_end))
    line_counts = {path: len(text.splitlines()) for path, text in files.items()}

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    styles: dict[str, str] = {}
    passages: dict[str, Passage] = {}
    try:
        for q in spec["queries"]:
            if q.get("set", 1) != query_set or (split != "all" and q["split"] != split):
                continue
            if q["style"] not in NOTES_STYLES:
                raise _bad(folder, f"query {q['id']}: unknown style {q['style']!r}")
            rel: dict[str, int] = {}
            for path, start, end, grade in q["relevant"]:
                if path not in files or not 1 <= start <= end <= line_counts[path]:
                    raise _bad(folder, f"query {q['id']}: {path}:{start}-{end} is outside the file")
                p = Passage(path, int(start), int(end))
                passages[p.id] = p
                rel[p.id] = int(grade)
            queries[q["id"]] = q["text"]
            qrels[q["id"]] = {pid: g for pid, g in rel.items() if g > 0}
            styles[q["id"]] = q["style"]
    except (KeyError, TypeError, ValueError) as e:
        raise _bad(folder, f"malformed query: {e!r}") from e

    covers = {
        doc_id: tuple(sorted(pid for pid, p in passages.items() if p.overlaps(*loc)))
        for doc_id, loc in zip(doc_ids, locations, strict=True)
    }
    h = hashlib.sha256(raw)
    h.update(f"\nmax_chars={max_chars}\nset={query_set}".encode())
    name = f"notes-{split}" if query_set == 1 else f"notes{query_set}-{split}"
    if max_chars != DEFAULT_MAX_CHARS:
        name += f"@{max_chars}"
    return Dataset(
        name=name,
        doc_ids=doc_ids,
        titles=[" > ".join(hp) for hp in heading_paths],
        texts=texts,
        queries=queries,
        qrels=qrels,
        fingerprint=h.hexdigest()[:16],
        heading_paths=heading_paths,
        locations=locations,
        covers={d: c for d, c in covers.items() if c},
        query_styles=styles,
    )
