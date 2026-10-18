"""Measure the "no close match" floor for bge-small (design notes, "No close match").

For each query this takes the best cosine between the query and any chunk,
as `Searcher.query` does (the `ternary+int8` vector search, k=1), and prints
how often it falls below candidate floors:

- on-topic queries: the notes benchmark (tune and held-out splits, against
  its own corpus) and the BEIR test queries against their own corpora; a
  query below the floor is a false "no close match";
- off-topic queries: `match_floor_offtopic.json` (written before any
  measurement; tune and held-out halves) and the BEIR queries against the
  notes corpus, plus FiQA (finance) against SciFact (science) and back; a
  query below the floor is a correct notice.

The floor was chosen from the tune columns only (notes tune, off-topic tune,
SciFact against notes). Needs the BEIR datasets and bge-small in the cache
(`notelight eval` downloads them); embeddings are cached like `notelight eval`.

    python scripts/measure_match_floor.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from notelight import _core, embed
from notelight.eval.datasets import load_dataset
from notelight.eval.runner import _prepare

HERE = Path(__file__).resolve().parent
FLOORS = [round(0.55 + 0.01 * i, 2) for i in range(12)]


def tops(reader: _core.IndexReader, vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Best cosine per query vector."""
    return np.array([reader.search_vector(v.astype(np.float32), k=1)[0][1] for v in vectors])


def main() -> None:
    e = embed.make_embedder("bge-small", offline=True)
    assert e is not None
    readers: dict[str, _core.IndexReader] = {}
    queries: dict[str, npt.NDArray[np.float32]] = {}
    on: dict[str, npt.NDArray[np.float64]] = {}
    for name in ("notes-tune", "notes-heldout", "scifact", "nfcorpus", "fiqa"):
        split = name.removeprefix("notes-") if name.startswith("notes") else "tune"
        ds = load_dataset("notes" if name.startswith("notes") else name, notes_split=split)
        reader, _, qv = _prepare(ds, list(ds.queries), e, None, None)
        readers[name], queries[name] = reader, qv
        on[name] = tops(reader, qv)

    notes = readers["notes-tune"]  # both splits share the corpus
    authored = json.loads((HERE / "match_floor_offtopic.json").read_text(encoding="utf-8"))
    off: dict[str, npt.NDArray[np.float64]] = {}
    for split in ("tune", "heldout"):
        vectors = np.stack([e.embed_query(q) for q in authored[split]])
        off[f"off-topic {split} > notes"] = tops(notes, vectors)
    for name in ("scifact", "nfcorpus", "fiqa"):
        off[f"{name} > notes"] = tops(notes, queries[name])
    off["fiqa > scifact"] = tops(readers["scifact"], queries["fiqa"])
    off["scifact > fiqa"] = tops(readers["fiqa"], queries["scifact"])

    for title, groups, what in (
        ("On-topic queries: share flagged (false notices)", on, "own corpus"),
        ("Off-topic queries: share flagged (correct notices)", off, "queries > corpus"),
    ):
        print(f"\n{title}\n")
        print(f"| floor | {' | '.join(groups)} |")
        print(f"|---|{'---:|' * len(groups)}")
        for floor in FLOORS:
            cells = " | ".join(f"{(v < floor).mean():.0%}" for v in groups.values())
            print(f"| {floor:.2f} | {cells} |")
        counts = ", ".join(f"{k} {len(v)}" for k, v in groups.items())
        print(f"\nQueries ({what}): {counts}")


if __name__ == "__main__":
    main()
