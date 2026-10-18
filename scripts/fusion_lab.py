"""Try hybrid fusion variants without changing the Rust core.

A Python copy of `IndexReader::search` (BM25 and vector lists, weighted RRF,
phrase rules) with named candidate weightings, scored with nDCG@10 and
compared to the default with a paired bootstrap. `--parity` first checks that
the copy returns exactly the core's top 10 for every query; run it before
trusting any number from here.

    python scripts/fusion_lab.py notes2 --split tune --parity
    python scripts/fusion_lab.py notes2 --split tune --cands C0,V1,V2,V3,B3
    python scripts/fusion_lab.py fiqa --cands C0,B3 --json fiqa.json

Datasets: scifact, nfcorpus, fiqa, notes (query set 1) and notes2 (set 2).
It uses the same cached embeddings and indexes as `notelight eval` (needs the
cached bge-small). Candidates, as tested in `docs/results.md`, "Fusion:
weighting vector hits (tested, not shipped)"; c1 is the share of distinct
query terms in BM25's top hit, cov(d) the share in chunk d:

    C0  the default: BM25 hit x cov(d)^2, vector hit x 1
    V1  vector hits x 0.5 when c1 = 1, else x 1
    V2  vector hits x (1 - 0.5 * c1^2)
    V3  vector hit d x (1 + cov(d)^2) / 2
    B3  BM25 hit x cov(d)^3 (combine with a V as "V1+B3")

Follow the notes benchmark's rules (CLAUDE.md, Milestone 3 notes): write the
candidates and the decision rule down before running them, pick on a tune
half, and run a held-out half once.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from notelight import _core
from notelight.embed import OnnxEmbedder
from notelight.eval import load_dataset, ndcg_at_k
from notelight.eval.runner import _prepare

F: type[np.float32] = np.float32
K, DEPTH, CANDIDATES = 10, 50, 200
RRF_K, PHRASE_BOOST = F(60.0), F(1.0)
QUOTES = {'"', "“", "”"}
CANDS = ("C0", "V1", "V2", "V3", "B3")


def parse(query: str) -> tuple[list[list[str]], list[str]]:
    """`PhraseQuery::parse`: quoted phrases (an unclosed quote runs to the end) and all terms."""
    quoted: list[list[str]] = []
    inside, current = False, list[str]()
    for c in query:
        if c in QUOTES:
            if inside and (terms := _core.tokenize("".join(current))):
                quoted.append(terms)
            current = []
            inside = not inside
        elif inside:
            current.append(c)
    if inside and (terms := _core.tokenize("".join(current))):
        quoted.append(terms)
    return quoted, _core.tokenize(query)


def contains_sequence(hay: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n == 0 or any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


class Chunks:
    """BM25 terms of each chunk (heading path, newline, text), as the core indexes them."""

    def __init__(self, reader: _core.IndexReader) -> None:
        self.reader = reader
        self._terms: dict[int, list[str]] = {}

    def terms(self, i: int) -> list[str]:
        if i not in self._terms:
            r = self.reader.chunk(i)
            self._terms[i] = _core.tokenize(" ".join(r.heading_path) + "\n" + r.text)
        return self._terms[i]

    def coverage(self, i: int, distinct: set[str]) -> np.float32:
        share = F(len(distinct & set(self.terms(i)))) / F(len(distinct)) if distinct else F(1.0)
        return cast("np.float32", share)


def vector_weights(
    cand: str, chunks: Chunks, bm25: list[int], vec: list[int], distinct: set[str]
) -> list[np.float32]:
    base = cand.split("+")[0]
    c1 = chunks.coverage(bm25[0], distinct) if bm25 else F(0.0)
    if base in ("C0", "B3"):
        return [F(1.0)] * len(vec)
    if base == "V1":
        return [F(0.5) if bm25 and c1 == F(1.0) else F(1.0)] * len(vec)
    if base == "V2":
        return [F(1.0) - F(0.5) * c1 * c1] * len(vec)
    if base == "V3":
        return [(F(1.0) + chunks.coverage(d, distinct) ** 2) / F(2.0) for d in vec]
    raise ValueError(f"unknown candidate {cand!r}; choose from {CANDS}")


def fuse(cand: str, chunks: Chunks, text: str, qvec: npt.NDArray[np.float32] | None) -> list[int]:
    """The top K ids `IndexReader::search` would return under candidate `cand`."""
    reader = chunks.reader
    quoted, terms = parse(text)
    bm25 = [i for i, _ in reader.search_bm25(text, DEPTH)]
    vec = [i for i, _ in reader.search_vector(qvec, DEPTH, CANDIDATES)] if qvec is not None else []
    distinct = set(terms)
    power = F(3.0) if "B3" in cand.split("+") else F(2.0)
    weigh = qvec is not None  # alone, BM25 keeps its own order
    bw = [F(np.power(chunks.coverage(d, distinct), power)) if weigh else F(1.0) for d in bm25]
    vw = vector_weights(cand, chunks, bm25, vec, distinct) if weigh else []

    scores: dict[int, np.float32] = {}
    for ids, weights in ((bm25, bw), (vec, vw)):
        seen: set[int] = set()
        for rank, (d, w) in enumerate(zip(ids, weights, strict=True)):
            if d not in seen:
                seen.add(d)
                scores[d] = F(scores.get(d, F(0.0)) + w / (RRF_K + F(rank) + F(1.0)))
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    has_sequence = len(terms) >= 2
    if not quoted and not has_sequence:
        return [d for d, _ in ranked[:K]]

    bonus = PHRASE_BOOST / (RRF_K + F(1.0))
    fused: list[tuple[int, np.float32, bool]] = []
    for d, s in ranked:
        all_quoted = False
        have = set(chunks.terms(d))
        if has_sequence and distinct <= have and contains_sequence(chunks.terms(d), terms):
            s = F(s + bonus)
        if quoted and all(set(p) <= have for p in quoted):
            all_quoted = all(contains_sequence(chunks.terms(d), p) for p in quoted)
        fused.append((d, s, all_quoted))
    fused.sort(key=lambda t: (not t[2], -t[1], t[0]))
    return [d for d, _, _ in fused[:K]]


def bootstrap(diffs: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Percentile 95% interval of the mean paired difference."""
    rng = random.Random(seed)
    m = len(diffs)
    means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dataset", help="scifact, nfcorpus, fiqa, notes or notes2")
    ap.add_argument("--split", default="tune", help="notes only: tune, heldout or all")
    ap.add_argument("--parity", action="store_true", help="check C0 against the core and stop")
    ap.add_argument("--cands", default="C0", help=f"comma-separated, from {', '.join(CANDS)}")
    ap.add_argument("--json", type=Path, help="write per-query nDCG@10 here")
    args = ap.parse_args()

    notes = args.dataset.startswith("notes")
    ds = load_dataset(args.dataset, notes_split=args.split) if notes else load_dataset(args.dataset)
    qids = list(ds.queries)
    reader, _, qvecs = _prepare(ds, qids, OnnxEmbedder(), None, None)
    chunks = Chunks(reader)
    print(f"{ds.name}: {len(ds.doc_ids)} docs, {len(qids)} queries", file=sys.stderr)

    def vector(qi: int) -> npt.NDArray[np.float32] | None:
        return qvecs[qi] if np.any(qvecs[qi]) else None  # an all-zero query has no direction

    if args.parity:
        same = sum(
            [h.id for h in reader.search(ds.queries[q], vector(qi), k=K)]
            == fuse("C0", chunks, ds.queries[q], vector(qi))
            for qi, q in enumerate(qids)
        )
        print(f"parity C0: {same}/{len(qids)} identical top-{K}")
        sys.exit(0 if same == len(qids) else 1)

    per: dict[str, dict[str, float]] = {}
    for cand in args.cands.split(","):
        per[cand] = {}
        for qi, q in enumerate(qids):
            ids = fuse(cand, chunks, ds.queries[q], vector(qi))
            rel = ds.qrels[q]
            ranked = ds.judged_ranking([ds.doc_ids[i] for i in ids], rel)
            per[cand][q] = ndcg_at_k(ranked, rel, K)

    styles = ds.query_styles or {}
    style_names = list(dict.fromkeys(styles[q] for q in qids if q in styles))
    base = per.get("C0")
    for cand, scores in per.items():
        line = f"{cand:8s} nDCG@10 {np.mean(list(scores.values())):.3f}"
        for st in style_names:
            line += f"  {st} {np.mean([scores[q] for q in qids if styles.get(q) == st]):.3f}"
        if base is not None and cand != "C0":
            d = [scores[q] - base[q] for q in qids]
            lo, hi = bootstrap(d)
            better, worse = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
            line += (
                f"  vs C0 {np.mean(d):+.3f} ({lo:+.3f}..{hi:+.3f}) better/worse {better}/{worse}"
            )
        print(line)
    if args.json:
        args.json.write_text(json.dumps({"dataset": ds.name, "per_query": per, "styles": styles}))


if __name__ == "__main__":
    main()
