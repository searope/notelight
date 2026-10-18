# Results

Retrieval quality and speed of every search method in notelight, measured on
three [BEIR](https://github.com/beir-cellar/beir) datasets and on a small
[notes benchmark](#notes-benchmark) with `notelight eval`. Every number on
this page comes from runs of that command; see [Reproducing](#reproducing).

## Setup

| | |
|---|---|
| Embedder | `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405e`, ONNX on CPU, 384 dimensions |
| Datasets | BEIR test splits: SciFact (5,183 docs, 300 queries), NFCorpus (3,633 docs, 323 queries), FiQA-2018 (57,638 docs, 648 queries) |
| Metrics | nDCG@10 and recall@10, computed like `pytrec_eval` (linear gain) |
| Machine | Intel Core i9-10900K (10 cores), Windows 11, Python 3.14, NumPy 2.5.4, ONNX Runtime 1.31 |
| Search settings | the `notelight search` defaults: 200 candidates re-scored, 50 results per retriever fused with RRF (k = 60), each BM25 hit weighted by the share of query terms it contains, squared ([why](#fusion-weighting-keyword-hits-by-coverage)) |

Each BEIR document is indexed as one chunk: the title becomes the heading
path and the text the chunk text, exactly as a section of a note would be.
The embedder truncates documents at 512 tokens; BM25 sees the full text.

## Quality

| Method | SciFact nDCG@10 | NFCorpus nDCG@10 | FiQA nDCG@10 | SciFact R@10 | NFCorpus R@10 | FiQA R@10 |
|---|---:|---:|---:|---:|---:|---:|
| `f32` (exact, reference) | 0.713 | 0.344 | 0.403 | 0.836 | 0.162 | 0.464 |
| `int8` | 0.711 | 0.342 | 0.403 | 0.840 | 0.162 | 0.463 |
| `binary` | 0.626 | 0.279 | 0.294 | 0.745 | 0.132 | 0.353 |
| `binary+int8` | 0.703 | 0.341 | 0.399 | 0.821 | 0.160 | 0.457 |
| `ternary+int8` (default) | 0.711 | 0.342 | 0.403 | 0.840 | 0.161 | 0.463 |
| `bm25` | 0.678 | 0.305 | 0.238 | 0.803 | 0.146 | 0.296 |
| `hybrid` (what `notelight search` returns) | 0.719 | 0.358 | 0.376 | 0.841 | 0.174 | 0.464 |

What the table says:

- **The f32 numbers match the published ones.** The MTEB leaderboard lists
  bge-small-en-v1.5 at about 0.71 (SciFact), 0.34 (NFCorpus) and 0.40 (FiQA)
  nDCG@10. That checks the embedder, the metrics and the dataset handling
  end to end.
- **int8 is free.** Exhaustive int8 is within 0.002 nDCG@10 of f32 on all
  three datasets, at a quarter of the size.
- **Binary alone is not enough.** Sign bits lose 0.06 to 0.11 nDCG@10.
- **Re-scoring 200 binary candidates recovers most of it.** `binary+int8`
  is 0.004 to 0.010 below f32.
- **Ternary candidates recover all of it, so they are the default.** With the
  same 200 candidates, `ternary+int8` is indistinguishable from exhaustive
  int8 (identical nDCG@10 to three decimals on all three datasets). Dropping
  the near-zero dimensions makes the candidate stage pick the right 200
  almost every time. Two bit planes instead of one cost about 0.3 ms more per
  query on FiQA (0.85 ms instead of 0.52 ms). The masks are stored in the
  index (schema version 2), about 11% more vector data.
- **Hybrid is not a free win.** Where BM25 is nearly as good as the
  embeddings (SciFact, NFCorpus), the fusion beats both (0.719 and 0.358,
  against 0.711 and 0.342 for the vectors). On FiQA, where BM25 is 0.17
  nDCG@10 behind, it still drags the fused list down (0.376 vs 0.403).
  Weighting each BM25 hit by the query terms it contains
  ([below](#fusion-weighting-keyword-hits-by-coverage)) halved that loss:
  with plain RRF, which gave each retriever equal say, FiQA's hybrid was
  0.351. Notes, with their exact names, identifiers and error codes, are
  closer to the first case, which is why hybrid stays the default, but the
  number is reported as it is. The [notes benchmark](#notes-benchmark)
  measures that case: there BM25 is ahead of the vectors, and hybrid scores
  highest.
- **BM25 has no stemming** (by design, see the design notes). Published
  BM25 (Anserini, with stemming) scores 0.665 / 0.325 / 0.236 nDCG@10 on these
  datasets; notelight's 0.678 / 0.305 / 0.238 is in the same range.

### Phrase matching

`hybrid` includes the phrase rules (see the design notes, "Phrase search
decisions"): chunks containing every quoted phrase rank first, and a chunk
with all query terms in order gains `phrase_boost / 61`. The weight was
swept with `notelight eval --methods hybrid --phrase-boost <w>`, with the
plain RRF of the time (before coverage weighting):

| `phrase_boost` | SciFact nDCG@10 | NFCorpus nDCG@10 | FiQA nDCG@10 |
|---|---:|---:|---:|
| none (before phrase matching) | 0.710 | 0.350 | 0.349 |
| 0 (quoted phrases only) | 0.710 | 0.350 | 0.352 |
| 0.5 | 0.712 | 0.350 | 0.351 |
| **1 (default)** | 0.712 | 0.350 | 0.351 |
| 2 | 0.712 | 0.349 | 0.351 |
| 4 | 0.712 | 0.349 | 0.351 |

BEIR queries are long questions whose terms rarely appear in order, so the
bonus barely moves them (at most 0.002). It is meant for short, phrase-like
queries over notes, which BEIR does not measure; the bar here is that it
does no harm. Among the evaluated (test) queries only FiQA has quotes: 33 of
648, such as `Why is “cheque cashing” a legitimate business?`. They account
for FiQA's +0.003 at weight 0.

## Notes benchmark

BEIR asks long questions of papers and forum posts. People search their notes
differently: two keywords, an identifier, a quoted phrase, a short question.
`notelight eval notes` measures that on a small, labelled benchmark bundled
with the package (`python/notelight/eval/notes/`).

### How it was built

- **Corpus**: this repository's own Markdown at commit `6a9d410` (README,
  CHANGELOG, CLAUDE.md, `docs/` and `docs/learning/`; 14 files), stored with a
  SHA-256 per file. The default chunker cuts it into 229 chunks.
- **Queries**: 82, in four styles: 28 keyword queries (1 to 4 words, such as
  `mmap windows delete`), 22 exact phrases or identifiers (`SCHEMA_VERSION`,
  `"lock file"`), 22 short questions (`what happens if a commit crashes
  halfway?`), and 10 paraphrases that share no indexed term with their answer
  (`stop two programs modifying simultaneously`, for the writer lock).
- **Labels**: each query lists the passages that answer it (grade 2) or
  partly answer it (grade 1), as a path and line range, 1 to 6 per query.
  Labels are line ranges rather than chunk ids, so they hold for any chunk
  size. A retrieved chunk counts for every labelled passage it overlaps, and
  each passage counts once, at the first chunk that shows it.
- **Process**: the queries and labels were written from what each section is
  for, after reading every file and before any search ran over the corpus.
  They were committed before the first search (commit `df61276`), and no
  label was changed after seeing results. (Before that commit, four
  paraphrases were reworded because a term-overlap check, not a search,
  showed they shared a word with their answer.) Each query records the
  section it came from. Within each style the queries alternate between a **tune** half
  (41) and a **held-out** half (41), fixed when the file was written.
- **Protocol**: settings were compared on the tune half only. The held-out
  runs, and the rule for changing a default, were written down before the
  held-out half was run once: change a default only if the held-out paired
  95% bootstrap interval of the nDCG@10 difference excludes zero and agrees
  with tune.
- **Query set 2** (`notelight eval notes2`): once set 1's halves had
  informed three decisions, 80 new queries over the same corpus were added
  the same way (24 keyword, 16 phrase, 20 question, 20 paraphrase; 40 per
  half), avoiding needs set 1 already covers, and committed with the next
  experiment's plan before any search with them (commit `833cc8c`). Their
  author knew set 1's results and the hypothesis to be tested, which the
  held-out half and BEIR are there to check. `notelight eval notes` is
  still set 1, so the numbers above reproduce.

**Limits.** The benchmark is small: 41 queries per half, and 5 paraphrases
per half. Paired differences under about 0.04 nDCG@10 are within noise, and
the two halves differ by 0.05 for the same settings. A single author wrote
every query and label, and that author is an LLM (Claude), not a person
searching their own notes. Queries written right after reading a document
tend to reuse its words, which favours BM25; only the paraphrases avoid that
by construction. The corpus is the project's own documentation, which is
more technical and more repetitive than personal notes (the same design is
described in the README, the design notes and a walkthrough). At 229 chunks,
200 candidates re-score nearly everything, so this benchmark says nothing
about the quantized candidate stage; BEIR covers that. Treat it as a check
of the defaults on note-style queries, not as a general result.

### Held-out results (default settings)

`max_chars` 1200 and `phrase_boost` 1, run once:

| Method | nDCG@10 | R@10 | MRR@10 |
|---|---:|---:|---:|
| `f32` (exact, reference) | 0.631 | 0.664 | 0.756 |
| `int8` | 0.639 | 0.684 | 0.756 |
| `binary` | 0.565 | 0.592 | 0.717 |
| `binary+int8` | 0.639 | 0.684 | 0.756 |
| `ternary+int8` (default) | 0.639 | 0.684 | 0.756 |
| `bm25` | 0.667 | 0.752 | 0.692 |
| `hybrid` (what `notelight search` returns) | **0.759** | **0.839** | **0.814** |
| `hybrid` with plain RRF (before coverage weighting) | 0.715 | 0.794 | 0.769 |

nDCG@10 by query style (number of queries in brackets):

| Method | keyword (14) | phrase (11) | question (11) | paraphrase (5) |
|---|---:|---:|---:|---:|
| `ternary+int8` | 0.699 | 0.753 | 0.622 | 0.257 |
| `bm25` | 0.813 | 0.907 | 0.546 | 0.000 |
| `hybrid` | 0.890 | 0.906 | 0.684 | 0.237 |
| `hybrid`, plain RRF | 0.833 | 0.907 | 0.659 | 0.082 |

The tune half, same settings, for comparison: `ternary+int8` 0.524,
`bm25` 0.647, `hybrid` 0.688 (0.663 with plain RRF) nDCG@10.

What the numbers say:

- **Hybrid is the best default for notes.** It beats vector search alone on
  both halves: by 0.121 nDCG@10 on held-out (95% CI 0.067 to 0.184; 27
  queries better, 7 worse) and by 0.164 on tune. It is also ahead of BM25
  alone on both halves: by 0.092 on held-out (0.045 to 0.143; 21 better,
  8 worse) and by 0.041 on tune, where the interval includes zero. With
  plain RRF the lead over BM25 was within noise on both halves.
- **Keywords carry note search.** Unlike on BEIR, BM25 alone beats the
  embeddings here. It is far ahead on identifiers and phrases (0.907 vs
  0.753), because a name like `IndexBusyError` means nothing to bge-small.
  The vectors win on questions and paraphrases.
- **Paraphrases are still the weak spot, but fusion no longer hides the
  vectors' hits.** With no shared word, BM25 scores 0. Plain RRF cut what the
  vectors alone find by two thirds (0.082 vs 0.257 nDCG@10, on 5 queries);
  coverage weighting keeps most of it (0.237). bge-small finds about half of
  these passages in its top 10, but rarely first. With 10 paraphrases in all
  this is a direction, not a measured trade-off on its own; the overall gain
  of the weighting (next section) is measured.
- **The quantized searches are exact here.** At 229 chunks `ternary+int8`
  and `binary+int8` return what exhaustive int8 does. `f32` is 0.008 below
  int8 on held-out and 0.007 above on tune: quantization noise in both
  directions.

### Fusion: weighting keyword hits by coverage

Plain RRF adds `1 / (60 + rank)` from each list, so a BM25 hit counts as much
as a vector hit at the same rank, whatever it matched. On a paraphrase BM25
still returns up to 50 hits, each on an incidental word (`two`, `update`,
`output`), and their top scores (5.6 to 9.2) are in the same range as real
keyword hits (5.0 to 15), so the score cannot tell them apart. What can is
term coverage: on the tune half, the best BM25 hit contains 25 to 40% of a
paraphrase's distinct terms, and at least 50% of every other query's
(keyword mean 93%, phrase 100%, question 74%). Of the relevant chunks in the
vector top 10, plain fusion pushed 3 of 4 out on paraphrases, against 3 of
24 on questions, 1 of 31 on keywords and 1 of 17 on phrases.

So hybrid search now weighs each BM25 hit: it adds
`(matched / total)² / (60 + rank)`, where the chunk contains `matched` of the
query's `total` distinct terms. A chunk with every term counts fully, one
with 1 of 4 terms counts 1/16. Vector hits are unchanged, and keyword-only
search (no embedder) keeps BM25's order. `--coverage-power 0` gives plain
RRF.

Seven candidates were written down after the diagnosis and before any of
them ran, with a rule to pick one on tune (highest hybrid nDCG@10 that lowers
no style by more than 0.02). Hybrid on the **tune** half:

| Candidate | nDCG@10 | keyword | phrase | question | paraphrase | vs plain RRF (95% CI) | better / worse |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain RRF, k = 60 (the old default) | 0.663 | 0.728 | 0.838 | 0.652 | 0.126 | | |
| RRF k = 20 | 0.672 | 0.731 | 0.838 | 0.680 | 0.126 | +0.009 (−0.000 to +0.021) | 5 / 1 |
| RRF k = 100 | 0.663 | 0.728 | 0.838 | 0.652 | 0.126 | 0.000 | 0 / 0 |
| BM25 weight 0.5 | 0.659 | 0.717 | 0.822 | 0.665 | 0.126 | −0.004 (−0.018 to +0.011) | 4 / 12 |
| BM25 hit × coverage | 0.682 | 0.746 | 0.838 | 0.648 | 0.233 | +0.018 (+0.000 to +0.040) | 10 / 5 |
| **BM25 hit × coverage² (picked)** | **0.688** | 0.749 | 0.838 | 0.672 | 0.220 | +0.024 (−0.005 to +0.058) | 12 / 5 |
| BM25 hit only if coverage ≥ 0.5 | 0.681 | 0.720 | 0.845 | 0.679 | 0.215 | +0.018 (−0.018 to +0.056) | 8 / 4 |
| no BM25 if its best hit has coverage < 0.5 | 0.674 | 0.728 | 0.838 | 0.652 | 0.215 | +0.011 (−0.016 to +0.042) | 3 / 1 |

The held-out half was then run once, for the pick and plain RRF:

| Hybrid on **held-out** | nDCG@10 | R@10 | MRR@10 | difference (95% CI) | queries better / worse |
|---|---:|---:|---:|---:|---:|
| plain RRF | 0.715 | 0.794 | 0.769 | | |
| coverage² | 0.759 | 0.839 | 0.814 | +0.045 (+0.012 to +0.083) | 14 / 7 |

That meets the benchmark's rule for changing a default: the held-out
interval excludes zero and agrees with tune. The gain is not only on
paraphrases (+0.155, on 5 queries): keyword queries gain 0.057 and questions
0.025, because a chunk that matches only some of the words no longer
outranks the vector hits that have the meaning. Phrase and identifier
queries match every term, so they barely move (−0.001).

The rule also required no BEIR regression beyond noise, judged by the
paired 95% bootstrap interval of the per-query difference (10,000
resamples) over the same queries. BEIR improved instead:

| Hybrid nDCG@10 | SciFact | NFCorpus | FiQA |
|---|---:|---:|---:|
| plain RRF | 0.712 | 0.350 | 0.351 |
| coverage² | 0.719 | 0.358 | 0.376 |
| difference (95% CI) | +0.006 (−0.013 to +0.026) | +0.008 (+0.000 to +0.017) | +0.025 (+0.014 to +0.037) |
| queries better / worse | 47 / 51 | 63 / 59 | 195 / 118 |

What did not work: changing RRF's `k` barely moves anything (k = 100 changes
no tune query), and a fixed lower BM25 weight costs keyword and phrase
queries as much as it saves elsewhere. A rule on the query as a whole (drop
BM25 when its best hit is weak) helps paraphrases and nothing else. Plain
coverage, without the square, came second on tune and is also positive on
BEIR (+0.014 on FiQA, +0.005 on NFCorpus, −0.003 on SciFact), but less so.
The square is one more choice made on the notes tune half; the held-out and
BEIR gains, which played no part in choosing it, are the evidence that it
is not just fitted to it. Counting the terms each hit contains costs 0.02 to
0.06 ms per query (see [Latency](#latency)).

### Fusion: weighting vector hits (tested, not shipped)

Coverage weighting left two gaps: on FiQA, hybrid (0.376) stays below the
vectors alone (0.403), and on set 1's tune half keyword queries scored lower
fused (0.749) than with BM25 alone (0.827), as if the vectors diluted good
keyword hits. The natural fix for the second is a weight on the vector side
when BM25 is clearly right. It was tested on query set 2, with five
candidates, the pick rule and the ship rule written down and committed
before any set-2 query ran (`c1` is the share of query terms in BM25's top
hit; BM25 hits keep coverage²):

- **V1**: vector hits count 0.5 when BM25's top hit has every query term;
- **V2**: vector hits count `1 − 0.5·c1²`;
- **V3**: each vector hit counts `(1 + coverage²) / 2`, by the terms it contains;
- **B3**: BM25 hits count coverage³ instead of coverage² (for FiQA, where the
  vectors are already better);
- the best V with B3, if both beat the default on tune (B3 did not).

The search was the Python copy of `IndexReader::search` used before, which
returned the core's top 10 for every notes and BEIR query. First, what set
2 says about the gap itself (core, default settings, nDCG@10):

| Set 2 | vectors | BM25 | hybrid | keyword: vectors / BM25 / hybrid |
|---|---:|---:|---:|---:|
| tune (40) | 0.417 | 0.692 | 0.695 | 0.427 / 0.933 / 0.913 |
| held-out (40) | 0.508 | 0.711 | 0.790 | 0.547 / 0.922 / 0.936 |

On new keyword queries the fused list is 0.020 below BM25 on tune and 0.014
above it on held-out: the large gap on set 1's tune half did not come back.
Hybrid on set 2's **tune** half:

| Candidate | nDCG@10 | keyword | phrase | question | paraphrase | vs default (95% CI) | better / worse |
|---|---:|---:|---:|---:|---:|---:|---:|
| default (coverage²) | 0.695 | 0.913 | 0.909 | 0.892 | 0.062 | | |
| **V1 (picked)** | 0.696 | 0.913 | 0.918 | 0.892 | 0.062 | +0.002 (−0.001 to +0.006) | 1 / 1 |
| V2 | 0.695 | 0.913 | 0.918 | 0.886 | 0.062 | +0.000 (−0.013 to +0.012) | 2 / 2 |
| V3 | 0.693 | 0.913 | 0.919 | 0.881 | 0.058 | −0.002 (−0.016 to +0.011) | 2 / 3 |
| B3 | 0.693 | 0.913 | 0.909 | 0.885 | 0.062 | −0.002 (−0.004 to +0.000) | 0 / 2 |

V1 won by the rule, on two changed queries. The **held-out** half, run once
for V1 and the default: 0.792 against 0.790, +0.002 (0.000 to +0.004; 3
queries better, none worse). The interval does not exclude zero (4.5% of
resamples show no gain), so by the rule the default stays. For information,
on BEIR (hybrid nDCG@10, difference to the default and its 95% interval):

| | SciFact | NFCorpus | FiQA |
|---|---:|---:|---:|
| default | 0.719 | 0.358 | 0.376 |
| V1 | +0.000 (+0.000 to +0.000) | +0.000 (−0.003 to +0.004) | −0.002 (−0.004 to −0.001) |
| V2 | −0.008 (−0.013 to −0.002) | −0.000 (−0.004 to +0.003) | −0.013 (−0.018 to −0.008) |
| V3 | −0.023 (−0.034 to −0.013) | −0.008 (−0.013 to −0.002) | −0.034 (−0.043 to −0.025) |
| B3 | +0.003 (−0.004 to +0.010) | +0.003 (−0.000 to +0.006) | +0.009 (+0.004 to +0.014) |

- **Weighting the vectors down buys nothing on notes and costs on BEIR.** V1
  fires rarely (BM25's top hit must contain every term) and changes almost
  nothing anywhere; on set 1 it is +0.006 on tune and −0.005 on held-out.
  The smoother and per-hit versions lose up to 0.034 on FiQA, where the
  vectors carry the result.
- **The keyword gap was mostly set 1's tune half.** With fresh queries the
  vectors no longer pull good keyword hits down on average, which is the
  kind of difference 12 to 14 queries per style can show by chance.
- **FiQA stays open.** By construction a vector-side weight cannot close a
  gap where the vectors are the better retriever. Sharper keyword weighting
  (B3) helps all three BEIR sets, FiQA outside noise (0.385, still below
  0.403), but it lost on both notes tune halves (−0.002 on set 2, −0.007 on
  set 1), so it was not picked. It is the lead for a next attempt, which
  would need queries that have not yet informed a decision.

### Tuning: chunk size and phrase weight

These sweeps ran with plain RRF, before coverage weighting.

Hybrid nDCG@10 / MRR@10 on the **tune** half:

| `max_chars` | chunks | `phrase_boost` 0 | 1 (default) | 4 |
|---|---:|---:|---:|---:|
| 600 | 374 | 0.648 / 0.793 | 0.676 / 0.804 | 0.685 / 0.816 |
| **1200 (default)** | 229 | 0.631 / 0.740 | 0.663 / 0.732 | 0.672 / 0.743 |
| 2000 | 186 | 0.622 / 0.733 | 0.656 / 0.734 | 0.665 / 0.746 |

Against the defaults, no cell's 95% interval for nDCG@10 excludes zero. The
best, 600 characters with weight 4, was +0.022 (−0.017 to +0.064). It went
to the held-out check with the single-factor variants, each run once:

| Hybrid on **held-out**, vs the defaults | nDCG@10 | difference (95% CI) | queries better / worse |
|---|---:|---:|---:|
| defaults: 1200, weight 1 | 0.715 | | |
| 600, weight 4 (tune's best) | 0.690 | −0.024 (−0.069 to +0.016) | 11 / 18 |
| 600, weight 1 | 0.690 | −0.024 (−0.069 to +0.016) | 11 / 18 |
| 1200, weight 4 | 0.715 | 0.000 | 0 / 0 |
| 1200, weight 0 (no bonus for terms in order) | 0.696 | −0.018 (−0.048 to +0.003) | 1 / 5 |

- **The default chunk size holds.** Smaller chunks looked better on tune and
  were worse on held-out, which is what noise looks like at 41 queries. They
  do raise MRR on both halves (+0.073 tune, +0.022 held-out, neither
  interval excluding zero), because a short chunk that hits ranks higher.
  But it shows less context per hit and needs 63% more chunks to embed.
- **The phrase bonus earns its place.** Weight 0 is worse on both halves
  (−0.032 tune, −0.018 held-out). The losses are on phrases and
  identifiers: without the bonus every phrase query that changed got worse
  (6 on tune, 4 of 5 on held-out), including single identifiers such as
  `IndexBusyError`, whose camelCase parts count as terms in order. On
  keyword queries it is mixed (3 better, 3 worse on tune). Weight 4 changes
  two tune queries (one up by 0.37, one down by 0.02) and no held-out query,
  so 1 stays; BEIR showed no harm at 1.
- No default changed. Differences are paired over the same queries (10,000
  bootstrap resamples); the per-query scores are in `notelight eval --json`.

## Latency

Median milliseconds per query, measured from Python around one call (so each
includes a few microseconds of binding overhead), excluding query embedding.
`f32` is NumPy; every other method is the Rust core on one thread.

| Method | SciFact | NFCorpus | FiQA |
|---|---:|---:|---:|
| `f32` (NumPy) | 0.22 | 0.20 | 3.53 |
| `int8` | 0.37 | 0.26 | 4.83 |
| `binary` | 0.05 | 0.04 | 0.46 |
| `binary+int8` | 0.07 | 0.06 | 0.52 |
| `ternary+int8` (default) | 0.10 | 0.08 | 0.85 |
| `bm25` | 0.05 | 0.01 | 0.27 |
| `hybrid` | 0.24 | 0.15 | 1.31 |

`hybrid` was re-measured after coverage weighting; with plain RRF, run back
to back on the same machine, it took 0.18, 0.13 and 1.27 ms, so counting the
query terms in each BM25 hit costs 0.02 to 0.06 ms per query. The other rows
are from the earlier full run.

Embedding the query with bge-small takes about 5 ms on this CPU, more than
any search here, so for a personal folder the search is never the
bottleneck.

Every search reads the ternary masks straight from the index, so there is no
warm-up cost. Before schema version 2 the masks were derived from the int8
codes on the first ternary search, which took 117 ms on FiQA for every newly
opened index.

## Rust vs NumPy

The same four searches, implemented in NumPy the way a NumPy user would
(`python/notelight/eval/baseline.py`), on the same queries:

| Dataset | Search | Rust ms | NumPy ms | NumPy / Rust | Top-10 agreement |
|---|---|---:|---:|---:|---:|
| SciFact | `int8` | 0.41 | 0.24 | 0.6x | 1.000 |
| SciFact | `binary` | 0.05 | 0.60 | 11.3x | 1.000 |
| SciFact | `binary+int8` | 0.10 | 0.69 | 6.9x | 1.000 |
| SciFact | `ternary+int8` | 0.13 | 1.39 | 10.5x | 1.000 |
| NFCorpus | `int8` | 0.28 | 0.22 | 0.8x | 1.000 |
| NFCorpus | `binary` | 0.04 | 0.43 | 11.6x | 1.000 |
| NFCorpus | `binary+int8` | 0.08 | 0.51 | 6.8x | 1.000 |
| NFCorpus | `ternary+int8` | 0.10 | 1.02 | 10.2x | 1.000 |
| FiQA | `int8` | 4.62 | 3.59 | 0.8x | 1.000 |
| FiQA | `binary` | 0.53 | 7.72 | 14.4x | 1.000 |
| FiQA | `binary+int8` | 0.59 | 7.85 | 13.3x | 1.000 |
| FiQA | `ternary+int8` | 0.95 | 17.8 | 18.8x | 1.000 |

- **Agreement is 1.000 everywhere**: for every query, both implementations
  return the same top 10. That cross-checks the Rust bit packing, int8
  rounding, ternary masks and tie-breaking against an independent
  implementation.
- **The two-stage searches are 7 to 19x faster in Rust**: 7 to 13x with
  binary candidates, 10 to 19x with ternary ones (the default), where NumPy
  makes two more full-size temporaries for the masks. XOR + popcount over
  packed `u64` words and a histogram cut-off beat NumPy's
  `bitwise_xor` + `bitwise_count` + `argpartition`, which allocate a full-size
  temporary at every step.
- **The exhaustive int8 scan is still 1.3 to 1.7x slower in Rust.** NumPy
  keeps the codes dequantized as float32 (4x the memory) and hands the
  product to a multi-threaded, SIMD BLAS kernel. The first version of the
  Rust `int8_dot` was a scalar loop with one `f32` accumulator, which the
  compiler may not vectorize because that would change the order of the
  additions, and it lost by 5x (18.6 ms vs 3.58 ms on FiQA). Keeping 16
  partial sums lets it use SIMD on one thread and cut the scan 4x, to
  4.6 ms; it also speeds up the rescoring in every two-stage search. See
  [the learning note](learning/m3-int8-dot-vectorization.md). Building for
  the local CPU (AVX2) would roughly halve it again, but wheels have to run
  on any x86-64.

## What changes next

- **AVX2 for `int8_dot` via runtime feature detection**, which would close
  the remaining gap to NumPy on exhaustive scans without giving up portable
  wheels. Done so far: 16 partial sums (4x on the default target).

## Reproducing

```powershell
.venv\Scripts\activate
notelight eval                                 # all three datasets, all methods
notelight eval scifact --methods f32,bm25      # a subset
notelight eval path\to\beir-folder --json      # any folder in the BEIR layout

# the notes benchmark: bundled, no download beyond the model
notelight eval notes                                        # tune half, defaults
notelight eval notes --max-chars 600 --phrase-boost 4       # one sweep cell
notelight eval notes --coverage-power 0                     # plain RRF fusion
notelight eval notes --split heldout                        # the held-out half
notelight eval notes2                                       # query set 2, tune half
```

Tune on the tune half. Run the held-out half once, for settings chosen
beforehand, or it stops being held out. `--json` includes per-query scores
for paired comparisons. The notes corpus is re-created from git with
`python scripts/make_notes_corpus.py` (`--check` verifies the bundled copy).

The first run downloads the datasets (23 MB, verified by SHA-256) and the
model (134 MB), and embeds the corpora: about 6 minutes for SciFact, 5 for
NFCorpus and 40 for FiQA on the machine above. Embeddings and indexes are
cached under the user cache directory (`NOTELIGHT_HOME\eval` if that is
set), so later runs take under a minute. Latencies vary with the machine and
its load; quality numbers do not.
