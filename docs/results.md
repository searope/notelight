# Results

Retrieval quality and speed of every search method in notelight, measured on
three [BEIR](https://github.com/beir-cellar/beir) datasets with
`notelight eval`. Every number on this page comes from one run of that
command; see [Reproducing](#reproducing).

## Setup

| | |
|---|---|
| Embedder | `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405e`, ONNX on CPU, 384 dimensions |
| Datasets | BEIR test splits: SciFact (5,183 docs, 300 queries), NFCorpus (3,633 docs, 323 queries), FiQA-2018 (57,638 docs, 648 queries) |
| Metrics | nDCG@10 and recall@10, computed like `pytrec_eval` (linear gain) |
| Machine | Intel Core i9-10900K (10 cores), Windows 11, Python 3.14, NumPy 2.5.4, ONNX Runtime 1.31 |
| Search settings | the `notelight search` defaults: 200 candidates re-scored, 50 results per retriever fused with RRF (k = 60) |

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
| `hybrid` (what `notelight search` returns) | 0.710 | 0.350 | 0.349 | 0.839 | 0.165 | 0.424 |

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
- **Hybrid is not a free win.** RRF gives each retriever equal say. Where
  BM25 is nearly as good as the embeddings (SciFact, NFCorpus), the fusion
  matches the vectors (SciFact) or beats both (NFCorpus). On FiQA, where BM25
  is 0.17 nDCG@10 behind, it drags the fused list down (0.349 vs 0.403). Notes, with their exact
  names, identifiers and error codes, are closer to the first case, which is
  why hybrid stays the default, but the number is reported as it is.
- **BM25 has no stemming** (by design, see the design notes). Published
  BM25 (Anserini, with stemming) scores 0.665 / 0.325 / 0.236 nDCG@10 on these
  datasets; notelight's 0.678 / 0.305 / 0.238 is in the same range.

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
| `hybrid` | 0.16 | 0.10 | 1.16 |

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
```

The first run downloads the datasets (23 MB, verified by SHA-256) and the
model (134 MB), and embeds the corpora: about 6 minutes for SciFact, 5 for
NFCorpus and 40 for FiQA on the machine above. Embeddings and indexes are
cached under the user cache directory (`NOTELIGHT_HOME\eval` if that is
set), so later runs take under a minute. Latencies vary with the machine and
its load; quality numbers do not.
