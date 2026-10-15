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
| `binary+int8` (default) | 0.703 | 0.341 | 0.399 | 0.821 | 0.160 | 0.457 |
| `ternary+int8` | 0.711 | 0.342 | 0.403 | 0.840 | 0.161 | 0.463 |
| `bm25` | 0.678 | 0.305 | 0.238 | 0.803 | 0.146 | 0.296 |
| `hybrid` (what `notelight search` returns) | 0.703 | 0.352 | 0.347 | 0.829 | 0.166 | 0.431 |

What the table says:

- **The f32 numbers match the published ones.** The MTEB leaderboard lists
  bge-small-en-v1.5 at about 0.71 (SciFact), 0.34 (NFCorpus) and 0.40 (FiQA)
  nDCG@10. That checks the embedder, the metrics and the dataset handling
  end to end.
- **int8 is free.** Exhaustive int8 is within 0.002 nDCG@10 of f32 on all
  three datasets, at a quarter of the size.
- **Binary alone is not enough.** Sign bits lose 0.06 to 0.11 nDCG@10.
- **Re-scoring 200 binary candidates recovers most of it.** The default
  `binary+int8` is 0.004 to 0.010 below f32.
- **Ternary candidates recover all of it.** With the same 200 candidates,
  `ternary+int8` is indistinguishable from exhaustive int8 (identical nDCG@10
  to three decimals on all three datasets). Dropping the near-zero dimensions
  makes the candidate stage pick the right 200 almost every time. It costs
  about 1.5x the binary scan (two bit planes instead of one), which on FiQA is
  1.0 ms instead of 0.6 ms per query. This is the strongest argument for
  making ternary the default; see [What changes next](#what-changes-next).
- **Hybrid is not a free win.** RRF gives each retriever equal say. Where
  BM25 is nearly as good as the embeddings (SciFact, NFCorpus), the fusion
  matches or beats either alone. On FiQA, where BM25 is 0.17 nDCG@10 behind,
  it drags the fused list down (0.347 vs 0.399). Notes, with their exact
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
| `f32` (NumPy) | 0.23 | 0.20 | 3.27 |
| `int8` | 1.65 | 1.15 | 18.7 |
| `binary` | 0.06 | 0.04 | 0.54 |
| `binary+int8` | 0.13 | 0.11 | 0.63 |
| `ternary+int8` | 0.16 | 0.13 | 1.01 |
| `bm25` | 0.04 | 0.01 | 0.26 |
| `hybrid` | 0.20 | 0.14 | 0.97 |

Embedding the query with bge-small takes about 5 ms on this CPU, more than
any search here except exhaustive int8 on FiQA, so for a personal folder the
default search is never the bottleneck.

The first `ternary+int8` search on an opened index also derives the ternary
masks from the int8 codes: 117 ms once for FiQA, excluded above (the harness
runs one warm-up query per method).

## Rust vs NumPy

The same three searches, implemented in NumPy the way a NumPy user would
(`python/notelight/eval/baseline.py`), on the same queries:

| Dataset | Search | Rust ms | NumPy ms | NumPy / Rust | Top-10 agreement |
|---|---|---:|---:|---:|---:|
| SciFact | `int8` | 1.67 | 0.34 | 0.2x | 1.000 |
| SciFact | `binary` | 0.06 | 0.59 | 10.0x | 1.000 |
| SciFact | `binary+int8` | 0.15 | 0.68 | 4.6x | 1.000 |
| NFCorpus | `int8` | 1.15 | 0.27 | 0.2x | 1.000 |
| NFCorpus | `binary` | 0.04 | 0.43 | 10.5x | 1.000 |
| NFCorpus | `binary+int8` | 0.12 | 0.52 | 4.2x | 1.000 |
| FiQA | `int8` | 18.6 | 3.58 | 0.2x | 1.000 |
| FiQA | `binary` | 0.63 | 7.64 | 12.1x | 1.000 |
| FiQA | `binary+int8` | 0.72 | 7.69 | 10.6x | 1.000 |

- **Agreement is 1.000 everywhere**: for every query, both implementations
  return the same top 10. That cross-checks the Rust bit packing, int8
  rounding and tie-breaking against an independent implementation.
- **The two-stage search is 4 to 11x faster in Rust.** XOR + popcount over
  packed `u64` words and a histogram cut-off beat NumPy's
  `bitwise_xor` + `bitwise_count` + `argpartition`, which allocate a full-size
  temporary at every step.
- **The exhaustive int8 scan is 5x slower in Rust.** This one Rust loses.
  NumPy keeps the codes dequantized as float32 (4x the memory) and hands the
  product to a multi-threaded, SIMD BLAS kernel. The Rust `int8_dot` is a
  scalar loop with one `f32` accumulator, which the compiler may not
  vectorize because that would change the order of the additions. The
  default search only re-scores 200 candidates, so this does not affect
  `notelight search`, but it is a clear target: several partial sums (or
  integer accumulation of a quantized query) would let the loop vectorize.

## What changes next

- **Ternary as the default candidate stage.** It matches exhaustive int8
  quality at about 1 ms per query on 57k documents. Making it the default means
  storing the masks (48 more bytes per 384-dim vector, about 11% more
  vector data, and a `SCHEMA_VERSION` bump) or keeping the one-time
  derivation on first search (117 ms at FiQA scale, once per opened index).
- **A vectorized `int8_dot`**, which also speeds up the rescoring step.

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
