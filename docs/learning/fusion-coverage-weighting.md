# Weighted fusion: the Rust behind coverage weighting

A short follow-up to the [milestone 1 walkthrough](m1-rust-walkthrough.md).
Hybrid search used to give BM25 and the vectors equal say. Now each BM25 hit
counts by the share of the query's terms it contains, squared (the numbers
are in [results.md](../results.md#fusion-weighting-keyword-hits-by-coverage)).
The change is about 60 lines in `fuse.rs`, `bm25.rs` and `index/reader.rs`.
It takes about 10 minutes to read.

## 1. A list of `(id, weight)` pairs: slices of tuples

Plain RRF took `&[&[u32]]`: a slice of borrowed lists of ids. The weighted
version takes `&[&[(u32, f32)]]`:

```rust
pub fn rrf_weighted(lists: &[&[(u32, f32)]], k: f32, limit: usize) -> Vec<(u32, f32)> {
    let mut scores: HashMap<u32, f32> = HashMap::new();
    for list in lists {
        let mut seen = HashSet::new();
        for (rank, &(id, weight)) in list.iter().enumerate() {
            if seen.insert(id) {
                *scores.entry(id).or_insert(0.0) += weight / (k + rank as f32 + 1.0);
            }
        }
    }
    top_k(scores, limit)
}
```

- `(u32, f32)` is a tuple, like C#'s `(uint Id, float Weight)` value tuple or
  a Python tuple, but stored inline: a `Vec<(u32, f32)>` is one block of
  8-byte pairs, not an array of pointers to tuple objects as in Python.
- `&(id, weight)` in the `for` pattern destructures each borrowed pair and
  copies both fields out, which is allowed because `u32` and `f32` are
  `Copy`. Python's `for rank, (id, weight) in enumerate(lst)` reads the same.
- `seen.insert(id)` returns `true` only the first time, like C#'s
  `HashSet<T>.Add`. That is how an id repeated in one list counts once, at
  its best rank.

Plain `rrf` keeps its signature and delegates: it turns each id list into
`(id, 1.0)` pairs. Multiplying by 1.0 changes no bits, so plain RRF gives
exactly the scores it did before, which the test
`weights_of_one_are_plain_rrf` checks.

```rust
let weighted: Vec<Vec<(u32, f32)>> = lists
    .iter()
    .map(|list| list.iter().map(|&id| (id, 1.0)).collect())
    .collect();
let refs: Vec<&[(u32, f32)]> = weighted.iter().map(Vec::as_slice).collect();
```

The second line is needed because a `Vec<Vec<T>>` is not a `&[&[T]]`: the
outer function wants a slice of *borrowed slices*, so we build a small `Vec`
of borrows (`Vec::as_slice` is the method used as a function, like
`list.AsSpan()` in C# passed as a method group). The owned `weighted` must
outlive `refs`, and the borrow checker makes sure it does.

## 2. Counting with an iterator: `filter(...).count()`

`Bm25` already answered "does chunk `doc` contain every one of these
terms?" with `contains_all`. Coverage needs "how many of them?". The binary
search over one term's postings moved into a private helper, and both
public methods became one line:

```rust
pub fn contains_all(&self, doc: u32, terms: &[String]) -> bool {
    terms.iter().all(|term| self.contains(doc, term))
}

pub fn count_contained(&self, doc: u32, terms: &[String]) -> usize {
    terms.iter().filter(|term| self.contains(doc, term)).count()
}
```

This is LINQ's `terms.All(t => Contains(doc, t))` and
`terms.Count(t => Contains(doc, t))`, or Python's `all(...)` and
`sum(1 for t in terms if ...)`. Like LINQ, `all` stops at the first `false`.
`self.contains(doc, term)` receives a `&String` where it asks for `&str`;
Rust converts it automatically (deref coercion), as C# would pass a
`string` where `ReadOnlySpan<char>` is wanted.

A proptest checks `count_contained` against the obvious reference
(`terms.iter().filter(|t| doc_terms.contains(t)).count()` over the raw
tokens), next to the existing one for `contains_all`.

## 3. The weight itself: integer-to-float casts and `powf`

```rust
pub fn coverage_weight(matched: usize, total: usize, power: f32) -> f32 {
    if total == 0 {
        return 1.0;
    }
    let share = matched.min(total) as f32 / total as f32;
    share.powf(power)
}
```

- Rust never converts integers to floats implicitly; `as f32` is the
  explicit cast, like `(float)matched` in C#. Writing `matched / total`
  without the casts would be integer division, 0 for any partial match, the
  same trap as in C#.
- `powf` is `MathF.Pow` / Python's `**`. `0.0_f32.powf(0.0)` is 1, so
  `power = 0` turns the weighting off for every hit, which is how
  `--coverage-power 0` reproduces plain RRF.
- `matched.min(total)` caps a caller mistake at 1 instead of returning a
  weight above 1. `usize::min` is a method on the integer, like
  `Math.Min(a, b)` written as `a.Min(b)`.

## 4. Adding a field to `SearchOpts`: struct update syntax

`SearchOpts` gained `coverage_power: f32`. Every place that builds one with
`..SearchOpts::default()` kept compiling, which is the point of that syntax:

```rust
let plain = SearchOpts {
    coverage_power: 0.0,
    ..one
};
```

`..one` means "every other field from `one`", like a C# `with` expression on
a record (`one with { CoveragePower = 0 }`). `SearchOpts` is `Copy`, so `one`
stays usable afterwards. The one place that listed every field by name, the
Python binding, stopped compiling until it passed the new field. That is a
feature: Rust has no default for a missing field in a struct literal, so a
new option cannot be silently left at zero.

## 5. Where the rule lives

The weighting happens in `IndexReader::search`, only when a query vector is
given:

```rust
let weigh = query_vec.is_some() && opts.coverage_power != 0.0;
```

Keyword-only search (an index without an embedder) keeps BM25's order: the
weights exist to balance two retrievers, and with one there is nothing to
balance. Because the change is in the core, the CLI, the MCP server, the
web UI, the browser app and the npm package all get it from the same code;
the parity fixtures in `web-demo/tests/fixtures/` were re-recorded, and the
browser's top 10 still match Python's exactly.
