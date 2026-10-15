# Milestone 3: Rust walkthrough

Milestone 3 is the evaluation harness, and most of it is Python
(`python/notelight/eval/`). The Rust in it is a new candidate strategy,
**ternary codes**, plus a `VectorMode` enum that lets the harness run each
quantization strategy on its own. This guide covers the Rust ideas involved,
mapped to C# and Python. It takes about 40 minutes.

**Suggested order:** `quant.rs` (`ternary_mask`, `ternary_dot`) →
`vectors.rs` (`VectorMode`, `ternary_masks`, `search_with`) →
`tests/properties.rs` (the three new properties) → the one-line binding change
in `crates/notelight-py/src/index.rs`.

## 0. What ternary codes are, and why they live in the core

A binary code keeps one bit per dimension: the sign. A ternary code keeps
`-1`, `0` or `+1`: dimensions whose magnitude is small (at most 0.7 × the
vector's mean absolute value) become `0`, because their sign is mostly noise.
Two ternary codes are compared with a dot product that only counts dimensions
significant in *both* vectors.

The question for this milestone was whether ternary belongs in the Rust core.
It does, for the same reason the binary stage does: it is a tight loop over
every vector per query, and the browser build (v2) will need the same code.
But the **on-disk format does not change**. A ternary code is the binary code
(the sign, already stored) plus a mask (the "is it significant" bit), and the
mask can be derived from the int8 codes that are also already stored. So the
mask is computed in memory on first use, and `SCHEMA_VERSION` stays at 1. If
ternary ever becomes the default, storing the masks is a later, separate
format change.

## 1. Two bit planes and one popcount formula (`ternary_dot`)

Each ternary code is two bit-packed planes: `sign` (bit set = positive) and
`mask` (bit set = significant). For one dimension:

| both significant? | signs agree? | contributes |
|---|---|---|
| no | – | 0 |
| yes | yes | +1 |
| yes | no | −1 |

Over 64 dimensions at once, with `m = mask_a & mask_b`:

```rust
both   += m.count_ones();                 // dimensions significant in both
differ += (m & (sa ^ sb)).count_ones();   // ...whose signs differ
// dot = (both - differ) - differ = both - 2 * differ
```

`count_ones` compiles to a single `popcnt` instruction, the same trick the
Hamming distance uses. In C# it's `BitOperations.PopCount`; in Python,
`int.bit_count()`.

The result is an `i64`, built with `i64::from(both)`. An earlier version wrote
`both as i32`, and clippy pedantic rejected it (`cast_possible_wrap`): a `u32`
that large would wrap to a negative `i32`. `i64::from` is a *lossless*
conversion, so there is nothing to argue about. Rust only implements `From`
for conversions that can never fail or lose data; everything else needs
`try_from` (see section 4).

## 2. Arrays of slices, and `map` on an array (`ternary_dot`)

```rust
// Each is (whole 8-byte words, leftover bytes).
let [sa, ma, sb, mb] = [sign_a, mask_a, sign_b, mask_b].map(<[u8]>::as_chunks::<8>);
for (((s1, m1), s2), m2) in sa.0.iter().zip(ma.0).zip(sb.0).zip(mb.0) {
    tally(w(*s1), w(*m1), w(*s2), w(*m2));
}
```

Three ideas in two lines:

- **`[T; 4].map(f)`** applies `f` to every element of a fixed-size array and
  returns a new array, like `Select` on a C# array, but the length (4) is part
  of the type, so the destructuring `let [sa, ma, sb, mb] = ...` cannot fail.
- **`<[u8]>::as_chunks::<8>`** names a method by its type, the way you'd
  pass `string.IsNullOrEmpty` as a method group in C#. `<[u8]>` needs angle
  brackets because `[u8]` is not a plain identifier. The result is a tuple
  `(&[[u8; 8]], &[u8])`, read as `.0` and `.1`, like a C# `ValueTuple`'s
  `Item1` and `Item2`.
- **`zip` three times** gives nested pairs, `(((a, b), c), d)`, and the `for`
  pattern takes them apart in place. Python's `zip(a, b, c, d)` is flatter;
  Rust's `zip` is strictly pairwise.

This version replaced one that copied each 8-byte window into a buffer by
index. Clippy didn't object to that one, but the bindings it then needed
(`words_sa`, `words_ma`, ...) tripped `similar_names`, and the slice-of-arrays
version has no indexing at all, so no bounds checks and nothing that can
panic.

`tally` is a closure that mutates `both` and `differ`, which is why it's
declared `let mut tally = |...| { ... }`. A closure that changes captured
variables is an `FnMut`; calling it requires a mutable binding, just as
calling a `&mut self` method does.

## 3. Lazy, thread-safe state: `OnceLock` (`Vectors::ternary_masks`)

> **Update after milestone 3:** once ternary became the default search, the
> masks moved into `vectors.bin` (schema version 2), so this `OnceLock` no
> longer exists: a CLI query opens a fresh reader every time and would pay for
> the derivation on each one (see `docs/design-notes.md`, "Ternary is the
> default"). The pattern still applies; to read the code as it was, run
> `git show d6d50aa:crates/notelight-core/src/vectors.rs`.

```rust
pub struct Vectors<B> {
    ...
    ternary_masks: OnceLock<Vec<u8>>,
}

fn ternary_masks(&self) -> &[u8] {
    self.ternary_masks.get_or_init(|| { /* one pass over the int8 codes */ })
}
```

`OnceLock<T>` is C#'s `Lazy<T>` with
`LazyThreadSafetyMode.ExecutionAndPublication`, or Python's
`functools.cached_property` plus a lock: the first caller runs the closure,
concurrent callers wait, and everyone gets a `&T` to the same value
afterwards.

Notice that `ternary_masks` takes `&self`, not `&mut self`, yet it fills in a
field. That's *interior mutability*: `OnceLock` is one of the few types allowed
to change behind a shared reference, because it guarantees the write happens
at most once and is finished before any read. This matters for the Python
binding: `IndexReader` is a `frozen` pyclass, shared across threads, so it
only ever hands out `&self`. A plain `Option<Vec<u8>>` field would need
`&mut self` and would not compile there.

The type system also checks the threading claim. A frozen pyclass must be
`Sync` ("safe to share between threads"). `OnceLock<Vec<u8>>` is `Sync`; the
single-threaded `OnceCell` is not, and using it here is a compile error, not a
data race.

## 4. Enums with names: `Display`, `FromStr` and `const fn`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorMode {
    #[default]
    BinaryInt8,
    TernaryInt8,
    Binary,
    Int8,
}

impl VectorMode {
    pub const ALL: [Self; 4] = [Self::Int8, Self::Binary, Self::BinaryInt8, Self::TernaryInt8];
    pub const fn name(self) -> &'static str { match self { ... } }
}

impl FromStr for VectorMode {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> { ... }
}
```

- **`#[default]`** on a variant makes `#[derive(Default)]` work for an enum,
  so `VectorMode::default()` is `BinaryInt8`.
- **`Copy`** makes the enum behave like a C# `enum` or `int`: passing it
  copies it, and the original stays usable.
- **`Display`** is Rust's `ToString()` override; implementing it gives you
  `.to_string()` and `format!("{mode}")` for free.
- **`FromStr`** is the parse side, like a static `Parse` method. Implementing
  it makes `"ternary+int8".parse::<VectorMode>()` work. The `type Err = Error;`
  line is an *associated type*: each implementation chooses its own error
  type. The binding turns that error into a Python `ValueError` with the usual
  `.map_err(to_py)?`, so the Python layer accepts a plain string and the
  validation lives in the core.
- **`const fn`** can run at compile time. `name` qualifies because it's a
  `match` returning string literals. `&'static str` means "a string that lives
  for the whole program": literals are baked into the binary.

There's one more conversion worth a look, in `search_with`:

```rust
let dim = i16::try_from(self.dim).map_err(|_| Error::Invalid(...))?;
```

Ternary "distances" are `dim - dot`, between `0` and `2 * dim`, and the
histogram selection stores them as `u16`. Requiring `dim` to fit an `i16`
guarantees `2 * dim` fits a `u16`. `try_from` is the checked conversion: it
returns a `Result` instead of silently truncating like an `as` cast, so an
oversized index gets a clear error rather than wrong results.

## 5. Testing against a naive reference (`tests/properties.rs`)

The new properties follow the milestone 1 pattern: write the obvious, slow
version and let proptest check the fast one against it on thousands of random
inputs.

- `ternary_dot_matches_reference`: map each value to `-1/0/+1` with plain
  `if`s, multiply, sum. It must equal the bit-plane formula.
- `binary_mode_matches_a_hamming_sort`: sort every id by
  `(hamming distance, id)` and truncate.
- `ternary_search_rescores_the_best_ternary_candidates`: rebuild every stored
  ternary code from the int8 codes, pick candidates by sorting, rescore them
  with the index's own int8 scores, and compare with `search_with`.

One detail in the first property is worth remembering. The reference computes
the threshold as `TERNARY_THRESHOLD * (sum / len)`, with the parentheses,
because that's the order the real code uses. Floating-point multiplication
and division are not associative: `0.7 * sum / len` can differ from
`0.7 * (sum / len)` in the last bit, and with 20 000 random cases one of them
would sit exactly on the threshold and fail for no real reason. When a
reference implementation does float math, mirror the operation order, or
compare with a tolerance.

Run them harder than CI does with:

```powershell
$env:PROPTEST_CASES=20000; cargo test --release -p notelight-core --test properties
```

## 6. What the measurements said

The harness (`notelight eval`) compares every mode on three BEIR datasets;
the numbers and their discussion are in [docs/results.md](../results.md).
Two results are directly about the Rust in this guide:

- The NumPy baseline returns the same top 10 as the Rust core for the same
  algorithm (agreement 1.000), which is a useful end-to-end check of the bit
  packing, the int8 rounding and the tie-breaking.
- An exhaustive int8 scan is *slower* in Rust than a NumPy (BLAS) float32
  matrix-vector product. `int8_dot` is a scalar loop with a single `f32`
  accumulator, which the compiler is not allowed to vectorize because that
  would reorder the additions. The two-stage search avoids the problem by
  rescoring only 200 candidates. A follow-up made the scan itself 4x faster
  with 16 partial sums; see
  [Making `int8_dot` vectorize](m3-int8-dot-vectorization.md).

## Exercises

1. On a checkout of `d6d50aa` (before the masks were stored), replace
   `OnceLock` with `std::cell::OnceCell` in `Vectors`. Build
   `notelight-py` and read the error. Which trait is missing, and which
   `#[pyclass]` option requires it?
2. Remove the parentheses in the reference threshold in
   `ternary_dot_matches_reference` and run with `PROPTEST_CASES=200000`. Does
   proptest find a failing case? What does the shrunk input look like?
3. `int8_dot` sums into one `f32`. Rewrite it to keep eight partial sums (one
   per lane of `as_chunks::<8>`), add them at the end, and compare
   `cargo bench -p notelight-core --bench search` before and after. Why may
   the results differ in the last decimal place, and why do the existing
   tests still pass?
