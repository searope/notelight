# Milestone 1: Rust walkthrough

A reading guide to the retrieval core, for someone fluent in C# and Python.
It takes 1–2 hours to read everything. Each section names the file, the Rust
idea it shows, and the closest C# equivalent.

**Suggested order:** `quant.rs` → `topk.rs` → `tokenize.rs` → `bm25.rs` →
`vectors.rs` → `fuse.rs` → `chunk.rs` → `index/` → tests.

## 1. Modules and visibility (`lib.rs`)

```rust
pub mod bm25;              // public module: part of the crate's API
pub(crate) mod bin_format; // visible inside this crate only, like C# `internal`
pub use error::{Error, Result};  // re-export, so users write notelight_core::Error
```

A file `bm25.rs` is the module `bm25`. A folder `index/` with `mod.rs` is the
module `index`, and its submodules are the other files in that folder.

## 2. Errors as values (`error.rs`)

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error on {path}: {source}")]
    Io { path: PathBuf, #[source] source: std::io::Error },
    #[error("invalid index data: {0}")]
    Format(String),
    ...
}
pub type Result<T, E = Error> = std::result::Result<T, E>;
```

- A Rust `enum` can carry data in each variant. It's closer to a sealed class
  hierarchy (or a discriminated union) than to a C# enum.
- `thiserror` generates the `Display` and `std::error::Error` impls from the
  `#[error(...)]` strings.
- Callers can `match` on the variant: `Err(Error::Schema { found, .. })` is how
  the Python layer will say "rebuild your index" instead of showing a generic
  error.

## 3. Traits for ordering (`topk.rs`)

```rust
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score).then_with(|| other.id.cmp(&self.id))
    }
}
```

This is `IComparable<T>`. `BinaryHeap` is a max-heap; wrapping items in
`Reverse(...)` turns it into a min-heap, which is how we keep the k *best*
items while evicting the worst. `f32` has no total order (NaN), so we use
`total_cmp`, and the code maps NaN to `-inf` first.

`smallest_k_by_histogram` is a small algorithmic idea worth knowing: when keys
are small integers, counting beats sorting.

## 4. Generic views over bytes (`bm25.rs`, `vectors.rs`, `index/docs.rs`)

```rust
pub struct Bm25<B> { bytes: B, n_docs: usize, ... }

impl<B: AsRef<[u8]>> Bm25<B> {
    pub fn from_bytes(bytes: B) -> Result<Self> { ... }
}
```

`B` can be a `Vec<u8>` (owned), a `&[u8]` (borrowed), or our `Storage` enum
(a memory map or a `Vec`). The search code only calls `self.bytes.as_ref()` to
get a `&[u8]`. In C# terms, it's a class generic over something that can hand
out a `ReadOnlySpan<byte>`. The compiler generates a specialized copy for each
`B`, so there is no virtual-call cost (monomorphization, like C# generics over
structs).

**Why not deserialize into `Vec<u32>` and friends?** Then a memory-mapped
100 MB file would be copied into the heap on open. Reading in place means
opening an index costs almost nothing, and the OS pages data in on demand.

## 5. Validate once, then index freely (`Bm25::validate`)

`from_bytes` checks every offset table, every term and every posting. After
that, `search_tokens` uses plain indexing (`u32_at(data, pos)`), which would
panic rather than corrupt memory if something were wrong. Bounds checks never
turn into undefined behavior in safe Rust; validating up front just turns
"panic during a query" into "clean error when opening".

## 6. Ownership in practice (`vectors.rs`)

```rust
pub fn add(&mut self, v: &[f32]) -> Result<u32>   // borrows self mutably and v read-only
pub fn finish(self) -> Result<Vec<u8>>           // takes self by value: the builder is consumed
```

`finish(self)` moves the builder, so using it after `finish` is a compile
error, not a runtime bug. C# can only document "don't call Add after Build";
Rust enforces it. `IndexWriter::commit(self, ...)` works the same way.

## 7. Small syntax you'll see everywhere

```rust
let Some(t) = self.find_term(term) else { continue };   // let-else: unwrap or bail out
if markdown && let Some((level, title)) = parse_heading(text) { ... }   // let-chain
.map_or(0, |t| ...)        // Option -> value with a default
.ok_or_else(|| Error::Invalid(...))?    // Option -> Result, then propagate with ?
(dim > 0).then(|| VectorBuilder::new(dim))   // bool -> Option
```

## 8. The chunker (`chunk.rs`)

This is the most "business logic" file, and it's plain structs, `Vec`s and
loops. Things to notice:

- Strings are UTF-8 and are indexed by **byte** offsets. Slicing in the middle
  of a character panics, which is why `byte_after_chars` walks `char_indices()`
  to find safe cut points. The property test generates Cyrillic and accented
  text to cover this.
- `split_inclusive('\n')` keeps the terminator, so byte offsets stay exact,
  and `\r\n` is stripped explicitly for Windows files.
- Section ids are dense integers, so `section_paths` is a `Vec` indexed by id
  instead of a `HashMap`.

## 9. Files, atomic renames and the one `unsafe` block (`index/`)

- `writer.rs` builds a generation in a temp directory, `fsync`s it, renames it
  into place, and swaps `CURRENT`. The closure `(|| { ... })()` groups the
  fallible steps so a single `if let Err(e)` can clean up the temp directory:
  a hand-rolled `try/finally`.
- `storage.rs` contains the only `unsafe` in the crate: `Mmap::map`. The
  workspace sets `unsafe_code = "deny"`, and this one function opts out with
  `#[allow(unsafe_code)]` and a `// SAFETY:` comment explaining why it's
  sound. Reviewers look for exactly this pattern.
- `#[cfg(not(target_arch = "wasm32"))]` compiles the memory map only on native
  targets, so the same crate will build for the browser in v2.

## 10. Tests and benchmarks

- **Unit tests** live next to the code in `#[cfg(test)] mod tests`.
- **Integration tests** (`tests/index.rs`) use only the public API, as a user
  would, and run against real temp directories via the `tempfile` crate.
- **Property tests** (`tests/properties.rs`, the `proptest` crate, like
  Python's Hypothesis or C#'s FsCheck) generate thousands of inputs and compare
  against naive reference implementations: textbook BM25, full sorts, and the
  chunking invariants. During development they caught a real tokenizer bug:
  `"İ".to_lowercase()` produces `i` plus a combining dot, which broke the
  "tokens are alphanumeric" rule. They also caught a char-boundary panic in one
  of the test's own helpers.
  `PROPTEST_CASES=20000 cargo test --release` runs a longer search.
- **Benchmarks** (`benches/search.rs`, `criterion`) back the numbers in the
  README. Run `cargo bench -p notelight-core`, and see `target/criterion/` for
  HTML reports.

## Exercises

1. In `vectors.rs`, the int8 dot product converts each byte with
   `i8::from_le_bytes`. Benchmark `int8_exhaustive` before and after
   accumulating in `i32` with the query pre-scaled to integers. Is it faster?
   Does `two_stage_search_with_all_candidates_equals_exhaustive` still pass?
2. Add stemming for English plural "s" to `tokenize.rs`. Which tests fail, and
   do you agree with them?
3. Change `SCHEMA_VERSION` to 2 and run the tests. What does a user see, and
   where does that message come from?
