# Milestone 0: Rust walkthrough

A reading guide to the Rust in milestone 0, aimed at someone who knows C# and
Python well and Rust somewhat. About 15 minutes.

## 1. The workspace (`Cargo.toml`)

A Cargo **workspace** is like a .NET solution: several crates (projects) sharing
one `target/` build folder and one lockfile.

- `[workspace.package]` holds shared metadata. Member crates opt in with
  `version.workspace = true`, much like `Directory.Build.props`.
- `[workspace.lints]` turns on `clippy::pedantic` (a strict analyzer, like
  enabling most Roslyn analyzers) and denies `unsafe` code in our crates.
- `edition = "2024"` is the language edition, similar to `<LangVersion>`.

## 2. The core crate (`crates/notelight-core/src/quant.rs`)

```rust
pub fn quantize_binary(v: &[f32]) -> Vec<u8>
```

- `&[f32]` is a **slice**: a borrowed view of contiguous floats (pointer + length).
  Think `ReadOnlySpan<float>`. The function reads the data without owning it.
- `Vec<u8>` is an owned, growable buffer, like `List<byte>`. The caller owns the
  result.

```rust
out[i / 8] |= 1 << (i % 8);
```

Sets bit `i % 8` of byte `i / 8`. Indexing is bounds-checked: an out-of-range
index panics instead of corrupting memory.

```rust
a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
```

An iterator chain, the equivalent of LINQ's `Zip(...).Select(...).Sum()`.
`count_ones()` compiles to a single CPU popcount instruction. Iterator chains
in Rust are zero-cost: they compile to the same loop you would write by hand.

`#[must_use]` warns if a caller ignores the return value. `#[cfg(test)]` marks a
module compiled only for `cargo test`; unit tests live next to the code.

## 3. The binding crate (`crates/notelight-py/src/lib.rs`)

```toml
[lib]
name = "_core"
crate-type = ["cdylib"]
```

`cdylib` builds a C-compatible shared library (`.pyd` on Windows, `.so` on
Linux), which Python imports as `notelight._core`.

```rust
#[pyfunction]
fn quantize_binary<'py>(py: Python<'py>, vectors: PyReadonlyArray2<'py, f32>)
    -> PyResult<Bound<'py, PyArray2<u8>>>
```

- `'py` is a **lifetime**: a compile-time proof that these values are only used
  while we hold Python's GIL. You will rarely write lifetimes in the core; they
  show up mostly at this boundary.
- `PyReadonlyArray2<f32>` borrows a NumPy array without copying. The `numpy`
  crate tracks the borrow at runtime, so other Rust code can't take a mutable
  borrow of the same array while we read it.
- `PyResult<T>` is `Result<T, PyErr>`. Returning `Err` raises a Python exception.
  Rust has no exceptions; errors are values.

```rust
let contiguous = view.as_standard_layout();
```

Returns a `Cow` ("clone on write"): a borrow if the array is already contiguous,
a copy only if the caller passed a strided view such as `arr[:, ::2]`.

```rust
py.detach(|| data.chunks_exact(dim).flat_map(quant::quantize_binary).collect())
```

`py.detach` releases the GIL while the closure runs, so other Python threads
keep working. The compiler checks that the closure touches no Python objects;
using one there would be a compile error, not a runtime crash.

`?` after an expression means "if this is an error, return it now". It replaces
most `try/catch` and `if (err != nil)` boilerplate.

## 4. How Python sees it

`python/notelight/_core.pyi` is a type stub, so mypy and editors know the
signatures of the compiled functions. `maturin develop` compiles the crate and
installs the package in editable mode: Python edits apply immediately, and Rust
edits need a rebuild.

## Try it

```bash
cargo test -p notelight-core   # Rust unit tests
pytest                         # checks Rust output against a NumPy reference
```

Exercise: change `x > 0.0` to `x >= 0.0` in `quant.rs`, then run both test
suites and see which tests catch it.
