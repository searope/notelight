# Making `int8_dot` vectorize: partial sums and `as_chunks`

A short follow-up to the [milestone 3 walkthrough](m3-rust-walkthrough.md).
Milestone 3's measurements showed the exhaustive int8 scan running 5x slower
in Rust than a NumPy (BLAS) matrix-vector product. The fix is ten lines in
`quant.rs` (`dot_in_lanes`), no `unsafe` and no new dependency, and it makes
the scan about 4x faster. It takes about 15 minutes to read.

## 1. Why the simple loop was slow

```rust
query.iter().zip(codes).map(|(&q, &c)| q * f32::from(c)).sum()
```

This is one long chain: every `+` needs the result of the previous one. A
modern core can start two float additions per cycle, but each takes about four
cycles to finish, so a single chain keeps the adder about one eighth busy.
And it runs one element at a time, while an SSE register holds four `f32`s.

The compiler sees this and still may not fix it. Turning the chain into four
SIMD lanes would add the numbers in a different order, and float addition is
not associative: `(a + b) + c` can differ from `a + (b + c)` in the last bit.
Rust (like C# and C without `-ffast-math`) promises the result you wrote, so
LLVM keeps the order and the loop stays scalar. Integer addition is
associative, so integer sums don't have this problem.

NumPy is not bound by that promise in the same way: `np.dot` calls a BLAS
kernel that picks its own order (and threads), and `np.sum` uses pairwise
summation. Different order, slightly different bits, much faster.

## 2. The fix: choose the order yourself

If a different order is fine, write it down, and the compiler is free to use
SIMD because the source now says what SIMD would do:

```rust
const LANES: usize = 16;

pub(crate) fn dot_in_lanes<C: Copy>(query: &[f32], codes: &[C], to_f32: impl Fn(C) -> f32) -> f32 {
    let (q_chunks, q_tail) = query.as_chunks::<LANES>();
    let (c_chunks, c_tail) = codes.as_chunks::<LANES>();
    let mut acc = [0.0_f32; LANES];
    for (q, c) in q_chunks.iter().zip(c_chunks) {
        for ((a, &x), &y) in acc.iter_mut().zip(q).zip(c) {
            *a += x * to_f32(y);
        }
    }
    let tail: f32 = q_tail.iter().zip(c_tail).map(|(&x, &y)| x * to_f32(y)).sum();
    acc.iter().sum::<f32>() + tail
}
```

- **`as_chunks::<16>()`** (stable since Rust 1.88, this crate's MSRV) splits a
  slice into `&[[f32; 16]]` plus a remainder shorter than 16. The chunk type is
  a fixed-size array, so the inner loop has a length the compiler knows, and it
  unrolls it completely with no bounds checks. In C# terms it's like
  `MemoryMarshal.Cast<float, Vector512<float>>` without the `unsafe`
  reinterpretation: the slice is not copied, only viewed differently. In
  Python terms it's `x[: n - n % 16].reshape(-1, 16)` plus `x[n - n % 16:]`.
- **`acc: [f32; 16]`** holds sixteen independent partial sums. Lane `j` only
  ever adds elements `j`, `j + 16`, `j + 32`, ..., so the sixteen chains don't
  wait on each other. On baseline x86-64 (SSE2) they live in four 4-wide
  registers, which keeps the adder busy.
- **`to_f32: impl Fn(C) -> f32`** lets one loop serve both callers: the public
  `dot_f32_i8` reads `&[i8]` (`f32::from`), and `Vectors::int8_dot` reads the
  raw bytes of a memory-mapped index (`f32::from(i8::from_le_bytes([c]))`).
  Each call site gets its own monomorphized copy, like a C# generic method
  over a struct, so the closure costs nothing after inlining.

The result is still **deterministic**: the order of additions depends only on
the length, never on the data or the CPU. It just isn't the left-to-right
order, so scores can differ from the old loop in the last bits. Tests that
compare two-stage search with exhaustive search still pass exactly, because
both call the same function.

## 3. How many lanes

Measured with `cargo bench -p notelight-core --bench search` (100k × 384
vectors, one thread, i9-10900K, default target):

| `LANES` | exhaustive int8 scan | binary candidates + rescore (10k) |
|---:|---:|---:|
| scalar loop (before) | 32.3 ms | 161 µs |
| 8 | 9.8 ms | 114 µs |
| **16** | **9.3 ms** | **98 µs** |
| 32 | 9.6 ms | 165 µs |

Eight lanes are only two SSE registers, not enough independent chains. At 32,
sixteen SSE registers are not enough for the accumulators plus the converted
codes, so the compiler spills to the stack, which shows up most in the short
rescoring loop. Sixteen is the sweet spot, and it is also a whole number of
AVX-512 or AVX2 registers if the target allows them.

That last point matters: compiled with `RUSTFLAGS="-C target-cpu=native"`
(AVX2 here, with a single-instruction `i8` → `i32` widening) the scan drops
again, from 9.3 to 5.0 ms. notelight doesn't do that, because a wheel built
for one CPU can crash on an older one. Runtime dispatch
(`is_x86_feature_detected!` + `#[target_feature]`) would get it safely, at
the cost of `unsafe` calls, so it stays a possible follow-up.

## 4. Testing a function whose answer changed

The old exact-equality tests no longer apply to `dot_f32_i8` alone, since the
new order rounds differently. `tests/properties.rs` instead checks it against
a sequential sum in `f64` with the textbook error bound for summing `n` float
terms in any order:

```rust
|got - reference| <= n * f32::EPSILON * sum(|q_i * c_i|)
```

That bound holds for every order, so the test doesn't encode which order the
code picked, and it would catch a real bug (a dropped tail element, a lane
counted twice) on the first case where it matters.

## Exercises

1. Change `LANES` to 4 and run the vector benches. Which of the two effects in
   section 1 is still unsolved?
2. Replace the inner `for` loop with `acc = std::array::from_fn(|j| acc[j] + q[j] * to_f32(c[j]))`.
   Does the benchmark change? Is the generated code different
   (`cargo asm` or Compiler Explorer)?
3. In the proptest, drop the `tail` term from `dot_in_lanes` and run with
   `PROPTEST_CASES=20000`. How short is the shrunk failing input, and why?
