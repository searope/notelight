//! Vector quantization primitives.
//!
//! Three encodings are stored for every embedding:
//!
//! - **Binary**: one bit per dimension (the sign). Comparing two codes is a
//!   Hamming distance (XOR + popcount), which is cheap enough to scan every
//!   vector for each query.
//! - **Ternary** (`-1`, `0` or `+1` per dimension): the binary code plus a
//!   mask of "significant" dimensions (see [`ternary_mask`]). Comparing two
//!   codes is [`ternary_dot`], two popcounts. It is used to pick candidates,
//!   and picks better ones than binary codes alone.
//! - **Int8**: each dimension scaled to `-127..=127` with one `f32` scale per
//!   vector. It is about 4x smaller than `f32` and keeps ~99% of the ranking
//!   quality. It is used to re-score the candidates.

/// Number of bytes needed to store a binary code of `dim` dimensions.
#[must_use]
pub const fn binary_code_len(dim: usize) -> usize {
    dim.div_ceil(8)
}

/// Quantize one vector to a binary code: bit `i` is set when `v[i] > 0`.
///
/// Bits are packed little-endian within each byte (dimension 0 is bit 0 of
/// byte 0). The last byte is zero-padded when `v.len()` is not a multiple of 8.
#[must_use]
pub fn quantize_binary(v: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; binary_code_len(v.len())];
    for (i, &x) in v.iter().enumerate() {
        if x > 0.0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// Hamming distance between two binary codes of equal length.
///
/// Processes eight bytes at a time as `u64` words, so the compiler emits one
/// `popcnt` per word instead of one per byte.
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
#[must_use]
pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "binary codes must have equal length");
    let (words_a, rest_a) = a.as_chunks::<8>();
    let (words_b, rest_b) = b.as_chunks::<8>();
    let mut dist: u32 = words_a
        .iter()
        .zip(words_b)
        .map(|(x, y)| (u64::from_le_bytes(*x) ^ u64::from_le_bytes(*y)).count_ones())
        .sum();
    for (x, y) in rest_a.iter().zip(rest_b) {
        dist += (x ^ y).count_ones();
    }
    dist
}

/// Quantize one vector to int8 with a single symmetric scale.
///
/// Returns `(codes, scale)` such that `v[i] ≈ codes[i] as f32 * scale`, with
/// an absolute error of at most `scale / 2` per dimension. An all-zero vector
/// gets scale `0.0`.
#[must_use]
pub fn quantize_int8(v: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = v.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
    if max_abs == 0.0 || !max_abs.is_finite() {
        return (vec![0; v.len()], 0.0);
    }
    let scale = max_abs / 127.0;
    let codes = v
        .iter()
        .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (codes, scale)
}

/// Dot product of a float query with int8 codes, *without* the vector's scale.
///
/// Multiply the result by the stored scale to approximate `query · v`.
/// Keeping the query in `f32` ("asymmetric" scoring) is more accurate than
/// quantizing it too, and costs nothing at this scale.
///
/// # Panics
/// Panics if the lengths differ.
#[must_use]
pub fn dot_f32_i8(query: &[f32], codes: &[i8]) -> f32 {
    assert_eq!(query.len(), codes.len(), "dimension mismatch");
    dot_in_lanes(query, codes, f32::from)
}

/// Number of independent partial sums in [`dot_in_lanes`].
const LANES: usize = 16;

/// `sum(query[i] * to_f32(codes[i]))`, written so the compiler can vectorize
/// it.
///
/// A plain `.sum()` has one accumulator, and every addition waits for the
/// previous one. The compiler may not split it into SIMD lanes on its own,
/// because float addition is not associative and that would change the
/// result. Here the split is explicit: `LANES` partial sums, one per position
/// in each `LANES`-wide chunk, added together at the end. Each chunk's
/// multiply-adds are independent of each other, so they become a few SIMD
/// instructions.
///
/// The result can differ from a sequential sum in the last bits, but it is
/// deterministic: the order of additions depends only on the length.
/// `to_f32` lets the same loop read `i8` codes or the raw bytes of a stored
/// index. Callers pass slices of the same length.
///
/// 16 lanes measured best on x86-64 (see `docs/learning/m3-rust-walkthrough.md`):
/// 8 leaves the SIMD units waiting on each other, 32 runs out of registers.
#[inline]
pub(crate) fn dot_in_lanes<C: Copy>(query: &[f32], codes: &[C], to_f32: impl Fn(C) -> f32) -> f32 {
    debug_assert_eq!(query.len(), codes.len());
    let (q_chunks, q_tail) = query.as_chunks::<LANES>();
    let (c_chunks, c_tail) = codes.as_chunks::<LANES>();
    let mut acc = [0.0_f32; LANES];
    for (q, c) in q_chunks.iter().zip(c_chunks) {
        for ((a, &x), &y) in acc.iter_mut().zip(q).zip(c) {
            *a += x * to_f32(y);
        }
    }
    let tail: f32 = q_tail
        .iter()
        .zip(c_tail)
        .map(|(&x, &y)| x * to_f32(y))
        .sum();
    acc.iter().sum::<f32>() + tail
}

/// Fraction of a vector's mean absolute value at or below which a dimension
/// counts as zero in its ternary code.
///
/// `0.7` is the threshold of Ternary Weight Networks (Li et al., 2016), the
/// usual choice for roughly Gaussian values: about 40% of dimensions become
/// zero, and the rest keep only their sign.
pub const TERNARY_THRESHOLD: f32 = 0.7;

/// The mask half of a ternary code: bit `i` is set when
/// `|v[i]| > TERNARY_THRESHOLD * mean(|v|)`.
///
/// Combined with the binary code (the sign), it gives each dimension a value
/// in `{-1, 0, +1}`. The threshold is relative to the vector's own mean, so
/// scaling `v` (for example, int8 codes instead of floats) gives the same
/// mask. Bits are packed like [`quantize_binary`]; an all-zero or non-finite
/// vector gets an empty mask.
#[must_use]
pub fn ternary_mask(v: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; binary_code_len(v.len())];
    if v.is_empty() {
        return out;
    }
    let mean_abs = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
    if !mean_abs.is_finite() {
        return out;
    }
    let threshold = TERNARY_THRESHOLD * mean_abs;
    for (i, &x) in v.iter().enumerate() {
        if x.abs() > threshold {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// Dot product of two ternary codes, each given as `(sign, mask)` bit planes.
///
/// A dimension contributes `+1` when it is significant in both vectors and the
/// signs agree, `-1` when they disagree, and `0` otherwise. With
/// `both = mask_a & mask_b`, that is
/// `popcount(both) - 2 * popcount(both & (sign_a ^ sign_b))`, eight bytes at a
/// time like [`hamming`].
///
/// # Panics
/// Panics if the four slices do not all have the same length.
#[must_use]
pub fn ternary_dot(sign_a: &[u8], mask_a: &[u8], sign_b: &[u8], mask_b: &[u8]) -> i64 {
    let len = sign_a.len();
    assert!(
        mask_a.len() == len && sign_b.len() == len && mask_b.len() == len,
        "ternary codes must have equal length"
    );
    let (mut both, mut differ) = (0u32, 0u32);
    // One step over 64 dimensions (or 8 for the tail), as integers.
    let mut tally = |sa: u64, ma: u64, sb: u64, mb: u64| {
        let m = ma & mb;
        both += m.count_ones();
        differ += (m & (sa ^ sb)).count_ones();
    };
    // Each is (whole 8-byte words, leftover bytes).
    let [sa, ma, sb, mb] = [sign_a, mask_a, sign_b, mask_b].map(<[u8]>::as_chunks::<8>);
    let w = u64::from_le_bytes;
    for (((s1, m1), s2), m2) in sa.0.iter().zip(ma.0).zip(sb.0).zip(mb.0) {
        tally(w(*s1), w(*m1), w(*s2), w(*m2));
    }
    for (((&s1, &m1), &s2), &m2) in sa.1.iter().zip(ma.1).zip(sb.1).zip(mb.1) {
        tally(s1.into(), m1.into(), s2.into(), m2.into());
    }
    i64::from(both) - 2 * i64::from(differ)
}

/// Return `v` scaled to unit L2 norm, or unchanged if its norm is zero.
#[must_use]
pub fn l2_normalized(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_signs_little_endian() {
        // dims 0 and 9 positive -> bit 0 of byte 0, bit 1 of byte 1
        let mut v = vec![-1.0_f32; 10];
        v[0] = 0.5;
        v[9] = 2.0;
        assert_eq!(quantize_binary(&v), vec![0b0000_0001, 0b0000_0010]);
    }

    #[test]
    fn zero_is_not_positive() {
        assert_eq!(quantize_binary(&[0.0; 8]), vec![0]);
    }

    #[test]
    fn hamming_counts_differing_bits() {
        assert_eq!(hamming(&[0b1010_1010], &[0b0101_0101]), 8);
        assert_eq!(hamming(&[0xFF, 0x00], &[0xFF, 0x00]), 0);
        // 8-byte word path plus a 3-byte remainder
        let a = [0xFFu8; 11];
        let b = [0x00u8; 11];
        assert_eq!(hamming(&a, &b), 88);
    }

    #[test]
    fn int8_round_trips_within_half_a_step() {
        let v = [0.5_f32, -1.0, 0.25, 0.0, 0.999];
        let (codes, scale) = quantize_int8(&v);
        assert_eq!(codes[1], -127);
        for (x, c) in v.iter().zip(&codes) {
            assert!((x - f32::from(*c) * scale).abs() <= scale / 2.0 + f32::EPSILON);
        }
    }

    #[test]
    fn int8_of_zero_vector_is_zero() {
        assert_eq!(quantize_int8(&[0.0; 4]), (vec![0; 4], 0.0));
    }

    #[test]
    fn asymmetric_dot_approximates_float_dot() {
        let q = [0.6_f32, 0.8, 0.0];
        let v = [0.8_f32, 0.6, 0.0];
        let (codes, scale) = quantize_int8(&v);
        let approx = dot_f32_i8(&q, &codes) * scale;
        assert!((approx - 0.96).abs() < 0.01, "{approx}");
    }

    #[test]
    fn ternary_mask_keeps_large_dimensions() {
        // mean |v| = 3.1 / 9 ≈ 0.344, threshold ≈ 0.241: dims 0, 2 and 8 count.
        let v = [1.0_f32, 0.1, -0.9, 0.0, 0.2, -0.2, 0.1, 0.1, 0.5];
        assert_eq!(ternary_mask(&v), vec![0b0000_0101, 0b0000_0001]);
        // Scale-invariant, so int8 codes give the same mask as floats.
        let scaled: Vec<f32> = v.iter().map(|x| x * 127.0).collect();
        assert_eq!(ternary_mask(&scaled), ternary_mask(&v));
        assert_eq!(ternary_mask(&[0.0; 4]), vec![0]);
        assert_eq!(ternary_mask(&[]), Vec::<u8>::new());
    }

    #[test]
    fn ternary_dot_counts_agreements_minus_disagreements() {
        // a = [+1, -1, 0, +1], b = [+1, +1, +1, 0] -> 1 - 1 + 0 + 0 = 0
        let a = [1.0_f32, -1.0, 0.0, 1.0];
        let b = [1.0_f32, 1.0, 1.0, 0.0];
        let (sa, ma) = (quantize_binary(&a), ternary_mask(&a));
        let (sb, mb) = (quantize_binary(&b), ternary_mask(&b));
        assert_eq!(ternary_dot(&sa, &ma, &sb, &mb), 0);
        // A vector with itself: one per significant dimension.
        assert_eq!(ternary_dot(&sa, &ma, &sa, &ma), 3);
        // 8-byte word path plus a 3-byte remainder.
        let ones = [0xFFu8; 11];
        let zeros = [0u8; 11];
        assert_eq!(ternary_dot(&ones, &ones, &ones, &ones), 88);
        assert_eq!(ternary_dot(&ones, &ones, &zeros, &ones), -88);
    }

    #[test]
    fn normalizes_to_unit_length() {
        let n = l2_normalized(&[3.0, 4.0]);
        assert_eq!(n, vec![0.6, 0.8]);
        assert_eq!(l2_normalized(&[0.0, 0.0]), vec![0.0, 0.0]);
    }
}
