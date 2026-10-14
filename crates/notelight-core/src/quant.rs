//! Vector quantization primitives.
//!
//! Two encodings are stored for every embedding:
//!
//! - **Binary**: one bit per dimension (the sign). Comparing two codes is a
//!   Hamming distance (XOR + popcount), which is cheap enough to scan every
//!   vector for each query. It is used to pick candidates.
//! - **Int8**: each dimension scaled to `-127..=127` with one `f32` scale per
//!   vector. It is about 4x smaller than `f32` and keeps ~99% of the ranking
//!   quality. It is used to re-score the binary candidates.

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
    let mut words_a = a.chunks_exact(8);
    let mut words_b = b.chunks_exact(8);
    let mut dist: u32 = words_a
        .by_ref()
        .zip(words_b.by_ref())
        .map(|(x, y)| (word(x) ^ word(y)).count_ones())
        .sum();
    for (x, y) in words_a.remainder().iter().zip(words_b.remainder()) {
        dist += (x ^ y).count_ones();
    }
    dist
}

#[inline]
fn word(bytes: &[u8]) -> u64 {
    // `chunks_exact(8)` guarantees the length, so this conversion cannot fail.
    u64::from_le_bytes(bytes.try_into().expect("chunk of 8 bytes"))
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
    query
        .iter()
        .zip(codes)
        .map(|(&q, &c)| q * f32::from(c))
        .sum()
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
    fn normalizes_to_unit_length() {
        let n = l2_normalized(&[3.0, 4.0]);
        assert_eq!(n, vec![0.6, 0.8]);
        assert_eq!(l2_normalized(&[0.0, 0.0]), vec![0.0, 0.0]);
    }
}
