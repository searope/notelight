//! Vector quantization primitives.
//!
//! Binary quantization keeps one bit per dimension (the sign). Similarity
//! between two binary codes is measured with Hamming distance: the number of
//! differing bits, computed with XOR + popcount.

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
/// # Panics
/// Panics if `a` and `b` have different lengths.
#[must_use]
pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "binary codes must have equal length");
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
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
    }
}
