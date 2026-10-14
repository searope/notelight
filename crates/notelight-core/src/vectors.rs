//! Quantized vector storage and two-stage nearest-neighbour search.
//!
//! Search is exhaustive (no approximate index): at personal-folder scale a
//! full Hamming scan takes milliseconds, and exhaustive search never silently
//! misses a document the way a mis-tuned ANN index can.
//!
//! 1. **Candidates**: Hamming distance between the query's binary code and
//!    every stored code; keep the `candidates` closest.
//! 2. **Rescore**: for those candidates only, compute `query · v` using the
//!    int8 codes and per-vector scale; return the best `k`.
//!
//! All vectors are L2-normalized on insert and queries on search, so scores
//! approximate cosine similarity.
//!
//! File layout (little-endian, header described in `docs/index-format.md`):
//!
//! ```text
//! header   magic "NLVC", version 1, a = n, b = dim
//! binary   [u8; n * ceil(dim/8)]  padded to 4 bytes
//! scales   [f32; n]
//! int8     [i8; n * dim]
//! ```

use crate::bin_format::{self, HEADER_LEN, f32_at, to_u32};
use crate::error::{Error, Result};
use crate::quant::{binary_code_len, hamming, l2_normalized, quantize_binary, quantize_int8};
use crate::topk::{smallest_k_by_histogram, top_k};

const MAGIC: [u8; 4] = *b"NLVC";
const VERSION: u16 = 1;

/// Accumulates vectors and serializes them.
#[derive(Debug)]
pub struct VectorBuilder {
    dim: usize,
    binary: Vec<u8>,
    scales: Vec<f32>,
    int8: Vec<i8>,
}

impl VectorBuilder {
    /// Create a builder for `dim`-dimensional vectors.
    ///
    /// # Panics
    /// Panics if `dim` is zero.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "dimension must be positive");
        Self {
            dim,
            binary: Vec::new(),
            scales: Vec::new(),
            int8: Vec::new(),
        }
    }

    /// Normalize, quantize and append `v`. Returns its id.
    ///
    /// # Errors
    /// Fails if `v` has the wrong dimension or contains non-finite values.
    pub fn add(&mut self, v: &[f32]) -> Result<u32> {
        if v.len() != self.dim {
            return Err(Error::Dimension {
                expected: self.dim,
                found: v.len(),
            });
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(Error::Invalid("vector contains NaN or infinity".into()));
        }
        let id = to_u32(self.scales.len(), "vector count")?;
        let v = l2_normalized(v);
        self.binary.extend(quantize_binary(&v));
        let (codes, scale) = quantize_int8(&v);
        self.int8.extend(codes);
        self.scales.push(scale);
        Ok(id)
    }

    /// Number of vectors added.
    #[must_use]
    pub fn len(&self) -> usize {
        self.scales.len()
    }

    /// Whether no vectors have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scales.is_empty()
    }

    /// Serialize the vectors.
    ///
    /// # Errors
    /// Fails if counts exceed the format's 32-bit limits.
    pub fn finish(self) -> Result<Vec<u8>> {
        let n = to_u32(self.scales.len(), "vector count")?;
        let dim = to_u32(self.dim, "dimension")?;
        let mut w = bin_format::Writer::with_header(MAGIC, VERSION, n, dim);
        w.bytes(&self.binary);
        w.pad4();
        for &s in &self.scales {
            w.f32(s);
        }
        w.bytes(
            &self
                .int8
                .iter()
                .map(|&c| c.to_le_bytes()[0])
                .collect::<Vec<u8>>(),
        );
        Ok(w.into_inner())
    }
}

/// Read-only quantized vectors backed by their serialized bytes.
#[derive(Debug)]
pub struct Vectors<B> {
    bytes: B,
    n: usize,
    dim: usize,
    code_len: usize,
    scales_at: usize,
    int8_at: usize,
}

impl<B: AsRef<[u8]>> Vectors<B> {
    /// Open vectors from their bytes, validating the layout.
    ///
    /// # Errors
    /// Fails if the bytes are not a well-formed vector file.
    pub fn from_bytes(bytes: B) -> Result<Self> {
        let data = bytes.as_ref();
        let header = bin_format::read_header(data, MAGIC, VERSION, "vectors")?;
        let (n, dim) = (header.a as usize, header.b as usize);
        if dim == 0 {
            return Err(Error::format("vectors: dimension is zero"));
        }
        let code_len = binary_code_len(dim);
        // Layout math in u128: with n and dim both up to u32::MAX, the total
        // exceeds even u64, and `usize` is only 32 bits on WebAssembly. Once
        // the total matches the real length, every offset fits in `usize`.
        let (n64, dim64, code64) = (n as u128, dim as u128, code_len as u128);
        let scales_at = HEADER_LEN as u128 + (n64 * code64).div_ceil(4) * 4;
        let int8_at = scales_at + 4 * n64;
        if data.len() as u128 != int8_at + n64 * dim64 {
            return Err(Error::format("vectors: size does not match header"));
        }
        let (scales_at, int8_at) = (scales_at as usize, int8_at as usize);
        Ok(Self {
            bytes,
            n,
            dim,
            code_len,
            scales_at,
            int8_at,
        })
    }

    /// Number of stored vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether no vectors are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Vector dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn code(&self, i: usize) -> &[u8] {
        let start = HEADER_LEN + i * self.code_len;
        &self.bytes.as_ref()[start..start + self.code_len]
    }

    /// `query · v_i` using int8 codes and the stored scale.
    fn int8_dot(&self, query: &[f32], i: usize) -> f32 {
        let data = self.bytes.as_ref();
        let start = self.int8_at + i * self.dim;
        let codes = &data[start..start + self.dim];
        // Codes are stored as raw bytes; `i8::from_le_bytes` reinterprets each
        // one without `unsafe`, and compiles to the same sign-extending load.
        let dot: f32 = query
            .iter()
            .zip(codes)
            .map(|(&q, &c)| q * f32::from(i8::from_le_bytes([c])))
            .sum();
        dot * f32_at(data, self.scales_at + 4 * i)
    }

    /// Approximate cosine similarity between `query` and vector `i`.
    ///
    /// # Errors
    /// Fails if `query` has the wrong dimension or `i` is out of range.
    pub fn score(&self, query: &[f32], i: u32) -> Result<f32> {
        self.check_query(query)?;
        if i as usize >= self.n {
            return Err(Error::Invalid(format!("vector id {i} out of range")));
        }
        Ok(self.int8_dot(&l2_normalized(query), i as usize))
    }

    fn check_query(&self, query: &[f32]) -> Result<()> {
        if query.len() != self.dim {
            return Err(Error::Dimension {
                expected: self.dim,
                found: query.len(),
            });
        }
        if query.iter().any(|x| !x.is_finite()) {
            return Err(Error::Invalid(
                "query vector contains NaN or infinity".into(),
            ));
        }
        if query.iter().all(|&x| x == 0.0) {
            return Err(Error::Invalid("query vector is all zeros".into()));
        }
        Ok(())
    }

    /// Hamming distances from `query`'s binary code to every stored code.
    fn hamming_scan(&self, query: &[f32]) -> Vec<u16> {
        let q_code = quantize_binary(query);
        (0..self.n)
            .map(|i| hamming(&q_code, self.code(i)) as u16)
            .collect()
    }

    /// Two-stage search: binary candidates, then int8 rescoring.
    ///
    /// Returns up to `k` `(id, score)` pairs, best first. `candidates` is
    /// raised to at least `k`.
    ///
    /// # Errors
    /// Fails if `query` has the wrong dimension.
    pub fn search(&self, query: &[f32], k: usize, candidates: usize) -> Result<Vec<(u32, f32)>> {
        self.check_query(query)?;
        let query = l2_normalized(query);
        let dists = self.hamming_scan(&query);
        let cands = smallest_k_by_histogram(&dists, candidates.max(k));
        Ok(top_k(
            cands
                .into_iter()
                .map(|i| (i, self.int8_dot(&query, i as usize))),
            k,
        ))
    }

    /// Exact int8 search over every vector, skipping the binary stage.
    /// Used by tests and benchmarks as the reference for [`Self::search`].
    ///
    /// # Errors
    /// Fails if `query` has the wrong dimension.
    pub fn search_int8_exhaustive(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        self.check_query(query)?;
        let query = l2_normalized(query);
        Ok(top_k(
            (0..self.n).map(|i| (i as u32, self.int8_dot(&query, i))),
            k,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random vectors (no RNG dependency).
    fn vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                    })
                    .collect()
            })
            .collect()
    }

    fn build(vs: &[Vec<f32>]) -> Vectors<Vec<u8>> {
        let mut b = VectorBuilder::new(vs[0].len());
        for v in vs {
            b.add(v).unwrap();
        }
        Vectors::from_bytes(b.finish().unwrap()).unwrap()
    }

    #[test]
    fn finds_itself_first() {
        let vs = vecs(200, 64, 1);
        let idx = build(&vs);
        for (i, v) in vs.iter().enumerate().step_by(17) {
            let hits = idx.search(v, 3, 20).unwrap();
            assert_eq!(hits[0].0 as usize, i);
            assert!(
                (hits[0].1 - 1.0).abs() < 0.02,
                "self-similarity {}",
                hits[0].1
            );
        }
    }

    #[test]
    fn two_stage_matches_exhaustive_with_enough_candidates() {
        let vs = vecs(300, 32, 2);
        let idx = build(&vs);
        let q = &vecs(1, 32, 99)[0];
        assert_eq!(
            idx.search(q, 5, 300).unwrap(),
            idx.search_int8_exhaustive(q, 5).unwrap()
        );
    }

    #[test]
    fn scores_approximate_float_cosine() {
        let vs = vecs(50, 48, 3);
        let idx = build(&vs);
        let q = &vecs(1, 48, 7)[0];
        let (qn, qd) = (l2_normalized(q), vs.iter().map(|v| l2_normalized(v)));
        for (i, v) in qd.enumerate() {
            let exact: f32 = qn.iter().zip(&v).map(|(a, b)| a * b).sum();
            let approx = idx.score(q, i as u32).unwrap();
            assert!((exact - approx).abs() < 0.02, "{exact} vs {approx}");
        }
    }

    #[test]
    fn rejects_bad_input() {
        let mut b = VectorBuilder::new(4);
        assert!(matches!(b.add(&[1.0; 3]), Err(Error::Dimension { .. })));
        assert!(b.add(&[f32::NAN, 0.0, 0.0, 0.0]).is_err());
        b.add(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        let idx = Vectors::from_bytes(b.finish().unwrap()).unwrap();
        assert!(idx.search(&[1.0; 5], 1, 1).is_err());
        assert!(idx.score(&[1.0; 4], 9).is_err());
        assert!(idx.search(&[f32::NAN; 4], 1, 1).is_err());
        assert!(idx.search(&[0.0; 4], 1, 1).is_err());
    }

    #[test]
    fn rejects_overflowing_headers() {
        let mut bytes = b"NLVC".to_vec();
        bytes.extend_from_slice(&[1, 0, 0, 0]);
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(Vectors::from_bytes(bytes).is_err());
    }

    #[test]
    fn rejects_truncated_file() {
        let mut b = VectorBuilder::new(8);
        b.add(&[1.0; 8]).unwrap();
        let bytes = b.finish().unwrap();
        assert!(Vectors::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    }
}
