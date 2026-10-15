//! Quantized vector storage and two-stage nearest-neighbour search.
//!
//! Search is exhaustive (no approximate index): at personal-folder scale a
//! full Hamming scan takes milliseconds, and exhaustive search never silently
//! misses a document the way a mis-tuned ANN index can.
//!
//! 1. **Candidates**: the ternary dot product between the query's ternary
//!    code and every stored one (two bit planes: sign and "significant"
//!    mask, see [`crate::quant::ternary_mask`]); keep the `candidates` best.
//! 2. **Rescore**: for those candidates only, compute `query · v` using the
//!    int8 codes and per-vector scale; return the best `k`.
//!
//! All vectors are L2-normalized on insert and queries on search, so scores
//! approximate cosine similarity.
//!
//! [`VectorMode`] selects other strategies, which the evaluation harness
//! compares: int8 over every vector, binary codes alone, or binary codes
//! instead of ternary ones for the candidate stage.
//!
//! File layout (little-endian, header described in `docs/index-format.md`):
//!
//! ```text
//! header   magic "NLVC", version 2, a = n, b = dim
//! binary   [u8; n * ceil(dim/8)]  sign bits, padded to 4 bytes
//! masks    [u8; n * ceil(dim/8)]  ternary masks, padded to 4 bytes
//! scales   [f32; n]
//! int8     [i8; n * dim]
//! ```

use std::fmt;
use std::str::FromStr;

use crate::bin_format::{self, HEADER_LEN, f32_at, to_u32};
use crate::error::{Error, Result};
use crate::quant::{
    binary_code_len, dot_in_lanes, hamming, l2_normalized, quantize_binary, quantize_int8,
    ternary_dot, ternary_mask,
};
use crate::topk::{smallest_k_by_histogram, top_k};

const MAGIC: [u8; 4] = *b"NLVC";
const VERSION: u16 = 2;

/// How [`Vectors::search_with`] finds and scores results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorMode {
    /// Binary candidates re-scored with int8 codes.
    BinaryInt8,
    /// Ternary candidates re-scored with int8 codes. The default, and what
    /// hybrid search uses: on BEIR it matches exhaustive int8 quality, where
    /// binary candidates lose up to 0.01 nDCG@10 (see `docs/results.md`).
    #[default]
    TernaryInt8,
    /// Rank by Hamming distance alone. Scores are `1 - 2 * distance / dim`,
    /// the fraction of agreeing signs mapped to `-1..=1`.
    Binary,
    /// Int8 scores for every vector, no candidate stage.
    Int8,
}

impl VectorMode {
    /// Every mode, in the order the evaluation reports them.
    pub const ALL: [Self; 4] = [
        Self::Int8,
        Self::Binary,
        Self::BinaryInt8,
        Self::TernaryInt8,
    ];

    /// The mode's name, as accepted by [`str::parse`].
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::BinaryInt8 => "binary+int8",
            Self::TernaryInt8 => "ternary+int8",
            Self::Binary => "binary",
            Self::Int8 => "int8",
        }
    }
}

impl fmt::Display for VectorMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for VectorMode {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|m| m.name() == s)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|m| m.name()).collect();
                Error::Invalid(format!(
                    "unknown vector mode {s:?}; expected one of {}",
                    names.join(", ")
                ))
            })
    }
}

/// Accumulates vectors and serializes them.
#[derive(Debug)]
pub struct VectorBuilder {
    dim: usize,
    binary: Vec<u8>,
    masks: Vec<u8>,
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
            masks: Vec::new(),
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
        // The mask comes from the int8 codes, not the floats, so it matches
        // what a reader could recompute from the stored codes alone.
        let as_f32: Vec<f32> = codes.iter().map(|&c| f32::from(c)).collect();
        self.masks.extend(ternary_mask(&as_f32));
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
        w.bytes(&self.masks);
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
    masks_at: usize,
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
        let plane = (n64 * code64).div_ceil(4) * 4;
        let masks_at = HEADER_LEN as u128 + plane;
        let scales_at = masks_at + plane;
        let int8_at = scales_at + 4 * n64;
        if data.len() as u128 != int8_at + n64 * dim64 {
            return Err(Error::format("vectors: size does not match header"));
        }
        let (masks_at, scales_at, int8_at) =
            (masks_at as usize, scales_at as usize, int8_at as usize);
        Ok(Self {
            bytes,
            n,
            dim,
            code_len,
            masks_at,
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

    /// Ternary mask of vector `i` (its sign plane is [`Self::code`]).
    fn mask(&self, i: usize) -> &[u8] {
        let start = self.masks_at + i * self.code_len;
        &self.bytes.as_ref()[start..start + self.code_len]
    }

    /// Raw int8 codes of vector `i`, as bytes.
    fn int8_codes(&self, i: usize) -> &[u8] {
        let start = self.int8_at + i * self.dim;
        &self.bytes.as_ref()[start..start + self.dim]
    }

    /// `query · v_i` using int8 codes and the stored scale.
    fn int8_dot(&self, query: &[f32], i: usize) -> f32 {
        let data = self.bytes.as_ref();
        let codes = self.int8_codes(i);
        // Codes are stored as raw bytes; `i8::from_le_bytes` reinterprets each
        // one without `unsafe`, and compiles to the same sign-extending load.
        let dot = dot_in_lanes(query, codes, |c| f32::from(i8::from_le_bytes([c])));
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

    /// Ternary "distances" from `query` to every stored vector: `dim` minus
    /// the ternary dot product, so `0` is the best match and every value fits
    /// the histogram selection used for Hamming distances.
    fn ternary_scan(&self, query: &[f32], dim: i64) -> Vec<u16> {
        let (q_sign, q_mask) = (quantize_binary(query), ternary_mask(query));
        (0..self.n)
            .map(|i| {
                let dot = ternary_dot(&q_sign, &q_mask, self.code(i), self.mask(i));
                // dot is in -dim..=dim, so this is in 0..=2 * dim.
                u16::try_from(dim - dot).unwrap_or(u16::MAX)
            })
            .collect()
    }

    /// Re-score `cands` with int8 codes and keep the best `k`.
    fn rescore(&self, query: &[f32], cands: Vec<u32>, k: usize) -> Vec<(u32, f32)> {
        top_k(
            cands
                .into_iter()
                .map(|i| (i, self.int8_dot(query, i as usize))),
            k,
        )
    }

    /// Two-stage search in the default [`VectorMode`]: ternary candidates,
    /// then int8 rescoring.
    ///
    /// Returns up to `k` `(id, score)` pairs, best first. `candidates` is
    /// raised to at least `k`.
    ///
    /// # Errors
    /// As [`Self::search_with`].
    pub fn search(&self, query: &[f32], k: usize, candidates: usize) -> Result<Vec<(u32, f32)>> {
        self.search_with(query, k, candidates, VectorMode::default())
    }

    /// Search with an explicit [`VectorMode`].
    ///
    /// Returns up to `k` `(id, score)` pairs, best first, ties broken by lower
    /// id. `candidates` applies to the two-stage modes and is raised to at
    /// least `k`; the other modes ignore it.
    ///
    /// # Errors
    /// Fails if `query` has the wrong dimension, is all zeros or contains
    /// non-finite values, or if a ternary search is asked of vectors with
    /// more than 32 767 dimensions.
    pub fn search_with(
        &self,
        query: &[f32],
        k: usize,
        candidates: usize,
        mode: VectorMode,
    ) -> Result<Vec<(u32, f32)>> {
        self.check_query(query)?;
        let query = l2_normalized(query);
        let candidates = candidates.max(k);
        Ok(match mode {
            VectorMode::BinaryInt8 => {
                let cands = smallest_k_by_histogram(&self.hamming_scan(&query), candidates);
                self.rescore(&query, cands, k)
            }
            VectorMode::TernaryInt8 => {
                // Distances go up to 2 * dim and must fit the u16 histogram.
                let dim = i16::try_from(self.dim).map_err(|_| {
                    Error::Invalid(format!(
                        "ternary search supports at most {} dimensions, not {}",
                        i16::MAX,
                        self.dim
                    ))
                })?;
                let dists = self.ternary_scan(&query, i64::from(dim));
                let cands = smallest_k_by_histogram(&dists, candidates);
                self.rescore(&query, cands, k)
            }
            VectorMode::Binary => {
                let dim = self.dim as f32;
                let dists = self.hamming_scan(&query);
                smallest_k_by_histogram(&dists, k)
                    .into_iter()
                    .map(|i| (i, 1.0 - 2.0 * f32::from(dists[i as usize]) / dim))
                    .collect()
            }
            VectorMode::Int8 => top_k((0..self.n).map(|i| (i as u32, self.int8_dot(&query, i))), k),
        })
    }

    /// Exact int8 search over every vector, skipping the binary stage.
    /// Used by tests and benchmarks as the reference for [`Self::search`].
    ///
    /// # Errors
    /// Fails if `query` has the wrong dimension.
    pub fn search_int8_exhaustive(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        self.search_with(query, k, 0, VectorMode::Int8)
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
    fn every_mode_finds_itself_first() {
        let vs = vecs(200, 64, 4);
        let idx = build(&vs);
        for mode in VectorMode::ALL {
            for (i, v) in vs.iter().enumerate().step_by(23) {
                let hits = idx.search_with(v, 5, 20, mode).unwrap();
                assert_eq!(hits.len(), 5, "{mode}");
                assert_eq!(hits[0].0 as usize, i, "{mode}");
                assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1), "{mode}");
            }
        }
    }

    #[test]
    fn ternary_with_all_candidates_equals_exhaustive() {
        let vs = vecs(120, 40, 5);
        let idx = build(&vs);
        let q = &vecs(1, 40, 77)[0];
        assert_eq!(
            idx.search_with(q, 7, 120, VectorMode::TernaryInt8).unwrap(),
            idx.search_with(q, 7, 0, VectorMode::Int8).unwrap()
        );
    }

    #[test]
    fn binary_mode_scores_are_sign_agreement() {
        let mut b = VectorBuilder::new(4);
        b.add(&[1.0, 1.0, 1.0, 1.0]).unwrap();
        b.add(&[1.0, 1.0, -1.0, -1.0]).unwrap();
        b.add(&[-1.0, -1.0, -1.0, -1.0]).unwrap();
        let idx = Vectors::from_bytes(b.finish().unwrap()).unwrap();
        let hits = idx
            .search_with(&[1.0, 1.0, 1.0, 1.0], 3, 0, VectorMode::Binary)
            .unwrap();
        assert_eq!(hits, vec![(0, 1.0), (1, 0.0), (2, -1.0)]);
    }

    #[test]
    fn modes_round_trip_through_their_names() {
        for mode in VectorMode::ALL {
            assert_eq!(mode.to_string().parse::<VectorMode>().unwrap(), mode);
        }
        assert_eq!(VectorMode::default(), VectorMode::TernaryInt8);
        assert!("hnsw".parse::<VectorMode>().is_err());
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
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(Vectors::from_bytes(bytes).is_err());
    }

    #[test]
    fn stores_ternary_masks_derived_from_int8_codes() {
        // 3 vectors of 12 dimensions: 2-byte codes, so each bit plane is 6
        // bytes padded to 8.
        let vs = vecs(3, 12, 6);
        let idx = build(&vs);
        assert_eq!(
            idx.bytes.len(),
            HEADER_LEN + 8 + 8 + 4 * 3 + 3 * 12,
            "header, signs, masks, scales, int8"
        );
        for (i, v) in vs.iter().enumerate() {
            let (codes, _) = quantize_int8(&l2_normalized(v));
            let as_f32: Vec<f32> = codes.iter().map(|&c| f32::from(c)).collect();
            assert_eq!(idx.mask(i), ternary_mask(&as_f32).as_slice());
        }
    }

    #[test]
    fn rejects_version_1_files() {
        let mut bytes = VectorBuilder::new(8).finish().unwrap();
        bytes[4..6].copy_from_slice(&1u16.to_le_bytes());
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
