//! Reciprocal Rank Fusion (RRF) of several ranked lists.
//!
//! BM25 scores and cosine similarities live on different scales, so adding
//! them directly needs per-corpus tuning. RRF uses only ranks:
//!
//! ```text
//! score(d) = Σ over lists containing d of 1 / (k + rank(d))      rank starts at 1
//! ```
//!
//! With the standard `k = 60`, a document ranked well by both retrievers beats
//! one ranked first by only one of them. Reference: Cormack, Clarke and
//! Buettcher, "Reciprocal Rank Fusion outperforms Condorcet and individual
//! Rank Learning Methods", SIGIR 2009.
//!
//! Equal say has a cost when one retriever has no real evidence. A query that
//! shares no word with its answer still gets 50 BM25 hits, matched on one
//! incidental word each, and their ranks push the right vector hits out. So
//! [`rrf_weighted`] lets every entry carry its own weight, and hybrid search
//! weighs each BM25 hit by [`coverage_weight`]: the share of the query's
//! distinct terms the chunk contains, squared. A chunk with every term counts
//! fully, one with 1 of 4 terms counts 1/16. See `docs/design-notes.md`,
//! "Fusion decisions".

use std::collections::{HashMap, HashSet};

use crate::topk::top_k;

/// The constant from the original RRF paper.
pub const DEFAULT_RRF_K: f32 = 60.0;

/// Default exponent of [`coverage_weight`], chosen on the notes benchmark
/// (see `docs/results.md`).
pub const DEFAULT_COVERAGE_POWER: f32 = 2.0;

/// Fuse ranked id lists (best first) and return the `limit` best `(id, score)`.
///
/// An id repeated within one list counts only at its best rank.
#[must_use]
pub fn rrf(lists: &[&[u32]], k: f32, limit: usize) -> Vec<(u32, f32)> {
    let weighted: Vec<Vec<(u32, f32)>> = lists
        .iter()
        .map(|list| list.iter().map(|&id| (id, 1.0)).collect())
        .collect();
    let refs: Vec<&[(u32, f32)]> = weighted.iter().map(Vec::as_slice).collect();
    rrf_weighted(&refs, k, limit)
}

/// [`rrf`] where each entry carries a weight: an id at rank `r` (from 1)
/// with weight `w` adds `w / (k + r)`. Weights of 1 give plain RRF.
///
/// An id repeated within one list counts only at its best rank, with that
/// entry's weight. An entry with weight 0 still holds its rank, so the ids
/// after it keep theirs.
#[must_use]
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

/// How much a keyword hit counts in fusion: `(matched / total).powf(power)`,
/// where `matched` of the query's `total` distinct terms are in the chunk.
///
/// The result is in `[0, 1]`, and 1 for a chunk with every term. A query
/// with no terms gives 1, and so does `power` 0 (plain RRF).
#[must_use]
pub fn coverage_weight(matched: usize, total: usize, power: f32) -> f32 {
    if total == 0 {
        return 1.0;
    }
    let share = matched.min(total) as f32 / total as f32;
    share.powf(power)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agreement_beats_a_single_first_place() {
        // 7 is 1st in list A only; 3 is 2nd in both lists.
        let fused = rrf(&[&[7, 3, 1], &[2, 3, 9]], DEFAULT_RRF_K, 10);
        assert_eq!(fused[0].0, 3);
    }

    #[test]
    fn single_list_keeps_its_order() {
        let fused = rrf(&[&[4, 2, 8]], DEFAULT_RRF_K, 10);
        assert_eq!(fused.iter().map(|f| f.0).collect::<Vec<_>>(), [4, 2, 8]);
    }

    #[test]
    fn duplicates_within_a_list_count_once() {
        let a = rrf(&[&[5, 5, 6]], DEFAULT_RRF_K, 10);
        let b = rrf(&[&[5, 6]], DEFAULT_RRF_K, 10);
        assert_eq!(a[0], b[0]);
    }

    #[test]
    fn weights_of_one_are_plain_rrf() {
        let a = [7, 3, 1];
        let b = [2, 3, 9];
        let ones = |l: &[u32]| l.iter().map(|&id| (id, 1.0)).collect::<Vec<_>>();
        let (wa, wb) = (ones(&a), ones(&b));
        assert_eq!(
            rrf_weighted(&[&wa, &wb], DEFAULT_RRF_K, 10),
            rrf(&[&a, &b], DEFAULT_RRF_K, 10)
        );
    }

    #[test]
    fn a_light_entry_counts_less_but_keeps_its_rank() {
        // 7 is first in A at weight 0.25; 2 is first in B at full weight.
        let a = [(7, 0.25), (3, 1.0)];
        let b = [(2, 1.0)];
        let fused = rrf_weighted(&[&a, &b], DEFAULT_RRF_K, 10);
        assert_eq!(fused.iter().map(|f| f.0).collect::<Vec<_>>(), [2, 3, 7]);
        assert!((fused[2].1 - 0.25 / 61.0).abs() < 1e-9);
        // 3 is still 2nd in A: weight 0 above it does not promote it.
        let a = [(7, 0.0), (3, 1.0)];
        let fused = rrf_weighted(&[&a], DEFAULT_RRF_K, 10);
        assert_eq!(fused[0], (3, 1.0 / 62.0));
    }

    #[test]
    fn coverage_weight_is_the_matched_share_raised_to_power() {
        assert!((coverage_weight(1, 4, 2.0) - 1.0 / 16.0).abs() < 1e-7);
        assert!((coverage_weight(1, 2, 1.0) - 0.5).abs() < 1e-7);
        assert!((coverage_weight(3, 3, 2.0) - 1.0).abs() < f32::EPSILON);
        assert!((coverage_weight(1, 5, 0.0) - 1.0).abs() < f32::EPSILON);
        assert!((coverage_weight(0, 0, 2.0) - 1.0).abs() < f32::EPSILON);
        assert!(coverage_weight(0, 3, 2.0).abs() < f32::EPSILON);
        // More matched than asked for (a caller bug) is capped at 1.
        assert!((coverage_weight(5, 3, 2.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn empty_input() {
        assert_eq!(rrf(&[], DEFAULT_RRF_K, 10), []);
        assert_eq!(rrf(&[&[]], DEFAULT_RRF_K, 10), []);
        assert_eq!(rrf_weighted(&[&[]], DEFAULT_RRF_K, 10), []);
    }
}
