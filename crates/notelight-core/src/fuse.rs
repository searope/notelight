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

use std::collections::HashMap;

use crate::topk::top_k;

/// The constant from the original RRF paper.
pub const DEFAULT_RRF_K: f32 = 60.0;

/// Fuse ranked id lists (best first) and return the `limit` best `(id, score)`.
///
/// An id repeated within one list counts only at its best rank.
#[must_use]
pub fn rrf(lists: &[&[u32]], k: f32, limit: usize) -> Vec<(u32, f32)> {
    let mut scores: HashMap<u32, f32> = HashMap::new();
    for list in lists {
        let mut seen = std::collections::HashSet::new();
        for (rank, &id) in list.iter().enumerate() {
            if seen.insert(id) {
                *scores.entry(id).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
            }
        }
    }
    top_k(scores, limit)
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
    fn empty_input() {
        assert!(rrf(&[], DEFAULT_RRF_K, 10).is_empty());
        assert!(rrf(&[&[]], DEFAULT_RRF_K, 10).is_empty());
    }
}
