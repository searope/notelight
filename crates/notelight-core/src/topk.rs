//! Top-k selection helpers shared by BM25 and vector search.
//!
//! Results are always ordered by score descending, then by id ascending, so
//! rankings are deterministic even when scores tie.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// A scored item. `Ord` puts the *best* item last: higher score, then lower id.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Scored {
    score: f32,
    id: u32,
}

impl Eq for Scored {}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Keep the `k` best `(id, score)` pairs and return them best first.
///
/// Runs in `O(n log k)` time and `O(k)` memory using a min-heap of size `k`.
/// `NaN` scores are ranked below every number.
pub fn top_k(items: impl IntoIterator<Item = (u32, f32)>, k: usize) -> Vec<(u32, f32)> {
    if k == 0 {
        return Vec::new();
    }
    let items = items.into_iter();
    // Never preallocate from `k` alone: callers may pass a huge k to mean "all".
    let capacity = k.min(items.size_hint().0).saturating_add(1);
    let mut heap: BinaryHeap<Reverse<Scored>> = BinaryHeap::with_capacity(capacity);
    for (id, score) in items {
        let score = if score.is_nan() {
            f32::NEG_INFINITY
        } else {
            score
        };
        let item = Scored { score, id };
        if heap.len() < k {
            heap.push(Reverse(item));
        } else if heap.peek().is_some_and(|worst| item > worst.0) {
            heap.pop();
            heap.push(Reverse(item));
        }
    }
    let mut out: Vec<Scored> = heap.into_iter().map(|r| r.0).collect();
    out.sort_unstable_by(|a, b| b.cmp(a));
    out.into_iter().map(|s| (s.id, s.score)).collect()
}

/// Indices of the `k` smallest values in `dists`, ties broken by lower index.
///
/// Hamming distances are small integers (`0..=dim`), so instead of sorting we
/// build a histogram, find the cut-off distance, and collect in one more pass:
/// `O(n + max_dist)` time with no comparisons. The result is sorted by
/// `(distance, index)`.
#[must_use]
pub fn smallest_k_by_histogram(dists: &[u16], k: usize) -> Vec<u32> {
    if k == 0 || dists.is_empty() {
        return Vec::new();
    }
    if k >= dists.len() {
        let mut all: Vec<u32> = (0..dists.len() as u32).collect();
        all.sort_by_key(|&i| (dists[i as usize], i));
        return all;
    }

    let max = usize::from(*dists.iter().max().unwrap_or(&0));
    let mut hist = vec![0usize; max + 1];
    for &d in dists {
        hist[usize::from(d)] += 1;
    }

    // Smallest threshold t such that at least k values are <= t.
    let mut cumulative = 0;
    let mut threshold = max;
    for (d, &count) in hist.iter().enumerate() {
        cumulative += count;
        if cumulative >= k {
            threshold = d;
            break;
        }
    }
    let below: usize = hist[..threshold].iter().sum();
    let mut room_at_threshold = k - below;

    let mut out = Vec::with_capacity(k);
    for (i, &d) in dists.iter().enumerate() {
        let d = usize::from(d);
        if d < threshold {
            out.push(i as u32);
        } else if d == threshold && room_at_threshold > 0 {
            out.push(i as u32);
            room_at_threshold -= 1;
        }
    }
    out.sort_by_key(|&i| (dists[i as usize], i));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_orders_by_score_then_id() {
        let items = [(5, 0.5), (1, 0.9), (3, 0.5), (2, 0.1)];
        assert_eq!(top_k(items, 3), [(1, 0.9), (3, 0.5), (5, 0.5)]);
        assert_eq!(top_k(items, 0), []);
        assert_eq!(top_k(items, 10).len(), 4);
    }

    #[test]
    fn huge_k_does_not_preallocate() {
        assert_eq!(top_k([(1, 1.0)], usize::MAX), [(1, 1.0)]);
    }

    #[test]
    fn nan_ranks_last() {
        let items = [(0, f32::NAN), (1, -5.0)];
        assert_eq!(top_k(items, 1), [(1, -5.0)]);
    }

    #[test]
    fn histogram_select_matches_sorting() {
        let dists = [7u16, 3, 3, 9, 0, 3, 7];
        assert_eq!(smallest_k_by_histogram(&dists, 3), [4, 1, 2]);
        assert_eq!(smallest_k_by_histogram(&dists, 5), [4, 1, 2, 5, 0]);
        assert_eq!(smallest_k_by_histogram(&dists, 100).len(), 7);
        assert_eq!(smallest_k_by_histogram(&dists, 0), Vec::<u32>::new());
    }
}
