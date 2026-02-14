//! Transposition table for caching neural network evaluations.
//!
//! A transposition table stores (policy, value) pairs keyed by Zobrist hash.
//! When the same chess position is reached via a different move order, we can
//! reuse the cached evaluation instead of calling the neural network again.
//!
//! # Design
//!
//! - We cache **evaluations only**, not subtrees. Sharing subtrees would create
//!   a DAG (directed acyclic graph) which complicates backpropagation. Keeping
//!   it simple: cache `(policy, value)` pairs keyed by Zobrist hash.
//!
//! - **Fixed size**: The table has a fixed number of slots. The capacity is
//!   rounded up to the nearest power of 2 so we can use bitwise AND instead
//!   of modulo for the index calculation.
//!
//! - **Replace-always**: On hash collision (two different positions map to the
//!   same slot), the newer entry overwrites the older one. This is the simplest
//!   and usually most effective strategy for MCTS, where recent evaluations are
//!   more relevant than old ones.
//!
//! - **Full hash verification**: Each entry stores the complete 64-bit Zobrist
//!   hash. On lookup, we verify that the stored hash matches the query hash
//!   before returning a hit. This catches false positives from hash collisions
//!   in the index calculation.

// =============================================================================
// CachedEval
// =============================================================================

/// Cached neural network evaluation for a position.
#[derive(Clone, Debug)]
pub struct CachedEval {
    /// Policy logits (full 4672-element vector from NN).
    pub policy: Vec<f32>,
    /// Value estimate from NN, in [-1, 1].
    pub value: f32,
}

// =============================================================================
// Entry (internal)
// =============================================================================

/// Internal table entry pairing a full Zobrist hash with a cached evaluation.
struct Entry {
    /// Full Zobrist hash for verification (to detect index collisions).
    hash: u64,
    /// The cached evaluation.
    eval: CachedEval,
}

// =============================================================================
// TranspositionTable
// =============================================================================

/// Fixed-size transposition table for caching NN evaluations.
///
/// Uses Zobrist hash as the key. On collision, the newer entry replaces
/// the older one (replace-always strategy).
pub struct TranspositionTable {
    /// The table slots, indexed by `hash & mask`.
    entries: Vec<Option<Entry>>,
    /// Bitmask for fast index calculation: `capacity - 1` (capacity is power of 2).
    mask: u64,
    /// Number of occupied entries.
    size: usize,
    /// Number of cache hits.
    hits: u64,
    /// Number of cache misses.
    misses: u64,
}

impl TranspositionTable {
    /// Create a new transposition table with the given capacity (number of entries).
    ///
    /// The capacity is rounded up to the nearest power of 2 for fast modulo
    /// via bitwise AND. Minimum capacity is 1 (rounded up to 1).
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1).next_power_of_two();
        let mut entries = Vec::with_capacity(capacity);
        entries.resize_with(capacity, || None);

        TranspositionTable {
            entries,
            mask: (capacity - 1) as u64,
            size: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a position by its Zobrist hash.
    ///
    /// Returns `Some(&CachedEval)` on a cache hit (hash matches), or `None`
    /// on a miss (empty slot or hash mismatch). Updates hit/miss statistics.
    pub fn get(&mut self, hash: u64) -> Option<&CachedEval> {
        let index = (hash & self.mask) as usize;
        match &self.entries[index] {
            Some(entry) if entry.hash == hash => {
                self.hits += 1;
                // Re-borrow to satisfy the borrow checker: we need to return
                // a reference tied to `&self`, not to the match arm.
                self.entries[index].as_ref().map(|e| &e.eval)
            }
            _ => {
                self.misses += 1;
                None
            }
        }
    }

    /// Store an evaluation for a position.
    ///
    /// If the slot is already occupied (by the same or a different hash),
    /// it is overwritten (replace-always strategy).
    pub fn put(&mut self, hash: u64, eval: CachedEval) {
        let index = (hash & self.mask) as usize;

        // Track size: only increment if the slot was empty.
        if self.entries[index].is_none() {
            self.size += 1;
        }

        self.entries[index] = Some(Entry { hash, eval });
    }

    /// Clear all entries and reset statistics.
    pub fn clear(&mut self) {
        for slot in self.entries.iter_mut() {
            *slot = None;
        }
        self.size = 0;
        self.hits = 0;
        self.misses = 0;
    }

    /// Get the hit rate as a fraction in [0.0, 1.0].
    ///
    /// Returns 0.0 if no lookups have been performed.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if no entries are stored.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the actual capacity (always a power of 2).
    pub fn capacity(&self) -> usize {
        self.entries.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a dummy CachedEval with a known value.
    fn make_eval(value: f32) -> CachedEval {
        CachedEval {
            policy: vec![value; 10], // Small policy for tests.
            value,
        }
    }

    /// Helper to create a full-sized CachedEval (4672 elements).
    fn make_full_eval(value: f32) -> CachedEval {
        CachedEval {
            policy: vec![value; 4672],
            value,
        }
    }

    // ========================================================================
    // 1. Put and get
    // ========================================================================

    #[test]
    fn put_and_get() {
        let mut tt = TranspositionTable::new(1024);

        let hash = 0xDEAD_BEEF_CAFE_BABE;
        let eval = make_eval(0.75);

        tt.put(hash, eval);

        let retrieved = tt.get(hash).expect("should find the entry");
        assert_eq!(retrieved.value, 0.75);
        assert_eq!(retrieved.policy.len(), 10);
        assert_eq!(retrieved.policy[0], 0.75);
    }

    // ========================================================================
    // 2. Cache miss
    // ========================================================================

    #[test]
    fn cache_miss_on_empty_table() {
        let mut tt = TranspositionTable::new(1024);
        assert!(tt.get(12345).is_none());
    }

    #[test]
    fn cache_miss_on_nonexistent_hash() {
        let mut tt = TranspositionTable::new(1024);
        tt.put(111, make_eval(0.5));
        assert!(tt.get(222).is_none());
    }

    // ========================================================================
    // 3. Collision handling (replace-always)
    // ========================================================================

    #[test]
    fn collision_replaces_old_entry() {
        // Use a tiny table so collisions are guaranteed.
        let mut tt = TranspositionTable::new(4); // capacity = 4, mask = 3

        // Two hashes that map to the same index (both & 3 == same value).
        let hash_a = 0b0101; // index = 1
        let hash_b = 0b1101; // index = 1 (same low 2 bits)

        tt.put(hash_a, make_eval(0.1));
        tt.put(hash_b, make_eval(0.9));

        // The newer entry (hash_b) should have replaced hash_a.
        let result = tt.get(hash_b).expect("should find newer entry");
        assert_eq!(result.value, 0.9);

        // hash_a should no longer be found (it was overwritten).
        assert!(tt.get(hash_a).is_none());
    }

    // ========================================================================
    // 4. Hash verification (different hash, same index)
    // ========================================================================

    #[test]
    fn hash_verification_prevents_false_positive() {
        let mut tt = TranspositionTable::new(4); // capacity = 4, mask = 3

        let hash_a = 0b0010; // index = 2
        let hash_b = 0b1110; // index = 2 (same low 2 bits, different full hash)

        tt.put(hash_a, make_eval(0.5));

        // Querying hash_b should return None even though it maps to the same
        // index, because the full hashes don't match.
        assert!(tt.get(hash_b).is_none());

        // hash_a should still be found.
        assert!(tt.get(hash_a).is_some());
    }

    // ========================================================================
    // 5. Clear resets everything
    // ========================================================================

    #[test]
    fn clear_resets_everything() {
        let mut tt = TranspositionTable::new(256);

        // Populate and trigger some stats.
        for i in 0..50u64 {
            tt.put(i, make_eval(i as f32));
        }
        for i in 0..50u64 {
            tt.get(i);
        }
        assert!(tt.len() > 0);
        assert!(tt.hits > 0);

        tt.clear();

        assert_eq!(tt.len(), 0);
        assert!(tt.is_empty());
        assert_eq!(tt.hit_rate(), 0.0);

        // All previous entries should be gone.
        for i in 0..50u64 {
            assert!(tt.get(i).is_none());
        }
    }

    // ========================================================================
    // 6. Hit rate tracking
    // ========================================================================

    #[test]
    fn hit_rate_tracking() {
        let mut tt = TranspositionTable::new(1024);

        tt.put(1, make_eval(0.1));
        tt.put(2, make_eval(0.2));

        // 2 hits.
        tt.get(1);
        tt.get(2);
        // 2 misses.
        tt.get(3);
        tt.get(4);

        // 2 hits / 4 total = 0.5.
        assert!((tt.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn hit_rate_zero_before_any_lookups() {
        let tt = TranspositionTable::new(128);
        assert_eq!(tt.hit_rate(), 0.0);
    }

    // ========================================================================
    // 7. Capacity is power of 2
    // ========================================================================

    #[test]
    fn capacity_is_power_of_two() {
        // Request non-power-of-2 capacities; actual capacity should round up.
        let tt = TranspositionTable::new(3);
        assert_eq!(tt.capacity(), 4);

        let tt = TranspositionTable::new(5);
        assert_eq!(tt.capacity(), 8);

        let tt = TranspositionTable::new(100);
        assert_eq!(tt.capacity(), 128);

        let tt = TranspositionTable::new(1024);
        assert_eq!(tt.capacity(), 1024);

        // Edge case: 0 rounds up to 1 (then to next power of 2 = 1).
        let tt = TranspositionTable::new(0);
        assert_eq!(tt.capacity(), 1);

        // Already a power of 2.
        let tt = TranspositionTable::new(256);
        assert_eq!(tt.capacity(), 256);
    }

    // ========================================================================
    // 8. Large table
    // ========================================================================

    #[test]
    fn large_table_store_and_retrieve() {
        let mut tt = TranspositionTable::new(1_000_000);
        // Capacity should be rounded up to the next power of 2 >= 1M.
        assert!(tt.capacity() >= 1_000_000);
        assert!(tt.capacity().is_power_of_two());

        // Store 1000 entries with full-size policy vectors.
        for i in 0..1000u64 {
            tt.put(i * 7919, make_full_eval(i as f32 * 0.001)); // use prime multiplier for spread
        }

        assert_eq!(tt.len(), 1000);

        // Retrieve and verify all 1000 entries.
        let mut found = 0;
        for i in 0..1000u64 {
            let hash = i * 7919;
            if let Some(eval) = tt.get(hash) {
                let expected = i as f32 * 0.001;
                assert!(
                    (eval.value - expected).abs() < 1e-6,
                    "Value mismatch for hash {}: expected {}, got {}",
                    hash,
                    expected,
                    eval.value
                );
                assert_eq!(eval.policy.len(), 4672);
                found += 1;
            }
        }

        // All 1000 should be retrievable (with 1M capacity, no collisions
        // among just 1000 entries unless very unlucky -- with prime spacing
        // and a power-of-2 table this should work).
        assert_eq!(
            found, 1000,
            "Expected all 1000 entries to be retrievable, got {}",
            found
        );
    }

    // ========================================================================
    // Extra: len tracking
    // ========================================================================

    #[test]
    fn len_tracks_occupied_slots() {
        let mut tt = TranspositionTable::new(64);

        assert_eq!(tt.len(), 0);
        assert!(tt.is_empty());

        tt.put(10, make_eval(0.1));
        assert_eq!(tt.len(), 1);

        tt.put(20, make_eval(0.2));
        assert_eq!(tt.len(), 2);

        // Overwriting the same hash shouldn't change the count.
        tt.put(10, make_eval(0.3));
        assert_eq!(tt.len(), 2);
    }
}
