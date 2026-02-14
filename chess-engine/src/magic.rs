//! Magic bitboard attack generation for sliding pieces (bishops, rooks, queens).
//!
//! # What are magic bitboards?
//!
//! Sliding pieces (bishops, rooks, queens) have attacks that depend on the
//! positions of other pieces that might block their rays. Unlike knights and
//! kings whose attacks depend only on their own square, a rook on e4 attacks
//! different squares depending on whether there is a piece on e6 blocking the
//! north ray.
//!
//! The naive approach is to walk each ray at runtime until hitting a blocker or
//! the board edge. That works but involves unpredictable branches. Magic
//! bitboards replace this with an O(1) table lookup:
//!
//! ```text
//! attacks = TABLE[square][(occupancy & mask) * magic >> shift]
//! ```
//!
//! # How it works
//!
//! For each square and piece type (rook or bishop):
//!
//! 1. **Occupancy mask**: Which squares along the piece's rays could potentially
//!    block it? Edge squares are excluded because a blocker on the edge does not
//!    change the attack set (there is nothing beyond the edge to block or unblock).
//!
//! 2. **Blocker enumeration**: For a mask with N bits set, there are 2^N possible
//!    blocker configurations. For each one, we compute the actual attack set using
//!    the slow ray-walking method.
//!
//! 3. **Magic number**: A 64-bit constant that, when multiplied by the masked
//!    occupancy and right-shifted, maps each blocker configuration to a unique
//!    index in a lookup table. Finding these numbers is done by trial: generate
//!    random sparse candidates and test if they produce a perfect (collision-free)
//!    hash.
//!
//! 4. **Lookup table**: An array of attack bitboards, indexed by the magic hash.
//!    At runtime, we just compute the index and read the answer.
//!
//! # Memory usage
//!
//! Rook tables total ~800KB (rooks have up to 12-bit masks, so 4096 entries per
//! square in the worst case). Bishop tables total ~40KB (bishops have up to
//! 9-bit masks, so 512 entries per square). This easily fits in L2 cache.
//!
//! # Initialization
//!
//! Tables are built lazily on first use via `std::sync::LazyLock`. The magic
//! number search takes <100ms total for all 128 squares (64 rook + 64 bishop),
//! so there is no perceptible delay.

use std::sync::LazyLock;

use crate::bitboard::Bitboard;
use crate::types::{Square, NUM_SQUARES};

// =============================================================================
// Edge masks (used in tests to verify occupancy masks exclude board edges)
// =============================================================================

// =============================================================================
// Occupancy masks
// =============================================================================
//
// An occupancy mask defines which squares along a piece's rays could affect
// its attack set. Edge squares are excluded because:
//   - A piece on the board edge is the last square in a ray regardless of
//     whether it is occupied. So it does not change the set of attacked squares.
//   - Excluding edges means fewer bits in the mask, which means smaller lookup
//     tables (2^N entries where N = popcount of mask).

/// Computes the rook occupancy mask for a given square.
///
/// This includes all squares along the four cardinal rays (north, south, east,
/// west) from the square, **excluding** the square itself and **excluding**
/// squares on the board edge (rank 1, rank 8, file A, file H) unless the rook
/// is on that edge itself (in which case that edge's squares are naturally part
/// of a different ray and are handled correctly).
///
/// More precisely: for each ray direction, we walk from the square outward and
/// include every square except the last one in that direction (since the last
/// square is always on the edge and cannot affect what is beyond it).
const fn rook_occupancy_mask(sq: u8) -> Bitboard {
    let file = sq % 8;
    let rank = sq / 8;
    let mut mask: u64 = 0;

    // North ray: from rank+1 to rank 6 (not rank 7, which is the edge rank 8)
    {
        let mut r = rank + 1;
        while r <= 6 {
            mask |= 1u64 << (r * 8 + file);
            r += 1;
        }
    }

    // South ray: from rank-1 down to rank 1 (not rank 0, which is the edge rank 1)
    {
        let mut r = rank as i8 - 1;
        while r >= 1 {
            mask |= 1u64 << (r as u8 * 8 + file);
            r -= 1;
        }
    }

    // East ray: from file+1 to file 6 (not file 7, which is the H file edge)
    {
        let mut f = file + 1;
        while f <= 6 {
            mask |= 1u64 << (rank * 8 + f);
            f += 1;
        }
    }

    // West ray: from file-1 down to file 1 (not file 0, which is the A file edge)
    {
        let mut f = file as i8 - 1;
        while f >= 1 {
            mask |= 1u64 << (rank * 8 + f as u8);
            f -= 1;
        }
    }

    Bitboard(mask)
}

/// Computes the bishop occupancy mask for a given square.
///
/// Same concept as the rook mask, but for diagonal rays. We walk in each of
/// the four diagonal directions and include all squares except those on the
/// board edges (since edge squares cannot affect what is beyond them).
const fn bishop_occupancy_mask(sq: u8) -> Bitboard {
    let file = sq % 8;
    let rank = sq / 8;
    let mut mask: u64 = 0;

    // Northeast diagonal: file+1, rank+1 -- stop before edges
    {
        let mut f = file + 1;
        let mut r = rank + 1;
        while f <= 6 && r <= 6 {
            mask |= 1u64 << (r * 8 + f);
            f += 1;
            r += 1;
        }
    }

    // Northwest diagonal: file-1, rank+1
    {
        let mut f = file as i8 - 1;
        let mut r = rank + 1;
        while f >= 1 && r <= 6 {
            mask |= 1u64 << (r * 8 + f as u8);
            f -= 1;
            r += 1;
        }
    }

    // Southeast diagonal: file+1, rank-1
    {
        let mut f = file + 1;
        let mut r = rank as i8 - 1;
        while f <= 6 && r >= 1 {
            mask |= 1u64 << (r as u8 * 8 + f);
            f += 1;
            r -= 1;
        }
    }

    // Southwest diagonal: file-1, rank-1
    {
        let mut f = file as i8 - 1;
        let mut r = rank as i8 - 1;
        while f >= 1 && r >= 1 {
            mask |= 1u64 << (r as u8 * 8 + f as u8);
            f -= 1;
            r -= 1;
        }
    }

    Bitboard(mask)
}

// =============================================================================
// Precomputed occupancy mask tables
// =============================================================================

/// Rook occupancy masks for all 64 squares, computed at compile time.
const ROOK_MASKS: [Bitboard; NUM_SQUARES] = {
    let mut table = [Bitboard::EMPTY; NUM_SQUARES];
    let mut sq = 0u8;
    while sq < 64 {
        table[sq as usize] = rook_occupancy_mask(sq);
        sq += 1;
    }
    table
};

/// Bishop occupancy masks for all 64 squares, computed at compile time.
const BISHOP_MASKS: [Bitboard; NUM_SQUARES] = {
    let mut table = [Bitboard::EMPTY; NUM_SQUARES];
    let mut sq = 0u8;
    while sq < 64 {
        table[sq as usize] = bishop_occupancy_mask(sq);
        sq += 1;
    }
    table
};

// =============================================================================
// Slow (reference) attack generators
// =============================================================================
//
// These walk each ray one square at a time, stopping when they hit a blocker
// or the board edge. They include the blocker square in the attack set (you
// can capture a blocking piece). These are used during initialization to fill
// the magic lookup tables, and also as reference implementations for testing.

/// Computes rook attacks by walking rays. The `blockers` bitboard represents
/// all occupied squares that could block the rook's movement.
///
/// The rook's attacked squares include every square along each cardinal ray
/// up to and including the first blocker encountered. If no blocker is found,
/// the ray extends to the board edge.
const fn rook_attacks_slow(sq: u8, blockers: u64) -> Bitboard {
    let file = sq % 8;
    let rank = sq / 8;
    let mut attacks: u64 = 0;

    // North ray
    {
        let mut r = rank + 1;
        while r <= 7 {
            let bit = 1u64 << (r * 8 + file);
            attacks |= bit;
            if blockers & bit != 0 {
                break; // hit a blocker -- include it (can capture) but stop
            }
            r += 1;
        }
    }

    // South ray
    {
        let mut r = rank as i8 - 1;
        while r >= 0 {
            let bit = 1u64 << (r as u8 * 8 + file);
            attacks |= bit;
            if blockers & bit != 0 {
                break;
            }
            r -= 1;
        }
    }

    // East ray
    {
        let mut f = file + 1;
        while f <= 7 {
            let bit = 1u64 << (rank * 8 + f);
            attacks |= bit;
            if blockers & bit != 0 {
                break;
            }
            f += 1;
        }
    }

    // West ray
    {
        let mut f = file as i8 - 1;
        while f >= 0 {
            let bit = 1u64 << (rank * 8 + f as u8);
            attacks |= bit;
            if blockers & bit != 0 {
                break;
            }
            f -= 1;
        }
    }

    Bitboard(attacks)
}

/// Computes bishop attacks by walking diagonal rays. Same logic as the rook
/// but in diagonal directions.
const fn bishop_attacks_slow(sq: u8, blockers: u64) -> Bitboard {
    let file = sq % 8;
    let rank = sq / 8;
    let mut attacks: u64 = 0;

    // Northeast
    {
        let mut f = file + 1;
        let mut r = rank + 1;
        while f <= 7 && r <= 7 {
            let bit = 1u64 << (r * 8 + f);
            attacks |= bit;
            if blockers & bit != 0 {
                break;
            }
            f += 1;
            r += 1;
        }
    }

    // Northwest
    {
        let mut f = file as i8 - 1;
        let mut r = rank + 1;
        while f >= 0 && r <= 7 {
            let bit = 1u64 << (r * 8 + f as u8);
            attacks |= bit;
            if blockers & bit != 0 {
                break;
            }
            f -= 1;
            r += 1;
        }
    }

    // Southeast
    {
        let mut f = file + 1;
        let mut r = rank as i8 - 1;
        while f <= 7 && r >= 0 {
            let bit = 1u64 << (r as u8 * 8 + f);
            attacks |= bit;
            if blockers & bit != 0 {
                break;
            }
            f += 1;
            r -= 1;
        }
    }

    // Southwest
    {
        let mut f = file as i8 - 1;
        let mut r = rank as i8 - 1;
        while f >= 0 && r >= 0 {
            let bit = 1u64 << (r as u8 * 8 + f as u8);
            attacks |= bit;
            if blockers & bit != 0 {
                break;
            }
            f -= 1;
            r -= 1;
        }
    }

    Bitboard(attacks)
}

// =============================================================================
// Blocker subset enumeration
// =============================================================================

/// Enumerates all subsets of a bitmask using the Carry-Rippler trick.
///
/// Given a mask with N bits set, this generates all 2^N subsets of those bits.
/// The trick works by using arithmetic carry to "ripple" through the set bits:
///
/// ```text
/// subset = (subset - mask) & mask
/// ```
///
/// Starting from 0, this visits every subset exactly once and returns to 0.
///
/// # Why do we need this?
///
/// For a rook on e4 with an occupancy mask of (say) 10 bits, there are 2^10 =
/// 1024 possible configurations of blockers within those 10 squares. We need
/// to compute the rook's attacks for each one and store them in the lookup table.
fn enumerate_subsets(mask: u64) -> Vec<u64> {
    let mut subsets = Vec::with_capacity(1usize << mask.count_ones());
    let mut subset: u64 = 0;
    loop {
        subsets.push(subset);
        // Carry-Rippler: advances to the next subset of mask
        subset = subset.wrapping_sub(mask) & mask;
        if subset == 0 {
            break;
        }
    }
    subsets
}

// =============================================================================
// Magic number search
// =============================================================================
//
// A magic number is a 64-bit constant M such that for a given square's
// occupancy mask, multiplying any blocker subset by M and right-shifting by
// (64 - N) produces a unique index (where N = popcount of the mask).
//
// The search is simple: generate random sparse u64 values and test each one.
// Sparse values (few bits set) tend to work well because the multiplication
// spreads the mask bits into the high bits of the result, creating a good hash.
// For most squares this finds a magic in <1000 attempts.

/// A simple pseudo-random number generator (xorshift64) for generating magic
/// number candidates. We use our own PRNG instead of pulling in a dependency,
/// and because we need deterministic, reproducible behavior across platforms.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generates the next pseudo-random u64.
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generates a sparse random u64 (few bits set), which tends to produce
    /// good magic numbers. We AND three random values together, which gives
    /// each bit roughly a 1/8 chance of being set.
    fn next_sparse_u64(&mut self) -> u64 {
        self.next_u64() & self.next_u64() & self.next_u64()
    }
}

/// Searches for a magic number for a given square, mask, and set of
/// (blocker -> attack) pairs.
///
/// Returns `(magic, shift, table)` where:
///   - `magic` is the found magic number
///   - `shift` is `64 - popcount(mask)` (the right-shift amount)
///   - `table` is the filled lookup table indexed by `(blockers * magic) >> shift`
fn find_magic(
    mask: Bitboard,
    blocker_attacks: &[(u64, Bitboard)],
    rng: &mut Rng,
) -> (u64, u32, Vec<Bitboard>) {
    let num_bits = mask.count();
    let shift = 64 - num_bits;
    let table_size = 1usize << num_bits;

    // Sentinel value used to detect collisions. We use a bitboard that should
    // never appear as a valid attack set.
    let unset = Bitboard::new(u64::MAX);

    loop {
        let candidate = rng.next_sparse_u64();

        // Quick reject: the multiplication should spread bits into the top
        // portion. If the top bits have too few set bits, the hash is likely
        // poor. We require at least 6 bits in the top 8 bits of the product
        // of the mask and candidate.
        if (mask.0.wrapping_mul(candidate) & 0xFF00_0000_0000_0000).count_ones() < 6 {
            continue;
        }

        let mut table = vec![unset; table_size];
        let mut valid = true;

        for &(blockers, attacks) in blocker_attacks {
            let index = (blockers.wrapping_mul(candidate) >> shift) as usize;

            if table[index] == unset {
                // Slot is empty -- fill it
                table[index] = attacks;
            } else if table[index] != attacks {
                // Collision with a different attack set -- this candidate fails.
                // (If it collides with the same attack set, that is fine -- it is
                // a "constructive collision" where two different blocker configs
                // happen to produce the same attacks.)
                valid = false;
                break;
            }
        }

        if valid {
            // Replace any unfilled slots with EMPTY (should not happen for a
            // correct magic, but defensive programming)
            for entry in table.iter_mut() {
                if *entry == unset {
                    *entry = Bitboard::EMPTY;
                }
            }
            return (candidate, shift, table);
        }
    }
}

// =============================================================================
// Magic table entry (per-square data)
// =============================================================================

/// All the data needed to perform a magic bitboard lookup for one square.
struct MagicEntry {
    /// The occupancy mask: which squares are relevant blockers for this square.
    mask: Bitboard,

    /// The magic number that hashes blocker configurations to table indices.
    magic: u64,

    /// The right-shift amount: `64 - popcount(mask)`.
    shift: u32,

    /// The lookup table: attack bitboards indexed by `(blockers & mask) * magic >> shift`.
    table: Vec<Bitboard>,
}

impl MagicEntry {
    /// Looks up the attack set for a given board occupancy.
    ///
    /// This is the hot path -- it should compile down to:
    ///   AND, MUL, SHR, array index
    /// which is about 4-5 instructions.
    #[inline]
    fn attacks(&self, occupancy: Bitboard) -> Bitboard {
        let blockers = (occupancy & self.mask).0;
        let index = (blockers.wrapping_mul(self.magic) >> self.shift) as usize;
        self.table[index]
    }
}

// =============================================================================
// Full magic table (all 64 squares for one piece type)
// =============================================================================

/// Magic tables for all 64 squares of one piece type (rook or bishop).
struct MagicTable {
    entries: Vec<MagicEntry>,
}

impl MagicTable {
    /// Builds the magic table for either rooks or bishops.
    ///
    /// The `masks` parameter provides the occupancy mask for each square, and
    /// `slow_attacks` is the reference function that computes attacks given a
    /// square index and blocker configuration.
    fn build(
        masks: &[Bitboard; NUM_SQUARES],
        slow_attacks: fn(u8, u64) -> Bitboard,
        rng: &mut Rng,
    ) -> Self {
        let mut entries = Vec::with_capacity(NUM_SQUARES);

        for sq in 0..64u8 {
            let mask = masks[sq as usize];

            // Enumerate all possible blocker configurations within the mask
            let subsets = enumerate_subsets(mask.0);
            let blocker_attacks: Vec<(u64, Bitboard)> = subsets
                .iter()
                .map(|&blockers| (blockers, slow_attacks(sq, blockers)))
                .collect();

            // Find a magic number that perfectly hashes all configurations
            let (magic, shift, table) = find_magic(mask, &blocker_attacks, rng);

            entries.push(MagicEntry {
                mask,
                magic,
                shift,
                table,
            });
        }

        MagicTable { entries }
    }

    /// Looks up the attacks for a piece on the given square with the given
    /// board occupancy.
    #[inline]
    fn attacks(&self, square: Square, occupancy: Bitboard) -> Bitboard {
        self.entries[square.index() as usize].attacks(occupancy)
    }
}

// =============================================================================
// Lazy initialization
// =============================================================================
//
// The magic tables are built once on first access. We use separate seeds for
// rooks and bishops so they search in different random sequences, which helps
// avoid accidentally finding poor magics in one table due to the other table's
// search consuming "good" random values.

/// Lazily initialized rook magic table.
static ROOK_MAGICS: LazyLock<MagicTable> = LazyLock::new(|| {
    let mut rng = Rng::new(0xDEAD_BEEF_CAFE_1234);
    MagicTable::build(&ROOK_MASKS, |sq, blockers| rook_attacks_slow(sq, blockers), &mut rng)
});

/// Lazily initialized bishop magic table.
static BISHOP_MAGICS: LazyLock<MagicTable> = LazyLock::new(|| {
    let mut rng = Rng::new(0xBADC_0FFE_E0DD_5678);
    MagicTable::build(&BISHOP_MASKS, |sq, blockers| bishop_attacks_slow(sq, blockers), &mut rng)
});

// =============================================================================
// Public API
// =============================================================================

/// Returns the attack bitboard for a rook on the given square, considering
/// the given board occupancy.
///
/// This is an O(1) lookup from the precomputed magic table. On first call,
/// the tables are initialized (takes <100ms).
///
/// # Example
///
/// ```ignore
/// use chess_engine::magic::rook_attacks;
/// use chess_engine::bitboard::Bitboard;
/// use chess_engine::types::Square;
///
/// let attacks = rook_attacks(Square::E4, Bitboard::EMPTY);
/// assert_eq!(attacks.count(), 14); // full rank + full file minus self
/// ```
#[inline]
pub fn rook_attacks(square: Square, occupancy: Bitboard) -> Bitboard {
    ROOK_MAGICS.attacks(square, occupancy)
}

/// Returns the attack bitboard for a bishop on the given square, considering
/// the given board occupancy.
///
/// This is an O(1) lookup from the precomputed magic table.
#[inline]
pub fn bishop_attacks(square: Square, occupancy: Bitboard) -> Bitboard {
    BISHOP_MAGICS.attacks(square, occupancy)
}

/// Returns the attack bitboard for a queen on the given square, considering
/// the given board occupancy.
///
/// A queen combines the movement of a rook and a bishop, so its attacks are
/// simply the union of both.
#[inline]
pub fn queen_attacks(square: Square, occupancy: Bitboard) -> Bitboard {
    rook_attacks(square, occupancy) | bishop_attacks(square, occupancy)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;

    // ---- Occupancy mask tests -----------------------------------------------

    #[test]
    fn rook_mask_e4_excludes_edges() {
        // A rook on e4 (file=4, rank=3). The occupancy mask should include:
        //   - North ray: e5, e6, e7 (not e8, that is rank 8 edge)
        //   - South ray: e3, e2 (not e1, that is rank 1 edge)
        //   - East ray:  f4, g4 (not h4, that is file H edge)
        //   - West ray:  d4, c4, b4 (not a4, that is file A edge)
        // Total: 3 + 2 + 2 + 3 = 10 bits
        let mask = ROOK_MASKS[Square::E4.index() as usize];
        assert_eq!(
            mask.count(), 10,
            "Rook occupancy mask on e4 should have 10 bits, got {}",
            mask.count()
        );

        // Verify specific squares are included
        assert!(mask.contains(Square::E5));
        assert!(mask.contains(Square::E6));
        assert!(mask.contains(Square::E7));
        assert!(mask.contains(Square::E3));
        assert!(mask.contains(Square::E2));
        assert!(mask.contains(Square::F4));
        assert!(mask.contains(Square::G4));
        assert!(mask.contains(Square::D4));
        assert!(mask.contains(Square::C4));
        assert!(mask.contains(Square::B4));

        // Verify edge squares are excluded
        assert!(!mask.contains(Square::E8), "rank 8 edge should be excluded");
        assert!(!mask.contains(Square::E1), "rank 1 edge should be excluded");
        assert!(!mask.contains(Square::H4), "file H edge should be excluded");
        assert!(!mask.contains(Square::A4), "file A edge should be excluded");

        // The square itself should not be in the mask
        assert!(!mask.contains(Square::E4));
    }

    #[test]
    fn rook_mask_a1_corner() {
        // A rook on a1 (file=0, rank=0).
        // North ray: a2, a3, a4, a5, a6, a7 (not a8) = 6 squares
        // South ray: none (rank 0, can't go lower)
        // East ray: b1, c1, d1, e1, f1, g1 (not h1) = 6 squares
        // West ray: none (file 0, can't go left)
        // Total: 12
        let mask = ROOK_MASKS[Square::A1.index() as usize];
        assert_eq!(mask.count(), 12);
    }

    #[test]
    fn rook_mask_h8_corner() {
        // Same as a1 by symmetry: 12 bits
        let mask = ROOK_MASKS[Square::H8.index() as usize];
        assert_eq!(mask.count(), 12);
    }

    #[test]
    fn bishop_mask_e4_excludes_edges() {
        // A bishop on e4 (file=4, rank=3). Diagonals excluding edges:
        //   - NE: f5, g6 (not h7 -- wait, h7 is file 7 which is the edge? Yes.)
        //     Actually: f5 (5,4), g6 (6,5) -- next would be h7 (7,6), but file 7
        //     is the H edge. So 2 squares.
        //   - NW: d5 (3,4), c6 (2,5), b7 (1,6) -- next would be a8 but that is
        //     both file A and rank 8 edge. So 3 squares.
        //     Wait -- b7 is file 1, rank 6. We stop before edges, meaning
        //     f >= 1 and r <= 6. b7 is file=1 and rank=6, so just barely included.
        //     Next would be a8 (file=0, rank=7) which fails both conditions.
        //     So NW: d5, c6, b7 = 3 squares.
        //   - SE: f3 (5,2), g2 (6,1) -- next would be h1, but file 7 is edge.
        //     So 2 squares.
        //   - SW: d3 (3,2), c2 (2,1) -- next would be b1 (1,0), but rank 0 is
        //     edge. So 2 squares.
        //     Wait: d3 is file=3, rank=2. c2 is file=2, rank=1. Both f>=1 and r>=1.
        //     Next is b1 (file=1, rank=0). r=0 fails r>=1. So 2 squares.
        // Total: 2 + 3 + 2 + 2 = 9 bits
        //
        // Hmm, let me recount NE:
        //   f5 (file=5, rank=4): f<=6 and r<=6, yes
        //   g6 (file=6, rank=5): f<=6 and r<=6, yes
        //   h7 (file=7, rank=6): f<=6 fails. So 2 squares.
        //
        // And NW:
        //   d5 (file=3, rank=4): f>=1 and r<=6, yes
        //   c6 (file=2, rank=5): yes
        //   b7 (file=1, rank=6): f>=1 and r<=6, yes
        //   a8 (file=0, rank=7): f>=1 fails. So 3 squares.
        //
        // SE:
        //   f3 (file=5, rank=2): f<=6 and r>=1, yes
        //   g2 (file=6, rank=1): yes
        //   h1 (file=7, rank=0): f<=6 fails. So 2 squares.
        //
        // SW:
        //   d3 (file=3, rank=2): f>=1 and r>=1, yes
        //   c2 (file=2, rank=1): yes
        //   b1 (file=1, rank=0): r>=1 fails. So 2 squares.
        //
        // Total: 2 + 3 + 2 + 2 = 9 bits
        let mask = BISHOP_MASKS[Square::E4.index() as usize];
        assert_eq!(
            mask.count(), 9,
            "Bishop occupancy mask on e4 should have 9 bits, got {}",
            mask.count()
        );
    }

    #[test]
    fn bishop_mask_a1_corner() {
        // A bishop on a1 (file=0, rank=0). Only the NE diagonal exists.
        // NE: b2 (1,1), c3 (2,2), d4 (3,3), e5 (4,4), f6 (5,5), g7 (6,6)
        //     All satisfy f<=6 and r<=6.
        //     h8 (7,7) would fail f<=6. So 6 squares.
        // Other diagonals: all go off-board immediately.
        // Total: 6
        let mask = BISHOP_MASKS[Square::A1.index() as usize];
        assert_eq!(mask.count(), 6);
    }

    #[test]
    fn occupancy_masks_exclude_own_square() {
        // Neither rook nor bishop masks should include the piece's own square.
        for sq_idx in 0..64u8 {
            let sq = Square::new(sq_idx);
            assert!(
                !ROOK_MASKS[sq_idx as usize].contains(sq),
                "Rook mask for {} should not contain the square itself", sq
            );
            assert!(
                !BISHOP_MASKS[sq_idx as usize].contains(sq),
                "Bishop mask for {} should not contain the square itself", sq
            );
        }
    }

    #[test]
    fn rook_masks_exclude_all_edges_for_center_squares() {
        // For squares not on any edge, the mask should not contain any edge squares.
        let edges = Bitboard::RANK_1 | Bitboard::RANK_8 | Bitboard::FILE_A | Bitboard::FILE_H;
        for rank in 1..7u8 {
            for file in 1..7u8 {
                let sq = Square::from_file_rank(file, rank);
                let mask = ROOK_MASKS[sq.index() as usize];
                let edge_overlap = mask & edges;
                assert!(
                    edge_overlap.is_empty(),
                    "Rook mask for center square {} should not include any edge squares, but includes {} edge squares",
                    sq, edge_overlap.count()
                );
            }
        }
    }

    // ---- Slow attack generator tests ----------------------------------------

    #[test]
    fn rook_attacks_slow_empty_board_e4() {
        // Rook on e4 with no blockers: attacks all of rank 4 (except e4 itself)
        // plus all of file e (except e4 itself).
        // Rank 4: a4, b4, c4, d4, f4, g4, h4 = 7 squares
        // File e: e1, e2, e3, e5, e6, e7, e8 = 7 squares
        // Total: 14
        let attacks = rook_attacks_slow(Square::E4.index(), 0);
        assert_eq!(attacks.count(), 14);

        // Spot check some squares
        assert!(attacks.contains(Square::A4));
        assert!(attacks.contains(Square::H4));
        assert!(attacks.contains(Square::E1));
        assert!(attacks.contains(Square::E8));
        assert!(!attacks.contains(Square::E4)); // not the square itself
    }

    #[test]
    fn rook_attacks_slow_empty_board_a1() {
        // Rook on a1 with no blockers: all of rank 1 (7 squares) + all of file A (7 squares) = 14
        let attacks = rook_attacks_slow(Square::A1.index(), 0);
        assert_eq!(attacks.count(), 14);
    }

    #[test]
    fn rook_attacks_slow_with_blocker_on_e7() {
        // Rook on e4 with a blocker on e7. The north ray should stop at e7.
        // North: e5, e6, e7 (blocked) = 3
        // South: e3, e2, e1 = 3
        // East:  f4, g4, h4 = 3
        // West:  d4, c4, b4, a4 = 4
        // Total: 3 + 3 + 3 + 4 = 13
        let blockers = Bitboard::from_square(Square::E7).0;
        let attacks = rook_attacks_slow(Square::E4.index(), blockers);
        assert_eq!(attacks.count(), 13);

        // e7 should be attacked (we can capture the blocker)
        assert!(attacks.contains(Square::E7));
        // e8 should NOT be attacked (blocked by e7)
        assert!(!attacks.contains(Square::E8));
    }

    #[test]
    fn bishop_attacks_slow_empty_board_e4() {
        // Bishop on e4 (file=4, rank=3) with no blockers.
        // NE: f5, g6, h7 = 3
        // NW: d5, c6, b7, a8 = 4
        // SE: f3, g2, h1 = 3
        // SW: d3, c2, b1 = 3
        // Total: 13
        let attacks = bishop_attacks_slow(Square::E4.index(), 0);
        assert_eq!(attacks.count(), 13);
    }

    #[test]
    fn bishop_attacks_slow_empty_board_a1() {
        // Bishop on a1 with no blockers. Only the NE diagonal.
        // NE: b2, c3, d4, e5, f6, g7, h8 = 7
        let attacks = bishop_attacks_slow(Square::A1.index(), 0);
        assert_eq!(attacks.count(), 7);
    }

    #[test]
    fn bishop_attacks_slow_with_blocker() {
        // Bishop on e4 with blocker on g6. NE ray stops at g6.
        // NE: f5, g6 (blocked) = 2 (instead of 3)
        // NW: d5, c6, b7, a8 = 4
        // SE: f3, g2, h1 = 3
        // SW: d3, c2, b1 = 3
        // Total: 12
        let blockers = Bitboard::from_square(Square::G6).0;
        let attacks = bishop_attacks_slow(Square::E4.index(), blockers);
        assert_eq!(attacks.count(), 12);

        assert!(attacks.contains(Square::G6)); // blocker is attacked (can capture)
        assert!(!attacks.contains(Square::H7)); // blocked by g6
    }

    // ---- Magic table lookup tests -------------------------------------------

    #[test]
    fn rook_attacks_empty_board_e4() {
        let attacks = rook_attacks(Square::E4, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 14);
    }

    #[test]
    fn rook_attacks_empty_board_a1() {
        let attacks = rook_attacks(Square::A1, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 14);
    }

    #[test]
    fn rook_attacks_with_blocker_on_e7() {
        let occupancy = Bitboard::from_square(Square::E7);
        let attacks = rook_attacks(Square::E4, occupancy);

        // North ray: e5, e6, e7 (stops at blocker, inclusive)
        assert!(attacks.contains(Square::E5));
        assert!(attacks.contains(Square::E6));
        assert!(attacks.contains(Square::E7));
        assert!(!attacks.contains(Square::E8));

        assert_eq!(attacks.count(), 13);
    }

    #[test]
    fn bishop_attacks_empty_board_e4() {
        let attacks = bishop_attacks(Square::E4, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 13);
    }

    #[test]
    fn bishop_attacks_empty_board_a1() {
        let attacks = bishop_attacks(Square::A1, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 7);
    }

    #[test]
    fn bishop_attacks_with_blockers() {
        // Bishop on e4 with blocker on g6 (blocks NE diagonal)
        let occupancy = Bitboard::from_square(Square::G6);
        let attacks = bishop_attacks(Square::E4, occupancy);
        assert_eq!(attacks.count(), 12);
        assert!(attacks.contains(Square::G6));
        assert!(!attacks.contains(Square::H7));
    }

    #[test]
    fn queen_attacks_is_rook_union_bishop() {
        // Test that queen_attacks = rook_attacks | bishop_attacks for several
        // squares and occupancies.
        let test_cases = [
            (Square::E4, Bitboard::EMPTY),
            (Square::A1, Bitboard::EMPTY),
            (Square::H8, Bitboard::EMPTY),
            (Square::D4, Bitboard::from_square(Square::D7)),
            (Square::E4, Bitboard::new(0x0000_0010_0000_1000)), // some blockers
        ];

        for (sq, occ) in test_cases {
            let queen = queen_attacks(sq, occ);
            let expected = rook_attacks(sq, occ) | bishop_attacks(sq, occ);
            assert_eq!(
                queen, expected,
                "queen_attacks({}, occ=0x{:016X}) should equal rook | bishop",
                sq, occ.0
            );
        }
    }

    #[test]
    fn queen_attacks_empty_board_e4_count() {
        // Queen on e4 empty board = rook (14) + bishop (13) = 27
        let attacks = queen_attacks(Square::E4, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 27);
    }

    // ---- Magic vs slow comparison (exhaustive correctness check) ------------

    #[test]
    fn rook_magic_matches_slow_for_all_squares_random_occupancies() {
        // For each square, test many random occupancy configurations and verify
        // the magic lookup matches the slow reference implementation.
        let mut rng = Rng::new(0x1234_5678_9ABC_DEF0);

        for sq_idx in 0..64u8 {
            let sq = Square::new(sq_idx);

            // Test with empty board
            let slow = rook_attacks_slow(sq_idx, 0);
            let fast = rook_attacks(sq, Bitboard::EMPTY);
            assert_eq!(slow, fast, "Rook mismatch for {} on empty board", sq);

            // Test with full board
            let slow = rook_attacks_slow(sq_idx, !0u64);
            let fast = rook_attacks(sq, Bitboard::FULL);
            assert_eq!(slow, fast, "Rook mismatch for {} on full board", sq);

            // Test with 100 random occupancies
            for _ in 0..100 {
                let occ = rng.next_u64();
                let slow = rook_attacks_slow(sq_idx, occ);
                let fast = rook_attacks(sq, Bitboard::new(occ));
                assert_eq!(
                    slow, fast,
                    "Rook mismatch for {} with occupancy 0x{:016X}",
                    sq, occ
                );
            }
        }
    }

    #[test]
    fn bishop_magic_matches_slow_for_all_squares_random_occupancies() {
        let mut rng = Rng::new(0xFEDC_BA98_7654_3210);

        for sq_idx in 0..64u8 {
            let sq = Square::new(sq_idx);

            let slow = bishop_attacks_slow(sq_idx, 0);
            let fast = bishop_attacks(sq, Bitboard::EMPTY);
            assert_eq!(slow, fast, "Bishop mismatch for {} on empty board", sq);

            let slow = bishop_attacks_slow(sq_idx, !0u64);
            let fast = bishop_attacks(sq, Bitboard::FULL);
            assert_eq!(slow, fast, "Bishop mismatch for {} on full board", sq);

            for _ in 0..100 {
                let occ = rng.next_u64();
                let slow = bishop_attacks_slow(sq_idx, occ);
                let fast = bishop_attacks(sq, Bitboard::new(occ));
                assert_eq!(
                    slow, fast,
                    "Bishop mismatch for {} with occupancy 0x{:016X}",
                    sq, occ
                );
            }
        }
    }

    // ---- Symmetry tests -----------------------------------------------------

    #[test]
    fn rook_attacks_symmetry_on_empty_board() {
        // If square B is in the rook attacks from square A on an empty board,
        // then square A must be in the rook attacks from square B. This is
        // because rook rays are bidirectional.
        for sq_idx in 0..64u8 {
            let sq = Square::new(sq_idx);
            let attacks = rook_attacks(sq, Bitboard::EMPTY);

            for target in attacks {
                let reverse = rook_attacks(target, Bitboard::EMPTY);
                assert!(
                    reverse.contains(sq),
                    "Rook symmetry violation: {} attacks {} but not vice versa",
                    sq, target
                );
            }
        }
    }

    #[test]
    fn bishop_attacks_symmetry_on_empty_board() {
        for sq_idx in 0..64u8 {
            let sq = Square::new(sq_idx);
            let attacks = bishop_attacks(sq, Bitboard::EMPTY);

            for target in attacks {
                let reverse = bishop_attacks(target, Bitboard::EMPTY);
                assert!(
                    reverse.contains(sq),
                    "Bishop symmetry violation: {} attacks {} but not vice versa",
                    sq, target
                );
            }
        }
    }

    // ---- Specific position tests --------------------------------------------

    #[test]
    fn rook_attacks_blocked_both_sides() {
        // Rook on e4 with blockers on e2, e6, c4, g4.
        // North: e5, e6 (blocked) = 2
        // South: e3, e2 (blocked) = 2
        // East:  f4, g4 (blocked) = 2
        // West:  d4, c4 (blocked) = 2
        // Total: 8
        let mut occupancy = Bitboard::EMPTY;
        occupancy.set(Square::E2);
        occupancy.set(Square::E6);
        occupancy.set(Square::C4);
        occupancy.set(Square::G4);

        let attacks = rook_attacks(Square::E4, occupancy);
        assert_eq!(attacks.count(), 8);
        assert!(attacks.contains(Square::E5));
        assert!(attacks.contains(Square::E6));
        assert!(attacks.contains(Square::E3));
        assert!(attacks.contains(Square::E2));
        assert!(attacks.contains(Square::F4));
        assert!(attacks.contains(Square::G4));
        assert!(attacks.contains(Square::D4));
        assert!(attacks.contains(Square::C4));
    }

    #[test]
    fn bishop_attacks_d4_empty_board() {
        // Bishop on d4 (file=3, rank=3).
        // NE: e5, f6, g7, h8 = 4
        // NW: c5, b6, a7 = 3
        // SE: e3, f2, g1 = 3
        // SW: c3, b2, a1 = 3
        // Total: 13
        let attacks = bishop_attacks(Square::D4, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 13);
    }

    #[test]
    fn bishop_attacks_h1_corner() {
        // Bishop on h1 (file=7, rank=0). Only NW diagonal exists.
        // NW: g2, f3, e4, d5, c6, b7, a8 = 7
        let attacks = bishop_attacks(Square::H1, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 7);
    }

    #[test]
    fn rook_all_squares_empty_board_attack_14() {
        // On an empty board, a rook always attacks 14 squares (7 on its rank +
        // 7 on its file, minus itself which is counted in neither).
        for sq_idx in 0..64u8 {
            let sq = Square::new(sq_idx);
            let attacks = rook_attacks(sq, Bitboard::EMPTY);
            assert_eq!(
                attacks.count(), 14,
                "Rook on {} empty board should attack 14 squares, got {}",
                sq, attacks.count()
            );
        }
    }

    #[test]
    fn rook_does_not_attack_own_square() {
        for sq_idx in 0..64u8 {
            let sq = Square::new(sq_idx);
            let attacks = rook_attacks(sq, Bitboard::EMPTY);
            assert!(
                !attacks.contains(sq),
                "Rook attacks should not include own square {}",
                sq
            );
        }
    }

    #[test]
    fn bishop_does_not_attack_own_square() {
        for sq_idx in 0..64u8 {
            let sq = Square::new(sq_idx);
            let attacks = bishop_attacks(sq, Bitboard::EMPTY);
            assert!(
                !attacks.contains(sq),
                "Bishop attacks should not include own square {}",
                sq
            );
        }
    }

    // ---- Subset enumeration test --------------------------------------------

    #[test]
    fn enumerate_subsets_count() {
        // A mask with N bits should produce exactly 2^N subsets.
        let mask = 0b1010_1100u64; // 4 bits set -> 16 subsets
        let subsets = enumerate_subsets(mask);
        assert_eq!(subsets.len(), 16);

        // All subsets should be subsets of the mask
        for &s in &subsets {
            assert_eq!(s & !mask, 0, "Subset 0x{:X} is not a subset of mask 0x{:X}", s, mask);
        }

        // All subsets should be unique
        let mut sorted = subsets.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), subsets.len(), "Subsets should all be unique");
    }

    #[test]
    fn enumerate_subsets_empty_mask() {
        let subsets = enumerate_subsets(0);
        assert_eq!(subsets.len(), 1);
        assert_eq!(subsets[0], 0);
    }

    #[test]
    fn enumerate_subsets_single_bit() {
        let mask = 1u64 << 20;
        let subsets = enumerate_subsets(mask);
        assert_eq!(subsets.len(), 2);
        assert!(subsets.contains(&0));
        assert!(subsets.contains(&mask));
    }

    // ---- Edge case: rook on edge squares ------------------------------------

    #[test]
    fn rook_attacks_a4_empty_board() {
        // Rook on a4 (file=0, rank=3).
        // North: a5, a6, a7, a8 = 4
        // South: a3, a2, a1 = 3
        // East:  b4, c4, d4, e4, f4, g4, h4 = 7
        // West:  none (file A is the leftmost)
        // Total: 14
        let attacks = rook_attacks(Square::A4, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 14);
    }

    #[test]
    fn rook_attacks_h1_empty_board() {
        let attacks = rook_attacks(Square::H1, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 14);
    }

    // ---- Blocker configurations on edges ------------------------------------

    #[test]
    fn rook_attacks_blocker_on_edge_square() {
        // Rook on e4 with a blocker on e8 (edge). Should include e8 in attacks.
        let occupancy = Bitboard::from_square(Square::E8);
        let attacks = rook_attacks(Square::E4, occupancy);
        assert!(attacks.contains(Square::E8));
        assert!(attacks.contains(Square::E5));
        assert!(attacks.contains(Square::E6));
        assert!(attacks.contains(Square::E7));
    }

    #[test]
    fn bishop_attacks_a8_corner() {
        // Bishop on a8 (file=0, rank=7). Only SE diagonal.
        // SE: b7, c6, d5, e4, f3, g2, h1 = 7
        let attacks = bishop_attacks(Square::A8, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 7);
    }

    #[test]
    fn bishop_attacks_h8_corner() {
        // Bishop on h8 (file=7, rank=7). Only SW diagonal.
        // SW: g7, f6, e5, d4, c3, b2, a1 = 7
        let attacks = bishop_attacks(Square::H8, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 7);
    }

    // ---- Multiple blockers on same ray --------------------------------------

    #[test]
    fn rook_stops_at_first_blocker_per_ray() {
        // Rook on e4 with blockers on e5 and e7. Should stop at e5 (nearest).
        let mut occupancy = Bitboard::EMPTY;
        occupancy.set(Square::E5);
        occupancy.set(Square::E7);

        let attacks = rook_attacks(Square::E4, occupancy);
        assert!(attacks.contains(Square::E5));
        assert!(!attacks.contains(Square::E6));
        assert!(!attacks.contains(Square::E7));
        assert!(!attacks.contains(Square::E8));
    }

    #[test]
    fn bishop_stops_at_first_blocker_per_ray() {
        // Bishop on e4 with blockers on f5 and h7. Should stop at f5 (nearest).
        let mut occupancy = Bitboard::EMPTY;
        occupancy.set(Square::F5);
        occupancy.set(Square::H7);

        let attacks = bishop_attacks(Square::E4, occupancy);
        assert!(attacks.contains(Square::F5));
        assert!(!attacks.contains(Square::G6));
        assert!(!attacks.contains(Square::H7));
    }

    // ---- Queen combines rook and bishop correctly ---------------------------

    #[test]
    fn queen_attacks_empty_board_a1() {
        // Queen on a1 = rook (14) + bishop (7) = 21
        let attacks = queen_attacks(Square::A1, Bitboard::EMPTY);
        assert_eq!(attacks.count(), 21);
    }

    #[test]
    fn queen_attacks_with_blockers() {
        // Test queen with some blockers, verify it matches rook | bishop
        let mut occupancy = Bitboard::EMPTY;
        occupancy.set(Square::E7);
        occupancy.set(Square::G6);
        occupancy.set(Square::B4);

        let queen = queen_attacks(Square::E4, occupancy);
        let expected = rook_attacks(Square::E4, occupancy) | bishop_attacks(Square::E4, occupancy);
        assert_eq!(queen, expected);
    }

    // ---- Bishop attack counts on empty board --------------------------------

    #[test]
    fn bishop_corner_squares_attack_7() {
        // All four corner bishops see exactly 7 squares on empty board.
        for &sq in &[Square::A1, Square::A8, Square::H1, Square::H8] {
            let count = bishop_attacks(sq, Bitboard::EMPTY).count();
            assert_eq!(count, 7, "Bishop on {} should attack 7, got {}", sq, count);
        }
    }
}
