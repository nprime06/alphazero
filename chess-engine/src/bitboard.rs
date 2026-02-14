//! Bitboard representation and operations.
//!
//! A bitboard is a 64-bit integer where each bit represents one square on the
//! chess board. This is the fundamental data structure for fast chess engines
//! because it allows set operations on groups of squares (union, intersection,
//! complement) to be performed in a single CPU instruction.
//!
//! # Bit-to-square mapping
//!
//! We use the same LERF (Little-Endian Rank-File) mapping as `Square`:
//! - Bit 0  = A1, Bit 1  = B1, ..., Bit 7  = H1
//! - Bit 8  = A2, Bit 9  = B2, ..., Bit 15 = H2
//! - ...
//! - Bit 56 = A8, Bit 57 = B8, ..., Bit 63 = H8
//!
//! This means `1u64 << square.index()` sets exactly the bit for that square.
//!
//! # Why bitboards?
//!
//! Consider "find all squares attacked by a knight on e4". With a mailbox
//! board representation you'd loop over 8 possible offsets, check bounds for
//! each one. With bitboards you do a single lookup: `KNIGHT_ATTACKS[E4]`
//! returns a u64 with all 8 target squares already set. Intersecting that
//! with "empty or enemy-occupied squares" is just one AND instruction.

use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

use crate::types::Square;

// =============================================================================
// Standalone bitboard utility functions
// =============================================================================

/// Returns a bitboard of squares strictly between `a` and `b` on a line
/// (rank, file, or diagonal). Returns EMPTY if the squares are not aligned.
///
/// Uses magic bitboard attack intersection: the ray from a to b is the
/// intersection of the attacks from a (blocked by b) and from b (blocked by a).
/// This works because each attack set extends along the ray only up to (and
/// including) the blocker, so their intersection gives exactly the squares
/// between them (exclusive of both endpoints).
pub fn between_bb(a: Square, b: Square) -> Bitboard {
    use crate::magic::{bishop_attacks, rook_attacks};

    let bb_a = Bitboard::from_square(a);
    let bb_b = Bitboard::from_square(b);

    // Only use rook intersection when squares share a rank or file.
    // The intersection trick produces spurious corner squares otherwise.
    if a.file() == b.file() || a.rank() == b.rank() {
        return rook_attacks(a, bb_b) & rook_attacks(b, bb_a);
    }

    // Only use bishop intersection when squares share a diagonal.
    let file_diff = (a.file() as i8 - b.file() as i8).abs();
    let rank_diff = (a.rank() as i8 - b.rank() as i8).abs();
    if file_diff == rank_diff {
        return bishop_attacks(a, bb_b) & bishop_attacks(b, bb_a);
    }

    // Squares are not aligned on any line
    Bitboard::EMPTY
}

// =============================================================================
// Bitboard struct
// =============================================================================

/// A set of squares represented as a 64-bit integer.
///
/// Each bit corresponds to one square on the board. Bitboards enable
/// extremely fast set operations: union (OR), intersection (AND),
/// difference (AND NOT), and complement (NOT) are all single instructions.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Bitboard(pub u64);

// =============================================================================
// Constants
// =============================================================================

impl Bitboard {
    /// The empty bitboard (no squares set).
    pub const EMPTY: Bitboard = Bitboard(0);

    /// The full bitboard (all 64 squares set).
    pub const FULL: Bitboard = Bitboard(!0u64);

    // ---- File masks ---------------------------------------------------------
    // A file mask has all 8 squares in that file (column) set.
    //
    // File A is the leftmost column. In LERF mapping, the A-file squares are
    // at indices 0, 8, 16, 24, 32, 40, 48, 56 -- i.e., every 8th bit starting
    // from bit 0. The hex value 0x0101_0101_0101_0101 has bits 0, 8, 16, ...
    // set, which is exactly the A-file. Each subsequent file shifts one bit
    // to the right within each byte.

    pub const FILE_A: Bitboard = Bitboard(0x0101_0101_0101_0101);
    pub const FILE_B: Bitboard = Bitboard(0x0202_0202_0202_0202);
    pub const FILE_C: Bitboard = Bitboard(0x0404_0404_0404_0404);
    pub const FILE_D: Bitboard = Bitboard(0x0808_0808_0808_0808);
    pub const FILE_E: Bitboard = Bitboard(0x1010_1010_1010_1010);
    pub const FILE_F: Bitboard = Bitboard(0x2020_2020_2020_2020);
    pub const FILE_G: Bitboard = Bitboard(0x4040_4040_4040_4040);
    pub const FILE_H: Bitboard = Bitboard(0x8080_8080_8080_8080);

    /// All file masks in an array, indexed by file number (0=A, 7=H).
    /// Useful for looking up the file mask for a given square's file.
    pub const FILES: [Bitboard; 8] = [
        Self::FILE_A,
        Self::FILE_B,
        Self::FILE_C,
        Self::FILE_D,
        Self::FILE_E,
        Self::FILE_F,
        Self::FILE_G,
        Self::FILE_H,
    ];

    // ---- Rank masks ---------------------------------------------------------
    // A rank mask has all 8 squares in that rank (row) set.
    //
    // Rank 1 occupies bits 0..7 (the lowest byte), rank 2 occupies bits 8..15,
    // and so on. So RANK_1 = 0xFF, RANK_2 = 0xFF00, etc.

    pub const RANK_1: Bitboard = Bitboard(0x0000_0000_0000_00FF);
    pub const RANK_2: Bitboard = Bitboard(0x0000_0000_0000_FF00);
    pub const RANK_3: Bitboard = Bitboard(0x0000_0000_00FF_0000);
    pub const RANK_4: Bitboard = Bitboard(0x0000_0000_FF00_0000);
    pub const RANK_5: Bitboard = Bitboard(0x0000_00FF_0000_0000);
    pub const RANK_6: Bitboard = Bitboard(0x0000_FF00_0000_0000);
    pub const RANK_7: Bitboard = Bitboard(0x00FF_0000_0000_0000);
    pub const RANK_8: Bitboard = Bitboard(0xFF00_0000_0000_0000);

    /// All rank masks in an array, indexed by rank number (0=rank 1, 7=rank 8).
    pub const RANKS: [Bitboard; 8] = [
        Self::RANK_1,
        Self::RANK_2,
        Self::RANK_3,
        Self::RANK_4,
        Self::RANK_5,
        Self::RANK_6,
        Self::RANK_7,
        Self::RANK_8,
    ];
}

// =============================================================================
// Core operations
// =============================================================================

impl Bitboard {
    /// Creates a bitboard from a raw u64 value.
    #[inline]
    pub const fn new(bits: u64) -> Self {
        Bitboard(bits)
    }

    /// Creates a bitboard with a single square set.
    #[inline]
    pub const fn from_square(square: Square) -> Self {
        Bitboard(1u64 << square.index())
    }

    /// Returns the underlying u64 value.
    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Sets the bit for the given square (adds it to the set).
    ///
    /// If the square is already set, this is a no-op.
    #[inline]
    pub fn set(&mut self, square: Square) {
        self.0 |= 1u64 << square.index();
    }

    /// Clears the bit for the given square (removes it from the set).
    ///
    /// If the square is already clear, this is a no-op.
    #[inline]
    pub fn clear(&mut self, square: Square) {
        self.0 &= !(1u64 << square.index());
    }

    /// Returns `true` if the given square is set in this bitboard.
    ///
    /// This is the membership test for the set of squares.
    #[inline]
    pub const fn contains(self, square: Square) -> bool {
        (self.0 & (1u64 << square.index())) != 0
    }

    /// Alias for `contains` -- tests whether a square's bit is set.
    #[inline]
    pub const fn test(self, square: Square) -> bool {
        self.contains(square)
    }

    /// Returns the number of squares set in this bitboard (population count).
    ///
    /// Uses the CPU's native `popcnt` instruction on x86, which completes
    /// in a single cycle. This is why bitboards are so fast for counting
    /// pieces, mobility, etc.
    #[inline]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Returns `true` if no squares are set.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns `true` if at least one square is set.
    #[inline]
    pub const fn is_not_empty(self) -> bool {
        self.0 != 0
    }

    /// Removes and returns the least significant set bit as a `Square`.
    ///
    /// This is the standard way to iterate over set bits in a bitboard:
    /// ```ignore
    /// while bb.is_not_empty() {
    ///     let sq = bb.pop_lsb();
    ///     // do something with sq
    /// }
    /// ```
    ///
    /// # How it works
    ///
    /// `trailing_zeros()` finds the index of the lowest set bit. We convert
    /// that to a `Square`, then clear that bit using the classic trick:
    /// `x & (x - 1)` clears the lowest set bit of x.
    ///
    /// For example, if x = 0b1010_1000:
    ///   - x - 1   = 0b1010_0111  (borrows through the lowest set bit)
    ///   - x & (x-1) = 0b1010_0000 (lowest set bit cleared)
    ///
    /// # Panics
    ///
    /// Panics if the bitboard is empty (no bits set). Always check
    /// `is_not_empty()` before calling.
    #[inline]
    pub fn pop_lsb(&mut self) -> Square {
        debug_assert!(self.is_not_empty(), "pop_lsb called on empty bitboard");

        let index = self.0.trailing_zeros() as u8;
        // Clear the lowest set bit. This is faster than `self.clear(square)`
        // because it avoids computing the mask separately.
        self.0 &= self.0 - 1;
        Square::new(index)
    }

    /// Returns the least significant set bit as a `Square` without removing it.
    ///
    /// # Panics
    ///
    /// Panics if the bitboard is empty.
    #[inline]
    pub const fn lsb(self) -> Square {
        debug_assert!(self.is_not_empty(), "lsb called on empty bitboard");
        Square::new(self.0.trailing_zeros() as u8)
    }
}

// =============================================================================
// Operator overloads
// =============================================================================
//
// Bitwise operators map directly to set operations:
//   - AND (&)  = intersection: squares in both sets
//   - OR  (|)  = union: squares in either set
//   - XOR (^)  = symmetric difference: squares in one set but not both
//   - NOT (!)  = complement: squares not in the set
//   - SHL (<<) = shift all squares "north" (toward higher ranks)
//   - SHR (>>) = shift all squares "south" (toward lower ranks)

impl BitAnd for Bitboard {
    type Output = Bitboard;

    #[inline]
    fn bitand(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 & rhs.0)
    }
}

impl BitAndAssign for Bitboard {
    #[inline]
    fn bitand_assign(&mut self, rhs: Bitboard) {
        self.0 &= rhs.0;
    }
}

impl BitOr for Bitboard {
    type Output = Bitboard;

    #[inline]
    fn bitor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 | rhs.0)
    }
}

impl BitOrAssign for Bitboard {
    #[inline]
    fn bitor_assign(&mut self, rhs: Bitboard) {
        self.0 |= rhs.0;
    }
}

impl BitXor for Bitboard {
    type Output = Bitboard;

    #[inline]
    fn bitxor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for Bitboard {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Bitboard) {
        self.0 ^= rhs.0;
    }
}

impl Not for Bitboard {
    type Output = Bitboard;

    #[inline]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

impl Shl<u8> for Bitboard {
    type Output = Bitboard;

    #[inline]
    fn shl(self, rhs: u8) -> Bitboard {
        Bitboard(self.0 << rhs)
    }
}

impl Shr<u8> for Bitboard {
    type Output = Bitboard;

    #[inline]
    fn shr(self, rhs: u8) -> Bitboard {
        Bitboard(self.0 >> rhs)
    }
}

// =============================================================================
// Iterator
// =============================================================================

/// An iterator over the set squares in a `Bitboard`.
///
/// Yields squares from lowest index (A1) to highest (H8). Each iteration
/// pops the least significant set bit, so this destructively consumes a
/// copy of the bitboard.
///
/// # Usage
///
/// ```ignore
/// let bb = Bitboard::RANK_1;
/// for square in bb {
///     println!("{}", square); // a1, b1, c1, ..., h1
/// }
/// ```
pub struct BitboardIterator {
    remaining: Bitboard,
}

impl Iterator for BitboardIterator {
    type Item = Square;

    #[inline]
    fn next(&mut self) -> Option<Square> {
        if self.remaining.is_empty() {
            None
        } else {
            Some(self.remaining.pop_lsb())
        }
    }

    /// Returns the exact number of remaining squares.
    ///
    /// We can compute this in O(1) via popcount, which means `len()`,
    /// `count()`, and `size_hint()` are all exact.
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining.count() as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BitboardIterator {}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = BitboardIterator;

    #[inline]
    fn into_iter(self) -> BitboardIterator {
        BitboardIterator { remaining: self }
    }
}

// =============================================================================
// Display and Debug
// =============================================================================

/// Prints a bitboard as an 8x8 grid.
///
/// The display follows standard chess board orientation:
/// - Rank 8 (the top of the board for White) is printed first (at the top)
/// - Rank 1 is printed last (at the bottom)
/// - Files go left to right: a, b, c, ..., h
///
/// Set bits are shown as `1`, clear bits as `.`. Rank labels are shown on
/// the left, file labels on the bottom.
///
/// Example output for `Bitboard::FILE_E`:
/// ```text
///   8 | . . . . 1 . . .
///   7 | . . . . 1 . . .
///   6 | . . . . 1 . . .
///   5 | . . . . 1 . . .
///   4 | . . . . 1 . . .
///   3 | . . . . 1 . . .
///   2 | . . . . 1 . . .
///   1 | . . . . 1 . . .
///       a b c d e f g h
/// ```
impl fmt::Display for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print from rank 8 (top) down to rank 1 (bottom), matching how
        // humans view a chess board from White's perspective.
        for rank in (0..8u8).rev() {
            write!(f, "  {} |", rank + 1)?;
            for file in 0..8u8 {
                let sq = Square::from_file_rank(file, rank);
                let c = if self.contains(sq) { '1' } else { '.' };
                write!(f, " {}", c)?;
            }
            writeln!(f)?;
        }
        writeln!(f, "      a b c d e f g h")?;
        Ok(())
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bitboard(0x{:016X}, {} bits set)\n{}", self.0, self.count(), self)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;

    // ---- Construction and basic queries ------------------------------------

    #[test]
    fn empty_bitboard_has_no_bits() {
        let bb = Bitboard::EMPTY;
        assert!(bb.is_empty());
        assert!(!bb.is_not_empty());
        assert_eq!(bb.count(), 0);
        assert_eq!(bb.bits(), 0);
    }

    #[test]
    fn full_bitboard_has_all_bits() {
        let bb = Bitboard::FULL;
        assert!(!bb.is_empty());
        assert!(bb.is_not_empty());
        assert_eq!(bb.count(), 64);
        assert_eq!(bb.bits(), !0u64);
    }

    #[test]
    fn from_square_sets_exactly_one_bit() {
        for i in 0..64u8 {
            let sq = Square::new(i);
            let bb = Bitboard::from_square(sq);
            assert_eq!(bb.count(), 1, "from_square should set exactly one bit");
            assert!(bb.contains(sq), "from_square should set the correct bit");
            assert_eq!(bb.bits(), 1u64 << i);
        }
    }

    // ---- Set, clear, contains (test) ----------------------------------------

    #[test]
    fn set_and_contains() {
        let mut bb = Bitboard::EMPTY;
        assert!(!bb.contains(Square::E4));

        bb.set(Square::E4);
        assert!(bb.contains(Square::E4));
        assert!(bb.test(Square::E4)); // alias
        assert_eq!(bb.count(), 1);

        // Setting the same square again is a no-op.
        bb.set(Square::E4);
        assert_eq!(bb.count(), 1);
    }

    #[test]
    fn set_multiple_squares() {
        let mut bb = Bitboard::EMPTY;
        bb.set(Square::A1);
        bb.set(Square::H8);
        bb.set(Square::D4);

        assert!(bb.contains(Square::A1));
        assert!(bb.contains(Square::H8));
        assert!(bb.contains(Square::D4));
        assert!(!bb.contains(Square::E4));
        assert_eq!(bb.count(), 3);
    }

    #[test]
    fn clear_removes_bit() {
        let mut bb = Bitboard::EMPTY;
        bb.set(Square::E4);
        bb.set(Square::D5);
        assert_eq!(bb.count(), 2);

        bb.clear(Square::E4);
        assert!(!bb.contains(Square::E4));
        assert!(bb.contains(Square::D5));
        assert_eq!(bb.count(), 1);

        // Clearing an already-clear bit is a no-op.
        bb.clear(Square::E4);
        assert_eq!(bb.count(), 1);
    }

    #[test]
    fn set_all_64_squares() {
        let mut bb = Bitboard::EMPTY;
        for i in 0..64u8 {
            bb.set(Square::new(i));
        }
        assert_eq!(bb, Bitboard::FULL);
    }

    // ---- pop_lsb ------------------------------------------------------------

    #[test]
    fn pop_lsb_returns_lowest_square() {
        let mut bb = Bitboard::EMPTY;
        bb.set(Square::C3); // index 18
        bb.set(Square::E4); // index 28
        bb.set(Square::A1); // index 0

        // Should return A1 first (lowest index).
        let sq = bb.pop_lsb();
        assert_eq!(sq, Square::A1);
        assert!(!bb.contains(Square::A1));
        assert_eq!(bb.count(), 2);

        // Then C3.
        let sq = bb.pop_lsb();
        assert_eq!(sq, Square::C3);
        assert_eq!(bb.count(), 1);

        // Then E4.
        let sq = bb.pop_lsb();
        assert_eq!(sq, Square::E4);
        assert!(bb.is_empty());
    }

    #[test]
    fn pop_lsb_all_64_squares() {
        let mut bb = Bitboard::FULL;
        let mut squares = Vec::new();
        while bb.is_not_empty() {
            squares.push(bb.pop_lsb());
        }
        // Should yield all 64 squares in index order.
        assert_eq!(squares.len(), 64);
        for (i, sq) in squares.iter().enumerate() {
            assert_eq!(sq.index() as usize, i, "squares should come out in order");
        }
    }

    #[test]
    fn lsb_does_not_modify() {
        let bb = Bitboard::from_square(Square::D4);
        let sq = bb.lsb();
        assert_eq!(sq, Square::D4);
        // bb should be unchanged (lsb is const, takes self by value).
        assert_eq!(bb.count(), 1);
    }

    // ---- Operator overloads -------------------------------------------------

    #[test]
    fn bitand_intersection() {
        let rank1 = Bitboard::RANK_1;
        let file_a = Bitboard::FILE_A;
        // Intersection of rank 1 and file A should be just A1.
        let result = rank1 & file_a;
        assert_eq!(result.count(), 1);
        assert!(result.contains(Square::A1));
    }

    #[test]
    fn bitor_union() {
        let a = Bitboard::from_square(Square::A1);
        let b = Bitboard::from_square(Square::H8);
        let result = a | b;
        assert_eq!(result.count(), 2);
        assert!(result.contains(Square::A1));
        assert!(result.contains(Square::H8));
    }

    #[test]
    fn bitxor_symmetric_difference() {
        let a = Bitboard::RANK_1;
        let b = Bitboard::FILE_A;
        let result = a ^ b;
        // XOR: squares in rank 1 or file A, but not both (not A1).
        assert!(!result.contains(Square::A1)); // in both, so XOR removes it
        assert!(result.contains(Square::B1)); // only in rank 1
        assert!(result.contains(Square::A2)); // only in file A
        assert_eq!(result.count(), 7 + 7); // 7 from rank 1 (minus A1) + 7 from file A (minus A1)
    }

    #[test]
    fn not_complement() {
        let empty = Bitboard::EMPTY;
        assert_eq!(!empty, Bitboard::FULL);
        assert_eq!(!Bitboard::FULL, Bitboard::EMPTY);

        let one_square = Bitboard::from_square(Square::E4);
        let complement = !one_square;
        assert_eq!(complement.count(), 63);
        assert!(!complement.contains(Square::E4));
    }

    #[test]
    fn shl_shift_north() {
        // Shifting rank 1 left by 8 should give rank 2.
        let result = Bitboard::RANK_1 << 8;
        assert_eq!(result, Bitboard::RANK_2);

        // Shifting rank 7 left by 8 should give rank 8.
        let result = Bitboard::RANK_7 << 8;
        assert_eq!(result, Bitboard::RANK_8);
    }

    #[test]
    fn shr_shift_south() {
        // Shifting rank 2 right by 8 should give rank 1.
        let result = Bitboard::RANK_2 >> 8;
        assert_eq!(result, Bitboard::RANK_1);

        // Shifting rank 8 right by 8 should give rank 7.
        let result = Bitboard::RANK_8 >> 8;
        assert_eq!(result, Bitboard::RANK_7);
    }

    #[test]
    fn assign_operators() {
        let mut bb = Bitboard::RANK_1;

        // BitAndAssign
        bb &= Bitboard::FILE_A;
        assert_eq!(bb, Bitboard::from_square(Square::A1));

        // BitOrAssign
        bb |= Bitboard::from_square(Square::H8);
        assert_eq!(bb.count(), 2);

        // BitXorAssign: XOR with itself clears all bits.
        let copy = bb;
        bb ^= copy;
        assert!(bb.is_empty());
    }

    // ---- File and rank constants -------------------------------------------

    #[test]
    fn file_constants_have_8_bits_each() {
        for (i, file_bb) in Bitboard::FILES.iter().enumerate() {
            assert_eq!(
                file_bb.count(),
                8,
                "FILE_{} should have 8 bits set",
                (b'A' + i as u8) as char
            );
        }
    }

    #[test]
    fn file_constants_are_disjoint() {
        // No two file masks should share a square.
        for i in 0..8 {
            for j in (i + 1)..8 {
                let overlap = Bitboard::FILES[i] & Bitboard::FILES[j];
                assert!(
                    overlap.is_empty(),
                    "FILE_{} and FILE_{} should not overlap",
                    (b'A' + i as u8) as char,
                    (b'A' + j as u8) as char
                );
            }
        }
    }

    #[test]
    fn file_constants_cover_all_squares() {
        let mut combined = Bitboard::EMPTY;
        for file_bb in &Bitboard::FILES {
            combined |= *file_bb;
        }
        assert_eq!(combined, Bitboard::FULL);
    }

    #[test]
    fn file_a_contains_correct_squares() {
        // File A should contain A1, A2, ..., A8.
        let file_a = Bitboard::FILE_A;
        for rank in 0..8u8 {
            let sq = Square::from_file_rank(0, rank);
            assert!(file_a.contains(sq), "FILE_A should contain {}", sq);
        }
        // And no other squares.
        for file in 1..8u8 {
            for rank in 0..8u8 {
                let sq = Square::from_file_rank(file, rank);
                assert!(!file_a.contains(sq), "FILE_A should not contain {}", sq);
            }
        }
    }

    #[test]
    fn rank_constants_have_8_bits_each() {
        for (i, rank_bb) in Bitboard::RANKS.iter().enumerate() {
            assert_eq!(
                rank_bb.count(),
                8,
                "RANK_{} should have 8 bits set",
                i + 1
            );
        }
    }

    #[test]
    fn rank_constants_are_disjoint() {
        for i in 0..8 {
            for j in (i + 1)..8 {
                let overlap = Bitboard::RANKS[i] & Bitboard::RANKS[j];
                assert!(
                    overlap.is_empty(),
                    "RANK_{} and RANK_{} should not overlap",
                    i + 1,
                    j + 1
                );
            }
        }
    }

    #[test]
    fn rank_constants_cover_all_squares() {
        let mut combined = Bitboard::EMPTY;
        for rank_bb in &Bitboard::RANKS {
            combined |= *rank_bb;
        }
        assert_eq!(combined, Bitboard::FULL);
    }

    #[test]
    fn rank_1_contains_correct_squares() {
        let rank1 = Bitboard::RANK_1;
        for file in 0..8u8 {
            let sq = Square::from_file_rank(file, 0);
            assert!(rank1.contains(sq), "RANK_1 should contain {}", sq);
        }
        for rank in 1..8u8 {
            for file in 0..8u8 {
                let sq = Square::from_file_rank(file, rank);
                assert!(!rank1.contains(sq), "RANK_1 should not contain {}", sq);
            }
        }
    }

    #[test]
    fn file_rank_intersection_is_single_square() {
        // The intersection of any file and any rank should be exactly one square.
        for file in 0..8usize {
            for rank in 0..8usize {
                let intersection = Bitboard::FILES[file] & Bitboard::RANKS[rank];
                assert_eq!(
                    intersection.count(),
                    1,
                    "FILE x RANK intersection should be one square"
                );
                let expected = Square::from_file_rank(file as u8, rank as u8);
                assert!(intersection.contains(expected));
            }
        }
    }

    // ---- Iterator -----------------------------------------------------------

    #[test]
    fn iterator_yields_squares_in_order() {
        let mut bb = Bitboard::EMPTY;
        bb.set(Square::H8); // index 63
        bb.set(Square::A1); // index 0
        bb.set(Square::E4); // index 28

        let squares: Vec<Square> = bb.into_iter().collect();
        assert_eq!(squares.len(), 3);
        assert_eq!(squares[0], Square::A1);
        assert_eq!(squares[1], Square::E4);
        assert_eq!(squares[2], Square::H8);
    }

    #[test]
    fn iterator_empty_bitboard() {
        let bb = Bitboard::EMPTY;
        let squares: Vec<Square> = bb.into_iter().collect();
        assert!(squares.is_empty());
    }

    #[test]
    fn iterator_full_bitboard() {
        let bb = Bitboard::FULL;
        let squares: Vec<Square> = bb.into_iter().collect();
        assert_eq!(squares.len(), 64);
        for (i, sq) in squares.iter().enumerate() {
            assert_eq!(sq.index() as usize, i);
        }
    }

    #[test]
    fn iterator_exact_size() {
        let mut bb = Bitboard::EMPTY;
        bb.set(Square::A1);
        bb.set(Square::B2);
        bb.set(Square::C3);

        let iter = bb.into_iter();
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn for_loop_over_bitboard() {
        // Verify the for-loop syntax works (IntoIterator).
        let rank1 = Bitboard::RANK_1;
        let mut count = 0;
        for sq in rank1 {
            assert_eq!(sq.rank(), 0, "all squares should be on rank 1");
            count += 1;
        }
        assert_eq!(count, 8);
    }

    // ---- Display/Debug ------------------------------------------------------

    #[test]
    fn display_empty_board() {
        let bb = Bitboard::EMPTY;
        let s = format!("{}", bb);
        // No set bits, so no " 1 " (with spaces) in the board area.
        // Rank labels like "1 |" will contain '1', so we check the board cells.
        for rank in 1..=8 {
            let label = format!("{} |", rank);
            let line = s.lines().find(|l| l.contains(&label)).unwrap();
            let after_bar = line.split('|').nth(1).unwrap();
            assert!(!after_bar.contains('1'), "rank {rank} should have no set bits");
        }
        // Should contain rank labels.
        assert!(s.contains("8 |"));
        assert!(s.contains("1 |"));
        // Should contain file labels.
        assert!(s.contains("a b c d e f g h"));
    }

    #[test]
    fn display_single_square() {
        let bb = Bitboard::from_square(Square::E4);
        let s = format!("{}", bb);
        // E4 is on rank 4, file e.
        // Count '1's only in board cells (after the '|' on each line).
        let board_ones: usize = s
            .lines()
            .filter_map(|line| line.split('|').nth(1))
            .map(|cells| cells.chars().filter(|&c| c == '1').count())
            .sum();
        assert_eq!(board_ones, 1, "should have exactly one set bit displayed");
    }

    #[test]
    fn display_rank_8_at_top() {
        let bb = Bitboard::EMPTY;
        let s = format!("{}", bb);
        let lines: Vec<&str> = s.lines().collect();
        // First line should be rank 8, last board line should be rank 1.
        assert!(lines[0].contains("8 |"), "rank 8 should be at the top");
        assert!(lines[7].contains("1 |"), "rank 1 should be at the bottom");
    }

    #[test]
    fn display_file_a_shows_leftmost_column() {
        let bb = Bitboard::FILE_A;
        let s = format!("{}", bb);
        // Every rank line should have '1' as the first square character.
        for line in s.lines().take(8) {
            // Format is "  R | x x x x x x x x" where x is '.' or '1'
            // The first square char comes after "  R | "
            let after_bar = line.split('|').nth(1).unwrap();
            let first_char = after_bar.trim_start().chars().next().unwrap();
            assert_eq!(first_char, '1', "FILE_A should set the leftmost column: {}", line);
        }
    }

    // ---- Edge cases and properties ------------------------------------------

    #[test]
    fn xor_with_self_is_empty() {
        let bb = Bitboard::RANK_4;
        assert_eq!(bb ^ bb, Bitboard::EMPTY);
    }

    #[test]
    fn and_with_complement_is_empty() {
        let bb = Bitboard::FILE_E;
        assert_eq!(bb & !bb, Bitboard::EMPTY);
    }

    #[test]
    fn or_with_complement_is_full() {
        let bb = Bitboard::FILE_E;
        assert_eq!(bb | !bb, Bitboard::FULL);
    }

    #[test]
    fn double_complement_is_identity() {
        let bb = Bitboard::new(0xDEAD_BEEF_CAFE_BABE);
        assert_eq!(!!bb, bb);
    }

    #[test]
    fn count_rank_1() {
        assert_eq!(Bitboard::RANK_1.count(), 8);
    }

    #[test]
    fn count_file_a() {
        assert_eq!(Bitboard::FILE_A.count(), 8);
    }

    #[test]
    fn shift_left_by_1_doubles_value_if_no_overflow() {
        // Shifting a single bit on A1 left by 1 gives B1.
        let a1 = Bitboard::from_square(Square::A1);
        let result = a1 << 1;
        assert!(result.contains(Square::B1));
        assert_eq!(result.count(), 1);
    }

    #[test]
    fn pop_lsb_single_bit() {
        let mut bb = Bitboard::from_square(Square::D5);
        let sq = bb.pop_lsb();
        assert_eq!(sq, Square::D5);
        assert!(bb.is_empty());
    }

    #[test]
    fn bitboard_is_copy() {
        // Verify Bitboard is Copy (this won't compile if it isn't).
        let bb = Bitboard::RANK_1;
        let bb2 = bb;
        assert_eq!(bb, bb2); // bb wasn't moved-from
    }
}
