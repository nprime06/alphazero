//! Precomputed attack tables for non-sliding pieces (knights, kings, pawns).
//!
//! Non-sliding pieces have attack patterns that depend only on the square they
//! occupy -- they are not affected by other pieces on the board. This means we
//! can precompute all possible attacks for each square once and store them in
//! lookup tables. At runtime, finding the attacks for a knight on e4 is a
//! single array lookup: `KNIGHT_ATTACKS[e4.index()]`.
//!
//! # Why precompute?
//!
//! Without precomputation, calculating knight attacks from a square requires
//! checking 8 possible offsets and handling edge-wrapping for each one. That
//! is 8 branches per call. With a lookup table, it is zero branches -- just
//! an array index and a load from memory (which will almost certainly be in
//! L1 cache during move generation since the tables are small).
//!
//! # Edge wrapping problem
//!
//! On a bitboard, "moving left" means shifting bits to a lower index and
//! "moving right" means shifting to a higher index. But the board wraps:
//! square H1 (index 7) is adjacent to A2 (index 8) in the bit layout, even
//! though they are on opposite sides of the board. We must mask out squares
//! that would wrap around file edges after shifting.
//!
//! For example, a knight on H1 could appear to "attack" squares on the A and
//! B files via wrapping. We prevent this by AND-ing with masks that exclude
//! the files the piece cannot legally reach.
//!
//! # Pawn attacks vs pawn moves
//!
//! This module only computes pawn *attacks* (diagonal captures), not pawn
//! *pushes* (forward moves). Pawn pushes depend on occupancy (a pawn can't
//! push through another piece), so they cannot be precomputed as simply.
//! Pawn attacks are occupancy-independent -- a pawn always threatens the
//! same diagonal squares regardless of what is there.

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::magic::{bishop_attacks, rook_attacks};
use crate::types::{Color, Piece, Square, NUM_SQUARES};

// =============================================================================
// File combination masks for edge detection
// =============================================================================
//
// When a piece near the edge of the board makes a move that includes a
// horizontal component, we need to ensure it does not wrap around to the
// other side. For moves with 1 square of horizontal movement, masking out
// FILE_A (for leftward moves) or FILE_H (for rightward moves) suffices.
//
// For knights, which can move 2 squares horizontally, we need wider masks:
// FILE_AB (files A and B) and FILE_GH (files G and H). A knight on file B
// trying to go 2 squares left would wrap to file H -- the FILE_AB mask
// catches this.

/// Files A and B combined. Used to mask out knights that would wrap 2 files
/// to the left (westward). A knight on the B file cannot go 2 files left.
const FILE_AB: u64 = Bitboard::FILE_A.0 | Bitboard::FILE_B.0;

/// Files G and H combined. Used to mask out knights that would wrap 2 files
/// to the right (eastward). A knight on the G file cannot go 2 files right.
const FILE_GH: u64 = Bitboard::FILE_G.0 | Bitboard::FILE_H.0;

// Single-file masks reexported as raw u64 for use in const fn (where we
// cannot use Bitboard operator overloads since those are trait methods).
const FILE_A: u64 = Bitboard::FILE_A.0;
const FILE_H: u64 = Bitboard::FILE_H.0;

// =============================================================================
// Knight attack generation
// =============================================================================

/// Computes the knight attack bitboard for a single square.
///
/// A knight moves in an "L" shape: 2 squares in one direction and 1 in a
/// perpendicular direction. From a central square this gives 8 possible
/// target squares:
///
/// ```text
///     . x . x .
///     x . . . x
///     . . N . .
///     x . . . x
///     . x . x .
/// ```
///
/// The 8 moves expressed as (file_delta, rank_delta) are:
///   (+1, +2), (+2, +1), (+2, -1), (+1, -2),
///   (-1, -2), (-2, -1), (-2, +1), (-1, +2)
///
/// In bitboard terms, these translate to bit shifts:
///   - North-northeast (+1 file, +2 ranks) = shift left by 17, mask out FILE_A
///   - East-northeast (+2 files, +1 rank)  = shift left by 10, mask out FILE_AB
///   - East-southeast (+2 files, -1 rank)  = shift right by 6, mask out FILE_AB
///   - South-southeast (+1 file, -2 ranks) = shift right by 15, mask out FILE_A
///   - South-southwest (-1 file, -2 ranks) = shift right by 17, mask out FILE_H
///   - West-southwest (-2 files, -1 rank)  = shift right by 10, mask out FILE_GH
///   - West-northwest (-2 files, +1 rank)  = shift left by 6,  mask out FILE_GH
///   - North-northwest (-1 file, +2 ranks) = shift left by 15, mask out FILE_H
///
/// Each shift is masked with the appropriate file exclusion to prevent wrapping.
const fn generate_knight_attacks(sq: usize) -> Bitboard {
    let bb = 1u64 << sq;

    let attacks =
        // Moves going "north" (higher ranks): shift left
        ((bb << 17) & !FILE_A)  // NNE: up 2, right 1 -- can't land on A file (wrapped from H)
      | ((bb << 10) & !FILE_AB) // ENE: up 1, right 2 -- can't land on A or B file
      | ((bb >> 6)  & !FILE_AB) // ESE: down 1, right 2 -- can't land on A or B file
      | ((bb >> 15) & !FILE_A)  // SSE: down 2, right 1 -- can't land on A file
        // Moves going "south" (lower ranks): shift right
      | ((bb >> 17) & !FILE_H)  // SSW: down 2, left 1 -- can't land on H file (wrapped from A)
      | ((bb >> 10) & !FILE_GH) // WSW: down 1, left 2 -- can't land on G or H file
      | ((bb << 6)  & !FILE_GH) // WNW: up 1, left 2 -- can't land on G or H file
      | ((bb << 15) & !FILE_H); // NNW: up 2, left 1 -- can't land on H file

    Bitboard(attacks)
}

/// Precomputed knight attack table. `KNIGHT_ATTACKS[sq]` gives the bitboard
/// of all squares attacked by a knight on square `sq`.
const KNIGHT_ATTACKS: [Bitboard; NUM_SQUARES] = {
    let mut table = [Bitboard::EMPTY; NUM_SQUARES];
    let mut sq = 0;
    while sq < NUM_SQUARES {
        table[sq] = generate_knight_attacks(sq);
        sq += 1;
    }
    table
};

/// Returns the precomputed attack bitboard for a knight on the given square.
///
/// This is a simple array lookup -- O(1) with no computation.
///
/// # Example
///
/// ```ignore
/// let attacks = knight_attacks(Square::E4);
/// assert_eq!(attacks.count(), 8); // center square: all 8 moves available
/// ```
#[inline]
pub fn knight_attacks(square: Square) -> Bitboard {
    KNIGHT_ATTACKS[square.index() as usize]
}

// =============================================================================
// King attack generation
// =============================================================================

/// Computes the king attack bitboard for a single square.
///
/// A king can move one square in any of 8 directions:
///
/// ```text
///     x x x
///     x K x
///     x x x
/// ```
///
/// The 8 moves as bit shifts:
///   - North (+8):     shift left by 8, no file mask needed
///   - South (-8):     shift right by 8, no file mask needed
///   - East (+1):      shift left by 1, mask out FILE_A (prevents H->A wrap)
///   - West (-1):      shift right by 1, mask out FILE_H (prevents A->H wrap)
///   - Northeast (+9): shift left by 9, mask out FILE_A
///   - Northwest (+7): shift left by 7, mask out FILE_H
///   - Southeast (-7): shift right by 7, mask out FILE_A
///   - Southwest (-9): shift right by 9, mask out FILE_H
///
/// Pure vertical moves (north/south) never wrap files, so no mask is needed.
/// Any move with an eastward component could wrap from H-file to A-file, so
/// we mask out FILE_A on the result. Any move with a westward component could
/// wrap from A-file to H-file, so we mask out FILE_H.
const fn generate_king_attacks(sq: usize) -> Bitboard {
    let bb = 1u64 << sq;

    let attacks =
        // Pure vertical (no wrapping possible)
         (bb << 8)               // North
       | (bb >> 8)               // South
        // Eastward moves (mask out FILE_A to prevent wrap from H to A)
       | ((bb << 1) & !FILE_A)  // East
       | ((bb << 9) & !FILE_A)  // Northeast
       | ((bb >> 7) & !FILE_A)  // Southeast
        // Westward moves (mask out FILE_H to prevent wrap from A to H)
       | ((bb >> 1) & !FILE_H)  // West
       | ((bb << 7) & !FILE_H)  // Northwest
       | ((bb >> 9) & !FILE_H); // Southwest

    Bitboard(attacks)
}

/// Precomputed king attack table. `KING_ATTACKS[sq]` gives the bitboard
/// of all squares attacked by a king on square `sq`.
const KING_ATTACKS: [Bitboard; NUM_SQUARES] = {
    let mut table = [Bitboard::EMPTY; NUM_SQUARES];
    let mut sq = 0;
    while sq < NUM_SQUARES {
        table[sq] = generate_king_attacks(sq);
        sq += 1;
    }
    table
};

/// Returns the precomputed attack bitboard for a king on the given square.
///
/// Note: this does NOT include castling moves, which are special moves
/// handled separately by the move generator.
#[inline]
pub fn king_attacks(square: Square) -> Bitboard {
    KING_ATTACKS[square.index() as usize]
}

// =============================================================================
// Pawn attack generation
// =============================================================================

/// Computes the pawn attack bitboard for a single square and color.
///
/// Pawns attack diagonally forward (one square):
/// - White pawns attack northeast (+9) and northwest (+7)
/// - Black pawns attack southeast (-7) and southwest (-9)
///
/// Pawns on the back rank of the opposing side (rank 8 for White, rank 1 for
/// Black) have no attacks, since they would have already promoted. Pawns on
/// their own back rank also technically should not exist in a legal game, but
/// we handle them gracefully by returning empty if the shift would go off the
/// board.
///
/// Edge masking:
/// - Northeast/Southeast (eastward component): mask out FILE_A
/// - Northwest/Southwest (westward component): mask out FILE_H
const fn generate_pawn_attacks(sq: usize, is_white: bool) -> Bitboard {
    let bb = 1u64 << sq;

    let attacks = if is_white {
        // White pawns attack toward higher ranks (northeast and northwest)
          ((bb << 9) & !FILE_A)  // Northeast: up-right, can't wrap to A file
        | ((bb << 7) & !FILE_H)  // Northwest: up-left, can't wrap to H file
    } else {
        // Black pawns attack toward lower ranks (southeast and southwest)
          ((bb >> 7) & !FILE_A)  // Southeast: down-right, can't wrap to A file
        | ((bb >> 9) & !FILE_H)  // Southwest: down-left, can't wrap to H file
    };

    Bitboard(attacks)
}

/// Precomputed pawn attack table. `PAWN_ATTACKS[color][sq]` gives the bitboard
/// of all squares attacked by a pawn of the given color on square `sq`.
///
/// Indexed as `PAWN_ATTACKS[Color::White.index()][square.index()]` for white
/// pawn attacks, etc.
const PAWN_ATTACKS: [[Bitboard; NUM_SQUARES]; 2] = {
    let mut table = [[Bitboard::EMPTY; NUM_SQUARES]; 2];
    let mut sq = 0;
    while sq < NUM_SQUARES {
        table[0][sq] = generate_pawn_attacks(sq, true);  // White
        table[1][sq] = generate_pawn_attacks(sq, false); // Black
        sq += 1;
    }
    table
};

/// Returns the precomputed attack bitboard for a pawn of the given color
/// on the given square.
///
/// This returns the squares the pawn *threatens* (diagonal captures), not
/// the squares it can *move to* (forward pushes). A pawn on a2 with color
/// White threatens b3 (and nothing else, since it cannot capture to the left
/// from the A file).
#[inline]
pub fn pawn_attacks(color: Color, square: Square) -> Bitboard {
    PAWN_ATTACKS[color.index()][square.index() as usize]
}

// =============================================================================
// Square attack detection
// =============================================================================

/// Determines whether a given square is attacked by any piece of the specified color.
///
/// Uses the "super-piece" approach: from the target square, we look outward using
/// each piece type's attack pattern. If we find an enemy piece of the matching type,
/// the square is attacked.
///
/// For example, to check if a knight attacks the square, we compute the knight attack
/// bitboard *from* that square and intersect it with the opponent's knight bitboard.
/// If the intersection is non-empty, there is an enemy knight that can reach the square.
///
/// This works because piece attacks are symmetric: if a knight on B can reach A, then
/// a knight on A can reach B. The same applies to all piece types (with a color flip
/// for pawns, since pawn attacks are directional).
///
/// # Arguments
///
/// * `board` - The board state to check.
/// * `square` - The target square to test.
/// * `by_color` - The color of the attacking side.
///
/// # Returns
///
/// `true` if any piece of `by_color` attacks the given square.
pub fn is_square_attacked(board: &Board, square: Square, by_color: Color) -> bool {
    let occupancy = board.all_pieces();

    // Pawn attacks: look from the target square with the *defending* side's pawn
    // attack pattern. If we find an enemy pawn there, the square is attacked.
    // We use by_color.flip() because pawn attacks are directional: a white pawn
    // attacks northeast/northwest, so to find white pawns that attack a square,
    // we look southeast/southwest from that square (i.e., black's attack pattern).
    if (pawn_attacks(by_color.flip(), square)
        & board.piece_bitboard(by_color, Piece::Pawn))
    .is_not_empty()
    {
        return true;
    }

    // Knight attacks (symmetric)
    if (knight_attacks(square) & board.piece_bitboard(by_color, Piece::Knight)).is_not_empty() {
        return true;
    }

    // King attacks (symmetric)
    if (king_attacks(square) & board.piece_bitboard(by_color, Piece::King)).is_not_empty() {
        return true;
    }

    // Diagonal attacks: bishop or queen
    let diagonal_attackers =
        board.piece_bitboard(by_color, Piece::Bishop) | board.piece_bitboard(by_color, Piece::Queen);
    if (bishop_attacks(square, occupancy) & diagonal_attackers).is_not_empty() {
        return true;
    }

    // Straight-line attacks: rook or queen
    let straight_attackers =
        board.piece_bitboard(by_color, Piece::Rook) | board.piece_bitboard(by_color, Piece::Queen);
    if (rook_attacks(square, occupancy) & straight_attackers).is_not_empty() {
        return true;
    }

    false
}

/// Determines whether a given square is attacked by any piece of the specified color,
/// while ignoring a specific square in the occupancy (treating it as empty).
///
/// This is essential for king move legality: when a king moves, we need to check if
/// the destination is attacked, but the king's current square should not block sliding
/// attack rays (otherwise the king could appear to "hide behind itself").
///
/// # Arguments
///
/// * `board` - The board state to check.
/// * `square` - The target square to test.
/// * `by_color` - The color of the attacking side.
/// * `ignore` - A square to treat as empty in the occupancy (typically the king's current square).
pub fn is_square_attacked_ignoring(
    board: &Board,
    square: Square,
    by_color: Color,
    ignore: Square,
) -> bool {
    let occupancy = board.all_pieces() & !Bitboard::from_square(ignore);

    // Pawn attacks (not affected by occupancy)
    if (pawn_attacks(by_color.flip(), square)
        & board.piece_bitboard(by_color, Piece::Pawn))
    .is_not_empty()
    {
        return true;
    }

    // Knight attacks (not affected by occupancy)
    if (knight_attacks(square) & board.piece_bitboard(by_color, Piece::Knight)).is_not_empty() {
        return true;
    }

    // King attacks (not affected by occupancy)
    if (king_attacks(square) & board.piece_bitboard(by_color, Piece::King)).is_not_empty() {
        return true;
    }

    // Diagonal attacks: bishop or queen (uses modified occupancy)
    let diagonal_attackers =
        board.piece_bitboard(by_color, Piece::Bishop) | board.piece_bitboard(by_color, Piece::Queen);
    if (bishop_attacks(square, occupancy) & diagonal_attackers).is_not_empty() {
        return true;
    }

    // Straight-line attacks: rook or queen (uses modified occupancy)
    let straight_attackers =
        board.piece_bitboard(by_color, Piece::Rook) | board.piece_bitboard(by_color, Piece::Queen);
    if (rook_attacks(square, occupancy) & straight_attackers).is_not_empty() {
        return true;
    }

    false
}

/// Returns `true` if the king of the given color is currently in check.
///
/// This finds the king's square and checks whether any enemy piece attacks it.
///
/// # Panics
///
/// In debug mode, panics if the given color has no king on the board
/// (which indicates a malformed position).
pub fn is_in_check(board: &Board, color: Color) -> bool {
    let king_bb = board.piece_bitboard(color, Piece::King);
    debug_assert!(
        king_bb.is_not_empty(),
        "is_in_check: no {:?} king on the board",
        color
    );

    let king_square = king_bb.lsb();
    is_square_attacked(board, king_square, color.flip())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::fen;
    use crate::types::Square;

    // ---- Knight attack tests ------------------------------------------------

    #[test]
    fn knight_attacks_center_e4_has_8_targets() {
        // A knight on e4 (center of the board) can reach all 8 L-shaped
        // destinations with no edge clipping.
        let attacks = knight_attacks(Square::E4);
        assert_eq!(
            attacks.count(), 8,
            "Knight on center square e4 should attack exactly 8 squares"
        );

        // Verify the specific target squares:
        //   From e4 (file=4, rank=3):
        //   (+1,+2) = f6, (+2,+1) = g5, (+2,-1) = g3, (+1,-2) = f2,
        //   (-1,-2) = d2, (-2,-1) = c3, (-2,+1) = c5, (-1,+2) = d6
        assert!(attacks.contains(Square::F6));
        assert!(attacks.contains(Square::G5));
        assert!(attacks.contains(Square::G3));
        assert!(attacks.contains(Square::F2));
        assert!(attacks.contains(Square::D2));
        assert!(attacks.contains(Square::C3));
        assert!(attacks.contains(Square::C5));
        assert!(attacks.contains(Square::D6));
    }

    #[test]
    fn knight_attacks_corner_a1_has_2_targets() {
        // A knight on a1 is maximally constrained: it can only reach b3 and c2.
        let attacks = knight_attacks(Square::A1);
        assert_eq!(
            attacks.count(), 2,
            "Knight on corner a1 should attack exactly 2 squares"
        );
        assert!(attacks.contains(Square::B3));
        assert!(attacks.contains(Square::C2));
    }

    #[test]
    fn knight_attacks_edge_b1_has_3_targets() {
        // b1 is on the bottom edge but not a corner. A knight can reach:
        // a3, c3, d2.
        let attacks = knight_attacks(Square::B1);
        assert_eq!(
            attacks.count(), 3,
            "Knight on edge b1 should attack exactly 3 squares"
        );
        assert!(attacks.contains(Square::A3));
        assert!(attacks.contains(Square::C3));
        assert!(attacks.contains(Square::D2));
    }

    #[test]
    fn knight_attacks_corner_h8_specific_squares() {
        // h8: file=7, rank=7. Possible targets:
        //   (+1,+2): file=8 -- off board
        //   (+2,+1): file=9 -- off board
        //   (+2,-1): file=9 -- off board
        //   (+1,-2): file=8 -- off board
        //   (-1,-2): g6 (file=6, rank=5) -- valid
        //   (-2,-1): f7 (file=5, rank=6) -- valid
        //   (-2,+1): f9 -- off board (rank 9)
        //   (-1,+2): g10 -- off board
        // So only g6 and f7.
        let attacks = knight_attacks(Square::H8);
        assert_eq!(attacks.count(), 2);
        assert!(attacks.contains(Square::G6));
        assert!(attacks.contains(Square::F7));
    }

    #[test]
    fn knight_attacks_symmetry() {
        // If square B is in the knight attacks from square A, then square A
        // must also be in the knight attacks from square B. This is because
        // knight moves are reversible: the L-shape from A to B is the same
        // L-shape from B to A (just in the opposite direction).
        for sq_idx in 0..64u8 {
            let sq = Square::new(sq_idx);
            let attacks = knight_attacks(sq);

            for target in attacks {
                let reverse_attacks = knight_attacks(target);
                assert!(
                    reverse_attacks.contains(sq),
                    "Symmetry violation: knight on {} attacks {}, but knight on {} does not attack {}",
                    sq, target, target, sq
                );
            }
        }
    }

    #[test]
    fn knight_attacks_all_squares_have_2_to_8_targets() {
        // A knight always has between 2 (corner) and 8 (center) attacks.
        for sq_idx in 0..64u8 {
            let sq = Square::new(sq_idx);
            let count = knight_attacks(sq).count();
            assert!(
                (2..=8).contains(&count),
                "Knight on {} has {} attacks (expected 2-8)",
                sq, count
            );
        }
    }

    #[test]
    fn knight_attacks_corner_h1() {
        let attacks = knight_attacks(Square::H1);
        assert_eq!(attacks.count(), 2);
        assert!(attacks.contains(Square::G3));
        assert!(attacks.contains(Square::F2));
    }

    #[test]
    fn knight_attacks_corner_a8() {
        let attacks = knight_attacks(Square::A8);
        assert_eq!(attacks.count(), 2);
        assert!(attacks.contains(Square::B6));
        assert!(attacks.contains(Square::C7));
    }

    #[test]
    fn knight_attacks_d4_center() {
        // d4 is also a central square, should have 8 attacks.
        let attacks = knight_attacks(Square::D4);
        assert_eq!(attacks.count(), 8);
        assert!(attacks.contains(Square::C6));
        assert!(attacks.contains(Square::E6));
        assert!(attacks.contains(Square::F5));
        assert!(attacks.contains(Square::F3));
        assert!(attacks.contains(Square::E2));
        assert!(attacks.contains(Square::C2));
        assert!(attacks.contains(Square::B3));
        assert!(attacks.contains(Square::B5));
    }

    // ---- King attack tests --------------------------------------------------

    #[test]
    fn king_attacks_center_e4_has_8_targets() {
        let attacks = king_attacks(Square::E4);
        assert_eq!(
            attacks.count(), 8,
            "King on center square e4 should attack exactly 8 squares"
        );

        // All 8 neighbors of e4 (file=4, rank=3):
        assert!(attacks.contains(Square::D3)); // Southwest
        assert!(attacks.contains(Square::E3)); // South
        assert!(attacks.contains(Square::F3)); // Southeast
        assert!(attacks.contains(Square::D4)); // West
        assert!(attacks.contains(Square::F4)); // East
        assert!(attacks.contains(Square::D5)); // Northwest
        assert!(attacks.contains(Square::E5)); // North
        assert!(attacks.contains(Square::F5)); // Northeast
    }

    #[test]
    fn king_attacks_corner_a1_has_3_targets() {
        // a1 is in the bottom-left corner. Only 3 neighbors: b1, a2, b2.
        let attacks = king_attacks(Square::A1);
        assert_eq!(
            attacks.count(), 3,
            "King on corner a1 should attack exactly 3 squares"
        );
        assert!(attacks.contains(Square::B1));
        assert!(attacks.contains(Square::A2));
        assert!(attacks.contains(Square::B2));
    }

    #[test]
    fn king_attacks_e1_has_5_targets() {
        // e1 is on the bottom edge (rank 1) but not on a file edge.
        // It has 5 neighbors: d1, f1, d2, e2, f2.
        let attacks = king_attacks(Square::E1);
        assert_eq!(
            attacks.count(), 5,
            "King on e1 (bottom edge, center file) should attack exactly 5 squares"
        );
        assert!(attacks.contains(Square::D1));
        assert!(attacks.contains(Square::F1));
        assert!(attacks.contains(Square::D2));
        assert!(attacks.contains(Square::E2));
        assert!(attacks.contains(Square::F2));
    }

    #[test]
    fn king_attacks_h4_has_5_targets() {
        // h4 is on the right edge (file H). 5 neighbors.
        let attacks = king_attacks(Square::H4);
        assert_eq!(attacks.count(), 5);
        assert!(attacks.contains(Square::G3));
        assert!(attacks.contains(Square::H3));
        assert!(attacks.contains(Square::G4));
        assert!(attacks.contains(Square::G5));
        assert!(attacks.contains(Square::H5));
    }

    #[test]
    fn king_attacks_all_squares_have_3_to_8_targets() {
        // A king always has between 3 (corner) and 8 (center) attacks.
        for sq_idx in 0..64u8 {
            let sq = Square::new(sq_idx);
            let count = king_attacks(sq).count();
            assert!(
                (3..=8).contains(&count),
                "King on {} has {} attacks (expected 3-8)",
                sq, count
            );
        }
    }

    #[test]
    fn king_attacks_corner_h8() {
        let attacks = king_attacks(Square::H8);
        assert_eq!(attacks.count(), 3);
        assert!(attacks.contains(Square::G8));
        assert!(attacks.contains(Square::G7));
        assert!(attacks.contains(Square::H7));
    }

    #[test]
    fn king_attacks_a4_edge() {
        // a4 is on the left edge (file A). 5 neighbors.
        let attacks = king_attacks(Square::A4);
        assert_eq!(attacks.count(), 5);
        assert!(attacks.contains(Square::A3));
        assert!(attacks.contains(Square::B3));
        assert!(attacks.contains(Square::B4));
        assert!(attacks.contains(Square::A5));
        assert!(attacks.contains(Square::B5));
    }

    // ---- Pawn attack tests --------------------------------------------------

    #[test]
    fn pawn_attacks_white_e4() {
        // White pawn on e4 attacks d5 and f5.
        let attacks = pawn_attacks(Color::White, Square::E4);
        assert_eq!(attacks.count(), 2);
        assert!(attacks.contains(Square::D5));
        assert!(attacks.contains(Square::F5));
    }

    #[test]
    fn pawn_attacks_white_a2_only_b3() {
        // White pawn on a2 (A file) can only attack b3. There is no square
        // to the left of the A file, so northwest is impossible.
        let attacks = pawn_attacks(Color::White, Square::A2);
        assert_eq!(
            attacks.count(), 1,
            "White pawn on a2 should only attack b3 (can't go left from A file)"
        );
        assert!(attacks.contains(Square::B3));
    }

    #[test]
    fn pawn_attacks_white_h2_only_g3() {
        // White pawn on h2 (H file) can only attack g3.
        let attacks = pawn_attacks(Color::White, Square::H2);
        assert_eq!(attacks.count(), 1);
        assert!(attacks.contains(Square::G3));
    }

    #[test]
    fn pawn_attacks_black_e5() {
        // Black pawn on e5 attacks d4 and f4.
        let attacks = pawn_attacks(Color::Black, Square::E5);
        assert_eq!(attacks.count(), 2);
        assert!(attacks.contains(Square::D4));
        assert!(attacks.contains(Square::F4));
    }

    #[test]
    fn pawn_attacks_black_a7_only_b6() {
        let attacks = pawn_attacks(Color::Black, Square::A7);
        assert_eq!(attacks.count(), 1);
        assert!(attacks.contains(Square::B6));
    }

    #[test]
    fn pawn_attacks_black_h7_only_g6() {
        let attacks = pawn_attacks(Color::Black, Square::H7);
        assert_eq!(attacks.count(), 1);
        assert!(attacks.contains(Square::G6));
    }

    #[test]
    fn pawn_attacks_white_rank_8_empty() {
        // A white pawn on rank 8 cannot attack further north (it would have
        // promoted). All rank-8 squares should return empty attacks for White.
        for file in 0..8u8 {
            let sq = Square::from_file_rank(file, 7); // rank 8 = index 7
            let attacks = pawn_attacks(Color::White, sq);
            assert!(
                attacks.is_empty(),
                "White pawn on {} (rank 8) should have no attacks, but has {} attack(s)",
                sq, attacks.count()
            );
        }
    }

    #[test]
    fn pawn_attacks_black_rank_1_empty() {
        // A black pawn on rank 1 cannot attack further south.
        for file in 0..8u8 {
            let sq = Square::from_file_rank(file, 0); // rank 1 = index 0
            let attacks = pawn_attacks(Color::Black, sq);
            assert!(
                attacks.is_empty(),
                "Black pawn on {} (rank 1) should have no attacks, but has {} attack(s)",
                sq, attacks.count()
            );
        }
    }

    #[test]
    fn pawn_attacks_white_d4() {
        let attacks = pawn_attacks(Color::White, Square::D4);
        assert_eq!(attacks.count(), 2);
        assert!(attacks.contains(Square::C5));
        assert!(attacks.contains(Square::E5));
    }

    #[test]
    fn pawn_attacks_black_d5() {
        let attacks = pawn_attacks(Color::Black, Square::D5);
        assert_eq!(attacks.count(), 2);
        assert!(attacks.contains(Square::C4));
        assert!(attacks.contains(Square::E4));
    }

    #[test]
    fn pawn_attacks_white_center_files_always_2() {
        // White pawns on files B-G (not A or H) on ranks 1-7 should always
        // attack exactly 2 squares (except rank 8 which has 0).
        for file in 1..7u8 {
            for rank in 0..7u8 {
                let sq = Square::from_file_rank(file, rank);
                let attacks = pawn_attacks(Color::White, sq);
                assert_eq!(
                    attacks.count(), 2,
                    "White pawn on {} should attack exactly 2 squares",
                    sq
                );
            }
        }
    }

    #[test]
    fn pawn_attacks_black_center_files_always_2() {
        // Black pawns on files B-G on ranks 2-8 should attack exactly 2 squares
        // (except rank 1 which has 0).
        for file in 1..7u8 {
            for rank in 1..8u8 {
                let sq = Square::from_file_rank(file, rank);
                let attacks = pawn_attacks(Color::Black, sq);
                assert_eq!(
                    attacks.count(), 2,
                    "Black pawn on {} should attack exactly 2 squares",
                    sq
                );
            }
        }
    }

    // ---- Cross-piece sanity checks ------------------------------------------

    #[test]
    fn knight_and_king_attacks_are_different() {
        // On a central square, knight and king attacks should be completely
        // different (no overlap), since they have different move patterns.
        let sq = Square::E4;
        let knight = knight_attacks(sq);
        let king = king_attacks(sq);

        // Both have 8 attacks from the center, but no square should be in both.
        let overlap = knight & king;
        assert!(
            overlap.is_empty(),
            "Knight and king attacks from e4 should not overlap, but {} squares do",
            overlap.count()
        );
    }

    #[test]
    fn pawn_attacks_are_subset_of_king_attacks() {
        // A pawn's diagonal attacks should always be a subset of the king's
        // attacks from the same square (since the king can move in all
        // 8 directions including the pawn's diagonals).
        for sq_idx in 0..64u8 {
            let sq = Square::new(sq_idx);
            let king = king_attacks(sq);

            for &color in &Color::ALL {
                let pawn = pawn_attacks(color, sq);
                let outside_king = pawn & !king;
                assert!(
                    outside_king.is_empty(),
                    "Pawn ({:?}) attacks from {} should be a subset of king attacks, but {} squares are outside",
                    color, sq, outside_king.count()
                );
            }
        }
    }

    #[test]
    fn white_and_black_pawn_attacks_are_mirrored() {
        // White pawn attacks from rank R should mirror black pawn attacks from
        // rank (7-R), flipped vertically. We test a specific case.
        //
        // White pawn on e2 (file=4, rank=1) attacks d3 and f3.
        // Black pawn on e7 (file=4, rank=6) attacks d6 and f6.
        // These are the vertical mirrors of each other.
        let white_attacks = pawn_attacks(Color::White, Square::E2);
        let black_attacks = pawn_attacks(Color::Black, Square::E7);

        assert_eq!(white_attacks.count(), 2);
        assert_eq!(black_attacks.count(), 2);

        // White attacks d3 (file=3, rank=2), black attacks d6 (file=3, rank=5)
        // The files match, and the ranks are mirrored (2 vs 5, sum = 7).
        assert!(white_attacks.contains(Square::D3));
        assert!(white_attacks.contains(Square::F3));
        assert!(black_attacks.contains(Square::D6));
        assert!(black_attacks.contains(Square::F6));
    }

    // ---- Specific regression / edge case tests ------------------------------

    #[test]
    fn knight_attacks_g1_starting_position() {
        // The g1 knight in the starting position. It can move to f3 and h3.
        let attacks = knight_attacks(Square::G1);
        assert_eq!(attacks.count(), 3);
        assert!(attacks.contains(Square::F3));
        assert!(attacks.contains(Square::H3));
        assert!(attacks.contains(Square::E2));
    }

    #[test]
    fn knight_attacks_b8_starting_position() {
        // The b8 knight. It can move to a6, c6, and d7.
        let attacks = knight_attacks(Square::B8);
        assert_eq!(attacks.count(), 3);
        assert!(attacks.contains(Square::A6));
        assert!(attacks.contains(Square::C6));
        assert!(attacks.contains(Square::D7));
    }

    #[test]
    fn king_attacks_does_not_include_castling() {
        // King on e1 should only have the 5 adjacent squares, not g1 or c1
        // (those would be castling destinations, handled separately).
        let attacks = king_attacks(Square::E1);
        assert!(!attacks.contains(Square::G1), "King attacks should not include castling squares");
        assert!(!attacks.contains(Square::C1), "King attacks should not include castling squares");
    }

    // ---- Check detection tests (is_square_attacked / is_in_check) -----------

    #[test]
    fn starting_position_no_check() {
        // In the starting position, neither king is in check.
        let board = Board::starting_position();
        assert!(
            !is_in_check(&board, Color::White),
            "White should not be in check in the starting position"
        );
        assert!(
            !is_in_check(&board, Color::Black),
            "Black should not be in check in the starting position"
        );
    }

    #[test]
    fn pawn_attacks_king() {
        // White pawn on d4 attacks e5 where the black king sits.
        let board = Board::from_fen("8/8/8/4k3/3P4/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        assert!(
            is_in_check(&board, Color::Black),
            "Black king on e5 should be attacked by white pawn on d4"
        );
        assert!(
            !is_in_check(&board, Color::White),
            "White king should not be in check"
        );
    }

    #[test]
    fn knight_attacks_king() {
        // White knight on f3 attacks e5 where the black king sits (f3->e5 is a valid knight jump).
        // Actually, let's verify: f3 (file=5,rank=2) to e5 (file=4,rank=4) = (-1,+2) = valid.
        let board = Board::from_fen("8/8/8/4k3/8/5N2/8/4K3 w - - 0 1")
            .expect("valid FEN");
        assert!(
            is_in_check(&board, Color::Black),
            "Black king on e5 should be in check by white knight on f3"
        );
    }

    #[test]
    fn bishop_attacks_king_through_empty_squares() {
        // White bishop on a1 attacks h8 diagonally (if the path is clear).
        let board = Board::from_fen("7k/8/8/8/8/8/8/B3K3 w - - 0 1")
            .expect("valid FEN");
        assert!(
            is_in_check(&board, Color::Black),
            "Black king on h8 should be in check by white bishop on a1"
        );
    }

    #[test]
    fn bishop_blocked_by_piece() {
        // White bishop on a1, a piece on d4 blocks the diagonal to the black king on h8.
        let board = Board::from_fen("7k/8/8/8/3P4/8/8/B3K3 w - - 0 1")
            .expect("valid FEN");
        assert!(
            !is_in_check(&board, Color::Black),
            "Black king on h8 should NOT be in check (bishop blocked by pawn on d4)"
        );
    }

    #[test]
    fn rook_attacks_king_on_same_file() {
        // White rook on e1 attacks black king on e8 along the e-file (if clear).
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
            .expect("valid FEN");
        // Rook is on h1, king on e8. They are not on the same file or rank. No check.
        assert!(!is_in_check(&board, Color::Black));

        // Now put the rook on the e-file.
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4KR2 w - - 0 1")
            .expect("valid FEN");
        // Rook on f1, king on e8, not the same file. No check.
        assert!(!is_in_check(&board, Color::Black));

        // Rook on e-file with clear path.
        let board = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
            .expect("valid FEN");
        // Rook on a1, king on e8. Same rank? No, a1 and e8 are different rank and file.
        assert!(!is_in_check(&board, Color::Black));

        // Actually test a proper rook check.
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4R1K1 w - - 0 1")
            .expect("valid FEN");
        assert!(
            is_in_check(&board, Color::Black),
            "Black king on e8 should be in check by white rook on e1"
        );
    }

    #[test]
    fn rook_attacks_blocked_by_piece() {
        // White rook on e1, pawn on e4 blocks the path to black king on e8.
        let board = Board::from_fen("4k3/8/8/8/4P3/8/8/4R1K1 w - - 0 1")
            .expect("valid FEN");
        assert!(
            !is_in_check(&board, Color::Black),
            "Black king on e8 should NOT be in check (rook blocked by pawn on e4)"
        );
    }

    #[test]
    fn queen_attacks_king_diagonal() {
        // White queen on a1 attacks black king on h8 diagonally.
        let board = Board::from_fen("7k/8/8/8/8/8/8/Q3K3 w - - 0 1")
            .expect("valid FEN");
        assert!(
            is_in_check(&board, Color::Black),
            "Black king on h8 should be in check by white queen on a1 (diagonal)"
        );
    }

    #[test]
    fn queen_attacks_king_straight() {
        // White queen on e1 attacks black king on e8 along the file.
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4Q1K1 w - - 0 1")
            .expect("valid FEN");
        assert!(
            is_in_check(&board, Color::Black),
            "Black king on e8 should be in check by white queen on e1 (file)"
        );
    }

    #[test]
    fn king_attacks_king() {
        // Two kings adjacent -- each "attacks" the other.
        let board = Board::from_fen("8/8/8/3kK3/8/8/8/8 w - - 0 1")
            .expect("valid FEN");
        assert!(
            is_in_check(&board, Color::Black),
            "Black king on d5 should be attacked by white king on e5"
        );
        assert!(
            is_in_check(&board, Color::White),
            "White king on e5 should be attacked by black king on d5"
        );
    }

    #[test]
    fn scholars_mate_check() {
        // After Scholar's mate: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7#
        // The black king on e8 is checkmated by the white queen on f7.
        let board = Board::from_fen("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
            .expect("valid FEN");
        assert!(
            is_in_check(&board, Color::Black),
            "Black king should be in check in Scholar's mate"
        );
        assert!(
            !is_in_check(&board, Color::White),
            "White should not be in check"
        );
    }

    #[test]
    fn double_check() {
        // A position where the black king is attacked by two pieces simultaneously.
        // White rook on e1, white bishop on b5, black king on e8 -- both attack e8.
        // Wait, bishop on b5 doesn't attack e8. Let me construct a real double check.
        // Rook on e1 checks along e-file. Knight on d6 checks e8.
        let board = Board::from_fen("4k3/8/3N4/8/8/8/8/4R1K1 w - - 0 1")
            .expect("valid FEN");
        assert!(
            is_in_check(&board, Color::Black),
            "Black king on e8 should be in double check (rook e1 + knight d6)"
        );
        // Verify both pieces individually attack e8
        assert!(
            is_square_attacked(&board, Square::E8, Color::White),
            "e8 should be attacked by white"
        );
    }

    #[test]
    fn is_square_attacked_starting_position_center_squares() {
        // In the starting position, e3 and d3 should be attacked by white pawns.
        let board = Board::starting_position();
        assert!(
            is_square_attacked(&board, Square::E3, Color::White),
            "e3 should be attacked by white pawn on d2 or f2"
        );
        assert!(
            is_square_attacked(&board, Square::D3, Color::White),
            "d3 should be attacked by white pawn on c2 or e2"
        );
        // e6 and d6 should be attacked by black pawns.
        assert!(
            is_square_attacked(&board, Square::E6, Color::Black),
            "e6 should be attacked by black pawn on d7 or f7"
        );
        assert!(
            is_square_attacked(&board, Square::D6, Color::Black),
            "d6 should be attacked by black pawn on c7 or e7"
        );
    }

    #[test]
    fn is_square_attacked_empty_square_no_attackers() {
        // A mostly empty board. Square e4 should not be attacked.
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        assert!(
            !is_square_attacked(&board, Square::E4, Color::White),
            "e4 should not be attacked by white on a nearly empty board"
        );
        assert!(
            !is_square_attacked(&board, Square::E4, Color::Black),
            "e4 should not be attacked by black on a nearly empty board"
        );
    }

    #[test]
    fn black_pawn_gives_check() {
        // Black pawn on f2 attacks white king on e1.
        let board = Board::from_fen("4k3/8/8/8/8/8/5p2/4K3 w - - 0 1")
            .expect("valid FEN");
        assert!(
            is_in_check(&board, Color::White),
            "White king on e1 should be in check by black pawn on f2"
        );
    }

    #[test]
    fn kiwipete_no_check() {
        // Kiwipete position -- neither side is in check.
        let board = Board::from_fen(fen::KIWIPETE_FEN).expect("valid FEN");
        assert!(!is_in_check(&board, Color::White), "White not in check in Kiwipete");
        assert!(!is_in_check(&board, Color::Black), "Black not in check in Kiwipete");
    }
}
