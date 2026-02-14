//! Perft (performance test) for validating move generation correctness.
//!
//! Perft counts the number of leaf nodes (positions) reachable at a given depth
//! from a starting position by exhaustively making all legal moves. The resulting
//! counts are well-established and published for many positions. If our engine
//! produces the correct perft numbers, we can be confident that move generation,
//! make/unmake, castling, en passant, promotions, and all other edge cases are
//! implemented correctly.
//!
//! This is the gold standard for chess engine correctness testing.
//!
//! # Bulk counting optimization
//!
//! At depth 1, instead of making each move and recursing to depth 0 (which would
//! return 1 for each), we simply return the number of legal moves. This avoids
//! the overhead of make/unmake for the final ply and roughly doubles the speed
//! of perft at any depth.

use crate::board::Board;
use crate::movegen::generate_legal_moves;

/// Counts the number of leaf nodes at the given depth from the current position.
///
/// - At depth 0, returns 1 (the current position itself is the only leaf).
/// - At depth 1, returns the number of legal moves (bulk counting optimization).
/// - At deeper depths, recursively makes each legal move and sums the subtree counts.
///
/// # Arguments
///
/// * `board` - A mutable reference to the board. The board is temporarily modified
///   by make/unmake during the search but is restored to its original state when
///   the function returns.
/// * `depth` - The number of plies (half-moves) to search. Depth 0 means "count
///   this position", depth 1 means "count all positions after one move", etc.
///
/// # Example
///
/// ```ignore
/// let mut board = Board::starting_position();
/// assert_eq!(perft(&mut board, 0), 1);
/// assert_eq!(perft(&mut board, 1), 20);   // 20 legal moves from starting position
/// assert_eq!(perft(&mut board, 2), 400);  // 20 * 20 = 400
/// ```
pub fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = generate_legal_moves(board);

    // Bulk counting optimization: at depth 1, the number of leaf nodes equals
    // the number of legal moves. No need to make/unmake each one.
    if depth == 1 {
        return moves.len() as u64;
    }

    let mut count = 0u64;
    for mv in moves {
        let undo = board.make_move(mv);
        count += perft(board, depth - 1);
        board.unmake_move(mv, &undo);
    }
    count
}

/// Divide: shows the perft count broken down by each root move.
///
/// This is the primary debugging tool when perft numbers don't match. By comparing
/// the per-move breakdown against a known-correct engine, you can identify exactly
/// which move (or moves) produce incorrect subtree counts, narrowing down the bug.
///
/// Returns a `Vec` of `(move_uci_string, count)` pairs, sorted alphabetically
/// by move name for consistent, comparable output.
///
/// # Arguments
///
/// * `board` - A mutable reference to the board (restored to original state on return).
/// * `depth` - The total depth to search. Each root move's count is `perft(depth - 1)`
///   from the position after that move.
///
/// # Example
///
/// ```ignore
/// let mut board = Board::starting_position();
/// let results = divide(&mut board, 2);
/// // Each entry shows a root move and how many leaf nodes it leads to at depth 2.
/// // The sum of all counts should equal perft(board, 2) = 400.
/// for (mv, count) in &results {
///     println!("{}: {}", mv, count);
/// }
/// ```
pub fn divide(board: &mut Board, depth: u32) -> Vec<(String, u64)> {
    let moves = generate_legal_moves(board);
    let mut results = Vec::new();

    for mv in moves {
        let undo = board.make_move(mv);
        let count = perft(board, depth - 1);
        board.unmake_move(mv, &undo);
        results.push((mv.to_uci(), count));
    }

    // Sort by move name for consistent, reproducible output.
    results.sort_by(|a, b| a.0.cmp(&b.0));
    results
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::{
        KIWIPETE_FEN, POSITION_3_FEN, POSITION_4_FEN, POSITION_5_FEN, POSITION_6_FEN,
        STARTING_FEN,
    };

    // =========================================================================
    // Starting position perft
    // =========================================================================
    //
    // These are universally agreed-upon values. If any of these fail, something
    // fundamental is wrong with move generation or make/unmake.

    #[test]
    fn perft_starting_depth_0() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        assert_eq!(perft(&mut board, 0), 1);
    }

    #[test]
    fn perft_starting_depth_1() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        assert_eq!(perft(&mut board, 1), 20);
    }

    #[test]
    fn perft_starting_depth_2() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        assert_eq!(perft(&mut board, 2), 400);
    }

    #[test]
    fn perft_starting_depth_3() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        assert_eq!(perft(&mut board, 3), 8_902);
    }

    #[test]
    #[ignore] // Takes a few seconds in debug mode
    fn perft_starting_depth_4() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        assert_eq!(perft(&mut board, 4), 197_281);
    }

    #[test]
    #[ignore] // Takes several seconds in debug mode
    fn perft_starting_depth_5() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        assert_eq!(perft(&mut board, 5), 4_865_609);
    }

    // =========================================================================
    // Kiwipete position perft
    // =========================================================================
    //
    // r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -
    //
    // This position has many edge cases: en passant, castling, pins,
    // discovered checks, etc. Created by Peter McKenzie for perft testing.

    #[test]
    fn perft_kiwipete_depth_1() {
        let mut board = Board::from_fen(KIWIPETE_FEN).unwrap();
        assert_eq!(perft(&mut board, 1), 48);
    }

    #[test]
    fn perft_kiwipete_depth_2() {
        let mut board = Board::from_fen(KIWIPETE_FEN).unwrap();
        assert_eq!(perft(&mut board, 2), 2_039);
    }

    #[test]
    fn perft_kiwipete_depth_3() {
        let mut board = Board::from_fen(KIWIPETE_FEN).unwrap();
        assert_eq!(perft(&mut board, 3), 97_862);
    }

    #[test]
    #[ignore] // May be slow in debug mode
    fn perft_kiwipete_depth_4() {
        let mut board = Board::from_fen(KIWIPETE_FEN).unwrap();
        assert_eq!(perft(&mut board, 4), 4_085_603);
    }

    // =========================================================================
    // Position 3 perft
    // =========================================================================
    //
    // 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -
    //
    // A sparse position with pawns, a rook, and kings. Tests pawn captures,
    // en passant edge cases, and positions where kings are exposed.

    #[test]
    fn perft_position3_depth_1() {
        let mut board = Board::from_fen(POSITION_3_FEN).unwrap();
        assert_eq!(perft(&mut board, 1), 14);
    }

    #[test]
    fn perft_position3_depth_2() {
        let mut board = Board::from_fen(POSITION_3_FEN).unwrap();
        assert_eq!(perft(&mut board, 2), 191);
    }

    #[test]
    fn perft_position3_depth_3() {
        let mut board = Board::from_fen(POSITION_3_FEN).unwrap();
        assert_eq!(perft(&mut board, 3), 2_812);
    }

    #[test]
    #[ignore] // May be slow in debug mode
    fn perft_position3_depth_4() {
        let mut board = Board::from_fen(POSITION_3_FEN).unwrap();
        assert_eq!(perft(&mut board, 4), 43_238);
    }

    #[test]
    #[ignore] // May be slow in debug mode
    fn perft_position3_depth_5() {
        let mut board = Board::from_fen(POSITION_3_FEN).unwrap();
        assert_eq!(perft(&mut board, 5), 674_624);
    }

    // =========================================================================
    // Position 4 perft
    // =========================================================================
    //
    // r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1
    //
    // A wild position with many promotions, captures, and castling rights
    // that are asymmetric (Black can castle, White cannot).

    #[test]
    fn perft_position4_depth_1() {
        let mut board = Board::from_fen(POSITION_4_FEN).unwrap();
        assert_eq!(perft(&mut board, 1), 6);
    }

    #[test]
    fn perft_position4_depth_2() {
        let mut board = Board::from_fen(POSITION_4_FEN).unwrap();
        assert_eq!(perft(&mut board, 2), 264);
    }

    #[test]
    fn perft_position4_depth_3() {
        let mut board = Board::from_fen(POSITION_4_FEN).unwrap();
        assert_eq!(perft(&mut board, 3), 9_467);
    }

    #[test]
    #[ignore] // May be slow in debug mode
    fn perft_position4_depth_4() {
        let mut board = Board::from_fen(POSITION_4_FEN).unwrap();
        assert_eq!(perft(&mut board, 4), 422_333);
    }

    // =========================================================================
    // Position 5 perft
    // =========================================================================
    //
    // rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8
    //
    // A position with a pawn on the 7th rank ready to promote, plus a black
    // knight on f2 attacking the white king. Tests promotion and check handling.

    #[test]
    fn perft_position5_depth_1() {
        let mut board = Board::from_fen(POSITION_5_FEN).unwrap();
        assert_eq!(perft(&mut board, 1), 44);
    }

    #[test]
    fn perft_position5_depth_2() {
        let mut board = Board::from_fen(POSITION_5_FEN).unwrap();
        assert_eq!(perft(&mut board, 2), 1_486);
    }

    #[test]
    fn perft_position5_depth_3() {
        let mut board = Board::from_fen(POSITION_5_FEN).unwrap();
        assert_eq!(perft(&mut board, 3), 62_379);
    }

    #[test]
    #[ignore] // May be slow in debug mode
    fn perft_position5_depth_4() {
        let mut board = Board::from_fen(POSITION_5_FEN).unwrap();
        assert_eq!(perft(&mut board, 4), 2_103_487);
    }

    // =========================================================================
    // Position 6 perft
    // =========================================================================
    //
    // r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10
    //
    // A symmetrical middlegame position. Both sides have developed pieces with
    // lots of tactical possibilities. Tests general move generation thoroughly.

    #[test]
    fn perft_position6_depth_1() {
        let mut board = Board::from_fen(POSITION_6_FEN).unwrap();
        assert_eq!(perft(&mut board, 1), 46);
    }

    #[test]
    fn perft_position6_depth_2() {
        let mut board = Board::from_fen(POSITION_6_FEN).unwrap();
        assert_eq!(perft(&mut board, 2), 2_079);
    }

    #[test]
    fn perft_position6_depth_3() {
        let mut board = Board::from_fen(POSITION_6_FEN).unwrap();
        assert_eq!(perft(&mut board, 3), 89_890);
    }

    #[test]
    #[ignore] // May be slow in debug mode (>30s possible)
    fn perft_position6_depth_4() {
        let mut board = Board::from_fen(POSITION_6_FEN).unwrap();
        assert_eq!(perft(&mut board, 4), 3_894_594);
    }

    // =========================================================================
    // Divide tests
    // =========================================================================

    #[test]
    fn divide_starting_depth_1() {
        // Divide at depth 1 should list all 20 root moves, each with a count of 1.
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        let results = divide(&mut board, 1);

        assert_eq!(results.len(), 20, "Starting position has 20 legal moves");

        let total: u64 = results.iter().map(|(_, count)| count).sum();
        assert_eq!(total, 20, "Sum of divide at depth 1 should equal perft(1)");

        // Every root move at depth 1 should have count 1 (since depth-1 = 0 => 1 leaf).
        for (mv, count) in &results {
            assert_eq!(*count, 1, "Move {} at depth 1 should have count 1", mv);
        }
    }

    #[test]
    fn divide_starting_depth_2() {
        // Divide at depth 2: each root move's subtotal should sum to 400.
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        let results = divide(&mut board, 2);

        assert_eq!(results.len(), 20, "Starting position has 20 legal moves");

        let total: u64 = results.iter().map(|(_, count)| count).sum();
        assert_eq!(
            total, 400,
            "Sum of divide at depth 2 should equal perft(2) = 400"
        );

        // Each root move at depth 2 should produce exactly 20 responses
        // (in the starting position, every first move leaves 20 legal replies).
        for (mv, count) in &results {
            assert_eq!(
                *count, 20,
                "Move {} at depth 2 should have count 20 (opponent has 20 replies)",
                mv
            );
        }
    }

    #[test]
    fn divide_results_are_sorted() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        let results = divide(&mut board, 1);

        // Verify the results are sorted alphabetically by move name.
        for i in 1..results.len() {
            assert!(
                results[i - 1].0 <= results[i].0,
                "Divide results should be sorted: '{}' should come before '{}'",
                results[i - 1].0,
                results[i].0
            );
        }
    }

    // =========================================================================
    // Board restoration tests
    // =========================================================================
    //
    // Verify that perft leaves the board in its original state after returning.

    #[test]
    fn perft_restores_board_state() {
        let mut board = Board::from_fen(STARTING_FEN).unwrap();
        let fen_before = board.to_fen();

        perft(&mut board, 3);

        let fen_after = board.to_fen();
        assert_eq!(
            fen_before, fen_after,
            "Board should be unchanged after perft completes"
        );
    }

    #[test]
    fn divide_restores_board_state() {
        let mut board = Board::from_fen(KIWIPETE_FEN).unwrap();
        let fen_before = board.to_fen();

        divide(&mut board, 2);

        let fen_after = board.to_fen();
        assert_eq!(
            fen_before, fen_after,
            "Board should be unchanged after divide completes"
        );
    }
}
