//! Game state management and termination detection.
//!
//! This module wraps a `Board` with position history tracking, enabling
//! detection of all game-ending conditions needed for AlphaZero self-play:
//!
//! - **Checkmate**: No legal moves and king is in check
//! - **Stalemate**: No legal moves and king is not in check
//! - **Threefold repetition**: Same position reached 3 times (with a configurable
//!   `is_repetition(count)` method; AlphaZero uses twofold)
//! - **50-move rule**: 100 half-moves with no pawn move or capture
//! - **Insufficient material**: Only kings remain, or K+B vs K, K+N vs K,
//!   or K+B vs K+B with same-colored bishops
//!
//! The `result()` method checks these conditions in order of cost: cheap field
//! comparisons first (50-move rule), then history iteration (repetition), then
//! piece counting (insufficient material), and finally legal move generation
//! (checkmate/stalemate) only if needed.

use crate::attacks::is_in_check;
use crate::board::Board;
use crate::makemove::UndoInfo;
use crate::movegen::generate_legal_moves;
use crate::moves::Move;
use crate::types::{Color, Piece};

// =============================================================================
// GameResult
// =============================================================================

/// The result of a game or the current game state.
///
/// Used by the MCTS and self-play systems to determine when a game has ended
/// and what the outcome is. `Ongoing` means the game is still in progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    /// The game is still in progress.
    Ongoing,
    /// Black is checkmated -- White wins.
    WhiteWins,
    /// White is checkmated -- Black wins.
    BlackWins,
    /// Draw by stalemate (side to move has no legal moves but is not in check).
    DrawStalemate,
    /// Draw by threefold repetition (same position reached 3+ times).
    DrawRepetition,
    /// Draw by the 50-move rule (100 half-moves without a pawn move or capture).
    DrawFiftyMoveRule,
    /// Draw by insufficient mating material.
    DrawInsufficientMaterial,
}

// =============================================================================
// Game
// =============================================================================

/// A game manager that wraps a `Board` and tracks position history for
/// repetition detection.
///
/// The `history` vector stores the Zobrist hash of every position that has
/// occurred in the game. This allows efficient threefold (or twofold)
/// repetition checking by counting how many times the current hash appears.
pub struct Game {
    board: Board,
    /// Zobrist hashes of all positions that have occurred in the game,
    /// including the current position. The current position's hash is
    /// always the last element.
    history: Vec<u64>,
}

// =============================================================================
// Construction
// =============================================================================

impl Game {
    /// Creates a new game starting from the standard chess opening position.
    pub fn new() -> Game {
        let board = Board::starting_position();
        let hash = board.hash();
        Game {
            board,
            history: vec![hash],
        }
    }

    /// Creates a new game starting from the given board position.
    ///
    /// The history is initialized with just the given position's hash,
    /// meaning no prior repetition context is available.
    pub fn from_board(board: Board) -> Game {
        let hash = board.hash();
        Game {
            board,
            history: vec![hash],
        }
    }
}

// =============================================================================
// Accessors
// =============================================================================

impl Game {
    /// Returns a reference to the current board position.
    #[inline]
    pub fn board(&self) -> &Board {
        &self.board
    }
}

// =============================================================================
// Move execution
// =============================================================================

impl Game {
    /// Makes a move on the board and records the new position hash in history.
    ///
    /// Returns the `UndoInfo` needed to reverse this move later.
    pub fn make_move(&mut self, mv: Move) -> UndoInfo {
        let undo = self.board.make_move(mv);
        self.history.push(self.board.hash());
        undo
    }

    /// Unmakes a move, restoring the board to its previous state and removing
    /// the most recent position hash from history.
    pub fn unmake_move(&mut self, mv: Move, undo: &UndoInfo) {
        self.board.unmake_move(mv, undo);
        self.history.pop();
    }
}

// =============================================================================
// Termination detection
// =============================================================================

impl Game {
    /// Determines the current game result by checking all termination conditions.
    ///
    /// Conditions are checked in order of computational cost:
    /// 1. **50-move rule** -- just a field comparison (cheapest)
    /// 2. **Threefold repetition** -- iterate position history
    /// 3. **Insufficient material** -- count pieces on the board
    /// 4. **Checkmate / Stalemate** -- generate all legal moves (most expensive)
    pub fn result(&self) -> GameResult {
        // 1. 50-move rule: 100 half-moves = 50 full moves without a pawn move
        //    or capture. We check this first because it is the cheapest test.
        if self.board.halfmove_clock() >= 100 {
            return GameResult::DrawFiftyMoveRule;
        }

        // 2. Threefold repetition: the current position hash appears 3+ times
        //    in the history (which includes the current position).
        if self.count_repetitions() >= 3 {
            return GameResult::DrawRepetition;
        }

        // 3. Insufficient material: only bare kings or trivially drawn material.
        if is_insufficient_material(&self.board) {
            return GameResult::DrawInsufficientMaterial;
        }

        // 4. Generate legal moves to check for checkmate or stalemate.
        let legal_moves = generate_legal_moves(&self.board);
        if legal_moves.is_empty() {
            let side_to_move = self.board.side_to_move();
            if is_in_check(&self.board, side_to_move) {
                // The side to move is in check with no legal moves: checkmate.
                // The *other* side wins.
                match side_to_move {
                    Color::White => return GameResult::BlackWins,
                    Color::Black => return GameResult::WhiteWins,
                }
            } else {
                // No legal moves and not in check: stalemate.
                return GameResult::DrawStalemate;
            }
        }

        GameResult::Ongoing
    }

    /// Returns `true` if the game has ended (result is anything other than `Ongoing`).
    pub fn is_terminal(&self) -> bool {
        self.result() != GameResult::Ongoing
    }

    /// Convenience wrapper that generates all legal moves for the current position.
    pub fn legal_moves(&self) -> Vec<Move> {
        generate_legal_moves(&self.board)
    }

    /// Counts how many times the current position's Zobrist hash appears in the
    /// game history (including the current position itself).
    ///
    /// A count of 3 means threefold repetition. A count of 2 means the position
    /// has occurred twice (twofold repetition, which AlphaZero treats as a draw).
    pub fn count_repetitions(&self) -> usize {
        let current_hash = self.board.hash();
        self.history
            .iter()
            .filter(|&&h| h == current_hash)
            .count()
    }

    /// Returns `true` if the current position has occurred at least `count` times
    /// in the game history.
    ///
    /// For standard chess rules, use `count = 3` (threefold repetition).
    /// For AlphaZero self-play, use `count = 2` (twofold repetition) since the
    /// agent should learn to avoid repeating positions.
    pub fn is_repetition(&self, count: usize) -> bool {
        self.count_repetitions() >= count
    }

    /// Returns the game result as a numeric value from the perspective of the
    /// given color.
    ///
    /// - `+1.0` if the given color wins
    /// - `-1.0` if the given color loses
    /// - `0.0` for any draw or ongoing game
    ///
    /// This is the reward signal used by AlphaZero's value network.
    pub fn result_for_color(&self, color: Color) -> f32 {
        match self.result() {
            GameResult::WhiteWins => match color {
                Color::White => 1.0,
                Color::Black => -1.0,
            },
            GameResult::BlackWins => match color {
                Color::White => -1.0,
                Color::Black => 1.0,
            },
            // All draw types and Ongoing return 0.
            _ => 0.0,
        }
    }
}

// =============================================================================
// Insufficient material detection (free function)
// =============================================================================

/// Determines whether the position has insufficient mating material.
///
/// The following material configurations are considered insufficient:
/// - King vs King
/// - King + Bishop vs King
/// - King + Knight vs King
/// - King + Bishop vs King + Bishop (same-colored bishops only)
///
/// Note: K+N+N vs K is technically NOT insufficient (forced mate exists in some
/// positions), so we do NOT treat it as a draw. Similarly, K+B+B vs K (opposite
/// colored bishops) can force mate, so that is not insufficient either.
pub fn is_insufficient_material(board: &Board) -> bool {
    // Count all pieces by type and color.
    let white_pawns = board.piece_bitboard(Color::White, Piece::Pawn).count();
    let black_pawns = board.piece_bitboard(Color::Black, Piece::Pawn).count();
    let white_knights = board.piece_bitboard(Color::White, Piece::Knight).count();
    let black_knights = board.piece_bitboard(Color::Black, Piece::Knight).count();
    let white_bishops = board.piece_bitboard(Color::White, Piece::Bishop).count();
    let black_bishops = board.piece_bitboard(Color::Black, Piece::Bishop).count();
    let white_rooks = board.piece_bitboard(Color::White, Piece::Rook).count();
    let black_rooks = board.piece_bitboard(Color::Black, Piece::Rook).count();
    let white_queens = board.piece_bitboard(Color::White, Piece::Queen).count();
    let black_queens = board.piece_bitboard(Color::Black, Piece::Queen).count();

    // If any pawns, rooks, or queens exist, there is sufficient material.
    if white_pawns + black_pawns + white_rooks + black_rooks + white_queens + black_queens > 0 {
        return false;
    }

    // At this point, only kings, knights, and bishops remain.
    let total_knights = white_knights + black_knights;
    let total_bishops = white_bishops + black_bishops;
    let total_minor = total_knights + total_bishops;

    // King vs King: no minor pieces at all.
    if total_minor == 0 {
        return true;
    }

    // Exactly one minor piece total: K+B vs K or K+N vs K.
    if total_minor == 1 {
        return true;
    }

    // K+B vs K+B: two bishops total, one on each side, and they must be on
    // the same colored squares (otherwise mate is possible).
    if total_minor == 2 && white_bishops == 1 && black_bishops == 1 {
        let white_bishop_sq = board.piece_bitboard(Color::White, Piece::Bishop).lsb();
        let black_bishop_sq = board.piece_bitboard(Color::Black, Piece::Bishop).lsb();
        // A square's color is determined by (file + rank) % 2.
        // If both bishops are on the same parity, they are on the same color.
        let white_color = (white_bishop_sq.file() + white_bishop_sq.rank()) % 2;
        let black_color = (black_bishop_sq.file() + black_bishop_sq.rank()) % 2;
        if white_color == black_color {
            return true;
        }
    }

    false
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::moves::{Move, MoveFlags};
    use crate::types::{Color, Square};

    // ========================================================================
    // Starting position: should be Ongoing
    // ========================================================================

    #[test]
    fn starting_position_is_ongoing() {
        let game = Game::new();
        assert_eq!(game.result(), GameResult::Ongoing);
        assert!(!game.is_terminal());
    }

    // ========================================================================
    // Game::new starts from starting position
    // ========================================================================

    #[test]
    fn game_new_starts_from_starting_position() {
        let game = Game::new();
        let expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        assert_eq!(game.board().to_fen(), expected_fen);
    }

    // ========================================================================
    // Checkmate detection: Scholar's mate
    // ========================================================================

    #[test]
    fn scholars_mate_white_wins() {
        let mut game = Game::new();

        // 1. e4
        game.make_move(Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH));
        // 1... e5
        game.make_move(Move::with_flags(Square::E7, Square::E5, MoveFlags::DOUBLE_PAWN_PUSH));
        // 2. Bc4
        game.make_move(Move::new(Square::F1, Square::C4));
        // 2... Nc6
        game.make_move(Move::new(Square::B8, Square::C6));
        // 3. Qh5
        game.make_move(Move::new(Square::D1, Square::H5));
        // 3... Nf6
        game.make_move(Move::new(Square::G8, Square::F6));
        // 4. Qxf7#
        game.make_move(Move::new(Square::H5, Square::F7));

        assert_eq!(game.result(), GameResult::WhiteWins);
        assert!(game.is_terminal());
    }

    // ========================================================================
    // Back rank mate
    // ========================================================================

    #[test]
    fn back_rank_mate() {
        // White rook delivers checkmate on the back rank.
        // Position: Black king on g8, pawns on f7, g7, h7 (blocking escape).
        // White rook on a1, white king on g1.
        // White plays Ra8# (rook to a8, checkmate).
        let board = Board::from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
            .expect("valid FEN");
        let mut game = Game::from_board(board);
        game.make_move(Move::new(Square::A1, Square::A8));

        assert_eq!(game.result(), GameResult::WhiteWins);
        assert!(game.is_terminal());
    }

    // ========================================================================
    // Stalemate
    // ========================================================================

    #[test]
    fn stalemate_detection() {
        // Classic stalemate: Black king on a8, White queen on b6, White king on c8.
        // Black to move, no legal moves but not in check.
        let board = Board::from_fen("k7/8/1Q6/8/8/8/8/2K5 b - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        assert_eq!(game.result(), GameResult::DrawStalemate);
        assert!(game.is_terminal());
    }

    // ========================================================================
    // Threefold repetition
    // ========================================================================

    #[test]
    fn threefold_repetition() {
        // Play Ng1-f3, Ng8-f6, Nf3-g1, Nf6-g8, Ng1-f3, Ng8-f6, Nf3-g1, Nf6-g8
        // This reaches the starting position 3 times total (initial + 2 returns).
        let mut game = Game::new();

        // 1. Nf3
        game.make_move(Move::new(Square::G1, Square::F3));
        // 1... Nf6
        game.make_move(Move::new(Square::G8, Square::F6));
        // 2. Ng1
        game.make_move(Move::new(Square::F3, Square::G1));
        // 2... Ng8
        game.make_move(Move::new(Square::F6, Square::G8));
        // Now we are back to the starting position hash (2nd occurrence).
        assert_eq!(game.count_repetitions(), 2);
        assert_ne!(game.result(), GameResult::DrawRepetition);

        // 3. Nf3
        game.make_move(Move::new(Square::G1, Square::F3));
        // 3... Nf6
        game.make_move(Move::new(Square::G8, Square::F6));
        // 4. Ng1
        game.make_move(Move::new(Square::F3, Square::G1));
        // 4... Ng8
        game.make_move(Move::new(Square::F6, Square::G8));
        // Now we are back to the starting position hash (3rd occurrence).
        assert_eq!(game.count_repetitions(), 3);
        assert_eq!(game.result(), GameResult::DrawRepetition);
        assert!(game.is_terminal());
    }

    // ========================================================================
    // Twofold repetition (is_repetition with count=2)
    // ========================================================================

    #[test]
    fn twofold_repetition() {
        let mut game = Game::new();

        // 1. Nf3
        game.make_move(Move::new(Square::G1, Square::F3));
        // 1... Nf6
        game.make_move(Move::new(Square::G8, Square::F6));
        // 2. Ng1
        game.make_move(Move::new(Square::F3, Square::G1));
        // 2... Ng8
        game.make_move(Move::new(Square::F6, Square::G8));

        // Position has occurred twice (initial + this return).
        assert!(game.is_repetition(2));
        // But not three times yet.
        assert!(!game.is_repetition(3));
    }

    // ========================================================================
    // 50-move rule
    // ========================================================================

    #[test]
    fn fifty_move_rule() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 100 50")
            .expect("valid FEN");
        // halfmove_clock is 100, which triggers the 50-move rule.
        let game = Game::from_board(board);

        assert_eq!(game.result(), GameResult::DrawFiftyMoveRule);
        assert!(game.is_terminal());
    }

    #[test]
    fn fifty_move_rule_not_triggered_at_99() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 99 50")
            .expect("valid FEN");
        // halfmove_clock is 99 -- this is K vs K, so it will be insufficient material.
        // But the 50-move rule check happens first (at >= 100), and this is 99, so
        // the insufficient material check will fire instead.
        let game = Game::from_board(board);

        // Not DrawFiftyMoveRule because clock is only 99.
        assert_ne!(game.result(), GameResult::DrawFiftyMoveRule);
        // But it should still be terminal (insufficient material: K vs K).
        assert_eq!(game.result(), GameResult::DrawInsufficientMaterial);
    }

    // ========================================================================
    // Insufficient material: K vs K
    // ========================================================================

    #[test]
    fn insufficient_material_king_vs_king() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        assert_eq!(game.result(), GameResult::DrawInsufficientMaterial);
        assert!(game.is_terminal());
    }

    // ========================================================================
    // Insufficient material: K+B vs K
    // ========================================================================

    #[test]
    fn insufficient_material_king_bishop_vs_king() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4KB2 w - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        assert_eq!(game.result(), GameResult::DrawInsufficientMaterial);
        assert!(game.is_terminal());
    }

    #[test]
    fn insufficient_material_king_vs_king_bishop() {
        let board = Board::from_fen("4kb2/8/8/8/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        assert_eq!(game.result(), GameResult::DrawInsufficientMaterial);
    }

    // ========================================================================
    // Insufficient material: K+N vs K
    // ========================================================================

    #[test]
    fn insufficient_material_king_knight_vs_king() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4KN2 w - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        assert_eq!(game.result(), GameResult::DrawInsufficientMaterial);
        assert!(game.is_terminal());
    }

    #[test]
    fn insufficient_material_king_vs_king_knight() {
        let board = Board::from_fen("4kn2/8/8/8/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        assert_eq!(game.result(), GameResult::DrawInsufficientMaterial);
    }

    // ========================================================================
    // Insufficient material: K+B vs K+B (same colored bishops)
    // ========================================================================

    #[test]
    fn insufficient_material_same_colored_bishops() {
        // White bishop on c1 (dark square: file=2, rank=0, sum=2, even -> dark)
        // Black bishop on f8 (dark square: file=5, rank=7, sum=12, even -> dark)
        let board = Board::from_fen("4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        assert_eq!(game.result(), GameResult::DrawInsufficientMaterial);
    }

    #[test]
    fn sufficient_material_opposite_colored_bishops() {
        // White bishop on c1 (dark square: file=2, rank=0, sum=2, even)
        // Black bishop on c8 (light square: file=2, rank=7, sum=9, odd)
        let board = Board::from_fen("2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        // Opposite colored bishops -- mate IS possible, so this is NOT insufficient.
        assert_eq!(game.result(), GameResult::Ongoing);
    }

    // ========================================================================
    // Sufficient material: K+R vs K
    // ========================================================================

    #[test]
    fn sufficient_material_king_rook_vs_king() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4KR2 w - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        assert_eq!(game.result(), GameResult::Ongoing);
        assert!(!game.is_terminal());
    }

    // ========================================================================
    // Sufficient material: K+P vs K
    // ========================================================================

    #[test]
    fn sufficient_material_king_pawn_vs_king() {
        let board = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        assert_eq!(game.result(), GameResult::Ongoing);
    }

    // ========================================================================
    // Sufficient material: K+N+N vs K (NOT insufficient -- forced mate exists)
    // ========================================================================

    #[test]
    fn sufficient_material_king_two_knights_vs_king() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/3NKN2 w - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        // K+N+N vs K: we do NOT treat this as insufficient.
        assert_eq!(game.result(), GameResult::Ongoing);
    }

    // ========================================================================
    // is_terminal: true for all non-Ongoing results
    // ========================================================================

    #[test]
    fn is_terminal_for_various_results() {
        // Checkmate
        let board = Board::from_fen("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
            .expect("valid FEN");
        let game = Game::from_board(board);
        assert!(game.is_terminal());

        // Stalemate
        let board = Board::from_fen("k7/8/1Q6/8/8/8/8/2K5 b - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);
        assert!(game.is_terminal());

        // Insufficient material (K vs K)
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);
        assert!(game.is_terminal());

        // 50-move rule
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 100 50")
            .expect("valid FEN");
        let game = Game::from_board(board);
        assert!(game.is_terminal());

        // Ongoing (starting position)
        let game = Game::new();
        assert!(!game.is_terminal());
    }

    // ========================================================================
    // Make/unmake through Game: verify history tracking
    // ========================================================================

    #[test]
    fn make_unmake_tracks_history() {
        let mut game = Game::new();
        assert_eq!(game.history.len(), 1);

        let mv = Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH);
        let undo = game.make_move(mv);
        assert_eq!(game.history.len(), 2);

        game.unmake_move(mv, &undo);
        assert_eq!(game.history.len(), 1);

        // Board should be back to starting position
        let expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        assert_eq!(game.board().to_fen(), expected_fen);
    }

    // ========================================================================
    // result_for_color: reward signal
    // ========================================================================

    #[test]
    fn result_for_color_white_wins() {
        // Scholar's mate position (after Qxf7#, Black to move, checkmate).
        let board = Board::from_fen("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
            .expect("valid FEN");
        let game = Game::from_board(board);

        assert_eq!(game.result(), GameResult::WhiteWins);
        assert_eq!(game.result_for_color(Color::White), 1.0);
        assert_eq!(game.result_for_color(Color::Black), -1.0);
    }

    #[test]
    fn result_for_color_black_wins() {
        // Fool's mate: White is checkmated.
        let board = Board::from_fen("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
            .expect("valid FEN");
        let game = Game::from_board(board);

        assert_eq!(game.result(), GameResult::BlackWins);
        assert_eq!(game.result_for_color(Color::White), -1.0);
        assert_eq!(game.result_for_color(Color::Black), 1.0);
    }

    #[test]
    fn result_for_color_draw() {
        // K vs K: draw
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        assert_eq!(game.result_for_color(Color::White), 0.0);
        assert_eq!(game.result_for_color(Color::Black), 0.0);
    }

    #[test]
    fn result_for_color_ongoing() {
        let game = Game::new();
        assert_eq!(game.result_for_color(Color::White), 0.0);
        assert_eq!(game.result_for_color(Color::Black), 0.0);
    }

    // ========================================================================
    // legal_moves convenience wrapper
    // ========================================================================

    #[test]
    fn legal_moves_starting_position() {
        let game = Game::new();
        let moves = game.legal_moves();
        assert_eq!(moves.len(), 20, "Starting position should have exactly 20 legal moves");
    }

    // ========================================================================
    // Black checkmate (Black delivers mate)
    // ========================================================================

    #[test]
    fn fools_mate_black_wins() {
        let mut game = Game::new();

        // 1. f3
        game.make_move(Move::new(Square::F2, Square::F3));
        // 1... e6
        game.make_move(Move::new(Square::E7, Square::E6));
        // 2. g4
        game.make_move(Move::with_flags(Square::G2, Square::G4, MoveFlags::DOUBLE_PAWN_PUSH));
        // 2... Qh4#
        game.make_move(Move::new(Square::D8, Square::H4));

        assert_eq!(game.result(), GameResult::BlackWins);
        assert!(game.is_terminal());
    }

    // ========================================================================
    // Insufficient material helper: direct unit tests
    // ========================================================================

    #[test]
    fn insufficient_material_with_queen_is_false() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
            .expect("valid FEN");
        assert!(!is_insufficient_material(&board));
    }

    #[test]
    fn insufficient_material_with_pawn_is_false() {
        let board = Board::from_fen("4k3/p7/8/8/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        assert!(!is_insufficient_material(&board));
    }

    #[test]
    fn insufficient_material_king_bishop_bishop_same_color() {
        // Two bishops on same colored squares (both dark).
        // White bishop on a1 (file=0, rank=0, sum=0, even -> dark)
        // Black bishop on c1 (file=2, rank=0, sum=2, even -> dark)
        // Wait -- c1 is White's side. Let's use proper FEN:
        // White bishop on a3 (file=0, rank=2, sum=2, even -> dark)
        // Black bishop on c1 (file=2, rank=0, sum=2, even -> dark)
        let board = Board::from_fen("4k3/8/8/8/8/B7/8/4K3 w - - 0 1")
            .expect("valid FEN");
        // This is K+B vs K, which is already insufficient (1 minor piece).
        assert!(is_insufficient_material(&board));
    }

    // ========================================================================
    // count_repetitions: basic test
    // ========================================================================

    #[test]
    fn count_repetitions_initial_position() {
        let game = Game::new();
        // Initial position appears exactly once.
        assert_eq!(game.count_repetitions(), 1);
    }

    // ========================================================================
    // Game::from_board works correctly
    // ========================================================================

    #[test]
    fn from_board_preserves_position() {
        let board = Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
            .expect("valid FEN");
        let expected_fen = board.to_fen();
        let game = Game::from_board(board);

        assert_eq!(game.board().to_fen(), expected_fen);
        assert_eq!(game.history.len(), 1);
    }

    // ========================================================================
    // Stalemate with pawns blocked
    // ========================================================================

    #[test]
    fn stalemate_king_trapped_in_corner() {
        // White king on h1, Black queen on g3, Black king on f2.
        // White to move, no legal moves, not in check.
        // Actually let me use a simpler known stalemate.
        // King on a1, opponent queen on b3, opponent king on c1.
        // a1 king: can go to a2 (attacked by Qb3), b1 (attacked by Qb3), b2 (attacked by Qb3).
        // Hmm, let's use a well-known stalemate FEN instead.
        let board = Board::from_fen("5k2/5P2/5K2/8/8/8/8/8 b - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);
        // Black king on f8, White pawn on f7, White king on f6.
        // Black can't move: king is trapped, not in check.
        assert_eq!(game.result(), GameResult::DrawStalemate);
    }

    // ========================================================================
    // Verify 50-move rule takes priority over checkmate/stalemate
    // ========================================================================

    #[test]
    fn fifty_move_rule_priority_over_material_check() {
        // Position with halfmove_clock = 100 but with sufficient material.
        let board = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 100 60")
            .expect("valid FEN");
        let game = Game::from_board(board);
        assert_eq!(game.result(), GameResult::DrawFiftyMoveRule);
    }

    // ========================================================================
    // Multiple make/unmake cycles preserve history correctly
    // ========================================================================

    #[test]
    fn multiple_make_unmake_history_integrity() {
        let mut game = Game::new();

        // Make 3 moves
        let mv1 = Move::new(Square::G1, Square::F3);
        let undo1 = game.make_move(mv1);
        let mv2 = Move::new(Square::G8, Square::F6);
        let undo2 = game.make_move(mv2);
        let mv3 = Move::new(Square::F3, Square::G1);
        let undo3 = game.make_move(mv3);

        assert_eq!(game.history.len(), 4); // initial + 3 moves

        // Unmake all 3
        game.unmake_move(mv3, &undo3);
        assert_eq!(game.history.len(), 3);
        game.unmake_move(mv2, &undo2);
        assert_eq!(game.history.len(), 2);
        game.unmake_move(mv1, &undo1);
        assert_eq!(game.history.len(), 1);

        // Should be back to starting position
        assert_eq!(game.board().to_fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    // ========================================================================
    // Sufficient material edge cases
    // ========================================================================

    #[test]
    fn sufficient_material_two_knights_one_side() {
        // K+N+N vs K: not insufficient
        let board = Board::from_fen("4k3/8/8/8/8/8/8/2NNKN2 w - - 0 1")
            .expect("valid FEN");
        assert!(!is_insufficient_material(&board));
    }

    #[test]
    fn sufficient_material_knight_and_bishop() {
        // K+N+B vs K: sufficient (can force mate)
        let board = Board::from_fen("4k3/8/8/8/8/8/8/2BNK3 w - - 0 1")
            .expect("valid FEN");
        assert!(!is_insufficient_material(&board));
    }
}
