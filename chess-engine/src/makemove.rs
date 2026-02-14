//! Make and unmake move operations on the board.
//!
//! These two methods are the core of the chess engine's move execution:
//!
//! - `make_move` applies a move to the board, updating all state (piece positions,
//!   castling rights, en passant square, clocks, side to move). It returns an
//!   `UndoInfo` struct that captures everything needed to reverse the move later.
//!
//! - `unmake_move` reverses a previously applied move, restoring the board to its
//!   exact prior state using the saved `UndoInfo`.
//!
//! Together, they enable the "make/unmake" pattern used in search: explore a move,
//! evaluate the resulting position, then undo it and try the next move -- all without
//! allocating a new board each time.
//!
//! For MCTS (which prefers the "copy-make" pattern), you can also do:
//! ```ignore
//! let mut new_board = board.clone();
//! new_board.make_move(mv);
//! ```
//! This is safe because `Board` derives `Clone`.

use crate::board::{Board, CastlingRights};
use crate::moves::{Move, MoveFlags};
use crate::types::{Color, Piece, Square};
use crate::zobrist;

// =============================================================================
// UndoInfo
// =============================================================================

/// Information needed to unmake a move, restoring the board to its previous state.
///
/// Every call to `make_move` returns one of these. Pass it back to `unmake_move`
/// along with the same `Move` to perfectly reverse the operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct UndoInfo {
    /// The piece that was captured (if any). For en passant, this is always a Pawn.
    pub captured_piece: Option<Piece>,

    /// The castling rights that were in effect *before* the move.
    pub castling_rights: CastlingRights,

    /// The en passant square that was in effect *before* the move.
    pub en_passant_square: Option<Square>,

    /// The halfmove clock value *before* the move.
    pub halfmove_clock: u16,

    /// The Zobrist hash of the position *before* the move.
    /// Restored on unmake to avoid recomputation.
    pub hash: u64,
}

// =============================================================================
// make_move
// =============================================================================

impl Board {
    /// Applies a move to the board and returns undo information.
    ///
    /// This method handles all move types: normal moves, captures, castling,
    /// en passant, double pawn pushes, and promotions (including promotion captures).
    ///
    /// After calling this method:
    /// - The piece has moved from `mv.from` to `mv.to`
    /// - Any captured piece has been removed
    /// - Special move mechanics (castling rook, en passant capture, promotion) are applied
    /// - Castling rights are updated (if king/rook moved or rook was captured)
    /// - En passant square is set (if double pawn push) or cleared
    /// - Halfmove clock is reset (pawn move or capture) or incremented
    /// - Fullmove number is incremented (after Black's move)
    /// - Side to move is flipped
    ///
    /// # Panics
    ///
    /// In debug mode, panics if no piece is found on `mv.from` for the side to move.
    pub fn make_move(&mut self, mv: Move) -> UndoInfo {
        let us = self.side_to_move();
        let them = us.flip();

        // Step 1: Save undo info (current state before any changes)
        let undo = UndoInfo {
            captured_piece: None, // will be set below if there is a capture
            castling_rights: self.castling_rights(),
            en_passant_square: self.en_passant_square(),
            halfmove_clock: self.halfmove_clock(),
            hash: self.hash(),
        };

        // We need a mutable copy of the undo's captured_piece field, since we may
        // discover a capture during move execution.
        let mut captured_piece: Option<Piece> = None;

        // Step 2: Determine the moving piece
        let moving_piece = self.piece_on_square_for_color(mv.from, us)
            .expect("make_move: no piece on 'from' square for the side to move");

        // --- Zobrist: XOR out old castling rights and old en passant ---
        let mut hash = self.hash();
        hash ^= zobrist::castling_key(self.castling_rights().bits());
        if let Some(ep_sq) = self.en_passant_square() {
            hash ^= zobrist::en_passant_key(ep_sq.file());
        }

        // Step 3: Handle the move based on type
        if mv.flags == MoveFlags::KINGSIDE_CASTLE || mv.flags == MoveFlags::QUEENSIDE_CASTLE {
            // --- Castling ---
            // Move the king
            self.remove_piece(mv.from, us, Piece::King);
            self.put_piece(mv.to, us, Piece::King);
            hash ^= zobrist::piece_key(us, Piece::King, mv.from);
            hash ^= zobrist::piece_key(us, Piece::King, mv.to);

            // Move the rook
            let (rook_from, rook_to) = castling_rook_squares(us, mv.flags);
            self.remove_piece(rook_from, us, Piece::Rook);
            self.put_piece(rook_to, us, Piece::Rook);
            hash ^= zobrist::piece_key(us, Piece::Rook, rook_from);
            hash ^= zobrist::piece_key(us, Piece::Rook, rook_to);

            // Clear all castling rights for this color (king has moved)
            let mut rights = self.castling_rights();
            rights.clear_color(us);
            self.set_castling_rights(rights);

        } else if mv.flags == MoveFlags::EN_PASSANT {
            // --- En passant ---
            // Move the pawn to the en passant target square
            self.remove_piece(mv.from, us, Piece::Pawn);
            self.put_piece(mv.to, us, Piece::Pawn);
            hash ^= zobrist::piece_key(us, Piece::Pawn, mv.from);
            hash ^= zobrist::piece_key(us, Piece::Pawn, mv.to);

            // Remove the captured pawn, which is on the same rank as our pawn's
            // original position (one rank "behind" the destination).
            let captured_pawn_sq = en_passant_captured_pawn_square(us, mv.to);
            self.remove_piece(captured_pawn_sq, them, Piece::Pawn);
            hash ^= zobrist::piece_key(them, Piece::Pawn, captured_pawn_sq);
            captured_piece = Some(Piece::Pawn);

        } else if mv.is_promotion() {
            // --- Promotion (possibly with capture) ---
            let promotion_piece = mv.promotion.expect("promotion flag set but no piece");

            // Remove the pawn from the origin square
            self.remove_piece(mv.from, us, Piece::Pawn);
            hash ^= zobrist::piece_key(us, Piece::Pawn, mv.from);

            // If there is a piece on the destination, it is captured
            if let Some((_cap_color, cap_piece)) = self.piece_at(mv.to) {
                self.remove_piece(mv.to, them, cap_piece);
                hash ^= zobrist::piece_key(them, cap_piece, mv.to);
                captured_piece = Some(cap_piece);
            }

            // Place the promotion piece
            self.put_piece(mv.to, us, promotion_piece);
            hash ^= zobrist::piece_key(us, promotion_piece, mv.to);

        } else if mv.flags == MoveFlags::DOUBLE_PAWN_PUSH {
            // --- Double pawn push ---
            self.remove_piece(mv.from, us, Piece::Pawn);
            self.put_piece(mv.to, us, Piece::Pawn);
            hash ^= zobrist::piece_key(us, Piece::Pawn, mv.from);
            hash ^= zobrist::piece_key(us, Piece::Pawn, mv.to);

            // Set the en passant square to the "middle" square
            let ep_square = en_passant_middle_square(us, mv.from);
            self.set_en_passant_square(Some(ep_square));

        } else {
            // --- Normal move (possibly a capture) ---
            self.remove_piece(mv.from, us, moving_piece);
            hash ^= zobrist::piece_key(us, moving_piece, mv.from);

            // Check for capture on the destination square
            if let Some((_cap_color, cap_piece)) = self.piece_at(mv.to) {
                self.remove_piece(mv.to, them, cap_piece);
                hash ^= zobrist::piece_key(them, cap_piece, mv.to);
                captured_piece = Some(cap_piece);
            }

            self.put_piece(mv.to, us, moving_piece);
            hash ^= zobrist::piece_key(us, moving_piece, mv.to);
        }

        // Step 4: Update state

        // 4a. Clear en passant square (unless we just set it via double pawn push)
        if mv.flags != MoveFlags::DOUBLE_PAWN_PUSH {
            self.set_en_passant_square(None);
        }

        // 4b. Update castling rights based on what moved and where
        //     (Castling moves already cleared their own rights above.)
        if mv.flags != MoveFlags::KINGSIDE_CASTLE && mv.flags != MoveFlags::QUEENSIDE_CASTLE {
            let mut rights = self.castling_rights();
            update_castling_rights_for_move(&mut rights, mv.from, mv.to, moving_piece, us);
            // Also update if a rook was captured on its starting square
            if let Some(cap) = captured_piece {
                update_castling_rights_for_capture(&mut rights, mv.to, cap);
            }
            self.set_castling_rights(rights);
        }

        // --- Zobrist: XOR in new castling rights and new en passant ---
        hash ^= zobrist::castling_key(self.castling_rights().bits());
        if let Some(ep_sq) = self.en_passant_square() {
            hash ^= zobrist::en_passant_key(ep_sq.file());
        }

        // 4c. Update halfmove clock: reset on pawn moves or captures, else increment
        if moving_piece == Piece::Pawn || captured_piece.is_some() {
            self.set_halfmove_clock(0);
        } else {
            self.set_halfmove_clock(self.halfmove_clock() + 1);
        }

        // 4d. Increment fullmove number after Black's move
        if us == Color::Black {
            self.set_fullmove_number(self.fullmove_number() + 1);
        }

        // 4e. Flip side to move
        self.set_side_to_move(them);
        hash ^= zobrist::side_to_move_key();
        self.set_hash(hash);

        // Return undo info with the captured piece filled in
        UndoInfo {
            captured_piece,
            ..undo
        }
    }

    /// Reverses a previously applied move, restoring the board exactly.
    ///
    /// This method must be called with the same `Move` that was passed to `make_move`,
    /// and the `UndoInfo` that was returned from that call.
    ///
    /// After calling this method, the board is identical to how it was before the
    /// corresponding `make_move` call (all bitboards, castling rights, en passant,
    /// clocks, side to move).
    pub fn unmake_move(&mut self, mv: Move, undo: &UndoInfo) {
        // Step 1: Flip side to move back (undo the flip from make_move)
        let them = self.side_to_move(); // "them" was set as side_to_move by make_move
        let us = them.flip(); // the color that actually made the move
        self.set_side_to_move(us);

        // Step 2: Decrement fullmove number if Black just moved
        if us == Color::Black {
            self.set_fullmove_number(self.fullmove_number() - 1);
        }

        // Step 3: Restore saved state from undo info
        self.set_halfmove_clock(undo.halfmove_clock);
        self.set_castling_rights(undo.castling_rights);
        self.set_en_passant_square(undo.en_passant_square);
        self.set_hash(undo.hash);

        // Step 4: Reverse the move based on type
        if mv.flags == MoveFlags::KINGSIDE_CASTLE || mv.flags == MoveFlags::QUEENSIDE_CASTLE {
            // --- Undo castling ---
            // Move the king back
            self.remove_piece(mv.to, us, Piece::King);
            self.put_piece(mv.from, us, Piece::King);

            // Move the rook back
            let (rook_from, rook_to) = castling_rook_squares(us, mv.flags);
            self.remove_piece(rook_to, us, Piece::Rook);
            self.put_piece(rook_from, us, Piece::Rook);

        } else if mv.flags == MoveFlags::EN_PASSANT {
            // --- Undo en passant ---
            // Move the pawn back
            self.remove_piece(mv.to, us, Piece::Pawn);
            self.put_piece(mv.from, us, Piece::Pawn);

            // Restore the captured pawn
            let captured_pawn_sq = en_passant_captured_pawn_square(us, mv.to);
            self.put_piece(captured_pawn_sq, them, Piece::Pawn);

        } else if mv.is_promotion() {
            // --- Undo promotion ---
            let promotion_piece = mv.promotion.expect("promotion flag set but no piece");

            // Remove the promoted piece from the destination
            self.remove_piece(mv.to, us, promotion_piece);

            // Restore the pawn on the origin square
            self.put_piece(mv.from, us, Piece::Pawn);

            // Restore captured piece (if any) on the destination
            if let Some(cap_piece) = undo.captured_piece {
                self.put_piece(mv.to, them, cap_piece);
            }

        } else {
            // --- Undo normal move (including double pawn push) ---
            // Determine the piece type that was on mv.to
            let moving_piece = self.piece_on_square_for_color(mv.to, us)
                .expect("unmake_move: no piece on 'to' square for the side that moved");

            // Move it back
            self.remove_piece(mv.to, us, moving_piece);
            self.put_piece(mv.from, us, moving_piece);

            // Restore captured piece on the destination
            if let Some(cap_piece) = undo.captured_piece {
                self.put_piece(mv.to, them, cap_piece);
            }
        }
    }

    /// Returns the piece type for the given color on the given square, if any.
    ///
    /// This is more efficient than `piece_at` when you already know the color,
    /// because it only checks 6 bitboards instead of 12.
    fn piece_on_square_for_color(&self, square: Square, color: Color) -> Option<Piece> {
        for &piece in &Piece::ALL {
            if self.piece_bitboard(color, piece).contains(square) {
                return Some(piece);
            }
        }
        None
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Returns the (rook_from, rook_to) squares for a castling move.
///
/// Given the color and castle type (kingside or queenside), returns the
/// square the rook starts on and the square it ends up on.
fn castling_rook_squares(color: Color, flags: MoveFlags) -> (Square, Square) {
    match (color, flags) {
        (Color::White, MoveFlags { .. }) if flags == MoveFlags::KINGSIDE_CASTLE => {
            (Square::H1, Square::F1)
        }
        (Color::White, MoveFlags { .. }) if flags == MoveFlags::QUEENSIDE_CASTLE => {
            (Square::A1, Square::D1)
        }
        (Color::Black, MoveFlags { .. }) if flags == MoveFlags::KINGSIDE_CASTLE => {
            (Square::H8, Square::F8)
        }
        (Color::Black, MoveFlags { .. }) if flags == MoveFlags::QUEENSIDE_CASTLE => {
            (Square::A8, Square::D8)
        }
        _ => unreachable!("castling_rook_squares called with non-castling flags"),
    }
}

/// Returns the square of the pawn captured during an en passant move.
///
/// For white capturing en passant, the captured pawn is one rank *below* the
/// destination (to - 8). For black, it is one rank *above* (to + 8).
fn en_passant_captured_pawn_square(capturing_color: Color, ep_target: Square) -> Square {
    match capturing_color {
        Color::White => Square::new(ep_target.index() - 8),
        Color::Black => Square::new(ep_target.index() + 8),
    }
}

/// Returns the en passant "middle" square for a double pawn push.
///
/// For white pushing from rank 2 to rank 4 (e.g., e2 -> e4), the en passant
/// target is on rank 3 (e3), which is from + 8. For black pushing from rank 7
/// to rank 5, the target is on rank 6, which is from - 8.
fn en_passant_middle_square(color: Color, from: Square) -> Square {
    match color {
        Color::White => Square::new(from.index() + 8),
        Color::Black => Square::new(from.index() - 8),
    }
}

/// Updates castling rights when a piece moves.
///
/// - If the king moves, clear both castling rights for that color.
/// - If a rook moves from its starting square, clear the corresponding right.
fn update_castling_rights_for_move(
    rights: &mut CastlingRights,
    from: Square,
    _to: Square,
    piece: Piece,
    color: Color,
) {
    match piece {
        Piece::King => {
            // King moved -- lose all castling rights for this color
            rights.clear_color(color);
        }
        Piece::Rook => {
            // Rook moved -- lose the right for whichever side this rook was on
            match (color, from) {
                (Color::White, sq) if sq == Square::A1 => rights.set_white_queenside(false),
                (Color::White, sq) if sq == Square::H1 => rights.set_white_kingside(false),
                (Color::Black, sq) if sq == Square::A8 => rights.set_black_queenside(false),
                (Color::Black, sq) if sq == Square::H8 => rights.set_black_kingside(false),
                _ => {} // Rook not on starting square, no rights to lose
            }
        }
        _ => {} // Other pieces don't affect castling rights
    }
}

/// Updates castling rights when an enemy piece is captured on a rook starting square.
///
/// If a rook is captured on a1/h1/a8/h8, the corresponding castling right is lost.
/// We check the *square* rather than relying on piece type because the captured piece
/// might not be a rook (e.g., if the original rook was captured earlier and another
/// piece now occupies the corner). However, the spec says to check the piece too,
/// so we do both for correctness.
fn update_castling_rights_for_capture(
    rights: &mut CastlingRights,
    capture_square: Square,
    captured_piece: Piece,
) {
    if captured_piece != Piece::Rook {
        return;
    }
    match capture_square {
        sq if sq == Square::A1 => rights.set_white_queenside(false),
        sq if sq == Square::H1 => rights.set_white_kingside(false),
        sq if sq == Square::A8 => rights.set_black_queenside(false),
        sq if sq == Square::H8 => rights.set_black_kingside(false),
        _ => {}
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::movegen::generate_pseudo_legal_moves;
    use crate::moves::{Move, MoveFlags};
    use crate::types::{Color, Piece, Square};

    // ---- Helper: assert full board equality ---------------------------------

    /// Asserts that two boards are identical in every respect: all bitboards,
    /// castling rights, en passant, clocks, and side to move.
    fn assert_boards_equal(a: &Board, b: &Board, context: &str) {
        assert_eq!(a.side_to_move(), b.side_to_move(),
            "{}: side_to_move mismatch", context);
        assert_eq!(a.castling_rights(), b.castling_rights(),
            "{}: castling_rights mismatch", context);
        assert_eq!(a.en_passant_square(), b.en_passant_square(),
            "{}: en_passant_square mismatch", context);
        assert_eq!(a.halfmove_clock(), b.halfmove_clock(),
            "{}: halfmove_clock mismatch", context);
        assert_eq!(a.fullmove_number(), b.fullmove_number(),
            "{}: fullmove_number mismatch", context);

        // Check every piece bitboard
        for &color in &Color::ALL {
            for &piece in &Piece::ALL {
                assert_eq!(
                    a.piece_bitboard(color, piece),
                    b.piece_bitboard(color, piece),
                    "{}: {:?} {:?} bitboard mismatch",
                    context, color, piece
                );
            }
        }
    }

    // ========================================================================
    // Simple pawn push
    // ========================================================================

    #[test]
    fn simple_pawn_push_e2_e3() {
        let mut board = Board::starting_position();
        let mv = Move::new(Square::E2, Square::E3);
        let undo = board.make_move(mv);

        // Pawn should be on e3, not on e2
        assert_eq!(board.piece_at(Square::E3), Some((Color::White, Piece::Pawn)));
        assert!(board.piece_at(Square::E2).is_none());

        // No capture
        assert_eq!(undo.captured_piece, None);

        // Side to move flipped
        assert_eq!(board.side_to_move(), Color::Black);

        // Halfmove clock reset (pawn move)
        assert_eq!(board.halfmove_clock(), 0);

        // Fullmove number unchanged (White just moved)
        assert_eq!(board.fullmove_number(), 1);
    }

    // ========================================================================
    // Pawn capture
    // ========================================================================

    #[test]
    fn pawn_capture() {
        // White pawn on e4, black pawn on d5 -- white captures d5
        let mut board = Board::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 5 10")
            .expect("valid FEN");
        let mv = Move::new(Square::E4, Square::D5);
        let undo = board.make_move(mv);

        // White pawn on d5, original squares empty
        assert_eq!(board.piece_at(Square::D5), Some((Color::White, Piece::Pawn)));
        assert!(board.piece_at(Square::E4).is_none());

        // Captured piece was a pawn
        assert_eq!(undo.captured_piece, Some(Piece::Pawn));

        // Halfmove clock reset (capture)
        assert_eq!(board.halfmove_clock(), 0);
    }

    // ========================================================================
    // Double pawn push
    // ========================================================================

    #[test]
    fn double_pawn_push_sets_en_passant() {
        let mut board = Board::starting_position();
        let mv = Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH);
        let _undo = board.make_move(mv);

        // Pawn on e4
        assert_eq!(board.piece_at(Square::E4), Some((Color::White, Piece::Pawn)));
        assert!(board.piece_at(Square::E2).is_none());

        // En passant square should be e3
        assert_eq!(board.en_passant_square(), Some(Square::E3));
    }

    #[test]
    fn black_double_pawn_push_sets_en_passant() {
        let mut board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
            .expect("valid FEN");
        let mv = Move::with_flags(Square::D7, Square::D5, MoveFlags::DOUBLE_PAWN_PUSH);
        let _undo = board.make_move(mv);

        // Pawn on d5
        assert_eq!(board.piece_at(Square::D5), Some((Color::Black, Piece::Pawn)));

        // En passant square should be d6
        assert_eq!(board.en_passant_square(), Some(Square::D6));

        // Fullmove incremented (Black just moved)
        assert_eq!(board.fullmove_number(), 2);
    }

    // ========================================================================
    // En passant capture
    // ========================================================================

    #[test]
    fn en_passant_capture_white() {
        // White pawn on e5, black pawn just pushed d7-d5. EP target = d6.
        let mut board = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
            .expect("valid FEN");
        let mv = Move::with_flags(Square::E5, Square::D6, MoveFlags::EN_PASSANT);
        let undo = board.make_move(mv);

        // White pawn on d6
        assert_eq!(board.piece_at(Square::D6), Some((Color::White, Piece::Pawn)));
        // Original square empty
        assert!(board.piece_at(Square::E5).is_none());
        // Captured pawn on d5 is gone
        assert!(board.piece_at(Square::D5).is_none());
        // Captured piece recorded
        assert_eq!(undo.captured_piece, Some(Piece::Pawn));
        // EP square cleared
        assert_eq!(board.en_passant_square(), None);
    }

    #[test]
    fn en_passant_capture_black() {
        // Black pawn on d4, white pawn just pushed e2-e4. EP target = e3.
        let mut board = Board::from_fen("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")
            .expect("valid FEN");
        let mv = Move::with_flags(Square::D4, Square::E3, MoveFlags::EN_PASSANT);
        let undo = board.make_move(mv);

        // Black pawn on e3
        assert_eq!(board.piece_at(Square::E3), Some((Color::Black, Piece::Pawn)));
        // Original square empty
        assert!(board.piece_at(Square::D4).is_none());
        // Captured pawn on e4 is gone
        assert!(board.piece_at(Square::E4).is_none());
        assert_eq!(undo.captured_piece, Some(Piece::Pawn));
    }

    // ========================================================================
    // Kingside castling
    // ========================================================================

    #[test]
    fn white_kingside_castling() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
            .expect("valid FEN");
        let mv = Move::with_flags(Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE);
        let _undo = board.make_move(mv);

        // King on g1, rook on f1
        assert_eq!(board.piece_at(Square::G1), Some((Color::White, Piece::King)));
        assert_eq!(board.piece_at(Square::F1), Some((Color::White, Piece::Rook)));
        // Original squares empty
        assert!(board.piece_at(Square::E1).is_none());
        assert!(board.piece_at(Square::H1).is_none());
        // White castling rights cleared
        assert!(!board.castling_rights().white_kingside());
        assert!(!board.castling_rights().white_queenside());
    }

    #[test]
    fn white_queenside_castling() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
            .expect("valid FEN");
        let mv = Move::with_flags(Square::E1, Square::C1, MoveFlags::QUEENSIDE_CASTLE);
        let _undo = board.make_move(mv);

        // King on c1, rook on d1
        assert_eq!(board.piece_at(Square::C1), Some((Color::White, Piece::King)));
        assert_eq!(board.piece_at(Square::D1), Some((Color::White, Piece::Rook)));
        // Original squares empty
        assert!(board.piece_at(Square::E1).is_none());
        assert!(board.piece_at(Square::A1).is_none());
    }

    #[test]
    fn black_kingside_castling() {
        let mut board = Board::from_fen("4k2r/8/8/8/8/8/8/4K3 b k - 0 1")
            .expect("valid FEN");
        let mv = Move::with_flags(Square::E8, Square::G8, MoveFlags::KINGSIDE_CASTLE);
        let _undo = board.make_move(mv);

        assert_eq!(board.piece_at(Square::G8), Some((Color::Black, Piece::King)));
        assert_eq!(board.piece_at(Square::F8), Some((Color::Black, Piece::Rook)));
        assert!(board.piece_at(Square::E8).is_none());
        assert!(board.piece_at(Square::H8).is_none());
        assert!(!board.castling_rights().black_kingside());
    }

    #[test]
    fn black_queenside_castling() {
        let mut board = Board::from_fen("r3k3/8/8/8/8/8/8/4K3 b q - 0 1")
            .expect("valid FEN");
        let mv = Move::with_flags(Square::E8, Square::C8, MoveFlags::QUEENSIDE_CASTLE);
        let _undo = board.make_move(mv);

        assert_eq!(board.piece_at(Square::C8), Some((Color::Black, Piece::King)));
        assert_eq!(board.piece_at(Square::D8), Some((Color::Black, Piece::Rook)));
        assert!(board.piece_at(Square::E8).is_none());
        assert!(board.piece_at(Square::A8).is_none());
    }

    // ========================================================================
    // Promotion
    // ========================================================================

    #[test]
    fn promotion_to_queen() {
        let mut board = Board::from_fen("8/4P3/8/8/8/6k1/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let mv = Move::with_promotion(Square::E7, Square::E8, Piece::Queen);
        let undo = board.make_move(mv);

        // Queen on e8, not a pawn
        assert_eq!(board.piece_at(Square::E8), Some((Color::White, Piece::Queen)));
        assert!(board.piece_at(Square::E7).is_none());
        assert_eq!(undo.captured_piece, None);
    }

    #[test]
    fn promotion_to_knight() {
        let mut board = Board::from_fen("8/4P3/8/8/8/6k1/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let mv = Move::with_promotion(Square::E7, Square::E8, Piece::Knight);
        board.make_move(mv);

        assert_eq!(board.piece_at(Square::E8), Some((Color::White, Piece::Knight)));
    }

    #[test]
    fn promotion_to_rook() {
        let mut board = Board::from_fen("8/4P3/8/8/8/6k1/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let mv = Move::with_promotion(Square::E7, Square::E8, Piece::Rook);
        board.make_move(mv);

        assert_eq!(board.piece_at(Square::E8), Some((Color::White, Piece::Rook)));
    }

    #[test]
    fn promotion_to_bishop() {
        let mut board = Board::from_fen("8/4P3/8/8/8/6k1/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let mv = Move::with_promotion(Square::E7, Square::E8, Piece::Bishop);
        board.make_move(mv);

        assert_eq!(board.piece_at(Square::E8), Some((Color::White, Piece::Bishop)));
    }

    // ========================================================================
    // Promotion capture
    // ========================================================================

    #[test]
    fn promotion_capture() {
        // White pawn on e7 captures black rook on d8 and promotes to queen
        let mut board = Board::from_fen("3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let mv = Move::with_promotion(Square::E7, Square::D8, Piece::Queen);
        let undo = board.make_move(mv);

        assert_eq!(board.piece_at(Square::D8), Some((Color::White, Piece::Queen)));
        assert!(board.piece_at(Square::E7).is_none());
        assert_eq!(undo.captured_piece, Some(Piece::Rook));
    }

    // ========================================================================
    // Castling rights updates
    // ========================================================================

    #[test]
    fn king_move_clears_both_castling_rights() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
            .expect("valid FEN");
        let mv = Move::new(Square::E1, Square::F1);
        board.make_move(mv);

        assert!(!board.castling_rights().white_kingside());
        assert!(!board.castling_rights().white_queenside());
    }

    #[test]
    fn rook_move_from_a1_clears_white_queenside() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
            .expect("valid FEN");
        let mv = Move::new(Square::A1, Square::B1);
        board.make_move(mv);

        // Queenside lost, kingside preserved
        assert!(!board.castling_rights().white_queenside());
        assert!(board.castling_rights().white_kingside());
    }

    #[test]
    fn rook_move_from_h1_clears_white_kingside() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
            .expect("valid FEN");
        let mv = Move::new(Square::H1, Square::G1);
        board.make_move(mv);

        assert!(board.castling_rights().white_queenside());
        assert!(!board.castling_rights().white_kingside());
    }

    #[test]
    fn capture_of_rook_on_a8_clears_black_queenside() {
        // White rook captures black rook on a8
        let mut board = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K3 w Qkq - 0 1")
            .expect("valid FEN");
        let mv = Move::new(Square::A1, Square::A8);
        let undo = board.make_move(mv);

        assert_eq!(undo.captured_piece, Some(Piece::Rook));
        // Black's queenside castling right should be gone
        assert!(!board.castling_rights().black_queenside());
        // Black's kingside should still be there
        assert!(board.castling_rights().black_kingside());
    }

    #[test]
    fn capture_of_rook_on_h8_clears_black_kingside() {
        // White rook on h1 captures black rook on h8
        let mut board = Board::from_fen("4k2r/8/8/8/8/8/8/4K2R w Kk - 0 1")
            .expect("valid FEN");
        let mv = Move::new(Square::H1, Square::H8);
        let undo = board.make_move(mv);

        assert_eq!(undo.captured_piece, Some(Piece::Rook));
        // Black's kingside right removed (rook captured on h8)
        assert!(!board.castling_rights().black_kingside());
        // White's kingside right also removed (rook moved from h1)
        assert!(!board.castling_rights().white_kingside());
    }

    #[test]
    fn capture_of_rook_on_a1_clears_white_queenside() {
        // Black captures white rook on a1
        let mut board = Board::from_fen("4k3/r7/8/8/8/8/8/R3K2R b KQ - 0 1")
            .expect("valid FEN");
        let mv = Move::new(Square::A7, Square::A1);
        let undo = board.make_move(mv);

        assert_eq!(undo.captured_piece, Some(Piece::Rook));
        assert!(!board.castling_rights().white_queenside());
        assert!(board.castling_rights().white_kingside());
    }

    #[test]
    fn capture_of_rook_on_h1_clears_white_kingside() {
        let mut board = Board::from_fen("4k3/7r/8/8/8/8/8/R3K2R b KQ - 0 1")
            .expect("valid FEN");
        let mv = Move::new(Square::H7, Square::H1);
        let undo = board.make_move(mv);

        assert_eq!(undo.captured_piece, Some(Piece::Rook));
        assert!(!board.castling_rights().white_kingside());
        assert!(board.castling_rights().white_queenside());
    }

    // ========================================================================
    // Halfmove clock
    // ========================================================================

    #[test]
    fn halfmove_clock_reset_on_pawn_move() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 10 5")
            .expect("valid FEN");
        let mv = Move::new(Square::E2, Square::E3);
        board.make_move(mv);

        assert_eq!(board.halfmove_clock(), 0);
    }

    #[test]
    fn halfmove_clock_reset_on_capture() {
        // Rook on e5 captures black pawn on e4
        let mut board = Board::from_fen("4k3/8/8/4R3/4p3/8/8/4K3 w - - 10 5")
            .expect("valid FEN");
        let mv = Move::new(Square::E5, Square::E4);
        let undo = board.make_move(mv);

        assert_eq!(undo.captured_piece, Some(Piece::Pawn));
        assert_eq!(board.halfmove_clock(), 0);
    }

    #[test]
    fn halfmove_clock_increments_on_quiet_move() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 5 10")
            .expect("valid FEN");
        let mv = Move::new(Square::A1, Square::A2);
        board.make_move(mv);

        assert_eq!(board.halfmove_clock(), 6);
    }

    // ========================================================================
    // Fullmove number
    // ========================================================================

    #[test]
    fn fullmove_increments_after_black_move() {
        let mut board = Board::from_fen("4k3/4p3/8/8/8/8/4P3/4K3 b - - 0 5")
            .expect("valid FEN");
        let mv = Move::new(Square::E7, Square::E6);
        board.make_move(mv);

        assert_eq!(board.fullmove_number(), 6);
    }

    #[test]
    fn fullmove_unchanged_after_white_move() {
        let mut board = Board::from_fen("4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 5")
            .expect("valid FEN");
        let mv = Move::new(Square::E2, Square::E3);
        board.make_move(mv);

        assert_eq!(board.fullmove_number(), 5);
    }

    // ========================================================================
    // Side to move flips
    // ========================================================================

    #[test]
    fn side_to_move_flips_on_every_make() {
        let mut board = Board::starting_position();
        assert_eq!(board.side_to_move(), Color::White);

        let mv = Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH);
        board.make_move(mv);
        assert_eq!(board.side_to_move(), Color::Black);

        let mv = Move::with_flags(Square::E7, Square::E5, MoveFlags::DOUBLE_PAWN_PUSH);
        board.make_move(mv);
        assert_eq!(board.side_to_move(), Color::White);
    }

    // ========================================================================
    // Roundtrip: make then unmake restores board exactly
    // ========================================================================

    #[test]
    fn roundtrip_simple_pawn_push() {
        let mut board = Board::starting_position();
        let original = board.clone();

        let mv = Move::new(Square::E2, Square::E3);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "simple pawn push roundtrip");
    }

    #[test]
    fn roundtrip_double_pawn_push() {
        let mut board = Board::starting_position();
        let original = board.clone();

        let mv = Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "double pawn push roundtrip");
    }

    #[test]
    fn roundtrip_pawn_capture() {
        let mut board = Board::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 5 10")
            .expect("valid FEN");
        let original = board.clone();

        let mv = Move::new(Square::E4, Square::D5);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "pawn capture roundtrip");
    }

    #[test]
    fn roundtrip_en_passant() {
        let mut board = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
            .expect("valid FEN");
        let original = board.clone();

        let mv = Move::with_flags(Square::E5, Square::D6, MoveFlags::EN_PASSANT);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "en passant roundtrip");
    }

    #[test]
    fn roundtrip_white_kingside_castling() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
            .expect("valid FEN");
        let original = board.clone();

        let mv = Move::with_flags(Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "white kingside castling roundtrip");
    }

    #[test]
    fn roundtrip_white_queenside_castling() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
            .expect("valid FEN");
        let original = board.clone();

        let mv = Move::with_flags(Square::E1, Square::C1, MoveFlags::QUEENSIDE_CASTLE);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "white queenside castling roundtrip");
    }

    #[test]
    fn roundtrip_black_kingside_castling() {
        let mut board = Board::from_fen("4k2r/8/8/8/8/8/8/4K3 b k - 0 1")
            .expect("valid FEN");
        let original = board.clone();

        let mv = Move::with_flags(Square::E8, Square::G8, MoveFlags::KINGSIDE_CASTLE);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "black kingside castling roundtrip");
    }

    #[test]
    fn roundtrip_black_queenside_castling() {
        let mut board = Board::from_fen("r3k3/8/8/8/8/8/8/4K3 b q - 0 1")
            .expect("valid FEN");
        let original = board.clone();

        let mv = Move::with_flags(Square::E8, Square::C8, MoveFlags::QUEENSIDE_CASTLE);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "black queenside castling roundtrip");
    }

    #[test]
    fn roundtrip_promotion() {
        let mut board = Board::from_fen("8/4P3/8/8/8/6k1/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let original = board.clone();

        let mv = Move::with_promotion(Square::E7, Square::E8, Piece::Queen);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "promotion roundtrip");
    }

    #[test]
    fn roundtrip_promotion_capture() {
        let mut board = Board::from_fen("3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let original = board.clone();

        let mv = Move::with_promotion(Square::E7, Square::D8, Piece::Queen);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "promotion capture roundtrip");
    }

    #[test]
    fn roundtrip_knight_under_promotion() {
        let mut board = Board::from_fen("8/4P3/8/8/8/6k1/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let original = board.clone();

        let mv = Move::with_promotion(Square::E7, Square::E8, Piece::Knight);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "knight under-promotion roundtrip");
    }

    #[test]
    fn roundtrip_rook_capture_on_starting_square() {
        // White rook captures black rook on a8 (loses black queenside castling right)
        let mut board = Board::from_fen("r3k2r/8/8/8/8/8/8/R3K3 w Qkq - 0 1")
            .expect("valid FEN");
        let original = board.clone();

        let mv = Move::new(Square::A1, Square::A8);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "rook capture on starting square roundtrip");
    }

    // ========================================================================
    // Copy-make for MCTS
    // ========================================================================

    #[test]
    fn copy_make_preserves_original() {
        let board = Board::starting_position();
        let mut new_board = board.clone();

        let mv = Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH);
        new_board.make_move(mv);

        // Original should be untouched
        assert_eq!(board, Board::starting_position());

        // New board should be different
        assert_ne!(board, new_board);
        assert_eq!(new_board.piece_at(Square::E4), Some((Color::White, Piece::Pawn)));
    }

    // ========================================================================
    // Make/unmake all starting moves
    // ========================================================================

    #[test]
    fn make_unmake_all_starting_position_moves() {
        let original = Board::starting_position();
        let moves = generate_pseudo_legal_moves(&original);
        assert_eq!(moves.len(), 20, "Starting position should have 20 moves");

        for mv in &moves {
            let mut board = original.clone();
            let undo = board.make_move(*mv);
            board.unmake_move(*mv, &undo);

            assert_boards_equal(
                &board,
                &original,
                &format!("roundtrip failed for move {}", mv.to_uci()),
            );
        }
    }

    // ========================================================================
    // Roundtrip with various FEN positions
    // ========================================================================

    #[test]
    fn roundtrip_all_moves_from_kiwipete() {
        let original = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&original);

        for mv in &moves {
            let mut board = original.clone();
            let undo = board.make_move(*mv);
            board.unmake_move(*mv, &undo);

            assert_boards_equal(
                &board,
                &original,
                &format!("kiwipete roundtrip failed for move {}", mv.to_uci()),
            );
        }
    }

    #[test]
    fn roundtrip_all_moves_from_position_3() {
        let original = Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&original);

        for mv in &moves {
            let mut board = original.clone();
            let undo = board.make_move(*mv);
            board.unmake_move(*mv, &undo);

            assert_boards_equal(
                &board,
                &original,
                &format!("position 3 roundtrip failed for move {}", mv.to_uci()),
            );
        }
    }

    #[test]
    fn roundtrip_all_moves_from_position_4() {
        let original = Board::from_fen(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        )
        .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&original);

        for mv in &moves {
            let mut board = original.clone();
            let undo = board.make_move(*mv);
            board.unmake_move(*mv, &undo);

            assert_boards_equal(
                &board,
                &original,
                &format!("position 4 roundtrip failed for move {}", mv.to_uci()),
            );
        }
    }

    #[test]
    fn roundtrip_all_moves_from_position_5() {
        let original = Board::from_fen(
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        )
        .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&original);

        for mv in &moves {
            let mut board = original.clone();
            let undo = board.make_move(*mv);
            board.unmake_move(*mv, &undo);

            assert_boards_equal(
                &board,
                &original,
                &format!("position 5 roundtrip failed for move {}", mv.to_uci()),
            );
        }
    }

    #[test]
    fn roundtrip_all_moves_from_position_with_en_passant() {
        // Position where en passant is available
        let original = Board::from_fen(
            "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3",
        )
        .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&original);

        for mv in &moves {
            let mut board = original.clone();
            let undo = board.make_move(*mv);
            board.unmake_move(*mv, &undo);

            assert_boards_equal(
                &board,
                &original,
                &format!("en passant position roundtrip failed for move {}", mv.to_uci()),
            );
        }
    }

    // ========================================================================
    // Sequence of moves: make two moves, unmake two
    // ========================================================================

    #[test]
    fn two_move_sequence_roundtrip() {
        let original = Board::starting_position();
        let mut board = original.clone();

        // 1. e4
        let mv1 = Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH);
        let undo1 = board.make_move(mv1);

        // 1... e5
        let mv2 = Move::with_flags(Square::E7, Square::E5, MoveFlags::DOUBLE_PAWN_PUSH);
        let undo2 = board.make_move(mv2);

        // Verify intermediate state
        assert_eq!(board.side_to_move(), Color::White);
        assert_eq!(board.fullmove_number(), 2);
        assert_eq!(board.en_passant_square(), Some(Square::E6));

        // Unmake in reverse order
        board.unmake_move(mv2, &undo2);

        // After undoing Black's move, should be Black to move again
        assert_eq!(board.side_to_move(), Color::Black);
        assert_eq!(board.en_passant_square(), Some(Square::E3));

        board.unmake_move(mv1, &undo1);

        assert_boards_equal(&board, &original, "two-move sequence roundtrip");
    }

    // ========================================================================
    // En passant square is cleared by non-double-push moves
    // ========================================================================

    #[test]
    fn en_passant_cleared_after_normal_move() {
        // Position with en passant available, but we play a different move
        let mut board = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        )
        .expect("valid FEN");

        // Black plays Nc6 (a non-pawn, non-double-push move)
        let mv = Move::new(Square::B8, Square::C6);
        board.make_move(mv);

        // En passant should be cleared
        assert_eq!(board.en_passant_square(), None);
    }

    // ========================================================================
    // Black promotion
    // ========================================================================

    #[test]
    fn black_promotion_to_queen() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/3p4/4K3 b - - 0 1")
            .expect("valid FEN");
        let mv = Move::with_promotion(Square::D2, Square::D1, Piece::Queen);
        let undo = board.make_move(mv);

        assert_eq!(board.piece_at(Square::D1), Some((Color::Black, Piece::Queen)));
        assert!(board.piece_at(Square::D2).is_none());
        assert_eq!(undo.captured_piece, None);
    }

    #[test]
    fn roundtrip_black_promotion() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/3p4/4K3 b - - 0 1")
            .expect("valid FEN");
        let original = board.clone();

        let mv = Move::with_promotion(Square::D2, Square::D1, Piece::Queen);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "black promotion roundtrip");
    }

    // ========================================================================
    // Non-pawn, non-capture quiet move increments halfmove clock
    // ========================================================================

    #[test]
    fn knight_move_increments_halfmove_clock() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/4K1N1 w - - 7 20")
            .expect("valid FEN");
        let mv = Move::new(Square::G1, Square::F3);
        board.make_move(mv);

        assert_eq!(board.halfmove_clock(), 8);
    }

    // ========================================================================
    // Multiple make/unmake cycles on the same board
    // ========================================================================

    #[test]
    fn repeated_make_unmake_same_move() {
        let mut board = Board::starting_position();
        let original = board.clone();

        let mv = Move::new(Square::G1, Square::F3);

        for _ in 0..10 {
            let undo = board.make_move(mv);
            board.unmake_move(mv, &undo);
            assert_boards_equal(&board, &original, "repeated make/unmake");
        }
    }

    // ========================================================================
    // Castling with full rights; verify other color's rights preserved
    // ========================================================================

    #[test]
    fn white_castling_preserves_black_rights() {
        let mut board = Board::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
            .expect("valid FEN");
        let mv = Move::with_flags(Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE);
        board.make_move(mv);

        // White rights gone
        assert!(!board.castling_rights().white_kingside());
        assert!(!board.castling_rights().white_queenside());
        // Black rights preserved
        assert!(board.castling_rights().black_kingside());
        assert!(board.castling_rights().black_queenside());
    }

    // ========================================================================
    // Edge case: promoting pawn captures and creates the promotion piece
    // ========================================================================

    #[test]
    fn black_promotion_capture_to_knight() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/1p6/R1B1K3 b - - 0 1")
            .expect("valid FEN");
        let mv = Move::with_promotion(Square::B2, Square::A1, Piece::Knight);
        let undo = board.make_move(mv);

        assert_eq!(board.piece_at(Square::A1), Some((Color::Black, Piece::Knight)));
        assert!(board.piece_at(Square::B2).is_none());
        assert_eq!(undo.captured_piece, Some(Piece::Rook));
    }

    #[test]
    fn roundtrip_black_promotion_capture() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/1p6/R1B1K3 b - - 0 1")
            .expect("valid FEN");
        let original = board.clone();

        let mv = Move::with_promotion(Square::B2, Square::A1, Piece::Knight);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_boards_equal(&board, &original, "black promotion capture roundtrip");
    }

    // ========================================================================
    // Verify make_move returns correct undo info
    // ========================================================================

    #[test]
    fn undo_info_preserves_previous_state() {
        let mut board = Board::from_fen(
            "r3k2r/pppppppp/8/8/4P3/8/PPPP1PPP/R3K2R b KQkq e3 3 15",
        )
        .expect("valid FEN");

        let mv = Move::new(Square::A7, Square::A6);
        let undo = board.make_move(mv);

        // Undo should have captured the pre-move state
        assert_eq!(undo.castling_rights, CastlingRights::ALL);
        assert_eq!(undo.en_passant_square, Some(Square::E3));
        assert_eq!(undo.halfmove_clock, 3);
        assert_eq!(undo.captured_piece, None);
    }

    // ========================================================================
    // Regression-style: castling then capture removes correct rights
    // ========================================================================

    #[test]
    fn rook_move_from_h8_clears_black_kingside() {
        let mut board = Board::from_fen("4k2r/8/8/8/8/8/8/4K3 b kq - 0 1")
            .expect("valid FEN");
        let mv = Move::new(Square::H8, Square::H7);
        board.make_move(mv);

        assert!(!board.castling_rights().black_kingside());
        assert!(board.castling_rights().black_queenside());
    }

    #[test]
    fn rook_move_from_a8_clears_black_queenside() {
        let mut board = Board::from_fen("r3k3/8/8/8/8/8/8/4K3 b kq - 0 1")
            .expect("valid FEN");
        let mv = Move::new(Square::A8, Square::A7);
        board.make_move(mv);

        assert!(board.castling_rights().black_kingside());
        assert!(!board.castling_rights().black_queenside());
    }

    // ========================================================================
    // Make multiple moves in sequence, verify full state
    // ========================================================================

    #[test]
    fn full_game_fragment_1_e4_e5_nf3() {
        let mut board = Board::starting_position();

        // 1. e4
        let mv1 = Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH);
        let _undo1 = board.make_move(mv1);
        assert_eq!(board.en_passant_square(), Some(Square::E3));
        assert_eq!(board.side_to_move(), Color::Black);
        assert_eq!(board.fullmove_number(), 1);

        // 1... e5
        let mv2 = Move::with_flags(Square::E7, Square::E5, MoveFlags::DOUBLE_PAWN_PUSH);
        let _undo2 = board.make_move(mv2);
        assert_eq!(board.en_passant_square(), Some(Square::E6));
        assert_eq!(board.side_to_move(), Color::White);
        assert_eq!(board.fullmove_number(), 2);

        // 2. Nf3
        let mv3 = Move::new(Square::G1, Square::F3);
        let _undo3 = board.make_move(mv3);
        assert_eq!(board.en_passant_square(), None); // cleared
        assert_eq!(board.side_to_move(), Color::Black);
        assert_eq!(board.fullmove_number(), 2);
        assert_eq!(board.halfmove_clock(), 1); // knight move, not pawn/capture
    }
}
