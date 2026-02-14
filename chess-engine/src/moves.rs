//! Chess move representation.
//!
//! A `Move` encodes everything needed to apply (and later undo) a chess move:
//! the origin and destination squares, an optional promotion piece, and flags
//! that distinguish special move types (castling, en passant, double pawn push).
//!
//! We use explicit named fields rather than packing bits into a u16/u32.
//! This is slightly less compact but far more readable, which aligns with
//! the project's educational goals.

use std::fmt;

use crate::types::{Piece, Square};

// =============================================================================
// MoveFlags
// =============================================================================

/// Flags that distinguish special move types.
///
/// In standard chess there are four "special" move mechanics that cannot be
/// inferred from just the from/to squares:
///
/// 1. **Castling**: The king moves two squares and the rook jumps over.
/// 2. **En passant**: A pawn captures diagonally but the captured pawn
///    is on a different square than the destination.
/// 3. **Double pawn push**: A pawn advances two ranks from its starting
///    position, which creates an en passant opportunity for the opponent.
/// 4. **Promotion**: A pawn reaching the last rank becomes another piece.
///    (Promotion is tracked via the `promotion` field on `Move`, not here.)
///
/// A move has at most one of these flags set. A "quiet" move (normal piece
/// movement or single pawn push) has `flags == MoveFlags::NONE`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MoveFlags(u8);

impl MoveFlags {
    /// Ordinary move (no special mechanics).
    pub const NONE: MoveFlags = MoveFlags(0);

    /// Kingside castling (O-O). The king moves two squares toward the h-file.
    pub const KINGSIDE_CASTLE: MoveFlags = MoveFlags(1);

    /// Queenside castling (O-O-O). The king moves two squares toward the a-file.
    pub const QUEENSIDE_CASTLE: MoveFlags = MoveFlags(2);

    /// En passant capture. The capturing pawn moves diagonally, and the
    /// captured pawn (which is on the same rank as the capturing pawn,
    /// not on the destination square) is removed.
    pub const EN_PASSANT: MoveFlags = MoveFlags(3);

    /// Double pawn push from the starting rank. This creates an en passant
    /// target square for the opponent on the next move.
    pub const DOUBLE_PAWN_PUSH: MoveFlags = MoveFlags(4);

    /// Returns `true` if this is a castling move (either side).
    #[inline]
    pub const fn is_castle(self) -> bool {
        self.0 == Self::KINGSIDE_CASTLE.0 || self.0 == Self::QUEENSIDE_CASTLE.0
    }

    /// Returns `true` if no special flags are set.
    #[inline]
    pub const fn is_quiet(self) -> bool {
        self.0 == Self::NONE.0
    }
}

// =============================================================================
// Move
// =============================================================================

/// A chess move.
///
/// Contains all the information needed to apply the move to a board position:
/// - `from`: the square the piece is moving from
/// - `to`: the square the piece is moving to
/// - `promotion`: if a pawn reaches the last rank, which piece it becomes
///   (Knight, Bishop, Rook, or Queen). `None` for non-promotion moves.
/// - `flags`: special move flags (castling, en passant, double pawn push)
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<Piece>,
    pub flags: MoveFlags,
}

impl Move {
    /// Creates an ordinary (non-special) move.
    #[inline]
    pub const fn new(from: Square, to: Square) -> Self {
        Move {
            from,
            to,
            promotion: None,
            flags: MoveFlags::NONE,
        }
    }

    /// Creates a promotion move.
    ///
    /// `promotion_piece` should be Knight, Bishop, Rook, or Queen.
    #[inline]
    pub const fn with_promotion(from: Square, to: Square, promotion_piece: Piece) -> Self {
        Move {
            from,
            to,
            promotion: Some(promotion_piece),
            flags: MoveFlags::NONE,
        }
    }

    /// Creates a move with the given flags (for castling, en passant, etc.).
    #[inline]
    pub const fn with_flags(from: Square, to: Square, flags: MoveFlags) -> Self {
        Move {
            from,
            to,
            promotion: None,
            flags,
        }
    }

    /// Returns `true` if this is a promotion move.
    #[inline]
    pub const fn is_promotion(&self) -> bool {
        self.promotion.is_some()
    }

    /// Returns `true` if this is a castling move.
    #[inline]
    pub const fn is_castle(&self) -> bool {
        self.flags.is_castle()
    }

    /// Returns `true` if this is an en passant capture.
    #[inline]
    pub const fn is_en_passant(&self) -> bool {
        self.flags.0 == MoveFlags::EN_PASSANT.0
    }

    /// Returns `true` if this is a double pawn push.
    #[inline]
    pub const fn is_double_pawn_push(&self) -> bool {
        self.flags.0 == MoveFlags::DOUBLE_PAWN_PUSH.0
    }

    /// Returns `true` if this is a quiet (non-special) move.
    /// Note: a capture that is not en passant is still "quiet" by this
    /// definition, since the special-ness refers to move mechanics, not
    /// whether a piece is captured.
    #[inline]
    pub const fn is_quiet(&self) -> bool {
        self.flags.is_quiet() && self.promotion.is_none()
    }

    /// Returns the move in UCI long algebraic notation (e.g., "e2e4", "e7e8q").
    ///
    /// This is the format used by the Universal Chess Interface protocol:
    /// - Four characters for from/to squares
    /// - An optional fifth character for promotion piece (lowercase)
    pub fn to_uci(&self) -> String {
        let mut s = format!("{}{}", self.from, self.to);
        if let Some(piece) = self.promotion {
            // UCI uses lowercase for promotion piece
            let c = match piece {
                Piece::Knight => 'n',
                Piece::Bishop => 'b',
                Piece::Rook => 'r',
                Piece::Queen => 'q',
                _ => unreachable!("only N, B, R, Q are valid promotion pieces"),
            };
            s.push(c);
        }
        s
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Move({})", self.to_uci())?;
        if self.flags != MoveFlags::NONE {
            write!(f, " [{:?}]", self.flags)?;
        }
        Ok(())
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;

    #[test]
    fn move_new_basic() {
        let mv = Move::new(Square::E2, Square::E4);
        assert_eq!(mv.from, Square::E2);
        assert_eq!(mv.to, Square::E4);
        assert_eq!(mv.promotion, None);
        assert_eq!(mv.flags, MoveFlags::NONE);
        assert!(mv.is_quiet());
        assert!(!mv.is_promotion());
        assert!(!mv.is_castle());
        assert!(!mv.is_en_passant());
        assert!(!mv.is_double_pawn_push());
    }

    #[test]
    fn move_double_pawn_push() {
        let mv = Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH);
        assert!(mv.is_double_pawn_push());
        assert!(!mv.is_quiet());
        assert!(!mv.is_castle());
    }

    #[test]
    fn move_kingside_castle() {
        // White kingside castling: King e1 -> g1
        let mv = Move::with_flags(Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE);
        assert!(mv.is_castle());
        assert!(!mv.is_quiet());
        assert_eq!(mv.flags, MoveFlags::KINGSIDE_CASTLE);
    }

    #[test]
    fn move_queenside_castle() {
        // White queenside castling: King e1 -> c1
        let mv = Move::with_flags(Square::E1, Square::C1, MoveFlags::QUEENSIDE_CASTLE);
        assert!(mv.is_castle());
        assert!(!mv.is_quiet());
        assert_eq!(mv.flags, MoveFlags::QUEENSIDE_CASTLE);
    }

    #[test]
    fn move_en_passant() {
        // White pawn on e5 captures en passant on d6
        let mv = Move::with_flags(Square::E5, Square::D6, MoveFlags::EN_PASSANT);
        assert!(mv.is_en_passant());
        assert!(!mv.is_quiet());
        assert!(!mv.is_castle());
    }

    #[test]
    fn move_promotion_queen() {
        let mv = Move::with_promotion(Square::E7, Square::E8, Piece::Queen);
        assert!(mv.is_promotion());
        assert!(!mv.is_quiet());
        assert_eq!(mv.promotion, Some(Piece::Queen));
    }

    #[test]
    fn move_promotion_knight() {
        // Under-promotions are important in chess!
        let mv = Move::with_promotion(Square::A7, Square::A8, Piece::Knight);
        assert!(mv.is_promotion());
        assert_eq!(mv.promotion, Some(Piece::Knight));
    }

    #[test]
    fn move_promotion_all_types() {
        let promotion_pieces = [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen];
        for piece in promotion_pieces {
            let mv = Move::with_promotion(Square::D7, Square::D8, piece);
            assert!(mv.is_promotion());
            assert_eq!(mv.promotion, Some(piece));
        }
    }

    #[test]
    fn move_uci_notation_basic() {
        let mv = Move::new(Square::E2, Square::E4);
        assert_eq!(mv.to_uci(), "e2e4");
    }

    #[test]
    fn move_uci_notation_promotion() {
        let mv = Move::with_promotion(Square::E7, Square::E8, Piece::Queen);
        assert_eq!(mv.to_uci(), "e7e8q");

        let mv = Move::with_promotion(Square::A7, Square::B8, Piece::Knight);
        assert_eq!(mv.to_uci(), "a7b8n");

        let mv = Move::with_promotion(Square::C7, Square::C8, Piece::Rook);
        assert_eq!(mv.to_uci(), "c7c8r");

        let mv = Move::with_promotion(Square::H7, Square::H8, Piece::Bishop);
        assert_eq!(mv.to_uci(), "h7h8b");
    }

    #[test]
    fn move_uci_notation_castling() {
        let mv = Move::with_flags(Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE);
        assert_eq!(mv.to_uci(), "e1g1");

        let mv = Move::with_flags(Square::E8, Square::C8, MoveFlags::QUEENSIDE_CASTLE);
        assert_eq!(mv.to_uci(), "e8c8");
    }

    #[test]
    fn move_display_matches_uci() {
        let mv = Move::new(Square::D2, Square::D4);
        assert_eq!(format!("{}", mv), "d2d4");

        let mv = Move::with_promotion(Square::E7, Square::E8, Piece::Queen);
        assert_eq!(format!("{}", mv), "e7e8q");
    }

    #[test]
    fn move_equality() {
        let mv1 = Move::new(Square::E2, Square::E4);
        let mv2 = Move::new(Square::E2, Square::E4);
        let mv3 = Move::new(Square::E2, Square::E3);
        assert_eq!(mv1, mv2);
        assert_ne!(mv1, mv3);
    }

    #[test]
    fn move_equality_flags_matter() {
        // A quiet e2e4 is not the same as a double-pawn-push e2e4.
        let quiet = Move::new(Square::E2, Square::E4);
        let double = Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH);
        assert_ne!(quiet, double);
    }

    #[test]
    fn move_equality_promotion_matters() {
        let queen_promo = Move::with_promotion(Square::E7, Square::E8, Piece::Queen);
        let knight_promo = Move::with_promotion(Square::E7, Square::E8, Piece::Knight);
        let no_promo = Move::new(Square::E7, Square::E8);
        assert_ne!(queen_promo, knight_promo);
        assert_ne!(queen_promo, no_promo);
    }

    #[test]
    fn move_is_copy() {
        // Verify Move is Copy (this won't compile if it isn't).
        let mv = Move::new(Square::A1, Square::A2);
        let mv2 = mv;
        assert_eq!(mv, mv2); // both are valid — mv wasn't moved-from
    }

    #[test]
    fn moveflags_is_quiet() {
        assert!(MoveFlags::NONE.is_quiet());
        assert!(!MoveFlags::KINGSIDE_CASTLE.is_quiet());
        assert!(!MoveFlags::QUEENSIDE_CASTLE.is_quiet());
        assert!(!MoveFlags::EN_PASSANT.is_quiet());
        assert!(!MoveFlags::DOUBLE_PAWN_PUSH.is_quiet());
    }

    #[test]
    fn moveflags_is_castle() {
        assert!(MoveFlags::KINGSIDE_CASTLE.is_castle());
        assert!(MoveFlags::QUEENSIDE_CASTLE.is_castle());
        assert!(!MoveFlags::NONE.is_castle());
        assert!(!MoveFlags::EN_PASSANT.is_castle());
        assert!(!MoveFlags::DOUBLE_PAWN_PUSH.is_castle());
    }
}
