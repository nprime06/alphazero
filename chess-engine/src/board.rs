//! Board representation.
//!
//! The `Board` struct holds the complete state of a chess position:
//! piece placement (via 12 bitboards), side to move, castling rights,
//! en passant square, and move clocks.
//!
//! # Piece storage
//!
//! We use 12 bitboards -- one per (Color, Piece) combination. They are stored
//! as `piece_bb[color.index()][piece.index()]`, so `piece_bb[0][3]` is the
//! bitboard for White Rooks.
//!
//! This representation makes it fast to answer questions like "where are all
//! the black pawns?" (a single bitboard lookup) and "what is on square e4?"
//! (check each bitboard until we find a hit).
//!
//! # Copy-make for MCTS
//!
//! The `Clone` derive gives us an O(1) copy of the entire board (it's just
//! a few hundred bytes of data). This enables the "copy-make" pattern used
//! in MCTS: clone the board, apply a move, evaluate the new position.

use std::fmt;

use crate::bitboard::Bitboard;
use crate::types::{Color, Piece, Square, NUM_COLORS, NUM_PIECE_TYPES};

// =============================================================================
// Castling Rights
// =============================================================================

/// Castling availability, stored as a 4-bit field inside a `u8`.
///
/// Each bit represents one castling right:
/// - Bit 0: White kingside  (O-O)
/// - Bit 1: White queenside (O-O-O)
/// - Bit 2: Black kingside  (O-O)
/// - Bit 3: Black queenside (O-O-O)
///
/// Using a bitfield (instead of four separate booleans) keeps the struct
/// compact and makes it easy to XOR with Zobrist keys later (Chunk 1.10).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CastlingRights(u8);

impl CastlingRights {
    /// Individual right flags.
    const WHITE_KINGSIDE: u8 = 0b0001;
    const WHITE_QUEENSIDE: u8 = 0b0010;
    const BLACK_KINGSIDE: u8 = 0b0100;
    const BLACK_QUEENSIDE: u8 = 0b1000;

    /// No castling rights at all.
    pub const NONE: CastlingRights = CastlingRights(0);

    /// All four castling rights (the starting position).
    pub const ALL: CastlingRights = CastlingRights(
        Self::WHITE_KINGSIDE | Self::WHITE_QUEENSIDE | Self::BLACK_KINGSIDE | Self::BLACK_QUEENSIDE,
    );

    /// Returns the underlying bits (useful for Zobrist hashing).
    #[inline]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Creates castling rights from raw bits.
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        CastlingRights(bits & 0b1111)
    }

    // ---- Query individual rights --------------------------------------------

    #[inline]
    pub const fn white_kingside(self) -> bool {
        self.0 & Self::WHITE_KINGSIDE != 0
    }

    #[inline]
    pub const fn white_queenside(self) -> bool {
        self.0 & Self::WHITE_QUEENSIDE != 0
    }

    #[inline]
    pub const fn black_kingside(self) -> bool {
        self.0 & Self::BLACK_KINGSIDE != 0
    }

    #[inline]
    pub const fn black_queenside(self) -> bool {
        self.0 & Self::BLACK_QUEENSIDE != 0
    }

    /// Returns `true` if the given color can castle kingside.
    #[inline]
    pub const fn kingside(self, color: Color) -> bool {
        match color {
            Color::White => self.white_kingside(),
            Color::Black => self.black_kingside(),
        }
    }

    /// Returns `true` if the given color can castle queenside.
    #[inline]
    pub const fn queenside(self, color: Color) -> bool {
        match color {
            Color::White => self.white_queenside(),
            Color::Black => self.black_queenside(),
        }
    }

    // ---- Mutators -----------------------------------------------------------

    /// Sets a specific castling right.
    #[inline]
    pub fn set_white_kingside(&mut self, value: bool) {
        if value {
            self.0 |= Self::WHITE_KINGSIDE;
        } else {
            self.0 &= !Self::WHITE_KINGSIDE;
        }
    }

    #[inline]
    pub fn set_white_queenside(&mut self, value: bool) {
        if value {
            self.0 |= Self::WHITE_QUEENSIDE;
        } else {
            self.0 &= !Self::WHITE_QUEENSIDE;
        }
    }

    #[inline]
    pub fn set_black_kingside(&mut self, value: bool) {
        if value {
            self.0 |= Self::BLACK_KINGSIDE;
        } else {
            self.0 &= !Self::BLACK_KINGSIDE;
        }
    }

    #[inline]
    pub fn set_black_queenside(&mut self, value: bool) {
        if value {
            self.0 |= Self::BLACK_QUEENSIDE;
        } else {
            self.0 &= !Self::BLACK_QUEENSIDE;
        }
    }

    /// Removes all castling rights for the given color.
    #[inline]
    pub fn clear_color(&mut self, color: Color) {
        match color {
            Color::White => self.0 &= !(Self::WHITE_KINGSIDE | Self::WHITE_QUEENSIDE),
            Color::Black => self.0 &= !(Self::BLACK_KINGSIDE | Self::BLACK_QUEENSIDE),
        }
    }
}

impl fmt::Display for CastlingRights {
    /// Displays castling rights in FEN notation: "KQkq", "-" if none.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            return write!(f, "-");
        }
        if self.white_kingside() {
            write!(f, "K")?;
        }
        if self.white_queenside() {
            write!(f, "Q")?;
        }
        if self.black_kingside() {
            write!(f, "k")?;
        }
        if self.black_queenside() {
            write!(f, "q")?;
        }
        Ok(())
    }
}

// =============================================================================
// Board
// =============================================================================

/// A complete chess position.
///
/// Contains everything needed to determine which moves are legal:
/// piece placement, side to move, castling rights, en passant square,
/// and move clocks (for the 50-move rule and move numbering).
///
/// # Cloning
///
/// `Board` derives `Clone`, enabling the "copy-make" pattern:
/// ```ignore
/// let new_board = board.clone();
/// new_board.make_move(mv);
/// ```
/// This is important for MCTS, where each simulation explores a different
/// sequence of moves from the same root position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    /// Piece bitboards: `piece_bb[color][piece]`.
    ///
    /// For example, `piece_bb[Color::White.index()][Piece::Pawn.index()]`
    /// holds a bitboard with a `1` for every square that has a white pawn.
    piece_bb: [[Bitboard; NUM_PIECE_TYPES]; NUM_COLORS],

    /// Which side's turn it is to move.
    side_to_move: Color,

    /// Which castling moves are still potentially available.
    ///
    /// A right being present here does NOT mean the castling move is legal
    /// right now -- just that the king and rook haven't moved yet.
    /// Legality also requires that the king isn't in check, doesn't pass
    /// through check, and the squares between king and rook are empty.
    castling_rights: CastlingRights,

    /// The square "behind" a pawn that just made a double push, if any.
    ///
    /// If White plays e2-e4, the en passant square is e3. If Black plays
    /// d7-d5, the en passant square is d6. This is `None` when the last
    /// move was not a double pawn push.
    en_passant_square: Option<Square>,

    /// Number of half-moves since the last pawn advance or capture.
    ///
    /// Used for the 50-move draw rule: if this reaches 100 (= 50 full moves
    /// with no pawn move or capture), either side can claim a draw.
    halfmove_clock: u16,

    /// The full move number, starting at 1 and incrementing after Black moves.
    ///
    /// In the starting position, this is 1. After 1. e4 e5, it becomes 2.
    fullmove_number: u16,

    /// Zobrist hash of the current position.
    ///
    /// Maintained incrementally by `make_move` and restored by `unmake_move`.
    /// Used for transposition detection and repetition checking.
    /// See the `zobrist` module for details on how the hash is computed.
    hash: u64,
}

// =============================================================================
// Construction
// =============================================================================

impl Board {
    /// Creates an empty board with no pieces.
    ///
    /// Useful as a starting point for building custom positions (e.g., from
    /// FEN parsing). Side to move is White, no castling rights, no en passant.
    pub fn empty() -> Self {
        use crate::zobrist;

        let mut board = Board {
            piece_bb: [[Bitboard::EMPTY; NUM_PIECE_TYPES]; NUM_COLORS],
            side_to_move: Color::White,
            castling_rights: CastlingRights::NONE,
            en_passant_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            hash: 0, // temporary; computed below
        };
        board.hash = zobrist::compute_hash(&board);
        board
    }

    /// Creates a board with the standard chess starting position.
    ///
    /// ```text
    ///   8 | r n b q k b n r
    ///   7 | p p p p p p p p
    ///   6 | . . . . . . . .
    ///   5 | . . . . . . . .
    ///   4 | . . . . . . . .
    ///   3 | . . . . . . . .
    ///   2 | P P P P P P P P
    ///   1 | R N B Q K B N R
    ///       a b c d e f g h
    /// ```
    ///
    /// White to move, all castling rights, no en passant, clocks at 0/1.
    pub fn starting_position() -> Self {
        let mut board = Board::empty();

        // --- White pieces (ranks 1 and 2) ---

        // White pawns: all of rank 2
        board.piece_bb[Color::White.index()][Piece::Pawn.index()] = Bitboard::RANK_2;

        // White rooks: a1 and h1
        board.piece_bb[Color::White.index()][Piece::Rook.index()] =
            Bitboard::from_square(Square::A1) | Bitboard::from_square(Square::H1);

        // White knights: b1 and g1
        board.piece_bb[Color::White.index()][Piece::Knight.index()] =
            Bitboard::from_square(Square::B1) | Bitboard::from_square(Square::G1);

        // White bishops: c1 and f1
        board.piece_bb[Color::White.index()][Piece::Bishop.index()] =
            Bitboard::from_square(Square::C1) | Bitboard::from_square(Square::F1);

        // White queen: d1
        board.piece_bb[Color::White.index()][Piece::Queen.index()] =
            Bitboard::from_square(Square::D1);

        // White king: e1
        board.piece_bb[Color::White.index()][Piece::King.index()] =
            Bitboard::from_square(Square::E1);

        // --- Black pieces (ranks 7 and 8) ---

        // Black pawns: all of rank 7
        board.piece_bb[Color::Black.index()][Piece::Pawn.index()] = Bitboard::RANK_7;

        // Black rooks: a8 and h8
        board.piece_bb[Color::Black.index()][Piece::Rook.index()] =
            Bitboard::from_square(Square::A8) | Bitboard::from_square(Square::H8);

        // Black knights: b8 and g8
        board.piece_bb[Color::Black.index()][Piece::Knight.index()] =
            Bitboard::from_square(Square::B8) | Bitboard::from_square(Square::G8);

        // Black bishops: c8 and f8
        board.piece_bb[Color::Black.index()][Piece::Bishop.index()] =
            Bitboard::from_square(Square::C8) | Bitboard::from_square(Square::F8);

        // Black queen: d8
        board.piece_bb[Color::Black.index()][Piece::Queen.index()] =
            Bitboard::from_square(Square::D8);

        // Black king: e8
        board.piece_bb[Color::Black.index()][Piece::King.index()] =
            Bitboard::from_square(Square::E8);

        // --- Game state ---

        board.side_to_move = Color::White;
        board.castling_rights = CastlingRights::ALL;
        board.en_passant_square = None;
        board.halfmove_clock = 0;
        board.fullmove_number = 1;

        // Compute the Zobrist hash from scratch for the starting position.
        board.hash = crate::zobrist::compute_hash(&board);

        board
    }
}

// =============================================================================
// Accessors (read-only queries)
// =============================================================================

impl Board {
    /// Returns the side to move.
    #[inline]
    pub const fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    /// Returns the current castling rights.
    #[inline]
    pub const fn castling_rights(&self) -> CastlingRights {
        self.castling_rights
    }

    /// Returns the en passant target square, if any.
    #[inline]
    pub const fn en_passant_square(&self) -> Option<Square> {
        self.en_passant_square
    }

    /// Returns the halfmove clock (moves since last pawn push or capture).
    #[inline]
    pub const fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    /// Returns the fullmove number (starts at 1, increments after Black moves).
    #[inline]
    pub const fn fullmove_number(&self) -> u16 {
        self.fullmove_number
    }

    /// Returns the Zobrist hash of the current position.
    ///
    /// This hash is maintained incrementally by `make_move` and restored
    /// by `unmake_move`. It encodes: piece placement, castling rights,
    /// en passant file, and side to move.
    #[inline]
    pub const fn hash(&self) -> u64 {
        self.hash
    }

    /// Returns the bitboard for a specific (color, piece) pair.
    ///
    /// Example: `board.piece_bitboard(Color::White, Piece::Pawn)` returns a
    /// bitboard with a 1 on every square that has a white pawn.
    #[inline]
    pub const fn piece_bitboard(&self, color: Color, piece: Piece) -> Bitboard {
        self.piece_bb[color.index()][piece.index()]
    }

    // ---- Occupancy helpers --------------------------------------------------

    /// Returns a bitboard of all pieces belonging to the given color.
    ///
    /// This OR-s together all 6 piece bitboards for that color.
    pub fn pieces(&self, color: Color) -> Bitboard {
        let c = color.index();
        self.piece_bb[c][0]
            | self.piece_bb[c][1]
            | self.piece_bb[c][2]
            | self.piece_bb[c][3]
            | self.piece_bb[c][4]
            | self.piece_bb[c][5]
    }

    /// Returns a bitboard of all white pieces.
    #[inline]
    pub fn white_pieces(&self) -> Bitboard {
        self.pieces(Color::White)
    }

    /// Returns a bitboard of all black pieces.
    #[inline]
    pub fn black_pieces(&self) -> Bitboard {
        self.pieces(Color::Black)
    }

    /// Returns a bitboard of all pieces on the board (both colors).
    #[inline]
    pub fn all_pieces(&self) -> Bitboard {
        self.white_pieces() | self.black_pieces()
    }

    /// Returns a bitboard of all empty squares (no piece of either color).
    #[inline]
    pub fn empty_squares(&self) -> Bitboard {
        !self.all_pieces()
    }

    // ---- Piece lookup -------------------------------------------------------

    /// Looks up what piece (if any) is on the given square.
    ///
    /// Returns `Some((color, piece))` if a piece is found, `None` if the
    /// square is empty.
    ///
    /// This performs a linear scan over all 12 bitboards. In hot paths where
    /// you already know the color, it's faster to check only the 6 bitboards
    /// for that color.
    pub fn piece_at(&self, square: Square) -> Option<(Color, Piece)> {
        for &color in &Color::ALL {
            for &piece in &Piece::ALL {
                if self.piece_bb[color.index()][piece.index()].contains(square) {
                    return Some((color, piece));
                }
            }
        }
        None
    }
}

// =============================================================================
// Mutators (piece placement and removal)
// =============================================================================

impl Board {
    /// Places a piece on the given square.
    ///
    /// Sets the corresponding bit in the `piece_bb[color][piece]` bitboard.
    ///
    /// # Preconditions
    ///
    /// The square should be empty. Calling `put_piece` on an occupied square
    /// does not automatically remove the existing piece -- that would require
    /// knowing which piece to remove. In debug builds, this will panic if the
    /// square is already occupied (to catch bugs early).
    pub fn put_piece(&mut self, square: Square, color: Color, piece: Piece) {
        debug_assert!(
            self.piece_at(square).is_none(),
            "put_piece called on occupied square {} (has {:?})",
            square,
            self.piece_at(square)
        );
        self.piece_bb[color.index()][piece.index()].set(square);
    }

    /// Removes a piece from the given square.
    ///
    /// Clears the corresponding bit in the `piece_bb[color][piece]` bitboard.
    ///
    /// # Preconditions
    ///
    /// The specified (color, piece) should actually be on that square. In
    /// debug builds, this will panic if the piece isn't found there.
    pub fn remove_piece(&mut self, square: Square, color: Color, piece: Piece) {
        debug_assert!(
            self.piece_bb[color.index()][piece.index()].contains(square),
            "remove_piece: no {:?} {:?} on square {}",
            color,
            piece,
            square
        );
        self.piece_bb[color.index()][piece.index()].clear(square);
    }

    /// Sets the side to move.
    #[inline]
    pub fn set_side_to_move(&mut self, color: Color) {
        self.side_to_move = color;
    }

    /// Sets the castling rights.
    #[inline]
    pub fn set_castling_rights(&mut self, rights: CastlingRights) {
        self.castling_rights = rights;
    }

    /// Sets the en passant target square.
    #[inline]
    pub fn set_en_passant_square(&mut self, square: Option<Square>) {
        self.en_passant_square = square;
    }

    /// Sets the halfmove clock.
    #[inline]
    pub fn set_halfmove_clock(&mut self, clock: u16) {
        self.halfmove_clock = clock;
    }

    /// Sets the fullmove number.
    #[inline]
    pub fn set_fullmove_number(&mut self, number: u16) {
        self.fullmove_number = number;
    }

    /// Sets the Zobrist hash directly.
    ///
    /// Used by `from_fen` to set the hash after building the board,
    /// and by `unmake_move` to restore the hash from `UndoInfo`.
    #[inline]
    pub fn set_hash(&mut self, hash: u64) {
        self.hash = hash;
    }
}

// =============================================================================
// Display
// =============================================================================

impl fmt::Display for Board {
    /// Displays the board as an 8x8 grid with piece characters.
    ///
    /// Uses standard piece notation:
    /// - Uppercase for White: P N B R Q K
    /// - Lowercase for Black: p n b r q k
    /// - `.` for empty squares
    ///
    /// Also shows side to move, castling rights, en passant square, and clocks.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print from rank 8 (top) down to rank 1 (bottom).
        for rank in (0..8u8).rev() {
            write!(f, "  {} |", rank + 1)?;
            for file in 0..8u8 {
                let square = Square::from_file_rank(file, rank);
                let c = match self.piece_at(square) {
                    None => '.',
                    Some((Color::White, piece)) => piece.char(),             // Uppercase
                    Some((Color::Black, piece)) => piece.char().to_ascii_lowercase(), // Lowercase
                };
                write!(f, " {}", c)?;
            }
            writeln!(f)?;
        }
        writeln!(f, "      a b c d e f g h")?;
        writeln!(f)?;
        writeln!(f, "  Side to move: {}", self.side_to_move)?;
        writeln!(f, "  Castling:     {}", self.castling_rights)?;
        match self.en_passant_square {
            Some(sq) => writeln!(f, "  En passant:   {}", sq)?,
            None => writeln!(f, "  En passant:   -")?,
        }
        writeln!(f, "  Halfmove:     {}", self.halfmove_clock)?;
        writeln!(f, "  Fullmove:     {}", self.fullmove_number)?;
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Color, Piece, Square};

    // ---- CastlingRights tests -----------------------------------------------

    #[test]
    fn castling_rights_none() {
        let cr = CastlingRights::NONE;
        assert!(!cr.white_kingside());
        assert!(!cr.white_queenside());
        assert!(!cr.black_kingside());
        assert!(!cr.black_queenside());
        assert_eq!(cr.bits(), 0);
    }

    #[test]
    fn castling_rights_all() {
        let cr = CastlingRights::ALL;
        assert!(cr.white_kingside());
        assert!(cr.white_queenside());
        assert!(cr.black_kingside());
        assert!(cr.black_queenside());
        assert_eq!(cr.bits(), 0b1111);
    }

    #[test]
    fn castling_rights_set_and_clear() {
        let mut cr = CastlingRights::NONE;

        cr.set_white_kingside(true);
        assert!(cr.white_kingside());
        assert_eq!(cr.bits(), 0b0001);

        cr.set_black_queenside(true);
        assert!(cr.black_queenside());
        assert_eq!(cr.bits(), 0b1001);

        cr.set_white_kingside(false);
        assert!(!cr.white_kingside());
        assert_eq!(cr.bits(), 0b1000);
    }

    #[test]
    fn castling_rights_clear_color() {
        let mut cr = CastlingRights::ALL;
        cr.clear_color(Color::White);
        assert!(!cr.white_kingside());
        assert!(!cr.white_queenside());
        assert!(cr.black_kingside());
        assert!(cr.black_queenside());

        cr.clear_color(Color::Black);
        assert_eq!(cr, CastlingRights::NONE);
    }

    #[test]
    fn castling_rights_kingside_queenside_by_color() {
        let cr = CastlingRights::ALL;
        assert!(cr.kingside(Color::White));
        assert!(cr.queenside(Color::White));
        assert!(cr.kingside(Color::Black));
        assert!(cr.queenside(Color::Black));

        let cr = CastlingRights::from_bits(0b0001); // Only white kingside
        assert!(cr.kingside(Color::White));
        assert!(!cr.queenside(Color::White));
        assert!(!cr.kingside(Color::Black));
        assert!(!cr.queenside(Color::Black));
    }

    #[test]
    fn castling_rights_display_all() {
        assert_eq!(format!("{}", CastlingRights::ALL), "KQkq");
    }

    #[test]
    fn castling_rights_display_none() {
        assert_eq!(format!("{}", CastlingRights::NONE), "-");
    }

    #[test]
    fn castling_rights_display_partial() {
        let cr = CastlingRights::from_bits(0b0101); // White kingside + Black kingside
        assert_eq!(format!("{}", cr), "Kk");

        let cr = CastlingRights::from_bits(0b1010); // White queenside + Black queenside
        assert_eq!(format!("{}", cr), "Qq");
    }

    #[test]
    fn castling_rights_from_bits_masks_extra() {
        // Only the low 4 bits should matter.
        let cr = CastlingRights::from_bits(0xFF);
        assert_eq!(cr.bits(), 0b1111);
    }

    // ---- Empty board tests --------------------------------------------------

    #[test]
    fn empty_board_has_no_pieces() {
        let board = Board::empty();
        assert_eq!(board.all_pieces(), Bitboard::EMPTY);
        assert_eq!(board.white_pieces(), Bitboard::EMPTY);
        assert_eq!(board.black_pieces(), Bitboard::EMPTY);
    }

    #[test]
    fn empty_board_defaults() {
        let board = Board::empty();
        assert_eq!(board.side_to_move(), Color::White);
        assert_eq!(board.castling_rights(), CastlingRights::NONE);
        assert_eq!(board.en_passant_square(), None);
        assert_eq!(board.halfmove_clock(), 0);
        assert_eq!(board.fullmove_number(), 1);
    }

    #[test]
    fn empty_board_piece_at_returns_none() {
        let board = Board::empty();
        for i in 0..64u8 {
            assert!(
                board.piece_at(Square::new(i)).is_none(),
                "Empty board should have no piece on {}",
                Square::new(i)
            );
        }
    }

    // ---- Put and remove piece -----------------------------------------------

    #[test]
    fn put_piece_and_query() {
        let mut board = Board::empty();
        board.put_piece(Square::E4, Color::White, Piece::Knight);

        assert_eq!(
            board.piece_at(Square::E4),
            Some((Color::White, Piece::Knight))
        );
        assert!(board
            .piece_bitboard(Color::White, Piece::Knight)
            .contains(Square::E4));
        assert_eq!(board.white_pieces().count(), 1);
        assert_eq!(board.all_pieces().count(), 1);
    }

    #[test]
    fn put_multiple_pieces() {
        let mut board = Board::empty();
        board.put_piece(Square::E1, Color::White, Piece::King);
        board.put_piece(Square::D8, Color::Black, Piece::Queen);
        board.put_piece(Square::A2, Color::White, Piece::Pawn);

        assert_eq!(
            board.piece_at(Square::E1),
            Some((Color::White, Piece::King))
        );
        assert_eq!(
            board.piece_at(Square::D8),
            Some((Color::Black, Piece::Queen))
        );
        assert_eq!(
            board.piece_at(Square::A2),
            Some((Color::White, Piece::Pawn))
        );
        assert_eq!(board.white_pieces().count(), 2);
        assert_eq!(board.black_pieces().count(), 1);
        assert_eq!(board.all_pieces().count(), 3);
    }

    #[test]
    fn remove_piece_clears_square() {
        let mut board = Board::empty();
        board.put_piece(Square::D4, Color::Black, Piece::Bishop);
        assert!(board.piece_at(Square::D4).is_some());

        board.remove_piece(Square::D4, Color::Black, Piece::Bishop);
        assert!(board.piece_at(Square::D4).is_none());
        assert_eq!(board.all_pieces().count(), 0);
    }

    #[test]
    fn put_and_remove_leaves_board_empty() {
        let mut board = Board::empty();
        board.put_piece(Square::A1, Color::White, Piece::Rook);
        board.put_piece(Square::H8, Color::Black, Piece::Rook);

        board.remove_piece(Square::A1, Color::White, Piece::Rook);
        board.remove_piece(Square::H8, Color::Black, Piece::Rook);

        assert_eq!(board.all_pieces(), Bitboard::EMPTY);
    }

    // ---- Starting position tests -------------------------------------------

    #[test]
    fn starting_position_piece_count() {
        let board = Board::starting_position();

        // Each side starts with 16 pieces.
        assert_eq!(board.white_pieces().count(), 16);
        assert_eq!(board.black_pieces().count(), 16);
        assert_eq!(board.all_pieces().count(), 32);
    }

    #[test]
    fn starting_position_pawn_placement() {
        let board = Board::starting_position();

        // White pawns on rank 2.
        let white_pawns = board.piece_bitboard(Color::White, Piece::Pawn);
        assert_eq!(white_pawns, Bitboard::RANK_2);
        assert_eq!(white_pawns.count(), 8);

        // Black pawns on rank 7.
        let black_pawns = board.piece_bitboard(Color::Black, Piece::Pawn);
        assert_eq!(black_pawns, Bitboard::RANK_7);
        assert_eq!(black_pawns.count(), 8);
    }

    #[test]
    fn starting_position_rook_placement() {
        let board = Board::starting_position();

        // White rooks on a1 and h1.
        let white_rooks = board.piece_bitboard(Color::White, Piece::Rook);
        assert_eq!(white_rooks.count(), 2);
        assert!(white_rooks.contains(Square::A1));
        assert!(white_rooks.contains(Square::H1));

        // Black rooks on a8 and h8.
        let black_rooks = board.piece_bitboard(Color::Black, Piece::Rook);
        assert_eq!(black_rooks.count(), 2);
        assert!(black_rooks.contains(Square::A8));
        assert!(black_rooks.contains(Square::H8));
    }

    #[test]
    fn starting_position_knight_placement() {
        let board = Board::starting_position();

        let white_knights = board.piece_bitboard(Color::White, Piece::Knight);
        assert_eq!(white_knights.count(), 2);
        assert!(white_knights.contains(Square::B1));
        assert!(white_knights.contains(Square::G1));

        let black_knights = board.piece_bitboard(Color::Black, Piece::Knight);
        assert_eq!(black_knights.count(), 2);
        assert!(black_knights.contains(Square::B8));
        assert!(black_knights.contains(Square::G8));
    }

    #[test]
    fn starting_position_bishop_placement() {
        let board = Board::starting_position();

        let white_bishops = board.piece_bitboard(Color::White, Piece::Bishop);
        assert_eq!(white_bishops.count(), 2);
        assert!(white_bishops.contains(Square::C1));
        assert!(white_bishops.contains(Square::F1));

        let black_bishops = board.piece_bitboard(Color::Black, Piece::Bishop);
        assert_eq!(black_bishops.count(), 2);
        assert!(black_bishops.contains(Square::C8));
        assert!(black_bishops.contains(Square::F8));
    }

    #[test]
    fn starting_position_queen_placement() {
        let board = Board::starting_position();

        let white_queen = board.piece_bitboard(Color::White, Piece::Queen);
        assert_eq!(white_queen.count(), 1);
        assert!(white_queen.contains(Square::D1));

        let black_queen = board.piece_bitboard(Color::Black, Piece::Queen);
        assert_eq!(black_queen.count(), 1);
        assert!(black_queen.contains(Square::D8));
    }

    #[test]
    fn starting_position_king_placement() {
        let board = Board::starting_position();

        let white_king = board.piece_bitboard(Color::White, Piece::King);
        assert_eq!(white_king.count(), 1);
        assert!(white_king.contains(Square::E1));

        let black_king = board.piece_bitboard(Color::Black, Piece::King);
        assert_eq!(black_king.count(), 1);
        assert!(black_king.contains(Square::E8));
    }

    #[test]
    fn starting_position_piece_at_all_squares() {
        let board = Board::starting_position();

        // Verify every square has the expected piece (or is empty).
        // Rank 1: R N B Q K B N R
        assert_eq!(board.piece_at(Square::A1), Some((Color::White, Piece::Rook)));
        assert_eq!(board.piece_at(Square::B1), Some((Color::White, Piece::Knight)));
        assert_eq!(board.piece_at(Square::C1), Some((Color::White, Piece::Bishop)));
        assert_eq!(board.piece_at(Square::D1), Some((Color::White, Piece::Queen)));
        assert_eq!(board.piece_at(Square::E1), Some((Color::White, Piece::King)));
        assert_eq!(board.piece_at(Square::F1), Some((Color::White, Piece::Bishop)));
        assert_eq!(board.piece_at(Square::G1), Some((Color::White, Piece::Knight)));
        assert_eq!(board.piece_at(Square::H1), Some((Color::White, Piece::Rook)));

        // Rank 2: all white pawns
        for file in 0..8u8 {
            let sq = Square::from_file_rank(file, 1);
            assert_eq!(board.piece_at(sq), Some((Color::White, Piece::Pawn)),
                "Expected white pawn on {}", sq);
        }

        // Ranks 3-6: empty
        for rank in 2..6u8 {
            for file in 0..8u8 {
                let sq = Square::from_file_rank(file, rank);
                assert!(board.piece_at(sq).is_none(), "Expected empty on {}", sq);
            }
        }

        // Rank 7: all black pawns
        for file in 0..8u8 {
            let sq = Square::from_file_rank(file, 6);
            assert_eq!(board.piece_at(sq), Some((Color::Black, Piece::Pawn)),
                "Expected black pawn on {}", sq);
        }

        // Rank 8: r n b q k b n r
        assert_eq!(board.piece_at(Square::A8), Some((Color::Black, Piece::Rook)));
        assert_eq!(board.piece_at(Square::B8), Some((Color::Black, Piece::Knight)));
        assert_eq!(board.piece_at(Square::C8), Some((Color::Black, Piece::Bishop)));
        assert_eq!(board.piece_at(Square::D8), Some((Color::Black, Piece::Queen)));
        assert_eq!(board.piece_at(Square::E8), Some((Color::Black, Piece::King)));
        assert_eq!(board.piece_at(Square::F8), Some((Color::Black, Piece::Bishop)));
        assert_eq!(board.piece_at(Square::G8), Some((Color::Black, Piece::Knight)));
        assert_eq!(board.piece_at(Square::H8), Some((Color::Black, Piece::Rook)));
    }

    #[test]
    fn starting_position_game_state() {
        let board = Board::starting_position();
        assert_eq!(board.side_to_move(), Color::White);
        assert_eq!(board.castling_rights(), CastlingRights::ALL);
        assert_eq!(board.en_passant_square(), None);
        assert_eq!(board.halfmove_clock(), 0);
        assert_eq!(board.fullmove_number(), 1);
    }

    #[test]
    fn starting_position_empty_squares() {
        let board = Board::starting_position();
        let empty = board.empty_squares();

        // Ranks 3-6 are empty: 4 ranks * 8 files = 32 squares.
        assert_eq!(empty.count(), 32);

        // Verify that occupied + empty = full board.
        assert_eq!(board.all_pieces() | empty, Bitboard::FULL);
    }

    // ---- Occupancy helpers --------------------------------------------------

    #[test]
    fn pieces_returns_union_of_all_piece_types() {
        let board = Board::starting_position();

        // White pieces should be exactly ranks 1 and 2.
        let expected_white = Bitboard::RANK_1 | Bitboard::RANK_2;
        assert_eq!(board.white_pieces(), expected_white);

        // Black pieces should be exactly ranks 7 and 8.
        let expected_black = Bitboard::RANK_7 | Bitboard::RANK_8;
        assert_eq!(board.black_pieces(), expected_black);
    }

    #[test]
    fn white_and_black_pieces_are_disjoint() {
        let board = Board::starting_position();
        let overlap = board.white_pieces() & board.black_pieces();
        assert!(
            overlap.is_empty(),
            "White and black pieces must never overlap"
        );
    }

    #[test]
    fn all_pieces_equals_white_or_black() {
        let board = Board::starting_position();
        assert_eq!(
            board.all_pieces(),
            board.white_pieces() | board.black_pieces()
        );
    }

    // ---- Bitboard consistency -----------------------------------------------

    #[test]
    fn no_two_piece_types_overlap_for_same_color() {
        let board = Board::starting_position();
        for &color in &Color::ALL {
            for i in 0..Piece::ALL.len() {
                for j in (i + 1)..Piece::ALL.len() {
                    let bb_i = board.piece_bitboard(color, Piece::ALL[i]);
                    let bb_j = board.piece_bitboard(color, Piece::ALL[j]);
                    let overlap = bb_i & bb_j;
                    assert!(
                        overlap.is_empty(),
                        "{:?}: {:?} and {:?} bitboards overlap!",
                        color,
                        Piece::ALL[i],
                        Piece::ALL[j]
                    );
                }
            }
        }
    }

    #[test]
    fn no_piece_bitboard_overlaps_across_colors() {
        let board = Board::starting_position();
        for &piece in &Piece::ALL {
            let white_bb = board.piece_bitboard(Color::White, piece);
            let black_bb = board.piece_bitboard(Color::Black, piece);
            let overlap = white_bb & black_bb;
            assert!(
                overlap.is_empty(),
                "{:?} bitboards overlap between White and Black!",
                piece
            );
        }
    }

    // ---- Clone tests --------------------------------------------------------

    #[test]
    fn clone_produces_equal_board() {
        let board = Board::starting_position();
        let cloned = board.clone();
        assert_eq!(board, cloned);
    }

    #[test]
    fn clone_is_independent() {
        let board = Board::starting_position();
        let mut cloned = board.clone();

        // Modify the clone.
        cloned.remove_piece(Square::E2, Color::White, Piece::Pawn);
        cloned.put_piece(Square::E4, Color::White, Piece::Pawn);

        // Original should be unchanged.
        assert_eq!(
            board.piece_at(Square::E2),
            Some((Color::White, Piece::Pawn)),
            "Original board should not be affected by clone modifications"
        );
        assert!(
            board.piece_at(Square::E4).is_none(),
            "Original board should not have piece on e4"
        );

        // Clone should reflect the change.
        assert!(cloned.piece_at(Square::E2).is_none());
        assert_eq!(
            cloned.piece_at(Square::E4),
            Some((Color::White, Piece::Pawn))
        );
    }

    // ---- Setters ------------------------------------------------------------

    #[test]
    fn set_side_to_move() {
        let mut board = Board::empty();
        assert_eq!(board.side_to_move(), Color::White);

        board.set_side_to_move(Color::Black);
        assert_eq!(board.side_to_move(), Color::Black);
    }

    #[test]
    fn set_castling_rights() {
        let mut board = Board::empty();
        assert_eq!(board.castling_rights(), CastlingRights::NONE);

        board.set_castling_rights(CastlingRights::ALL);
        assert_eq!(board.castling_rights(), CastlingRights::ALL);
    }

    #[test]
    fn set_en_passant_square() {
        let mut board = Board::empty();
        assert_eq!(board.en_passant_square(), None);

        board.set_en_passant_square(Some(Square::E3));
        assert_eq!(board.en_passant_square(), Some(Square::E3));

        board.set_en_passant_square(None);
        assert_eq!(board.en_passant_square(), None);
    }

    #[test]
    fn set_halfmove_clock() {
        let mut board = Board::empty();
        board.set_halfmove_clock(42);
        assert_eq!(board.halfmove_clock(), 42);
    }

    #[test]
    fn set_fullmove_number() {
        let mut board = Board::empty();
        board.set_fullmove_number(100);
        assert_eq!(board.fullmove_number(), 100);
    }

    // ---- Display ------------------------------------------------------------

    #[test]
    fn display_starting_position_contains_expected_pieces() {
        let board = Board::starting_position();
        let s = format!("{}", board);

        // Should contain rank labels.
        assert!(s.contains("8 |"));
        assert!(s.contains("1 |"));

        // Should contain file labels.
        assert!(s.contains("a b c d e f g h"));

        // Should show White is to move.
        assert!(s.contains("Side to move: White"));

        // Should show all castling rights.
        assert!(s.contains("Castling:     KQkq"));

        // Should show no en passant.
        assert!(s.contains("En passant:   -"));
    }

    #[test]
    fn display_empty_board_shows_all_dots() {
        let board = Board::empty();
        let s = format!("{}", board);

        // Count pieces on board area (after '|' on each rank line).
        let piece_chars = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'];
        for line in s.lines() {
            if line.contains('|') {
                let after_bar = line.split('|').nth(1).unwrap();
                for &pc in &piece_chars {
                    assert!(
                        !after_bar.contains(pc),
                        "Empty board should have no piece characters, found '{}' in: {}",
                        pc,
                        line
                    );
                }
            }
        }
    }

    // ---- Custom position test -----------------------------------------------

    #[test]
    fn build_custom_position() {
        // Build a simple endgame: White Ke1, Rd1 vs Black Ke8.
        let mut board = Board::empty();
        board.put_piece(Square::E1, Color::White, Piece::King);
        board.put_piece(Square::D1, Color::White, Piece::Rook);
        board.put_piece(Square::E8, Color::Black, Piece::King);
        board.set_side_to_move(Color::White);

        assert_eq!(board.white_pieces().count(), 2);
        assert_eq!(board.black_pieces().count(), 1);
        assert_eq!(board.all_pieces().count(), 3);
        assert_eq!(board.empty_squares().count(), 61);

        assert_eq!(
            board.piece_at(Square::E1),
            Some((Color::White, Piece::King))
        );
        assert_eq!(
            board.piece_at(Square::D1),
            Some((Color::White, Piece::Rook))
        );
        assert_eq!(
            board.piece_at(Square::E8),
            Some((Color::Black, Piece::King))
        );
        assert!(board.piece_at(Square::D4).is_none());
    }

    // ---- Piece counts for starting position ---------------------------------

    #[test]
    fn starting_position_piece_type_counts() {
        let board = Board::starting_position();

        // White piece counts.
        assert_eq!(board.piece_bitboard(Color::White, Piece::Pawn).count(), 8);
        assert_eq!(board.piece_bitboard(Color::White, Piece::Knight).count(), 2);
        assert_eq!(board.piece_bitboard(Color::White, Piece::Bishop).count(), 2);
        assert_eq!(board.piece_bitboard(Color::White, Piece::Rook).count(), 2);
        assert_eq!(board.piece_bitboard(Color::White, Piece::Queen).count(), 1);
        assert_eq!(board.piece_bitboard(Color::White, Piece::King).count(), 1);

        // Black piece counts.
        assert_eq!(board.piece_bitboard(Color::Black, Piece::Pawn).count(), 8);
        assert_eq!(board.piece_bitboard(Color::Black, Piece::Knight).count(), 2);
        assert_eq!(board.piece_bitboard(Color::Black, Piece::Bishop).count(), 2);
        assert_eq!(board.piece_bitboard(Color::Black, Piece::Rook).count(), 2);
        assert_eq!(board.piece_bitboard(Color::Black, Piece::Queen).count(), 1);
        assert_eq!(board.piece_bitboard(Color::Black, Piece::King).count(), 1);
    }
}
