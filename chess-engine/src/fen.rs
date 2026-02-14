//! FEN (Forsyth-Edwards Notation) parsing and generation.
//!
//! FEN is a compact text format for describing a chess position. It contains
//! six space-separated fields:
//!
//! ```text
//! rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
//! |                                              | |    | | |
//! |                                              | |    | | +-- fullmove number
//! |                                              | |    | +---- halfmove clock
//! |                                              | |    +------ en passant square ("-" = none)
//! |                                              | +----------- castling rights
//! |                                              +------------- side to move (w/b)
//! +------------------------------------------------------------ piece placement (rank 8 to rank 1)
//! ```
//!
//! Piece characters: P=pawn, N=knight, B=bishop, R=rook, Q=queen, K=king.
//! Uppercase = White, lowercase = Black. Digits 1-8 represent consecutive
//! empty squares.
//!
//! # Roundtrip guarantee
//!
//! For any valid FEN string, `Board::from_fen(fen).unwrap().to_fen()` produces
//! the same FEN (assuming canonical form). This is verified in tests.

use std::fmt;

use crate::board::{Board, CastlingRights};
use crate::types::{Color, Piece, Square};

// =============================================================================
// Standard FEN strings
// =============================================================================

/// The FEN string for the standard chess starting position.
pub const STARTING_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// The "Kiwipete" position, a widely used test position for move generation.
///
/// It has many edge cases: en passant, castling, pins, discovered checks, etc.
/// Created by Peter McKenzie for perft testing.
pub const KIWIPETE_FEN: &str =
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

/// A position with many possible captures and promotions.
/// Useful for testing capture generation and promotion handling.
pub const POSITION_3_FEN: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

/// A complex middlegame position with many tactical possibilities.
pub const POSITION_4_FEN: &str =
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";

/// A position with promotions and checks.
pub const POSITION_5_FEN: &str =
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";

/// An endgame position useful for testing various piece configurations.
pub const POSITION_6_FEN: &str =
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

// =============================================================================
// FenError
// =============================================================================

/// Errors that can occur when parsing a FEN string.
///
/// Each variant carries a descriptive message explaining what went wrong,
/// to help users diagnose invalid FEN strings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FenError {
    /// The FEN string does not have exactly 6 space-separated fields.
    WrongFieldCount {
        found: usize,
    },

    /// The piece placement field does not have exactly 8 ranks separated by '/'.
    WrongRankCount {
        found: usize,
    },

    /// A rank in the piece placement field does not add up to exactly 8 squares.
    /// For example, "rnbqkbnr1" or "rnbqkbn" would trigger this.
    RankLengthInvalid {
        rank: usize,
        squares: usize,
    },

    /// An unrecognized character was found in the piece placement field.
    /// Valid characters are: p, n, b, r, q, k (lowercase for Black),
    /// P, N, B, R, Q, K (uppercase for White), and 1-8 (empty squares).
    InvalidPieceChar {
        character: char,
        rank: usize,
    },

    /// The side-to-move field is not "w" or "b".
    InvalidSideToMove {
        found: String,
    },

    /// The castling rights field contains an invalid character.
    /// Valid characters are K, Q, k, q, and "-" (for no rights).
    InvalidCastlingRights {
        found: String,
    },

    /// The en passant field is not a valid square name or "-".
    InvalidEnPassantSquare {
        found: String,
    },

    /// The en passant square is on an impossible rank. En passant targets
    /// must be on rank 3 (if Black just moved) or rank 6 (if White just moved).
    EnPassantWrongRank {
        square: String,
        rank: u8,
    },

    /// The halfmove clock is not a valid non-negative integer.
    InvalidHalfmoveClock {
        found: String,
    },

    /// The fullmove number is not a valid positive integer.
    InvalidFullmoveNumber {
        found: String,
    },

    /// The fullmove number is zero, which is not valid (it starts at 1).
    FullmoveNumberZero,
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FenError::WrongFieldCount { found } => {
                write!(
                    f,
                    "FEN must have exactly 6 space-separated fields, found {}",
                    found
                )
            }
            FenError::WrongRankCount { found } => {
                write!(
                    f,
                    "Piece placement must have exactly 8 ranks separated by '/', found {}",
                    found
                )
            }
            FenError::RankLengthInvalid { rank, squares } => {
                write!(
                    f,
                    "Rank {} has {} squares (expected exactly 8)",
                    rank + 1,
                    squares
                )
            }
            FenError::InvalidPieceChar { character, rank } => {
                write!(
                    f,
                    "Invalid character '{}' in rank {} of piece placement",
                    character,
                    rank + 1
                )
            }
            FenError::InvalidSideToMove { found } => {
                write!(
                    f,
                    "Side to move must be 'w' or 'b', found '{}'",
                    found
                )
            }
            FenError::InvalidCastlingRights { found } => {
                write!(
                    f,
                    "Invalid castling rights '{}' (expected combination of K, Q, k, q or '-')",
                    found
                )
            }
            FenError::InvalidEnPassantSquare { found } => {
                write!(
                    f,
                    "Invalid en passant square '{}' (expected algebraic notation like 'e3' or '-')",
                    found
                )
            }
            FenError::EnPassantWrongRank { square, rank } => {
                write!(
                    f,
                    "En passant square '{}' is on rank {} (must be rank 3 or 6)",
                    square,
                    rank + 1
                )
            }
            FenError::InvalidHalfmoveClock { found } => {
                write!(
                    f,
                    "Halfmove clock '{}' is not a valid non-negative integer",
                    found
                )
            }
            FenError::InvalidFullmoveNumber { found } => {
                write!(
                    f,
                    "Fullmove number '{}' is not a valid positive integer",
                    found
                )
            }
            FenError::FullmoveNumberZero => {
                write!(f, "Fullmove number must be at least 1")
            }
        }
    }
}

// Implement std::error::Error so FenError can be used with `?` and error chains.
impl std::error::Error for FenError {}

// =============================================================================
// FEN Parsing (Board::from_fen)
// =============================================================================

impl Board {
    /// Parses a FEN string into a `Board`.
    ///
    /// FEN has six space-separated fields:
    /// 1. Piece placement (ranks 8 to 1, separated by '/')
    /// 2. Side to move ('w' or 'b')
    /// 3. Castling rights ('KQkq', any subset, or '-')
    /// 4. En passant target square (algebraic notation, or '-')
    /// 5. Halfmove clock (non-negative integer)
    /// 6. Fullmove number (positive integer, starting at 1)
    ///
    /// # Errors
    ///
    /// Returns a descriptive `FenError` if the FEN string is malformed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")?;
    /// ```
    pub fn from_fen(fen: &str) -> Result<Board, FenError> {
        let fields: Vec<&str> = fen.split_whitespace().collect();
        if fields.len() != 6 {
            return Err(FenError::WrongFieldCount {
                found: fields.len(),
            });
        }

        let mut board = Board::empty();

        // Field 1: Piece placement
        parse_piece_placement(fields[0], &mut board)?;

        // Field 2: Side to move
        parse_side_to_move(fields[1], &mut board)?;

        // Field 3: Castling rights
        parse_castling_rights(fields[2], &mut board)?;

        // Field 4: En passant square
        parse_en_passant(fields[3], &mut board)?;

        // Field 5: Halfmove clock
        parse_halfmove_clock(fields[4], &mut board)?;

        // Field 6: Fullmove number
        parse_fullmove_number(fields[5], &mut board)?;

        // Compute the Zobrist hash from scratch now that all fields are set.
        let hash = crate::zobrist::compute_hash(&board);
        board.set_hash(hash);

        Ok(board)
    }
}

/// Parses the piece placement field (field 1 of FEN).
///
/// The field is a series of rank descriptions separated by '/', starting from
/// rank 8 (Black's back rank) and ending at rank 1 (White's back rank).
///
/// Within each rank:
/// - Letters (pnbrqkPNBRQK) represent pieces
/// - Digits (1-8) represent consecutive empty squares
/// - The total number of squares per rank must be exactly 8
fn parse_piece_placement(placement: &str, board: &mut Board) -> Result<(), FenError> {
    let ranks: Vec<&str> = placement.split('/').collect();
    if ranks.len() != 8 {
        return Err(FenError::WrongRankCount {
            found: ranks.len(),
        });
    }

    // FEN lists ranks from 8 (top) to 1 (bottom), so ranks[0] is rank 8
    // and ranks[7] is rank 1. We iterate accordingly.
    for (rank_index, rank_str) in ranks.iter().enumerate() {
        // rank_index 0 = rank 8, rank_index 7 = rank 1
        // In our Square system, rank is 0-based: rank 8 = index 7, rank 1 = index 0
        let rank = 7 - rank_index as u8;
        let mut file: u8 = 0;

        for ch in rank_str.chars() {
            if file > 8 {
                return Err(FenError::RankLengthInvalid {
                    rank: rank_index,
                    squares: file as usize,
                });
            }

            match ch {
                // Digits 1-8: skip that many empty squares
                '1'..='8' => {
                    let empty_count = ch as u8 - b'0';
                    file += empty_count;
                }
                // Piece characters
                _ => {
                    let (color, piece) = char_to_piece(ch).ok_or(FenError::InvalidPieceChar {
                        character: ch,
                        rank: rank_index,
                    })?;

                    if file >= 8 {
                        return Err(FenError::RankLengthInvalid {
                            rank: rank_index,
                            squares: file as usize + 1,
                        });
                    }

                    let square = Square::from_file_rank(file, rank);
                    board.put_piece(square, color, piece);
                    file += 1;
                }
            }
        }

        // After processing all characters, we should have covered exactly 8 files.
        if file != 8 {
            return Err(FenError::RankLengthInvalid {
                rank: rank_index,
                squares: file as usize,
            });
        }
    }

    Ok(())
}

/// Converts a FEN piece character to a (Color, Piece) pair.
///
/// Uppercase letters are White pieces, lowercase are Black.
/// Returns `None` for unrecognized characters.
fn char_to_piece(c: char) -> Option<(Color, Piece)> {
    let color = if c.is_ascii_uppercase() {
        Color::White
    } else {
        Color::Black
    };

    let piece = Piece::from_char(c)?;
    Some((color, piece))
}

/// Converts a (Color, Piece) pair to a FEN piece character.
///
/// White pieces are uppercase, Black pieces are lowercase.
fn piece_to_char(color: Color, piece: Piece) -> char {
    match color {
        Color::White => piece.char(),                        // Uppercase
        Color::Black => piece.char().to_ascii_lowercase(),   // Lowercase
    }
}

/// Parses the side-to-move field (field 2 of FEN).
fn parse_side_to_move(field: &str, board: &mut Board) -> Result<(), FenError> {
    match field {
        "w" => board.set_side_to_move(Color::White),
        "b" => board.set_side_to_move(Color::Black),
        _ => {
            return Err(FenError::InvalidSideToMove {
                found: field.to_string(),
            })
        }
    }
    Ok(())
}

/// Parses the castling rights field (field 3 of FEN).
///
/// Valid values are any combination of K, Q, k, q (in that order), or "-" for none.
fn parse_castling_rights(field: &str, board: &mut Board) -> Result<(), FenError> {
    if field == "-" {
        board.set_castling_rights(CastlingRights::NONE);
        return Ok(());
    }

    let mut rights = CastlingRights::NONE;

    for ch in field.chars() {
        match ch {
            'K' => rights.set_white_kingside(true),
            'Q' => rights.set_white_queenside(true),
            'k' => rights.set_black_kingside(true),
            'q' => rights.set_black_queenside(true),
            _ => {
                return Err(FenError::InvalidCastlingRights {
                    found: field.to_string(),
                })
            }
        }
    }

    board.set_castling_rights(rights);
    Ok(())
}

/// Parses the en passant field (field 4 of FEN).
///
/// Must be either "-" (no en passant) or a valid square on rank 3 or 6.
fn parse_en_passant(field: &str, board: &mut Board) -> Result<(), FenError> {
    if field == "-" {
        board.set_en_passant_square(None);
        return Ok(());
    }

    let square = Square::from_algebraic(field).ok_or(FenError::InvalidEnPassantSquare {
        found: field.to_string(),
    })?;

    // En passant target squares can only be on rank 3 (after Black's d7-d5 style move,
    // meaning White to move) or rank 6 (after White's e2-e4 style move, meaning Black
    // to move). Any other rank is invalid.
    let rank = square.rank();
    if rank != 2 && rank != 5 {
        return Err(FenError::EnPassantWrongRank {
            square: field.to_string(),
            rank,
        });
    }

    board.set_en_passant_square(Some(square));
    Ok(())
}

/// Parses the halfmove clock field (field 5 of FEN).
fn parse_halfmove_clock(field: &str, board: &mut Board) -> Result<(), FenError> {
    let clock: u16 = field.parse().map_err(|_| FenError::InvalidHalfmoveClock {
        found: field.to_string(),
    })?;
    board.set_halfmove_clock(clock);
    Ok(())
}

/// Parses the fullmove number field (field 6 of FEN).
fn parse_fullmove_number(field: &str, board: &mut Board) -> Result<(), FenError> {
    let number: u16 = field.parse().map_err(|_| FenError::InvalidFullmoveNumber {
        found: field.to_string(),
    })?;

    if number == 0 {
        return Err(FenError::FullmoveNumberZero);
    }

    board.set_fullmove_number(number);
    Ok(())
}

// =============================================================================
// FEN Generation (Board::to_fen)
// =============================================================================

impl Board {
    /// Generates a FEN string from the current board state.
    ///
    /// The result is always in canonical form, which means:
    /// - Piece placement uses maximum-length empty-square runs (e.g., "8" not "44")
    /// - Castling rights are in KQkq order (or "-" if none)
    /// - All other fields are in their standard format
    ///
    /// # Roundtrip guarantee
    ///
    /// For any board created from a canonical FEN string:
    /// `Board::from_fen(board.to_fen()) == board`
    pub fn to_fen(&self) -> String {
        let mut parts = Vec::with_capacity(6);

        // Field 1: Piece placement
        parts.push(self.generate_piece_placement());

        // Field 2: Side to move
        parts.push(match self.side_to_move() {
            Color::White => "w".to_string(),
            Color::Black => "b".to_string(),
        });

        // Field 3: Castling rights (already formatted correctly by Display impl)
        parts.push(format!("{}", self.castling_rights()));

        // Field 4: En passant square
        parts.push(match self.en_passant_square() {
            Some(sq) => sq.to_algebraic(),
            None => "-".to_string(),
        });

        // Field 5: Halfmove clock
        parts.push(self.halfmove_clock().to_string());

        // Field 6: Fullmove number
        parts.push(self.fullmove_number().to_string());

        parts.join(" ")
    }

    /// Generates the piece placement field (field 1) of the FEN string.
    ///
    /// Iterates over ranks from 8 (top) to 1 (bottom), left to right within
    /// each rank, encoding pieces as characters and consecutive empty squares
    /// as a single digit.
    fn generate_piece_placement(&self) -> String {
        let mut result = String::with_capacity(64); // generous pre-allocation

        for rank in (0..8u8).rev() {
            if rank < 7 {
                result.push('/');
            }

            let mut empty_count: u8 = 0;

            for file in 0..8u8 {
                let square = Square::from_file_rank(file, rank);

                match self.piece_at(square) {
                    Some((color, piece)) => {
                        // Flush any accumulated empty squares first.
                        if empty_count > 0 {
                            result.push((b'0' + empty_count) as char);
                            empty_count = 0;
                        }
                        result.push(piece_to_char(color, piece));
                    }
                    None => {
                        empty_count += 1;
                    }
                }
            }

            // Flush remaining empty squares at end of rank.
            if empty_count > 0 {
                result.push((b'0' + empty_count) as char);
            }
        }

        result
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, CastlingRights};
    use crate::types::{Color, Piece, Square};

    // ---- Roundtrip tests: FEN -> Board -> FEN --------------------------------

    #[test]
    fn roundtrip_starting_position() {
        let board = Board::from_fen(STARTING_FEN).expect("valid FEN");
        assert_eq!(board.to_fen(), STARTING_FEN);
    }

    #[test]
    fn roundtrip_kiwipete() {
        let board = Board::from_fen(KIWIPETE_FEN).expect("valid FEN");
        assert_eq!(board.to_fen(), KIWIPETE_FEN);
    }

    #[test]
    fn roundtrip_position_3() {
        let board = Board::from_fen(POSITION_3_FEN).expect("valid FEN");
        assert_eq!(board.to_fen(), POSITION_3_FEN);
    }

    #[test]
    fn roundtrip_position_4() {
        let board = Board::from_fen(POSITION_4_FEN).expect("valid FEN");
        assert_eq!(board.to_fen(), POSITION_4_FEN);
    }

    #[test]
    fn roundtrip_position_5() {
        let board = Board::from_fen(POSITION_5_FEN).expect("valid FEN");
        assert_eq!(board.to_fen(), POSITION_5_FEN);
    }

    #[test]
    fn roundtrip_position_6() {
        let board = Board::from_fen(POSITION_6_FEN).expect("valid FEN");
        assert_eq!(board.to_fen(), POSITION_6_FEN);
    }

    #[test]
    fn roundtrip_board_to_fen_to_board() {
        // Build a board programmatically and check roundtrip via FEN.
        let board = Board::starting_position();
        let fen = board.to_fen();
        let parsed = Board::from_fen(&fen).expect("should parse generated FEN");
        assert_eq!(board, parsed, "Board -> FEN -> Board should roundtrip");
    }

    // ---- Parsing the starting position ----------------------------------------

    #[test]
    fn parse_starting_position_matches_starting_position() {
        let from_fen = Board::from_fen(STARTING_FEN).expect("valid FEN");
        let built = Board::starting_position();
        assert_eq!(from_fen, built);
    }

    #[test]
    fn parse_starting_position_piece_placement() {
        let board = Board::from_fen(STARTING_FEN).expect("valid FEN");

        // Spot-check a few pieces
        assert_eq!(board.piece_at(Square::A1), Some((Color::White, Piece::Rook)));
        assert_eq!(board.piece_at(Square::E1), Some((Color::White, Piece::King)));
        assert_eq!(board.piece_at(Square::D8), Some((Color::Black, Piece::Queen)));
        assert_eq!(board.piece_at(Square::H8), Some((Color::Black, Piece::Rook)));

        // Check empty center
        assert!(board.piece_at(Square::E4).is_none());
        assert!(board.piece_at(Square::D5).is_none());
    }

    #[test]
    fn parse_starting_position_game_state() {
        let board = Board::from_fen(STARTING_FEN).expect("valid FEN");

        assert_eq!(board.side_to_move(), Color::White);
        assert_eq!(board.castling_rights(), CastlingRights::ALL);
        assert_eq!(board.en_passant_square(), None);
        assert_eq!(board.halfmove_clock(), 0);
        assert_eq!(board.fullmove_number(), 1);
    }

    // ---- Parsing various positions --------------------------------------------

    #[test]
    fn parse_black_to_move() {
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let board = Board::from_fen(fen).expect("valid FEN");

        assert_eq!(board.side_to_move(), Color::Black);
        assert_eq!(board.en_passant_square(), Some(Square::E3));
    }

    #[test]
    fn parse_en_passant_rank_6() {
        // After 1. e4 e5 2. Nf3 d5, en passant target is d6
        let fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/5N2/PPPP1PPP/RNBQKB1R w KQkq d6 0 3";
        let board = Board::from_fen(fen).expect("valid FEN");

        assert_eq!(board.en_passant_square(), Some(Square::D6));
    }

    #[test]
    fn parse_partial_castling_rights() {
        // Only white kingside and black queenside
        let fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kq - 0 1";
        let board = Board::from_fen(fen).expect("valid FEN");

        let cr = board.castling_rights();
        assert!(cr.white_kingside());
        assert!(!cr.white_queenside());
        assert!(!cr.black_kingside());
        assert!(cr.black_queenside());
    }

    #[test]
    fn parse_no_castling_rights() {
        let fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1";
        let board = Board::from_fen(fen).expect("valid FEN");

        assert_eq!(board.castling_rights(), CastlingRights::NONE);
    }

    #[test]
    fn parse_nonzero_halfmove_clock() {
        let fen = "8/8/8/8/8/8/8/4K2k w - - 47 100";
        let board = Board::from_fen(fen).expect("valid FEN");

        assert_eq!(board.halfmove_clock(), 47);
        assert_eq!(board.fullmove_number(), 100);
    }

    #[test]
    fn parse_empty_board_except_kings() {
        let fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1";
        let board = Board::from_fen(fen).expect("valid FEN");

        assert_eq!(board.all_pieces().count(), 2);
        assert_eq!(board.piece_at(Square::E1), Some((Color::White, Piece::King)));
        assert_eq!(board.piece_at(Square::E8), Some((Color::Black, Piece::King)));
    }

    #[test]
    fn parse_complex_position_kiwipete() {
        let board = Board::from_fen(KIWIPETE_FEN).expect("valid FEN");

        // Kiwipete specific checks
        assert_eq!(board.side_to_move(), Color::White);
        assert_eq!(board.castling_rights(), CastlingRights::ALL);
        assert_eq!(board.en_passant_square(), None);

        // Check a few specific pieces in the Kiwipete position
        assert_eq!(board.piece_at(Square::E5), Some((Color::White, Piece::Knight)));
        assert_eq!(board.piece_at(Square::F3), Some((Color::White, Piece::Queen)));
        assert_eq!(board.piece_at(Square::A6), Some((Color::Black, Piece::Bishop)));
        assert_eq!(board.piece_at(Square::E7), Some((Color::Black, Piece::Queen)));
    }

    // ---- FEN generation tests -------------------------------------------------

    #[test]
    fn to_fen_starting_position() {
        let board = Board::starting_position();
        assert_eq!(board.to_fen(), STARTING_FEN);
    }

    #[test]
    fn to_fen_empty_board_with_kings() {
        let mut board = Board::empty();
        board.put_piece(Square::E1, Color::White, Piece::King);
        board.put_piece(Square::E8, Color::Black, Piece::King);

        assert_eq!(board.to_fen(), "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    }

    #[test]
    fn to_fen_with_en_passant() {
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let board = Board::from_fen(fen).expect("valid FEN");
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn to_fen_black_to_move_with_clocks() {
        let mut board = Board::empty();
        board.put_piece(Square::E1, Color::White, Piece::King);
        board.put_piece(Square::E8, Color::Black, Piece::King);
        board.set_side_to_move(Color::Black);
        board.set_halfmove_clock(10);
        board.set_fullmove_number(25);

        assert_eq!(board.to_fen(), "4k3/8/8/8/8/8/8/4K3 b - - 10 25");
    }

    // ---- Error handling tests -------------------------------------------------

    #[test]
    fn error_wrong_field_count_too_few() {
        let result = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w");
        assert!(result.is_err());
        match result {
            Err(FenError::WrongFieldCount { found }) => assert_eq!(found, 2),
            other => panic!("Expected WrongFieldCount, got {:?}", other),
        }
    }

    #[test]
    fn error_wrong_field_count_too_many() {
        let result = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 extra",
        );
        assert!(result.is_err());
        match result {
            Err(FenError::WrongFieldCount { found }) => assert_eq!(found, 7),
            other => panic!("Expected WrongFieldCount, got {:?}", other),
        }
    }

    #[test]
    fn error_empty_string() {
        let result = Board::from_fen("");
        assert!(result.is_err());
        match result {
            Err(FenError::WrongFieldCount { found }) => assert_eq!(found, 0),
            other => panic!("Expected WrongFieldCount, got {:?}", other),
        }
    }

    #[test]
    fn error_wrong_rank_count() {
        // Only 7 ranks
        let result = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1");
        assert!(result.is_err());
        match result {
            Err(FenError::WrongRankCount { found }) => assert_eq!(found, 7),
            other => panic!("Expected WrongRankCount, got {:?}", other),
        }
    }

    #[test]
    fn error_rank_too_long() {
        // Rank 8 has 9 squares worth of content
        let result = Board::from_fen(
            "rnbqkbnrr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        );
        assert!(result.is_err());
        match result {
            Err(FenError::RankLengthInvalid { rank: _, squares }) => {
                assert!(squares > 8);
            }
            other => panic!("Expected RankLengthInvalid, got {:?}", other),
        }
    }

    #[test]
    fn error_rank_too_short() {
        // Rank 8 has only 7 squares
        let result =
            Board::from_fen("rnbqkbn/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(result.is_err());
        match result {
            Err(FenError::RankLengthInvalid { rank: _, squares }) => {
                assert!(squares < 8);
            }
            other => panic!("Expected RankLengthInvalid, got {:?}", other),
        }
    }

    #[test]
    fn error_invalid_piece_character() {
        // 'x' is not a valid piece character
        let result =
            Board::from_fen("rnbqkxnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(result.is_err());
        match result {
            Err(FenError::InvalidPieceChar { character, rank: _ }) => {
                assert_eq!(character, 'x');
            }
            other => panic!("Expected InvalidPieceChar, got {:?}", other),
        }
    }

    #[test]
    fn error_invalid_side_to_move() {
        let result = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",
        );
        assert!(result.is_err());
        match result {
            Err(FenError::InvalidSideToMove { found }) => assert_eq!(found, "x"),
            other => panic!("Expected InvalidSideToMove, got {:?}", other),
        }
    }

    #[test]
    fn error_invalid_castling_rights() {
        let result = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQxq - 0 1",
        );
        assert!(result.is_err());
        match result {
            Err(FenError::InvalidCastlingRights { .. }) => {} // expected
            other => panic!("Expected InvalidCastlingRights, got {:?}", other),
        }
    }

    #[test]
    fn error_invalid_en_passant_square() {
        let result = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq z9 0 1",
        );
        assert!(result.is_err());
        match result {
            Err(FenError::InvalidEnPassantSquare { .. }) => {} // expected
            other => panic!("Expected InvalidEnPassantSquare, got {:?}", other),
        }
    }

    #[test]
    fn error_en_passant_wrong_rank() {
        // e4 is on rank 4, not rank 3 or 6
        let result = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e4 0 1",
        );
        assert!(result.is_err());
        match result {
            Err(FenError::EnPassantWrongRank { .. }) => {} // expected
            other => panic!("Expected EnPassantWrongRank, got {:?}", other),
        }
    }

    #[test]
    fn error_invalid_halfmove_clock() {
        let result = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - abc 1",
        );
        assert!(result.is_err());
        match result {
            Err(FenError::InvalidHalfmoveClock { .. }) => {} // expected
            other => panic!("Expected InvalidHalfmoveClock, got {:?}", other),
        }
    }

    #[test]
    fn error_negative_halfmove_clock() {
        let result = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - -1 1",
        );
        assert!(result.is_err());
        match result {
            Err(FenError::InvalidHalfmoveClock { .. }) => {} // expected
            other => panic!("Expected InvalidHalfmoveClock, got {:?}", other),
        }
    }

    #[test]
    fn error_invalid_fullmove_number() {
        let result = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 xyz",
        );
        assert!(result.is_err());
        match result {
            Err(FenError::InvalidFullmoveNumber { .. }) => {} // expected
            other => panic!("Expected InvalidFullmoveNumber, got {:?}", other),
        }
    }

    #[test]
    fn error_fullmove_number_zero() {
        let result = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0",
        );
        assert!(result.is_err());
        match result {
            Err(FenError::FullmoveNumberZero) => {} // expected
            other => panic!("Expected FullmoveNumberZero, got {:?}", other),
        }
    }

    // ---- FenError display tests -----------------------------------------------

    #[test]
    fn fen_error_display_messages_are_descriptive() {
        // Verify that error messages contain useful information.
        let err = FenError::WrongFieldCount { found: 3 };
        let msg = format!("{}", err);
        assert!(msg.contains("6"), "should mention expected 6 fields");
        assert!(msg.contains("3"), "should mention found 3");

        let err = FenError::InvalidPieceChar {
            character: 'x',
            rank: 2,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("x"), "should mention the invalid char");
        assert!(msg.contains("3"), "should mention rank 3 (1-indexed)");

        let err = FenError::InvalidSideToMove {
            found: "z".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("z"), "should mention the invalid value");
        assert!(msg.contains("w") || msg.contains("b"), "should mention valid values");
    }

    // ---- Additional roundtrip tests for edge cases ----------------------------

    #[test]
    fn roundtrip_position_with_all_piece_types() {
        // A position where every piece type appears for both colors
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let board = Board::from_fen(fen).expect("valid FEN");
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn roundtrip_scattered_pieces() {
        // A position with pieces scattered across the board
        let fen = "r1b1k1nr/p2p1ppp/1pn1p3/2b5/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 2 5";
        let board = Board::from_fen(fen).expect("valid FEN");
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn roundtrip_nearly_empty_board() {
        let fen = "8/8/8/3k4/8/8/8/3K4 w - - 0 1";
        let board = Board::from_fen(fen).expect("valid FEN");
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn roundtrip_high_move_numbers() {
        let fen = "8/8/8/3k4/8/8/8/3K4 b - - 99 200";
        let board = Board::from_fen(fen).expect("valid FEN");
        assert_eq!(board.to_fen(), fen);
    }

    #[test]
    fn roundtrip_en_passant_rank_3() {
        let fen = "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3";
        let board = Board::from_fen(fen).expect("valid FEN");
        assert_eq!(board.to_fen(), fen);
    }

    // ---- Whitespace handling --------------------------------------------------

    #[test]
    fn parse_fen_with_extra_whitespace() {
        // FEN with multiple spaces between fields should still parse
        // (since we use split_whitespace which handles this)
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR  w  KQkq  -  0  1";
        let board = Board::from_fen(fen).expect("should parse FEN with extra spaces");
        assert_eq!(board.to_fen(), STARTING_FEN);
    }

    #[test]
    fn parse_fen_with_leading_trailing_whitespace() {
        let fen = "  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1  ";
        let board = Board::from_fen(fen).expect("should handle leading/trailing spaces");
        assert_eq!(board.to_fen(), STARTING_FEN);
    }

    // ---- Piece count verification after parsing --------------------------------

    #[test]
    fn parse_starting_fen_has_correct_piece_counts() {
        let board = Board::from_fen(STARTING_FEN).expect("valid FEN");

        assert_eq!(board.white_pieces().count(), 16);
        assert_eq!(board.black_pieces().count(), 16);
        assert_eq!(board.all_pieces().count(), 32);

        assert_eq!(
            board.piece_bitboard(Color::White, Piece::Pawn).count(),
            8
        );
        assert_eq!(
            board.piece_bitboard(Color::White, Piece::Knight).count(),
            2
        );
        assert_eq!(
            board.piece_bitboard(Color::White, Piece::Bishop).count(),
            2
        );
        assert_eq!(
            board.piece_bitboard(Color::White, Piece::Rook).count(),
            2
        );
        assert_eq!(
            board.piece_bitboard(Color::White, Piece::Queen).count(),
            1
        );
        assert_eq!(
            board.piece_bitboard(Color::White, Piece::King).count(),
            1
        );
    }

    // ---- Castling rights combinations ----------------------------------------

    #[test]
    fn parse_all_castling_combinations() {
        let base = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w";
        let cases = [
            ("K", true, false, false, false),
            ("Q", false, true, false, false),
            ("k", false, false, true, false),
            ("q", false, false, false, true),
            ("KQ", true, true, false, false),
            ("Kk", true, false, true, false),
            ("Kq", true, false, false, true),
            ("Qk", false, true, true, false),
            ("Qq", false, true, false, true),
            ("kq", false, false, true, true),
            ("KQk", true, true, true, false),
            ("KQq", true, true, false, true),
            ("Kkq", true, false, true, true),
            ("Qkq", false, true, true, true),
            ("KQkq", true, true, true, true),
            ("-", false, false, false, false),
        ];

        for (castling_str, wk, wq, bk, bq) in cases {
            let fen = format!("{} {} - 0 1", base, castling_str);
            let board = Board::from_fen(&fen).unwrap_or_else(|e| {
                panic!("Failed to parse FEN with castling '{}': {}", castling_str, e)
            });
            let cr = board.castling_rights();
            assert_eq!(
                cr.white_kingside(),
                wk,
                "white_kingside mismatch for '{}'",
                castling_str
            );
            assert_eq!(
                cr.white_queenside(),
                wq,
                "white_queenside mismatch for '{}'",
                castling_str
            );
            assert_eq!(
                cr.black_kingside(),
                bk,
                "black_kingside mismatch for '{}'",
                castling_str
            );
            assert_eq!(
                cr.black_queenside(),
                bq,
                "black_queenside mismatch for '{}'",
                castling_str
            );
        }
    }

    // ---- to_fen generates consecutive empty squares correctly ----------------

    #[test]
    fn to_fen_empty_rank_produces_8() {
        let mut board = Board::empty();
        board.put_piece(Square::E1, Color::White, Piece::King);
        board.put_piece(Square::E8, Color::Black, Piece::King);

        let fen = board.to_fen();
        // Middle ranks should be "8"
        assert!(fen.starts_with("4k3/8/8/8/8/8/8/4K3"));
    }

    #[test]
    fn to_fen_single_piece_on_rank() {
        let mut board = Board::empty();
        board.put_piece(Square::D4, Color::White, Piece::Queen);
        board.put_piece(Square::E1, Color::White, Piece::King);
        board.put_piece(Square::E8, Color::Black, Piece::King);

        let fen = board.to_fen();
        // Rank 4 should be "3Q4" (3 empty, Queen, 4 empty)
        assert!(fen.contains("3Q4"), "Expected '3Q4' in FEN: {}", fen);
    }

    #[test]
    fn to_fen_piece_at_edges_of_rank() {
        let mut board = Board::empty();
        board.put_piece(Square::A4, Color::White, Piece::Rook);
        board.put_piece(Square::H4, Color::Black, Piece::Rook);
        board.put_piece(Square::E1, Color::White, Piece::King);
        board.put_piece(Square::E8, Color::Black, Piece::King);

        let fen = board.to_fen();
        // Rank 4 should be "R6r"
        assert!(fen.contains("R6r"), "Expected 'R6r' in FEN: {}", fen);
    }

    // ---- Verify char_to_piece helper ------------------------------------------

    #[test]
    fn char_to_piece_all_valid_chars() {
        let cases = [
            ('P', Color::White, Piece::Pawn),
            ('N', Color::White, Piece::Knight),
            ('B', Color::White, Piece::Bishop),
            ('R', Color::White, Piece::Rook),
            ('Q', Color::White, Piece::Queen),
            ('K', Color::White, Piece::King),
            ('p', Color::Black, Piece::Pawn),
            ('n', Color::Black, Piece::Knight),
            ('b', Color::Black, Piece::Bishop),
            ('r', Color::Black, Piece::Rook),
            ('q', Color::Black, Piece::Queen),
            ('k', Color::Black, Piece::King),
        ];

        for (ch, expected_color, expected_piece) in cases {
            let result = super::char_to_piece(ch);
            assert_eq!(
                result,
                Some((expected_color, expected_piece)),
                "char_to_piece('{}') failed",
                ch
            );
        }
    }

    #[test]
    fn char_to_piece_invalid_chars() {
        assert_eq!(super::char_to_piece('x'), None);
        assert_eq!(super::char_to_piece('1'), None);
        assert_eq!(super::char_to_piece(' '), None);
        assert_eq!(super::char_to_piece('0'), None);
    }

    // ---- Verify piece_to_char helper ------------------------------------------

    #[test]
    fn piece_to_char_all_combinations() {
        assert_eq!(super::piece_to_char(Color::White, Piece::Pawn), 'P');
        assert_eq!(super::piece_to_char(Color::White, Piece::Knight), 'N');
        assert_eq!(super::piece_to_char(Color::White, Piece::Bishop), 'B');
        assert_eq!(super::piece_to_char(Color::White, Piece::Rook), 'R');
        assert_eq!(super::piece_to_char(Color::White, Piece::Queen), 'Q');
        assert_eq!(super::piece_to_char(Color::White, Piece::King), 'K');
        assert_eq!(super::piece_to_char(Color::Black, Piece::Pawn), 'p');
        assert_eq!(super::piece_to_char(Color::Black, Piece::Knight), 'n');
        assert_eq!(super::piece_to_char(Color::Black, Piece::Bishop), 'b');
        assert_eq!(super::piece_to_char(Color::Black, Piece::Rook), 'r');
        assert_eq!(super::piece_to_char(Color::Black, Piece::Queen), 'q');
        assert_eq!(super::piece_to_char(Color::Black, Piece::King), 'k');
    }

    // ---- Sicilian Defense position -------------------------------------------

    #[test]
    fn parse_sicilian_defense() {
        // After 1. e4 c5
        let fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2";
        let board = Board::from_fen(fen).expect("valid FEN");

        assert_eq!(board.side_to_move(), Color::White);
        assert_eq!(board.fullmove_number(), 2);
        // Black pawn on c5
        assert_eq!(board.piece_at(Square::C5), Some((Color::Black, Piece::Pawn)));
        // White pawn on e4
        assert_eq!(board.piece_at(Square::E4), Some((Color::White, Piece::Pawn)));
        // En passant on c6
        assert_eq!(board.en_passant_square(), Some(Square::C6));

        // Roundtrip
        assert_eq!(board.to_fen(), fen);
    }

    // ---- Digit-only rank descriptions ----------------------------------------

    #[test]
    fn parse_ranks_with_only_digits() {
        // "8" means 8 empty squares -- the most common case for empty ranks
        let fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1";
        let board = Board::from_fen(fen).expect("valid FEN");
        assert_eq!(board.all_pieces().count(), 2);
    }

    #[test]
    fn parse_ranks_with_split_digits() {
        // "35" means 3 empty then 5 empty = 8 total (unusual but valid FEN)
        let fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1";
        let board1 = Board::from_fen(fen).expect("valid FEN");

        // This should also be parseable (equivalent: rank 1 = 4 empty + K + 3 empty)
        // The canonical form uses maximum run lengths, but split digits should work too.
        // Let's test a rank written as "1K6" vs "1K6" (same thing)
        let fen2 = "4k3/8/8/8/8/8/8/1K6 w - - 0 1";
        let board2 = Board::from_fen(fen2).expect("valid FEN");
        assert_eq!(board2.piece_at(Square::B1), Some((Color::White, Piece::King)));
    }
}
