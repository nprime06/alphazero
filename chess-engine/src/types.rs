//! Core chess types: Square, Piece, and Color.
//!
//! These are the building blocks used throughout the engine. All types are
//! lightweight (Copy) and designed for use in arrays and bitboard operations.

use std::fmt;

// =============================================================================
// Square
// =============================================================================

/// The number of squares on a chess board.
pub const NUM_SQUARES: usize = 64;

/// The number of files (columns) on a chess board: a through h.
pub const NUM_FILES: usize = 8;

/// The number of ranks (rows) on a chess board: 1 through 8.
pub const NUM_RANKS: usize = 8;

/// A square on the chess board, represented as an index from 0 to 63.
///
/// The mapping uses rank-major order (also called Little-Endian Rank-File, or LERF):
///   - A1 = 0, B1 = 1, ..., H1 = 7
///   - A2 = 8, B2 = 9, ..., H2 = 15
///   - ...
///   - A8 = 56, B8 = 57, ..., H8 = 63
///
/// This layout means:
///   - `index = rank * 8 + file`  (rank and file are 0-based)
///   - `file  = index % 8`        (column: 0=a, 1=b, ..., 7=h)
///   - `rank  = index / 8`        (row:    0=1st rank, ..., 7=8th rank)
///
/// This is the standard convention for bitboard engines because bit 0
/// corresponds to A1 and bit 63 to H8, which keeps the mapping simple.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Square(u8);

impl Square {
    /// Creates a square from a raw index (0..63).
    ///
    /// # Panics
    /// Panics in debug mode if `index >= 64`.
    #[inline]
    pub const fn new(index: u8) -> Self {
        debug_assert!(index < NUM_SQUARES as u8, "Square index out of range");
        Square(index)
    }

    /// Creates a square from file (0=a..7=h) and rank (0=1st..7=8th).
    ///
    /// # Panics
    /// Panics in debug mode if file or rank is out of range.
    #[inline]
    pub const fn from_file_rank(file: u8, rank: u8) -> Self {
        debug_assert!(file < NUM_FILES as u8, "File out of range");
        debug_assert!(rank < NUM_RANKS as u8, "Rank out of range");
        Square(rank * NUM_FILES as u8 + file)
    }

    /// Returns the raw index (0..63).
    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Returns the file (column) as 0=a, 1=b, ..., 7=h.
    #[inline]
    pub const fn file(self) -> u8 {
        self.0 % NUM_FILES as u8
    }

    /// Returns the rank (row) as 0=1st rank, 1=2nd rank, ..., 7=8th rank.
    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 / NUM_FILES as u8
    }

    /// Returns the file as a lowercase character: 'a'..'h'.
    #[inline]
    pub const fn file_char(self) -> char {
        (b'a' + self.file()) as char
    }

    /// Returns the rank as a character: '1'..'8'.
    #[inline]
    pub const fn rank_char(self) -> char {
        (b'1' + self.rank()) as char
    }

    /// Creates a square from algebraic notation like "e4" or "a1".
    ///
    /// Returns `None` if the string is not a valid square name.
    pub fn from_algebraic(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return None;
        }

        let file_byte = bytes[0];
        let rank_byte = bytes[1];

        // File must be 'a'..'h' (or 'A'..'H')
        let file = match file_byte {
            b'a'..=b'h' => file_byte - b'a',
            b'A'..=b'H' => file_byte - b'A',
            _ => return None,
        };

        // Rank must be '1'..'8'
        let rank = match rank_byte {
            b'1'..=b'8' => rank_byte - b'1',
            _ => return None,
        };

        Some(Square::from_file_rank(file, rank))
    }

    /// Returns the algebraic notation for this square (e.g., "e4").
    pub fn to_algebraic(self) -> String {
        format!("{}{}", self.file_char(), self.rank_char())
    }
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Square({}={})", self.0, self.to_algebraic())
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_algebraic())
    }
}

// ---------------------------------------------------------------------------
// Well-known square constants, named for convenient use in tests and move
// generation. We define all 64 but group them by rank for readability.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
impl Square {
    // Rank 1
    pub const A1: Square = Square(0);
    pub const B1: Square = Square(1);
    pub const C1: Square = Square(2);
    pub const D1: Square = Square(3);
    pub const E1: Square = Square(4);
    pub const F1: Square = Square(5);
    pub const G1: Square = Square(6);
    pub const H1: Square = Square(7);

    // Rank 2
    pub const A2: Square = Square(8);
    pub const B2: Square = Square(9);
    pub const C2: Square = Square(10);
    pub const D2: Square = Square(11);
    pub const E2: Square = Square(12);
    pub const F2: Square = Square(13);
    pub const G2: Square = Square(14);
    pub const H2: Square = Square(15);

    // Rank 3
    pub const A3: Square = Square(16);
    pub const B3: Square = Square(17);
    pub const C3: Square = Square(18);
    pub const D3: Square = Square(19);
    pub const E3: Square = Square(20);
    pub const F3: Square = Square(21);
    pub const G3: Square = Square(22);
    pub const H3: Square = Square(23);

    // Rank 4
    pub const A4: Square = Square(24);
    pub const B4: Square = Square(25);
    pub const C4: Square = Square(26);
    pub const D4: Square = Square(27);
    pub const E4: Square = Square(28);
    pub const F4: Square = Square(29);
    pub const G4: Square = Square(30);
    pub const H4: Square = Square(31);

    // Rank 5
    pub const A5: Square = Square(32);
    pub const B5: Square = Square(33);
    pub const C5: Square = Square(34);
    pub const D5: Square = Square(35);
    pub const E5: Square = Square(36);
    pub const F5: Square = Square(37);
    pub const G5: Square = Square(38);
    pub const H5: Square = Square(39);

    // Rank 6
    pub const A6: Square = Square(40);
    pub const B6: Square = Square(41);
    pub const C6: Square = Square(42);
    pub const D6: Square = Square(43);
    pub const E6: Square = Square(44);
    pub const F6: Square = Square(45);
    pub const G6: Square = Square(46);
    pub const H6: Square = Square(47);

    // Rank 7
    pub const A7: Square = Square(48);
    pub const B7: Square = Square(49);
    pub const C7: Square = Square(50);
    pub const D7: Square = Square(51);
    pub const E7: Square = Square(52);
    pub const F7: Square = Square(53);
    pub const G7: Square = Square(54);
    pub const H7: Square = Square(55);

    // Rank 8
    pub const A8: Square = Square(56);
    pub const B8: Square = Square(57);
    pub const C8: Square = Square(58);
    pub const D8: Square = Square(59);
    pub const E8: Square = Square(60);
    pub const F8: Square = Square(61);
    pub const G8: Square = Square(62);
    pub const H8: Square = Square(63);
}

// =============================================================================
// Piece
// =============================================================================

/// The number of distinct piece types in chess.
pub const NUM_PIECE_TYPES: usize = 6;

/// A chess piece type (without color information).
///
/// The integer values (0..5) are chosen so they can index into arrays of size 6.
/// Pawn = 0 because pawns are the most common piece and are often special-cased,
/// so having them at index 0 keeps things tidy.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl Piece {
    /// All piece types in order, useful for iteration.
    pub const ALL: [Piece; NUM_PIECE_TYPES] = [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ];

    /// Returns the piece type as an array index (0..5).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Creates a piece from an index (0..5).
    ///
    /// Returns `None` if the index is out of range.
    pub const fn from_index(index: usize) -> Option<Piece> {
        match index {
            0 => Some(Piece::Pawn),
            1 => Some(Piece::Knight),
            2 => Some(Piece::Bishop),
            3 => Some(Piece::Rook),
            4 => Some(Piece::Queen),
            5 => Some(Piece::King),
            _ => None,
        }
    }

    /// Returns the standard single-character abbreviation for this piece.
    /// Uses uppercase: P, N, B, R, Q, K.
    pub const fn char(self) -> char {
        match self {
            Piece::Pawn => 'P',
            Piece::Knight => 'N',
            Piece::Bishop => 'B',
            Piece::Rook => 'R',
            Piece::Queen => 'Q',
            Piece::King => 'K',
        }
    }

    /// Creates a piece from its single-character abbreviation.
    /// Accepts both upper and lowercase: p/P, n/N, b/B, r/R, q/Q, k/K.
    ///
    /// Returns `None` for unrecognized characters.
    pub const fn from_char(c: char) -> Option<Piece> {
        match c {
            'P' | 'p' => Some(Piece::Pawn),
            'N' | 'n' => Some(Piece::Knight),
            'B' | 'b' => Some(Piece::Bishop),
            'R' | 'r' => Some(Piece::Rook),
            'Q' | 'q' => Some(Piece::Queen),
            'K' | 'k' => Some(Piece::King),
            _ => None,
        }
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.char())
    }
}

// =============================================================================
// Color
// =============================================================================

/// The number of colors (sides) in chess.
pub const NUM_COLORS: usize = 2;

/// A player color / side.
///
/// White = 0, Black = 1 so they can index into arrays of size 2.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    /// Both colors in order, useful for iteration.
    pub const ALL: [Color; NUM_COLORS] = [Color::White, Color::Black];

    /// Returns the color as an array index (0 or 1).
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Returns the opposite color.
    #[inline]
    pub const fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Color::White => write!(f, "White"),
            Color::Black => write!(f, "Black"),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Square tests -------------------------------------------------------

    #[test]
    fn square_new_and_index_roundtrip() {
        for i in 0..64u8 {
            let sq = Square::new(i);
            assert_eq!(sq.index(), i);
        }
    }

    #[test]
    fn square_file_rank_roundtrip() {
        for rank in 0..8u8 {
            for file in 0..8u8 {
                let sq = Square::from_file_rank(file, rank);
                assert_eq!(sq.file(), file, "file mismatch for ({file}, {rank})");
                assert_eq!(sq.rank(), rank, "rank mismatch for ({file}, {rank})");
                assert_eq!(
                    sq.index(),
                    rank * 8 + file,
                    "index mismatch for ({file}, {rank})"
                );
            }
        }
    }

    #[test]
    fn square_algebraic_roundtrip() {
        // Check every square: index -> algebraic -> index
        for i in 0..64u8 {
            let sq = Square::new(i);
            let alg = sq.to_algebraic();
            let parsed = Square::from_algebraic(&alg).expect("should parse");
            assert_eq!(parsed, sq, "roundtrip failed for index {i}");
        }
    }

    #[test]
    fn square_known_values() {
        // Spot-check a few well-known squares.
        assert_eq!(Square::A1.index(), 0);
        assert_eq!(Square::H1.index(), 7);
        assert_eq!(Square::A8.index(), 56);
        assert_eq!(Square::H8.index(), 63);
        assert_eq!(Square::E1.index(), 4);
        assert_eq!(Square::E4.index(), 28);
        assert_eq!(Square::D5.index(), 35);
    }

    #[test]
    fn square_a1_is_file_0_rank_0() {
        let sq = Square::A1;
        assert_eq!(sq.file(), 0);
        assert_eq!(sq.rank(), 0);
        assert_eq!(sq.file_char(), 'a');
        assert_eq!(sq.rank_char(), '1');
        assert_eq!(sq.to_algebraic(), "a1");
    }

    #[test]
    fn square_h8_is_file_7_rank_7() {
        let sq = Square::H8;
        assert_eq!(sq.file(), 7);
        assert_eq!(sq.rank(), 7);
        assert_eq!(sq.file_char(), 'h');
        assert_eq!(sq.rank_char(), '8');
        assert_eq!(sq.to_algebraic(), "h8");
    }

    #[test]
    fn square_e4_breakdown() {
        let sq = Square::from_algebraic("e4").unwrap();
        assert_eq!(sq.file(), 4); // e = file 4
        assert_eq!(sq.rank(), 3); // 4th rank = index 3 (0-based)
        assert_eq!(sq.index(), 28); // 3 * 8 + 4 = 28
    }

    #[test]
    fn square_from_algebraic_case_insensitive() {
        assert_eq!(
            Square::from_algebraic("E4"),
            Square::from_algebraic("e4")
        );
        assert_eq!(
            Square::from_algebraic("A1"),
            Square::from_algebraic("a1")
        );
    }

    #[test]
    fn square_from_algebraic_invalid() {
        assert!(Square::from_algebraic("").is_none());
        assert!(Square::from_algebraic("a").is_none());
        assert!(Square::from_algebraic("a9").is_none());
        assert!(Square::from_algebraic("i1").is_none());
        assert!(Square::from_algebraic("a0").is_none());
        assert!(Square::from_algebraic("abc").is_none());
        assert!(Square::from_algebraic("11").is_none());
    }

    #[test]
    fn square_display() {
        assert_eq!(format!("{}", Square::E4), "e4");
        assert_eq!(format!("{}", Square::A1), "a1");
        assert_eq!(format!("{}", Square::H8), "h8");
    }

    #[test]
    fn square_constants_match_from_file_rank() {
        // Verify every named constant matches the from_file_rank constructor.
        let all_named = [
            (Square::A1, 0, 0), (Square::B1, 1, 0), (Square::C1, 2, 0), (Square::D1, 3, 0),
            (Square::E1, 4, 0), (Square::F1, 5, 0), (Square::G1, 6, 0), (Square::H1, 7, 0),
            (Square::A2, 0, 1), (Square::B2, 1, 1), (Square::C2, 2, 1), (Square::D2, 3, 1),
            (Square::E2, 4, 1), (Square::F2, 5, 1), (Square::G2, 6, 1), (Square::H2, 7, 1),
            (Square::A3, 0, 2), (Square::B3, 1, 2), (Square::C3, 2, 2), (Square::D3, 3, 2),
            (Square::E3, 4, 2), (Square::F3, 5, 2), (Square::G3, 6, 2), (Square::H3, 7, 2),
            (Square::A4, 0, 3), (Square::B4, 1, 3), (Square::C4, 2, 3), (Square::D4, 3, 3),
            (Square::E4, 4, 3), (Square::F4, 5, 3), (Square::G4, 6, 3), (Square::H4, 7, 3),
            (Square::A5, 0, 4), (Square::B5, 1, 4), (Square::C5, 2, 4), (Square::D5, 3, 4),
            (Square::E5, 4, 4), (Square::F5, 5, 4), (Square::G5, 6, 4), (Square::H5, 7, 4),
            (Square::A6, 0, 5), (Square::B6, 1, 5), (Square::C6, 2, 5), (Square::D6, 3, 5),
            (Square::E6, 4, 5), (Square::F6, 5, 5), (Square::G6, 6, 5), (Square::H6, 7, 5),
            (Square::A7, 0, 6), (Square::B7, 1, 6), (Square::C7, 2, 6), (Square::D7, 3, 6),
            (Square::E7, 4, 6), (Square::F7, 5, 6), (Square::G7, 6, 6), (Square::H7, 7, 6),
            (Square::A8, 0, 7), (Square::B8, 1, 7), (Square::C8, 2, 7), (Square::D8, 3, 7),
            (Square::E8, 4, 7), (Square::F8, 5, 7), (Square::G8, 6, 7), (Square::H8, 7, 7),
        ];
        for (constant, file, rank) in all_named {
            assert_eq!(
                constant,
                Square::from_file_rank(file, rank),
                "Mismatch for file={file}, rank={rank}"
            );
        }
    }

    // ---- Piece tests --------------------------------------------------------

    #[test]
    fn piece_index_roundtrip() {
        for piece in Piece::ALL {
            let idx = piece.index();
            let roundtripped = Piece::from_index(idx).expect("valid index");
            assert_eq!(roundtripped, piece);
        }
    }

    #[test]
    fn piece_from_index_out_of_range() {
        assert!(Piece::from_index(6).is_none());
        assert!(Piece::from_index(255).is_none());
    }

    #[test]
    fn piece_char_roundtrip() {
        for piece in Piece::ALL {
            let c = piece.char();
            let roundtripped = Piece::from_char(c).expect("valid char");
            assert_eq!(roundtripped, piece);
        }
    }

    #[test]
    fn piece_from_char_lowercase() {
        assert_eq!(Piece::from_char('p'), Some(Piece::Pawn));
        assert_eq!(Piece::from_char('n'), Some(Piece::Knight));
        assert_eq!(Piece::from_char('b'), Some(Piece::Bishop));
        assert_eq!(Piece::from_char('r'), Some(Piece::Rook));
        assert_eq!(Piece::from_char('q'), Some(Piece::Queen));
        assert_eq!(Piece::from_char('k'), Some(Piece::King));
    }

    #[test]
    fn piece_from_char_invalid() {
        assert!(Piece::from_char('x').is_none());
        assert!(Piece::from_char('1').is_none());
        assert!(Piece::from_char(' ').is_none());
    }

    #[test]
    fn piece_display() {
        assert_eq!(format!("{}", Piece::Pawn), "P");
        assert_eq!(format!("{}", Piece::Knight), "N");
        assert_eq!(format!("{}", Piece::King), "K");
    }

    #[test]
    fn piece_all_has_correct_count() {
        assert_eq!(Piece::ALL.len(), NUM_PIECE_TYPES);
    }

    // ---- Color tests --------------------------------------------------------

    #[test]
    fn color_index() {
        assert_eq!(Color::White.index(), 0);
        assert_eq!(Color::Black.index(), 1);
    }

    #[test]
    fn color_flip() {
        assert_eq!(Color::White.flip(), Color::Black);
        assert_eq!(Color::Black.flip(), Color::White);
    }

    #[test]
    fn color_flip_is_involution() {
        // Flipping twice returns to the original color.
        for color in Color::ALL {
            assert_eq!(color.flip().flip(), color);
        }
    }

    #[test]
    fn color_display() {
        assert_eq!(format!("{}", Color::White), "White");
        assert_eq!(format!("{}", Color::Black), "Black");
    }

    #[test]
    fn color_all_has_correct_count() {
        assert_eq!(Color::ALL.len(), NUM_COLORS);
    }
}
