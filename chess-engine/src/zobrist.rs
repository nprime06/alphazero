//! Zobrist hashing for position identification.
//!
//! Zobrist hashing assigns a random 64-bit key to every "feature" of a chess
//! position (each piece on each square, castling rights, en passant file, side
//! to move). The hash of a position is the XOR of all its feature keys.
//!
//! The key property: XOR is its own inverse. To add or remove a feature,
//! simply XOR the same key. This makes incremental updates during make/unmake
//! extremely efficient -- instead of recomputing from scratch, we just XOR
//! in the changes.
//!
//! # Determinism
//!
//! The random keys are generated from a fixed-seed PRNG (SplitMix64), so
//! they are identical across runs, platforms, and compilations. This is
//! essential for reproducibility and for sharing transposition tables.

use crate::board::Board;
use crate::types::{Color, Piece, Square, NUM_COLORS, NUM_PIECE_TYPES, NUM_SQUARES};

// =============================================================================
// PRNG: SplitMix64
// =============================================================================

/// A simple, fast, high-quality PRNG used to generate Zobrist keys.
///
/// SplitMix64 was designed by Sebastiano Vigna. It has excellent statistical
/// properties and is the recommended way to seed other PRNGs. For our use
/// case (generating a fixed table of random numbers), it is ideal.
///
/// Reference: <https://prng.di.unimi.it/splitmix64.c>
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a new SplitMix64 generator with the given seed.
    const fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Returns the next pseudo-random u64, advancing the internal state.
    ///
    /// This is a `const fn`-compatible version (no mutable references needed
    /// at the call site because we take `&mut self`).
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

// =============================================================================
// ZobristKeys
// =============================================================================

/// Pre-computed random keys for Zobrist hashing.
///
/// There is one global instance of this struct, lazily initialized on first
/// access. All boards share the same keys (they must, for hashes to be
/// comparable).
pub struct ZobristKeys {
    /// Random key for each (color, piece_type, square) combination.
    /// Indexed as `pieces[color][piece][square]`.
    pub pieces: [[[u64; NUM_SQUARES]; NUM_PIECE_TYPES]; NUM_COLORS],

    /// Random key for each possible castling rights configuration.
    /// There are 4 independent castling bits, giving 16 combinations (0..15).
    /// Indexed by `CastlingRights::bits()` as a `usize`.
    pub castling: [u64; 16],

    /// Random key for the en passant file (0..7).
    /// Only XORed into the hash when an en passant square is actually set.
    pub en_passant: [u64; 8],

    /// Random key XORed when it is Black's turn to move.
    /// When White is to move, this key is NOT included in the hash.
    pub side_to_move: u64,
}

impl ZobristKeys {
    /// Generates the full set of Zobrist keys from a fixed seed.
    ///
    /// The seed is an arbitrary constant chosen to produce a good distribution.
    /// Changing this seed would change all hashes (breaking any saved data),
    /// so it should never be modified.
    fn generate() -> Self {
        // Seed chosen arbitrarily. The hex spells out "AlphaZero Chess" in a
        // playful way, but any non-zero seed would work fine.
        let mut rng = SplitMix64::new(0xA17BA_2E20_C8E55);

        let mut pieces = [[[0u64; NUM_SQUARES]; NUM_PIECE_TYPES]; NUM_COLORS];
        for color in 0..NUM_COLORS {
            for piece in 0..NUM_PIECE_TYPES {
                for square in 0..NUM_SQUARES {
                    pieces[color][piece][square] = rng.next_u64();
                }
            }
        }

        let mut castling = [0u64; 16];
        for rights in 0..16 {
            castling[rights] = rng.next_u64();
        }

        let mut en_passant = [0u64; 8];
        for file in 0..8 {
            en_passant[file] = rng.next_u64();
        }

        let side_to_move = rng.next_u64();

        ZobristKeys {
            pieces,
            castling,
            en_passant,
            side_to_move,
        }
    }
}

// =============================================================================
// Global key table (lazy initialization)
// =============================================================================

use std::sync::LazyLock;

/// The global Zobrist key table, shared by all boards.
///
/// Initialized exactly once on first access. Since the keys are derived from
/// a fixed seed, they are deterministic and identical across runs.
pub static ZOBRIST_KEYS: LazyLock<ZobristKeys> = LazyLock::new(ZobristKeys::generate);

// =============================================================================
// Hash computation
// =============================================================================

/// Computes the Zobrist hash of a board position from scratch.
///
/// This XORs together keys for:
/// 1. Every piece on every square
/// 2. The current castling rights
/// 3. The en passant file (if an en passant square exists)
/// 4. The side to move (if Black)
///
/// This is the "reference" implementation. During normal play, the hash is
/// maintained incrementally by `make_move`. This function is used to:
/// - Initialize the hash when creating a board from FEN or `starting_position`
/// - Verify the incremental hash in debug/test builds
pub fn compute_hash(board: &Board) -> u64 {
    let keys = &*ZOBRIST_KEYS;
    let mut hash: u64 = 0;

    // 1. Piece keys: iterate over every square and XOR in the key for
    //    whatever piece (if any) occupies it.
    for &color in &Color::ALL {
        for &piece in &Piece::ALL {
            let mut bb = board.piece_bitboard(color, piece);
            while !bb.is_empty() {
                let sq = bb.pop_lsb();
                hash ^= keys.pieces[color.index()][piece.index()][sq.index() as usize];
            }
        }
    }

    // 2. Castling rights key
    hash ^= keys.castling[board.castling_rights().bits() as usize];

    // 3. En passant file key (only if en passant is possible)
    if let Some(ep_sq) = board.en_passant_square() {
        hash ^= keys.en_passant[ep_sq.file() as usize];
    }

    // 4. Side to move (XOR when Black to move)
    if board.side_to_move() == Color::Black {
        hash ^= keys.side_to_move;
    }

    hash
}

// =============================================================================
// Convenience accessors for incremental updates
// =============================================================================

/// Returns the Zobrist key for a piece of the given color and type on the given square.
#[inline]
pub fn piece_key(color: Color, piece: Piece, square: Square) -> u64 {
    ZOBRIST_KEYS.pieces[color.index()][piece.index()][square.index() as usize]
}

/// Returns the Zobrist key for the given castling rights configuration.
#[inline]
pub fn castling_key(castling_bits: u8) -> u64 {
    ZOBRIST_KEYS.castling[castling_bits as usize]
}

/// Returns the Zobrist key for the en passant file.
#[inline]
pub fn en_passant_key(file: u8) -> u64 {
    ZOBRIST_KEYS.en_passant[file as usize]
}

/// Returns the Zobrist key for the side-to-move toggle.
#[inline]
pub fn side_to_move_key() -> u64 {
    ZOBRIST_KEYS.side_to_move
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::fen::STARTING_FEN;
    use crate::movegen::generate_legal_moves;
    use crate::moves::{Move, MoveFlags};
    use crate::types::{Piece, Square};

    // ---- Key table properties ------------------------------------------------

    #[test]
    fn keys_are_nonzero() {
        let keys = &*ZOBRIST_KEYS;

        // Side to move key should be nonzero
        assert_ne!(keys.side_to_move, 0, "side_to_move key should be nonzero");

        // Spot-check some piece keys
        assert_ne!(keys.pieces[0][0][0], 0, "piece key [0][0][0] should be nonzero");
        assert_ne!(keys.pieces[1][5][63], 0, "piece key [1][5][63] should be nonzero");
    }

    #[test]
    fn keys_are_deterministic() {
        // Generating keys twice with the same seed should produce the same values.
        let keys1 = ZobristKeys::generate();
        let keys2 = ZobristKeys::generate();

        assert_eq!(keys1.side_to_move, keys2.side_to_move);

        for color in 0..NUM_COLORS {
            for piece in 0..NUM_PIECE_TYPES {
                for sq in 0..NUM_SQUARES {
                    assert_eq!(keys1.pieces[color][piece][sq], keys2.pieces[color][piece][sq]);
                }
            }
        }

        for i in 0..16 {
            assert_eq!(keys1.castling[i], keys2.castling[i]);
        }

        for i in 0..8 {
            assert_eq!(keys1.en_passant[i], keys2.en_passant[i]);
        }
    }

    #[test]
    fn piece_keys_are_all_distinct() {
        // All 768 piece keys (2 colors * 6 pieces * 64 squares) should be unique.
        let keys = &*ZOBRIST_KEYS;
        let mut all_keys = Vec::with_capacity(768);

        for color in 0..NUM_COLORS {
            for piece in 0..NUM_PIECE_TYPES {
                for sq in 0..NUM_SQUARES {
                    all_keys.push(keys.pieces[color][piece][sq]);
                }
            }
        }

        let original_len = all_keys.len();
        all_keys.sort();
        all_keys.dedup();
        assert_eq!(all_keys.len(), original_len, "piece keys should all be unique");
    }

    // ---- From-scratch vs. incremental ----------------------------------------

    #[test]
    fn from_scratch_matches_incremental_after_e4() {
        let mut board = Board::starting_position();
        let mv = Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH);
        board.make_move(mv);

        let expected = compute_hash(&board);
        assert_eq!(board.hash(), expected,
            "Incremental hash after 1.e4 should match from-scratch computation");
    }

    #[test]
    fn from_scratch_matches_incremental_after_move_sequence() {
        let mut board = Board::starting_position();

        // Play the opening: 1.e4 e5 2.Nf3 Nc6 3.Bb5
        let moves = [
            Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH),
            Move::with_flags(Square::E7, Square::E5, MoveFlags::DOUBLE_PAWN_PUSH),
            Move::new(Square::G1, Square::F3),
            Move::new(Square::B8, Square::C6),
            Move::new(Square::F1, Square::B5),
        ];

        for (i, &mv) in moves.iter().enumerate() {
            board.make_move(mv);
            let expected = compute_hash(&board);
            assert_eq!(board.hash(), expected,
                "Hash mismatch after move {} ({})", i + 1, mv.to_uci());
        }
    }

    // ---- Same position, different move order => same hash --------------------

    #[test]
    fn same_position_different_move_order() {
        // Path 1: 1.e4 e5 2.Nf3 Nc6
        let mut board1 = Board::starting_position();
        board1.make_move(Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH));
        board1.make_move(Move::with_flags(Square::E7, Square::E5, MoveFlags::DOUBLE_PAWN_PUSH));
        board1.make_move(Move::new(Square::G1, Square::F3));
        board1.make_move(Move::new(Square::B8, Square::C6));

        // Path 2: 1.Nf3 Nc6 2.e4 e5
        let mut board2 = Board::starting_position();
        board2.make_move(Move::new(Square::G1, Square::F3));
        board2.make_move(Move::new(Square::B8, Square::C6));
        board2.make_move(Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH));
        board2.make_move(Move::with_flags(Square::E7, Square::E5, MoveFlags::DOUBLE_PAWN_PUSH));

        // Note: the en passant square differs (e6 vs e6). Actually, in path 2
        // the last move was e5 (double pawn push), so EP = e6 in both cases.
        // But the halfmove clocks differ (0 vs 0 -- both reset by pawn move).
        // Actually both positions have EP=e6 and halfmove=0.
        // However, the fullmove numbers differ: board1 is on move 3 (white),
        // board2 is on move 3 (white). Both have fullmove=3.
        // Wait -- let me trace this carefully:
        //
        // Path 1: start(1,W) -> e4(1,B) -> e5(2,W) -> Nf3(2,B) -> Nc6(3,W)
        // Path 2: start(1,W) -> Nf3(1,B) -> Nc6(2,W) -> e4(2,B) -> e5(3,W)
        //
        // Board1: fullmove=3, side=W, EP=None (Nc6 is not a double push)
        // Board2: fullmove=3, side=W, EP=e6 (e5 was last move, a double push)
        //
        // The positions have the same pieces but different EP squares!
        // So their hashes SHOULD differ. Let me use a sequence that actually
        // produces the same final position.

        // Instead: compare positions where EP is the same (no EP):
        // Path A: 1.e4 e5 2.Nf3 Nc6 3.d3
        // Path B: 1.Nf3 Nc6 2.d3 e5 3.e4
        //
        // Wait, that's still tricky because d3 is not a double push.
        //
        // Let me just use positions where the last move clears EP for both:
        // Path A: 1.e4 e5 2.Nf3 Nc6
        //   After Nc6: EP=None, side=W, fullmove=3, halfmove=2
        // Path B: 1.Nf3 Nc6 2.e4 e5
        //   After e5: EP=e6, side=W, fullmove=3, halfmove=0
        //
        // Different EP and halfmove. The key insight: halfmove and fullmove
        // are NOT part of Zobrist hash. But EP IS. So we need same EP.
        //
        // Path A: 1.e4 Nc6 2.Nf3 e5
        //   After e5: EP=e6, side=W, fullmove=3, halfmove=0
        // Path B: 1.Nf3 e5 2.e4 Nc6
        //   After Nc6: EP=None, side=W, fullmove=3, halfmove=1
        //
        // Still different EP. The problem is that EP depends on the last move.
        //
        // To get the same hash we need: same pieces, same castling, same EP, same side.
        // Path A: 1.e4 e5 2.Nf3 Nc6 3.Bc4 (no EP, side=B)
        // Path B: 1.Nf3 Nc6 2.Bc4 e5 3.e4 (EP=e3, side=B)
        //   Hmm, e4 is a double push so EP=e3. Different.
        //
        // Actually the simplest approach: end both paths with a non-pawn move.
        // Path A: 1.e4 Nf6 2.Nc3 (EP=None, side=B, fullmove=2, halfmove=1)
        // Path B: 1.Nc3 Nf6 2.e4 (EP=e3, side=B, fullmove=2, halfmove=0)
        //   Different EP again!
        //
        // OK, let me think differently. Both paths need to end with the same
        // "last move type". If both end with a knight move, EP is None for both.
        //
        // Path A: 1.e4 e5 2.Nf3 Nc6 => EP=None (last=Nc6), side=W, pieces same
        // Path B: 1.Nf3 Nc6 2.e4 e5 => EP=e6 (last=e5), side=W, pieces differ in EP
        //
        // Try: make more moves so both end with knight moves.
        // Path A: 1.Nf3 e5 2.e4 Nc6 => EP=None (Nc6 last), side=W
        // Path B: 1.e4 Nc6 2.Nf3 e5 => EP=e6 (e5 last), side=W
        //   Still different!
        //
        // The issue: we need the last move in both paths to be the same type.
        // Path A: 1.e4 e5 2.Nf3 Nc6 3.a3 a6 => EP=None, side=W, fullmove=4
        // Path B: 1.Nf3 Nc6 2.e4 e5 3.a3 a6 => EP=None, side=W, fullmove=4
        //   Now both have same pieces, same EP=None, same side=W, same castling=KQkq.
        //   These should have the same hash!

        // Let me redo this properly:
        let mut board_a = Board::starting_position();
        // Path A: 1.e4 e5 2.Nf3 Nc6 3.a3 a6
        board_a.make_move(Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH));
        board_a.make_move(Move::with_flags(Square::E7, Square::E5, MoveFlags::DOUBLE_PAWN_PUSH));
        board_a.make_move(Move::new(Square::G1, Square::F3));
        board_a.make_move(Move::new(Square::B8, Square::C6));
        board_a.make_move(Move::new(Square::A2, Square::A3));
        board_a.make_move(Move::new(Square::A7, Square::A6));

        let mut board_b = Board::starting_position();
        // Path B: 1.Nf3 Nc6 2.e4 e5 3.a3 a6
        board_b.make_move(Move::new(Square::G1, Square::F3));
        board_b.make_move(Move::new(Square::B8, Square::C6));
        board_b.make_move(Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH));
        board_b.make_move(Move::with_flags(Square::E7, Square::E5, MoveFlags::DOUBLE_PAWN_PUSH));
        board_b.make_move(Move::new(Square::A2, Square::A3));
        board_b.make_move(Move::new(Square::A7, Square::A6));

        assert_eq!(board_a.hash(), board_b.hash(),
            "Same position reached via different move orders should have the same hash");
    }

    // ---- Different positions have different hashes ----------------------------

    #[test]
    fn different_positions_have_different_hashes() {
        let start = Board::starting_position();

        let mut after_e4 = Board::starting_position();
        after_e4.make_move(Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH));

        let mut after_d4 = Board::starting_position();
        after_d4.make_move(Move::with_flags(Square::D2, Square::D4, MoveFlags::DOUBLE_PAWN_PUSH));

        let mut after_nf3 = Board::starting_position();
        after_nf3.make_move(Move::new(Square::G1, Square::F3));

        let hashes = [
            start.hash(),
            after_e4.hash(),
            after_d4.hash(),
            after_nf3.hash(),
        ];

        // All four should be distinct
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j],
                    "Hashes for positions {} and {} should differ", i, j);
            }
        }
    }

    // ---- Side to move matters ------------------------------------------------

    #[test]
    fn side_to_move_affects_hash() {
        // Same pieces but different side to move
        let board_w = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let board_b = Board::from_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1").unwrap();

        assert_ne!(board_w.hash(), board_b.hash(),
            "Same position with different side to move should have different hashes");
    }

    // ---- Castling rights matter -----------------------------------------------

    #[test]
    fn castling_rights_affect_hash() {
        let with_castling = Board::from_fen(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        ).unwrap();
        let without_castling = Board::from_fen(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1"
        ).unwrap();

        assert_ne!(with_castling.hash(), without_castling.hash(),
            "Same position with different castling rights should have different hashes");
    }

    #[test]
    fn partial_castling_rights_differ() {
        let kq = Board::from_fen(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQ - 0 1"
        ).unwrap();
        let kq_kq = Board::from_fen(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        ).unwrap();
        let k = Board::from_fen(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w K - 0 1"
        ).unwrap();

        assert_ne!(kq.hash(), kq_kq.hash());
        assert_ne!(kq.hash(), k.hash());
        assert_ne!(kq_kq.hash(), k.hash());
    }

    // ---- En passant matters ---------------------------------------------------

    #[test]
    fn en_passant_affects_hash() {
        let with_ep = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        ).unwrap();
        let without_ep = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        ).unwrap();

        assert_ne!(with_ep.hash(), without_ep.hash(),
            "Same position with/without en passant should have different hashes");
    }

    // ---- Make/unmake restores hash -------------------------------------------

    #[test]
    fn make_unmake_restores_hash() {
        let mut board = Board::starting_position();
        let original_hash = board.hash();

        let mv = Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH);
        let undo = board.make_move(mv);

        assert_ne!(board.hash(), original_hash,
            "Hash should change after a move");

        board.unmake_move(mv, &undo);

        assert_eq!(board.hash(), original_hash,
            "Hash should be restored after unmake");
    }

    #[test]
    fn make_unmake_restores_hash_all_starting_moves() {
        let original = Board::starting_position();
        let original_hash = original.hash();
        let moves = generate_legal_moves(&original);

        for mv in &moves {
            let mut board = original.clone();
            let undo = board.make_move(*mv);
            board.unmake_move(*mv, &undo);

            assert_eq!(board.hash(), original_hash,
                "Hash not restored after make/unmake of {}", mv.to_uci());
        }
    }

    // ---- Empty board hash ----------------------------------------------------

    #[test]
    fn empty_board_hash_is_consistent() {
        let board = Board::empty();
        let expected = compute_hash(&board);
        assert_eq!(board.hash(), expected,
            "Empty board hash should match from-scratch computation");
    }

    // ---- Starting position hash consistency ----------------------------------

    #[test]
    fn starting_position_hash_consistent_with_fen() {
        let from_constructor = Board::starting_position();
        let from_fen = Board::from_fen(STARTING_FEN).unwrap();

        assert_eq!(from_constructor.hash(), from_fen.hash(),
            "Board::starting_position().hash() should equal Board::from_fen(STARTING_FEN).hash()");
    }

    #[test]
    fn starting_position_hash_matches_from_scratch() {
        let board = Board::starting_position();
        let from_scratch = compute_hash(&board);
        assert_eq!(board.hash(), from_scratch,
            "Starting position hash should match from-scratch computation");
    }

    // ---- All 20 starting moves produce unique hashes -------------------------

    #[test]
    fn all_starting_moves_produce_unique_hashes() {
        let original = Board::starting_position();
        let moves = generate_legal_moves(&original);
        assert_eq!(moves.len(), 20, "Starting position should have 20 legal moves");

        let mut hashes: Vec<u64> = Vec::with_capacity(20);
        for mv in &moves {
            let mut board = original.clone();
            board.make_move(*mv);
            hashes.push(board.hash());
        }

        // Check all hashes are unique
        let original_len = hashes.len();
        hashes.sort();
        hashes.dedup();
        assert_eq!(hashes.len(), original_len,
            "All 20 starting moves should produce unique hashes");
    }

    // ---- Perft-like hash verification ----------------------------------------

    #[test]
    fn hash_matches_from_scratch_after_long_sequence() {
        let mut board = Board::starting_position();

        // Play a longer game fragment with various move types:
        // 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 4.Ba4 Nf6 5.O-O Be7 6.d3 b5 7.Bb3 d6
        let moves = [
            Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH),  // 1. e4
            Move::with_flags(Square::E7, Square::E5, MoveFlags::DOUBLE_PAWN_PUSH),  // 1... e5
            Move::new(Square::G1, Square::F3),                                       // 2. Nf3
            Move::new(Square::B8, Square::C6),                                       // 2... Nc6
            Move::new(Square::F1, Square::B5),                                       // 3. Bb5
            Move::new(Square::A7, Square::A6),                                       // 3... a6
            Move::new(Square::B5, Square::A4),                                       // 4. Ba4
            Move::new(Square::G8, Square::F6),                                       // 4... Nf6
            Move::with_flags(Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE),    // 5. O-O
            Move::new(Square::F8, Square::E7),                                       // 5... Be7
            Move::new(Square::D2, Square::D3),                                       // 6. d3
            Move::with_flags(Square::B7, Square::B5, MoveFlags::DOUBLE_PAWN_PUSH),   // 6... b5
            Move::new(Square::A4, Square::B3),                                       // 7. Bb3
            Move::new(Square::D7, Square::D6),                                       // 7... d6
        ];

        for (i, &mv) in moves.iter().enumerate() {
            board.make_move(mv);
            let from_scratch = compute_hash(&board);
            assert_eq!(board.hash(), from_scratch,
                "Hash mismatch at move {} ({})", i + 1, mv.to_uci());
        }
    }

    // ---- Hash verification with captures and special moves -------------------

    #[test]
    fn hash_correct_after_capture() {
        let mut board = Board::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1").unwrap();
        let mv = Move::new(Square::E4, Square::D5);
        board.make_move(mv);

        let expected = compute_hash(&board);
        assert_eq!(board.hash(), expected, "Hash should be correct after capture");
    }

    #[test]
    fn hash_correct_after_en_passant() {
        let mut board = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let mv = Move::with_flags(Square::E5, Square::D6, MoveFlags::EN_PASSANT);
        board.make_move(mv);

        let expected = compute_hash(&board);
        assert_eq!(board.hash(), expected, "Hash should be correct after en passant");
    }

    #[test]
    fn hash_correct_after_castling() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1").unwrap();
        let mv = Move::with_flags(Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE);
        board.make_move(mv);

        let expected = compute_hash(&board);
        assert_eq!(board.hash(), expected, "Hash should be correct after kingside castling");
    }

    #[test]
    fn hash_correct_after_queenside_castling() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1").unwrap();
        let mv = Move::with_flags(Square::E1, Square::C1, MoveFlags::QUEENSIDE_CASTLE);
        board.make_move(mv);

        let expected = compute_hash(&board);
        assert_eq!(board.hash(), expected, "Hash should be correct after queenside castling");
    }

    #[test]
    fn hash_correct_after_promotion() {
        let mut board = Board::from_fen("8/4P3/8/8/8/6k1/8/4K3 w - - 0 1").unwrap();
        let mv = Move::with_promotion(Square::E7, Square::E8, Piece::Queen);
        board.make_move(mv);

        let expected = compute_hash(&board);
        assert_eq!(board.hash(), expected, "Hash should be correct after promotion");
    }

    #[test]
    fn hash_correct_after_promotion_capture() {
        let mut board = Board::from_fen("3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mv = Move::with_promotion(Square::E7, Square::D8, Piece::Queen);
        board.make_move(mv);

        let expected = compute_hash(&board);
        assert_eq!(board.hash(), expected, "Hash should be correct after promotion capture");
    }

    // ---- Make/unmake roundtrip for special moves -----------------------------

    #[test]
    fn make_unmake_hash_en_passant() {
        let mut board = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let original_hash = board.hash();
        let mv = Move::with_flags(Square::E5, Square::D6, MoveFlags::EN_PASSANT);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_eq!(board.hash(), original_hash,
            "Hash should be restored after en passant make/unmake");
    }

    #[test]
    fn make_unmake_hash_castling() {
        let mut board = Board::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1").unwrap();
        let original_hash = board.hash();
        let mv = Move::with_flags(Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_eq!(board.hash(), original_hash,
            "Hash should be restored after castling make/unmake");
    }

    #[test]
    fn make_unmake_hash_promotion() {
        let mut board = Board::from_fen("8/4P3/8/8/8/6k1/8/4K3 w - - 0 1").unwrap();
        let original_hash = board.hash();
        let mv = Move::with_promotion(Square::E7, Square::E8, Piece::Queen);
        let undo = board.make_move(mv);
        board.unmake_move(mv, &undo);

        assert_eq!(board.hash(), original_hash,
            "Hash should be restored after promotion make/unmake");
    }

    // ---- Comprehensive: verify hash at every ply of all legal moves ----------

    #[test]
    fn hash_correct_for_all_legal_moves_from_kiwipete() {
        let board = Board::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        ).unwrap();
        let moves = generate_legal_moves(&board);

        for mv in &moves {
            let mut b = board.clone();
            let undo = b.make_move(*mv);
            let expected = compute_hash(&b);
            assert_eq!(b.hash(), expected,
                "Hash mismatch for kiwipete move {}", mv.to_uci());

            // Also verify unmake restores
            b.unmake_move(*mv, &undo);
            assert_eq!(b.hash(), board.hash(),
                "Hash not restored after unmake of kiwipete move {}", mv.to_uci());
        }
    }

    #[test]
    fn hash_correct_for_all_legal_moves_from_position_4() {
        let board = Board::from_fen(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
        ).unwrap();
        let moves = generate_legal_moves(&board);

        for mv in &moves {
            let mut b = board.clone();
            let undo = b.make_move(*mv);
            let expected = compute_hash(&b);
            assert_eq!(b.hash(), expected,
                "Hash mismatch for position 4 move {}", mv.to_uci());

            b.unmake_move(*mv, &undo);
            assert_eq!(b.hash(), board.hash(),
                "Hash not restored after unmake of position 4 move {}", mv.to_uci());
        }
    }

    // ---- Two-ply verification ------------------------------------------------

    #[test]
    fn hash_correct_at_depth_2_from_starting_position() {
        let root = Board::starting_position();
        let moves1 = generate_legal_moves(&root);

        for mv1 in &moves1 {
            let mut board1 = root.clone();
            board1.make_move(*mv1);
            assert_eq!(board1.hash(), compute_hash(&board1),
                "Depth 1 hash mismatch for {}", mv1.to_uci());

            let moves2 = generate_legal_moves(&board1);
            for mv2 in &moves2 {
                let mut board2 = board1.clone();
                board2.make_move(*mv2);
                assert_eq!(board2.hash(), compute_hash(&board2),
                    "Depth 2 hash mismatch for {} {}", mv1.to_uci(), mv2.to_uci());
            }
        }
    }
}
