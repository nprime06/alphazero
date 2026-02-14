//! Pseudo-legal move generation.
//!
//! This module generates all pseudo-legal moves for a given position. "Pseudo-legal"
//! means the moves follow piece movement rules but might leave the king in check.
//! Filtering out illegal moves (those that leave the king in check) is handled
//! separately in Chunk 1.9.
//!
//! # Move generation by piece type
//!
//! Each piece type has its own generation logic:
//! - **Pawns**: The most complex piece -- single pushes, double pushes, captures,
//!   en passant, and promotions (which multiply into 4 moves each).
//! - **Knights**: Simple lookup from precomputed attack tables.
//! - **Bishops/Rooks/Queens**: Magic bitboard lookups with occupancy.
//! - **King**: Precomputed attack table plus special castling logic.
//!
//! # Why pseudo-legal?
//!
//! Generating pseudo-legal moves first and filtering later is the standard approach
//! in bitboard engines. It is simpler to implement and often faster overall, because
//! many positions have few moves that actually leave the king in check. The filtering
//! step (make the move, check if own king is attacked, unmake) is done in the legal
//! move generator.

use crate::attacks::{is_in_check, is_square_attacked, is_square_attacked_ignoring, king_attacks, knight_attacks, pawn_attacks};
use crate::bitboard::{between_bb, Bitboard};
use crate::board::Board;
use crate::magic::{bishop_attacks, queen_attacks, rook_attacks};
use crate::moves::{Move, MoveFlags};
use crate::types::{Color, Piece, Square};

// =============================================================================
// Promotion piece list
// =============================================================================

/// The four pieces a pawn can promote to, in a standard order.
/// Queen is first because it is by far the most common promotion.
const PROMOTION_PIECES: [Piece; 4] = [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight];

// =============================================================================
// Main entry point
// =============================================================================

/// Generates all pseudo-legal moves for the given board position.
///
/// Returns a `Vec<Move>` containing every move that follows piece movement rules
/// for the side to move. Some of these moves may leave the king in check and
/// thus be illegal -- that filtering happens in a later step (Chunk 1.9).
///
/// The order of moves in the returned vector is not specified and should not be
/// relied upon. For move ordering (important for search efficiency), a separate
/// scoring/sorting step is applied.
pub fn generate_pseudo_legal_moves(board: &Board) -> Vec<Move> {
    // Pre-allocate with a reasonable capacity. Most positions have 30-40 legal
    // moves; pseudo-legal is slightly more. 50 is a good average.
    let mut moves = Vec::with_capacity(50);

    let us = board.side_to_move();
    let them = us.flip();
    let our_pieces = board.pieces(us);
    let their_pieces = board.pieces(them);
    let all_pieces = our_pieces | their_pieces;

    // Generate moves for each piece type
    generate_pawn_moves(board, us, our_pieces, their_pieces, all_pieces, &mut moves);
    generate_knight_moves(board, us, our_pieces, &mut moves);
    generate_bishop_moves(board, us, our_pieces, all_pieces, &mut moves);
    generate_rook_moves(board, us, our_pieces, all_pieces, &mut moves);
    generate_queen_moves(board, us, our_pieces, all_pieces, &mut moves);
    generate_king_moves(board, us, our_pieces, all_pieces, &mut moves);

    moves
}

// =============================================================================
// Legal move generation
// =============================================================================

/// Generates all legal moves for the given board position.
///
/// Uses pin detection and check evasion to avoid board cloning. The only
/// exception is en passant captures, which use the old clone+check approach
/// because removing two pawns from the same rank can create discovered checks
/// that pin analysis alone doesn't catch.
///
/// # Approach
///
/// 1. Compute checkers (pieces giving check) and pin information once.
/// 2. Always generate king moves (king can always try to escape).
/// 3. In double check, only king moves are legal (return early).
/// 4. For non-king pieces:
///    - If in single check, moves must capture the checker or block the check ray.
///    - Pinned pieces can only move along their pin ray.
///    - Pinned knights can never move (a knight can't stay on a line).
pub fn generate_legal_moves(board: &Board) -> Vec<Move> {
    let us = board.side_to_move();
    let them = us.flip();
    let king_bb = board.piece_bitboard(us, Piece::King);
    if king_bb.is_empty() {
        return Vec::new();
    }
    let king_sq = king_bb.lsb();

    let (num_checkers, _checkers, check_ray) = compute_checkers(board, king_sq, them);
    let (pinned, pin_rays) = compute_pins(board, king_sq, us);

    let mut moves = Vec::with_capacity(50);

    // King moves are always generated (king can always try to escape)
    generate_king_legal_moves(board, king_sq, us, them, &mut moves);

    // In double check, only king moves are legal
    if num_checkers >= 2 {
        return moves;
    }

    let our_pieces = board.pieces(us);

    // For non-king pieces: if in single check, moves must either
    // capture the checker or block the check ray
    let move_mask = if num_checkers == 1 {
        check_ray // must block or capture
    } else {
        !our_pieces // any square not occupied by our pieces
    };

    // Generate moves for each non-king piece type
    generate_pawn_legal_moves(board, us, them, king_sq, &pinned, &pin_rays, &move_mask, &mut moves);
    generate_knight_legal_moves(board, us, &pinned, &move_mask, &mut moves);
    generate_slider_legal_moves(board, us, &pinned, &pin_rays, &move_mask, &mut moves);

    // Castling (only when not in check)
    if num_checkers == 0 {
        generate_castling_legal_moves(board, us, them, &mut moves);
    }

    moves
}

// =============================================================================
// Pin and check computation
// =============================================================================

/// Computes which of our pieces are absolutely pinned to the king, and the
/// ray along which each pinned piece can move.
///
/// Returns `(pinned_bitboard, pin_ray_per_square)` where `pin_ray_per_square[sq]`
/// is the set of squares a piece on `sq` may move to (includes the pinner and
/// all squares between king and pinner). For non-pinned squares, the ray is
/// `Bitboard::FULL` (no restriction).
fn compute_pins(board: &Board, king_sq: Square, us: Color) -> (Bitboard, [Bitboard; 64]) {
    let them = us.flip();
    let our_pieces = board.pieces(us);

    let mut pinned = Bitboard::EMPTY;
    let mut pin_rays = [Bitboard::FULL; 64];

    // Check rook/queen pins (straight lines)
    let enemy_straight = board.piece_bitboard(them, Piece::Rook)
                       | board.piece_bitboard(them, Piece::Queen);
    // Get potential pinners: enemy straight sliders that could see the king
    // if only enemy pieces were blocking (our pieces are transparent)
    let potential_straight_pinners = rook_attacks(king_sq, board.pieces(them)) & enemy_straight;

    for pinner_sq in potential_straight_pinners {
        let between = between_bb(king_sq, pinner_sq);
        let blockers = between & our_pieces;
        if blockers.count() == 1 {
            let pinned_sq = blockers.lsb();
            pinned = pinned | Bitboard::from_square(pinned_sq);
            pin_rays[pinned_sq.index() as usize] = between | Bitboard::from_square(pinner_sq);
        }
    }

    // Check bishop/queen pins (diagonals)
    let enemy_diagonal = board.piece_bitboard(them, Piece::Bishop)
                       | board.piece_bitboard(them, Piece::Queen);
    let potential_diag_pinners = bishop_attacks(king_sq, board.pieces(them)) & enemy_diagonal;

    for pinner_sq in potential_diag_pinners {
        let between = between_bb(king_sq, pinner_sq);
        let blockers = between & our_pieces;
        if blockers.count() == 1 {
            let pinned_sq = blockers.lsb();
            pinned = pinned | Bitboard::from_square(pinned_sq);
            pin_rays[pinned_sq.index() as usize] = between | Bitboard::from_square(pinner_sq);
        }
    }

    (pinned, pin_rays)
}

/// Computes which enemy pieces are giving check, how many, and the check ray.
///
/// Returns `(num_checkers, checkers_bitboard, check_ray)`:
/// - `num_checkers`: 0, 1, or 2
/// - `checkers`: bitboard of the checking piece(s)
/// - `check_ray`: if exactly one checker, the ray between king and checker
///   (inclusive of checker, exclusive of king). Blocking or capturing on this
///   ray resolves the check. If 0 or 2 checkers, this is `Bitboard::FULL`.
fn compute_checkers(board: &Board, king_sq: Square, them: Color) -> (u8, Bitboard, Bitboard) {
    let occupancy = board.all_pieces();
    let mut checkers = Bitboard::EMPTY;

    // Pawn checks
    checkers = checkers | (pawn_attacks(them.flip(), king_sq) & board.piece_bitboard(them, Piece::Pawn));
    // Knight checks
    checkers = checkers | (knight_attacks(king_sq) & board.piece_bitboard(them, Piece::Knight));
    // Bishop/Queen checks (diagonal)
    let diag = board.piece_bitboard(them, Piece::Bishop) | board.piece_bitboard(them, Piece::Queen);
    checkers = checkers | (bishop_attacks(king_sq, occupancy) & diag);
    // Rook/Queen checks (straight)
    let straight = board.piece_bitboard(them, Piece::Rook) | board.piece_bitboard(them, Piece::Queen);
    checkers = checkers | (rook_attacks(king_sq, occupancy) & straight);

    let num = checkers.count() as u8;

    let check_ray = if num == 1 {
        let checker_sq = checkers.lsb();
        between_bb(king_sq, checker_sq) | checkers
    } else {
        Bitboard::FULL
    };

    (num, checkers, check_ray)
}

// =============================================================================
// Piece-specific legal move generators
// =============================================================================

/// Generates legal king moves (normal moves, not castling).
///
/// For each adjacent square, checks if it is attacked by the opponent
/// with the king removed from the board (to prevent the king from
/// hiding behind itself along a sliding ray).
fn generate_king_legal_moves(
    board: &Board,
    king_sq: Square,
    us: Color,
    them: Color,
    moves: &mut Vec<Move>,
) {
    let our_pieces = board.pieces(us);
    let targets = king_attacks(king_sq) & !our_pieces;

    for to_sq in targets {
        if !is_square_attacked_ignoring(board, to_sq, them, king_sq) {
            moves.push(Move::new(king_sq, to_sq));
        }
    }
}

/// Generates legal pawn moves (pushes, captures, en passant, promotions).
///
/// Handles:
/// - Single and double pushes (filtered by pin ray and check ray)
/// - Captures including promotion captures
/// - En passant (uses clone+check for the rare horizontal discovered check case)
fn generate_pawn_legal_moves(
    board: &Board,
    us: Color,
    them: Color,
    _king_sq: Square,
    pinned: &Bitboard,
    pin_rays: &[Bitboard; 64],
    move_mask: &Bitboard,
    moves: &mut Vec<Move>,
) {
    let pawns = board.piece_bitboard(us, Piece::Pawn);
    let their_pieces = board.pieces(them);
    let all_pieces = board.all_pieces();
    let empty = !all_pieces;

    let (push_offset, start_rank, promo_rank): (i8, Bitboard, u8) = match us {
        Color::White => (8, Bitboard::RANK_2, 7),
        Color::Black => (-8, Bitboard::RANK_7, 0),
    };

    // --- Single pushes ---
    let single_pushes = match us {
        Color::White => (pawns << 8) & empty,
        Color::Black => (pawns >> 8) & empty,
    };

    for to_sq in single_pushes {
        let from_sq = Square::new((to_sq.index() as i8 - push_offset) as u8);

        // Apply pin constraint
        let pin_ok = !pinned.contains(from_sq) || pin_rays[from_sq.index() as usize].contains(to_sq);
        // Apply check evasion constraint
        let check_ok = move_mask.contains(to_sq);

        if pin_ok && check_ok {
            if to_sq.rank() == promo_rank {
                for &promo_piece in &PROMOTION_PIECES {
                    moves.push(Move::with_promotion(from_sq, to_sq, promo_piece));
                }
            } else {
                moves.push(Move::new(from_sq, to_sq));
            }
        }
    }

    // --- Double pushes ---
    let on_start_rank = pawns & start_rank;
    let intermediate = match us {
        Color::White => (on_start_rank << 8) & empty,
        Color::Black => (on_start_rank >> 8) & empty,
    };
    let double_pushes = match us {
        Color::White => (intermediate << 8) & empty,
        Color::Black => (intermediate >> 8) & empty,
    };

    for to_sq in double_pushes {
        let from_sq = Square::new((to_sq.index() as i8 - 2 * push_offset) as u8);

        let pin_ok = !pinned.contains(from_sq) || pin_rays[from_sq.index() as usize].contains(to_sq);
        let check_ok = move_mask.contains(to_sq);

        if pin_ok && check_ok {
            moves.push(Move::with_flags(from_sq, to_sq, MoveFlags::DOUBLE_PAWN_PUSH));
        }
    }

    // --- Captures ---
    for from_sq in pawns {
        let attack_targets = pawn_attacks(us, from_sq) & their_pieces;

        for to_sq in attack_targets {
            let pin_ok = !pinned.contains(from_sq) || pin_rays[from_sq.index() as usize].contains(to_sq);
            let check_ok = move_mask.contains(to_sq);

            if pin_ok && check_ok {
                if to_sq.rank() == promo_rank {
                    for &promo_piece in &PROMOTION_PIECES {
                        moves.push(Move::with_promotion(from_sq, to_sq, promo_piece));
                    }
                } else {
                    moves.push(Move::new(from_sq, to_sq));
                }
            }
        }
    }

    // --- En passant ---
    // En passant is tricky: removing both the capturing pawn and the captured pawn
    // from the same rank can create a discovered horizontal check. We use the old
    // clone+check approach only for en passant (which is rare).
    if let Some(ep_square) = board.en_passant_square() {
        let ep_bb = Bitboard::from_square(ep_square);

        for from_sq in pawns {
            if (pawn_attacks(us, from_sq) & ep_bb).is_not_empty() {
                // The captured pawn's square
                let captured_pawn_sq = match us {
                    Color::White => Square::new(ep_square.index() - 8),
                    Color::Black => Square::new(ep_square.index() + 8),
                };

                // Check if the ep capture satisfies the check evasion constraint.
                // The ep capture can resolve check by either:
                // - Moving to a square on the check ray (blocking)
                // - Capturing the checker (the captured pawn IS the checker)
                let check_ok = move_mask.contains(ep_square) || move_mask.contains(captured_pawn_sq);

                if !check_ok {
                    continue;
                }

                // For en passant, use clone+check to handle the tricky discovered check case
                let mut new_board = board.clone();
                new_board.make_move(Move::with_flags(from_sq, ep_square, MoveFlags::EN_PASSANT));
                if !is_in_check(&new_board, us) {
                    moves.push(Move::with_flags(from_sq, ep_square, MoveFlags::EN_PASSANT));
                }
            }
        }
    }
}

/// Generates legal knight moves.
///
/// A pinned knight can never move (it can't stay on the pin line),
/// so we only generate moves for non-pinned knights.
fn generate_knight_legal_moves(
    board: &Board,
    us: Color,
    pinned: &Bitboard,
    move_mask: &Bitboard,
    moves: &mut Vec<Move>,
) {
    // Only non-pinned knights can move
    let knights = board.piece_bitboard(us, Piece::Knight) & !*pinned;

    for from_sq in knights {
        let targets = knight_attacks(from_sq) & *move_mask;

        for to_sq in targets {
            moves.push(Move::new(from_sq, to_sq));
        }
    }
}

/// Generates legal moves for sliding pieces (bishops, rooks, queens).
///
/// For pinned sliders, intersects their attack set with the pin ray.
/// For all sliders, intersects with the move_mask (check evasion).
fn generate_slider_legal_moves(
    board: &Board,
    us: Color,
    pinned: &Bitboard,
    pin_rays: &[Bitboard; 64],
    move_mask: &Bitboard,
    moves: &mut Vec<Move>,
) {
    let our_pieces = board.pieces(us);
    let all_pieces = board.all_pieces();

    // Bishops
    let bishops = board.piece_bitboard(us, Piece::Bishop);
    for from_sq in bishops {
        let mut targets = bishop_attacks(from_sq, all_pieces) & !our_pieces & *move_mask;
        if pinned.contains(from_sq) {
            targets = targets & pin_rays[from_sq.index() as usize];
        }
        for to_sq in targets {
            moves.push(Move::new(from_sq, to_sq));
        }
    }

    // Rooks
    let rooks = board.piece_bitboard(us, Piece::Rook);
    for from_sq in rooks {
        let mut targets = rook_attacks(from_sq, all_pieces) & !our_pieces & *move_mask;
        if pinned.contains(from_sq) {
            targets = targets & pin_rays[from_sq.index() as usize];
        }
        for to_sq in targets {
            moves.push(Move::new(from_sq, to_sq));
        }
    }

    // Queens
    let queens = board.piece_bitboard(us, Piece::Queen);
    for from_sq in queens {
        let mut targets = queen_attacks(from_sq, all_pieces) & !our_pieces & *move_mask;
        if pinned.contains(from_sq) {
            targets = targets & pin_rays[from_sq.index() as usize];
        }
        for to_sq in targets {
            moves.push(Move::new(from_sq, to_sq));
        }
    }
}

/// Generates legal castling moves.
///
/// Only called when not in check. Checks:
/// - Castling rights are available
/// - Squares between king and rook are empty
/// - King does not pass through or land on an attacked square
fn generate_castling_legal_moves(
    board: &Board,
    us: Color,
    them: Color,
    moves: &mut Vec<Move>,
) {
    let all_pieces = board.all_pieces();
    let castling = board.castling_rights();

    match us {
        Color::White => {
            if castling.white_kingside() {
                let between = Bitboard::from_square(Square::F1) | Bitboard::from_square(Square::G1);
                if (all_pieces & between).is_empty()
                    && !is_square_attacked(board, Square::F1, them)
                    && !is_square_attacked(board, Square::G1, them)
                {
                    moves.push(Move::with_flags(Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE));
                }
            }
            if castling.white_queenside() {
                let between = Bitboard::from_square(Square::B1)
                    | Bitboard::from_square(Square::C1)
                    | Bitboard::from_square(Square::D1);
                if (all_pieces & between).is_empty()
                    && !is_square_attacked(board, Square::D1, them)
                    && !is_square_attacked(board, Square::C1, them)
                {
                    moves.push(Move::with_flags(Square::E1, Square::C1, MoveFlags::QUEENSIDE_CASTLE));
                }
            }
        }
        Color::Black => {
            if castling.black_kingside() {
                let between = Bitboard::from_square(Square::F8) | Bitboard::from_square(Square::G8);
                if (all_pieces & between).is_empty()
                    && !is_square_attacked(board, Square::F8, them)
                    && !is_square_attacked(board, Square::G8, them)
                {
                    moves.push(Move::with_flags(Square::E8, Square::G8, MoveFlags::KINGSIDE_CASTLE));
                }
            }
            if castling.black_queenside() {
                let between = Bitboard::from_square(Square::B8)
                    | Bitboard::from_square(Square::C8)
                    | Bitboard::from_square(Square::D8);
                if (all_pieces & between).is_empty()
                    && !is_square_attacked(board, Square::D8, them)
                    && !is_square_attacked(board, Square::C8, them)
                {
                    moves.push(Move::with_flags(Square::E8, Square::C8, MoveFlags::QUEENSIDE_CASTLE));
                }
            }
        }
    }
}

// =============================================================================
// Pawn move generation
// =============================================================================

/// Generates all pseudo-legal pawn moves: single pushes, double pushes,
/// captures (including en passant), and promotions.
///
/// Pawn moves are the most complex because:
/// 1. Pawns move differently depending on color (north vs south).
/// 2. Pushes require empty target squares (unlike other pieces).
/// 3. Double pushes are only from the starting rank and require two empty squares.
/// 4. Promotions generate 4 moves each (Q, R, B, N).
/// 5. En passant is a special capture with its own flag.
fn generate_pawn_moves(
    board: &Board,
    us: Color,
    _our_pieces: Bitboard,
    their_pieces: Bitboard,
    all_pieces: Bitboard,
    moves: &mut Vec<Move>,
) {
    let pawns = board.piece_bitboard(us, Piece::Pawn);
    let empty = !all_pieces;

    // Direction constants depend on color.
    // White pawns move "north" (toward higher ranks), black pawns move "south".
    let (push_offset, start_rank, promo_rank): (i8, Bitboard, u8) = match us {
        Color::White => (8, Bitboard::RANK_2, 7),   // rank index 7 = rank 8
        Color::Black => (-8, Bitboard::RANK_7, 0),  // rank index 0 = rank 1
    };

    // --- Single pushes ---
    // Advance each pawn one rank forward. The target must be empty.
    // We compute all single pushes at once using bitboard shifts.
    let single_pushes = match us {
        Color::White => (pawns << 8) & empty,
        Color::Black => (pawns >> 8) & empty,
    };

    for to_sq in single_pushes {
        let from_sq = Square::new((to_sq.index() as i8 - push_offset) as u8);

        if to_sq.rank() == promo_rank {
            // Pawn reaches promotion rank -- generate all four promotion moves
            for &promo_piece in &PROMOTION_PIECES {
                moves.push(Move::with_promotion(from_sq, to_sq, promo_piece));
            }
        } else {
            moves.push(Move::new(from_sq, to_sq));
        }
    }

    // --- Double pushes ---
    // Only from the starting rank, and both the intermediate and destination
    // squares must be empty.
    let on_start_rank = pawns & start_rank;
    let intermediate = match us {
        Color::White => (on_start_rank << 8) & empty,
        Color::Black => (on_start_rank >> 8) & empty,
    };
    let double_pushes = match us {
        Color::White => (intermediate << 8) & empty,
        Color::Black => (intermediate >> 8) & empty,
    };

    for to_sq in double_pushes {
        let from_sq = Square::new((to_sq.index() as i8 - 2 * push_offset) as u8);
        moves.push(Move::with_flags(from_sq, to_sq, MoveFlags::DOUBLE_PAWN_PUSH));
    }

    // --- Captures ---
    // Use the precomputed pawn attack tables. A pawn can capture to a square
    // if it is occupied by an enemy piece.
    for from_sq in pawns {
        let attack_targets = pawn_attacks(us, from_sq) & their_pieces;

        for to_sq in attack_targets {
            if to_sq.rank() == promo_rank {
                // Capture onto promotion rank -- generate all four promotions
                for &promo_piece in &PROMOTION_PIECES {
                    moves.push(Move::with_promotion(from_sq, to_sq, promo_piece));
                }
            } else {
                moves.push(Move::new(from_sq, to_sq));
            }
        }
    }

    // --- En passant ---
    // If the opponent just made a double pawn push, we can capture en passant
    // to the square "behind" that pawn.
    if let Some(ep_square) = board.en_passant_square() {
        let ep_bb = Bitboard::from_square(ep_square);

        for from_sq in pawns {
            if (pawn_attacks(us, from_sq) & ep_bb).is_not_empty() {
                moves.push(Move::with_flags(from_sq, ep_square, MoveFlags::EN_PASSANT));
            }
        }
    }
}

// =============================================================================
// Knight move generation
// =============================================================================

/// Generates all pseudo-legal knight moves.
///
/// For each knight, look up its precomputed attack table and exclude squares
/// occupied by our own pieces (we cannot capture our own pieces).
fn generate_knight_moves(
    board: &Board,
    us: Color,
    our_pieces: Bitboard,
    moves: &mut Vec<Move>,
) {
    let knights = board.piece_bitboard(us, Piece::Knight);

    for from_sq in knights {
        let targets = knight_attacks(from_sq) & !our_pieces;

        for to_sq in targets {
            moves.push(Move::new(from_sq, to_sq));
        }
    }
}

// =============================================================================
// Bishop move generation
// =============================================================================

/// Generates all pseudo-legal bishop moves using magic bitboard lookups.
fn generate_bishop_moves(
    board: &Board,
    us: Color,
    our_pieces: Bitboard,
    all_pieces: Bitboard,
    moves: &mut Vec<Move>,
) {
    let bishops = board.piece_bitboard(us, Piece::Bishop);

    for from_sq in bishops {
        let targets = bishop_attacks(from_sq, all_pieces) & !our_pieces;

        for to_sq in targets {
            moves.push(Move::new(from_sq, to_sq));
        }
    }
}

// =============================================================================
// Rook move generation
// =============================================================================

/// Generates all pseudo-legal rook moves using magic bitboard lookups.
fn generate_rook_moves(
    board: &Board,
    us: Color,
    our_pieces: Bitboard,
    all_pieces: Bitboard,
    moves: &mut Vec<Move>,
) {
    let rooks = board.piece_bitboard(us, Piece::Rook);

    for from_sq in rooks {
        let targets = rook_attacks(from_sq, all_pieces) & !our_pieces;

        for to_sq in targets {
            moves.push(Move::new(from_sq, to_sq));
        }
    }
}

// =============================================================================
// Queen move generation
// =============================================================================

/// Generates all pseudo-legal queen moves using magic bitboard lookups.
///
/// A queen combines the movement of a bishop and a rook, so we use
/// `queen_attacks` which is `bishop_attacks | rook_attacks`.
fn generate_queen_moves(
    board: &Board,
    us: Color,
    our_pieces: Bitboard,
    all_pieces: Bitboard,
    moves: &mut Vec<Move>,
) {
    let queens = board.piece_bitboard(us, Piece::Queen);

    for from_sq in queens {
        let targets = queen_attacks(from_sq, all_pieces) & !our_pieces;

        for to_sq in targets {
            moves.push(Move::new(from_sq, to_sq));
        }
    }
}

// =============================================================================
// King move generation (including castling)
// =============================================================================

/// Generates all pseudo-legal king moves, including castling.
///
/// Normal king moves use the precomputed attack table, excluding own pieces.
/// Castling requires:
/// - The appropriate castling right is still available
/// - The squares between the king and rook are empty
///
/// Note: For pseudo-legal generation, we do NOT check whether the king passes
/// through or lands on an attacked square. That check is deferred to the legal
/// move filter in Chunk 1.9.
fn generate_king_moves(
    board: &Board,
    us: Color,
    our_pieces: Bitboard,
    all_pieces: Bitboard,
    moves: &mut Vec<Move>,
) {
    let king_bb = board.piece_bitboard(us, Piece::King);

    // There should always be exactly one king, but if the bitboard is empty
    // (malformed position), bail out gracefully.
    if king_bb.is_empty() {
        return;
    }

    let from_sq = king_bb.lsb();

    // --- Normal king moves ---
    let targets = king_attacks(from_sq) & !our_pieces;
    for to_sq in targets {
        moves.push(Move::new(from_sq, to_sq));
    }

    // --- Castling ---
    let castling = board.castling_rights();

    match us {
        Color::White => {
            // White kingside castling: King e1 -> g1
            // Requires: f1 and g1 empty
            if castling.white_kingside() {
                let between = Bitboard::from_square(Square::F1) | Bitboard::from_square(Square::G1);
                if (all_pieces & between).is_empty() {
                    moves.push(Move::with_flags(
                        Square::E1,
                        Square::G1,
                        MoveFlags::KINGSIDE_CASTLE,
                    ));
                }
            }

            // White queenside castling: King e1 -> c1
            // Requires: b1, c1, and d1 empty (3 squares for queenside!)
            if castling.white_queenside() {
                let between = Bitboard::from_square(Square::B1)
                    | Bitboard::from_square(Square::C1)
                    | Bitboard::from_square(Square::D1);
                if (all_pieces & between).is_empty() {
                    moves.push(Move::with_flags(
                        Square::E1,
                        Square::C1,
                        MoveFlags::QUEENSIDE_CASTLE,
                    ));
                }
            }
        }
        Color::Black => {
            // Black kingside castling: King e8 -> g8
            // Requires: f8 and g8 empty
            if castling.black_kingside() {
                let between = Bitboard::from_square(Square::F8) | Bitboard::from_square(Square::G8);
                if (all_pieces & between).is_empty() {
                    moves.push(Move::with_flags(
                        Square::E8,
                        Square::G8,
                        MoveFlags::KINGSIDE_CASTLE,
                    ));
                }
            }

            // Black queenside castling: King e8 -> c8
            // Requires: b8, c8, and d8 empty
            if castling.black_queenside() {
                let between = Bitboard::from_square(Square::B8)
                    | Bitboard::from_square(Square::C8)
                    | Bitboard::from_square(Square::D8);
                if (all_pieces & between).is_empty() {
                    moves.push(Move::with_flags(
                        Square::E8,
                        Square::C8,
                        MoveFlags::QUEENSIDE_CASTLE,
                    ));
                }
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::fen;
    use crate::moves::MoveFlags;
    use crate::types::{Piece, Square};

    // ---- Helper functions ---------------------------------------------------

    /// Returns true if the move list contains a move with the given from/to squares.
    fn has_move(moves: &[Move], from: Square, to: Square) -> bool {
        moves.iter().any(|m| m.from == from && m.to == to)
    }

    /// Returns true if the move list contains a move matching the given from/to/flags.
    fn has_move_with_flags(moves: &[Move], from: Square, to: Square, flags: MoveFlags) -> bool {
        moves
            .iter()
            .any(|m| m.from == from && m.to == to && m.flags == flags)
    }

    /// Returns true if the move list contains a promotion move from->to with the given piece.
    fn has_promotion(moves: &[Move], from: Square, to: Square, piece: Piece) -> bool {
        moves
            .iter()
            .any(|m| m.from == from && m.to == to && m.promotion == Some(piece))
    }

    /// Counts how many moves originate from the given square.
    fn count_moves_from(moves: &[Move], from: Square) -> usize {
        moves.iter().filter(|m| m.from == from).count()
    }

    // ---- Starting position --------------------------------------------------

    #[test]
    fn starting_position_has_20_moves() {
        let board = Board::starting_position();
        let moves = generate_pseudo_legal_moves(&board);

        // Starting position: 16 pawn moves (8 single + 8 double) + 4 knight moves = 20
        assert_eq!(
            moves.len(),
            20,
            "Starting position should have exactly 20 pseudo-legal moves, got {}",
            moves.len()
        );
    }

    #[test]
    fn starting_position_pawn_moves() {
        let board = Board::starting_position();
        let moves = generate_pseudo_legal_moves(&board);

        // Each of the 8 pawns should have exactly 2 moves (single push + double push)
        for file in 0..8u8 {
            let from = Square::from_file_rank(file, 1); // rank 2 for white pawns
            let count = count_moves_from(&moves, from);
            assert_eq!(
                count, 2,
                "White pawn on {} should have 2 moves, got {}",
                from, count
            );
        }
    }

    #[test]
    fn starting_position_knight_moves() {
        let board = Board::starting_position();
        let moves = generate_pseudo_legal_moves(&board);

        // Each knight should have 2 moves (the only squares not blocked by own pieces)
        assert_eq!(count_moves_from(&moves, Square::B1), 2);
        assert_eq!(count_moves_from(&moves, Square::G1), 2);

        // Verify specific knight targets
        assert!(has_move(&moves, Square::B1, Square::A3));
        assert!(has_move(&moves, Square::B1, Square::C3));
        assert!(has_move(&moves, Square::G1, Square::F3));
        assert!(has_move(&moves, Square::G1, Square::H3));
    }

    // ---- Kiwipete position --------------------------------------------------

    #[test]
    fn kiwipete_has_48_pseudo_legal_moves() {
        let board = Board::from_fen(fen::KIWIPETE_FEN).expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        assert_eq!(
            moves.len(),
            48,
            "Kiwipete position should have 48 pseudo-legal moves, got {}.\nMoves: {:?}",
            moves.len(),
            moves
        );
    }

    // ---- Pawn single pushes -------------------------------------------------

    #[test]
    fn pawn_single_push_from_custom_position() {
        // White pawn on e4, black pawn on e5 blocks the push.
        // White pawn on d2 can push to d3.
        let board = Board::from_fen("4k3/8/8/4p3/4P3/8/3P4/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // e4 pawn cannot push (e5 is blocked)
        assert!(
            !has_move(&moves, Square::E4, Square::E5),
            "e4 pawn should not be able to push to e5 (blocked)"
        );

        // d2 pawn can push to d3
        assert!(
            has_move(&moves, Square::D2, Square::D3),
            "d2 pawn should be able to push to d3"
        );
    }

    // ---- Pawn double push ---------------------------------------------------

    #[test]
    fn double_pawn_push_has_correct_flag() {
        let board = Board::starting_position();
        let moves = generate_pseudo_legal_moves(&board);

        // e2-e4 should have the DOUBLE_PAWN_PUSH flag
        assert!(
            has_move_with_flags(&moves, Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH),
            "e2e4 should have DOUBLE_PAWN_PUSH flag"
        );

        // e2-e3 should NOT have the flag
        let e2e3 = moves
            .iter()
            .find(|m| m.from == Square::E2 && m.to == Square::E3);
        assert!(e2e3.is_some(), "e2e3 should exist");
        assert_eq!(
            e2e3.unwrap().flags,
            MoveFlags::NONE,
            "e2e3 should have no special flags"
        );
    }

    #[test]
    fn double_pawn_push_blocked_by_intermediate_square() {
        // White pawn on e2, piece on e3 blocks double push but e3 is blocked too.
        let board = Board::from_fen("4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // e2 cannot push at all (e3 is blocked)
        assert!(
            !has_move(&moves, Square::E2, Square::E3),
            "e2 pawn cannot push to e3 (blocked)"
        );
        assert!(
            !has_move(&moves, Square::E2, Square::E4),
            "e2 pawn cannot double push to e4 (e3 blocked)"
        );
    }

    #[test]
    fn double_pawn_push_blocked_by_destination() {
        // White pawn on e2, piece on e4 blocks the double push destination.
        // But e3 is open, so single push is fine.
        let board = Board::from_fen("4k3/8/8/8/4p3/8/4P3/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // e2 can push to e3
        assert!(has_move(&moves, Square::E2, Square::E3));
        // e2 cannot double push to e4 (destination blocked)
        assert!(!has_move(&moves, Square::E2, Square::E4));
    }

    // ---- Pawn captures ------------------------------------------------------

    #[test]
    fn pawn_captures_diagonal_enemies() {
        // White pawn on e4, black pawns on d5 and f5.
        let board = Board::from_fen("4k3/8/8/3p1p2/4P3/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // e4 pawn can capture d5 and f5
        assert!(
            has_move(&moves, Square::E4, Square::D5),
            "e4 pawn should capture on d5"
        );
        assert!(
            has_move(&moves, Square::E4, Square::F5),
            "e4 pawn should capture on f5"
        );
    }

    #[test]
    fn pawn_does_not_capture_own_pieces() {
        // White pawn on e4 with white pieces on d5 and f5.
        let board = Board::from_fen("4k3/8/8/3P1P2/4P3/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // e4 pawn cannot capture own pieces
        assert!(!has_move(&moves, Square::E4, Square::D5));
        assert!(!has_move(&moves, Square::E4, Square::F5));
    }

    // ---- Pawn promotions ----------------------------------------------------

    #[test]
    fn pawn_promotion_generates_four_moves() {
        // White pawn on e7, no pieces blocking e8. Black king is off to the side.
        let board = Board::from_fen("8/4P3/8/8/8/8/6k1/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // Should generate 4 promotion moves for e7-e8
        assert!(has_promotion(&moves, Square::E7, Square::E8, Piece::Queen));
        assert!(has_promotion(&moves, Square::E7, Square::E8, Piece::Rook));
        assert!(has_promotion(&moves, Square::E7, Square::E8, Piece::Bishop));
        assert!(has_promotion(&moves, Square::E7, Square::E8, Piece::Knight));

        // Count: should be exactly 4 moves from e7 to e8
        let promo_moves: Vec<_> = moves
            .iter()
            .filter(|m| m.from == Square::E7 && m.to == Square::E8)
            .collect();
        assert_eq!(
            promo_moves.len(),
            4,
            "Should have exactly 4 promotion moves e7e8"
        );
    }

    #[test]
    fn promotion_capture_generates_four_moves() {
        // White pawn on e7, black piece on d8 -- can capture-promote.
        let board = Board::from_fen("3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // Capture-promotion to d8 should generate 4 moves
        assert!(has_promotion(&moves, Square::E7, Square::D8, Piece::Queen));
        assert!(has_promotion(&moves, Square::E7, Square::D8, Piece::Rook));
        assert!(has_promotion(&moves, Square::E7, Square::D8, Piece::Bishop));
        assert!(has_promotion(&moves, Square::E7, Square::D8, Piece::Knight));

        let capture_promos: Vec<_> = moves
            .iter()
            .filter(|m| m.from == Square::E7 && m.to == Square::D8)
            .collect();
        assert_eq!(capture_promos.len(), 4);
    }

    #[test]
    fn black_pawn_promotion() {
        // Black pawn on a2 can promote on a1.
        let board = Board::from_fen("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        assert!(has_promotion(&moves, Square::A2, Square::A1, Piece::Queen));
        assert!(has_promotion(&moves, Square::A2, Square::A1, Piece::Rook));
        assert!(has_promotion(&moves, Square::A2, Square::A1, Piece::Bishop));
        assert!(has_promotion(&moves, Square::A2, Square::A1, Piece::Knight));
    }

    // ---- En passant ---------------------------------------------------------

    #[test]
    fn en_passant_generates_move_with_flag() {
        // White pawn on e5, black pawn just played d7-d5. En passant target is d6.
        let board = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        assert!(
            has_move_with_flags(&moves, Square::E5, Square::D6, MoveFlags::EN_PASSANT),
            "Should generate en passant capture e5xd6"
        );
    }

    #[test]
    fn en_passant_not_generated_without_ep_square() {
        // Same position but no en passant square set.
        let board = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // No en passant move should be generated
        let ep_moves: Vec<_> = moves.iter().filter(|m| m.is_en_passant()).collect();
        assert!(
            ep_moves.is_empty(),
            "Should not generate en passant without ep square"
        );
    }

    #[test]
    fn black_en_passant() {
        // Black pawn on d4, white pawn just played e2-e4. En passant target is e3.
        let board = Board::from_fen("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        assert!(
            has_move_with_flags(&moves, Square::D4, Square::E3, MoveFlags::EN_PASSANT),
            "Should generate black en passant capture d4xe3"
        );
    }

    // ---- Knight moves -------------------------------------------------------

    #[test]
    fn knight_on_e4_attacks_8_squares_on_empty_board() {
        // Lone white knight on e4, only kings on board.
        let board = Board::from_fen("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // Knight on e4 should have 8 target squares (minus any own pieces).
        // Only own piece is King on e1, which is not in the knight's attack set.
        let knight_moves = count_moves_from(&moves, Square::E4);
        assert_eq!(
            knight_moves, 8,
            "Knight on e4 should have 8 moves, got {}",
            knight_moves
        );

        // Verify specific targets
        assert!(has_move(&moves, Square::E4, Square::F6));
        assert!(has_move(&moves, Square::E4, Square::G5));
        assert!(has_move(&moves, Square::E4, Square::G3));
        assert!(has_move(&moves, Square::E4, Square::F2));
        assert!(has_move(&moves, Square::E4, Square::D2));
        assert!(has_move(&moves, Square::E4, Square::C3));
        assert!(has_move(&moves, Square::E4, Square::C5));
        assert!(has_move(&moves, Square::E4, Square::D6));
    }

    #[test]
    fn knight_blocked_by_own_pieces() {
        // Knight on e4 with own pieces on some target squares.
        let board = Board::from_fen("4k3/8/3P1P2/8/4N3/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // Knight cannot go to d6 or f6 (own pawns)
        let knight_moves: Vec<_> = moves
            .iter()
            .filter(|m| m.from == Square::E4 && m.to == Square::D6)
            .collect();
        assert!(knight_moves.is_empty(), "Knight cannot capture own pawn on d6");

        let knight_moves: Vec<_> = moves
            .iter()
            .filter(|m| m.from == Square::E4 && m.to == Square::F6)
            .collect();
        assert!(knight_moves.is_empty(), "Knight cannot capture own pawn on f6");

        // But can still go to the other 6 squares
        assert_eq!(count_moves_from(&moves, Square::E4), 6);
    }

    // ---- Bishop moves -------------------------------------------------------

    #[test]
    fn bishop_on_e4_empty_board() {
        let board = Board::from_fen("4k3/8/8/8/4B3/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // Bishop on e4 empty board = 13 squares.
        // But our king is on e1, which is not on any of the bishop's diagonals, so 13.
        let bishop_moves = count_moves_from(&moves, Square::E4);
        assert_eq!(
            bishop_moves, 13,
            "Bishop on e4 should have 13 moves on near-empty board, got {}",
            bishop_moves
        );
    }

    // ---- Rook moves ---------------------------------------------------------

    #[test]
    fn rook_on_a1_empty_board() {
        // Rook on a1, kings elsewhere.
        let board = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // Rook on a1: 14 squares on empty board, but king on e1 blocks part of rank 1.
        // Rook sees: a2-a8 (7 north) + b1, c1, d1 (3 east, blocked by own King on e1).
        // Total from rook: 10
        // Actually: file a up = a2,a3,a4,a5,a6,a7,a8 = 7, rank 1 east = b1,c1,d1 = 3 (e1 is own king, blocked)
        // Total: 10
        let rook_moves = count_moves_from(&moves, Square::A1);
        assert_eq!(
            rook_moves, 10,
            "Rook on a1 with king on e1 should have 10 moves, got {}",
            rook_moves
        );
    }

    #[test]
    fn rook_on_a1_truly_empty_board() {
        // Just a rook on a1 and kings far away.
        let board = Board::from_fen("4k3/8/8/8/8/8/8/R6K w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // Rook on a1: a2-a8 (7) + b1-g1 (6, h1 is own king) = 13
        let rook_moves = count_moves_from(&moves, Square::A1);
        assert_eq!(
            rook_moves, 13,
            "Rook on a1 with king on h1 should have 13 moves, got {}",
            rook_moves
        );
    }

    // ---- Queen moves --------------------------------------------------------

    #[test]
    fn queen_generates_union_of_bishop_and_rook_moves() {
        // Queen on e4 with minimal other pieces.
        let board = Board::from_fen("4k3/8/8/8/4Q3/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // Queen on e4 on a near-empty board:
        // Rook component: 14 squares, minus e1 (own king) = 13
        // Bishop component: 13 squares
        // But we need to check overlap: none between rook and bishop moves.
        // So queen should have 13 + 13 = 26 moves.
        // Wait: rook has e1-e8 vertically (7), a4-h4 horizontally (7) = 14.
        // But e1 has own king, so the rook ray south stops at e2 (e1 is own piece).
        // Actually rook sees: north e5,e6,e7,e8 (4), south e3,e2 (2, e1 blocked by own king),
        // east f4,g4,h4 (3), west d4,c4,b4,a4 (4). Total rook = 4+2+3+4 = 13.
        // Bishop: NE f5,g6,h7 (3), NW d5,c6,b7,a8 (4), SE f3,g2,h1 (3), SW d3,c2,b1 (3) = 13.
        // Queen total = 13 + 13 = 26.
        let queen_moves = count_moves_from(&moves, Square::E4);
        assert_eq!(
            queen_moves, 26,
            "Queen on e4 should have 26 moves, got {}",
            queen_moves
        );
    }

    // ---- King moves ---------------------------------------------------------

    #[test]
    fn king_on_e4_has_8_moves_on_empty_board() {
        // King on e4 with only the black king far away.
        let board = Board::from_fen("4k3/8/8/8/4K3/8/8/8 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        let king_moves = count_moves_from(&moves, Square::E4);
        assert_eq!(
            king_moves, 8,
            "King on e4 should have 8 moves, got {}",
            king_moves
        );
    }

    // ---- Kingside castling --------------------------------------------------

    #[test]
    fn kingside_castling_generated_when_path_clear() {
        // White king on e1, rook on h1, kingside castling right, f1+g1 empty.
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        assert!(
            has_move_with_flags(&moves, Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE),
            "Should generate white kingside castling"
        );
    }

    #[test]
    fn black_kingside_castling() {
        let board = Board::from_fen("4k2r/8/8/8/8/8/8/4K3 b k - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        assert!(
            has_move_with_flags(&moves, Square::E8, Square::G8, MoveFlags::KINGSIDE_CASTLE),
            "Should generate black kingside castling"
        );
    }

    // ---- Queenside castling -------------------------------------------------

    #[test]
    fn queenside_castling_generated_when_path_clear() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        assert!(
            has_move_with_flags(&moves, Square::E1, Square::C1, MoveFlags::QUEENSIDE_CASTLE),
            "Should generate white queenside castling"
        );
    }

    #[test]
    fn black_queenside_castling() {
        let board = Board::from_fen("r3k3/8/8/8/8/8/8/4K3 b q - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        assert!(
            has_move_with_flags(&moves, Square::E8, Square::C8, MoveFlags::QUEENSIDE_CASTLE),
            "Should generate black queenside castling"
        );
    }

    // ---- No castling when blocked -------------------------------------------

    #[test]
    fn no_kingside_castling_when_f1_blocked() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4KB1R w K - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        let castle = moves.iter().find(|m| {
            m.from == Square::E1 && m.to == Square::G1 && m.flags == MoveFlags::KINGSIDE_CASTLE
        });
        assert!(
            castle.is_none(),
            "Should NOT generate kingside castling when f1 is blocked"
        );
    }

    #[test]
    fn no_kingside_castling_when_g1_blocked() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K1NR w K - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        let castle = moves.iter().find(|m| {
            m.from == Square::E1 && m.to == Square::G1 && m.flags == MoveFlags::KINGSIDE_CASTLE
        });
        assert!(
            castle.is_none(),
            "Should NOT generate kingside castling when g1 is blocked"
        );
    }

    #[test]
    fn no_queenside_castling_when_d1_blocked() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/R2QK3 w Q - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        let castle = moves.iter().find(|m| {
            m.from == Square::E1
                && m.to == Square::C1
                && m.flags == MoveFlags::QUEENSIDE_CASTLE
        });
        assert!(
            castle.is_none(),
            "Should NOT generate queenside castling when d1 is blocked"
        );
    }

    #[test]
    fn no_queenside_castling_when_b1_blocked() {
        let board = Board::from_fen("4k3/8/8/8/8/8/8/RN2K3 w Q - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        let castle = moves.iter().find(|m| {
            m.from == Square::E1
                && m.to == Square::C1
                && m.flags == MoveFlags::QUEENSIDE_CASTLE
        });
        assert!(
            castle.is_none(),
            "Should NOT generate queenside castling when b1 is blocked"
        );
    }

    // ---- No castling without rights -----------------------------------------

    #[test]
    fn no_castling_without_rights_even_if_path_clear() {
        // King on e1, rooks on a1 and h1, but NO castling rights.
        let board = Board::from_fen("4k3/8/8/8/8/8/8/R3K2R w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        let castles: Vec<_> = moves.iter().filter(|m| m.flags.is_castle()).collect();
        assert!(
            castles.is_empty(),
            "Should NOT generate any castling without rights"
        );
    }

    // ---- Position 3 from plan -----------------------------------------------

    #[test]
    fn position_3_move_count() {
        // 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -
        // This position should have a specific number of pseudo-legal moves.
        let board = Board::from_fen(fen::POSITION_3_FEN).expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // Let's count manually:
        // White pieces: King on a5, Pawn on b5, Rook on b4, Pawn on e2, Pawn on g2
        //
        // King on a5: king_attacks(a5) & !own = {a4, b4, a6, b6, b5} & !{a5, b5, b4, e2, g2}
        //   king_attacks(a5) = a4, b4, a6, b6, b5
        //   own pieces at: b5 (pawn), b4 (rook)
        //   So king targets: a4, a6, b6 = 3 moves
        //
        // Pawn on b5: single push to b6 (if empty - b6 is empty, yes), no double push (not on rank 2).
        //   Capture: pawn_attacks(White, b5) = {a6, c6}, enemy on c6? No enemy there.
        //   Just b5->b6 = 1 move
        //   Wait, actually: c6 has no piece, a6 has no piece. So just push.
        //   But wait, there's a black pawn on c7. pawn_attacks(White, b5) hits a6 and c6.
        //   Neither a6 nor c6 has an enemy piece. So just b6 push.
        //   b5 -> b6: 1 move
        //
        // Rook on b4: rook_attacks(b4, all_pieces). All pieces are at:
        //   a5, b5, b4, e2, g2 (white), c7, d6, h5, f4, h4 (black)
        //   North ray from b4: b5 (own piece, blocked) = 0 targets
        //   South ray: b3, b2, b1 = 3
        //   East ray: c4, d4, e4, f4 (enemy, capture and stop) = 4
        //   West ray: a4 = 1
        //   Total: 0 + 3 + 4 + 1 = 8 moves
        //
        // Pawn on e2: single push e3 (empty: yes), double push e4 (e3 empty, e4 empty: yes)
        //   Captures: pawn_attacks(White, e2) = {d3, f3}. No enemies there.
        //   e2->e3, e2->e4 = 2 moves
        //
        // Pawn on g2: single push g3 (empty: yes), double push g4 (g3 empty, g4 empty: yes)
        //   Captures: pawn_attacks(White, g2) = {f3, h3}. No enemies there.
        //   g2->g3, g2->g4 = 2 moves
        //
        // Total: 3 + 1 + 8 + 2 + 2 = 16
        assert_eq!(
            moves.len(),
            16,
            "Position 3 should have 16 pseudo-legal moves, got {}.\nMoves: {:?}",
            moves.len(),
            moves
        );
    }

    // ---- Black pawn moves ---------------------------------------------------

    #[test]
    fn black_pawn_pushes_south() {
        // Black pawn on e7 can push to e6 and double push to e5.
        let board = Board::from_fen("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        assert!(
            has_move(&moves, Square::E7, Square::E6),
            "Black pawn on e7 should push to e6"
        );
        assert!(
            has_move_with_flags(&moves, Square::E7, Square::E5, MoveFlags::DOUBLE_PAWN_PUSH),
            "Black pawn on e7 should double push to e5"
        );
    }

    // ---- Both castling directions -------------------------------------------

    #[test]
    fn both_castling_directions_possible() {
        let board =
            Board::from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1").expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        assert!(has_move_with_flags(
            &moves,
            Square::E1,
            Square::G1,
            MoveFlags::KINGSIDE_CASTLE
        ));
        assert!(has_move_with_flags(
            &moves,
            Square::E1,
            Square::C1,
            MoveFlags::QUEENSIDE_CASTLE
        ));
    }

    // ---- Move list does not contain moves for the wrong side ----------------

    #[test]
    fn only_generates_moves_for_side_to_move() {
        let board = Board::starting_position();
        let moves = generate_pseudo_legal_moves(&board);

        // In the starting position with White to move, no move should originate
        // from a rank 7 or 8 square (those are Black's pieces).
        for mv in &moves {
            assert!(
                mv.from.rank() < 2,
                "White to move: move from {} should not originate from Black's side",
                mv.from
            );
        }
    }

    // ---- Regression: pawn on A or H file captures only one direction --------

    #[test]
    fn pawn_on_a_file_captures_only_right() {
        // White pawn on a5 with black pawns on b6.
        let board = Board::from_fen("4k3/8/1p6/P7/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // Pawn on a5 can capture b6
        assert!(has_move(&moves, Square::A5, Square::B6));

        // But should NOT have a capture to the left (there's nothing there)
        // Verify no move from a5 goes to a square that doesn't exist
        let a5_moves: Vec<_> = moves
            .iter()
            .filter(|m| m.from == Square::A5)
            .collect();
        for mv in &a5_moves {
            assert!(mv.to.file() <= 1, "a-file pawn should only move right or straight");
        }
    }

    #[test]
    fn pawn_on_h_file_captures_only_left() {
        // White pawn on h5 with black pawn on g6.
        let board = Board::from_fen("4k3/8/6p1/7P/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        assert!(has_move(&moves, Square::H5, Square::G6));
    }

    // ---- Multiple queens on board -------------------------------------------

    #[test]
    fn multiple_queens_generate_moves() {
        // Two white queens.
        let board = Board::from_fen("4k3/8/8/8/3Q4/8/8/3QK3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // Both queens should generate moves
        assert!(count_moves_from(&moves, Square::D4) > 0);
        assert!(count_moves_from(&moves, Square::D1) > 0);
    }

    // ---- Pawn capture and push to promotion rank ----------------------------

    #[test]
    fn multiple_pawns_promote_simultaneously() {
        // Two white pawns on 7th rank.
        let board = Board::from_fen("4k3/3PP3/8/8/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_pseudo_legal_moves(&board);

        // d7->d8: 4 promotions, e7->e8: 4 promotions = 8 promotion moves
        let promo_count = moves.iter().filter(|m| m.is_promotion()).count();
        assert_eq!(
            promo_count, 8,
            "Two pawns on 7th rank should generate 8 promotion moves, got {}",
            promo_count
        );
    }

    // ========================================================================
    // Legal move generation tests
    // ========================================================================

    #[test]
    fn legal_moves_starting_position_has_20_moves() {
        // In the starting position, all 20 pseudo-legal moves are legal
        // (no move exposes the king to check).
        let board = Board::starting_position();
        let moves = generate_legal_moves(&board);
        assert_eq!(
            moves.len(),
            20,
            "Starting position should have exactly 20 legal moves, got {}",
            moves.len()
        );
    }

    #[test]
    fn legal_moves_kiwipete_has_48_moves() {
        // Kiwipete: all 48 pseudo-legal moves are legal.
        let board = Board::from_fen(fen::KIWIPETE_FEN).expect("valid FEN");
        let moves = generate_legal_moves(&board);
        assert_eq!(
            moves.len(),
            48,
            "Kiwipete should have 48 legal moves, got {}",
            moves.len()
        );
    }

    #[test]
    fn legal_moves_check_evasion() {
        // Black king on e8 in check by white rook on e1. Black must deal with the check.
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4R1K1 b - - 0 1")
            .expect("valid FEN");
        let moves = generate_legal_moves(&board);

        // The black king can escape: d8, f8, d7, f7. But not e7 (still on the e-file).
        // Let's verify all moves are king moves (the only piece black has).
        for mv in &moves {
            assert_eq!(mv.from, Square::E8, "Only the king can move");
        }

        // Count: d8, d7, f8, f7 = 4 legal moves
        assert_eq!(
            moves.len(),
            4,
            "Black king in check by rook on e1 should have 4 legal moves, got {}.\nMoves: {:?}",
            moves.len(),
            moves
        );

        // Verify specific squares
        assert!(has_move(&moves, Square::E8, Square::D8));
        assert!(has_move(&moves, Square::E8, Square::D7));
        assert!(has_move(&moves, Square::E8, Square::F8));
        assert!(has_move(&moves, Square::E8, Square::F7));

        // King cannot stay on e-file (still in check)
        assert!(!has_move(&moves, Square::E8, Square::E7));
    }

    #[test]
    fn cannot_castle_out_of_check() {
        // White king on e1 in check by black rook on e8. Castling should not be available.
        let board = Board::from_fen("4r3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
            .expect("valid FEN");
        let moves = generate_legal_moves(&board);

        // No castling moves should be generated
        let castles: Vec<_> = moves.iter().filter(|m| m.flags.is_castle()).collect();
        assert!(
            castles.is_empty(),
            "Cannot castle while in check, but found {} castling moves",
            castles.len()
        );
    }

    #[test]
    fn cannot_castle_through_check_kingside() {
        // White king on e1, black rook on f8 attacks f1 -- cannot castle kingside
        // because the king would pass through f1 which is attacked.
        let board = Board::from_fen("5r1k/8/8/8/8/8/8/R3K2R w KQ - 0 1")
            .expect("valid FEN");
        let moves = generate_legal_moves(&board);

        // Kingside castling should not be available (king passes through f1)
        assert!(
            !has_move_with_flags(&moves, Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE),
            "Cannot castle kingside when f1 is attacked"
        );
        // Queenside castling should still be available (d1 is not attacked)
        assert!(
            has_move_with_flags(&moves, Square::E1, Square::C1, MoveFlags::QUEENSIDE_CASTLE),
            "Queenside castling should still be legal"
        );
    }

    #[test]
    fn cannot_castle_through_check_queenside() {
        // White king on e1, black rook on d8 attacks d1 -- cannot castle queenside
        // because the king would pass through d1 which is attacked.
        let board = Board::from_fen("3r3k/8/8/8/8/8/8/R3K2R w KQ - 0 1")
            .expect("valid FEN");
        let moves = generate_legal_moves(&board);

        // Queenside castling should not be available (king passes through d1)
        assert!(
            !has_move_with_flags(&moves, Square::E1, Square::C1, MoveFlags::QUEENSIDE_CASTLE),
            "Cannot castle queenside when d1 is attacked"
        );
        // Kingside should still be legal
        assert!(
            has_move_with_flags(&moves, Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE),
            "Kingside castling should still be legal"
        );
    }

    #[test]
    fn cannot_castle_into_check() {
        // White king on e1, black rook on g8 attacks g1 -- cannot castle kingside
        // because the king would land on g1 which is attacked.
        // (This is caught by the general "own king in check after move" filter.)
        let board = Board::from_fen("6rk/8/8/8/8/8/8/R3K2R w KQ - 0 1")
            .expect("valid FEN");
        let moves = generate_legal_moves(&board);

        assert!(
            !has_move_with_flags(&moves, Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE),
            "Cannot castle kingside when g1 is attacked (king lands in check)"
        );
    }

    #[test]
    fn pinned_piece_cannot_move_off_pin_line() {
        // White bishop on e2 is pinned by a black rook on e8 (white king on e1).
        // The bishop cannot move because that would expose the king to the rook.
        let board = Board::from_fen("4r2k/8/8/8/8/8/4B3/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_legal_moves(&board);

        // The bishop on e2 should have NO legal moves (it's absolutely pinned).
        let bishop_moves = count_moves_from(&moves, Square::E2);
        assert_eq!(
            bishop_moves, 0,
            "Pinned bishop on e2 should have 0 legal moves, got {}",
            bishop_moves
        );
    }

    #[test]
    fn pinned_rook_can_move_along_pin_line() {
        // White rook on e4 is pinned by a black rook on e8 (white king on e1).
        // The rook CAN move along the pin line (e2, e3, e5, e6, e7, e8-capture).
        let board = Board::from_fen("4r2k/8/8/8/4R3/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_legal_moves(&board);

        let rook_moves = count_moves_from(&moves, Square::E4);
        // Along the pin line: e2, e3, e5, e6, e7, e8 (capture) = 6 moves
        assert_eq!(
            rook_moves, 6,
            "Pinned rook on e4 should have 6 legal moves along the pin line, got {}",
            rook_moves
        );
    }

    #[test]
    fn legal_moves_position_3() {
        // Position 3: 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -
        // Known perft depth 1 = 14 legal moves.
        let board = Board::from_fen(fen::POSITION_3_FEN).expect("valid FEN");
        let moves = generate_legal_moves(&board);
        assert_eq!(
            moves.len(),
            14,
            "Position 3 should have 14 legal moves, got {}.\nMoves: {:?}",
            moves.len(),
            moves
        );
    }

    #[test]
    fn legal_moves_position_4() {
        // Position 4: r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1
        // Known perft depth 1 = 6 legal moves.
        let board = Board::from_fen(fen::POSITION_4_FEN).expect("valid FEN");
        let moves = generate_legal_moves(&board);
        assert_eq!(
            moves.len(),
            6,
            "Position 4 should have 6 legal moves, got {}.\nMoves: {:?}",
            moves.len(),
            moves
        );
    }

    #[test]
    fn en_passant_discovered_check() {
        // A position where en passant would leave the own king in check via a
        // discovered attack. The en passant capture should be illegal.
        //
        // White king on a5, white pawn on b5, black pawn on c5 (just double-pushed),
        // black rook on h5. If white captures en passant (b5xc6), both the white pawn
        // on b5 and the captured black pawn on c5 are removed from rank 5, exposing
        // the white king to the black rook.
        let board = Board::from_fen("4k3/8/8/K1pP3r/8/8/8/8 w - c6 0 1")
            .expect("valid FEN");
        let moves = generate_legal_moves(&board);

        // The en passant capture d5xc6 should NOT be in the legal moves because
        // it would expose the white king on a5 to the black rook on h5.
        // However, note: the d5 pawn would remain and block. Let me reconsider.
        //
        // Actually: white pawn is on d5, black pawn on c5. EP capture d5xc6 removes
        // the black pawn from c5 and moves white pawn from d5 to c6. That leaves
        // rank 5 with just K on a5 and r on h5. But wait, d5 is vacated and c5 is
        // vacated. The rook on h5 has a clear line to a5. So the EP is illegal.
        let ep_moves: Vec<_> = moves.iter().filter(|m| m.is_en_passant()).collect();
        assert!(
            ep_moves.is_empty(),
            "En passant should be illegal due to discovered check, but found {:?}",
            ep_moves
        );
    }

    #[test]
    fn checkmate_position_has_zero_legal_moves() {
        // Scholar's mate final position: black is in checkmate, no legal moves.
        let board = Board::from_fen("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
            .expect("valid FEN");
        let moves = generate_legal_moves(&board);
        assert_eq!(
            moves.len(),
            0,
            "Checkmate position should have 0 legal moves, got {}.\nMoves: {:?}",
            moves.len(),
            moves
        );
    }

    #[test]
    fn stalemate_position_has_zero_legal_moves() {
        // King on a1, opponent queen on b3, opponent king on c1 -- white to move, stalemate.
        // Actually let me use a well-known stalemate: King h1, Queen g3, King f2 (not check, no moves).
        // Wait, I need to be careful. Let me use: White king on a1, no other white pieces,
        // Black queen on b3, black king on c2.  White king on a1: can go to a2 (but b3 queen
        // attacks a2), b1 (queen on b3 attacks b1 diagonally? b3 to b1: same file, yes).
        // Actually queen on b3 attacks: b-file, 3rd rank, diagonals. b1 is on the b-file.
        // a2 is on the a2-e6 diagonal but also the queen attacks a2 from b3? Yes, b3 to a2 is diagonal.
        // So a1 king: a2 attacked, b1 attacked, b2 attacked. No legal moves = stalemate.
        let board = Board::from_fen("8/8/8/8/8/1q6/8/K1k5 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_legal_moves(&board);
        // Verify the king is NOT in check (otherwise it's checkmate, not stalemate)
        assert!(
            !crate::attacks::is_in_check(&board, Color::White),
            "This should be stalemate, not check"
        );
        assert_eq!(
            moves.len(),
            0,
            "Stalemate position should have 0 legal moves, got {}.\nMoves: {:?}",
            moves.len(),
            moves
        );
    }

    #[test]
    fn legal_moves_block_check_with_interposition() {
        // White king on e1 in check by black rook on e8. White has a bishop on c3
        // that can block by going to e5 (interposing between the rook and king).
        // Wait, bishop on c3 going to e5 is diagonal, that works.
        // Actually c3 to e5 is a valid bishop move (+2,+2 = diagonal).
        let board = Board::from_fen("4r2k/8/8/8/8/2B5/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let moves = generate_legal_moves(&board);

        // The bishop on c3 should be able to interpose on e5 (or other squares on the e-file
        // diagonal). Let me figure out which squares the bishop can reach that are on the
        // e-file between e1 and e8:
        // Bishop on c3 diagonals: c3-d4-e5-f6-g7-h8 and c3-b4-a5 and c3-d2-e1 and c3-b2-a1.
        // Squares on the e-file between e2 and e7: e2,e3,e4,e5,e6,e7.
        // Bishop can reach e5 (blocks), and e1 is the king square.
        // Also bishop can reach d2 which doesn't help.
        // So bishop has one blocking move: c3->e5.
        // King moves: d1, d2, f1, f2 (not e2 -- still on e-file and rook on e8 attacks it).
        // Actually d2: is d2 attacked? Rook on e8 doesn't attack d2. King attacks from e1
        // include d1, d2, f1, f2 (not e2 since rook attacks it).
        // But wait, we also need to check if the king destinations are attacked by the rook.
        // Rook on e8: attacks the entire e-file and 8th rank. So e2 is attacked.
        // d1, d2, f1, f2 are not attacked by the rook (different file and rank).
        // So king has 4 moves + bishop has 1 blocking move = 5 total legal moves.
        // But bishop might have other moves too... bishop on c3 can go to:
        // d4, e5 (blocks check!), f6, g7, h8 (but h8 doesn't block, rook is on e8 not h8...
        // wait, the black king is on h8 -- so bishop can't go to h8 which is occupied).
        // Actually f6 -- does it block the check? No, f6 is not on the e-file.
        // g7: doesn't block. d2: doesn't block. b4: doesn't block. a5: doesn't block.
        // b2: doesn't block. a1: doesn't block. d4: doesn't block (not on e-file).
        // Only e5 blocks the rook check along the e-file.
        // But wait -- the bishop can capture the rook! c3 to e5 to g7... no, that's two moves.
        // Can the bishop reach e8? c3 is on a light square (c3: file=2, rank=2, sum=4, even=light).
        // e8: file=4, rank=7, sum=11, odd=dark. Different color squares! Bishop can't reach e8.
        // So only 1 bishop move (e5) + 4 king moves = 5 total.
        assert_eq!(
            moves.len(),
            5,
            "Should have 5 legal moves (4 king + 1 bishop interposition), got {}.\nMoves: {:?}",
            moves.len(),
            moves
        );
        assert!(
            has_move(&moves, Square::C3, Square::E5),
            "Bishop should be able to interpose on e5"
        );
    }

    #[test]
    fn legal_moves_position_5() {
        // Position 5: rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8
        // Known perft depth 1 = 44 legal moves.
        let board = Board::from_fen(fen::POSITION_5_FEN).expect("valid FEN");
        let moves = generate_legal_moves(&board);
        assert_eq!(
            moves.len(),
            44,
            "Position 5 should have 44 legal moves, got {}.\nMoves: {:?}",
            moves.len(),
            moves
        );
    }

    #[test]
    fn legal_moves_position_6() {
        // Position 6: r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/3P1N1P/PPP1NPP1/R2Q1RK1 w - - 0 10
        // Known perft depth 1 = 46 legal moves.
        let board = Board::from_fen(fen::POSITION_6_FEN).expect("valid FEN");
        let moves = generate_legal_moves(&board);
        assert_eq!(
            moves.len(),
            46,
            "Position 6 should have 46 legal moves, got {}.\nMoves: {:?}",
            moves.len(),
            moves
        );
    }
}
