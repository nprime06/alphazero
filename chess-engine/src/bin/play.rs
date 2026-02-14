//! Interactive chess CLI for testing the engine.
//!
//! Play chess using standard algebraic notation (SAN):
//!   - Pawn moves: `e4`, `d5`, `exd5`, `e8=Q`
//!   - Piece moves: `Nf3`, `Bxc6`, `Rdf1`, `R1a3`
//!   - Castling: `O-O`, `O-O-O`
//!   - Check/checkmate indicators (`+`, `#`) are accepted but ignored
//!
//! Commands:
//!   moves     — list all legal moves
//!   fen       — print current FEN
//!   fen <FEN> — load a position from FEN
//!   board     — redisplay the board
//!   undo      — take back the last move
//!   new       — start a new game
//!   quit      — exit
//!
//! Run with: `cargo run -p chess-engine --bin play`

use chess_engine::board::Board;
use chess_engine::game::{Game, GameResult};
use chess_engine::moves::{Move, MoveFlags};
use chess_engine::types::{Color, Piece, Square};
use chess_engine::attacks::is_in_check;

use std::io::{self, BufRead, Write};

// =============================================================================
// SAN parsing
// =============================================================================

/// Converts a SAN string (e.g. "Nf3", "exd5", "O-O") into a Move by matching
/// against the list of legal moves in the current position.
fn parse_san(san: &str, game: &Game) -> Result<Move, String> {
    let legal = game.legal_moves();
    if legal.is_empty() {
        return Err("No legal moves in this position".into());
    }

    // Strip check/checkmate indicators and whitespace
    let san = san.trim().trim_end_matches('#').trim_end_matches('+');

    // Castling
    if san == "O-O" || san == "0-0" {
        let castle_mv = legal.iter().find(|m| m.flags == MoveFlags::KINGSIDE_CASTLE);
        return castle_mv.copied().ok_or_else(|| "Kingside castling is not legal here".into());
    }
    if san == "O-O-O" || san == "0-0-0" {
        let castle_mv = legal.iter().find(|m| m.flags == MoveFlags::QUEENSIDE_CASTLE);
        return castle_mv.copied().ok_or_else(|| "Queenside castling is not legal here".into());
    }

    let bytes = san.as_bytes();
    if bytes.len() < 2 {
        return Err(format!("Move too short: '{san}'"));
    }

    // Determine piece type (uppercase letter at start, or pawn if lowercase/none)
    let (piece, rest) = if bytes[0] >= b'A' && bytes[0] <= b'Z' && bytes[0] != b'O' {
        let p = match bytes[0] {
            b'N' => Piece::Knight,
            b'B' => Piece::Bishop,
            b'R' => Piece::Rook,
            b'Q' => Piece::Queen,
            b'K' => Piece::King,
            _ => return Err(format!("Unknown piece letter: '{}'", bytes[0] as char)),
        };
        (p, &san[1..])
    } else {
        (Piece::Pawn, san)
    };

    // Parse the rest: possible disambiguation, 'x' for capture, destination, '=' promotion
    // Examples: "f3", "xc6", "df1", "1a3", "xe5", "8=Q", "exd5"
    let rest = rest.replace('x', ""); // strip capture indicator

    // Extract promotion if present
    let (rest, promotion) = if rest.contains('=') {
        let parts: Vec<&str> = rest.splitn(2, '=').collect();
        let promo_piece = match parts[1].chars().next() {
            Some('Q') | Some('q') => Piece::Queen,
            Some('R') | Some('r') => Piece::Rook,
            Some('B') | Some('b') => Piece::Bishop,
            Some('N') | Some('n') => Piece::Knight,
            _ => return Err(format!("Invalid promotion piece in '{san}'")),
        };
        (parts[0].to_string(), Some(promo_piece))
    } else {
        (rest.to_string(), None)
    };

    let rest_bytes = rest.as_bytes();

    // The last two characters must be the destination square
    if rest_bytes.len() < 2 {
        return Err(format!("Cannot parse destination in '{san}'"));
    }

    let dest_str = &rest[rest.len() - 2..];
    let dest = Square::from_algebraic(dest_str)
        .ok_or_else(|| format!("Invalid destination square: '{dest_str}'"))?;

    // Everything before the destination is disambiguation (0, 1, or 2 chars)
    let disambig = &rest[..rest.len() - 2];

    let disambig_file: Option<u8> = disambig.chars().find(|c| c.is_ascii_lowercase())
        .map(|c| c as u8 - b'a');
    let disambig_rank: Option<u8> = disambig.chars().find(|c| c.is_ascii_digit())
        .map(|c| c as u8 - b'1');

    // Find matching legal moves
    let board = game.board();
    let candidates: Vec<Move> = legal
        .iter()
        .filter(|m| {
            // Must go to the right destination
            if m.to != dest {
                return false;
            }
            // Must be the right piece type
            let moving_piece = board.piece_at(m.from);
            match moving_piece {
                Some((_, p)) if p == piece => {}
                _ => return false,
            }
            // Must match promotion
            if m.promotion != promotion {
                return false;
            }
            // Disambiguation: file
            if let Some(f) = disambig_file {
                if m.from.file() != f {
                    return false;
                }
            }
            // Disambiguation: rank
            if let Some(r) = disambig_rank {
                if m.from.rank() != r {
                    return false;
                }
            }
            true
        })
        .copied()
        .collect();

    match candidates.len() {
        0 => Err(format!("No legal move matches '{san}'")),
        1 => Ok(candidates[0]),
        _ => Err(format!(
            "Ambiguous move '{san}': matches {}",
            candidates.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Converts a Move to SAN notation given the current position.
fn move_to_san(mv: Move, game: &Game) -> String {
    let board = game.board();

    // Castling
    if mv.flags == MoveFlags::KINGSIDE_CASTLE {
        return "O-O".to_string();
    }
    if mv.flags == MoveFlags::QUEENSIDE_CASTLE {
        return "O-O-O".to_string();
    }

    let (_, piece) = board.piece_at(mv.from).unwrap();
    let is_capture = board.piece_at(mv.to).is_some() || mv.is_en_passant();

    let mut san = String::new();

    if piece == Piece::Pawn {
        // Pawn moves: capture uses file prefix
        if is_capture {
            san.push(mv.from.file_char());
            san.push('x');
        }
        san.push_str(&mv.to.to_algebraic());
        // Promotion
        if let Some(promo) = mv.promotion {
            san.push('=');
            san.push(promo.char());
        }
    } else {
        // Piece letter
        san.push(piece.char());

        // Disambiguation: check if other pieces of the same type can reach the same square
        let legal = game.legal_moves();
        let ambiguous: Vec<&Move> = legal
            .iter()
            .filter(|m| {
                m.to == mv.to
                    && m.from != mv.from
                    && board.piece_at(m.from).map(|(_, p)| p) == Some(piece)
            })
            .collect();

        if !ambiguous.is_empty() {
            let same_file = ambiguous.iter().any(|m| m.from.file() == mv.from.file());
            let same_rank = ambiguous.iter().any(|m| m.from.rank() == mv.from.rank());

            if !same_file {
                san.push(mv.from.file_char());
            } else if !same_rank {
                san.push(mv.from.rank_char());
            } else {
                san.push(mv.from.file_char());
                san.push(mv.from.rank_char());
            }
        }

        if is_capture {
            san.push('x');
        }
        san.push_str(&mv.to.to_algebraic());
    }

    // Check/checkmate indicator
    // We need to look ahead: make the move, see if the opponent is in check
    let mut game_clone = Game::from_board(board.clone());
    game_clone.make_move(mv);
    let opponent = board.side_to_move().flip();
    if is_in_check(game_clone.board(), opponent) {
        let opp_legal = game_clone.legal_moves();
        if opp_legal.is_empty() {
            san.push('#');
        } else {
            san.push('+');
        }
    }

    san
}

// =============================================================================
// Display helpers
// =============================================================================

fn print_board(game: &Game) {
    let board = game.board();
    println!();
    for rank in (0..8u8).rev() {
        print!("  {} |", rank + 1);
        for file in 0..8u8 {
            let square = Square::from_file_rank(file, rank);
            let c = match board.piece_at(square) {
                None => '.',
                Some((Color::White, piece)) => piece.char(),
                Some((Color::Black, piece)) => piece.char().to_ascii_lowercase(),
            };
            print!(" {c}");
        }
        println!();
    }
    println!("      a b c d e f g h");
    println!();

    let stm = board.side_to_move();
    let in_check = is_in_check(board, stm);
    print!("  {stm} to move");
    if in_check {
        print!(" (CHECK)");
    }
    println!();
}

fn print_result(result: GameResult) {
    match result {
        GameResult::WhiteWins => println!("\n  *** CHECKMATE — White wins! ***\n"),
        GameResult::BlackWins => println!("\n  *** CHECKMATE — Black wins! ***\n"),
        GameResult::DrawStalemate => println!("\n  *** STALEMATE — Draw! ***\n"),
        GameResult::DrawRepetition => println!("\n  *** THREEFOLD REPETITION — Draw! ***\n"),
        GameResult::DrawFiftyMoveRule => println!("\n  *** 50-MOVE RULE — Draw! ***\n"),
        GameResult::DrawInsufficientMaterial => println!("\n  *** INSUFFICIENT MATERIAL — Draw! ***\n"),
        GameResult::Ongoing => {}
    }
}

fn print_legal_moves(game: &Game) {
    let moves = game.legal_moves();
    if moves.is_empty() {
        println!("  No legal moves.");
        return;
    }
    let san_moves: Vec<String> = moves.iter().map(|m| move_to_san(*m, game)).collect();
    println!("  {} legal moves:", san_moves.len());
    // Print in rows of 10
    for chunk in san_moves.chunks(10) {
        println!("    {}", chunk.join("  "));
    }
}

// =============================================================================
// Main loop
// =============================================================================

fn main() {
    println!("=== AlphaZero Chess Engine — Interactive Play ===");
    println!("Type moves in SAN (e.g. e4, Nf3, O-O) or a command:");
    println!("  moves, fen, fen <FEN>, board, undo, new, quit");
    println!();

    let mut game = Game::new();
    let mut move_history: Vec<(Move, chess_engine::makemove::UndoInfo)> = Vec::new();
    let mut move_number: usize = 1;

    print_board(&game);

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Prompt
        let stm = game.board().side_to_move();
        let num_display = if stm == Color::White {
            format!("{}.", move_number)
        } else {
            format!("{}...", move_number)
        };
        print!("\n  {num_display} ");
        stdout.flush().unwrap();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            }
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "quit" | "exit" | "q" => break,

            "new" => {
                game = Game::new();
                move_history.clear();
                move_number = 1;
                println!("\n  New game started.");
                print_board(&game);
                continue;
            }

            "board" | "b" => {
                print_board(&game);
                continue;
            }

            "moves" | "m" => {
                print_legal_moves(&game);
                continue;
            }

            "fen" => {
                println!("  {}", game.board().to_fen());
                continue;
            }

            "undo" | "u" => {
                if let Some((mv, undo)) = move_history.pop() {
                    game.unmake_move(mv, &undo);
                    // Adjust move number
                    if game.board().side_to_move() == Color::White {
                        move_number = move_number.saturating_sub(1);
                    }
                    println!("  Undid {}", mv.to_uci());
                    print_board(&game);
                } else {
                    println!("  Nothing to undo.");
                }
                continue;
            }

            _ => {}
        }

        // Check for "fen <FEN>" command
        if input.starts_with("fen ") {
            let fen_str = &input[4..].trim();
            match Board::from_fen(fen_str) {
                Ok(board) => {
                    move_number = 1;
                    move_history.clear();
                    game = Game::from_board(board);
                    println!("  Position loaded.");
                    print_board(&game);
                }
                Err(e) => println!("  Invalid FEN: {e}"),
            }
            continue;
        }

        // Check if game is already over
        let result = game.result();
        if result != GameResult::Ongoing {
            print_result(result);
            println!("  Game is over. Type 'new' to start a new game or 'fen <FEN>' to load a position.");
            continue;
        }

        // Try to parse as SAN move
        match parse_san(input, &game) {
            Ok(mv) => {
                let san = move_to_san(mv, &game);
                let was_white = game.board().side_to_move() == Color::White;
                let undo = game.make_move(mv);
                move_history.push((mv, undo));

                if was_white {
                    print!("  {}. {san}", move_number);
                } else {
                    print!("  {}... {san}", move_number);
                    move_number += 1;
                }
                println!();

                print_board(&game);

                let result = game.result();
                if result != GameResult::Ongoing {
                    print_result(result);
                }
            }
            Err(e) => {
                println!("  {e}");
                println!("  Type 'moves' to see legal moves.");
            }
        }
    }

    println!("\n  Goodbye!");
}
