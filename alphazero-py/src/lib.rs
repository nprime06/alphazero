//! Python bindings for the AlphaZero chess engine and MCTS.
//!
//! Provides PyO3 wrappers around the `chess-engine` and `mcts` crates,
//! exposing them as a native Python module via `alphazero_py`.
//!
//! # Classes
//!
//! - [`PyBoard`] -- Chess board/game wrapper with python-chess-like API.
//! - [`PySearchResult`] -- MCTS search results (move visits, value).
//!
//! # Module functions
//!
//! - [`search`] -- MCTS with uniform evaluator.
//! - [`search_with_model`] -- MCTS with neural network evaluator.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use chess_engine::attacks::is_in_check;
use chess_engine::board::Board;
use chess_engine::game::{Game, GameResult};
use chess_engine::makemove::UndoInfo;
use chess_engine::moves::Move;
use chess_engine::types::{Color, Piece, Square};

use mcts::config::MctsConfig;
use mcts::nn::NnModel;
use mcts::search;

// =============================================================================
// UCI parsing helper
// =============================================================================

/// Parse a UCI move string (e.g. "e2e4", "e7e8q") by matching it against the
/// legal moves in the current game position.
///
/// This approach is robust because it handles all special move flags (castling,
/// en passant, double pawn push) automatically -- we just need to match the
/// from/to squares and optional promotion piece.
fn parse_uci(uci: &str, game: &Game) -> Result<Move, String> {
    let uci = uci.trim().to_lowercase();
    let bytes = uci.as_bytes();

    if bytes.len() < 4 || bytes.len() > 5 {
        return Err(format!(
            "Invalid UCI move '{}': expected 4 or 5 characters (e.g. 'e2e4', 'e7e8q')",
            uci
        ));
    }

    let from = Square::from_algebraic(&uci[0..2])
        .ok_or_else(|| format!("Invalid source square: '{}'", &uci[0..2]))?;
    let to = Square::from_algebraic(&uci[2..4])
        .ok_or_else(|| format!("Invalid destination square: '{}'", &uci[2..4]))?;

    let promotion = if bytes.len() == 5 {
        Some(
            Piece::from_char(bytes[4] as char)
                .ok_or_else(|| format!("Invalid promotion piece: '{}'", bytes[4] as char))?,
        )
    } else {
        None
    };

    // Find the matching legal move. We match on from, to, and promotion,
    // which uniquely identifies a move (flags like castling/en passant are
    // determined by the position, not the UCI string).
    let legal_moves = game.legal_moves();
    let matching: Vec<Move> = legal_moves
        .iter()
        .filter(|mv| {
            mv.from == from && mv.to == to && mv.promotion == promotion
        })
        .copied()
        .collect();

    match matching.len() {
        0 => Err(format!("Illegal move: '{}'", uci)),
        1 => Ok(matching[0]),
        _ => {
            // This should not happen in a legal chess position (from+to+promotion
            // is always unique), but handle it gracefully.
            Ok(matching[0])
        }
    }
}

// =============================================================================
// Device parsing helper
// =============================================================================

/// Parse a device string into a `tch::Device`.
///
/// Supported formats:
/// - "cpu" -> `Device::Cpu`
/// - "cuda" or "cuda:0" -> `Device::Cuda(0)`
/// - "cuda:1" -> `Device::Cuda(1)`
/// - etc.
fn parse_device(device_str: &str) -> Result<tch::Device, String> {
    let s = device_str.trim().to_lowercase();
    if s == "cpu" {
        return Ok(tch::Device::Cpu);
    }
    if s == "cuda" {
        return Ok(tch::Device::Cuda(0));
    }
    if let Some(rest) = s.strip_prefix("cuda:") {
        let idx: i64 = rest
            .parse()
            .map_err(|_| format!("Invalid CUDA device index: '{}'", rest))?;
        return Ok(tch::Device::Cuda(idx as usize));
    }
    Err(format!(
        "Unknown device '{}'. Expected 'cpu', 'cuda', or 'cuda:N'",
        device_str
    ))
}

// =============================================================================
// PyBoard
// =============================================================================

/// A Python-accessible chess board/game wrapper.
///
/// Wraps a `Game` internally for proper repetition tracking, and maintains
/// a stack of `(Move, UndoInfo)` for `pop()` support.
///
/// Provides a python-chess-like API with UCI move notation.
#[pyclass(name = "Board")]
struct PyBoard {
    game: Game,
    /// Stack of (move, undo_info) for pop() support.
    history: Vec<(Move, UndoInfo)>,
}

#[pymethods]
impl PyBoard {
    /// Create a new board from the standard starting position.
    #[new]
    fn new() -> Self {
        PyBoard {
            game: Game::new(),
            history: Vec::new(),
        }
    }

    /// Create a board from a FEN string.
    #[staticmethod]
    fn from_fen(fen: &str) -> PyResult<Self> {
        let board = Board::from_fen(fen)
            .map_err(|e| PyValueError::new_err(format!("Invalid FEN: {}", e)))?;
        Ok(PyBoard {
            game: Game::from_board(board),
            history: Vec::new(),
        })
    }

    /// The current position as a FEN string.
    #[getter]
    fn fen(&self) -> String {
        self.game.board().to_fen()
    }

    /// The side to move: "white" or "black".
    #[getter]
    fn turn(&self) -> &'static str {
        match self.game.board().side_to_move() {
            Color::White => "white",
            Color::Black => "black",
        }
    }

    /// Return a list of all legal moves as UCI strings.
    fn legal_moves(&self) -> Vec<String> {
        self.game
            .legal_moves()
            .iter()
            .map(|mv| mv.to_uci())
            .collect()
    }

    /// Make a move given in UCI notation (e.g. "e2e4", "e7e8q").
    ///
    /// Raises ValueError if the move is illegal or malformed.
    fn push(&mut self, uci: &str) -> PyResult<()> {
        let mv = parse_uci(uci, &self.game)
            .map_err(|e| PyValueError::new_err(e))?;
        let undo = self.game.make_move(mv);
        self.history.push((mv, undo));
        Ok(())
    }

    /// Undo the last move.
    ///
    /// Raises ValueError if there are no moves to undo.
    fn pop(&mut self) -> PyResult<()> {
        let (mv, undo) = self
            .history
            .pop()
            .ok_or_else(|| PyValueError::new_err("No moves to undo"))?;
        self.game.unmake_move(mv, &undo);
        Ok(())
    }

    /// Return True if the game is over (checkmate, stalemate, draw).
    fn is_game_over(&self) -> bool {
        self.game.is_terminal()
    }

    /// Return the game result as a string.
    ///
    /// - "1-0" if White wins
    /// - "0-1" if Black wins
    /// - "1/2-1/2" for any draw
    /// - "*" if the game is still ongoing
    fn result(&self) -> &'static str {
        match self.game.result() {
            GameResult::WhiteWins => "1-0",
            GameResult::BlackWins => "0-1",
            GameResult::DrawStalemate
            | GameResult::DrawRepetition
            | GameResult::DrawFiftyMoveRule
            | GameResult::DrawInsufficientMaterial => "1/2-1/2",
            GameResult::Ongoing => "*",
        }
    }

    /// Return True if the side to move is in check.
    fn is_check(&self) -> bool {
        let side = self.game.board().side_to_move();
        is_in_check(self.game.board(), side)
    }

    /// Return a deep copy of this board.
    ///
    /// The copy preserves the full game history (for repetition detection)
    /// and the move stack (for pop() support).
    fn copy(&self) -> PyResult<PyBoard> {
        // Reconstruct the game by starting from the initial position
        // and replaying all moves. This preserves the full history for
        // repetition detection.
        //
        // We need to figure out the starting position. If history is empty,
        // the current board IS the starting position. Otherwise, we
        // reconstruct by undoing all moves to find the starting position,
        // then replaying.

        // Approach: clone the board, create a new Game from it, and replay
        // the move stack. For moves before our history (if the board was
        // created from FEN), repetition history starts fresh -- which matches
        // how the original PyBoard was created.
        if self.history.is_empty() {
            // No moves made -- just clone from current position.
            return Ok(PyBoard {
                game: Game::from_board(self.game.board().clone()),
                history: Vec::new(),
            });
        }

        // We need to undo all moves to find the root position, then replay.
        // But we cannot mutate self. Instead, clone the board and undo on
        // the clone.
        let mut temp_board = self.game.board().clone();
        // Undo in reverse order to get back to the starting position.
        for (mv, undo) in self.history.iter().rev() {
            temp_board.unmake_move(*mv, undo);
        }

        // Now replay all moves on a fresh Game.
        let mut new_game = Game::from_board(temp_board);
        let mut new_history = Vec::with_capacity(self.history.len());
        for &(mv, _) in &self.history {
            let undo = new_game.make_move(mv);
            new_history.push((mv, undo));
        }

        Ok(PyBoard {
            game: new_game,
            history: new_history,
        })
    }

    /// ASCII board display.
    fn __str__(&self) -> String {
        format!("{}", self.game.board())
    }

    /// Repr with FEN.
    fn __repr__(&self) -> String {
        format!("Board('{}')", self.game.board().to_fen())
    }
}

// =============================================================================
// PySearchResult
// =============================================================================

/// MCTS search result exposed to Python.
///
/// Contains the visit counts for each move, total simulations, root value,
/// and the best move (most visited).
#[pyclass(name = "SearchResult")]
struct PySearchResult {
    /// List of (uci_string, visit_count) tuples.
    #[pyo3(get)]
    moves: Vec<(String, u32)>,

    /// Total number of simulations run.
    #[pyo3(get)]
    total_simulations: u32,

    /// Root value estimate after search.
    #[pyo3(get)]
    root_value: f32,

    /// The move string with the most visits.
    #[pyo3(get)]
    best_move: Option<String>,
}

// =============================================================================
// Module-level search functions
// =============================================================================

/// Run MCTS search with uniform evaluator (no neural network).
///
/// Args:
///     board: The board position to search from.
///     num_simulations: Number of MCTS simulations (default 800).
///     temperature: Temperature for root noise and move selection (default 1.0).
///     c_puct: Exploration constant (default 2.5).
///
/// Returns:
///     SearchResult with move visits, total simulations, and root value.
#[pyfunction]
#[pyo3(signature = (board, num_simulations=800, temperature=1.0, c_puct=2.5))]
fn search_uniform(
    board: &PyBoard,
    num_simulations: u32,
    temperature: f32,
    c_puct: f32,
) -> PySearchResult {
    let config = MctsConfig {
        c_puct,
        num_simulations,
        temperature,
        ..MctsConfig::default()
    };

    let result = search::search(&board.game, &config);
    convert_search_result(result)
}

/// Run MCTS search with a neural network model.
///
/// Args:
///     board: The board position to search from.
///     model_path: Path to the TorchScript model file (.pt).
///     num_simulations: Number of MCTS simulations (default 800).
///     temperature: Temperature for root noise and move selection (default 1.0).
///     c_puct: Exploration constant (default 2.5).
///     device: Device string: "cpu", "cuda", or "cuda:N" (default "cpu").
///
/// Returns:
///     SearchResult with move visits, total simulations, and root value.
#[pyfunction]
#[pyo3(signature = (board, model_path, num_simulations=800, temperature=1.0, c_puct=2.5, device="cpu"))]
fn search_with_model(
    board: &PyBoard,
    model_path: &str,
    num_simulations: u32,
    temperature: f32,
    c_puct: f32,
    device: &str,
) -> PyResult<PySearchResult> {
    let device = parse_device(device)
        .map_err(|e| PyValueError::new_err(e))?;

    let model = NnModel::load(model_path, device)
        .map_err(|e| PyValueError::new_err(format!("Failed to load model: {}", e)))?;

    let config = MctsConfig {
        c_puct,
        num_simulations,
        temperature,
        ..MctsConfig::default()
    };

    let result = search::search_with_nn(&board.game, &config, &model);
    Ok(convert_search_result(result))
}

/// Convert a Rust SearchResult into a PySearchResult.
fn convert_search_result(result: search::SearchResult) -> PySearchResult {
    let moves: Vec<(String, u32)> = result
        .move_visits
        .iter()
        .map(|(mv, visits)| (mv.to_uci(), *visits))
        .collect();

    let best_move = moves.first().map(|(uci, _)| uci.clone());

    PySearchResult {
        moves,
        total_simulations: result.total_simulations,
        root_value: result.root_value,
        best_move,
    }
}

// =============================================================================
// Python module
// =============================================================================

/// AlphaZero chess engine and MCTS, exposed as a Python module.
#[pymodule]
fn alphazero_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add_class::<PySearchResult>()?;
    m.add_function(wrap_pyfunction!(search_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(search_with_model, m)?)?;
    Ok(())
}
