//! Neural network integration via TorchScript.
//!
//! This module wraps the `tch-rs` crate to load and run TorchScript models
//! exported from the Python training pipeline. It provides:
//!
//! - **Board encoding**: Converts a [`Board`] into the 119x8x8 tensor format
//!   expected by the AlphaZero neural network, matching the Python encoding
//!   in `neural/encoding.py` exactly.
//!
//! - **Move encoding/decoding**: Bidirectional mapping between chess-engine
//!   [`Move`] values and policy vector indices (0..4671), matching the Python
//!   move encoding in `neural/moves.py`.
//!
//! - **Model wrapper**: [`NnModel`] loads a TorchScript `.pt` file and provides
//!   single-position and batched evaluation methods.
//!
//! # Board orientation
//!
//! The network always sees the board from the current player's perspective.
//! When it is black's turn, the board is flipped vertically: rank 0 becomes
//! rank 7, and piece colors are swapped (so the current player's pieces always
//! appear on planes 0-5, and the opponent's on planes 6-11). Moves are also
//! flipped accordingly.
//!
//! # Encoding layout (119 planes)
//!
//! - **History planes** (14 planes x 8 time steps = 112 planes):
//!   - 6 planes for current player's pieces (P, N, B, R, Q, K)
//!   - 6 planes for opponent's pieces (P, N, B, R, Q, K)
//!   - 2 repetition count planes
//!
//! - **Auxiliary planes** (7 planes):
//!   - Color (1 if white to move, 0 if black)
//!   - Fullmove number / 200.0
//!   - White kingside castling right
//!   - White queenside castling right
//!   - Black kingside castling right
//!   - Black queenside castling right
//!   - Halfmove clock / 100.0
//!
//! # Policy vector (4672 = 8 x 8 x 73)
//!
//! Each index encodes (source_rank, source_file, move_type):
//! - 56 queen-type moves: 8 directions x 7 distances
//! - 8 knight moves
//! - 9 underpromotions: 3 piece types x 3 directions

use tch::{CModule, Device, Kind, Tensor};

use chess_engine::board::Board;
use chess_engine::moves::Move;
use chess_engine::types::{Color, Piece, Square};

// =============================================================================
// Constants
// =============================================================================

/// Total number of input planes for the neural network.
const TOTAL_PLANES: usize = 119;

/// Number of planes per history time step (12 piece + 2 repetition).
const PLANES_PER_TIME_STEP: usize = 14;

/// Number of history time steps (T=8 in the AlphaZero paper).
const HISTORY_STEPS: usize = 8;

/// Total history planes: 14 * 8 = 112.
const TOTAL_HISTORY_PLANES: usize = PLANES_PER_TIME_STEP * HISTORY_STEPS;

/// Board dimension.
const BOARD_SIZE: usize = 8;

/// Total number of floats in the encoded tensor: 119 * 8 * 8 = 7616.
const ENCODING_SIZE: usize = TOTAL_PLANES * BOARD_SIZE * BOARD_SIZE;

/// Normalization constant for fullmove number.
const MOVE_COUNT_NORMALIZATION: f32 = 200.0;

/// Normalization constant for halfmove clock.
const HALFMOVE_CLOCK_NORMALIZATION: f32 = 100.0;

/// Offset for opponent's piece planes within a time step.
const OPPONENT_PIECE_OFFSET: usize = 6;

// --- Auxiliary plane indices (absolute) ---

/// Plane 112: color (1 if white to move).
const COLOR_PLANE: usize = TOTAL_HISTORY_PLANES;

/// Plane 113: fullmove number / 200.
const MOVE_COUNT_PLANE: usize = TOTAL_HISTORY_PLANES + 1;

/// Plane 114: white kingside castling.
const CASTLING_WK_PLANE: usize = TOTAL_HISTORY_PLANES + 2;

/// Plane 115: white queenside castling.
const CASTLING_WQ_PLANE: usize = TOTAL_HISTORY_PLANES + 3;

/// Plane 116: black kingside castling.
const CASTLING_BK_PLANE: usize = TOTAL_HISTORY_PLANES + 4;

/// Plane 117: black queenside castling.
const CASTLING_BQ_PLANE: usize = TOTAL_HISTORY_PLANES + 5;

/// Plane 118: halfmove clock / 100.
const NO_PROGRESS_PLANE: usize = TOTAL_HISTORY_PLANES + 6;

// --- Policy vector constants ---

/// Total policy vector size: 8 * 8 * 73 = 4672.
pub const POLICY_SIZE: usize = BOARD_SIZE * BOARD_SIZE * NUM_MOVE_TYPES;

/// Number of move types per source square.
const NUM_MOVE_TYPES: usize = 73;

/// Number of queen-type move types: 8 directions * 7 distances = 56.
const NUM_QUEEN_MOVE_TYPES: usize = 56;

/// Knight move offset within move types.
const KNIGHT_MOVE_OFFSET: usize = NUM_QUEEN_MOVE_TYPES; // 56

/// Underpromotion offset within move types.
const UNDERPROMOTION_OFFSET: usize = NUM_QUEEN_MOVE_TYPES + 8; // 64

/// Queen move directions as (delta_rank, delta_file).
/// Order matches the Python encoding exactly.
const QUEEN_DIRECTIONS: [(i8, i8); 8] = [
    (1, 0),   // 0: North
    (1, 1),   // 1: Northeast
    (0, 1),   // 2: East
    (-1, 1),  // 3: Southeast
    (-1, 0),  // 4: South
    (-1, -1), // 5: Southwest
    (0, -1),  // 6: West
    (1, -1),  // 7: Northwest
];

/// Knight move deltas, matching the Python encoding order.
const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (2, 1),   // index 0 (move_type 56)
    (2, -1),  // index 1 (move_type 57)
    (1, 2),   // index 2 (move_type 58)
    (1, -2),  // index 3 (move_type 59)
    (-1, 2),  // index 4 (move_type 60)
    (-1, -2), // index 5 (move_type 61)
    (-2, 1),  // index 6 (move_type 62)
    (-2, -1), // index 7 (move_type 63)
];

/// Underpromotion file deltas: left-capture, straight, right-capture.
const UNDERPROMOTION_FILE_DELTAS: [i8; 3] = [-1, 0, 1];

/// Underpromotion piece types in order: knight, bishop, rook.
const UNDERPROMOTION_PIECES: [Piece; 3] = [Piece::Knight, Piece::Bishop, Piece::Rook];

// =============================================================================
// Neural network evaluation result
// =============================================================================

/// Neural network evaluation result for a single position.
#[derive(Debug, Clone)]
pub struct NnEval {
    /// Policy logits for all 4672 possible moves (before masking).
    pub policy: Vec<f32>,
    /// Value estimate in [-1, 1] from the current player's perspective.
    pub value: f32,
}

// =============================================================================
// Board Encoding
// =============================================================================

/// Encode a board position into 119 x 8 x 8 = 7616 f32 values.
///
/// This must match the Python encoding in `neural/encoding.py` exactly.
/// For a single position with no history, only the current board planes
/// (time step 0) and auxiliary planes are populated. History steps 1-7
/// are left as zeros.
///
/// The tensor layout is `[channel][rank][file]` where:
/// - `rank 0` = rank 1 (a1-h1) when white to move
/// - When black to move, the board is flipped: `rank 0` = rank 8 (a8-h8)
///
/// Piece planes use "current player" / "opponent" perspective:
/// - Planes 0-5: current player's P, N, B, R, Q, K
/// - Planes 6-11: opponent's P, N, B, R, Q, K
pub fn encode_board(board: &Board) -> Vec<f32> {
    let mut planes = vec![0.0f32; ENCODING_SIZE];

    let side = board.side_to_move();
    let flip = side == Color::Black;

    // --- Encode current position (time step 0) ---
    encode_piece_planes(board, side, flip, 0, &mut planes);
    // Repetition planes (indices 12-13 within time step 0) are left as zeros
    // since we don't have repetition history for a single Board.

    // --- History steps 1-7 are zeros (no history available yet) ---

    // --- Encode auxiliary planes ---
    encode_auxiliary_planes(board, &mut planes);

    planes
}

/// Encode piece planes for one time step into the output buffer.
///
/// `time_step_offset` is the starting plane index (0 for current position,
/// 14 for one move ago, etc.).
fn encode_piece_planes(
    board: &Board,
    side_to_move: Color,
    flip: bool,
    time_step_offset: usize,
    planes: &mut [f32],
) {
    // Iterate over all squares and place pieces into the correct planes
    for rank in 0..8u8 {
        for file in 0..8u8 {
            let sq = Square::from_file_rank(file, rank);
            if let Some((color, piece)) = board.piece_at(sq) {
                // Determine output rank (flip if black to move)
                let out_rank = if flip { 7 - rank } else { rank } as usize;

                // Determine plane index within the time step
                let plane_within_step = if color == side_to_move {
                    piece.index() // 0-5 for current player
                } else {
                    OPPONENT_PIECE_OFFSET + piece.index() // 6-11 for opponent
                };

                let plane_idx = time_step_offset + plane_within_step;
                let flat_idx = plane_idx * BOARD_SIZE * BOARD_SIZE
                    + out_rank * BOARD_SIZE
                    + file as usize;
                planes[flat_idx] = 1.0;
            }
        }
    }
}

/// Encode the 7 auxiliary planes (planes 112-118).
fn encode_auxiliary_planes(board: &Board, planes: &mut [f32]) {
    // Plane 112: Color -- all 1s if white to move
    if board.side_to_move() == Color::White {
        fill_plane(planes, COLOR_PLANE, 1.0);
    }

    // Plane 113: Fullmove number, normalized
    let move_count_value = board.fullmove_number() as f32 / MOVE_COUNT_NORMALIZATION;
    fill_plane(planes, MOVE_COUNT_PLANE, move_count_value);

    // Planes 114-117: Castling rights
    let castling = board.castling_rights();
    if castling.white_kingside() {
        fill_plane(planes, CASTLING_WK_PLANE, 1.0);
    }
    if castling.white_queenside() {
        fill_plane(planes, CASTLING_WQ_PLANE, 1.0);
    }
    if castling.black_kingside() {
        fill_plane(planes, CASTLING_BK_PLANE, 1.0);
    }
    if castling.black_queenside() {
        fill_plane(planes, CASTLING_BQ_PLANE, 1.0);
    }

    // Plane 118: Halfmove clock, normalized
    let no_progress_value = board.halfmove_clock() as f32 / HALFMOVE_CLOCK_NORMALIZATION;
    fill_plane(planes, NO_PROGRESS_PLANE, no_progress_value);
}

/// Fill an entire 8x8 plane with a constant value.
#[inline]
fn fill_plane(planes: &mut [f32], plane_idx: usize, value: f32) {
    let start = plane_idx * BOARD_SIZE * BOARD_SIZE;
    let end = start + BOARD_SIZE * BOARD_SIZE;
    for v in &mut planes[start..end] {
        *v = value;
    }
}

// =============================================================================
// Move Encoding / Decoding
// =============================================================================

/// Convert a chess [`Move`] to a policy index (0..4671).
///
/// Returns `None` if the move cannot be mapped to a valid policy index
/// (e.g., the delta does not match any known move pattern).
///
/// When the side to move is black, ranks are flipped to match the board
/// encoding convention (the network always sees the board from the current
/// player's perspective).
pub fn move_to_policy_index(mv: &Move, board: &Board) -> Option<usize> {
    let flip = board.side_to_move() == Color::Black;

    let mut from_rank = mv.from.rank() as i8;
    let from_file = mv.from.file() as i8;
    let mut to_rank = mv.to.rank() as i8;
    let to_file = mv.to.file() as i8;

    // Flip ranks for black's perspective
    if flip {
        from_rank = 7 - from_rank;
        to_rank = 7 - to_rank;
    }

    let dr = to_rank - from_rank;
    let df = to_file - from_file;

    let move_type = determine_move_type(dr, df, mv.promotion)?;

    let square_index = from_rank as usize * BOARD_SIZE + from_file as usize;
    let index = square_index * NUM_MOVE_TYPES + move_type;

    if index < POLICY_SIZE {
        Some(index)
    } else {
        None
    }
}

/// Convert a policy index (0..4671) to a chess [`Move`].
///
/// Returns `None` if the index is out of range or maps to an off-board move.
///
/// When the side to move is black, the returned move has its ranks flipped
/// back to real board coordinates.
pub fn policy_index_to_move(index: usize, board: &Board) -> Option<Move> {
    if index >= POLICY_SIZE {
        return None;
    }

    let flip = board.side_to_move() == Color::Black;

    let square_index = index / NUM_MOVE_TYPES;
    let move_type = index % NUM_MOVE_TYPES;

    let from_rank = (square_index / BOARD_SIZE) as i8;
    let from_file = (square_index % BOARD_SIZE) as i8;

    let (dr, df, promotion) = move_type_to_delta(move_type)?;

    let to_rank = from_rank + dr;
    let to_file = from_file + df;

    // Check bounds
    if to_rank < 0 || to_rank > 7 || to_file < 0 || to_file > 7 {
        return None;
    }

    // Unflip ranks for black's perspective
    let (actual_from_rank, actual_to_rank) = if flip {
        (7 - from_rank, 7 - to_rank)
    } else {
        (from_rank, to_rank)
    };

    let from_sq = Square::from_file_rank(from_file as u8, actual_from_rank as u8);
    let to_sq = Square::from_file_rank(to_file as u8, actual_to_rank as u8);

    match promotion {
        Some(piece) => Some(Move::with_promotion(from_sq, to_sq, piece)),
        None => {
            // Check if this is a queen-type move from rank 6 to rank 7
            // with distance 1 and |df| <= 1 -- that's a queen promotion
            // in the canonical (flipped) view.
            if to_rank == 7 && from_rank == 6 && move_type < NUM_QUEEN_MOVE_TYPES {
                let direction_idx = move_type / 7;
                let distance = (move_type % 7) + 1;
                let (d_r, d_f) = QUEEN_DIRECTIONS[direction_idx];
                if distance == 1 && d_f.abs() <= 1 && d_r == 1 {
                    // This is a queen promotion
                    Some(Move::with_promotion(from_sq, to_sq, Piece::Queen))
                } else {
                    Some(Move::new(from_sq, to_sq))
                }
            } else {
                Some(Move::new(from_sq, to_sq))
            }
        }
    }
}

/// Determine the move_type index (0..72) from the rank/file deltas and
/// optional promotion piece.
fn determine_move_type(dr: i8, df: i8, promotion: Option<Piece>) -> Option<usize> {
    // Case A: Underpromotion (knight, bishop, or rook)
    if let Some(piece) = promotion {
        if piece != Piece::Queen {
            let piece_index = match piece {
                Piece::Knight => 0,
                Piece::Bishop => 1,
                Piece::Rook => 2,
                _ => return None,
            };
            let dir_index = UNDERPROMOTION_FILE_DELTAS
                .iter()
                .position(|&d| d == df)?;
            return Some(UNDERPROMOTION_OFFSET + piece_index * 3 + dir_index);
        }
        // Queen promotion falls through to queen-type move encoding
    }

    // Case B: Knight move
    if is_knight_move(dr, df) {
        let knight_index = KNIGHT_DELTAS.iter().position(|&(r, f)| r == dr && f == df)?;
        return Some(KNIGHT_MOVE_OFFSET + knight_index);
    }

    // Case C: Queen-type move (including queen promotion)
    let direction_index = delta_to_queen_direction(dr, df)?;
    let distance = dr.abs().max(df.abs()) as usize;
    if distance < 1 || distance > 7 {
        return None;
    }
    Some(direction_index * 7 + (distance - 1))
}

/// Convert a move_type index back to (delta_rank, delta_file, promotion).
fn move_type_to_delta(move_type: usize) -> Option<(i8, i8, Option<Piece>)> {
    if move_type < NUM_QUEEN_MOVE_TYPES {
        // Queen-type move
        let direction_idx = move_type / 7;
        let distance = (move_type % 7) + 1;
        let (dr, df) = QUEEN_DIRECTIONS[direction_idx];
        Some((dr * distance as i8, df * distance as i8, None))
    } else if move_type < KNIGHT_MOVE_OFFSET + 8 {
        // Knight move
        let knight_idx = move_type - KNIGHT_MOVE_OFFSET;
        let (dr, df) = KNIGHT_DELTAS[knight_idx];
        Some((dr, df, None))
    } else if move_type < UNDERPROMOTION_OFFSET + 9 {
        // Underpromotion
        let idx = move_type - UNDERPROMOTION_OFFSET;
        let piece_idx = idx / 3;
        let dir_idx = idx % 3;
        let piece = UNDERPROMOTION_PIECES[piece_idx];
        let df = UNDERPROMOTION_FILE_DELTAS[dir_idx];
        // Underpromotions always go one rank forward
        Some((1, df, Some(piece)))
    } else {
        None
    }
}

/// Check if a delta is a knight move.
#[inline]
fn is_knight_move(dr: i8, df: i8) -> bool {
    let (ar, af) = (dr.abs(), df.abs());
    (ar == 2 && af == 1) || (ar == 1 && af == 2)
}

/// Convert a delta to a queen direction index (0..7).
fn delta_to_queen_direction(dr: i8, df: i8) -> Option<usize> {
    if dr == 0 && df == 0 {
        return None;
    }

    let abs_dr = dr.abs();
    let abs_df = df.abs();

    // Validate: for non-zero components, they must be equal (diagonal)
    // or one must be zero (cardinal).
    if abs_dr != 0 && abs_df != 0 && abs_dr != abs_df {
        return None;
    }

    let sign_r = dr.signum();
    let sign_f = df.signum();
    let unit = (sign_r, sign_f);

    QUEEN_DIRECTIONS.iter().position(|&d| d == unit)
}

// =============================================================================
// Neural Network Model Wrapper
// =============================================================================

/// Wrapper around a TorchScript model for position evaluation.
///
/// Loads a `.pt` file exported by `neural.export.export_torchscript` (FP32)
/// or `neural.export.export_fp16` (FP16) and provides methods to evaluate
/// single positions or batches.
///
/// The input dtype is auto-detected from the model's parameters on load,
/// so both FP32 and FP16 models work transparently.
pub struct NnModel {
    model: CModule,
    device: Device,
    /// Dtype for input tensors, auto-detected from the model's weights.
    input_kind: Kind,
}

/// Extract a flat Vec<f32> from a 1-D tensor.
fn tensor_to_vec_f32(t: &Tensor) -> Vec<f32> {
    let numel = t.numel();
    let flat = t.reshape([numel as i64]).to_kind(Kind::Float).to_device(Device::Cpu);
    let mut out = vec![0f32; numel];
    flat.copy_data(&mut out, numel);
    out
}

/// Extract a scalar f32 from a 0-D or 1-element tensor.
fn tensor_to_scalar_f32(t: &Tensor) -> f32 {
    let flat = t.reshape([1]).to_kind(Kind::Float).to_device(Device::Cpu);
    let mut out = [0f32; 1];
    flat.copy_data(&mut out, 1);
    out[0]
}

impl NnModel {
    /// Load a TorchScript model from a `.pt` file.
    ///
    /// Auto-detects whether the model uses FP32 or FP16 weights and sets
    /// the input dtype accordingly. This allows both `export_torchscript`
    /// (FP32) and `export_fp16` (FP16) models to work transparently.
    ///
    /// # Arguments
    /// * `path` - Path to the TorchScript model file.
    /// * `device` - Device to run inference on (CPU or CUDA).
    pub fn load(path: &str, device: Device) -> Result<Self, tch::TchError> {
        let model = CModule::load_on_device(path, device)?;

        // Detect model dtype from the first parameter's kind.
        let input_kind = model
            .named_parameters()
            .ok()
            .and_then(|params| params.first().map(|(_, t)| t.kind()))
            .unwrap_or(Kind::Float);

        let kind_name = match input_kind {
            Kind::Half => "FP16",
            Kind::Float => "FP32",
            other => {
                eprintln!("[WARN] Unexpected model dtype {:?}, falling back to FP32", other);
                return Ok(Self { model, device, input_kind: Kind::Float });
            }
        };
        eprintln!("[INFO] Model loaded: {} ({})", path, kind_name);

        Ok(Self { model, device, input_kind })
    }

    /// Evaluate a single board position.
    ///
    /// Returns an [`NnEval`] containing policy logits (4672 values) and a
    /// value estimate in [-1, 1].
    pub fn eval_position(&self, board: &Board) -> NnEval {
        let input = encode_board(board);
        let input_tensor = Tensor::from_slice(&input)
            .reshape([1, TOTAL_PLANES as i64, BOARD_SIZE as i64, BOARD_SIZE as i64])
            .to_kind(self.input_kind)
            .to_device(self.device);

        let output = self
            .model
            .forward_is(&[tch::IValue::Tensor(input_tensor)])
            .expect("Model forward pass failed");

        // The output is an IValue::Tuple containing [policy_tensor, value_tensor]
        match output {
            tch::IValue::Tuple(ref values) if values.len() == 2 => {
                let policy_tensor = match &values[0] {
                    tch::IValue::Tensor(t) => t,
                    _ => panic!("Expected tensor for policy output"),
                };
                let value_tensor = match &values[1] {
                    tch::IValue::Tensor(t) => t,
                    _ => panic!("Expected tensor for value output"),
                };

                // policy_tensor shape: (1, 4672)
                let policy = tensor_to_vec_f32(&policy_tensor.squeeze_dim(0));

                // value_tensor shape: (1, 1)
                let value = tensor_to_scalar_f32(&value_tensor.squeeze());

                NnEval { policy, value }
            }
            _ => panic!(
                "Model output is not a 2-tuple. Got: {:?}",
                output_type_name(&output)
            ),
        }
    }

    /// Evaluate a batch of board positions.
    ///
    /// Encodes all boards, runs a single batched forward pass, and splits
    /// the results into per-position [`NnEval`]s.
    pub fn eval_batch(&self, boards: &[Board]) -> Vec<NnEval> {
        if boards.is_empty() {
            return Vec::new();
        }

        if boards.len() == 1 {
            return vec![self.eval_position(&boards[0])];
        }

        let batch_size = boards.len();

        let mut all_data = Vec::with_capacity(batch_size * ENCODING_SIZE);
        for board in boards {
            all_data.extend_from_slice(&encode_board(board));
        }

        let input_tensor = Tensor::from_slice(&all_data)
            .reshape([
                batch_size as i64,
                TOTAL_PLANES as i64,
                BOARD_SIZE as i64,
                BOARD_SIZE as i64,
            ])
            .to_kind(self.input_kind)
            .to_device(self.device);

        let output = self
            .model
            .forward_is(&[tch::IValue::Tensor(input_tensor)])
            .expect("Model forward pass failed");

        match output {
            tch::IValue::Tuple(ref values) if values.len() == 2 => {
                let policy_tensor = match &values[0] {
                    tch::IValue::Tensor(t) => t,
                    _ => panic!("Expected tensor for policy output"),
                };
                let value_tensor = match &values[1] {
                    tch::IValue::Tensor(t) => t,
                    _ => panic!("Expected tensor for value output"),
                };

                // policy_tensor shape: (batch_size, 4672)
                // value_tensor shape: (batch_size, 1)
                let mut results = Vec::with_capacity(batch_size);
                for i in 0..batch_size {
                    let policy = tensor_to_vec_f32(&policy_tensor.get(i as i64));
                    let value = tensor_to_scalar_f32(&value_tensor.get(i as i64));
                    results.push(NnEval { policy, value });
                }
                results
            }
            _ => panic!(
                "Model output is not a 2-tuple. Got: {:?}",
                output_type_name(&output)
            ),
        }
    }
}

/// Helper to describe IValue type for error messages.
fn output_type_name(iv: &tch::IValue) -> &'static str {
    match iv {
        tch::IValue::Tensor(_) => "Tensor",
        tch::IValue::Tuple(_) => "Tuple",
        tch::IValue::Double(_) => "Double",
        tch::IValue::Int(_) => "Int",
        tch::IValue::Bool(_) => "Bool",
        tch::IValue::String(_) => "String",
        _ => "Other",
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chess_engine::board::Board;
    use chess_engine::moves::{Move, MoveFlags};
    use chess_engine::types::{Color, Piece, Square};

    // ---- Encoding shape tests -----------------------------------------------

    #[test]
    fn encode_board_returns_correct_size() {
        let board = Board::starting_position();
        let encoded = encode_board(&board);
        assert_eq!(encoded.len(), ENCODING_SIZE);
        assert_eq!(encoded.len(), 119 * 8 * 8);
    }

    #[test]
    fn encode_board_all_values_finite() {
        let board = Board::starting_position();
        let encoded = encode_board(&board);
        for (i, &val) in encoded.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Non-finite value at index {}: {}",
                i,
                val
            );
        }
    }

    // ---- Starting position piece plane tests --------------------------------

    /// Helper to read a specific plane value at (rank, file).
    fn plane_value(encoded: &[f32], plane: usize, rank: usize, file: usize) -> f32 {
        encoded[plane * 64 + rank * 8 + file]
    }

    #[test]
    fn starting_position_white_to_move_color_plane() {
        let board = Board::starting_position();
        let encoded = encode_board(&board);

        // Plane 112 should be all 1s (white to move)
        for r in 0..8 {
            for f in 0..8 {
                assert_eq!(
                    plane_value(&encoded, COLOR_PLANE, r, f),
                    1.0,
                    "Color plane at ({}, {}) should be 1.0 for white to move",
                    r,
                    f
                );
            }
        }
    }

    #[test]
    fn starting_position_castling_planes() {
        let board = Board::starting_position();
        let encoded = encode_board(&board);

        // All 4 castling planes should be all 1s in starting position
        for plane in [
            CASTLING_WK_PLANE,
            CASTLING_WQ_PLANE,
            CASTLING_BK_PLANE,
            CASTLING_BQ_PLANE,
        ] {
            for r in 0..8 {
                for f in 0..8 {
                    assert_eq!(
                        plane_value(&encoded, plane, r, f),
                        1.0,
                        "Castling plane {} at ({}, {}) should be 1.0",
                        plane,
                        r,
                        f
                    );
                }
            }
        }
    }

    #[test]
    fn starting_position_move_count_plane() {
        let board = Board::starting_position();
        let encoded = encode_board(&board);

        // Fullmove number is 1, so plane value is 1/200 = 0.005
        let expected = 1.0 / 200.0;
        for r in 0..8 {
            for f in 0..8 {
                assert!(
                    (plane_value(&encoded, MOVE_COUNT_PLANE, r, f) - expected).abs() < 1e-6,
                    "Move count plane at ({}, {}): expected {}, got {}",
                    r,
                    f,
                    expected,
                    plane_value(&encoded, MOVE_COUNT_PLANE, r, f)
                );
            }
        }
    }

    #[test]
    fn starting_position_no_progress_plane() {
        let board = Board::starting_position();
        let encoded = encode_board(&board);

        // Halfmove clock is 0, so plane value is 0/100 = 0.0
        for r in 0..8 {
            for f in 0..8 {
                assert_eq!(
                    plane_value(&encoded, NO_PROGRESS_PLANE, r, f),
                    0.0,
                    "No-progress plane at ({}, {}) should be 0.0",
                    r,
                    f
                );
            }
        }
    }

    #[test]
    fn starting_position_white_pawns_on_rank_1() {
        // White to move: current player = white, no flip.
        // White pawns are on rank 1 (index 1). In the encoding, that's
        // plane 0 (current player's pawns), rank index 1.
        let board = Board::starting_position();
        let encoded = encode_board(&board);

        // Plane 0: current player's pawns (white's pawns)
        // White pawns are on rank 1 (0-indexed), all 8 files
        for f in 0..8 {
            assert_eq!(
                plane_value(&encoded, 0, 1, f),
                1.0,
                "White pawn expected at rank 1, file {}",
                f
            );
        }

        // No pawns on other ranks in plane 0
        for r in [0, 2, 3, 4, 5, 6, 7] {
            for f in 0..8 {
                assert_eq!(
                    plane_value(&encoded, 0, r, f),
                    0.0,
                    "No current-player pawn expected at rank {}, file {}",
                    r,
                    f
                );
            }
        }
    }

    #[test]
    fn starting_position_black_pawns_on_rank_6() {
        // White to move: opponent = black, no flip.
        // Black pawns are on rank 6 (0-indexed).
        // They go on plane 6 (opponent's pawns).
        let board = Board::starting_position();
        let encoded = encode_board(&board);

        // Plane 6: opponent's pawns (black's pawns)
        for f in 0..8 {
            assert_eq!(
                plane_value(&encoded, 6, 6, f),
                1.0,
                "Black pawn expected at rank 6, file {}",
                f
            );
        }
    }

    #[test]
    fn starting_position_white_king_on_e1() {
        // White to move: current player = white, king on e1 = (rank 0, file 4).
        // King is piece index 5, so plane 5.
        let board = Board::starting_position();
        let encoded = encode_board(&board);

        assert_eq!(
            plane_value(&encoded, 5, 0, 4),
            1.0,
            "White king should be at rank 0, file 4 (e1)"
        );
    }

    #[test]
    fn starting_position_back_rank_pieces() {
        // White to move. White back rank: rank 0
        // Pieces: R(3) N(1) B(2) Q(4) K(5) B(2) N(1) R(3)
        // Files:   0     1    2    3    4    5    6    7
        let board = Board::starting_position();
        let encoded = encode_board(&board);

        // Current player's pieces (white), planes 0-5
        // Rooks (plane 3): a1 (rank 0, file 0) and h1 (rank 0, file 7)
        assert_eq!(plane_value(&encoded, 3, 0, 0), 1.0); // a1 rook
        assert_eq!(plane_value(&encoded, 3, 0, 7), 1.0); // h1 rook

        // Knights (plane 1): b1 (rank 0, file 1) and g1 (rank 0, file 6)
        assert_eq!(plane_value(&encoded, 1, 0, 1), 1.0); // b1 knight
        assert_eq!(plane_value(&encoded, 1, 0, 6), 1.0); // g1 knight

        // Bishops (plane 2): c1 (rank 0, file 2) and f1 (rank 0, file 5)
        assert_eq!(plane_value(&encoded, 2, 0, 2), 1.0); // c1 bishop
        assert_eq!(plane_value(&encoded, 2, 0, 5), 1.0); // f1 bishop

        // Queen (plane 4): d1 (rank 0, file 3)
        assert_eq!(plane_value(&encoded, 4, 0, 3), 1.0); // d1 queen

        // King (plane 5): e1 (rank 0, file 4)
        assert_eq!(plane_value(&encoded, 5, 0, 4), 1.0); // e1 king
    }

    #[test]
    fn starting_position_opponent_back_rank() {
        // White to move. Opponent (black) back rank: rank 7
        // Opponent pieces go on planes 6-11
        let board = Board::starting_position();
        let encoded = encode_board(&board);

        // Opponent rooks (plane 9): a8 (rank 7, file 0) and h8 (rank 7, file 7)
        assert_eq!(plane_value(&encoded, 9, 7, 0), 1.0);
        assert_eq!(plane_value(&encoded, 9, 7, 7), 1.0);

        // Opponent knights (plane 7): b8 (rank 7, file 1) and g8 (rank 7, file 6)
        assert_eq!(plane_value(&encoded, 7, 7, 1), 1.0);
        assert_eq!(plane_value(&encoded, 7, 7, 6), 1.0);

        // Opponent bishops (plane 8): c8 (rank 7, file 2) and f8 (rank 7, file 5)
        assert_eq!(plane_value(&encoded, 8, 7, 2), 1.0);
        assert_eq!(plane_value(&encoded, 8, 7, 5), 1.0);

        // Opponent queen (plane 10): d8 (rank 7, file 3)
        assert_eq!(plane_value(&encoded, 10, 7, 3), 1.0);

        // Opponent king (plane 11): e8 (rank 7, file 4)
        assert_eq!(plane_value(&encoded, 11, 7, 4), 1.0);
    }

    // ---- Board flipping for black tests ------------------------------------

    #[test]
    fn black_to_move_flips_board() {
        // Create a board where it's black's turn
        let board = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        )
        .expect("Valid FEN");

        let encoded = encode_board(&board);

        // Black to move: current player = black.
        // After flip: black's pieces appear on low ranks, white's on high ranks.

        // Color plane should be 0 (black to move)
        assert_eq!(plane_value(&encoded, COLOR_PLANE, 0, 0), 0.0);

        // Current player (black) pawns should be on rank 1 (after flip: rank 7 -> rank 0... wait)
        // When flip=true: out_rank = 7 - rank
        // Black pawns on rank 6 (0-indexed) -> out_rank = 7 - 6 = 1
        for f in 0..8 {
            assert_eq!(
                plane_value(&encoded, 0, 1, f),
                1.0,
                "After flip, black pawns (current player) should be at out_rank 1, file {}",
                f
            );
        }

        // Current player (black) back rank at rank 7 -> out_rank = 7 - 7 = 0
        // Black king at e8 (rank 7, file 4) -> out_rank 0, file 4
        assert_eq!(
            plane_value(&encoded, 5, 0, 4),
            1.0,
            "After flip, black king should be at out_rank 0, file 4"
        );

        // Opponent (white) pawns: rank 1 -> out_rank = 7 - 1 = 6
        // But white moved e-pawn to e4 (rank 3), so 7 pawns on rank 1 and 1 on rank 3
        // Opponent pawn plane is 6
        for f in 0..8 {
            if f == 4 {
                // e-pawn moved to e4 (rank 3) -> out_rank = 7 - 3 = 4
                assert_eq!(
                    plane_value(&encoded, 6, 6, f),
                    0.0,
                    "e-pawn is not on rank 1 anymore"
                );
                assert_eq!(
                    plane_value(&encoded, 6, 4, f),
                    1.0,
                    "e-pawn should be at out_rank 4 (rank 3 flipped)"
                );
            } else {
                assert_eq!(
                    plane_value(&encoded, 6, 6, f),
                    1.0,
                    "Opponent pawn at out_rank 6, file {}",
                    f
                );
            }
        }
    }

    // ---- History planes are zeros test --------------------------------------

    #[test]
    fn history_planes_are_zeros() {
        let board = Board::starting_position();
        let encoded = encode_board(&board);

        // Time steps 1-7 (planes 14..112) should all be zeros
        for plane in PLANES_PER_TIME_STEP..TOTAL_HISTORY_PLANES {
            for r in 0..8 {
                for f in 0..8 {
                    assert_eq!(
                        plane_value(&encoded, plane, r, f),
                        0.0,
                        "History plane {} at ({}, {}) should be 0.0",
                        plane,
                        r,
                        f
                    );
                }
            }
        }
    }

    // ---- Move encoding tests ------------------------------------------------

    #[test]
    fn white_pawn_e2_e4_move_encoding() {
        // e2 to e4: rank 1 file 4 -> rank 3 file 4
        // Direction: North (0), distance: 2
        // move_type = 0 * 7 + (2 - 1) = 1
        // square_index = 1 * 8 + 4 = 12
        // index = 12 * 73 + 1 = 877
        let board = Board::starting_position();
        let mv = Move::with_flags(Square::E2, Square::E4, MoveFlags::DOUBLE_PAWN_PUSH);
        let idx = move_to_policy_index(&mv, &board);
        assert_eq!(idx, Some(877));
    }

    #[test]
    fn knight_g1_f3_move_encoding() {
        // g1 to f3: rank 0 file 6 -> rank 2 file 5
        // dr = 2, df = -1 -> knight delta index 1 (2, -1)
        // move_type = 56 + 1 = 57
        // square_index = 0 * 8 + 6 = 6
        // index = 6 * 73 + 57 = 495
        let board = Board::starting_position();
        let mv = Move::new(Square::G1, Square::F3);
        let idx = move_to_policy_index(&mv, &board);
        assert_eq!(idx, Some(495));
    }

    #[test]
    fn move_encoding_roundtrip_starting_position() {
        // For every legal move in the starting position, encoding then
        // decoding should produce a move with the same from/to squares.
        let board = Board::starting_position();
        let legal_moves = chess_engine::movegen::generate_legal_moves(&board);

        for mv in &legal_moves {
            let idx = move_to_policy_index(mv, &board)
                .unwrap_or_else(|| panic!("Failed to encode move: {}", mv));

            assert!(
                idx < POLICY_SIZE,
                "Index {} out of range for move {}",
                idx,
                mv
            );

            let decoded = policy_index_to_move(idx, &board)
                .unwrap_or_else(|| panic!("Failed to decode index {} from move {}", idx, mv));

            assert_eq!(
                decoded.from, mv.from,
                "From-square mismatch for move {}: encoded {}, decoded {}",
                mv, idx, decoded
            );
            assert_eq!(
                decoded.to, mv.to,
                "To-square mismatch for move {}: encoded {}, decoded {}",
                mv, idx, decoded
            );
            // Promotion piece should match (for non-promotion moves, both are None)
            assert_eq!(
                decoded.promotion, mv.promotion,
                "Promotion mismatch for move {}: encoded {}, decoded {}",
                mv, idx, decoded
            );
        }
    }

    #[test]
    fn move_encoding_roundtrip_unique_indices() {
        // All legal moves in the starting position should map to distinct indices.
        let board = Board::starting_position();
        let legal_moves = chess_engine::movegen::generate_legal_moves(&board);

        let mut indices: Vec<usize> = legal_moves
            .iter()
            .map(|mv| move_to_policy_index(mv, &board).unwrap())
            .collect();
        let original_len = indices.len();
        indices.sort();
        indices.dedup();
        assert_eq!(
            indices.len(),
            original_len,
            "Some legal moves mapped to the same policy index"
        );
    }

    #[test]
    fn move_encoding_black_perspective_flip() {
        // After 1. e4, it's black's turn. Test that black's moves are
        // encoded correctly with the flip.
        let board = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        )
        .expect("Valid FEN");

        let legal_moves = chess_engine::movegen::generate_legal_moves(&board);

        // Roundtrip test for black's moves
        for mv in &legal_moves {
            let idx = move_to_policy_index(mv, &board)
                .unwrap_or_else(|| panic!("Failed to encode black move: {}", mv));

            let decoded = policy_index_to_move(idx, &board)
                .unwrap_or_else(|| panic!("Failed to decode index {} for black move {}", idx, mv));

            assert_eq!(
                decoded.from, mv.from,
                "From-square mismatch for black move {}",
                mv
            );
            assert_eq!(
                decoded.to, mv.to,
                "To-square mismatch for black move {}",
                mv
            );
        }
    }

    #[test]
    fn move_encoding_black_moves_unique() {
        let board = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        )
        .expect("Valid FEN");

        let legal_moves = chess_engine::movegen::generate_legal_moves(&board);
        let mut indices: Vec<usize> = legal_moves
            .iter()
            .map(|mv| move_to_policy_index(mv, &board).unwrap())
            .collect();
        let original_len = indices.len();
        indices.sort();
        indices.dedup();
        assert_eq!(indices.len(), original_len);
    }

    #[test]
    fn policy_index_out_of_range_returns_none() {
        let board = Board::starting_position();
        assert!(policy_index_to_move(POLICY_SIZE, &board).is_none());
        assert!(policy_index_to_move(POLICY_SIZE + 100, &board).is_none());
    }

    #[test]
    fn queen_promotion_encoding() {
        // White pawn on e7 promotes to queen on e8
        // After the flip logic for white (no flip): from_rank=6, to_rank=7
        // direction: North (0), distance: 1
        // move_type = 0 * 7 + 0 = 0
        // square_index = 6 * 8 + 4 = 52
        // index = 52 * 73 + 0 = 3796
        let mut board = Board::empty();
        board.put_piece(Square::E7, Color::White, Piece::Pawn);
        board.put_piece(Square::E1, Color::White, Piece::King);
        board.put_piece(Square::E8, Color::Black, Piece::King);
        // Don't need legal position for encoding test

        let mv = Move::with_promotion(Square::E7, Square::E8, Piece::Queen);
        let idx = move_to_policy_index(&mv, &board);
        assert_eq!(idx, Some(3796));

        // Decode should give back a queen promotion
        let decoded = policy_index_to_move(3796, &board).unwrap();
        assert_eq!(decoded.from, Square::E7);
        assert_eq!(decoded.to, Square::E8);
        assert_eq!(decoded.promotion, Some(Piece::Queen));
    }

    #[test]
    fn knight_underpromotion_encoding() {
        // White pawn on a7 promotes to knight on a8 (straight push)
        // from_rank=6, from_file=0, to_rank=7, to_file=0
        // Underpromotion: piece=knight(0), direction=straight(1)
        // move_type = 64 + 0*3 + 1 = 65
        // square_index = 6 * 8 + 0 = 48
        // index = 48 * 73 + 65 = 3569
        let board = Board::starting_position(); // just need white to move

        let mv = Move::with_promotion(Square::A7, Square::A8, Piece::Knight);
        let idx = move_to_policy_index(&mv, &board);
        assert_eq!(idx, Some(3569));

        let decoded = policy_index_to_move(3569, &board).unwrap();
        assert_eq!(decoded.from, Square::A7);
        assert_eq!(decoded.to, Square::A8);
        assert_eq!(decoded.promotion, Some(Piece::Knight));
    }

    #[test]
    fn bishop_underpromotion_capture_right() {
        // White pawn on a7 captures on b8 and promotes to bishop
        // from_rank=6, from_file=0, to_rank=7, to_file=1
        // df = 1 -> right capture (index 2)
        // Underpromotion: piece=bishop(1), direction=right(2)
        // move_type = 64 + 1*3 + 2 = 69
        // square_index = 6 * 8 + 0 = 48
        // index = 48 * 73 + 69 = 3573
        let board = Board::starting_position();

        let mv = Move::with_promotion(Square::A7, Square::B8, Piece::Bishop);
        let idx = move_to_policy_index(&mv, &board);
        assert_eq!(idx, Some(3573));
    }

    #[test]
    fn rook_underpromotion_capture_left() {
        // White pawn on b7 captures on a8 and promotes to rook
        // from_rank=6, from_file=1, to_rank=7, to_file=0
        // df = -1 -> left capture (index 0)
        // Underpromotion: piece=rook(2), direction=left(0)
        // move_type = 64 + 2*3 + 0 = 70
        // square_index = 6 * 8 + 1 = 49
        // index = 49 * 73 + 70 = 3647
        let board = Board::starting_position();

        let mv = Move::with_promotion(Square::B7, Square::A8, Piece::Rook);
        let idx = move_to_policy_index(&mv, &board);
        assert_eq!(idx, Some(3647));
    }

    // ---- Encoding plane count sanity check ----------------------------------

    #[test]
    fn encoding_has_correct_number_of_nonzero_planes_starting_position() {
        let board = Board::starting_position();
        let encoded = encode_board(&board);

        let mut nonzero_planes = 0;
        for plane in 0..TOTAL_PLANES {
            let mut has_nonzero = false;
            for r in 0..8 {
                for f in 0..8 {
                    if plane_value(&encoded, plane, r, f) != 0.0 {
                        has_nonzero = true;
                        break;
                    }
                }
                if has_nonzero {
                    break;
                }
            }
            if has_nonzero {
                nonzero_planes += 1;
            }
        }

        // In starting position with white to move:
        // - 12 piece planes (6 white + 6 black) for time step 0
        // - 0 repetition planes (count is 0)
        // - Color plane (1)
        // - Move count plane (1)
        // - 4 castling planes (all rights)
        // - No-progress plane (0, so zero)
        // Total: 12 + 0 + 1 + 1 + 4 + 0 = 18
        assert_eq!(
            nonzero_planes, 18,
            "Expected 18 non-zero planes in starting position, got {}",
            nonzero_planes
        );
    }

    // ---- Cross-validation with Python encoding values ----------------------

    #[test]
    fn starting_position_matches_python_encoding_spot_check() {
        // Spot-check specific values that we can verify by hand against
        // the Python encoding.
        let board = Board::starting_position();
        let encoded = encode_board(&board);

        // === Piece planes (time step 0, white to move, no flip) ===

        // Plane 0 (current player pawn = white pawn):
        //   Rank 1 (index 1) should be all 1s, everything else 0
        for f in 0..8 {
            assert_eq!(plane_value(&encoded, 0, 1, f), 1.0);
            assert_eq!(plane_value(&encoded, 0, 0, f), 0.0);
            assert_eq!(plane_value(&encoded, 0, 2, f), 0.0);
        }

        // Plane 6 (opponent pawn = black pawn):
        //   Rank 6 (index 6) should be all 1s
        for f in 0..8 {
            assert_eq!(plane_value(&encoded, 6, 6, f), 1.0);
            assert_eq!(plane_value(&encoded, 6, 5, f), 0.0);
            assert_eq!(plane_value(&encoded, 6, 7, f), 0.0);
        }

        // === Auxiliary planes ===

        // Plane 112 (color): all 1s
        assert_eq!(plane_value(&encoded, 112, 0, 0), 1.0);
        assert_eq!(plane_value(&encoded, 112, 7, 7), 1.0);

        // Plane 113 (move count): 1/200 = 0.005
        let mc = plane_value(&encoded, 113, 0, 0);
        assert!((mc - 0.005).abs() < 1e-6, "move count: {}", mc);

        // Plane 114 (WK castling): all 1s
        assert_eq!(plane_value(&encoded, 114, 4, 3), 1.0);

        // Plane 118 (no progress): all 0s
        assert_eq!(plane_value(&encoded, 118, 0, 0), 0.0);
    }

    // ---- Additional move encoding edge cases --------------------------------

    #[test]
    fn castling_move_encodes_as_queen_type() {
        // Kingside castle: e1 -> g1 (king moves 2 squares east)
        // Direction: East (2), distance: 2
        // move_type = 2 * 7 + 1 = 15
        // square_index = 0 * 8 + 4 = 4
        // index = 4 * 73 + 15 = 307
        let board = Board::starting_position();
        let mv = Move::with_flags(Square::E1, Square::G1, MoveFlags::KINGSIDE_CASTLE);
        let idx = move_to_policy_index(&mv, &board);
        assert_eq!(idx, Some(307));
    }

    #[test]
    fn en_passant_encodes_as_queen_type() {
        // White pawn on e5 captures en passant on d6
        // e5 = (rank 4, file 4), d6 = (rank 5, file 3)
        // dr=1, df=-1 -> Northwest (7), distance 1
        // move_type = 7 * 7 + 0 = 49
        // square_index = 4 * 8 + 4 = 36
        // index = 36 * 73 + 49 = 2677
        let board = Board::starting_position();
        let mv = Move::with_flags(Square::E5, Square::D6, MoveFlags::EN_PASSANT);
        let idx = move_to_policy_index(&mv, &board);
        assert_eq!(idx, Some(2677));
    }

    #[test]
    fn all_knight_deltas_encode_correctly() {
        let board = Board::starting_position();
        // From e4 (rank 3, file 4), test all 8 knight moves
        let from_sq = Square::E4;
        let expected_deltas = [
            (2, 1),
            (2, -1),
            (1, 2),
            (1, -2),
            (-1, 2),
            (-1, -2),
            (-2, 1),
            (-2, -1),
        ];

        for (i, &(dr, df)) in expected_deltas.iter().enumerate() {
            let to_rank = 3 + dr;
            let to_file = 4 + df;
            if to_rank >= 0 && to_rank < 8 && to_file >= 0 && to_file < 8 {
                let to_sq = Square::from_file_rank(to_file as u8, to_rank as u8);
                let mv = Move::new(from_sq, to_sq);
                let idx = move_to_policy_index(&mv, &board).unwrap();

                // Verify it's a knight-type index
                let move_type = idx % NUM_MOVE_TYPES;
                assert_eq!(
                    move_type,
                    KNIGHT_MOVE_OFFSET + i,
                    "Knight delta {:?} should map to move_type {}",
                    (dr, df),
                    KNIGHT_MOVE_OFFSET + i
                );
            }
        }
    }

    #[test]
    fn queen_directions_all_encode_correctly() {
        let board = Board::starting_position();
        // From d4 (rank 3, file 3), test all 8 queen directions at distance 1
        let from_sq = Square::D4;

        for (dir_idx, &(dr, df)) in QUEEN_DIRECTIONS.iter().enumerate() {
            let to_rank = 3 + dr;
            let to_file = 3 + df;
            if to_rank >= 0 && to_rank < 8 && to_file >= 0 && to_file < 8 {
                let to_sq = Square::from_file_rank(to_file as u8, to_rank as u8);
                let mv = Move::new(from_sq, to_sq);
                let idx = move_to_policy_index(&mv, &board).unwrap();

                let move_type = idx % NUM_MOVE_TYPES;
                // Distance 1 -> offset 0 within direction
                let expected_mt = dir_idx * 7 + 0;
                assert_eq!(
                    move_type, expected_mt,
                    "Queen direction {:?} should map to move_type {}",
                    (dr, df),
                    expected_mt
                );
            }
        }
    }

    // ---- Model loading test (requires model file) ---------------------------

    #[test]
    fn load_model_if_available() {
        let model_path = "/Users/william/Desktop/Random/alphazero/test_model.pt";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping model load test: {} not found", model_path);
            return;
        }

        let model = NnModel::load(model_path, Device::Cpu);
        assert!(model.is_ok(), "Failed to load model: {:?}", model.err());
    }

    #[test]
    fn model_inference_if_available() {
        let model_path = "/Users/william/Desktop/Random/alphazero/test_model.pt";
        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping model inference test: {} not found", model_path);
            return;
        }

        let model = NnModel::load(model_path, Device::Cpu).unwrap();
        let board = Board::starting_position();
        let eval = model.eval_position(&board);

        assert_eq!(
            eval.policy.len(),
            POLICY_SIZE,
            "Policy vector should have {} entries",
            POLICY_SIZE
        );
        assert!(
            eval.value >= -1.0 && eval.value <= 1.0,
            "Value should be in [-1, 1], got {}",
            eval.value
        );
    }

    #[test]
    fn batch_inference_matches_single_if_available() {
        let model_path = "/Users/william/Desktop/Random/alphazero/test_model.pt";
        if !std::path::Path::new(model_path).exists() {
            eprintln!(
                "Skipping batch inference test: {} not found",
                model_path
            );
            return;
        }

        let model = NnModel::load(model_path, Device::Cpu).unwrap();
        let board = Board::starting_position();

        let single_eval = model.eval_position(&board);
        let batch_evals = model.eval_batch(&[board.clone(), board.clone()]);

        assert_eq!(batch_evals.len(), 2);

        // Both batch results should match the single evaluation
        for (i, batch_eval) in batch_evals.iter().enumerate() {
            assert!(
                (batch_eval.value - single_eval.value).abs() < 1e-5,
                "Batch[{}] value mismatch: {} vs {}",
                i,
                batch_eval.value,
                single_eval.value
            );

            // Check policy values match
            for j in 0..POLICY_SIZE {
                assert!(
                    (batch_eval.policy[j] - single_eval.policy[j]).abs() < 1e-5,
                    "Batch[{}] policy[{}] mismatch: {} vs {}",
                    i,
                    j,
                    batch_eval.policy[j],
                    single_eval.policy[j]
                );
            }
        }
    }

    // ---- Cross-validation with Python encoding (exact match) ----------------

    /// Helper to load a JSON file containing a flat array of numbers.
    /// Uses a simple manual parser to avoid additional dependencies.
    fn load_python_encoding(path: &str) -> Option<Vec<f32>> {
        let data = std::fs::read_to_string(path).ok()?;
        // The file contains a JSON array of floating-point numbers.
        // Strip brackets and split by comma.
        let trimmed = data.trim();
        if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
            return None;
        }
        let inner = &trimmed[1..trimmed.len() - 1];
        let values: Vec<f32> = inner
            .split(',')
            .filter_map(|s| s.trim().parse::<f64>().ok().map(|v| v as f32))
            .collect();
        if values.is_empty() {
            return None;
        }
        Some(values)
    }

    #[test]
    fn encoding_matches_python_starting_position() {
        let json_path = "/Users/william/Desktop/Random/alphazero/test_encoding_start.json";
        let python_encoding = match load_python_encoding(json_path) {
            Some(enc) => enc,
            None => {
                eprintln!(
                    "Skipping Python cross-validation: {} not found. \
                     Run the Python export script first.",
                    json_path
                );
                return;
            }
        };

        let board = Board::starting_position();
        let rust_encoding = encode_board(&board);

        assert_eq!(
            rust_encoding.len(),
            python_encoding.len(),
            "Encoding length mismatch: Rust={}, Python={}",
            rust_encoding.len(),
            python_encoding.len()
        );

        let mut max_diff = 0.0f32;
        let mut first_mismatch = None;

        for i in 0..rust_encoding.len() {
            let diff = (rust_encoding[i] - python_encoding[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 1e-6 && first_mismatch.is_none() {
                let plane = i / 64;
                let rank = (i % 64) / 8;
                let file = i % 8;
                first_mismatch = Some((
                    i,
                    plane,
                    rank,
                    file,
                    rust_encoding[i],
                    python_encoding[i],
                ));
            }
        }

        if let Some((idx, plane, rank, file, rust_val, py_val)) = first_mismatch {
            panic!(
                "Encoding mismatch at index {} (plane {}, rank {}, file {}): \
                 Rust={}, Python={}. Max diff={:.8}",
                idx, plane, rank, file, rust_val, py_val, max_diff
            );
        }

        assert!(
            max_diff < 1e-6,
            "Max encoding difference: {:.8} (should be < 1e-6)",
            max_diff
        );
    }

    #[test]
    fn encoding_matches_python_after_e4() {
        let json_path =
            "/Users/william/Desktop/Random/alphazero/test_encoding_after_e4.json";
        let python_encoding = match load_python_encoding(json_path) {
            Some(enc) => enc,
            None => {
                eprintln!(
                    "Skipping Python cross-validation: {} not found.",
                    json_path
                );
                return;
            }
        };

        let board = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        )
        .expect("Valid FEN");
        let rust_encoding = encode_board(&board);

        assert_eq!(rust_encoding.len(), python_encoding.len());

        let mut max_diff = 0.0f32;
        let mut first_mismatch = None;

        for i in 0..rust_encoding.len() {
            let diff = (rust_encoding[i] - python_encoding[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 1e-6 && first_mismatch.is_none() {
                let plane = i / 64;
                let rank = (i % 64) / 8;
                let file = i % 8;
                first_mismatch = Some((
                    i,
                    plane,
                    rank,
                    file,
                    rust_encoding[i],
                    python_encoding[i],
                ));
            }
        }

        if let Some((idx, plane, rank, file, rust_val, py_val)) = first_mismatch {
            panic!(
                "Encoding mismatch at index {} (plane {}, rank {}, file {}): \
                 Rust={}, Python={}. Max diff={:.8}",
                idx, plane, rank, file, rust_val, py_val, max_diff
            );
        }

        assert!(
            max_diff < 1e-6,
            "Max encoding difference: {:.8} (should be < 1e-6)",
            max_diff
        );
    }
}
