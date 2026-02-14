//! Training data collection.
//!
//! This module converts [`GameRecord`](crate::game::GameRecord)s from
//! completed self-play games into [`TrainingSample`]s for neural network
//! training.
//!
//! Each training sample contains:
//! - The board state (to be encoded into a tensor)
//! - The MCTS policy target (normalized visit counts over legal moves)
//! - The value target (game outcome from the side-to-move's perspective)
//!
//! ## Value propagation
//!
//! The game result is converted to a numeric value from White's perspective:
//! - White wins: +1.0
//! - Black wins: -1.0
//! - Draw: 0.0
//!
//! Each position's value is then adjusted to the side-to-move's perspective:
//! - If White is to move, the value is used as-is.
//! - If Black is to move, the value is negated.

use chess_engine::board::Board;
use chess_engine::game::GameResult;
use chess_engine::moves::Move;
use chess_engine::types::Color;

use crate::game::GameRecord;

// =============================================================================
// TrainingSample
// =============================================================================

/// A single training sample for the neural network.
///
/// Each sample represents one position from a self-play game, with:
/// - The board state (to be encoded into the NN input tensor)
/// - The MCTS policy target (normalized visit counts)
/// - The value target (game outcome from this position's perspective)
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// The board state at this position.
    pub board: Board,
    /// Policy target: probability distribution over legal moves.
    /// Each (Move, f32) pair maps a legal move to its visit probability.
    pub policy: Vec<(Move, f32)>,
    /// Value target from the side-to-move's perspective:
    /// +1.0 = side to move won, -1.0 = side to move lost, 0.0 = draw.
    pub value: f32,
}

// =============================================================================
// extract_samples
// =============================================================================

/// Convert a GameRecord into training samples.
///
/// The game result is propagated to each position:
/// - +1 for positions where the winning side was to move
/// - -1 for positions where the losing side was to move
/// - 0 for draws
pub fn extract_samples(record: &GameRecord) -> Vec<TrainingSample> {
    let outcome = result_to_value(record.result);

    record
        .positions
        .iter()
        .map(|pos| {
            // Value from the side-to-move's perspective.
            let value = match pos.side_to_move {
                Color::White => outcome,  // White's perspective
                Color::Black => -outcome, // Black's perspective (negate)
            };

            TrainingSample {
                board: pos.board.clone(),
                policy: pos.policy.clone(),
                value,
            }
        })
        .collect()
}

// =============================================================================
// result_to_value
// =============================================================================

/// Convert a GameResult to a numeric value from White's perspective.
/// +1 = White wins, -1 = Black wins, 0 = draw.
fn result_to_value(result: GameResult) -> f32 {
    match result {
        GameResult::WhiteWins => 1.0,
        GameResult::BlackWins => -1.0,
        _ => 0.0, // All draws and Ongoing
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::PositionRecord;
    use chess_engine::board::Board;
    use chess_engine::game::GameResult;
    use chess_engine::moves::Move;
    use chess_engine::types::{Color, Square};

    /// Helper: create a simple GameRecord with specified result and alternating sides.
    fn make_game_record(result: GameResult, num_positions: u32) -> GameRecord {
        let mut positions = Vec::new();
        for i in 0..num_positions {
            let board = Board::starting_position();
            let side_to_move = if i % 2 == 0 {
                Color::White
            } else {
                Color::Black
            };

            // Create a simple policy with one move.
            let policy = vec![(Move::new(Square::E2, Square::E4), 1.0)];

            positions.push(PositionRecord {
                board,
                side_to_move,
                policy,
                move_number: i,
            });
        }

        GameRecord {
            positions,
            result,
            num_moves: num_positions,
        }
    }

    // ========================================================================
    // 1. White wins: white positions get +1, black positions get -1
    // ========================================================================

    #[test]
    fn white_wins_gives_correct_values() {
        let record = make_game_record(GameResult::WhiteWins, 4);
        let samples = extract_samples(&record);

        assert_eq!(samples.len(), 4);

        // Position 0: White to move -> +1 (White won)
        assert_eq!(samples[0].value, 1.0);
        // Position 1: Black to move -> -1 (White won, Black lost)
        assert_eq!(samples[1].value, -1.0);
        // Position 2: White to move -> +1
        assert_eq!(samples[2].value, 1.0);
        // Position 3: Black to move -> -1
        assert_eq!(samples[3].value, -1.0);
    }

    // ========================================================================
    // 2. Black wins: white positions get -1, black positions get +1
    // ========================================================================

    #[test]
    fn black_wins_gives_correct_values() {
        let record = make_game_record(GameResult::BlackWins, 4);
        let samples = extract_samples(&record);

        assert_eq!(samples.len(), 4);

        // Position 0: White to move -> -1 (Black won, White lost)
        assert_eq!(samples[0].value, -1.0);
        // Position 1: Black to move -> +1 (Black won)
        assert_eq!(samples[1].value, 1.0);
        // Position 2: White to move -> -1
        assert_eq!(samples[2].value, -1.0);
        // Position 3: Black to move -> +1
        assert_eq!(samples[3].value, 1.0);
    }

    // ========================================================================
    // 3. Draw: all positions get 0
    // ========================================================================

    #[test]
    fn draw_gives_zero_values() {
        // Test all draw types.
        let draw_types = [
            GameResult::DrawStalemate,
            GameResult::DrawRepetition,
            GameResult::DrawFiftyMoveRule,
            GameResult::DrawInsufficientMaterial,
        ];

        for draw_result in draw_types {
            let record = make_game_record(draw_result, 4);
            let samples = extract_samples(&record);

            for (i, sample) in samples.iter().enumerate() {
                assert_eq!(
                    sample.value, 0.0,
                    "Draw result {:?}, position {}: expected 0.0, got {}",
                    draw_result, i, sample.value,
                );
            }
        }
    }

    // ========================================================================
    // 4. Number of samples equals number of positions
    // ========================================================================

    #[test]
    fn sample_count_equals_position_count() {
        for &n in &[0, 1, 5, 10] {
            let record = make_game_record(GameResult::WhiteWins, n);
            let samples = extract_samples(&record);
            assert_eq!(
                samples.len(),
                n as usize,
                "Expected {} samples for {} positions",
                n,
                n,
            );
        }
    }

    // ========================================================================
    // 5. Policy is preserved from game record
    // ========================================================================

    #[test]
    fn policy_is_preserved() {
        let policy = vec![
            (Move::new(Square::E2, Square::E4), 0.6),
            (Move::new(Square::D2, Square::D4), 0.3),
            (Move::new(Square::G1, Square::F3), 0.1),
        ];

        let record = GameRecord {
            positions: vec![PositionRecord {
                board: Board::starting_position(),
                side_to_move: Color::White,
                policy: policy.clone(),
                move_number: 0,
            }],
            result: GameResult::WhiteWins,
            num_moves: 1,
        };

        let samples = extract_samples(&record);
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].policy.len(), policy.len());

        for (i, ((sample_mv, sample_p), (record_mv, record_p))) in samples[0]
            .policy
            .iter()
            .zip(policy.iter())
            .enumerate()
        {
            assert_eq!(
                sample_mv, record_mv,
                "Position 0, policy entry {}: move mismatch",
                i,
            );
            assert!(
                (sample_p - record_p).abs() < 1e-6,
                "Position 0, policy entry {}: probability mismatch ({} vs {})",
                i,
                sample_p,
                record_p,
            );
        }
    }

    // ========================================================================
    // 6. All values are exactly in {-1.0, 0.0, +1.0}
    // ========================================================================

    #[test]
    fn all_values_in_valid_set() {
        let results = [
            GameResult::WhiteWins,
            GameResult::BlackWins,
            GameResult::DrawStalemate,
            GameResult::DrawRepetition,
            GameResult::DrawFiftyMoveRule,
            GameResult::DrawInsufficientMaterial,
        ];

        for result in results {
            let record = make_game_record(result, 6);
            let samples = extract_samples(&record);

            for (i, sample) in samples.iter().enumerate() {
                assert!(
                    sample.value == -1.0 || sample.value == 0.0 || sample.value == 1.0,
                    "Result {:?}, position {}: value {} not in {{-1, 0, +1}}",
                    result,
                    i,
                    sample.value,
                );
            }
        }
    }

    // ========================================================================
    // 7. result_to_value unit tests
    // ========================================================================

    #[test]
    fn result_to_value_white_wins() {
        assert_eq!(result_to_value(GameResult::WhiteWins), 1.0);
    }

    #[test]
    fn result_to_value_black_wins() {
        assert_eq!(result_to_value(GameResult::BlackWins), -1.0);
    }

    #[test]
    fn result_to_value_draws() {
        assert_eq!(result_to_value(GameResult::DrawStalemate), 0.0);
        assert_eq!(result_to_value(GameResult::DrawRepetition), 0.0);
        assert_eq!(result_to_value(GameResult::DrawFiftyMoveRule), 0.0);
        assert_eq!(result_to_value(GameResult::DrawInsufficientMaterial), 0.0);
    }

    #[test]
    fn result_to_value_ongoing() {
        assert_eq!(result_to_value(GameResult::Ongoing), 0.0);
    }

    // ========================================================================
    // 8. Empty game record produces no samples
    // ========================================================================

    #[test]
    fn empty_game_record_produces_no_samples() {
        let record = GameRecord {
            positions: Vec::new(),
            result: GameResult::DrawStalemate,
            num_moves: 0,
        };

        let samples = extract_samples(&record);
        assert!(samples.is_empty());
    }

    // ========================================================================
    // 9. Board state is preserved in samples
    // ========================================================================

    #[test]
    fn board_state_is_preserved() {
        let board = Board::starting_position();
        let record = GameRecord {
            positions: vec![PositionRecord {
                board: board.clone(),
                side_to_move: Color::White,
                policy: vec![(Move::new(Square::E2, Square::E4), 1.0)],
                move_number: 0,
            }],
            result: GameResult::DrawStalemate,
            num_moves: 1,
        };

        let samples = extract_samples(&record);
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].board, board);
    }

    // ========================================================================
    // 10. Integration: extract_samples from a real play_game
    // ========================================================================

    #[test]
    fn extract_samples_from_real_game() {
        use crate::game::{play_game, SelfPlayConfig};
        use mcts::config::MctsConfig;
        use mcts::search::UniformEvaluator;

        let config = SelfPlayConfig {
            mcts_config: MctsConfig {
                num_simulations: 30,
                dirichlet_epsilon: 0.0,
                ..MctsConfig::default()
            },
            temperature_threshold: 30,
            max_moves: 512,
            add_noise: false,
        };
        let evaluator = UniformEvaluator;

        let record = play_game(&config, &evaluator);
        let samples = extract_samples(&record);

        // Same number of samples as positions.
        assert_eq!(samples.len(), record.positions.len());

        // All values should be valid.
        for sample in &samples {
            assert!(
                sample.value == -1.0 || sample.value == 0.0 || sample.value == 1.0,
                "Sample value {} not in {{-1, 0, +1}}",
                sample.value,
            );

            // Policy should be non-empty.
            assert!(
                !sample.policy.is_empty(),
                "Sample policy should not be empty",
            );

            // Policy should sum to approximately 1.0.
            let sum: f32 = sample.policy.iter().map(|(_, p)| p).sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Sample policy sum {} should be ~1.0",
                sum,
            );
        }
    }
}
