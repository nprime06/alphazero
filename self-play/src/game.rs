//! Self-play game loop.
//!
//! This module implements a single self-play game using MCTS. Each game
//! produces a [`GameRecord`] containing every position's board state,
//! MCTS policy (visit distribution), and the final game outcome. These
//! records are later converted into neural network training samples by
//! the [`data`](crate::data) module.
//!
//! The game loop follows the AlphaZero protocol:
//!
//! 1. From the current position, run MCTS search.
//! 2. Record the position and the search's visit distribution (policy).
//! 3. Select a move using a temperature schedule:
//!    - **T=1** for the first N moves (proportional to visits, diverse).
//!    - **T->0** after move N (greedy, strongest play).
//! 4. Apply the move and repeat until the game terminates or reaches a
//!    maximum move limit.

use chess_engine::board::Board;
use chess_engine::game::{Game, GameResult};
use chess_engine::moves::Move;
use chess_engine::types::Color;
use mcts::batch::InferenceServer;
use mcts::config::MctsConfig;
use mcts::search::{search_parallel, search_with_evaluator, select_move, LeafEvaluator, SearchResult};

// =============================================================================
// SelfPlayConfig
// =============================================================================

/// Configuration for a single self-play game.
pub struct SelfPlayConfig {
    /// MCTS parameters (simulations, c_puct, etc.)
    pub mcts_config: MctsConfig,
    /// Temperature schedule: use T=1 for first N moves, then T->0.
    /// Default: 30.
    pub temperature_threshold: u32,
    /// Maximum game length before declaring a draw.
    /// Default: 512.
    pub max_moves: u32,
    /// Whether to add Dirichlet noise at root (true for self-play, false for eval).
    pub add_noise: bool,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            mcts_config: MctsConfig::default(),
            temperature_threshold: 30,
            max_moves: 512,
            add_noise: true,
        }
    }
}

// =============================================================================
// PositionRecord
// =============================================================================

/// A record of a single position during the game, before the result is known.
pub struct PositionRecord {
    /// The game state at this position (needed for encoding later).
    pub board: Board,
    /// Side to move at this position.
    pub side_to_move: Color,
    /// MCTS policy (normalized visit counts) for each legal move.
    pub policy: Vec<(Move, f32)>,
    /// Move number (0-indexed).
    pub move_number: u32,
}

// =============================================================================
// GameRecord
// =============================================================================

/// The result of a complete self-play game.
pub struct GameRecord {
    /// Position records for each move.
    pub positions: Vec<PositionRecord>,
    /// Final game result.
    pub result: GameResult,
    /// Total number of moves played.
    pub num_moves: u32,
}

// =============================================================================
// play_game
// =============================================================================

/// Play a single self-play game using the given evaluator.
///
/// The evaluator determines how leaf positions are evaluated during MCTS.
/// For real training, this would be an `NnEvaluator`. For testing, use
/// `UniformEvaluator`.
///
/// # Algorithm
///
/// 1. Check for game termination (checkmate, stalemate, draws).
/// 2. If the game has reached `max_moves`, declare a draw.
/// 3. Run MCTS search from the current position.
/// 4. Record the position and normalized visit counts as the policy target.
/// 5. Select a move using the temperature schedule.
/// 6. Apply the move and increment the move counter.
/// 7. Repeat from step 1.
pub fn play_game(config: &SelfPlayConfig, evaluator: &impl LeafEvaluator) -> GameRecord {
    let mut game = Game::new();
    let mut positions: Vec<PositionRecord> = Vec::new();
    let mut move_number: u32 = 0;

    loop {
        // Check for game termination.
        let result = game.result();
        if result != GameResult::Ongoing {
            return GameRecord {
                positions,
                result,
                num_moves: move_number,
            };
        }

        // Max move limit -- treat as a draw.
        if move_number >= config.max_moves {
            return GameRecord {
                positions,
                result: GameResult::DrawFiftyMoveRule, // treat as draw
                num_moves: move_number,
            };
        }

        // Configure MCTS -- set noise based on config.
        let mut mcts_config = config.mcts_config.clone();
        if !config.add_noise {
            mcts_config.dirichlet_epsilon = 0.0;
        }

        // Run MCTS search.
        let search_result = search_with_evaluator(&game, &mcts_config, evaluator);

        // If search returns empty (terminal position detected by MCTS), stop.
        if search_result.move_visits.is_empty() {
            let result = game.result();
            return GameRecord {
                positions,
                result,
                num_moves: move_number,
            };
        }

        // Record the position BEFORE making the move.
        let policy = normalize_visits(&search_result);
        positions.push(PositionRecord {
            board: game.board().clone(),
            side_to_move: game.board().side_to_move(),
            policy,
            move_number,
        });

        // Select move using temperature schedule.
        let temperature = if move_number < config.temperature_threshold {
            1.0 // Proportional to visits (diverse)
        } else {
            0.0 // Greedy (strongest play)
        };

        let chosen_move = select_move(&search_result, temperature);

        // Make the move.
        game.make_move(chosen_move);
        move_number += 1;
    }
}

// =============================================================================
// play_game_parallel
// =============================================================================

/// Play a single self-play game using parallel MCTS search.
///
/// This is the parallel counterpart of [`play_game`]. Instead of using a
/// single-threaded [`LeafEvaluator`], it uses [`search_parallel`] with an
/// [`InferenceServer`] and multiple search threads. The batched inference
/// server naturally groups leaf evaluations from multiple threads into
/// efficient GPU batches.
///
/// # Arguments
///
/// * `config` -- Self-play configuration (simulations, temperature, etc.)
/// * `server` -- The batched inference server for neural network evaluation.
/// * `num_threads` -- Number of parallel search threads for MCTS.
///
/// # Algorithm
///
/// Same as [`play_game`], but each MCTS search call uses `search_parallel`
/// instead of `search_with_evaluator`.
pub fn play_game_parallel(
    config: &SelfPlayConfig,
    server: &InferenceServer,
    num_threads: usize,
) -> GameRecord {
    let mut game = Game::new();
    let mut positions: Vec<PositionRecord> = Vec::new();
    let mut move_number: u32 = 0;

    loop {
        // Check for game termination.
        let result = game.result();
        if result != GameResult::Ongoing {
            return GameRecord {
                positions,
                result,
                num_moves: move_number,
            };
        }

        // Max move limit -- treat as a draw.
        if move_number >= config.max_moves {
            return GameRecord {
                positions,
                result: GameResult::DrawFiftyMoveRule, // treat as draw
                num_moves: move_number,
            };
        }

        // Configure MCTS -- set noise based on config.
        let mut mcts_config = config.mcts_config.clone();
        if !config.add_noise {
            mcts_config.dirichlet_epsilon = 0.0;
        }

        // Run parallel MCTS search.
        let search_result = search_parallel(&game, &mcts_config, server, num_threads);

        // If search returns empty (terminal position detected by MCTS), stop.
        if search_result.move_visits.is_empty() {
            let result = game.result();
            return GameRecord {
                positions,
                result,
                num_moves: move_number,
            };
        }

        // Record the position BEFORE making the move.
        let policy = normalize_visits(&search_result);
        positions.push(PositionRecord {
            board: game.board().clone(),
            side_to_move: game.board().side_to_move(),
            policy,
            move_number,
        });

        // Select move using temperature schedule.
        let temperature = if move_number < config.temperature_threshold {
            1.0 // Proportional to visits (diverse)
        } else {
            0.0 // Greedy (strongest play)
        };

        let chosen_move = select_move(&search_result, temperature);

        // Make the move.
        game.make_move(chosen_move);
        move_number += 1;
    }
}

// =============================================================================
// normalize_visits
// =============================================================================

/// Convert MCTS visit counts to a probability distribution.
fn normalize_visits(result: &SearchResult) -> Vec<(Move, f32)> {
    let total: u32 = result.move_visits.iter().map(|(_, v)| v).sum();
    if total == 0 {
        // Uniform if no visits.
        let n = result.move_visits.len() as f32;
        return result
            .move_visits
            .iter()
            .map(|(m, _)| (*m, 1.0 / n))
            .collect();
    }
    result
        .move_visits
        .iter()
        .map(|(m, v)| (*m, *v as f32 / total as f32))
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mcts::search::UniformEvaluator;

    /// Helper: create a SelfPlayConfig with low simulation count for fast tests.
    fn test_config(num_simulations: u32) -> SelfPlayConfig {
        SelfPlayConfig {
            mcts_config: MctsConfig {
                num_simulations,
                dirichlet_epsilon: 0.0, // No noise for deterministic tests.
                ..MctsConfig::default()
            },
            temperature_threshold: 30,
            max_moves: 512,
            add_noise: false,
        }
    }

    // ========================================================================
    // 1. Play a game to completion with UniformEvaluator
    // ========================================================================

    #[test]
    fn play_game_completes() {
        let config = test_config(50);
        let evaluator = UniformEvaluator;

        let record = play_game(&config, &evaluator);

        // The game should have ended with a non-Ongoing result.
        assert_ne!(
            record.result,
            GameResult::Ongoing,
            "Game should have terminated"
        );

        // There should be at least one position recorded (games have moves).
        // Note: if the starting position is terminal somehow, positions could
        // be empty -- but that won't happen from a standard starting position.
        assert!(
            !record.positions.is_empty(),
            "Game should have recorded at least one position"
        );

        // num_moves should match the number of position records.
        assert_eq!(
            record.num_moves as usize,
            record.positions.len(),
            "num_moves should equal the number of position records"
        );
    }

    // ========================================================================
    // 2. All moves are legal
    // ========================================================================

    #[test]
    fn all_recorded_moves_are_legal() {
        let config = test_config(30);
        let evaluator = UniformEvaluator;

        let record = play_game(&config, &evaluator);

        for (i, pos) in record.positions.iter().enumerate() {
            let game = Game::from_board(pos.board.clone());
            let legal_moves = game.legal_moves();

            // Every move in the policy should be a legal move.
            for (mv, prob) in &pos.policy {
                assert!(
                    legal_moves.contains(mv),
                    "Position {}: move {} in policy is not legal",
                    i,
                    mv,
                );
                assert!(
                    *prob >= 0.0,
                    "Position {}: move {} has negative probability {}",
                    i,
                    mv,
                    prob,
                );
            }
        }
    }

    // ========================================================================
    // 3. Game ends with a valid non-Ongoing result
    // ========================================================================

    #[test]
    fn game_ends_with_valid_result() {
        let config = test_config(30);
        let evaluator = UniformEvaluator;

        let record = play_game(&config, &evaluator);

        let valid_results = [
            GameResult::WhiteWins,
            GameResult::BlackWins,
            GameResult::DrawStalemate,
            GameResult::DrawRepetition,
            GameResult::DrawFiftyMoveRule,
            GameResult::DrawInsufficientMaterial,
        ];

        assert!(
            valid_results.contains(&record.result),
            "Game result {:?} should be a valid terminal result",
            record.result,
        );
    }

    // ========================================================================
    // 4. PositionRecord count equals number of moves played
    // ========================================================================

    #[test]
    fn position_count_equals_moves_played() {
        let config = test_config(30);
        let evaluator = UniformEvaluator;

        let record = play_game(&config, &evaluator);

        assert_eq!(
            record.positions.len(),
            record.num_moves as usize,
            "Position records ({}) should equal num_moves ({})",
            record.positions.len(),
            record.num_moves,
        );
    }

    // ========================================================================
    // 5. Policy values sum to approximately 1.0 for each position
    // ========================================================================

    #[test]
    fn policy_sums_to_one() {
        let config = test_config(30);
        let evaluator = UniformEvaluator;

        let record = play_game(&config, &evaluator);

        for (i, pos) in record.positions.iter().enumerate() {
            let policy_sum: f32 = pos.policy.iter().map(|(_, p)| p).sum();
            assert!(
                (policy_sum - 1.0).abs() < 0.01,
                "Position {}: policy sum {} should be approximately 1.0",
                i,
                policy_sum,
            );
        }
    }

    // ========================================================================
    // 6. Temperature threshold behavior
    // ========================================================================

    #[test]
    fn temperature_threshold_applied() {
        // We can't directly observe what temperature was used, but we can
        // verify the config structure is respected by checking that games
        // complete without issues when the threshold is set to different values.
        let config_early = SelfPlayConfig {
            mcts_config: MctsConfig {
                num_simulations: 30,
                dirichlet_epsilon: 0.0,
                ..MctsConfig::default()
            },
            temperature_threshold: 5, // Switch to greedy after 5 moves.
            max_moves: 512,
            add_noise: false,
        };
        let evaluator = UniformEvaluator;

        let record = play_game(&config_early, &evaluator);
        assert_ne!(record.result, GameResult::Ongoing);
    }

    // ========================================================================
    // 7. Max moves limit terminates the game
    // ========================================================================

    #[test]
    fn max_moves_limit_terminates_game() {
        // Use a very low max_moves to force early termination.
        let config = SelfPlayConfig {
            mcts_config: MctsConfig {
                num_simulations: 10,
                dirichlet_epsilon: 0.0,
                ..MctsConfig::default()
            },
            temperature_threshold: 30,
            max_moves: 5, // Only allow 5 moves.
            add_noise: false,
        };
        let evaluator = UniformEvaluator;

        let record = play_game(&config, &evaluator);

        // The game should have terminated at or before 5 moves.
        assert!(
            record.num_moves <= 5,
            "Game should have terminated within max_moves (5), but played {} moves",
            record.num_moves,
        );

        // If the game hit the max_moves limit (not a natural termination),
        // the result should be a draw.
        if record.num_moves == 5 {
            // The result should be the draw type we assign for max-move limit.
            assert_eq!(
                record.result,
                GameResult::DrawFiftyMoveRule,
                "Max-moves termination should be treated as a draw",
            );
        }
    }

    // ========================================================================
    // 8. Default config has reasonable values
    // ========================================================================

    #[test]
    fn default_config_has_reasonable_values() {
        let config = SelfPlayConfig::default();

        assert_eq!(
            config.temperature_threshold, 30,
            "Default temperature_threshold should be 30"
        );
        assert_eq!(config.max_moves, 512, "Default max_moves should be 512");
        assert!(config.add_noise, "Default add_noise should be true");

        // MCTS config should have the default AlphaZero values.
        assert_eq!(config.mcts_config.num_simulations, 800);
        assert_eq!(config.mcts_config.c_puct, 2.5);
        assert_eq!(config.mcts_config.dirichlet_alpha, 0.3);
        assert_eq!(config.mcts_config.dirichlet_epsilon, 0.25);
    }

    // ========================================================================
    // 9. Move numbers are sequential
    // ========================================================================

    #[test]
    fn move_numbers_are_sequential() {
        let config = test_config(30);
        let evaluator = UniformEvaluator;

        let record = play_game(&config, &evaluator);

        for (i, pos) in record.positions.iter().enumerate() {
            assert_eq!(
                pos.move_number, i as u32,
                "Position {}: move_number should be {}, got {}",
                i, i, pos.move_number,
            );
        }
    }

    // ========================================================================
    // 10. Side to move alternates (mostly)
    // ========================================================================

    #[test]
    fn side_to_move_alternates() {
        let config = test_config(30);
        let evaluator = UniformEvaluator;

        let record = play_game(&config, &evaluator);

        // The starting position is White, so position 0 should be White,
        // position 1 should be Black, etc.
        for (i, pos) in record.positions.iter().enumerate() {
            let expected = if i % 2 == 0 {
                Color::White
            } else {
                Color::Black
            };
            assert_eq!(
                pos.side_to_move, expected,
                "Position {}: expected {:?} to move, got {:?}",
                i, expected, pos.side_to_move,
            );
        }
    }

    // ========================================================================
    // 11. normalize_visits produces valid distribution
    // ========================================================================

    #[test]
    fn normalize_visits_produces_valid_distribution() {
        use chess_engine::types::Square;

        let result = SearchResult {
            move_visits: vec![
                (Move::new(Square::E2, Square::E4), 30),
                (Move::new(Square::D2, Square::D4), 20),
                (Move::new(Square::G1, Square::F3), 10),
            ],
            total_simulations: 60,
            root_value: 0.0,
        };

        let policy = normalize_visits(&result);

        assert_eq!(policy.len(), 3);

        // Check values are proportional to visits.
        assert!((policy[0].1 - 0.5).abs() < 1e-6, "30/60 = 0.5");
        assert!(
            (policy[1].1 - 1.0 / 3.0).abs() < 1e-6,
            "20/60 = 0.333..."
        );
        assert!(
            (policy[2].1 - 1.0 / 6.0).abs() < 1e-6,
            "10/60 = 0.166..."
        );

        // Sum should be 1.0.
        let sum: f32 = policy.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    // ========================================================================
    // 12. normalize_visits handles zero total visits
    // ========================================================================

    #[test]
    fn normalize_visits_handles_zero_visits() {
        use chess_engine::types::Square;

        let result = SearchResult {
            move_visits: vec![
                (Move::new(Square::E2, Square::E4), 0),
                (Move::new(Square::D2, Square::D4), 0),
            ],
            total_simulations: 0,
            root_value: 0.0,
        };

        let policy = normalize_visits(&result);

        assert_eq!(policy.len(), 2);

        // Should be uniform: 1/2 each.
        for (_, p) in &policy {
            assert!((*p - 0.5).abs() < 1e-6, "Expected 0.5, got {}", p);
        }
    }

    // ========================================================================
    // 13. Noise configuration is respected
    // ========================================================================

    #[test]
    fn noise_disabled_when_add_noise_false() {
        // When add_noise is false, the mcts_config should have epsilon=0.
        // We verify this by running a game with add_noise=false and checking
        // it completes without issue.
        let config = SelfPlayConfig {
            mcts_config: MctsConfig {
                num_simulations: 30,
                dirichlet_epsilon: 0.25, // Set non-zero, but add_noise=false.
                ..MctsConfig::default()
            },
            temperature_threshold: 30,
            max_moves: 512,
            add_noise: false, // Should override epsilon to 0.
        };
        let evaluator = UniformEvaluator;

        let record = play_game(&config, &evaluator);
        assert_ne!(record.result, GameResult::Ongoing);
    }

    // ========================================================================
    // 14. Game with noise enabled completes
    // ========================================================================

    #[test]
    fn game_with_noise_completes() {
        let config = SelfPlayConfig {
            mcts_config: MctsConfig {
                num_simulations: 30,
                dirichlet_epsilon: 0.25,
                dirichlet_alpha: 0.3,
                ..MctsConfig::default()
            },
            temperature_threshold: 30,
            max_moves: 512,
            add_noise: true,
        };
        let evaluator = UniformEvaluator;

        let record = play_game(&config, &evaluator);
        assert_ne!(record.result, GameResult::Ongoing);
    }

    // ========================================================================
    // Parallel game tests (gated on model file existence)
    // ========================================================================

    /// Path to the test model file.
    const TEST_MODEL_PATH: &str = "/Users/william/Desktop/Random/alphazero/test_model.pt";

    /// Helper to create an InferenceServer from the test model.
    fn get_test_server() -> Option<InferenceServer> {
        if !std::path::Path::new(TEST_MODEL_PATH).exists() {
            eprintln!(
                "Skipping parallel game test: {} not found. \
                 Run the Python export script to create it.",
                TEST_MODEL_PATH
            );
            return None;
        }
        let model = mcts::nn::NnModel::load(TEST_MODEL_PATH, tch::Device::Cpu)
            .expect("Failed to load test model");
        Some(InferenceServer::new(model, 4, 50))
    }

    // 15. play_game_parallel completes a game
    #[test]
    fn play_game_parallel_completes() {
        let server = match get_test_server() {
            Some(s) => s,
            None => return,
        };

        let config = SelfPlayConfig {
            mcts_config: MctsConfig {
                num_simulations: 30,
                dirichlet_epsilon: 0.0,
                ..MctsConfig::default()
            },
            temperature_threshold: 10,
            max_moves: 20, // Keep short for test speed.
            add_noise: false,
        };

        let record = play_game_parallel(&config, &server, 2);

        assert_ne!(
            record.result,
            GameResult::Ongoing,
            "Parallel game should have terminated"
        );
        assert!(
            !record.positions.is_empty(),
            "Parallel game should have recorded at least one position"
        );
        assert_eq!(
            record.num_moves as usize,
            record.positions.len(),
            "num_moves should equal position count"
        );

        server.shutdown();
    }

    // 16. play_game_parallel produces valid results (legal moves, policy sums to 1)
    #[test]
    fn play_game_parallel_produces_valid_results() {
        let server = match get_test_server() {
            Some(s) => s,
            None => return,
        };

        let config = SelfPlayConfig {
            mcts_config: MctsConfig {
                num_simulations: 30,
                dirichlet_epsilon: 0.25,
                ..MctsConfig::default()
            },
            temperature_threshold: 10,
            max_moves: 20,
            add_noise: true,
        };

        let record = play_game_parallel(&config, &server, 2);

        // Check all positions have valid policies.
        for (i, pos) in record.positions.iter().enumerate() {
            let game = Game::from_board(pos.board.clone());
            let legal_moves = game.legal_moves();

            // Every move in the policy should be legal.
            for (mv, prob) in &pos.policy {
                assert!(
                    legal_moves.contains(mv),
                    "Parallel game position {}: move {} is not legal",
                    i, mv,
                );
                assert!(
                    *prob >= 0.0,
                    "Parallel game position {}: move {} has negative probability {}",
                    i, mv, prob,
                );
            }

            // Policy should sum to approximately 1.0.
            let policy_sum: f32 = pos.policy.iter().map(|(_, p)| p).sum();
            assert!(
                (policy_sum - 1.0).abs() < 0.01,
                "Parallel game position {}: policy sum {} should be ~1.0",
                i, policy_sum,
            );
        }

        // Move numbers should be sequential.
        for (i, pos) in record.positions.iter().enumerate() {
            assert_eq!(
                pos.move_number, i as u32,
                "Parallel game: move_number mismatch at position {}",
                i,
            );
        }

        server.shutdown();
    }
}
