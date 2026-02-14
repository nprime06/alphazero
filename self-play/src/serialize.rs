//! Data serialization for training samples.
//!
//! This module converts [`TrainingSample`]s to a compact, serializable format
//! using [MessagePack](https://msgpack.org/) via `rmp-serde`. Rather than adding
//! serde derives to all chess-engine types (which would create a tight coupling),
//! we use an intermediate [`SerializedSample`] struct that stores:
//!
//! - The board as a FEN string
//! - The policy as `(u16, f32)` pairs where the `u16` is a policy index (0..4671)
//! - The value as an `f32`
//!
//! # File format
//!
//! A game file is a single MessagePack-encoded [`GameFile`] struct containing:
//! - A [`GameFileHeader`] with a format version and sample count
//! - A vector of [`SerializedSample`]s
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use self_play::serialize::{write_game_file, read_game_file};
//!
//! // After playing a game and extracting samples...
//! // write_game_file(Path::new("game_001.msgpack"), &samples).unwrap();
//! // let loaded = read_game_file(Path::new("game_001.msgpack")).unwrap();
//! ```

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

use chess_engine::board::Board;
use chess_engine::moves::Move;
use mcts::nn::{move_to_policy_index, policy_index_to_move};

use crate::data::TrainingSample;

// =============================================================================
// Constants
// =============================================================================

/// Version of the serialization format.
///
/// Bump this when making breaking changes to the file format so that readers
/// can detect and reject incompatible files.
const FORMAT_VERSION: u32 = 1;

// =============================================================================
// Serializable types
// =============================================================================

/// Header for a game data file.
///
/// Stored at the beginning of each `.msgpack` file to enable quick validation
/// without deserializing all samples.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GameFileHeader {
    /// Format version (must match [`FORMAT_VERSION`] when reading).
    pub version: u32,
    /// Number of training samples in this file.
    pub num_samples: u32,
}

/// Compact, serializable representation of a training sample.
///
/// This is the on-disk format. Instead of storing the full `Board` (which has
/// complex non-serde types), we store:
/// - FEN string for the board position
/// - Policy as `Vec<(u16, f32)>` where `u16` is the policy index (0..4671)
/// - Value as `f32`
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializedSample {
    /// Board position as a FEN string.
    pub fen: String,
    /// Policy targets: `(policy_index, probability)` pairs.
    /// Policy index is from `move_to_policy_index` (0..4671).
    pub policy: Vec<(u16, f32)>,
    /// Value target from the side-to-move's perspective.
    pub value: f32,
}

/// A complete game file containing all training samples from one game.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GameFile {
    /// File header with version and sample count.
    pub header: GameFileHeader,
    /// Training samples from this game.
    pub samples: Vec<SerializedSample>,
}

// =============================================================================
// Conversion functions
// =============================================================================

/// Convert a [`TrainingSample`] to a [`SerializedSample`].
///
/// Each `(Move, f32)` in the policy is mapped to `(u16, f32)` using
/// `move_to_policy_index`. Moves that cannot be mapped (which should never
/// happen for legal moves from MCTS) are skipped with a warning.
pub fn serialize_sample(sample: &TrainingSample) -> SerializedSample {
    let fen = sample.board.to_fen();
    let policy: Vec<(u16, f32)> = sample
        .policy
        .iter()
        .filter_map(|(mv, prob)| {
            match move_to_policy_index(mv, &sample.board) {
                Some(idx) => Some((idx as u16, *prob)),
                None => {
                    // This should not happen for legal moves from MCTS search,
                    // but we handle it gracefully rather than panicking.
                    eprintln!(
                        "WARNING: could not map move {} to policy index (FEN: {})",
                        mv,
                        sample.board.to_fen()
                    );
                    None
                }
            }
        })
        .collect();

    SerializedSample {
        fen,
        policy,
        value: sample.value,
    }
}

/// Convert a [`SerializedSample`] back to a [`TrainingSample`].
///
/// # Panics
///
/// Panics if:
/// - The FEN string is invalid
/// - Any policy index cannot be mapped back to a move
pub fn deserialize_sample(sample: &SerializedSample) -> TrainingSample {
    let board = Board::from_fen(&sample.fen).expect("invalid FEN in serialized sample");
    let policy: Vec<(Move, f32)> = sample
        .policy
        .iter()
        .map(|(idx, prob)| {
            let mv = policy_index_to_move(*idx as usize, &board)
                .expect("invalid policy index in serialized sample");
            (mv, *prob)
        })
        .collect();

    TrainingSample {
        board,
        policy,
        value: sample.value,
    }
}

// =============================================================================
// File I/O
// =============================================================================

/// Write a list of training samples to a file as MessagePack.
///
/// Creates a [`GameFile`] with a header and the serialized samples, then
/// writes it as a single MessagePack blob.
pub fn write_game_file(path: &Path, samples: &[TrainingSample]) -> io::Result<()> {
    let serialized: Vec<SerializedSample> = samples.iter().map(serialize_sample).collect();
    let game_file = GameFile {
        header: GameFileHeader {
            version: FORMAT_VERSION,
            num_samples: serialized.len() as u32,
        },
        samples: serialized,
    };
    let bytes =
        rmp_serde::to_vec(&game_file).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    std::fs::write(path, bytes)
}

/// Read training samples from a MessagePack game file.
///
/// Validates the format version and deserializes all samples back to
/// [`TrainingSample`]s.
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - The MessagePack data is malformed
/// - The format version does not match [`FORMAT_VERSION`]
pub fn read_game_file(path: &Path) -> io::Result<Vec<TrainingSample>> {
    let bytes = std::fs::read(path)?;
    let game_file: GameFile =
        rmp_serde::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    if game_file.header.version != FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported format version: {} (expected {})",
                game_file.header.version, FORMAT_VERSION
            ),
        ));
    }
    Ok(game_file.samples.iter().map(deserialize_sample).collect())
}

/// Read the raw serialized game file (without deserializing samples to TrainingSample).
///
/// This is useful when you only need the serialized form (e.g., for Python
/// interop testing).
pub fn read_game_file_raw(path: &Path) -> io::Result<GameFile> {
    let bytes = std::fs::read(path)?;
    let game_file: GameFile =
        rmp_serde::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    if game_file.header.version != FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported format version: {} (expected {})",
                game_file.header.version, FORMAT_VERSION
            ),
        ));
    }
    Ok(game_file)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chess_engine::board::Board;
    use chess_engine::moves::Move;
    use chess_engine::types::Square;
    use mcts::nn::POLICY_SIZE;

    /// Helper: create a TrainingSample from the starting position with a simple policy.
    fn make_starting_sample(value: f32) -> TrainingSample {
        let board = Board::starting_position();
        let policy = vec![
            (Move::new(Square::E2, Square::E4), 0.5),
            (Move::new(Square::D2, Square::D4), 0.3),
            (Move::new(Square::G1, Square::F3), 0.2),
        ];
        TrainingSample {
            board,
            policy,
            value,
        }
    }

    // ========================================================================
    // 1. Roundtrip: serialize_sample -> deserialize_sample preserves data
    // ========================================================================

    #[test]
    fn roundtrip_preserves_board_fen() {
        let original = make_starting_sample(1.0);
        let serialized = serialize_sample(&original);
        let deserialized = deserialize_sample(&serialized);

        assert_eq!(
            original.board.to_fen(),
            deserialized.board.to_fen(),
            "FEN should match after roundtrip"
        );
    }

    #[test]
    fn roundtrip_preserves_policy() {
        let original = make_starting_sample(1.0);
        let serialized = serialize_sample(&original);
        let deserialized = deserialize_sample(&serialized);

        assert_eq!(
            original.policy.len(),
            deserialized.policy.len(),
            "Policy length should match"
        );

        for (i, ((orig_mv, orig_p), (deser_mv, deser_p))) in original
            .policy
            .iter()
            .zip(deserialized.policy.iter())
            .enumerate()
        {
            assert_eq!(
                orig_mv.from, deser_mv.from,
                "Policy entry {}: from-square mismatch",
                i
            );
            assert_eq!(
                orig_mv.to, deser_mv.to,
                "Policy entry {}: to-square mismatch",
                i
            );
            assert_eq!(
                orig_mv.promotion, deser_mv.promotion,
                "Policy entry {}: promotion mismatch",
                i
            );
            assert!(
                (orig_p - deser_p).abs() < 1e-6,
                "Policy entry {}: probability mismatch ({} vs {})",
                i,
                orig_p,
                deser_p
            );
        }
    }

    #[test]
    fn roundtrip_preserves_value() {
        for &value in &[-1.0, 0.0, 1.0] {
            let original = make_starting_sample(value);
            let serialized = serialize_sample(&original);
            let deserialized = deserialize_sample(&serialized);

            assert_eq!(
                original.value, deserialized.value,
                "Value should be preserved exactly for {}",
                value
            );
        }
    }

    // ========================================================================
    // 2. write_game_file -> read_game_file roundtrip
    // ========================================================================

    #[test]
    fn file_roundtrip_with_multiple_samples() {
        let samples = vec![
            make_starting_sample(1.0),
            make_starting_sample(-1.0),
            make_starting_sample(0.0),
        ];

        let dir = std::env::temp_dir().join("alphazero_test_serialize");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_roundtrip.msgpack");

        write_game_file(&path, &samples).unwrap();
        let loaded = read_game_file(&path).unwrap();

        assert_eq!(
            samples.len(),
            loaded.len(),
            "Should load same number of samples"
        );

        for (i, (orig, loaded)) in samples.iter().zip(loaded.iter()).enumerate() {
            assert_eq!(
                orig.board.to_fen(),
                loaded.board.to_fen(),
                "Sample {}: FEN mismatch",
                i
            );
            assert_eq!(
                orig.value, loaded.value,
                "Sample {}: value mismatch",
                i
            );
            assert_eq!(
                orig.policy.len(),
                loaded.policy.len(),
                "Sample {}: policy length mismatch",
                i
            );
        }

        // Clean up
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    // ========================================================================
    // 3. Policy indices are valid (0..4671)
    // ========================================================================

    #[test]
    fn policy_indices_are_valid() {
        let sample = make_starting_sample(1.0);
        let serialized = serialize_sample(&sample);

        for (idx, _prob) in &serialized.policy {
            assert!(
                (*idx as usize) < POLICY_SIZE,
                "Policy index {} should be < {} (POLICY_SIZE)",
                idx,
                POLICY_SIZE
            );
        }
    }

    // ========================================================================
    // 4. Value is preserved exactly
    // ========================================================================

    #[test]
    fn value_preserved_exactly() {
        let values = [-1.0f32, -0.5, 0.0, 0.5, 1.0];
        for &v in &values {
            let sample = TrainingSample {
                board: Board::starting_position(),
                policy: vec![(Move::new(Square::E2, Square::E4), 1.0)],
                value: v,
            };
            let ser = serialize_sample(&sample);
            assert_eq!(ser.value, v, "Serialized value should be exact");
            let deser = deserialize_sample(&ser);
            assert_eq!(deser.value, v, "Deserialized value should be exact");
        }
    }

    // ========================================================================
    // 5. File format version check (reject wrong version)
    // ========================================================================

    #[test]
    fn rejects_wrong_format_version() {
        let game_file = GameFile {
            header: GameFileHeader {
                version: 999, // Wrong version
                num_samples: 0,
            },
            samples: vec![],
        };
        let bytes = rmp_serde::to_vec(&game_file).unwrap();

        let dir = std::env::temp_dir().join("alphazero_test_version");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_version.msgpack");
        std::fs::write(&path, &bytes).unwrap();

        let result = read_game_file(&path);
        assert!(result.is_err(), "Should reject wrong format version");
        let err = result.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("unsupported format version"),
            "Error message should mention version: {}",
            err
        );

        // Clean up
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    // ========================================================================
    // 6. Integration with play_game: play, extract, serialize, deserialize
    // ========================================================================

    #[test]
    fn integration_play_serialize_deserialize() {
        use crate::data::extract_samples;
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
            max_moves: 50, // Keep it short for testing
            add_noise: false,
        };
        let evaluator = UniformEvaluator;

        // Play a game and extract samples
        let record = play_game(&config, &evaluator);
        let samples = extract_samples(&record);
        assert!(!samples.is_empty(), "Should have at least one sample");

        // Write to file
        let dir = std::env::temp_dir().join("alphazero_test_integration");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("integration_game.msgpack");

        write_game_file(&path, &samples).unwrap();

        // Read back
        let loaded = read_game_file(&path).unwrap();
        assert_eq!(samples.len(), loaded.len(), "Sample count should match");

        // Verify each sample
        for (i, (orig, loaded)) in samples.iter().zip(loaded.iter()).enumerate() {
            assert_eq!(
                orig.board.to_fen(),
                loaded.board.to_fen(),
                "Sample {}: FEN mismatch",
                i
            );
            assert_eq!(
                orig.value, loaded.value,
                "Sample {}: value mismatch",
                i
            );
            assert_eq!(
                orig.policy.len(),
                loaded.policy.len(),
                "Sample {}: policy length mismatch",
                i
            );

            // Verify policy sums approximately to 1.0
            let sum: f32 = loaded.policy.iter().map(|(_, p)| p).sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Sample {}: loaded policy sum {} should be ~1.0",
                i,
                sum
            );
        }

        // Clean up
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    // ========================================================================
    // 7. Empty samples list writes and reads correctly
    // ========================================================================

    #[test]
    fn empty_samples_roundtrip() {
        let dir = std::env::temp_dir().join("alphazero_test_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.msgpack");

        write_game_file(&path, &[]).unwrap();
        let loaded = read_game_file(&path).unwrap();
        assert!(loaded.is_empty(), "Empty samples should roundtrip to empty");

        // Clean up
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    // ========================================================================
    // 8. Serialized FEN matches original board
    // ========================================================================

    #[test]
    fn serialized_fen_matches_original_board() {
        let sample = make_starting_sample(0.0);
        let serialized = serialize_sample(&sample);
        assert_eq!(
            serialized.fen,
            sample.board.to_fen(),
            "Serialized FEN should match board's FEN"
        );
    }

    // ========================================================================
    // 9. GameFileHeader has correct sample count
    // ========================================================================

    #[test]
    fn header_sample_count_matches() {
        let samples = vec![
            make_starting_sample(1.0),
            make_starting_sample(-1.0),
        ];

        let dir = std::env::temp_dir().join("alphazero_test_header");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("header_test.msgpack");

        write_game_file(&path, &samples).unwrap();

        // Read raw to check header
        let game_file = read_game_file_raw(&path).unwrap();
        assert_eq!(
            game_file.header.num_samples, 2,
            "Header should report 2 samples"
        );
        assert_eq!(
            game_file.header.version, FORMAT_VERSION,
            "Header should have correct version"
        );
        assert_eq!(
            game_file.samples.len(), 2,
            "Should have 2 samples in file"
        );

        // Clean up
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    // ========================================================================
    // 10. Non-starting position roundtrips correctly
    // ========================================================================

    #[test]
    fn non_starting_position_roundtrip() {
        // Create a board from a FEN after 1. e4
        let board = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        )
        .expect("Valid FEN");

        let sample = TrainingSample {
            board,
            policy: vec![
                (Move::new(Square::E7, Square::E5), 0.7),
                (Move::new(Square::D7, Square::D5), 0.3),
            ],
            value: -0.5,
        };

        let serialized = serialize_sample(&sample);
        let deserialized = deserialize_sample(&serialized);

        assert_eq!(
            sample.board.to_fen(),
            deserialized.board.to_fen(),
            "Non-starting FEN should roundtrip"
        );
        assert_eq!(sample.value, deserialized.value);
        assert_eq!(sample.policy.len(), deserialized.policy.len());

        // Verify the moves are the same
        for (i, ((orig_mv, _), (deser_mv, _))) in sample
            .policy
            .iter()
            .zip(deserialized.policy.iter())
            .enumerate()
        {
            assert_eq!(
                orig_mv.from, deser_mv.from,
                "Move {}: from mismatch",
                i
            );
            assert_eq!(
                orig_mv.to, deser_mv.to,
                "Move {}: to mismatch",
                i
            );
        }
    }
}
