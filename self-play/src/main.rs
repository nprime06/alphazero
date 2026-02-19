//! Self-play worker binary for the AlphaZero training pipeline.
//!
//! Generates training data by playing games against itself using MCTS guided
//! by a neural network. Each game produces training samples (board positions
//! with MCTS policy targets and game outcome values) that are written to a
//! replay buffer on disk.
//!
//! # Usage
//!
//! ```text
//! self-play --model model.pt --games 100 --output ./data/ --sims 800
//! ```
//!
//! For multi-game parallelism (recommended on GPU nodes):
//!
//! ```text
//! self-play --model model.pt --games 250 --parallel-games 16 --batch-size 32
//! ```
//!
//! For within-game parallel MCTS only:
//!
//! ```text
//! self-play --model model.pt --games 100 --threads 4 --batch-size 8
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use clap::Parser;
use tch::Device;

use mcts::batch::InferenceServer;
use mcts::config::MctsConfig;
use mcts::nn::NnModel;
use mcts::search::NnEvaluator;
use self_play::buffer::ReplayBuffer;
use self_play::data::extract_samples;
use self_play::game::{play_game, play_game_parallel, SelfPlayConfig};

use chess_engine::game::GameResult;

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Parser)]
#[command(name = "self-play")]
#[command(about = "AlphaZero self-play game generator")]
struct Args {
    /// Path to TorchScript model file
    #[arg(long)]
    model: PathBuf,

    /// Number of games to generate
    #[arg(long, default_value = "100")]
    games: u32,

    /// Output directory for game files
    #[arg(long, default_value = "./data")]
    output: PathBuf,

    /// MCTS simulations per move
    #[arg(long, default_value = "800")]
    sims: u32,

    /// Max games in replay buffer
    #[arg(long, default_value = "100000")]
    buffer_capacity: usize,

    /// Move number to switch from T=1 to T=0
    #[arg(long, default_value = "30")]
    temperature_threshold: u32,

    /// Max moves per game
    #[arg(long, default_value = "512")]
    max_moves: u32,

    /// Disable Dirichlet noise at root
    #[arg(long)]
    no_noise: bool,

    /// MCTS exploration constant
    #[arg(long, default_value = "2.5")]
    c_puct: f32,

    /// Number of search threads for parallel MCTS (per game)
    #[arg(long, default_value = "1")]
    threads: u32,

    /// Number of games to play concurrently
    #[arg(long, default_value = "1")]
    parallel_games: u32,

    /// Batch size for NN inference
    #[arg(long, default_value = "8")]
    batch_size: usize,
}

// =============================================================================
// Result tracking
// =============================================================================

/// Tracks game result statistics across all games.
struct ResultStats {
    white_wins: u32,
    black_wins: u32,
    draws: u32,
}

impl ResultStats {
    fn new() -> Self {
        Self {
            white_wins: 0,
            black_wins: 0,
            draws: 0,
        }
    }

    fn record(&mut self, result: GameResult) {
        match result {
            GameResult::WhiteWins => self.white_wins += 1,
            GameResult::BlackWins => self.black_wins += 1,
            _ => self.draws += 1,
        }
    }
}

// =============================================================================
// Result formatting
// =============================================================================

/// Format a GameResult as a short string for progress output.
fn result_str(result: GameResult) -> &'static str {
    match result {
        GameResult::WhiteWins => "WhiteWins",
        GameResult::BlackWins => "BlackWins",
        GameResult::DrawStalemate => "Draw(Stalemate)",
        GameResult::DrawRepetition => "Draw(Repetition)",
        GameResult::DrawFiftyMoveRule => "Draw(50-move)",
        GameResult::DrawInsufficientMaterial => "Draw(Material)",
        GameResult::Ongoing => "Ongoing",
    }
}

/// Format a duration as "Xm Ys" or "X.Ys".
fn format_duration(secs: f64) -> String {
    if secs >= 60.0 {
        let minutes = (secs / 60.0).floor() as u64;
        let remaining = secs - (minutes as f64 * 60.0);
        format!("{}m {:.0}s", minutes, remaining)
    } else {
        format!("{:.1}s", secs)
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let args = Args::parse();

    // Determine device: use CUDA if available, otherwise CPU.
    let device = if tch::Cuda::is_available() {
        eprintln!("[INFO] CUDA available, using GPU");
        Device::Cuda(0)
    } else {
        eprintln!("[INFO] CUDA not available, using CPU");
        Device::Cpu
    };

    // Load the TorchScript model.
    eprintln!(
        "[INFO] Loading model from: {}",
        args.model.display()
    );
    let model = NnModel::load(
        args.model.to_str().expect("Model path is not valid UTF-8"),
        device,
    )
    .expect("Failed to load TorchScript model");

    // Build MCTS config.
    let mcts_config = MctsConfig {
        num_simulations: args.sims,
        c_puct: args.c_puct,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: if args.no_noise { 0.0 } else { 0.25 },
        ..MctsConfig::default()
    };

    // Build self-play config.
    let self_play_config = SelfPlayConfig {
        mcts_config,
        temperature_threshold: args.temperature_threshold,
        max_moves: args.max_moves,
        add_noise: !args.no_noise,
    };

    // Create the replay buffer.
    let buffer = ReplayBuffer::new(args.output.clone(), args.buffer_capacity)
        .expect("Failed to create replay buffer directory");

    eprintln!(
        "[INFO] Config: games={}, sims={}, threads={}, parallel_games={}, batch_size={}, c_puct={}, temp_threshold={}, max_moves={}, noise={}",
        args.games,
        args.sims,
        args.threads,
        args.parallel_games,
        args.batch_size,
        args.c_puct,
        args.temperature_threshold,
        args.max_moves,
        !args.no_noise,
    );
    eprintln!("[INFO] Output directory: {}", args.output.display());
    eprintln!();

    // Track statistics (atomic for thread-safe access).
    let stats = Mutex::new(ResultStats::new());
    let total_moves = AtomicU64::new(0);
    let total_samples = AtomicU64::new(0);
    let session_start = Instant::now();

    if args.parallel_games > 1 || args.threads > 1 {
        // =====================================================================
        // Batched mode: use InferenceServer for GPU batching.
        //
        // Two sub-modes:
        // - parallel_games > 1: multiple games concurrently (main speedup)
        // - threads > 1 only: single game with multi-threaded MCTS
        // =====================================================================

        // Load model for the inference server (it takes ownership).
        let server_model = NnModel::load(
            args.model.to_str().expect("Model path is not valid UTF-8"),
            device,
        )
        .expect("Failed to load model for inference server");

        // Scale max_wait based on parallelism: more concurrent games means
        // requests arrive faster, so we can use a shorter wait before batching.
        // With parallel_games > 1, batch_size should equal parallel_games * threads
        // so the batch fills immediately without needing a long timeout.  Keep
        // max_wait at 1ms as a safety fallback for thread scheduling jitter.
        let max_wait_ms: u64 = if args.parallel_games > 1 { 1 } else { 50 };
        let server = InferenceServer::new(server_model, args.batch_size, max_wait_ms);

        if args.parallel_games > 1 {
            // -----------------------------------------------------------------
            // Multi-game parallel mode: N games running concurrently
            // -----------------------------------------------------------------
            let total_workers = args.parallel_games * args.threads;
            eprintln!(
                "[INFO] Running {} games in parallel ({} threads/game, {} total workers, batch_size={})",
                args.parallel_games, args.threads, total_workers, args.batch_size
            );

            // Work-stealing counter: each thread grabs the next game index.
            let game_counter = AtomicU32::new(0);

            std::thread::scope(|s| {
                for _ in 0..args.parallel_games {
                    s.spawn(|| {
                        loop {
                            let game_idx = game_counter.fetch_add(1, Ordering::Relaxed);
                            if game_idx >= args.games {
                                break;
                            }

                            let game_start = Instant::now();

                            let record = play_game_parallel(
                                &self_play_config,
                                &server,
                                args.threads as usize,
                            );
                            let samples = extract_samples(&record);

                            buffer
                                .add_game(&samples)
                                .expect("Failed to write game to replay buffer");

                            let game_time = game_start.elapsed().as_secs_f64();
                            let num_moves = record.num_moves;
                            let num_samples = samples.len();

                            total_moves.fetch_add(num_moves as u64, Ordering::Relaxed);
                            total_samples.fetch_add(num_samples as u64, Ordering::Relaxed);
                            {
                                let mut s = stats.lock().unwrap();
                                s.record(record.result);
                            }

                            // Compute approximate progress.
                            let completed = (game_idx + 1).min(args.games);
                            let elapsed_total = session_start.elapsed().as_secs_f64();
                            let games_per_hour = if elapsed_total > 0.0 {
                                completed as f64 / elapsed_total * 3600.0
                            } else {
                                0.0
                            };

                            println!(
                                "[Game {}/{}] Moves: {} | Result: {} | Samples: {} | Time: {} | Games/hr: {:.0}",
                                completed,
                                args.games,
                                num_moves,
                                result_str(record.result),
                                num_samples,
                                format_duration(game_time),
                                games_per_hour,
                            );
                        }
                    });
                }
            });
        } else {
            // -----------------------------------------------------------------
            // Single game, multi-threaded MCTS
            // -----------------------------------------------------------------
            eprintln!(
                "[INFO] Running in parallel MCTS mode ({} threads, batch_size={})",
                args.threads, args.batch_size
            );

            for game_idx in 0..args.games {
                let game_start = Instant::now();

                let record = play_game_parallel(
                    &self_play_config,
                    &server,
                    args.threads as usize,
                );
                let samples = extract_samples(&record);

                buffer
                    .add_game(&samples)
                    .expect("Failed to write game to replay buffer");

                let game_time = game_start.elapsed().as_secs_f64();
                let elapsed_total = session_start.elapsed().as_secs_f64();
                let games_per_hour = if elapsed_total > 0.0 {
                    (game_idx as f64 + 1.0) / elapsed_total * 3600.0
                } else {
                    0.0
                };

                stats.lock().unwrap().record(record.result);
                total_moves.fetch_add(record.num_moves as u64, Ordering::Relaxed);
                total_samples.fetch_add(samples.len() as u64, Ordering::Relaxed);

                println!(
                    "[Game {}/{}] Moves: {} | Result: {} | Samples: {} | Time: {} | Games/hr: {:.0}",
                    game_idx + 1,
                    args.games,
                    record.num_moves,
                    result_str(record.result),
                    samples.len(),
                    format_duration(game_time),
                    games_per_hour,
                );
            }
        }

        server.shutdown();
    } else {
        // =====================================================================
        // Single-threaded mode: use NnEvaluator + search_with_evaluator
        // =====================================================================
        eprintln!("[INFO] Running in single-threaded mode");

        let evaluator = NnEvaluator::new(&model);

        for game_idx in 0..args.games {
            let game_start = Instant::now();

            let record = play_game(&self_play_config, &evaluator);
            let samples = extract_samples(&record);

            buffer
                .add_game(&samples)
                .expect("Failed to write game to replay buffer");

            let game_time = game_start.elapsed().as_secs_f64();
            let elapsed_total = session_start.elapsed().as_secs_f64();
            let games_per_hour = if elapsed_total > 0.0 {
                (game_idx as f64 + 1.0) / elapsed_total * 3600.0
            } else {
                0.0
            };

            stats.lock().unwrap().record(record.result);
            total_moves.fetch_add(record.num_moves as u64, Ordering::Relaxed);
            total_samples.fetch_add(samples.len() as u64, Ordering::Relaxed);

            println!(
                "[Game {}/{}] Moves: {} | Result: {} | Samples: {} | Time: {} | Games/hr: {:.0}",
                game_idx + 1,
                args.games,
                record.num_moves,
                result_str(record.result),
                samples.len(),
                format_duration(game_time),
                games_per_hour,
            );
        }
    }

    // Print summary.
    let final_stats = stats.into_inner().unwrap();
    let total_time = session_start.elapsed().as_secs_f64();
    let total_moves_val = total_moves.load(Ordering::Relaxed);
    let total_samples_val = total_samples.load(Ordering::Relaxed);
    let avg_length = if args.games > 0 {
        total_moves_val as f64 / args.games as f64
    } else {
        0.0
    };

    println!();
    println!("=== Summary ===");
    println!(
        "Games: {} | Avg length: {:.1} | W/D/L: {}/{}/{} | Total samples: {} | Total time: {}",
        args.games,
        avg_length,
        final_stats.white_wins,
        final_stats.draws,
        final_stats.black_wins,
        total_samples_val,
        format_duration(total_time),
    );
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // CLI argument parsing tests
    // ========================================================================

    #[test]
    fn parse_minimal_args() {
        let args = Args::try_parse_from(["self-play", "--model", "test.pt"]);
        assert!(args.is_ok(), "Should parse with only --model");

        let args = args.unwrap();
        assert_eq!(args.model, PathBuf::from("test.pt"));
        assert_eq!(args.games, 100);
        assert_eq!(args.output, PathBuf::from("./data"));
        assert_eq!(args.sims, 800);
        assert_eq!(args.buffer_capacity, 100000);
        assert_eq!(args.temperature_threshold, 30);
        assert_eq!(args.max_moves, 512);
        assert!(!args.no_noise);
        assert_eq!(args.c_puct, 2.5);
        assert_eq!(args.threads, 1);
        assert_eq!(args.parallel_games, 1);
        assert_eq!(args.batch_size, 8);
    }

    #[test]
    fn parse_all_args() {
        let args = Args::try_parse_from([
            "self-play",
            "--model", "model.pt",
            "--games", "50",
            "--output", "/tmp/data",
            "--sims", "400",
            "--buffer-capacity", "5000",
            "--temperature-threshold", "15",
            "--max-moves", "256",
            "--no-noise",
            "--c-puct", "3.0",
            "--threads", "4",
            "--parallel-games", "8",
            "--batch-size", "16",
        ]);
        assert!(args.is_ok(), "Should parse all args: {:?}", args.err());

        let args = args.unwrap();
        assert_eq!(args.model, PathBuf::from("model.pt"));
        assert_eq!(args.games, 50);
        assert_eq!(args.output, PathBuf::from("/tmp/data"));
        assert_eq!(args.sims, 400);
        assert_eq!(args.buffer_capacity, 5000);
        assert_eq!(args.temperature_threshold, 15);
        assert_eq!(args.max_moves, 256);
        assert!(args.no_noise);
        assert_eq!(args.c_puct, 3.0);
        assert_eq!(args.threads, 4);
        assert_eq!(args.parallel_games, 8);
        assert_eq!(args.batch_size, 16);
    }

    #[test]
    fn parse_missing_required_arg_fails() {
        let args = Args::try_parse_from(["self-play"]);
        assert!(args.is_err(), "Should fail without --model");
    }

    #[test]
    fn parse_no_noise_flag() {
        let args_with = Args::try_parse_from([
            "self-play", "--model", "m.pt", "--no-noise",
        ]).unwrap();
        assert!(args_with.no_noise);

        let args_without = Args::try_parse_from([
            "self-play", "--model", "m.pt",
        ]).unwrap();
        assert!(!args_without.no_noise);
    }

    // ========================================================================
    // Helper function tests
    // ========================================================================

    #[test]
    fn format_duration_seconds() {
        assert_eq!(format_duration(12.3), "12.3s");
        assert_eq!(format_duration(0.5), "0.5s");
        assert_eq!(format_duration(59.9), "59.9s");
    }

    #[test]
    fn format_duration_minutes() {
        assert_eq!(format_duration(60.0), "1m 0s");
        assert_eq!(format_duration(123.0), "2m 3s");
        assert_eq!(format_duration(323.0), "5m 23s");
    }

    #[test]
    fn result_str_all_variants() {
        assert_eq!(result_str(GameResult::WhiteWins), "WhiteWins");
        assert_eq!(result_str(GameResult::BlackWins), "BlackWins");
        assert_eq!(result_str(GameResult::DrawStalemate), "Draw(Stalemate)");
        assert_eq!(result_str(GameResult::DrawRepetition), "Draw(Repetition)");
        assert_eq!(result_str(GameResult::DrawFiftyMoveRule), "Draw(50-move)");
        assert_eq!(result_str(GameResult::DrawInsufficientMaterial), "Draw(Material)");
        assert_eq!(result_str(GameResult::Ongoing), "Ongoing");
    }

    #[test]
    fn result_stats_tracking() {
        let mut stats = ResultStats::new();
        assert_eq!(stats.white_wins, 0);
        assert_eq!(stats.black_wins, 0);
        assert_eq!(stats.draws, 0);

        stats.record(GameResult::WhiteWins);
        stats.record(GameResult::WhiteWins);
        stats.record(GameResult::BlackWins);
        stats.record(GameResult::DrawStalemate);
        stats.record(GameResult::DrawRepetition);
        stats.record(GameResult::DrawFiftyMoveRule);

        assert_eq!(stats.white_wins, 2);
        assert_eq!(stats.black_wins, 1);
        assert_eq!(stats.draws, 3);
    }
}
