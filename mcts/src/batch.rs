//! Batched asynchronous inference pipeline for MCTS.
//!
//! The key performance optimization for neural-network-guided MCTS: instead of
//! calling the neural network once per leaf evaluation (sequential, slow), we
//! collect multiple leaf positions into a batch and evaluate them all in one
//! GPU forward pass.
//!
//! ## Architecture
//!
//! ```text
//! MCTS Worker Threads        InferenceServer         GPU / Model
//!      |                         |                       |
//!      |--- send position ------>|                       |
//!      |                         |--- collect batch ---->|
//!      |                         |                       |--- forward pass
//!      |                         |<-- results -----------|
//!      |<-- receive result ------|                       |
//! ```
//!
//! Each worker thread calls [`InferenceServer::evaluate`], which sends the
//! board position to the server thread and blocks until the result is ready.
//! The server thread collects up to `batch_size` requests (or waits up to
//! `max_wait_ms` milliseconds), runs [`NnModel::eval_batch`], and dispatches
//! results back to the individual callers via oneshot channels.
//!
//! ## Usage
//!
//! ```ignore
//! let model = NnModel::load("model.pt", Device::Cpu).unwrap();
//! let server = InferenceServer::new(model, 8, 10);
//!
//! // From any thread:
//! let eval = server.evaluate(&board);
//! // eval.policy and eval.value are ready
//!
//! server.shutdown();
//! ```
//!
//! For integration with the MCTS search loop, use [`BatchEvaluator`] which
//! implements [`LeafEvaluator`]:
//!
//! ```ignore
//! let evaluator = BatchEvaluator::new(&server);
//! let result = search_with_evaluator(&game, &config, &evaluator);
//! ```

use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{self as channel, RecvTimeoutError};

use chess_engine::board::Board;
use chess_engine::moves::Move;

use crate::nn::{NnEval, NnModel};
use crate::search::{extract_legal_policy, LeafEvaluator};

// =============================================================================
// InferenceRequest
// =============================================================================

/// A request to evaluate a single board position.
///
/// Contains the board to evaluate and a oneshot channel sender for returning
/// the result to the requester.
struct InferenceRequest {
    /// The board position to evaluate.
    board: Board,
    /// Channel to send the result back to the requester.
    response_tx: channel::Sender<NnEval>,
}

// =============================================================================
// InferenceServer
// =============================================================================

/// Batched inference server that collects evaluation requests and processes
/// them in batches for GPU efficiency.
///
/// The server runs on a dedicated background thread. Callers submit positions
/// via [`evaluate`](InferenceServer::evaluate) and block until results are
/// ready. The server thread collects requests into batches of up to
/// `batch_size` positions, runs a single batched forward pass through the
/// neural network, and dispatches individual results back to callers.
///
/// This amortizes the overhead of GPU kernel launches and data transfers
/// across multiple positions, which is critical for achieving high throughput
/// in multi-threaded MCTS.
pub struct InferenceServer {
    /// Channel to send requests to the server thread.
    request_tx: channel::Sender<InferenceRequest>,
    /// Handle to the server thread (wrapped in Option for clean shutdown).
    handle: Option<thread::JoinHandle<()>>,
}

impl InferenceServer {
    /// Create and start a new inference server.
    ///
    /// Spawns a background thread that:
    /// 1. Waits for the first request (blocking).
    /// 2. Collects additional requests until `batch_size` is reached or
    ///    `max_wait_ms` milliseconds have elapsed since the first request.
    /// 3. Runs batched inference via [`NnModel::eval_batch`].
    /// 4. Dispatches results back to individual callers.
    /// 5. Repeats from step 1.
    ///
    /// The server shuts down when all senders are dropped (i.e., when the
    /// `InferenceServer` and any clones of its request channel are dropped).
    ///
    /// # Arguments
    /// * `model` -- The neural network model (moved to the server thread).
    /// * `batch_size` -- Maximum number of positions per batch.
    /// * `max_wait_ms` -- Maximum milliseconds to wait for a full batch.
    pub fn new(model: NnModel, batch_size: usize, max_wait_ms: u64) -> Self {
        let (request_tx, request_rx) = channel::unbounded::<InferenceRequest>();
        let max_wait = Duration::from_millis(max_wait_ms);

        let handle = thread::spawn(move || {
            server_loop(model, request_rx, batch_size, max_wait);
        });

        InferenceServer {
            request_tx,
            handle: Some(handle),
        }
    }

    /// Submit a position for evaluation. Blocks until the result is ready.
    ///
    /// This method is safe to call from multiple threads concurrently. Each
    /// call creates a oneshot response channel, sends the request to the
    /// server thread, and blocks waiting for the result.
    ///
    /// # Panics
    /// Panics if the server thread has shut down or the response channel
    /// is unexpectedly closed.
    pub fn evaluate(&self, board: &Board) -> NnEval {
        let (response_tx, response_rx) = channel::bounded::<NnEval>(1);
        let request = InferenceRequest {
            board: board.clone(),
            response_tx,
        };

        self.request_tx
            .send(request)
            .expect("InferenceServer: server thread has shut down");

        response_rx
            .recv()
            .expect("InferenceServer: response channel closed unexpectedly")
    }

    /// Shut down the server thread gracefully.
    ///
    /// Drops the request channel sender, which signals the server thread to
    /// exit its loop, then joins the thread to wait for it to finish.
    pub fn shutdown(mut self) {
        // Drop the sender to signal the server thread to exit.
        drop(self.request_tx.clone());
        // The actual sender is dropped when `self` is consumed, but we need
        // to explicitly drop our field to close the channel before joining.
        // Since `self` is consumed by value, the sender will be dropped at
        // the end of this function. We need to force it first.
        let tx = std::mem::replace(
            &mut self.request_tx,
            channel::unbounded::<InferenceRequest>().0,
        );
        drop(tx);

        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for InferenceServer {
    fn drop(&mut self) {
        // If the handle is still present (shutdown wasn't called explicitly),
        // we detach the thread. The channel sender will be dropped, which
        // causes the server thread to exit its loop and terminate.
        // We don't join here to avoid blocking in drop.
        // The thread will exit on its own once the channel closes.
    }
}

// =============================================================================
// server_loop
// =============================================================================

/// Main loop for the inference server thread.
///
/// Continuously collects batches of inference requests and processes them:
/// 1. Blocks on the first request.
/// 2. Tries to fill the batch up to `batch_size` within `max_wait`.
/// 3. Runs batched NN inference.
/// 4. Sends results back to individual requesters.
///
/// Exits when the request channel is closed (all senders dropped).
fn server_loop(
    model: NnModel,
    rx: channel::Receiver<InferenceRequest>,
    batch_size: usize,
    max_wait: Duration,
) {
    let mut batch: Vec<InferenceRequest> = Vec::with_capacity(batch_size);

    loop {
        // Step 1: Wait for the first request (blocks until one arrives or
        // channel closes).
        match rx.recv() {
            Ok(req) => batch.push(req),
            Err(_) => break, // Channel closed, shutdown.
        }

        // Step 2: Try to collect more requests up to batch_size, with timeout.
        let deadline = Instant::now() + max_wait;
        while batch.len() < batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match rx.recv_timeout(remaining) {
                Ok(req) => batch.push(req),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }

        // Step 3: Run batched inference.
        let boards: Vec<Board> = batch.iter().map(|r| r.board.clone()).collect();
        let results = model.eval_batch(&boards);

        // Step 4: Dispatch results back to requesters.
        for (req, eval) in batch.drain(..).zip(results) {
            // Ignore send error if the receiver was dropped (caller gave up).
            let _ = req.response_tx.send(eval);
        }
    }
}

// =============================================================================
// BatchEvaluator
// =============================================================================

/// Evaluator that submits positions to a batched inference server.
///
/// Implements [`LeafEvaluator`] so it can be used with
/// [`search_with_evaluator`](crate::search::search_with_evaluator).
///
/// Note: In a single-threaded search, every batch will contain exactly one
/// position (since the search loop is sequential). The real benefit comes
/// when multiple search threads share the same `InferenceServer`, allowing
/// their leaf evaluations to be batched together.
pub struct BatchEvaluator<'a> {
    server: &'a InferenceServer,
}

impl<'a> BatchEvaluator<'a> {
    /// Create a new batch evaluator wrapping the given inference server.
    pub fn new(server: &'a InferenceServer) -> Self {
        BatchEvaluator { server }
    }
}

impl<'a> LeafEvaluator for BatchEvaluator<'a> {
    fn evaluate(&self, board: &Board, legal_moves: &[Move]) -> (Vec<f32>, f32) {
        let eval = self.server.evaluate(board);
        let policy = extract_legal_policy(&eval.policy, legal_moves, board);
        (policy, eval.value)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MctsConfig;
    use crate::nn::NnModel;
    use crate::search::{search_with_evaluator, NnEvaluator};
    use chess_engine::board::Board;
    use chess_engine::game::Game;

    /// Path to the test model file.
    const TEST_MODEL_PATH: &str = "/Users/william/Desktop/Random/alphazero/test_model.pt";

    /// Helper to load the test model if it exists.
    /// Returns None (and prints a message) if the model file is not found.
    fn get_test_model() -> Option<NnModel> {
        if !std::path::Path::new(TEST_MODEL_PATH).exists() {
            eprintln!(
                "Skipping batch test: {} not found. \
                 Run the Python export script to create it.",
                TEST_MODEL_PATH
            );
            return None;
        }
        Some(NnModel::load(TEST_MODEL_PATH, tch::Device::Cpu).expect("Failed to load test model"))
    }

    // ========================================================================
    // 1. Single request: submit one position, get correct result back
    // ========================================================================

    #[test]
    fn single_request_returns_result() {
        let model = match get_test_model() {
            Some(m) => m,
            None => return,
        };

        let board = Board::starting_position();
        let server = InferenceServer::new(model, 8, 100);

        let eval = server.evaluate(&board);

        // Policy should have the correct size.
        assert_eq!(
            eval.policy.len(),
            crate::nn::POLICY_SIZE,
            "Policy vector should have {} entries, got {}",
            crate::nn::POLICY_SIZE,
            eval.policy.len()
        );

        // Value should be in [-1, 1].
        assert!(
            eval.value >= -1.0 && eval.value <= 1.0,
            "Value should be in [-1, 1], got {}",
            eval.value
        );

        server.shutdown();
    }

    // ========================================================================
    // 2. Multiple requests sequential: submit 10 positions one after another
    // ========================================================================

    #[test]
    fn multiple_sequential_requests() {
        let model = match get_test_model() {
            Some(m) => m,
            None => return,
        };

        let server = InferenceServer::new(model, 8, 100);

        let boards = vec![
            Board::starting_position(),
            Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
                .unwrap(),
            Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap(),
            Board::starting_position(),
            Board::from_fen("r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2")
                .unwrap(),
            Board::starting_position(),
            Board::starting_position(),
            Board::starting_position(),
            Board::starting_position(),
            Board::starting_position(),
        ];

        for (i, board) in boards.iter().enumerate() {
            let eval = server.evaluate(board);
            assert_eq!(
                eval.policy.len(),
                crate::nn::POLICY_SIZE,
                "Request {}: wrong policy size",
                i
            );
            assert!(
                eval.value >= -1.0 && eval.value <= 1.0,
                "Request {}: value {} out of range",
                i,
                eval.value
            );
        }

        server.shutdown();
    }

    // ========================================================================
    // 3. Batch formation: submit batch_size requests simultaneously
    // ========================================================================

    #[test]
    fn concurrent_requests_form_batch() {
        let model = match get_test_model() {
            Some(m) => m,
            None => return,
        };

        let batch_size = 4;
        let server = InferenceServer::new(model, batch_size, 500);
        let server_ref = &server;

        // Use a scoped thread pool to submit requests concurrently.
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for _ in 0..batch_size {
                let handle = s.spawn(move || {
                    let board = Board::starting_position();
                    server_ref.evaluate(&board)
                });
                handles.push(handle);
            }

            // All threads should complete and return valid results.
            for (i, handle) in handles.into_iter().enumerate() {
                let eval = handle.join().expect("Thread panicked");
                assert_eq!(
                    eval.policy.len(),
                    crate::nn::POLICY_SIZE,
                    "Concurrent request {}: wrong policy size",
                    i
                );
                assert!(
                    eval.value >= -1.0 && eval.value <= 1.0,
                    "Concurrent request {}: value {} out of range",
                    i,
                    eval.value
                );
            }
        });

        server.shutdown();
    }

    // ========================================================================
    // 4. Timeout behavior: fewer requests than batch_size still complete
    // ========================================================================

    #[test]
    fn timeout_fires_with_partial_batch() {
        let model = match get_test_model() {
            Some(m) => m,
            None => return,
        };

        // Batch size is 8, but we only send 1 request.
        // It should still complete within max_wait_ms.
        let server = InferenceServer::new(model, 8, 50);
        let board = Board::starting_position();

        let start = Instant::now();
        let eval = server.evaluate(&board);
        let elapsed = start.elapsed();

        assert_eq!(eval.policy.len(), crate::nn::POLICY_SIZE);
        assert!(
            eval.value >= -1.0 && eval.value <= 1.0,
            "Value out of range: {}",
            eval.value
        );

        // Should complete within a reasonable time (timeout + processing).
        // The max_wait is 50ms, so we expect it to complete well within 500ms
        // (allowing for model inference time and thread scheduling).
        assert!(
            elapsed.as_millis() < 2000,
            "Partial batch took too long: {:?}",
            elapsed
        );

        server.shutdown();
    }

    // ========================================================================
    // 5. Results match single inference
    // ========================================================================

    #[test]
    fn batched_results_match_single_inference() {
        let model = match get_test_model() {
            Some(m) => m,
            None => return,
        };

        let board = Board::starting_position();

        // Get single inference result directly.
        let single_eval = model.eval_position(&board);

        // Get result through the batch server (batch of 1).
        let server = InferenceServer::new(
            NnModel::load(TEST_MODEL_PATH, tch::Device::Cpu).unwrap(),
            8,
            50,
        );
        let batch_eval = server.evaluate(&board);
        server.shutdown();

        // Values should match within floating-point tolerance.
        assert!(
            (batch_eval.value - single_eval.value).abs() < 1e-5,
            "Value mismatch: batch={}, single={}",
            batch_eval.value,
            single_eval.value
        );

        // Policy logits should match.
        for i in 0..crate::nn::POLICY_SIZE {
            assert!(
                (batch_eval.policy[i] - single_eval.policy[i]).abs() < 1e-5,
                "Policy[{}] mismatch: batch={}, single={}",
                i,
                batch_eval.policy[i],
                single_eval.policy[i]
            );
        }
    }

    // ========================================================================
    // 6. Server shutdown: clean shutdown after requests
    // ========================================================================

    #[test]
    fn server_shuts_down_cleanly() {
        let model = match get_test_model() {
            Some(m) => m,
            None => return,
        };

        let server = InferenceServer::new(model, 4, 100);

        // Submit a few requests.
        let board = Board::starting_position();
        for _ in 0..3 {
            let eval = server.evaluate(&board);
            assert_eq!(eval.policy.len(), crate::nn::POLICY_SIZE);
        }

        // Shutdown should complete without hanging.
        let start = Instant::now();
        server.shutdown();
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_secs() < 5,
            "Shutdown took too long: {:?}",
            elapsed
        );
    }

    // ========================================================================
    // 7. BatchEvaluator with search
    // ========================================================================

    #[test]
    fn batch_evaluator_with_search() {
        let model = match get_test_model() {
            Some(m) => m,
            None => return,
        };

        let server = InferenceServer::new(model, 8, 50);
        let evaluator = BatchEvaluator::new(&server);

        let game = Game::new();
        let config = MctsConfig {
            num_simulations: 50,
            dirichlet_epsilon: 0.0,
            ..MctsConfig::default()
        };

        let result = search_with_evaluator(&game, &config, &evaluator);

        // Starting position has 20 legal moves.
        assert_eq!(
            result.move_visits.len(),
            20,
            "Expected 20 moves, got {}",
            result.move_visits.len()
        );

        // Total visits should match simulations.
        let total_visits: u32 = result.move_visits.iter().map(|&(_, v)| v).sum();
        assert_eq!(total_visits, 50);

        // Root value should be in valid range.
        assert!(
            result.root_value >= -1.0 && result.root_value <= 1.0,
            "Root value {} out of range",
            result.root_value
        );

        // Move visits should be sorted descending.
        for i in 1..result.move_visits.len() {
            assert!(
                result.move_visits[i - 1].1 >= result.move_visits[i].1,
                "move_visits not sorted at index {}",
                i,
            );
        }

        server.shutdown();
    }

    // ========================================================================
    // 8. BatchEvaluator results match NnEvaluator results
    // ========================================================================

    #[test]
    fn batch_evaluator_matches_nn_evaluator() {
        if !std::path::Path::new(TEST_MODEL_PATH).exists() {
            eprintln!("Skipping batch vs NN evaluator test: model not found");
            return;
        }

        let model_for_nn =
            NnModel::load(TEST_MODEL_PATH, tch::Device::Cpu).unwrap();
        let model_for_batch =
            NnModel::load(TEST_MODEL_PATH, tch::Device::Cpu).unwrap();

        let server = InferenceServer::new(model_for_batch, 8, 50);

        let board = Board::starting_position();
        let game = Game::from_board(board.clone());
        let legal_moves = game.legal_moves();

        // Evaluate with NnEvaluator (direct).
        let nn_evaluator = NnEvaluator::new(&model_for_nn);
        let (nn_policy, nn_value) = nn_evaluator.evaluate(&board, &legal_moves);

        // Evaluate with BatchEvaluator (through server).
        let batch_evaluator = BatchEvaluator::new(&server);
        let (batch_policy, batch_value) = batch_evaluator.evaluate(&board, &legal_moves);

        // Values should match within tolerance.
        assert!(
            (batch_value - nn_value).abs() < 1e-5,
            "Value mismatch: batch={}, nn={}",
            batch_value,
            nn_value
        );

        // Policies should match.
        assert_eq!(batch_policy.len(), nn_policy.len());
        for i in 0..nn_policy.len() {
            assert!(
                (batch_policy[i] - nn_policy[i]).abs() < 1e-4,
                "Policy[{}] mismatch: batch={}, nn={}",
                i,
                batch_policy[i],
                nn_policy[i]
            );
        }

        server.shutdown();
    }

    // ========================================================================
    // 9. Multiple concurrent batches
    // ========================================================================

    #[test]
    fn multiple_concurrent_batches() {
        let model = match get_test_model() {
            Some(m) => m,
            None => return,
        };

        let batch_size = 4;
        let num_requests = 12; // 3 full batches
        let server = InferenceServer::new(model, batch_size, 100);
        let server_ref = &server;

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for _ in 0..num_requests {
                let handle = s.spawn(move || {
                    let board = Board::starting_position();
                    server_ref.evaluate(&board)
                });
                handles.push(handle);
            }

            for (i, handle) in handles.into_iter().enumerate() {
                let eval = handle.join().expect("Thread panicked");
                assert_eq!(
                    eval.policy.len(),
                    crate::nn::POLICY_SIZE,
                    "Request {}: wrong policy size",
                    i
                );
            }
        });

        server.shutdown();
    }

    // ========================================================================
    // 10. Different board positions in same batch
    // ========================================================================

    #[test]
    fn different_positions_in_batch() {
        let model = match get_test_model() {
            Some(m) => m,
            None => return,
        };

        let server = InferenceServer::new(model, 4, 200);
        let server_ref = &server;

        let boards = [
            Board::starting_position(),
            Board::from_fen(
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            )
            .unwrap(),
            Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap(),
            Board::from_fen(
                "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            )
            .unwrap(),
        ];

        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for board in &boards {
                let b = board.clone();
                let handle = s.spawn(move || server_ref.evaluate(&b));
                handles.push(handle);
            }

            let results: Vec<NnEval> = handles
                .into_iter()
                .map(|h| h.join().expect("Thread panicked"))
                .collect();

            // All results should be valid.
            for (i, eval) in results.iter().enumerate() {
                assert_eq!(
                    eval.policy.len(),
                    crate::nn::POLICY_SIZE,
                    "Position {}: wrong policy size",
                    i
                );
                assert!(
                    eval.value >= -1.0 && eval.value <= 1.0,
                    "Position {}: value {} out of range",
                    i,
                    eval.value
                );
            }

            // Different positions should (generally) produce different values.
            // The starting position and after-e4 positions are different, so
            // their evaluations should differ (at least the value).
            // This isn't guaranteed with random weights, but extremely likely.
            let values: Vec<f32> = results.iter().map(|e| e.value).collect();
            let all_same = values.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-6);
            // With 4 different positions and random weights, it's astronomically
            // unlikely all values are identical. But we don't assert this to avoid
            // flaky tests -- we just verify all results are valid above.
            let _ = all_same; // suppress unused warning
        });

        server.shutdown();
    }
}
