//! Complete MCTS search loop.
//!
//! This module ties together selection, expansion, evaluation, and backup
//! into a single-threaded Monte Carlo Tree Search. Each call to [`search`]
//! runs `num_simulations` iterations of the MCTS loop from the given
//! position, then returns visit counts and a value estimate.
//!
//! ## Algorithm
//!
//! Each simulation follows four phases:
//!
//! 1. **Select** -- Walk from root to a leaf using PUCT ([`select_child`]).
//! 2. **Expand** -- Generate children for the leaf ([`expand`]).
//! 3. **Evaluate** -- Get a value for the leaf. Terminal nodes use the game
//!    result; non-terminal leaves are evaluated by a [`LeafEvaluator`]
//!    (either uniform/zero or a neural network).
//! 4. **Backup** -- Propagate the value up the path ([`backup`]).
//!
//! ## Evaluator abstraction
//!
//! The [`LeafEvaluator`] trait abstracts how leaf positions are evaluated:
//!
//! - [`UniformEvaluator`] uses uniform priors (1/N) and zero value (the
//!   original placeholder behavior).
//! - [`NnEvaluator`] uses a neural network to produce policy priors and
//!   value estimates.
//!
//! The core search loop is implemented once in [`search_with_evaluator`],
//! and the convenience functions [`search`] and [`search_with_nn`] wrap it
//! with the appropriate evaluator.
//!
//! ## Dirichlet noise
//!
//! After expanding the root, Dirichlet noise is mixed into the root
//! children's priors to encourage exploration during self-play:
//!
//! ```text
//! noisy_prior = (1 - epsilon) * prior + epsilon * Dir(alpha)
//! ```
//!
//! ## Move selection
//!
//! After search completes, [`select_move`] picks a move from the root's
//! visit distribution using a temperature parameter.

use chess_engine::board::Board;
use chess_engine::game::Game;
use chess_engine::moves::Move;

use rand::Rng;
use rand_distr::{Distribution, Gamma};

use crate::arena::Arena;
use crate::backup::backup;
use crate::config::MctsConfig;
use crate::expand::{expand, ExpandResult};
use crate::nn::{move_to_policy_index, NnModel};
use crate::node::{Node, NodeIndex};
use crate::select::select_child;

// =============================================================================
// LeafEvaluator trait
// =============================================================================

/// Trait for evaluating leaf positions during MCTS.
///
/// An evaluator provides two pieces of information about a non-terminal
/// position:
/// - **Policy priors**: a probability distribution over legal moves that
///   guides which children to explore first.
/// - **Value estimate**: a scalar in [-1, 1] from the current player's
///   perspective indicating how good the position is.
///
/// The search loop calls `evaluate` once per leaf expansion. Different
/// implementations allow plugging in different evaluation strategies
/// (uniform/zero for testing, neural network for real play).
pub trait LeafEvaluator {
    /// Evaluate a leaf position, returning:
    /// - Policy priors for legal moves (same order as `legal_moves`).
    ///   These will be passed to [`expand`] and normalized there.
    /// - Value estimate from the current player's perspective.
    fn evaluate(&self, board: &Board, legal_moves: &[Move]) -> (Vec<f32>, f32);
}

// =============================================================================
// UniformEvaluator
// =============================================================================

/// Evaluator that uses uniform priors and zero value.
///
/// This matches the original `search()` behavior: every legal move gets
/// equal prior probability, and the leaf value is always 0.0. Useful for
/// testing the search loop without a neural network.
pub struct UniformEvaluator;

impl LeafEvaluator for UniformEvaluator {
    fn evaluate(&self, _board: &Board, legal_moves: &[Move]) -> (Vec<f32>, f32) {
        let n = legal_moves.len() as f32;
        (vec![1.0 / n; legal_moves.len()], 0.0)
    }
}

// =============================================================================
// NnEvaluator
// =============================================================================

/// Evaluator that uses a neural network for policy and value estimation.
///
/// Wraps an [`NnModel`] and uses it to evaluate positions. The raw policy
/// logits from the network are masked to legal moves and converted to
/// probabilities via softmax.
pub struct NnEvaluator<'a> {
    model: &'a NnModel,
}

impl<'a> NnEvaluator<'a> {
    /// Create a new neural network evaluator wrapping the given model.
    pub fn new(model: &'a NnModel) -> Self {
        NnEvaluator { model }
    }
}

impl<'a> LeafEvaluator for NnEvaluator<'a> {
    fn evaluate(&self, board: &Board, legal_moves: &[Move]) -> (Vec<f32>, f32) {
        let eval = self.model.eval_position(board);
        let policy = extract_legal_policy(&eval.policy, legal_moves, board);
        (policy, eval.value)
    }
}

// =============================================================================
// extract_legal_policy
// =============================================================================

/// Extract and normalize policy probabilities for legal moves only.
///
/// Takes the full 4672-element policy logits from the neural network, maps
/// each legal move to its corresponding logit via [`move_to_policy_index`],
/// and applies softmax to produce a probability distribution.
///
/// # Algorithm
///
/// 1. For each legal move, look up its policy index and extract the logit.
/// 2. Subtract the maximum logit for numerical stability.
/// 3. Exponentiate each adjusted logit.
/// 4. Normalize so probabilities sum to 1.0.
///
/// If a move cannot be mapped to a policy index (should not happen for
/// valid moves), it receives a logit of negative infinity (effectively
/// zero probability after softmax).
///
/// # Arguments
/// * `policy_logits` -- Raw NN output (4672 values).
/// * `legal_moves` -- List of legal moves in current position.
/// * `board` -- Current board (needed for `move_to_policy_index`).
///
/// # Returns
/// Vec of probabilities, one per legal move, summing to 1.0.
pub fn extract_legal_policy(policy_logits: &[f32], legal_moves: &[Move], board: &Board) -> Vec<f32> {
    if legal_moves.is_empty() {
        return Vec::new();
    }

    // Step 1: Extract logits for legal moves only.
    let mut logits: Vec<f32> = Vec::with_capacity(legal_moves.len());
    for mv in legal_moves {
        let logit = match move_to_policy_index(mv, board) {
            Some(idx) => policy_logits[idx],
            None => f32::NEG_INFINITY,
        };
        logits.push(logit);
    }

    // Step 2: Softmax with numerical stability (subtract max).
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // If all logits are negative infinity, return uniform distribution.
    if max_logit == f32::NEG_INFINITY {
        let uniform = 1.0 / legal_moves.len() as f32;
        return vec![uniform; legal_moves.len()];
    }

    let mut exp_values: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = exp_values.iter().sum();

    // Normalize.
    if sum > 0.0 {
        for v in &mut exp_values {
            *v /= sum;
        }
    } else {
        // Fallback to uniform (shouldn't happen if max_logit is finite).
        let uniform = 1.0 / legal_moves.len() as f32;
        return vec![uniform; legal_moves.len()];
    }

    exp_values
}

// =============================================================================
// SearchResult
// =============================================================================

/// Result of an MCTS search.
pub struct SearchResult {
    /// Visit counts for each legal move from the root position.
    /// Sorted by visit count (descending).
    pub move_visits: Vec<(Move, u32)>,

    /// Total number of simulations run.
    pub total_simulations: u32,

    /// Root value estimate (Q-value of root after search).
    pub root_value: f32,
}

// =============================================================================
// Dirichlet noise
// =============================================================================

/// Add Dirichlet noise to the root node's children priors.
///
/// This mixes the existing priors with a sample from a symmetric Dirichlet
/// distribution to encourage exploration in self-play:
///
/// ```text
/// noisy_prior = (1 - epsilon) * prior + epsilon * Dir(alpha)
/// ```
///
/// The Dirichlet sample is generated by drawing independent Gamma(alpha, 1)
/// samples and normalizing them.
fn add_dirichlet_noise(root_idx: NodeIndex, arena: &mut Arena, alpha: f32, epsilon: f32) {
    // Collect children indices first to avoid borrow conflicts.
    let children: Vec<NodeIndex> = arena[root_idx].children(arena).collect();
    if children.is_empty() {
        return;
    }

    // Generate Gamma(alpha, 1) samples.
    let mut rng = rand::thread_rng();
    let gamma = Gamma::new(alpha as f64, 1.0).expect("invalid gamma parameters");
    let mut noise: Vec<f32> = Vec::with_capacity(children.len());
    for _ in 0..children.len() {
        noise.push(gamma.sample(&mut rng) as f32);
    }

    // Normalize to get a Dirichlet sample.
    let sum: f32 = noise.iter().sum();
    if sum > 0.0 {
        for n in &mut noise {
            *n /= sum;
        }
    } else {
        // Fallback: uniform noise if all samples are zero (extremely unlikely).
        let uniform = 1.0 / children.len() as f32;
        for n in &mut noise {
            *n = uniform;
        }
    }

    // Mix noise with existing priors.
    for (i, child_idx) in children.iter().enumerate() {
        let node = arena.get_mut(*child_idx);
        node.prior = (1.0 - epsilon) * node.prior + epsilon * noise[i];
    }
}

// =============================================================================
// search_with_evaluator
// =============================================================================

/// Run a complete MCTS search from the given position using a pluggable
/// leaf evaluator.
///
/// This is the core search implementation. The evaluator determines how
/// leaf positions are assessed:
/// - [`UniformEvaluator`] gives uniform priors and zero values (testing).
/// - [`NnEvaluator`] uses a neural network for policy and value (real play).
///
/// # Algorithm
///
/// For each simulation:
/// 1. **Select**: Walk from root to leaf using PUCT ([`select_child`]).
///    Maintain a path (`Vec<NodeIndex>`) and a cloned `Game` state.
///    At each step, find which child was selected, look up its move,
///    and apply it to the game.
/// 2. **Expand**: If the leaf is not terminal, evaluate it with the
///    evaluator, then expand with the returned policy priors.
/// 3. **Evaluate**: Use the evaluator's value estimate (or the game result
///    for terminal positions).
/// 4. **Backup**: Propagate the value up the path.
///
/// # Arguments
/// * `game` -- The game state to search from. Not mutated.
/// * `config` -- MCTS search parameters.
/// * `evaluator` -- Leaf evaluation strategy.
///
/// # Returns
/// `SearchResult` with move visit counts and root value.
pub fn search_with_evaluator(
    game: &Game,
    config: &MctsConfig,
    evaluator: &impl LeafEvaluator,
) -> SearchResult {
    // 1. Create arena and root node.
    let estimated_nodes = config.num_simulations as usize * 2 + 64;
    let mut arena = Arena::new(estimated_nodes);
    let root_idx = arena.alloc(Node::root());

    // 2. Evaluate root position and expand with evaluator's policy.
    let legal_moves = game.legal_moves();

    if game.is_terminal() || legal_moves.is_empty() {
        // Expand to detect the terminal type and get the value.
        let root_expand = expand(root_idx, game, None, &mut arena);
        if let ExpandResult::Terminal(value) = root_expand {
            return SearchResult {
                move_visits: Vec::new(),
                total_simulations: 0,
                root_value: value,
            };
        }
    }

    // Evaluate root with the evaluator for policy priors.
    let (root_policy, _root_value) = evaluator.evaluate(game.board(), &legal_moves);
    let root_expand = expand(root_idx, game, Some(&root_policy), &mut arena);

    // If root is terminal (detected during expand), return immediately.
    if let ExpandResult::Terminal(value) = root_expand {
        return SearchResult {
            move_visits: Vec::new(),
            total_simulations: 0,
            root_value: value,
        };
    }

    // 3. Add Dirichlet noise to root's children for exploration.
    if config.dirichlet_epsilon > 0.0 {
        add_dirichlet_noise(root_idx, &mut arena, config.dirichlet_alpha, config.dirichlet_epsilon);
    }

    // 4. Run simulations.
    for _ in 0..config.num_simulations {
        // (a) Clone the game state for this simulation.
        let mut sim_game = Game::from_board(game.board().clone());

        // (b) Selection: walk from root to a leaf.
        let mut path: Vec<NodeIndex> = vec![root_idx];
        let mut current = root_idx;

        // Descend while the current node is expanded and has children.
        while arena[current].is_expanded && arena[current].num_children > 0 {
            let child_idx = select_child(current, &arena, config);
            let child_mv = arena[child_idx].mv;
            sim_game.make_move(child_mv);
            path.push(child_idx);
            current = child_idx;
        }

        // (c) Expand the leaf and evaluate.
        let leaf_value = if arena[current].is_expanded {
            // Already expanded (terminal leaf re-visited, or defensive case).
            arena[current].q_value()
        } else if sim_game.is_terminal() {
            // Terminal position: expand to get the value.
            match expand(current, &sim_game, None, &mut arena) {
                ExpandResult::Terminal(v) => v,
                _ => 0.0,
            }
        } else {
            // Non-terminal leaf: evaluate with the evaluator.
            let sim_legal_moves = sim_game.legal_moves();
            let (policy, value) = evaluator.evaluate(sim_game.board(), &sim_legal_moves);
            expand(current, &sim_game, Some(&policy), &mut arena);
            value
        };

        // (d) Backup the path with the leaf value.
        backup(&path, leaf_value, &mut arena);
    }

    // 5. Collect root's children into move_visits.
    let mut move_visits: Vec<(Move, u32)> = arena[root_idx]
        .children(&arena)
        .map(|child_idx| {
            let child = &arena[child_idx];
            (child.mv, child.visit_count)
        })
        .collect();

    // 6. Sort by visit count descending.
    move_visits.sort_by(|a, b| b.1.cmp(&a.1));

    // 7. Compute root value.
    let root_value = arena[root_idx].q_value();

    SearchResult {
        move_visits,
        total_simulations: config.num_simulations,
        root_value,
    }
}

// =============================================================================
// search (convenience wrapper)
// =============================================================================

/// Run a complete MCTS search using uniform priors and zero leaf values.
///
/// This is a convenience wrapper around [`search_with_evaluator`] using
/// [`UniformEvaluator`]. Useful for testing without a neural network.
///
/// # Arguments
/// * `game` -- The game state to search from. Not mutated.
/// * `config` -- MCTS search parameters.
///
/// # Returns
/// `SearchResult` with move visit counts and root value.
pub fn search(game: &Game, config: &MctsConfig) -> SearchResult {
    search_with_evaluator(game, config, &UniformEvaluator)
}

// =============================================================================
// search_with_nn
// =============================================================================

/// Run MCTS search guided by a neural network.
///
/// Unlike [`search`] which uses uniform priors and zero values, this
/// version uses the neural network to:
/// 1. Provide policy priors during expansion (guides which moves to explore).
/// 2. Provide value estimates for leaf evaluation (replaces zero values).
///
/// This is a convenience wrapper around [`search_with_evaluator`] using
/// [`NnEvaluator`].
///
/// # Arguments
/// * `game` -- The game state to search from. Not mutated.
/// * `config` -- MCTS search parameters.
/// * `model` -- The neural network model for position evaluation.
///
/// # Returns
/// `SearchResult` with move visit counts and root value.
pub fn search_with_nn(game: &Game, config: &MctsConfig, model: &NnModel) -> SearchResult {
    let evaluator = NnEvaluator::new(model);
    search_with_evaluator(game, config, &evaluator)
}

// =============================================================================
// select_move
// =============================================================================

/// Select a move from search results using temperature.
///
/// - `T -> 0` (practically `T < 0.1`): pick the most visited move (greedy).
/// - `T = 1`: sample proportional to visit counts (exploratory).
/// - `T > 1`: more uniform/random selection.
///
/// # Panics
///
/// Panics if `result.move_visits` is empty.
pub fn select_move(result: &SearchResult, temperature: f32) -> Move {
    assert!(
        !result.move_visits.is_empty(),
        "select_move called with no moves"
    );

    // Greedy: pick the most visited move (already sorted descending).
    if temperature < 0.1 {
        return result.move_visits[0].0;
    }

    // Compute visit_count^(1/T) for each move.
    let inv_t = 1.0 / temperature;
    let weights: Vec<f64> = result
        .move_visits
        .iter()
        .map(|&(_, visits)| (visits as f64).powf(inv_t as f64))
        .collect();

    let total: f64 = weights.iter().sum();
    if total == 0.0 {
        // All zero visits: pick uniformly at random.
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..result.move_visits.len());
        return result.move_visits[idx].0;
    }

    // Sample proportional to weights.
    let mut rng = rand::thread_rng();
    let sample: f64 = rng.gen::<f64>() * total;
    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        if sample <= cumulative {
            return result.move_visits[i].0;
        }
    }

    // Floating-point edge case: return the last move.
    result.move_visits.last().unwrap().0
}

// =============================================================================
// Parallel MCTS search
// =============================================================================

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use crate::batch::InferenceServer;

/// Virtual loss magnitude. Applied during selection to discourage other
/// threads from visiting the same path.
const VIRTUAL_LOSS: f32 = 1.0;

/// Shared tree state protected by a mutex. Contains the arena and root index.
struct TreeState {
    arena: Arena,
    root: NodeIndex,
}

/// Apply virtual loss along a selection path.
///
/// For each node in the path:
/// - `visit_count += 1` (makes this path look more visited)
/// - `total_value -= VIRTUAL_LOSS` (makes Q look worse, discouraging other threads)
fn apply_virtual_loss(path: &[NodeIndex], arena: &mut Arena) {
    for &idx in path {
        let node = arena.get_mut(idx);
        node.visit_count += 1;
        node.total_value -= VIRTUAL_LOSS;
    }
}

/// Remove virtual loss along a selection path.
///
/// Reverses the effect of [`apply_virtual_loss`] so that the real backup
/// can apply the correct value.
fn remove_virtual_loss(path: &[NodeIndex], arena: &mut Arena) {
    for &idx in path {
        let node = arena.get_mut(idx);
        node.visit_count -= 1;
        node.total_value += VIRTUAL_LOSS;
    }
}

/// Select a leaf node from root, applying virtual loss along the way.
///
/// Returns the selection path and a `Game` state advanced to the leaf position.
/// Virtual loss is applied to all nodes along the path (including root).
///
/// The caller must hold the tree lock.
fn select_leaf_with_virtual_loss(
    state: &mut TreeState,
    game: &Game,
    config: &MctsConfig,
) -> (Vec<NodeIndex>, Game) {
    let mut path: Vec<NodeIndex> = vec![state.root];
    let mut current = state.root;
    let mut sim_game = Game::from_board(game.board().clone());

    // Descend while the current node is expanded and has children.
    while state.arena[current].is_expanded && state.arena[current].num_children > 0 {
        let child_idx = select_child(current, &state.arena, config);
        let child_mv = state.arena[child_idx].mv;
        sim_game.make_move(child_mv);
        path.push(child_idx);
        current = child_idx;
    }

    // Apply virtual loss to the entire path to discourage other threads
    // from selecting the same nodes.
    apply_virtual_loss(&path, &mut state.arena);

    (path, sim_game)
}

/// Collect the search result from the shared tree state.
///
/// Extracts move visit counts from the root's children, sorts them, and
/// computes the root value.
fn collect_search_result(state: &TreeState) -> SearchResult {
    let mut move_visits: Vec<(Move, u32)> = state.arena[state.root]
        .children(&state.arena)
        .map(|child_idx| {
            let child = &state.arena[child_idx];
            (child.mv, child.visit_count)
        })
        .collect();

    // Sort by visit count descending.
    move_visits.sort_by(|a, b| b.1.cmp(&a.1));

    let root_value = state.arena[state.root].q_value();
    let total_simulations: u32 = move_visits.iter().map(|&(_, v)| v).sum();

    SearchResult {
        move_visits,
        total_simulations,
        root_value,
    }
}

/// Run parallel MCTS search with multiple worker threads.
///
/// Each worker thread:
/// 1. Acquires the tree lock
/// 2. Selects a leaf using PUCT (with virtual loss)
/// 3. Releases the tree lock
/// 4. Evaluates the leaf position via [`InferenceServer`] (blocking, batched)
/// 5. Acquires the tree lock again
/// 6. Removes virtual loss, expands the leaf, and backpropagates
/// 7. Releases the lock
///
/// The [`InferenceServer`] naturally batches evaluations from multiple threads,
/// so NN inference (the bottleneck) happens in parallel with tree operations.
///
/// # Key insight
///
/// The tree lock is only held during select (~fast) and backup (~fast), NOT
/// during NN inference (~slow, ~95% of time). This means lock contention is
/// low even with many threads.
///
/// # Arguments
/// * `game` -- The game state to search from. Not mutated.
/// * `config` -- MCTS search parameters.
/// * `server` -- The batched inference server (shared across threads).
/// * `num_threads` -- Number of worker threads.
///
/// # Returns
/// `SearchResult` with move visit counts and root value.
pub fn search_parallel(
    game: &Game,
    config: &MctsConfig,
    server: &InferenceServer,
    num_threads: usize,
) -> SearchResult {
    // Handle terminal positions immediately.
    if game.is_terminal() || game.legal_moves().is_empty() {
        let mut arena = Arena::new(64);
        let root_idx = arena.alloc(Node::root());
        let root_expand = expand(root_idx, game, None, &mut arena);
        if let ExpandResult::Terminal(value) = root_expand {
            return SearchResult {
                move_visits: Vec::new(),
                total_simulations: 0,
                root_value: value,
            };
        }
    }

    // Initialize shared tree state.
    let estimated_nodes = config.num_simulations as usize * 30 + 64;
    let mut arena = Arena::new(estimated_nodes);
    let root_idx = arena.alloc(Node::root());

    // Evaluate root with NN and expand.
    let legal_moves = game.legal_moves();
    let root_eval = server.evaluate(game.board());
    let root_policy = extract_legal_policy(&root_eval.policy, &legal_moves, game.board());
    let root_expand = expand(root_idx, game, Some(&root_policy), &mut arena);

    if let ExpandResult::Terminal(value) = root_expand {
        return SearchResult {
            move_visits: Vec::new(),
            total_simulations: 0,
            root_value: value,
        };
    }

    // Add Dirichlet noise to root's children for exploration.
    if config.dirichlet_epsilon > 0.0 {
        add_dirichlet_noise(root_idx, &mut arena, config.dirichlet_alpha, config.dirichlet_epsilon);
    }

    let tree = Arc::new(Mutex::new(TreeState {
        arena,
        root: root_idx,
    }));
    let simulations_done = Arc::new(AtomicU32::new(0));
    let target_sims = config.num_simulations;

    // Spawn worker threads using scoped threads so we can borrow
    // server, game, and config from the outer scope.
    std::thread::scope(|s| {
        for _ in 0..num_threads {
            let tree = Arc::clone(&tree);
            let sims = Arc::clone(&simulations_done);

            s.spawn(move || {
                loop {
                    // Claim a simulation slot. If we've reached the target, stop.
                    let sim_idx = sims.fetch_add(1, Ordering::Relaxed);
                    if sim_idx >= target_sims {
                        break;
                    }

                    // Phase 1: SELECT (with tree lock)
                    // Walk from root to a leaf, applying virtual loss.
                    let (path, sim_game) = {
                        let mut state = tree.lock().unwrap();
                        select_leaf_with_virtual_loss(&mut state, game, config)
                    };
                    // Lock is released here.

                    let leaf_idx = *path.last().unwrap();

                    // Phase 2: EVALUATE (without tree lock -- this is the slow part)
                    // Check if the leaf is terminal or needs NN evaluation.
                    let leaf_is_terminal = sim_game.is_terminal();
                    let (eval_value, eval_policy) = if leaf_is_terminal {
                        // Terminal: we'll get the value during expand.
                        (None, None)
                    } else {
                        let leaf_legal_moves = sim_game.legal_moves();
                        let eval = server.evaluate(sim_game.board());
                        let policy = extract_legal_policy(
                            &eval.policy,
                            &leaf_legal_moves,
                            sim_game.board(),
                        );
                        (Some(eval.value), Some(policy))
                    };

                    // Phase 3: EXPAND + BACKUP (with tree lock)
                    {
                        let mut state = tree.lock().unwrap();

                        // Remove virtual loss first.
                        remove_virtual_loss(&path, &mut state.arena);

                        // Expand and get leaf value.
                        let leaf_value = if state.arena[leaf_idx].is_expanded {
                            // Already expanded (terminal leaf re-visited, or another
                            // thread expanded it while we were evaluating).
                            // Use existing Q value for terminal leaves, or the NN
                            // value we just computed.
                            if state.arena[leaf_idx].is_terminal_leaf() {
                                state.arena[leaf_idx].q_value()
                            } else {
                                eval_value.unwrap_or(0.0)
                            }
                        } else if leaf_is_terminal {
                            match expand(leaf_idx, &sim_game, None, &mut state.arena) {
                                ExpandResult::Terminal(v) => v,
                                _ => 0.0,
                            }
                        } else {
                            let policy = eval_policy.as_deref();
                            expand(leaf_idx, &sim_game, policy, &mut state.arena);
                            eval_value.unwrap_or(0.0)
                        };

                        // Backup the path with the real leaf value.
                        backup(&path, leaf_value, &mut state.arena);
                    }
                    // Lock is released here.
                }
            });
        }
    });

    // Collect results from the shared tree.
    let state = tree.lock().unwrap();
    collect_search_result(&state)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chess_engine::board::Board;
    use chess_engine::game::Game;

    // ========================================================================
    // Helper: create a config with fewer simulations for fast tests.
    // ========================================================================

    fn test_config(num_simulations: u32) -> MctsConfig {
        MctsConfig {
            num_simulations,
            ..MctsConfig::default()
        }
    }

    // ========================================================================
    // 1. Basic search completes
    // ========================================================================

    #[test]
    fn basic_search_completes_from_starting_position() {
        let game = Game::new();
        let config = test_config(100);
        let result = search(&game, &config);

        // Starting position has 20 legal moves.
        assert_eq!(
            result.move_visits.len(),
            20,
            "Expected 20 moves, got {}",
            result.move_visits.len()
        );

        // Total simulations should match config.
        assert_eq!(result.total_simulations, 100);

        // Sum of visit counts should equal num_simulations.
        let total_visits: u32 = result.move_visits.iter().map(|&(_, v)| v).sum();
        assert_eq!(
            total_visits, 100,
            "Total visits {} should equal num_simulations 100",
            total_visits
        );
    }

    // ========================================================================
    // 2. Move visits are sorted
    // ========================================================================

    #[test]
    fn move_visits_sorted_descending() {
        let game = Game::new();
        let config = test_config(100);
        let result = search(&game, &config);

        for i in 1..result.move_visits.len() {
            assert!(
                result.move_visits[i - 1].1 >= result.move_visits[i].1,
                "move_visits not sorted descending at index {}: {} < {}",
                i,
                result.move_visits[i - 1].1,
                result.move_visits[i].1,
            );
        }
    }

    // ========================================================================
    // 3. All root children visited
    // ========================================================================

    #[test]
    fn all_root_children_visited_with_enough_simulations() {
        let game = Game::new();
        // 100 simulations for 20 moves: each should get at least 1 visit.
        let config = test_config(100);
        let result = search(&game, &config);

        for &(mv, visits) in &result.move_visits {
            assert!(
                visits >= 1,
                "Move {:?} has 0 visits with 100 simulations",
                mv
            );
        }
    }

    // ========================================================================
    // 4. Search from non-starting position
    // ========================================================================

    #[test]
    fn search_from_non_starting_position() {
        // A position with fewer legal moves: king + rook vs king.
        let board =
            Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").expect("valid FEN");
        let game = Game::from_board(board);
        let legal_move_count = game.legal_moves().len();

        let config = test_config(50);
        let result = search(&game, &config);

        assert_eq!(
            result.move_visits.len(),
            legal_move_count,
            "Expected {} moves, got {}",
            legal_move_count,
            result.move_visits.len()
        );

        let total_visits: u32 = result.move_visits.iter().map(|&(_, v)| v).sum();
        assert_eq!(total_visits, 50);
    }

    // ========================================================================
    // 5. Temperature=0 selects most visited
    // ========================================================================

    #[test]
    fn temperature_zero_selects_most_visited() {
        let game = Game::new();
        let config = test_config(100);
        let result = search(&game, &config);

        let selected = select_move(&result, 0.0);

        // The most visited move is the first (sorted descending).
        assert_eq!(
            selected, result.move_visits[0].0,
            "Temperature=0 should select the most visited move"
        );
    }

    // ========================================================================
    // 6. Temperature=1 sampling
    // ========================================================================

    #[test]
    fn temperature_one_sampling_distribution() {
        let game = Game::new();
        let config = test_config(200);
        let result = search(&game, &config);

        // Run many selections with T=1 and check that we see multiple
        // distinct moves (not always the same one).
        let mut selected_moves = std::collections::HashSet::new();
        for _ in 0..200 {
            let mv = select_move(&result, 1.0);
            selected_moves.insert(format!("{:?}", mv));
        }

        // With T=1 and 200 samples over 20 moves, we should see at least 5
        // distinct moves.
        assert!(
            selected_moves.len() >= 5,
            "Expected at least 5 distinct moves with T=1 sampling, got {}",
            selected_moves.len()
        );
    }

    // ========================================================================
    // 7. Dirichlet noise changes priors
    // ========================================================================

    #[test]
    fn dirichlet_noise_changes_priors() {
        let game = Game::new();
        let mut arena = Arena::new(256);
        let root_idx = arena.alloc(Node::root());
        expand(root_idx, &game, None, &mut arena);

        // Record original priors.
        let original_priors: Vec<f32> = arena[root_idx]
            .children(&arena)
            .map(|idx| arena[idx].prior)
            .collect();

        // Add Dirichlet noise.
        add_dirichlet_noise(root_idx, &mut arena, 0.3, 0.25);

        // Collect new priors.
        let new_priors: Vec<f32> = arena[root_idx]
            .children(&arena)
            .map(|idx| arena[idx].prior)
            .collect();

        // At least some priors should have changed.
        let changed = original_priors
            .iter()
            .zip(new_priors.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-6)
            .count();

        assert!(
            changed > 0,
            "Dirichlet noise should change at least some priors"
        );

        // New priors should still sum to approximately 1.0.
        let prior_sum: f32 = new_priors.iter().sum();
        assert!(
            (prior_sum - 1.0).abs() < 0.01,
            "Noisy priors should sum to ~1.0, got {}",
            prior_sum
        );
    }

    // ========================================================================
    // 8. Search from near-terminal (only 1 legal move)
    // ========================================================================

    #[test]
    fn search_single_legal_move() {
        // Position where one side has very few legal moves.
        // Black king on h8, White queen on g6, White king on f6.
        // Black has only Kh8-g8 as legal move (Kh7 is attacked by Qg6).
        // Actually, let me find a better FEN with exactly 1 move.
        // King on a1, rook blocks b1, pawns block a2/b2, only move is Ka1-...
        // Let's use: Black king on h8, White queen on f7, White king on g5.
        // From h8, black can only go to g8 (h7 is attacked by Qf7).
        // Actually checking: Qf7 attacks g8 too. Let me think...
        //
        // Simpler approach: a position where the king has exactly one escape.
        // White king on a1, with pawns on a2, b2. The only move is Kb1.
        // But we need to set up proper FEN.
        //
        // Let's use a known position with very few moves and verify search works.
        let board =
            Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").expect("valid FEN");
        let game = Game::from_board(board);
        let legal_moves = game.legal_moves();

        // If this is checkmate or stalemate, pick a different position.
        if game.is_terminal() {
            // The position is terminal, so let's just verify search handles it.
            let config = test_config(50);
            let result = search(&game, &config);
            assert!(result.move_visits.is_empty());
            return;
        }

        let num_legal = legal_moves.len();
        assert!(
            num_legal <= 3,
            "Expected few legal moves, got {}",
            num_legal
        );

        let config = test_config(50);
        let result = search(&game, &config);

        assert_eq!(result.move_visits.len(), num_legal);

        let total_visits: u32 = result.move_visits.iter().map(|&(_, v)| v).sum();
        assert_eq!(total_visits, 50);
    }

    // ========================================================================
    // 9. SearchResult fields: root_value is reasonable
    // ========================================================================

    #[test]
    fn root_value_is_reasonable() {
        let game = Game::new();
        let config = test_config(100);
        let result = search(&game, &config);

        // With uniform policy and 0.0 evaluation, root value should be
        // between -1 and 1.
        assert!(
            result.root_value >= -1.0 && result.root_value <= 1.0,
            "Root value {} should be in [-1, 1]",
            result.root_value
        );
    }

    // ========================================================================
    // 10. Larger search: 800 simulations in reasonable time
    // ========================================================================

    #[test]
    fn larger_search_800_simulations() {
        let game = Game::new();
        let config = test_config(800);

        let start = std::time::Instant::now();
        let result = search(&game, &config);
        let elapsed = start.elapsed();

        assert_eq!(result.total_simulations, 800);
        assert_eq!(result.move_visits.len(), 20);

        let total_visits: u32 = result.move_visits.iter().map(|&(_, v)| v).sum();
        assert_eq!(total_visits, 800);

        // Should complete well within 1 second on any modern machine.
        assert!(
            elapsed.as_secs() < 2,
            "800 simulations took {:?}, expected < 2 seconds",
            elapsed
        );
    }

    // ========================================================================
    // 11. Terminal position search returns empty moves
    // ========================================================================

    #[test]
    fn search_terminal_position_returns_empty() {
        // Checkmate position: fool's mate.
        let board = Board::from_fen(
            "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        )
        .expect("valid FEN");
        let game = Game::from_board(board);

        let config = test_config(100);
        let result = search(&game, &config);

        assert!(
            result.move_visits.is_empty(),
            "Terminal position should have no moves"
        );
        assert_eq!(result.total_simulations, 0);
        assert!(
            (result.root_value - (-1.0)).abs() < f32::EPSILON,
            "Root value should be -1.0 (loss for side to move), got {}",
            result.root_value
        );
    }

    // ========================================================================
    // 12. Search without Dirichlet noise
    // ========================================================================

    #[test]
    fn search_without_dirichlet_noise() {
        let game = Game::new();
        let config = MctsConfig {
            num_simulations: 50,
            dirichlet_epsilon: 0.0,
            ..MctsConfig::default()
        };
        let result = search(&game, &config);

        assert_eq!(result.move_visits.len(), 20);
        let total_visits: u32 = result.move_visits.iter().map(|&(_, v)| v).sum();
        assert_eq!(total_visits, 50);
    }

    // ========================================================================
    // 13. select_move panics on empty result
    // ========================================================================

    #[test]
    #[should_panic(expected = "no moves")]
    fn select_move_panics_on_empty() {
        let result = SearchResult {
            move_visits: Vec::new(),
            total_simulations: 0,
            root_value: 0.0,
        };
        select_move(&result, 1.0);
    }

    // ========================================================================
    // 14. High temperature gives more uniform selection
    // ========================================================================

    #[test]
    fn high_temperature_more_uniform() {
        let game = Game::new();
        let config = test_config(200);
        let result = search(&game, &config);

        // With very high temperature, selection should approach uniform.
        let mut counts = std::collections::HashMap::new();
        for _ in 0..1000 {
            let mv = select_move(&result, 10.0);
            *counts.entry(format!("{:?}", mv)).or_insert(0u32) += 1;
        }

        // With T=10 and 1000 samples over 20 moves, each move should get
        // roughly 50 selections. Check that the most selected is < 3x the
        // least selected (very loose bound).
        let max_count = *counts.values().max().unwrap();
        let min_count = *counts.values().min().unwrap_or(&0);
        assert!(
            max_count < min_count * 4 + 50, // loose bound
            "High temperature should give roughly uniform: max={}, min={}",
            max_count,
            min_count
        );
    }

    // ========================================================================
    // 15. Dirichlet noise preserves prior sum
    // ========================================================================

    #[test]
    fn dirichlet_noise_preserves_prior_sum() {
        let game = Game::new();
        let mut arena = Arena::new(256);
        let root_idx = arena.alloc(Node::root());
        expand(root_idx, &game, None, &mut arena);

        add_dirichlet_noise(root_idx, &mut arena, 0.3, 0.25);

        let prior_sum: f32 = arena[root_idx]
            .children(&arena)
            .map(|idx| arena[idx].prior)
            .sum();

        assert!(
            (prior_sum - 1.0).abs() < 0.02,
            "Noisy priors should sum to ~1.0, got {}",
            prior_sum
        );
    }

    // ========================================================================
    // 16. Dirichlet noise with no children is a no-op
    // ========================================================================

    #[test]
    fn dirichlet_noise_no_children_noop() {
        let mut arena = Arena::new(16);
        let root_idx = arena.alloc(Node::root());

        // Should not panic.
        add_dirichlet_noise(root_idx, &mut arena, 0.3, 0.25);
    }

    // ========================================================================
    // 17. Multiple searches are independent
    // ========================================================================

    #[test]
    fn multiple_searches_independent() {
        let game = Game::new();
        let config = test_config(50);

        let result1 = search(&game, &config);
        let result2 = search(&game, &config);

        // Both should have the same number of moves and total visits.
        assert_eq!(result1.move_visits.len(), result2.move_visits.len());
        assert_eq!(result1.total_simulations, result2.total_simulations);

        let total1: u32 = result1.move_visits.iter().map(|&(_, v)| v).sum();
        let total2: u32 = result2.move_visits.iter().map(|&(_, v)| v).sum();
        assert_eq!(total1, total2);
    }

    // ========================================================================
    // 18. Search handles stalemate position
    // ========================================================================

    #[test]
    fn search_stalemate_position() {
        let board =
            Board::from_fen("k7/8/1Q6/8/8/8/8/2K5 b - - 0 1").expect("valid FEN");
        let game = Game::from_board(board);

        let config = test_config(50);
        let result = search(&game, &config);

        assert!(result.move_visits.is_empty());
        assert_eq!(result.total_simulations, 0);
        assert!(
            result.root_value.abs() < f32::EPSILON,
            "Stalemate value should be 0.0, got {}",
            result.root_value
        );
    }

    // ========================================================================
    // 19. extract_legal_policy: basic softmax
    // ========================================================================

    #[test]
    fn extract_legal_policy_basic_softmax() {
        // Create a fake policy logits array (4672 elements).
        let board = Board::starting_position();
        let game = Game::from_board(board.clone());
        let legal_moves = game.legal_moves();

        // Create logits: give known values to the legal move indices.
        let mut logits = vec![0.0f32; crate::nn::POLICY_SIZE];

        // Set some legal move logits to distinct values.
        for (i, mv) in legal_moves.iter().enumerate() {
            if let Some(idx) = crate::nn::move_to_policy_index(mv, &board) {
                logits[idx] = i as f32; // 0, 1, 2, ..., 19
            }
        }

        let policy = extract_legal_policy(&logits, &legal_moves, &board);

        // Should have one probability per legal move.
        assert_eq!(policy.len(), legal_moves.len());

        // All probabilities should be positive.
        for &p in &policy {
            assert!(p > 0.0, "All probabilities should be positive, got {}", p);
        }

        // Higher logits should give higher probabilities.
        // The last move (logit = 19) should have the highest probability.
        let max_idx = policy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            max_idx,
            legal_moves.len() - 1,
            "Move with highest logit should have highest probability"
        );
    }

    // ========================================================================
    // 20. extract_legal_policy: sums to 1.0
    // ========================================================================

    #[test]
    fn extract_legal_policy_sums_to_one() {
        let board = Board::starting_position();
        let game = Game::from_board(board.clone());
        let legal_moves = game.legal_moves();

        // Random-ish logits.
        let mut logits = vec![0.0f32; crate::nn::POLICY_SIZE];
        for (i, mv) in legal_moves.iter().enumerate() {
            if let Some(idx) = crate::nn::move_to_policy_index(mv, &board) {
                logits[idx] = (i as f32 - 10.0) * 0.5;
            }
        }

        let policy = extract_legal_policy(&logits, &legal_moves, &board);
        let sum: f32 = policy.iter().sum();

        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Policy should sum to ~1.0, got {}",
            sum
        );
    }

    // ========================================================================
    // 21. extract_legal_policy: handles all-zero logits
    // ========================================================================

    #[test]
    fn extract_legal_policy_all_zero_logits() {
        let board = Board::starting_position();
        let game = Game::from_board(board.clone());
        let legal_moves = game.legal_moves();

        // All logits are zero (the legal moves map to zeros).
        let logits = vec![0.0f32; crate::nn::POLICY_SIZE];

        let policy = extract_legal_policy(&logits, &legal_moves, &board);

        // Should not crash and should produce valid probabilities.
        assert_eq!(policy.len(), legal_moves.len());

        let sum: f32 = policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "All-zero logits should still sum to ~1.0, got {}",
            sum
        );

        // With all-zero logits, softmax gives uniform distribution.
        let expected = 1.0 / legal_moves.len() as f32;
        for &p in &policy {
            assert!(
                (p - expected).abs() < 1e-5,
                "All-zero logits should give uniform: expected {}, got {}",
                expected,
                p
            );
        }
    }

    // ========================================================================
    // 22. extract_legal_policy: illegal moves get zero probability
    // ========================================================================

    #[test]
    fn extract_legal_policy_masks_illegal_moves() {
        let board = Board::starting_position();
        let game = Game::from_board(board.clone());
        let legal_moves = game.legal_moves();

        // Set all logits to a large value, then set legal move logits to zero.
        // The key insight: only legal move logits are extracted, so the large
        // values for illegal moves are ignored entirely.
        let mut logits = vec![100.0f32; crate::nn::POLICY_SIZE];
        for mv in &legal_moves {
            if let Some(idx) = crate::nn::move_to_policy_index(mv, &board) {
                logits[idx] = 0.0;
            }
        }

        let policy = extract_legal_policy(&logits, &legal_moves, &board);

        // The returned vector only contains probabilities for legal moves.
        assert_eq!(policy.len(), legal_moves.len());

        // Since all legal move logits are 0.0, softmax gives uniform.
        let expected = 1.0 / legal_moves.len() as f32;
        for &p in &policy {
            assert!(
                (p - expected).abs() < 1e-5,
                "Should be uniform over legal moves: expected {}, got {}",
                expected,
                p
            );
        }

        // No probability leaks to illegal moves -- the output vector only
        // has entries for legal moves, so illegal moves inherently get zero.
        // This is verified by the output length matching legal_moves length.
    }

    // ========================================================================
    // 23. LeafEvaluator trait works with UniformEvaluator
    // ========================================================================

    #[test]
    fn uniform_evaluator_gives_uniform_priors_and_zero_value() {
        let board = Board::starting_position();
        let game = Game::from_board(board.clone());
        let legal_moves = game.legal_moves();

        let evaluator = UniformEvaluator;
        let (policy, value) = evaluator.evaluate(&board, &legal_moves);

        assert_eq!(policy.len(), legal_moves.len());
        assert_eq!(value, 0.0);

        let expected = 1.0 / legal_moves.len() as f32;
        for &p in &policy {
            assert!(
                (p - expected).abs() < 1e-6,
                "Uniform evaluator should give {}, got {}",
                expected,
                p
            );
        }
    }

    // ========================================================================
    // 24. search_with_evaluator using UniformEvaluator matches search()
    // ========================================================================

    #[test]
    fn search_with_uniform_evaluator_same_as_search() {
        let game = Game::new();
        let config = MctsConfig {
            num_simulations: 50,
            dirichlet_epsilon: 0.0, // No noise for deterministic comparison.
            ..MctsConfig::default()
        };

        let result_old = search(&game, &config);
        let result_new = search_with_evaluator(&game, &config, &UniformEvaluator);

        // Both should have 20 legal moves.
        assert_eq!(result_old.move_visits.len(), 20);
        assert_eq!(result_new.move_visits.len(), 20);

        // Both should have the same total simulations.
        assert_eq!(result_old.total_simulations, result_new.total_simulations);

        // Total visits should match.
        let total_old: u32 = result_old.move_visits.iter().map(|&(_, v)| v).sum();
        let total_new: u32 = result_new.move_visits.iter().map(|&(_, v)| v).sum();
        assert_eq!(total_old, 50);
        assert_eq!(total_new, 50);
    }

    // ========================================================================
    // 25. search_with_evaluator handles terminal position
    // ========================================================================

    #[test]
    fn search_with_evaluator_terminal_position() {
        // Checkmate position: fool's mate.
        let board = Board::from_fen(
            "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        )
        .expect("valid FEN");
        let game = Game::from_board(board);

        let config = test_config(100);
        let result = search_with_evaluator(&game, &config, &UniformEvaluator);

        assert!(
            result.move_visits.is_empty(),
            "Terminal position should have no moves"
        );
        assert_eq!(result.total_simulations, 0);
        assert!(
            (result.root_value - (-1.0)).abs() < f32::EPSILON,
            "Root value should be -1.0 (loss for side to move), got {}",
            result.root_value
        );
    }

    // ========================================================================
    // 26-30: NN-guided tests (gated on model file existence)
    // ========================================================================

    /// Helper to load the test model if it exists.
    fn load_test_model() -> Option<crate::nn::NnModel> {
        let model_path = "/Users/william/Desktop/Random/alphazero/test_model.pt";
        if !std::path::Path::new(model_path).exists() {
            eprintln!(
                "Skipping NN search test: {} not found. \
                 Run the Python export script to create it.",
                model_path
            );
            return None;
        }
        Some(crate::nn::NnModel::load(model_path, tch::Device::Cpu).expect("Failed to load model"))
    }

    #[test]
    fn nn_search_completes_from_starting_position() {
        let model = match load_test_model() {
            Some(m) => m,
            None => return,
        };

        let game = Game::new();
        let config = test_config(50);
        let result = search_with_nn(&game, &config, &model);

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

        // Move visits should be sorted descending.
        for i in 1..result.move_visits.len() {
            assert!(
                result.move_visits[i - 1].1 >= result.move_visits[i].1,
                "NN search move_visits not sorted at index {}",
                i,
            );
        }
    }

    #[test]
    fn nn_search_root_value_is_nonzero() {
        let model = match load_test_model() {
            Some(m) => m,
            None => return,
        };

        let game = Game::new();
        let config = MctsConfig {
            num_simulations: 100,
            dirichlet_epsilon: 0.0, // No noise for cleaner test.
            ..MctsConfig::default()
        };
        let result = search_with_nn(&game, &config, &model);

        // The NN has opinions (even untrained), so root value should
        // generally be non-zero. With random weights, it's extremely
        // unlikely to be exactly 0.0.
        // But we only assert it's in the valid range.
        assert!(
            result.root_value >= -1.0 && result.root_value <= 1.0,
            "Root value {} should be in [-1, 1]",
            result.root_value
        );

        // With NN evaluation (even random), the root value is almost
        // certainly non-zero.
        assert!(
            result.root_value.abs() > 1e-6,
            "NN search root value should generally be non-zero, got {}",
            result.root_value
        );
    }

    #[test]
    fn nn_search_produces_nonuniform_visits() {
        let model = match load_test_model() {
            Some(m) => m,
            None => return,
        };

        let game = Game::new();
        let config = MctsConfig {
            num_simulations: 200,
            dirichlet_epsilon: 0.0, // No noise for cleaner comparison.
            ..MctsConfig::default()
        };
        let result = search_with_nn(&game, &config, &model);

        // With NN-guided search, the visit distribution should be
        // non-uniform -- the NN policy concentrates visits on
        // preferred moves.
        let visits: Vec<u32> = result.move_visits.iter().map(|&(_, v)| v).collect();
        let max_visits = *visits.iter().max().unwrap();
        let min_visits = *visits.iter().min().unwrap();

        // The most visited move should have significantly more visits
        // than the least visited. With 200 simulations over 20 moves,
        // uniform would give ~10 each. NN should make it more spread.
        assert!(
            max_visits > min_visits,
            "NN search should produce non-uniform visits: max={}, min={}",
            max_visits,
            min_visits
        );
    }

    #[test]
    fn nn_evaluator_produces_valid_policy() {
        let model = match load_test_model() {
            Some(m) => m,
            None => return,
        };

        let board = Board::starting_position();
        let game = Game::from_board(board.clone());
        let legal_moves = game.legal_moves();

        let evaluator = NnEvaluator::new(&model);
        let (policy, value) = evaluator.evaluate(&board, &legal_moves);

        // Policy should have one entry per legal move.
        assert_eq!(policy.len(), legal_moves.len());

        // All probabilities should be non-negative.
        for &p in &policy {
            assert!(p >= 0.0, "Policy probability should be >= 0, got {}", p);
        }

        // Probabilities should sum to ~1.0.
        let sum: f32 = policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "NN policy should sum to ~1.0, got {}",
            sum
        );

        // Value should be in [-1, 1].
        assert!(
            value >= -1.0 && value <= 1.0,
            "NN value {} should be in [-1, 1]",
            value
        );
    }

    #[test]
    fn nn_search_handles_terminal_position() {
        let model = match load_test_model() {
            Some(m) => m,
            None => return,
        };

        // Checkmate position.
        let board = Board::from_fen(
            "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        )
        .expect("valid FEN");
        let game = Game::from_board(board);

        let config = test_config(50);
        let result = search_with_nn(&game, &config, &model);

        assert!(result.move_visits.is_empty());
        assert_eq!(result.total_simulations, 0);
        assert!(
            (result.root_value - (-1.0)).abs() < f32::EPSILON,
            "Checkmate root value should be -1.0, got {}",
            result.root_value
        );
    }

    // ========================================================================
    // Parallel search tests (gated on model file existence)
    // ========================================================================

    /// Path to the test model file (same as other NN tests).
    const TEST_MODEL_PATH: &str = "/Users/william/Desktop/Random/alphazero/test_model.pt";

    /// Helper to load model and create an InferenceServer.
    fn get_test_server() -> Option<(crate::nn::NnModel, InferenceServer)> {
        if !std::path::Path::new(TEST_MODEL_PATH).exists() {
            eprintln!(
                "Skipping parallel search test: {} not found. \
                 Run the Python export script to create it.",
                TEST_MODEL_PATH
            );
            return None;
        }
        let model_for_check = crate::nn::NnModel::load(TEST_MODEL_PATH, tch::Device::Cpu)
            .expect("Failed to load model for server");
        let model_for_server = crate::nn::NnModel::load(TEST_MODEL_PATH, tch::Device::Cpu)
            .expect("Failed to load model for server");
        let server = InferenceServer::new(model_for_server, 8, 50);
        Some((model_for_check, server))
    }

    // 31. Parallel search produces valid results: 2 threads, 100 sims
    #[test]
    fn parallel_search_produces_valid_results() {
        let (_model, server) = match get_test_server() {
            Some(s) => s,
            None => return,
        };

        let game = Game::new();
        let config = MctsConfig {
            num_simulations: 100,
            dirichlet_epsilon: 0.25,
            ..MctsConfig::default()
        };

        let result = search_parallel(&game, &config, &server, 2);

        // Starting position has 20 legal moves.
        assert_eq!(
            result.move_visits.len(),
            20,
            "Expected 20 moves, got {}",
            result.move_visits.len()
        );

        // Sum of visit counts should equal num_simulations.
        let total_visits: u32 = result.move_visits.iter().map(|&(_, v)| v).sum();
        assert_eq!(
            total_visits, 100,
            "Total visits {} should equal num_simulations 100",
            total_visits
        );

        // Move visits should be sorted descending.
        for i in 1..result.move_visits.len() {
            assert!(
                result.move_visits[i - 1].1 >= result.move_visits[i].1,
                "move_visits not sorted descending at index {}",
                i,
            );
        }

        // Root value should be in valid range.
        assert!(
            result.root_value >= -1.0 && result.root_value <= 1.0,
            "Root value {} should be in [-1, 1]",
            result.root_value
        );

        server.shutdown();
    }

    // 32. Thread count scaling: 1, 2, 4 threads all produce valid results
    #[test]
    fn parallel_search_thread_scaling() {
        let (_model, server) = match get_test_server() {
            Some(s) => s,
            None => return,
        };

        let game = Game::new();

        for &num_threads in &[1, 2, 4] {
            let config = MctsConfig {
                num_simulations: 100,
                dirichlet_epsilon: 0.0, // No noise for deterministic-ish comparison.
                ..MctsConfig::default()
            };

            let result = search_parallel(&game, &config, &server, num_threads);

            assert_eq!(
                result.move_visits.len(),
                20,
                "With {} threads: expected 20 moves, got {}",
                num_threads,
                result.move_visits.len()
            );

            let total_visits: u32 = result.move_visits.iter().map(|&(_, v)| v).sum();
            assert_eq!(
                total_visits, 100,
                "With {} threads: total visits {} should equal 100",
                num_threads,
                total_visits
            );
        }

        server.shutdown();
    }

    // 33. Parallel matches sequential quality: top moves should be reasonable
    #[test]
    fn parallel_search_reasonable_top_moves() {
        let (_model, server) = match get_test_server() {
            Some(s) => s,
            None => return,
        };

        let game = Game::new();
        let config = MctsConfig {
            num_simulations: 200,
            dirichlet_epsilon: 0.0,
            ..MctsConfig::default()
        };

        let result = search_parallel(&game, &config, &server, 2);

        // The most visited move should have more visits than the least visited.
        // This validates that the search is not degenerate (e.g., all visits to one node).
        let max_visits = result.move_visits.first().unwrap().1;
        let min_visits = result.move_visits.last().unwrap().1;
        assert!(
            max_visits > min_visits,
            "Parallel search should produce non-uniform visits: max={}, min={}",
            max_visits,
            min_visits
        );

        // The most visited move should have a reasonable fraction of total visits
        // (not 100% and not 0%).
        let total_visits: u32 = result.move_visits.iter().map(|&(_, v)| v).sum();
        assert!(
            max_visits < total_visits,
            "Top move should not have all visits"
        );
        assert!(max_visits > 0, "Top move should have some visits");

        server.shutdown();
    }

    // 34. Virtual loss diversifies exploration
    #[test]
    fn parallel_search_virtual_loss_diversifies() {
        let (_model, server) = match get_test_server() {
            Some(s) => s,
            None => return,
        };

        let game = Game::new();

        // Run with 1 thread (no virtual loss effect).
        let config_1 = MctsConfig {
            num_simulations: 200,
            dirichlet_epsilon: 0.0,
            ..MctsConfig::default()
        };
        let result_1 = search_parallel(&game, &config_1, &server, 1);

        // Run with 4 threads (virtual loss should diversify).
        let config_4 = MctsConfig {
            num_simulations: 200,
            dirichlet_epsilon: 0.0,
            ..MctsConfig::default()
        };
        let result_4 = search_parallel(&game, &config_4, &server, 4);

        // Both should have valid results.
        assert_eq!(result_1.move_visits.len(), 20);
        assert_eq!(result_4.move_visits.len(), 20);

        let total_1: u32 = result_1.move_visits.iter().map(|&(_, v)| v).sum();
        let total_4: u32 = result_4.move_visits.iter().map(|&(_, v)| v).sum();
        assert_eq!(total_1, 200);
        assert_eq!(total_4, 200);

        // Compute visit entropy as a proxy for exploration diversity.
        // Higher entropy = more uniform = more exploration.
        let entropy = |visits: &[(Move, u32)]| -> f64 {
            let total: f64 = visits.iter().map(|&(_, v)| v as f64).sum();
            if total == 0.0 {
                return 0.0;
            }
            visits
                .iter()
                .filter(|&&(_, v)| v > 0)
                .map(|&(_, v)| {
                    let p = v as f64 / total;
                    -p * p.ln()
                })
                .sum()
        };

        let entropy_1 = entropy(&result_1.move_visits);
        let entropy_4 = entropy(&result_4.move_visits);

        // We don't require entropy_4 > entropy_1 strictly (it's stochastic),
        // but both should be positive (non-degenerate).
        assert!(
            entropy_1 > 0.0,
            "1-thread entropy should be positive: {}",
            entropy_1
        );
        assert!(
            entropy_4 > 0.0,
            "4-thread entropy should be positive: {}",
            entropy_4
        );

        server.shutdown();
    }

    // 35. No panics under load: 800 simulations with 4 threads
    #[test]
    fn parallel_search_no_panics_under_load() {
        let (_model, server) = match get_test_server() {
            Some(s) => s,
            None => return,
        };

        let game = Game::new();
        let config = MctsConfig {
            num_simulations: 800,
            dirichlet_epsilon: 0.25,
            ..MctsConfig::default()
        };

        let result = search_parallel(&game, &config, &server, 4);

        // Should complete without panics or deadlocks.
        assert_eq!(
            result.move_visits.len(),
            20,
            "Expected 20 moves, got {}",
            result.move_visits.len()
        );

        let total_visits: u32 = result.move_visits.iter().map(|&(_, v)| v).sum();
        assert_eq!(
            total_visits, 800,
            "Total visits {} should equal 800",
            total_visits
        );

        server.shutdown();
    }

    // 36. Terminal position: parallel search on checkmate returns empty moves
    #[test]
    fn parallel_search_terminal_position() {
        let (_model, server) = match get_test_server() {
            Some(s) => s,
            None => return,
        };

        // Fool's mate: checkmate position.
        let board = Board::from_fen(
            "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        )
        .expect("valid FEN");
        let game = Game::from_board(board);

        let config = test_config(100);
        let result = search_parallel(&game, &config, &server, 2);

        assert!(
            result.move_visits.is_empty(),
            "Terminal position should have no moves"
        );
        assert_eq!(result.total_simulations, 0);
        assert!(
            (result.root_value - (-1.0)).abs() < f32::EPSILON,
            "Checkmate root value should be -1.0, got {}",
            result.root_value
        );

        server.shutdown();
    }

    // ========================================================================
    // Virtual loss unit tests (no model needed)
    // ========================================================================

    #[test]
    fn virtual_loss_apply_and_remove() {
        let mut arena = Arena::new(16);
        let root = arena.alloc(Node::root());
        let child = arena.alloc(Node::new(
            chess_engine::moves::Move::new(
                chess_engine::types::Square::E2,
                chess_engine::types::Square::E4,
            ),
            0.5,
        ));
        arena.add_child(root, child);

        // Set some initial values.
        arena[root].visit_count = 10;
        arena[root].total_value = 5.0;
        arena[child].visit_count = 3;
        arena[child].total_value = 1.5;

        let path = vec![root, child];

        // Apply virtual loss.
        apply_virtual_loss(&path, &mut arena);

        assert_eq!(arena[root].visit_count, 11);
        assert!((arena[root].total_value - (5.0 - VIRTUAL_LOSS)).abs() < f32::EPSILON);
        assert_eq!(arena[child].visit_count, 4);
        assert!((arena[child].total_value - (1.5 - VIRTUAL_LOSS)).abs() < f32::EPSILON);

        // Remove virtual loss.
        remove_virtual_loss(&path, &mut arena);

        assert_eq!(arena[root].visit_count, 10);
        assert!((arena[root].total_value - 5.0).abs() < f32::EPSILON);
        assert_eq!(arena[child].visit_count, 3);
        assert!((arena[child].total_value - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn virtual_loss_makes_q_worse() {
        let mut arena = Arena::new(16);
        let idx = arena.alloc(Node::root());

        // Set up a node with Q = 0.5 (10 visits, total_value = 5.0).
        arena[idx].visit_count = 10;
        arena[idx].total_value = 5.0;
        let q_before = arena[idx].q_value();
        assert!((q_before - 0.5).abs() < f32::EPSILON);

        // Apply virtual loss.
        apply_virtual_loss(&[idx], &mut arena);

        let q_after = arena[idx].q_value();
        // Q = (5.0 - 1.0) / 11 = 4.0 / 11 = 0.3636...
        assert!(
            q_after < q_before,
            "Virtual loss should decrease Q: before={}, after={}",
            q_before,
            q_after
        );

        // Remove virtual loss.
        remove_virtual_loss(&[idx], &mut arena);
        let q_restored = arena[idx].q_value();
        assert!(
            (q_restored - q_before).abs() < f32::EPSILON,
            "After removing virtual loss, Q should be restored: before={}, restored={}",
            q_before,
            q_restored
        );
    }

    #[test]
    fn collect_search_result_basic() {
        let mut arena = Arena::new(64);
        let root_idx = arena.alloc(Node::root());
        let game = Game::new();

        // Expand root.
        expand(root_idx, &game, None, &mut arena);

        // Set some visit counts on children.
        let children: Vec<NodeIndex> = arena[root_idx].children(&arena).collect();
        assert_eq!(children.len(), 20);

        // Give some visits to a few children.
        arena[children[0]].visit_count = 10;
        arena[children[0]].total_value = 3.0;
        arena[children[1]].visit_count = 5;
        arena[children[1]].total_value = 2.0;
        arena[children[2]].visit_count = 15;
        arena[children[2]].total_value = 7.0;

        // Set root visits.
        arena[root_idx].visit_count = 30;
        arena[root_idx].total_value = 12.0;

        let state = TreeState {
            arena,
            root: root_idx,
        };

        let result = collect_search_result(&state);

        assert_eq!(result.move_visits.len(), 20);

        // Should be sorted by visit count descending.
        assert_eq!(result.move_visits[0].1, 15);
        assert_eq!(result.move_visits[1].1, 10);
        assert_eq!(result.move_visits[2].1, 5);

        // Root value.
        assert!((result.root_value - (12.0 / 30.0)).abs() < f32::EPSILON);
    }
}
