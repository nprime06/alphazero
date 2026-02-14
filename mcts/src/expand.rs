//! Tree expansion for MCTS.
//!
//! When MCTS selection reaches a leaf node (one that has not yet been expanded),
//! we need to generate all legal moves for the current position and create a
//! child node for each move. Each child receives a prior probability from the
//! neural network's policy head (or uniform priors as a placeholder).
//!
//! Terminal positions (checkmate, stalemate, draws) are detected during
//! expansion and return a value directly, avoiding unnecessary work.

use chess_engine::game::{Game, GameResult};

use crate::arena::Arena;
use crate::node::{Node, NodeIndex};

// =============================================================================
// ExpandResult
// =============================================================================

/// Result of expanding a node.
pub enum ExpandResult {
    /// Node was successfully expanded with children created for each legal move.
    Expanded,

    /// Node is a terminal position (checkmate, stalemate, or draw).
    /// Contains the value from the perspective of the player to move:
    /// - `-1.0` = loss (checkmate — the player to move has been mated)
    /// - `0.0` = draw (stalemate, repetition, fifty-move rule, insufficient material)
    Terminal(f32),

    /// Node was already expanded (no-op).
    AlreadyExpanded,
}

// =============================================================================
// expand
// =============================================================================

/// Expand a leaf node by generating legal moves and creating child nodes.
///
/// This is the expansion phase of MCTS. When tree selection reaches an
/// unexpanded leaf, this function:
/// 1. Checks whether the node was already expanded (returns [`AlreadyExpanded`]).
/// 2. Checks whether the position is terminal (returns [`Terminal`] with a value).
/// 3. Generates all legal moves for the current position.
/// 4. Creates a child node for each move, assigning policy priors.
/// 5. Links each child to the parent via [`Arena::add_child`].
/// 6. Marks the node as expanded.
///
/// # Arguments
/// * `node_idx` — Index of the leaf node to expand.
/// * `game` — Current game state at this node.
/// * `policy` — Policy priors for each legal move. Must have the same length
///   as the number of legal moves. Will be normalized to sum to 1.0.
///   Pass `None` to use uniform priors (`1 / num_moves` each).
/// * `arena` — The node arena.
///
/// # Returns
/// * [`ExpandResult::Expanded`] if the node was successfully expanded.
/// * [`ExpandResult::Terminal(value)`] if the position is terminal.
/// * [`ExpandResult::AlreadyExpanded`] if the node was already expanded.
///
/// # Panics
/// Panics if `policy` is `Some` and its length does not match the number of
/// legal moves.
///
/// [`AlreadyExpanded`]: ExpandResult::AlreadyExpanded
/// [`Terminal`]: ExpandResult::Terminal
pub fn expand(
    node_idx: NodeIndex,
    game: &Game,
    policy: Option<&[f32]>,
    arena: &mut Arena,
) -> ExpandResult {
    // 1. Already expanded — nothing to do.
    if arena[node_idx].is_expanded {
        return ExpandResult::AlreadyExpanded;
    }

    // 2. Check for terminal position.
    let result = game.result();
    if result != GameResult::Ongoing {
        // Mark as expanded (with zero children) so it becomes a "terminal leaf".
        arena[node_idx].is_expanded = true;

        // Compute value from the perspective of the player to move.
        let side_to_move = game.board().side_to_move();
        let value = game.result_for_color(side_to_move);
        return ExpandResult::Terminal(value);
    }

    // 3. Generate legal moves.
    let legal_moves = game.legal_moves();
    let num_moves = legal_moves.len();
    debug_assert!(num_moves > 0, "Ongoing game must have at least one legal move");

    // 4. Compute priors.
    let priors: Vec<f32> = match policy {
        Some(raw_policy) => {
            assert_eq!(
                raw_policy.len(),
                num_moves,
                "Policy length ({}) must match number of legal moves ({})",
                raw_policy.len(),
                num_moves,
            );
            normalize_policy(raw_policy)
        }
        None => {
            // Uniform priors: each move gets equal probability.
            let uniform = 1.0 / num_moves as f32;
            vec![uniform; num_moves]
        }
    };

    // 5. Create child nodes and link them to the parent.
    for (mv, &prior) in legal_moves.iter().zip(priors.iter()) {
        let child = Node::new(*mv, prior);
        let child_idx = arena.alloc(child);
        arena.add_child(node_idx, child_idx);
    }

    // 6. Mark as expanded.
    arena[node_idx].is_expanded = true;

    ExpandResult::Expanded
}

// =============================================================================
// Helper: policy normalization
// =============================================================================

/// Normalizes a policy vector so that its elements sum to 1.0.
///
/// If the sum is zero or very small (< 1e-8), falls back to uniform priors
/// to avoid division by zero.
fn normalize_policy(raw: &[f32]) -> Vec<f32> {
    let sum: f32 = raw.iter().sum();

    if sum < 1e-8 {
        // All-zero or near-zero policy — fall back to uniform.
        let uniform = 1.0 / raw.len() as f32;
        vec![uniform; raw.len()]
    } else {
        raw.iter().map(|&p| p / sum).collect()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use crate::node::Node;
    use chess_engine::board::Board;
    use chess_engine::game::Game;

    // ========================================================================
    // Helper: create arena with a root node, return (arena, root_idx)
    // ========================================================================

    fn setup_arena() -> (Arena, NodeIndex) {
        let mut arena = Arena::new(256);
        let root_idx = arena.alloc(Node::root());
        (arena, root_idx)
    }

    // ========================================================================
    // 1. Starting position expansion with uniform priors
    // ========================================================================

    #[test]
    fn expand_starting_position_creates_20_children() {
        let (mut arena, root_idx) = setup_arena();
        let game = Game::new();

        let result = expand(root_idx, &game, None, &mut arena);

        // Should be Expanded, not Terminal or AlreadyExpanded.
        assert!(matches!(result, ExpandResult::Expanded));

        // Starting position has exactly 20 legal moves.
        let root = &arena[root_idx];
        assert_eq!(root.num_children, 20);
        assert!(root.is_expanded);

        // Each child should have uniform prior = 1/20 = 0.05.
        let expected_prior = 1.0 / 20.0;
        for child_idx in root.children(&arena) {
            let child = &arena[child_idx];
            assert!(
                (child.prior - expected_prior).abs() < 1e-6,
                "Expected prior ~{}, got {}",
                expected_prior,
                child.prior,
            );
        }
    }

    // ========================================================================
    // 2. Custom policy priors are correctly normalized
    // ========================================================================

    #[test]
    fn expand_with_custom_policy_normalizes_priors() {
        let (mut arena, root_idx) = setup_arena();
        let game = Game::new();

        // 20 legal moves in starting position. Provide unnormalized policy.
        let num_moves = game.legal_moves().len();
        assert_eq!(num_moves, 20);

        // Create policy that sums to 10.0 (0.5 each).
        let raw_policy: Vec<f32> = vec![0.5; num_moves];

        let result = expand(root_idx, &game, Some(&raw_policy), &mut arena);
        assert!(matches!(result, ExpandResult::Expanded));

        // After normalization, each prior should be 0.5/10.0 = 0.05.
        let expected = 0.05;
        let root = &arena[root_idx];
        for child_idx in root.children(&arena) {
            let child = &arena[child_idx];
            assert!(
                (child.prior - expected).abs() < 1e-6,
                "Expected prior ~{}, got {}",
                expected,
                child.prior,
            );
        }
    }

    #[test]
    fn expand_with_nonuniform_policy() {
        let (mut arena, root_idx) = setup_arena();
        let game = Game::new();

        let num_moves = game.legal_moves().len();
        assert_eq!(num_moves, 20);

        // Give the first move a large weight, the rest small weights.
        let mut raw_policy = vec![1.0; num_moves];
        raw_policy[0] = 19.0; // first move gets weight 19, others get 1
        // Sum = 19 + 19*1 = 38. So first prior = 19/38 = 0.5, others = 1/38.
        let sum: f32 = raw_policy.iter().sum();

        let result = expand(root_idx, &game, Some(&raw_policy), &mut arena);
        assert!(matches!(result, ExpandResult::Expanded));

        // Collect children's priors. Since add_child prepends, the iteration
        // order is reversed from the order we added them.
        let root = &arena[root_idx];
        let children: Vec<NodeIndex> = root.children(&arena).collect();
        assert_eq!(children.len(), 20);

        // The last child in the iteration corresponds to the first legal move
        // (because add_child prepends).
        let first_move_prior = arena[children[num_moves - 1]].prior;
        let expected_first = 19.0 / sum;
        assert!(
            (first_move_prior - expected_first).abs() < 1e-5,
            "First move prior: expected {}, got {}",
            expected_first,
            first_move_prior,
        );

        let other_expected = 1.0 / sum;
        for &child_idx in &children[..num_moves - 1] {
            let p = arena[child_idx].prior;
            assert!(
                (p - other_expected).abs() < 1e-5,
                "Other prior: expected {}, got {}",
                other_expected,
                p,
            );
        }
    }

    // ========================================================================
    // 3. Terminal node — checkmate
    // ========================================================================

    #[test]
    fn expand_checkmate_returns_terminal_loss() {
        let (mut arena, root_idx) = setup_arena();

        // Fool's mate position: White is checkmated, it's White's turn,
        // Black wins.
        let board = Board::from_fen(
            "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        )
        .expect("valid FEN");
        let game = Game::from_board(board);

        let result = expand(root_idx, &game, None, &mut arena);

        // Should be Terminal with -1.0 (loss for the player to move, White).
        match result {
            ExpandResult::Terminal(value) => {
                assert!(
                    (value - (-1.0)).abs() < f32::EPSILON,
                    "Checkmate value should be -1.0, got {}",
                    value,
                );
            }
            _ => panic!("Expected Terminal, got something else"),
        }

        // Node should be marked as expanded but have no children.
        let root = &arena[root_idx];
        assert!(root.is_expanded);
        assert_eq!(root.num_children, 0);
        assert!(root.is_terminal_leaf());
    }

    // ========================================================================
    // 4. Terminal node — stalemate
    // ========================================================================

    #[test]
    fn expand_stalemate_returns_terminal_draw() {
        let (mut arena, root_idx) = setup_arena();

        // Classic stalemate: Black king on a8, White queen on b6, White king on c8.
        // Black to move, no legal moves but not in check.
        let board = Board::from_fen("k7/8/1Q6/8/8/8/8/2K5 b - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        let result = expand(root_idx, &game, None, &mut arena);

        match result {
            ExpandResult::Terminal(value) => {
                assert!(
                    value.abs() < f32::EPSILON,
                    "Stalemate value should be 0.0, got {}",
                    value,
                );
            }
            _ => panic!("Expected Terminal for stalemate, got something else"),
        }

        let root = &arena[root_idx];
        assert!(root.is_expanded);
        assert_eq!(root.num_children, 0);
        assert!(root.is_terminal_leaf());
    }

    // ========================================================================
    // 5. Already expanded — second expansion is no-op
    // ========================================================================

    #[test]
    fn expand_already_expanded_returns_already_expanded() {
        let (mut arena, root_idx) = setup_arena();
        let game = Game::new();

        // First expansion.
        let result1 = expand(root_idx, &game, None, &mut arena);
        assert!(matches!(result1, ExpandResult::Expanded));
        assert!(arena[root_idx].is_expanded);
        let num_children = arena[root_idx].num_children;
        assert_eq!(num_children, 20);

        // Second expansion — should be AlreadyExpanded.
        let result2 = expand(root_idx, &game, None, &mut arena);
        assert!(matches!(result2, ExpandResult::AlreadyExpanded));

        // State should not have changed.
        assert_eq!(arena[root_idx].num_children, num_children);
    }

    // ========================================================================
    // 6. Priors sum to ~1.0
    // ========================================================================

    #[test]
    fn priors_sum_to_one_uniform() {
        let (mut arena, root_idx) = setup_arena();
        let game = Game::new();

        expand(root_idx, &game, None, &mut arena);

        let root = &arena[root_idx];
        let prior_sum: f32 = root
            .children(&arena)
            .map(|idx| arena[idx].prior)
            .sum();

        assert!(
            (prior_sum - 1.0).abs() < 1e-5,
            "Priors should sum to ~1.0, got {}",
            prior_sum,
        );
    }

    #[test]
    fn priors_sum_to_one_custom_policy() {
        let (mut arena, root_idx) = setup_arena();
        let game = Game::new();

        // Use varied policy values.
        let num_moves = 20;
        let raw_policy: Vec<f32> = (0..num_moves).map(|i| (i + 1) as f32).collect();

        expand(root_idx, &game, Some(&raw_policy), &mut arena);

        let root = &arena[root_idx];
        let prior_sum: f32 = root
            .children(&arena)
            .map(|idx| arena[idx].prior)
            .sum();

        assert!(
            (prior_sum - 1.0).abs() < 1e-5,
            "Priors should sum to ~1.0, got {}",
            prior_sum,
        );
    }

    // ========================================================================
    // 7. Children are accessible with valid moves and priors
    // ========================================================================

    #[test]
    fn children_have_valid_moves_and_priors() {
        let (mut arena, root_idx) = setup_arena();
        let game = Game::new();

        let legal_moves = game.legal_moves();
        expand(root_idx, &game, None, &mut arena);

        let root = &arena[root_idx];
        let children: Vec<NodeIndex> = root.children(&arena).collect();
        assert_eq!(children.len(), legal_moves.len());

        for child_idx in &children {
            let child = &arena[*child_idx];

            // Prior should be positive and at most 1.0.
            assert!(child.prior > 0.0, "Prior should be positive");
            assert!(child.prior <= 1.0, "Prior should be at most 1.0");

            // Visit count should be 0 (fresh node).
            assert_eq!(child.visit_count, 0);

            // Total value should be 0.
            assert_eq!(child.total_value, 0.0);

            // Should not be expanded yet.
            assert!(!child.is_expanded);

            // Move should be one of the legal moves.
            assert!(
                legal_moves.contains(&child.mv),
                "Child move {:?} should be a legal move",
                child.mv,
            );
        }
    }

    // ========================================================================
    // 8. Node is marked as expanded
    // ========================================================================

    #[test]
    fn node_marked_as_expanded_after_expansion() {
        let (mut arena, root_idx) = setup_arena();
        let game = Game::new();

        // Before expansion.
        assert!(!arena[root_idx].is_expanded);
        assert!(arena[root_idx].is_leaf());

        // After expansion.
        expand(root_idx, &game, None, &mut arena);

        assert!(arena[root_idx].is_expanded);
        assert!(!arena[root_idx].is_leaf());
    }

    // ========================================================================
    // Additional: terminal draw types
    // ========================================================================

    #[test]
    fn expand_insufficient_material_returns_terminal_draw() {
        let (mut arena, root_idx) = setup_arena();

        // King vs King: insufficient material.
        let board = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        let result = expand(root_idx, &game, None, &mut arena);

        match result {
            ExpandResult::Terminal(value) => {
                assert!(
                    value.abs() < f32::EPSILON,
                    "Insufficient material value should be 0.0, got {}",
                    value,
                );
            }
            _ => panic!("Expected Terminal for insufficient material"),
        }
    }

    // ========================================================================
    // Additional: policy length mismatch panics
    // ========================================================================

    #[test]
    #[should_panic(expected = "Policy length")]
    fn expand_policy_length_mismatch_panics() {
        let (mut arena, root_idx) = setup_arena();
        let game = Game::new();

        // Starting position has 20 legal moves. Provide wrong-length policy.
        let bad_policy = vec![0.1; 5];
        expand(root_idx, &game, Some(&bad_policy), &mut arena);
    }

    // ========================================================================
    // Additional: zero policy falls back to uniform
    // ========================================================================

    #[test]
    fn expand_zero_policy_falls_back_to_uniform() {
        let (mut arena, root_idx) = setup_arena();
        let game = Game::new();

        let num_moves = game.legal_moves().len();
        let zero_policy = vec![0.0; num_moves];

        expand(root_idx, &game, Some(&zero_policy), &mut arena);

        let root = &arena[root_idx];
        let expected = 1.0 / num_moves as f32;
        for child_idx in root.children(&arena) {
            let child = &arena[child_idx];
            assert!(
                (child.prior - expected).abs() < 1e-6,
                "Zero policy should fall back to uniform: expected {}, got {}",
                expected,
                child.prior,
            );
        }
    }

    // ========================================================================
    // Additional: another stalemate position
    // ========================================================================

    #[test]
    fn expand_stalemate_king_trapped() {
        let (mut arena, root_idx) = setup_arena();

        // Black king on f8, White pawn on f7, White king on f6.
        // Black can't move: stalemate.
        let board = Board::from_fen("5k2/5P2/5K2/8/8/8/8/8 b - - 0 1")
            .expect("valid FEN");
        let game = Game::from_board(board);

        let result = expand(root_idx, &game, None, &mut arena);

        match result {
            ExpandResult::Terminal(value) => {
                assert!(
                    value.abs() < f32::EPSILON,
                    "Stalemate value should be 0.0, got {}",
                    value,
                );
            }
            _ => panic!("Expected Terminal for stalemate"),
        }
    }

    // ========================================================================
    // Additional: arena grows correctly
    // ========================================================================

    #[test]
    fn expand_increases_arena_size_correctly() {
        let (mut arena, root_idx) = setup_arena();
        let game = Game::new();

        // Before: just the root node.
        assert_eq!(arena.len(), 1);

        expand(root_idx, &game, None, &mut arena);

        // After: root + 20 children = 21.
        assert_eq!(arena.len(), 21);
    }
}
