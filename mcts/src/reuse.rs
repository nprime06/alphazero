//! Tree reuse for MCTS.
//!
//! After making a move in the actual game, we can reuse the subtree
//! corresponding to the chosen move instead of building a new tree from
//! scratch. This saves many NN evaluations -- any nodes that were explored
//! under the chosen child in the previous search carry over to the next one.
//!
//! The implementation copies nodes into a **new** arena rather than trying to
//! prune the old one in place. This avoids fragmentation and keeps the arena
//! contiguous, which is important for cache performance.

use chess_engine::moves::Move;

use crate::arena::Arena;
use crate::node::{NodeIndex, NULL_NODE};

// =============================================================================
// Public API
// =============================================================================

/// Reuse the subtree rooted at the chosen move.
///
/// After the search selects a move and it is played on the board, the subtree
/// under that move becomes the starting point for the next search. All other
/// subtrees are discarded.
///
/// # Arguments
/// * `root` -- Current root node index.
/// * `chosen_move` -- The move that was played.
/// * `old_arena` -- The arena from the previous search.
///
/// # Returns
/// A new `(arena, root_index)` pair where:
/// - The root is the child node corresponding to `chosen_move`.
/// - All descendants are deep-copied into the new arena.
/// - Visit counts and values are preserved.
///
/// If the chosen move is not found among the root's children (e.g., the
/// opponent played an unexpected move, or the root has no children), returns
/// `None` and the caller should start a fresh search.
pub fn reuse_subtree(
    root: NodeIndex,
    chosen_move: Move,
    old_arena: &Arena,
) -> Option<(Arena, NodeIndex)> {
    // Find the child of root whose move matches `chosen_move`.
    let root_node = old_arena.get(root);
    let mut found_idx = NULL_NODE;

    for child_idx in root_node.children(old_arena) {
        if old_arena.get(child_idx).mv == chosen_move {
            found_idx = child_idx;
            break;
        }
    }

    if found_idx == NULL_NODE {
        return None;
    }

    // Estimate size: the old arena's size is an upper bound for the subtree.
    let mut new_arena = Arena::new(old_arena.len());
    let new_root = copy_subtree(found_idx, old_arena, &mut new_arena);

    Some((new_arena, new_root))
}

// =============================================================================
// Internal: deep copy
// =============================================================================

/// Recursively copy a subtree from `src_arena` into `dst_arena`.
///
/// Returns the index of the copied root node in the destination arena.
/// All descendants are copied with their visit counts and values preserved.
/// Sibling links are re-wired to point to the new copies in `dst_arena`.
fn copy_subtree(
    src_idx: NodeIndex,
    src_arena: &Arena,
    dst_arena: &mut Arena,
) -> NodeIndex {
    let src_node = src_arena.get(src_idx);

    // Create a copy of the node with cleared structural links (they will be
    // re-established when we add children below).
    let mut new_node = src_node.clone();
    new_node.first_child = NULL_NODE;
    new_node.next_sibling = NULL_NODE;
    new_node.num_children = 0;

    let new_idx = dst_arena.alloc(new_node);

    // Collect children indices first to avoid borrow issues.
    let children: Vec<NodeIndex> = src_node.children(src_arena).collect();

    for child_idx in children {
        let new_child = copy_subtree(child_idx, src_arena, dst_arena);
        dst_arena.add_child(new_idx, new_child);
    }

    new_idx
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use crate::node::{Node, NodeIndex};
    use chess_engine::moves::Move;
    use chess_engine::types::Square;

    /// Helper: create a move from two squares for concise test code.
    fn mv(from: Square, to: Square) -> Move {
        Move::new(from, to)
    }

    // ========================================================================
    // 1. Find correct child
    // ========================================================================

    #[test]
    fn reuse_finds_correct_child() {
        let mut arena = Arena::new(64);
        let root_idx = arena.alloc(Node::root());

        let move_e4 = mv(Square::E2, Square::E4);
        let move_d4 = mv(Square::D2, Square::D4);
        let move_nf3 = mv(Square::G1, Square::F3);

        // Create three children with distinct stats.
        let c1 = arena.alloc(Node::new(move_e4, 0.5));
        arena.get_mut(c1).visit_count = 100;
        arena.get_mut(c1).total_value = 60.0;

        let c2 = arena.alloc(Node::new(move_d4, 0.3));
        arena.get_mut(c2).visit_count = 80;
        arena.get_mut(c2).total_value = 40.0;

        let c3 = arena.alloc(Node::new(move_nf3, 0.2));
        arena.get_mut(c3).visit_count = 20;
        arena.get_mut(c3).total_value = 5.0;

        arena.add_child(root_idx, c1);
        arena.add_child(root_idx, c2);
        arena.add_child(root_idx, c3);

        // Reuse the subtree for d4.
        let (new_arena, new_root) = reuse_subtree(root_idx, move_d4, &arena).unwrap();

        let new_root_node = new_arena.get(new_root);
        assert_eq!(new_root_node.mv, move_d4);
        assert_eq!(new_root_node.visit_count, 80);
        assert_eq!(new_root_node.total_value, 40.0);
        assert_eq!(new_root_node.prior, 0.3);
    }

    // ========================================================================
    // 2. Deep copy preserves structure
    // ========================================================================

    #[test]
    fn deep_copy_preserves_structure() {
        let mut arena = Arena::new(64);
        let root_idx = arena.alloc(Node::root());

        let move_e4 = mv(Square::E2, Square::E4);
        let child_idx = arena.alloc(Node::new(move_e4, 0.5));
        arena.get_mut(child_idx).visit_count = 50;
        arena.get_mut(child_idx).is_expanded = true;
        arena.add_child(root_idx, child_idx);

        // Add two grandchildren under child_idx.
        let move_e5 = mv(Square::E7, Square::E5);
        let gc1 = arena.alloc(Node::new(move_e5, 0.6));
        arena.get_mut(gc1).visit_count = 30;
        arena.add_child(child_idx, gc1);

        let move_d5 = mv(Square::D7, Square::D5);
        let gc2 = arena.alloc(Node::new(move_d5, 0.4));
        arena.get_mut(gc2).visit_count = 20;
        arena.add_child(child_idx, gc2);

        // Reuse the subtree rooted at child (move e4).
        let (new_arena, new_root) = reuse_subtree(root_idx, move_e4, &arena).unwrap();

        let new_root_node = new_arena.get(new_root);
        assert_eq!(new_root_node.mv, move_e4);
        assert_eq!(new_root_node.visit_count, 50);
        assert!(new_root_node.is_expanded);
        assert_eq!(new_root_node.num_children, 2);

        // Verify grandchildren were copied.
        let grandchildren: Vec<NodeIndex> = new_root_node.children(&new_arena).collect();
        assert_eq!(grandchildren.len(), 2);

        // Children are prepended, so the order in the new arena reverses the
        // original insertion order. The original order was gc1 (e5) then gc2 (d5),
        // both prepended, so in old arena: first_child = gc2, then gc1.
        // After deep copy, the copy iterates gc2 then gc1 and prepends each,
        // resulting in first_child = gc1 (last prepended), then gc2.
        // Actually let's just check both grandchildren are present by moves:
        let gc_moves: Vec<Move> = grandchildren
            .iter()
            .map(|&idx| new_arena.get(idx).mv)
            .collect();
        assert!(gc_moves.contains(&move_e5));
        assert!(gc_moves.contains(&move_d5));

        // Verify visit counts are preserved.
        for &gc_idx in &grandchildren {
            let gc = new_arena.get(gc_idx);
            if gc.mv == move_e5 {
                assert_eq!(gc.visit_count, 30);
            } else {
                assert_eq!(gc.visit_count, 20);
            }
        }
    }

    // ========================================================================
    // 3. Visit counts preserved
    // ========================================================================

    #[test]
    fn visit_counts_preserved_after_reuse() {
        let mut arena = Arena::new(32);
        let root_idx = arena.alloc(Node::root());

        let move_e4 = mv(Square::E2, Square::E4);
        let child_idx = arena.alloc(Node::new(move_e4, 0.7));
        arena.get_mut(child_idx).visit_count = 200;
        arena.get_mut(child_idx).total_value = 150.0;
        arena.add_child(root_idx, child_idx);

        let (new_arena, new_root) = reuse_subtree(root_idx, move_e4, &arena).unwrap();

        let new_root_node = new_arena.get(new_root);
        assert_eq!(new_root_node.visit_count, 200);
        assert_eq!(new_root_node.total_value, 150.0);
        assert!((new_root_node.q_value() - 0.75).abs() < f32::EPSILON);
    }

    // ========================================================================
    // 4. Move not found returns None
    // ========================================================================

    #[test]
    fn move_not_found_returns_none() {
        let mut arena = Arena::new(32);
        let root_idx = arena.alloc(Node::root());

        let move_e4 = mv(Square::E2, Square::E4);
        let child_idx = arena.alloc(Node::new(move_e4, 0.5));
        arena.add_child(root_idx, child_idx);

        // Try to reuse with a move that doesn't exist.
        let move_d4 = mv(Square::D2, Square::D4);
        let result = reuse_subtree(root_idx, move_d4, &arena);
        assert!(result.is_none());
    }

    // ========================================================================
    // 5. Empty tree (root with no children) returns None
    // ========================================================================

    #[test]
    fn empty_tree_returns_none() {
        let mut arena = Arena::new(16);
        let root_idx = arena.alloc(Node::root());

        let move_e4 = mv(Square::E2, Square::E4);
        let result = reuse_subtree(root_idx, move_e4, &arena);
        assert!(result.is_none());
    }

    // ========================================================================
    // 6. New arena is independent from old arena
    // ========================================================================

    #[test]
    fn new_arena_is_independent() {
        let mut arena = Arena::new(32);
        let root_idx = arena.alloc(Node::root());

        let move_e4 = mv(Square::E2, Square::E4);
        let child_idx = arena.alloc(Node::new(move_e4, 0.5));
        arena.get_mut(child_idx).visit_count = 42;
        arena.add_child(root_idx, child_idx);

        let (mut new_arena, new_root) =
            reuse_subtree(root_idx, move_e4, &arena).unwrap();

        // Modify the new arena.
        new_arena.get_mut(new_root).visit_count = 999;

        // Old arena should be unchanged.
        assert_eq!(arena.get(child_idx).visit_count, 42);
        assert_eq!(new_arena.get(new_root).visit_count, 999);
    }

    // ========================================================================
    // 7. Integration with search
    // ========================================================================

    #[test]
    fn integration_with_search() {
        use chess_engine::game::Game;
        use crate::config::MctsConfig;
        use crate::search::{search_with_evaluator, UniformEvaluator};

        let game = Game::new();
        let config = MctsConfig {
            num_simulations: 100,
            dirichlet_epsilon: 0.0, // Deterministic for test stability.
            ..MctsConfig::default()
        };

        // Run first search.
        let result = search_with_evaluator(&game, &config, &UniformEvaluator);
        assert!(!result.move_visits.is_empty());

        // The best move is the one with the most visits.
        let best_move = result.move_visits[0].0;

        // We need to build a tree we can reuse from. Since `search_with_evaluator`
        // currently doesn't expose the arena, we test the reuse primitives directly
        // by building a mock tree that represents a plausible post-search state.
        let mut arena = Arena::new(256);
        let root_idx = arena.alloc(Node::root());
        arena.get_mut(root_idx).is_expanded = true;

        // Add children corresponding to legal moves with visit counts.
        for &(mv, visits) in &result.move_visits {
            let child_idx = arena.alloc(Node::new(mv, 0.05));
            arena.get_mut(child_idx).visit_count = visits;
            arena.get_mut(child_idx).total_value = visits as f32 * 0.5;
            arena.add_child(root_idx, child_idx);
        }

        // Reuse the subtree for the best move.
        let (new_arena, new_root) =
            reuse_subtree(root_idx, best_move, &arena).unwrap();

        let new_root_node = new_arena.get(new_root);
        assert_eq!(new_root_node.mv, best_move);
        assert_eq!(new_root_node.visit_count, result.move_visits[0].1);

        // The new arena should contain exactly 1 node (the child had no
        // children of its own in this mock).
        assert_eq!(new_arena.len(), 1);
    }
}
