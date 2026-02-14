//! Backpropagation for MCTS.
//!
//! After a leaf node is evaluated (by the neural network or by detecting a
//! terminal state), the resulting value must be propagated back up through every
//! node on the selection path. This updates each node's visit count and
//! cumulative value so that future selections are better informed.
//!
//! ## Sign flipping
//!
//! Chess is a zero-sum game: what is good for one player is equally bad for the
//! opponent. The neural network value head produces a score from the perspective
//! of the player **to move at the evaluated position** (the leaf). As we walk
//! back toward the root, the player to move alternates at each level, so we
//! **negate the value at every step**:
//!
//! - Leaf (depth N): receives `+leaf_value`
//! - Parent (depth N-1): receives `-leaf_value`
//! - Grandparent (depth N-2): receives `+leaf_value`
//! - … and so on.

use crate::arena::Arena;
use crate::node::NodeIndex;

/// Backpropagate a leaf evaluation value up through the search path.
///
/// Starting from the leaf, walks back to the root, updating each node:
/// - `visit_count += 1`
/// - `total_value += value` (with sign flip at each level)
///
/// The value is negated at each level because chess is zero-sum:
/// what's good for one player is bad for the opponent.
///
/// # Arguments
///
/// * `path`       - Node indices from root to leaf (`path[0]` = root,
///                  `path[last]` = leaf).
/// * `leaf_value` - Value from the perspective of the player at the **leaf**
///                  node (+1 = winning for player to move at leaf, -1 = losing).
/// * `arena`      - The node arena.
///
/// # Panics
///
/// Panics if any `NodeIndex` in `path` is `NULL_NODE` or out of bounds
/// (delegated to `Arena::get_mut`).
pub fn backup(path: &[NodeIndex], leaf_value: f32, arena: &mut Arena) {
    // Walk from leaf to root.
    // The leaf gets the value as-is (from its own perspective).
    // The parent gets -value (opponent's perspective).
    // The grandparent gets +value again, etc.
    let mut value = leaf_value;
    for &node_idx in path.iter().rev() {
        let node = arena.get_mut(node_idx);
        node.visit_count += 1;
        node.total_value += value;
        value = -value; // negate for opponent
    }
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

    /// Helper: create a dummy move so we can build nodes quickly.
    fn dummy_move() -> Move {
        Move::new(Square::E2, Square::E4)
    }

    // -----------------------------------------------------------------
    // 1. Single node path (root only)
    // -----------------------------------------------------------------

    #[test]
    fn single_node_path() {
        let mut arena = Arena::new(16);
        let root = arena.alloc(Node::root());

        backup(&[root], 0.5, &mut arena);

        assert_eq!(arena[root].visit_count, 1);
        assert!((arena[root].total_value - 0.5).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------
    // 2. Two-node path (root -> leaf)
    // -----------------------------------------------------------------

    #[test]
    fn two_node_path() {
        let mut arena = Arena::new(16);
        let root = arena.alloc(Node::root());
        let leaf = arena.alloc(Node::new(dummy_move(), 0.3));
        arena.add_child(root, leaf);

        backup(&[root, leaf], 1.0, &mut arena);

        // Leaf: receives +1.0 (its own perspective)
        assert_eq!(arena[leaf].visit_count, 1);
        assert!((arena[leaf].total_value - 1.0).abs() < f32::EPSILON);

        // Root: receives -1.0 (opponent's perspective)
        assert_eq!(arena[root].visit_count, 1);
        assert!((arena[root].total_value - (-1.0)).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------
    // 3. Three-node path
    // -----------------------------------------------------------------

    #[test]
    fn three_node_path() {
        let mut arena = Arena::new(16);
        let n0 = arena.alloc(Node::root());
        let n1 = arena.alloc(Node::new(dummy_move(), 0.4));
        let n2 = arena.alloc(Node::new(dummy_move(), 0.2));
        arena.add_child(n0, n1);
        arena.add_child(n1, n2);

        backup(&[n0, n1, n2], 0.5, &mut arena);

        // n2 (leaf): +0.5
        assert_eq!(arena[n2].visit_count, 1);
        assert!((arena[n2].total_value - 0.5).abs() < f32::EPSILON);

        // n1 (middle): -0.5
        assert_eq!(arena[n1].visit_count, 1);
        assert!((arena[n1].total_value - (-0.5)).abs() < f32::EPSILON);

        // n0 (root): +0.5
        assert_eq!(arena[n0].visit_count, 1);
        assert!((arena[n0].total_value - 0.5).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------
    // 4. Multiple backups on same path
    // -----------------------------------------------------------------

    #[test]
    fn multiple_backups_accumulate() {
        let mut arena = Arena::new(16);
        let root = arena.alloc(Node::root());
        let leaf = arena.alloc(Node::new(dummy_move(), 0.5));
        arena.add_child(root, leaf);

        // First backup: leaf_value = +1.0
        backup(&[root, leaf], 1.0, &mut arena);
        // Second backup: leaf_value = -0.5
        backup(&[root, leaf], -0.5, &mut arena);

        // Leaf: 2 visits, total_value = 1.0 + (-0.5) = 0.5
        assert_eq!(arena[leaf].visit_count, 2);
        assert!((arena[leaf].total_value - 0.5).abs() < f32::EPSILON);

        // Root: 2 visits, total_value = -1.0 + 0.5 = -0.5
        assert_eq!(arena[root].visit_count, 2);
        assert!((arena[root].total_value - (-0.5)).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------
    // 5. Q-value after backup
    // -----------------------------------------------------------------

    #[test]
    fn q_value_after_backup() {
        let mut arena = Arena::new(16);
        let root = arena.alloc(Node::root());
        let child = arena.alloc(Node::new(dummy_move(), 0.6));
        let leaf = arena.alloc(Node::new(dummy_move(), 0.2));
        arena.add_child(root, child);
        arena.add_child(child, leaf);

        // backup value 0.8 along [root, child, leaf]
        backup(&[root, child, leaf], 0.8, &mut arena);

        // leaf: 1 visit, total = 0.8 => Q = 0.8
        assert!((arena[leaf].q_value() - 0.8).abs() < f32::EPSILON);

        // child: 1 visit, total = -0.8 => Q = -0.8
        assert!((arena[child].q_value() - (-0.8)).abs() < f32::EPSILON);

        // root: 1 visit, total = 0.8 => Q = 0.8
        assert!((arena[root].q_value() - 0.8).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------
    // 6. Two paths sharing a prefix
    // -----------------------------------------------------------------

    #[test]
    fn two_paths_sharing_prefix() {
        let mut arena = Arena::new(16);
        let root = arena.alloc(Node::root());
        let a = arena.alloc(Node::new(dummy_move(), 0.5));
        let b = arena.alloc(Node::new(Move::new(Square::D2, Square::D4), 0.3));
        let c = arena.alloc(Node::new(Move::new(Square::G1, Square::F3), 0.2));
        arena.add_child(root, a);
        arena.add_child(a, b);
        arena.add_child(a, c);

        // Path 1: root -> A -> B, value = 1.0
        backup(&[root, a, b], 1.0, &mut arena);

        // Path 2: root -> A -> C, value = -1.0
        backup(&[root, a, c], -1.0, &mut arena);

        // Root: 2 visits. First backup: root gets +1.0 (negated twice from leaf).
        //   Path1: B=+1.0, A=-1.0, root=+1.0
        //   Path2: C=-1.0, A=+1.0, root=-1.0
        //   Root total = +1.0 + (-1.0) = 0.0
        assert_eq!(arena[root].visit_count, 2);
        assert!((arena[root].total_value - 0.0).abs() < f32::EPSILON);

        // A: 2 visits, total = -1.0 + 1.0 = 0.0
        assert_eq!(arena[a].visit_count, 2);
        assert!((arena[a].total_value - 0.0).abs() < f32::EPSILON);

        // B: 1 visit, total = 1.0
        assert_eq!(arena[b].visit_count, 1);
        assert!((arena[b].total_value - 1.0).abs() < f32::EPSILON);

        // C: 1 visit, total = -1.0
        assert_eq!(arena[c].visit_count, 1);
        assert!((arena[c].total_value - (-1.0)).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------
    // 7. Zero value backup (draw)
    // -----------------------------------------------------------------

    #[test]
    fn zero_value_backup() {
        let mut arena = Arena::new(16);
        let n0 = arena.alloc(Node::root());
        let n1 = arena.alloc(Node::new(dummy_move(), 0.5));
        let n2 = arena.alloc(Node::new(dummy_move(), 0.3));
        arena.add_child(n0, n1);
        arena.add_child(n1, n2);

        backup(&[n0, n1, n2], 0.0, &mut arena);

        // All nodes get +1 visit, 0.0 value (negating zero is still zero)
        for &idx in &[n0, n1, n2] {
            assert_eq!(arena[idx].visit_count, 1);
            assert!((arena[idx].total_value - 0.0).abs() < f32::EPSILON);
        }
    }

    // -----------------------------------------------------------------
    // 8. Long path — verify alternating signs
    // -----------------------------------------------------------------

    #[test]
    fn long_path_alternating_signs() {
        let mut arena = Arena::new(32);
        let mut path = Vec::new();

        // Create a chain of 10 nodes
        let root = arena.alloc(Node::root());
        path.push(root);

        for i in 1..10 {
            let node = arena.alloc(Node::new(dummy_move(), 0.1 * i as f32));
            if i == 1 {
                arena.add_child(root, node);
            } else {
                arena.add_child(path[i - 1], node);
            }
            path.push(node);
        }

        assert_eq!(path.len(), 10);

        let leaf_value: f32 = 0.7;
        backup(&path, leaf_value, &mut arena);

        // Verify each node. Walking from the leaf (index 9) to root (index 0):
        // leaf (idx 9): +0.7
        // idx 8: -0.7
        // idx 7: +0.7
        // ...
        // The sign alternates based on distance from the leaf.
        for (i, &idx) in path.iter().enumerate() {
            assert_eq!(arena[idx].visit_count, 1);

            // Distance from leaf = (path.len() - 1) - i
            let dist_from_leaf = (path.len() - 1) - i;
            let expected = if dist_from_leaf % 2 == 0 {
                leaf_value
            } else {
                -leaf_value
            };

            assert!(
                (arena[idx].total_value - expected).abs() < f32::EPSILON,
                "Node at path[{}]: expected {}, got {}",
                i,
                expected,
                arena[idx].total_value
            );
        }
    }

    // -----------------------------------------------------------------
    // 9. Empty path — no-op, should not panic
    // -----------------------------------------------------------------

    #[test]
    fn empty_path_is_noop() {
        let mut arena = Arena::new(16);
        let _root = arena.alloc(Node::root());

        // An empty path should do nothing.
        backup(&[], 1.0, &mut arena);

        // Root should be untouched.
        assert_eq!(arena[NodeIndex(0)].visit_count, 0);
        assert!((arena[NodeIndex(0)].total_value - 0.0).abs() < f32::EPSILON);
    }
}
