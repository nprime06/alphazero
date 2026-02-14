//! MCTS tree node.
//!
//! Each `Node` represents a state reached by playing a specific move from its
//! parent. The root node has a dummy move (a1a1) since it has no parent.
//!
//! Nodes are stored in an [`Arena`](crate::arena::Arena) and reference each
//! other via [`NodeIndex`] values. Children of a node are stored as a singly
//! linked list: the parent holds a `first_child` index, and each child holds
//! a `next_sibling` index pointing to the next child (or `NULL_NODE`).
//!
//! ## Memory layout
//!
//! Fields are ordered so the struct packs into 28 bytes (with padding to 32),
//! allowing two nodes per 64-byte cache line:
//!
//! | Field         | Type      | Size |
//! |---------------|-----------|------|
//! | visit_count   | u32       | 4    |
//! | total_value   | f32       | 4    |
//! | prior         | f32       | 4    |
//! | mv            | Move      | 4    |
//! | first_child   | NodeIndex | 4    |
//! | next_sibling  | NodeIndex | 4    |
//! | num_children  | u16       | 2    |
//! | is_expanded   | bool      | 1    |
//! | (padding)     |           | 1    |
//! | **Total**     |           | **28** |

use chess_engine::moves::Move;
use chess_engine::types::Square;

use crate::arena::Arena;

// =============================================================================
// NodeIndex
// =============================================================================

/// A lightweight handle into the [`Arena`]. This is a newtype around `u32`,
/// giving us up to ~4 billion addressable nodes — far more than any realistic
/// MCTS tree.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeIndex(pub u32);

/// Sentinel value meaning "no node". Analogous to a null pointer.
/// We use `u32::MAX` because a real arena will never have 4 billion nodes.
pub const NULL_NODE: NodeIndex = NodeIndex(u32::MAX);

// =============================================================================
// Node
// =============================================================================

/// A single node in the MCTS search tree.
///
/// See the [module-level documentation](self) for the memory layout rationale.
#[derive(Clone, Debug)]
pub struct Node {
    /// Number of times this node has been visited during search.
    pub visit_count: u32,

    /// Cumulative value backpropagated through this node.
    /// The mean action-value is `Q = total_value / visit_count`.
    pub total_value: f32,

    /// Prior probability P(s, a) from the neural network policy head.
    /// Represents how promising this move looked before any search.
    pub prior: f32,

    /// The move that was played from the parent to reach this node.
    /// For the root node, this is a dummy move (a1a1).
    pub mv: Move,

    /// Index of this node's first child in the arena, or `NULL_NODE` if the
    /// node has no children (either unexpanded or a terminal state).
    pub first_child: NodeIndex,

    /// Index of this node's next sibling, forming a linked list of the
    /// parent's children. `NULL_NODE` if this is the last (or only) child.
    pub next_sibling: NodeIndex,

    /// Number of children this node has. Stored explicitly to avoid walking
    /// the entire sibling chain just to count children.
    pub num_children: u16,

    /// Whether this node has been expanded (i.e., its children have been
    /// generated). A node can be expanded and still have zero children if
    /// the position is a terminal state (checkmate or stalemate).
    pub is_expanded: bool,
}

impl Node {
    /// Creates a new unexpanded node representing the result of playing `mv`
    /// from the parent position. `prior` is the neural network's policy
    /// probability for this move.
    #[inline]
    pub fn new(mv: Move, prior: f32) -> Self {
        Node {
            visit_count: 0,
            total_value: 0.0,
            prior,
            mv,
            first_child: NULL_NODE,
            next_sibling: NULL_NODE,
            num_children: 0,
            is_expanded: false,
        }
    }

    /// Creates a root node for the search tree. The root has no meaningful
    /// move (we use a dummy a1->a1) and zero prior.
    #[inline]
    pub fn root() -> Self {
        Node {
            visit_count: 0,
            total_value: 0.0,
            prior: 0.0,
            mv: Move::new(Square::A1, Square::A1),
            first_child: NULL_NODE,
            next_sibling: NULL_NODE,
            num_children: 0,
            is_expanded: false,
        }
    }

    /// Returns the mean action-value Q(s, a) = total_value / visit_count.
    ///
    /// Returns 0.0 if the node has never been visited, which is a safe
    /// default — unvisited nodes will be selected based on their prior
    /// and the exploration term, not their Q value.
    #[inline]
    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f32
        }
    }

    /// Returns `true` if this node has not yet been expanded.
    /// A leaf node needs to be evaluated by the neural network before
    /// its children can be generated.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        !self.is_expanded
    }

    /// Returns `true` if this node has been expanded but has no children.
    /// This means the position is a terminal state (checkmate, stalemate,
    /// or draw by rule), so the game is over and no further moves exist.
    #[inline]
    pub fn is_terminal_leaf(&self) -> bool {
        self.is_expanded && self.num_children == 0
    }

    /// Returns an iterator over this node's children, following the
    /// sibling linked list through the given arena.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for child_idx in node.children(&arena) {
    ///     let child = &arena[child_idx];
    ///     // ... do something with child
    /// }
    /// ```
    #[inline]
    pub fn children<'a>(&self, arena: &'a Arena) -> ChildrenIter<'a> {
        ChildrenIter {
            arena,
            current: self.first_child,
        }
    }

}

// =============================================================================
// ChildrenIter
// =============================================================================

/// Iterator over a node's children, walking the sibling linked list.
///
/// Yields `NodeIndex` values. To get the actual `Node`, index into the arena:
/// ```ignore
/// let child_node = &arena[child_index];
/// ```
pub struct ChildrenIter<'a> {
    arena: &'a Arena,
    current: NodeIndex,
}

impl<'a> Iterator for ChildrenIter<'a> {
    type Item = NodeIndex;

    #[inline]
    fn next(&mut self) -> Option<NodeIndex> {
        if self.current == NULL_NODE {
            None
        } else {
            let idx = self.current;
            self.current = self.arena[idx].next_sibling;
            Some(idx)
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chess_engine::moves::Move;
    use chess_engine::types::Square;
    use std::mem;

    #[test]
    fn node_size_fits_in_32_bytes() {
        // Two nodes per 64-byte cache line is our target.
        let size = mem::size_of::<Node>();
        assert!(
            size <= 32,
            "Node is {} bytes, must be <= 32 to fit two per cache line",
            size
        );
    }

    #[test]
    fn root_node_defaults() {
        let root = Node::root();
        assert_eq!(root.visit_count, 0);
        assert_eq!(root.total_value, 0.0);
        assert_eq!(root.prior, 0.0);
        assert_eq!(root.first_child, NULL_NODE);
        assert_eq!(root.next_sibling, NULL_NODE);
        assert_eq!(root.num_children, 0);
        assert!(!root.is_expanded);
        assert!(root.is_leaf());
    }

    #[test]
    fn new_node_with_move_and_prior() {
        let mv = Move::new(Square::E2, Square::E4);
        let node = Node::new(mv, 0.75);
        assert_eq!(node.mv, mv);
        assert_eq!(node.prior, 0.75);
        assert_eq!(node.visit_count, 0);
        assert_eq!(node.total_value, 0.0);
        assert!(node.is_leaf());
        assert!(!node.is_expanded);
        assert_eq!(node.first_child, NULL_NODE);
        assert_eq!(node.next_sibling, NULL_NODE);
        assert_eq!(node.num_children, 0);
    }

    #[test]
    fn q_value_unvisited_returns_zero() {
        let node = Node::root();
        assert_eq!(node.q_value(), 0.0);
    }

    #[test]
    fn q_value_after_visits() {
        let mut node = Node::root();
        // Simulate 4 visits with total value 3.0 => Q = 0.75
        node.visit_count = 4;
        node.total_value = 3.0;
        let q = node.q_value();
        assert!((q - 0.75).abs() < f32::EPSILON, "Expected 0.75, got {}", q);
    }

    #[test]
    fn q_value_negative() {
        let mut node = Node::root();
        // Losing position: 10 visits, total value -8.0 => Q = -0.8
        node.visit_count = 10;
        node.total_value = -8.0;
        let q = node.q_value();
        assert!(
            (q - (-0.8)).abs() < f32::EPSILON,
            "Expected -0.8, got {}",
            q
        );
    }

    #[test]
    fn is_leaf_true_initially() {
        let node = Node::new(Move::new(Square::A2, Square::A3), 0.1);
        assert!(node.is_leaf());
    }

    #[test]
    fn is_leaf_false_after_expansion() {
        let mut node = Node::new(Move::new(Square::A2, Square::A3), 0.1);
        node.is_expanded = true;
        assert!(!node.is_leaf());
    }

    #[test]
    fn is_terminal_leaf() {
        let mut node = Node::root();
        // Not expanded, no children -> not terminal (just unexpanded)
        assert!(!node.is_terminal_leaf());

        // Expanded with no children -> terminal (checkmate/stalemate)
        node.is_expanded = true;
        assert!(node.is_terminal_leaf());

        // Expanded with children -> not terminal
        node.num_children = 3;
        assert!(!node.is_terminal_leaf());
    }

    #[test]
    fn children_iter_empty() {
        let arena = Arena::new(16);
        let node = Node::root();
        let children: Vec<NodeIndex> = node.children(&arena).collect();
        assert!(children.is_empty());
    }

    #[test]
    fn children_iter_one_child() {
        let mut arena = Arena::new(16);
        let root_idx = arena.alloc(Node::root());
        let child = Node::new(Move::new(Square::E2, Square::E4), 0.5);
        let child_idx = arena.alloc(child);

        arena.add_child(root_idx, child_idx);

        // Collect children of root
        let root = &arena[root_idx];
        let children: Vec<NodeIndex> = root.children(&arena).collect();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0], child_idx);
        assert_eq!(root.num_children, 1);
    }

    #[test]
    fn children_iter_multiple_children() {
        let mut arena = Arena::new(16);
        let root_idx = arena.alloc(Node::root());

        let moves = [
            Move::new(Square::E2, Square::E4),
            Move::new(Square::D2, Square::D4),
            Move::new(Square::G1, Square::F3),
        ];

        let mut child_indices = Vec::new();
        for (i, mv) in moves.iter().enumerate() {
            let child = Node::new(*mv, (i as f32 + 1.0) * 0.1);
            let idx = arena.alloc(child);
            child_indices.push(idx);
        }

        // Add children to root
        for &idx in &child_indices {
            arena.add_child(root_idx, idx);
        }

        let root = &arena[root_idx];
        assert_eq!(root.num_children, 3);

        let children: Vec<NodeIndex> = root.children(&arena).collect();
        assert_eq!(children.len(), 3);

        // Because we prepend, the order is reversed: last added is first child
        assert_eq!(children[0], child_indices[2]); // Nf3 added last, appears first
        assert_eq!(children[1], child_indices[1]); // d4
        assert_eq!(children[2], child_indices[0]); // e4 added first, appears last

        // Verify each child's move
        assert_eq!(arena[children[0]].mv, moves[2]);
        assert_eq!(arena[children[1]].mv, moves[1]);
        assert_eq!(arena[children[2]].mv, moves[0]);
    }

    #[test]
    fn null_node_sentinel() {
        assert_eq!(NULL_NODE, NodeIndex(u32::MAX));
        // Make sure it's distinct from any plausible index
        assert_ne!(NULL_NODE, NodeIndex(0));
        assert_ne!(NULL_NODE, NodeIndex(1));
    }

    #[test]
    fn node_index_is_copy() {
        let idx = NodeIndex(42);
        let idx2 = idx;
        assert_eq!(idx, idx2);
    }
}
