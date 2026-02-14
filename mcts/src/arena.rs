//! Arena allocator for MCTS nodes.
//!
//! All [`Node`](crate::node::Node)s live in a single contiguous `Vec<Node>`,
//! referenced by [`NodeIndex`](crate::node::NodeIndex). This design gives us:
//!
//! - **Cache friendliness**: Nodes are packed contiguously in memory, so
//!   traversals hit L1/L2 cache far more often than a pointer-chasing tree.
//! - **Simple lifetime management**: No `Rc`, `RefCell`, or raw pointers.
//!   The arena owns all nodes and hands out indices.
//! - **Fast allocation**: Appending to a `Vec` is amortized O(1).
//! - **Cheap reuse**: `clear()` resets the length without deallocating,
//!   so the next search reuses the same memory.

use std::ops::{Index, IndexMut};

use crate::node::{Node, NodeIndex, NULL_NODE};

// =============================================================================
// Arena
// =============================================================================

/// Contiguous storage for MCTS tree nodes.
///
/// Nodes are allocated sequentially and referenced by [`NodeIndex`].
/// The arena never shrinks its backing allocation — calling [`clear`](Arena::clear)
/// resets the logical length to zero but keeps the memory for reuse.
pub struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    /// Creates a new arena with pre-allocated capacity for `capacity` nodes.
    ///
    /// Pre-allocating avoids repeated reallocations during search. A typical
    /// AlphaZero search might use 800 simulations with a branching factor of
    /// ~30, so capacities in the range of 50k-500k are common.
    pub fn new(capacity: usize) -> Self {
        Arena {
            nodes: Vec::with_capacity(capacity),
        }
    }

    /// Allocates a node in the arena and returns its index.
    ///
    /// This is an amortized O(1) operation (occasional reallocation when the
    /// `Vec` grows). The returned [`NodeIndex`] is valid for the lifetime of
    /// the arena (until [`clear`](Arena::clear) is called).
    ///
    /// # Panics
    ///
    /// Panics if the arena contains `u32::MAX` nodes (extremely unlikely — that
    /// would require ~128 GB of memory for the nodes alone).
    pub fn alloc(&mut self, node: Node) -> NodeIndex {
        let index = self.nodes.len();
        assert!(
            index < u32::MAX as usize,
            "Arena overflow: cannot allocate more than {} nodes",
            u32::MAX
        );
        self.nodes.push(node);
        NodeIndex(index as u32)
    }

    /// Returns an immutable reference to the node at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is `NULL_NODE` or out of bounds.
    #[inline]
    pub fn get(&self, index: NodeIndex) -> &Node {
        assert_ne!(index, NULL_NODE, "attempted to access NULL_NODE");
        &self.nodes[index.0 as usize]
    }

    /// Returns a mutable reference to the node at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is `NULL_NODE` or out of bounds.
    #[inline]
    pub fn get_mut(&mut self, index: NodeIndex) -> &mut Node {
        assert_ne!(index, NULL_NODE, "attempted to mutably access NULL_NODE");
        &mut self.nodes[index.0 as usize]
    }

    /// Resets the arena, logically removing all nodes.
    ///
    /// The backing memory is *not* freed — it will be reused by subsequent
    /// allocations. This is the intended way to prepare for a new search
    /// without the cost of reallocation.
    #[inline]
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// Returns the number of allocated (live) nodes.
    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the arena has no allocated nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the current capacity (number of nodes that can be stored
    /// without reallocation).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.nodes.capacity()
    }

    /// Adds `child_index` as a child of `parent_index`.
    ///
    /// The new child is **prepended** to the parent's sibling linked list,
    /// so it becomes the new `first_child`. This is O(1).
    ///
    /// This method lives on `Arena` rather than `Node` because both the
    /// parent and child are stored in the same arena, and Rust's borrow
    /// checker won't allow `&mut parent` and `&mut arena` simultaneously.
    /// Here we use index-based access to avoid that conflict.
    ///
    /// # Panics
    ///
    /// Panics if either index is `NULL_NODE` or out of bounds.
    pub fn add_child(&mut self, parent_index: NodeIndex, child_index: NodeIndex) {
        // Read the parent's current first_child so we can chain the new child.
        let old_first = self.nodes[parent_index.0 as usize].first_child;

        // Point the new child's next_sibling to the old first child.
        self.nodes[child_index.0 as usize].next_sibling = old_first;

        // Update the parent: new first_child and increment count.
        let parent = &mut self.nodes[parent_index.0 as usize];
        parent.first_child = child_index;
        parent.num_children += 1;
    }
}

// =============================================================================
// Index trait impls
// =============================================================================

impl Index<NodeIndex> for Arena {
    type Output = Node;

    #[inline]
    fn index(&self, index: NodeIndex) -> &Node {
        self.get(index)
    }
}

impl IndexMut<NodeIndex> for Arena {
    #[inline]
    fn index_mut(&mut self, index: NodeIndex) -> &mut Node {
        self.get_mut(index)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;
    use chess_engine::moves::Move;
    use chess_engine::types::Square;

    #[test]
    fn alloc_returns_sequential_indices() {
        let mut arena = Arena::new(16);
        let idx0 = arena.alloc(Node::root());
        let idx1 = arena.alloc(Node::new(Move::new(Square::E2, Square::E4), 0.5));
        let idx2 = arena.alloc(Node::new(Move::new(Square::D2, Square::D4), 0.3));

        assert_eq!(idx0, NodeIndex(0));
        assert_eq!(idx1, NodeIndex(1));
        assert_eq!(idx2, NodeIndex(2));
        assert_eq!(arena.len(), 3);
    }

    #[test]
    fn access_by_index_returns_correct_node() {
        let mut arena = Arena::new(16);
        let mv_e4 = Move::new(Square::E2, Square::E4);
        let mv_d4 = Move::new(Square::D2, Square::D4);

        let idx0 = arena.alloc(Node::root());
        let idx1 = arena.alloc(Node::new(mv_e4, 0.5));
        let idx2 = arena.alloc(Node::new(mv_d4, 0.3));

        // Check via get()
        assert_eq!(arena.get(idx0).prior, 0.0); // root
        assert_eq!(arena.get(idx1).mv, mv_e4);
        assert_eq!(arena.get(idx1).prior, 0.5);
        assert_eq!(arena.get(idx2).mv, mv_d4);
        assert_eq!(arena.get(idx2).prior, 0.3);

        // Check via Index trait (bracket syntax)
        assert_eq!(arena[idx1].mv, mv_e4);
        assert_eq!(arena[idx2].mv, mv_d4);
    }

    #[test]
    fn clear_resets_length_but_not_capacity() {
        let mut arena = Arena::new(64);

        for i in 0..50 {
            arena.alloc(Node::new(
                Move::new(Square::E2, Square::E4),
                i as f32 * 0.01,
            ));
        }
        assert_eq!(arena.len(), 50);
        let cap_before = arena.capacity();
        assert!(cap_before >= 50);

        arena.clear();
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
        // Capacity should not decrease after clear
        assert_eq!(arena.capacity(), cap_before);
    }

    #[test]
    fn allocate_many_nodes() {
        let n = 100_000;
        let mut arena = Arena::new(n);

        for i in 0..n {
            arena.alloc(Node::new(
                Move::new(Square::E2, Square::E4),
                (i as f32) / (n as f32),
            ));
        }

        assert_eq!(arena.len(), n);

        // Spot-check a few nodes
        assert!((arena[NodeIndex(0)].prior - 0.0).abs() < f32::EPSILON);
        let mid = n / 2;
        let expected_prior = (mid as f32) / (n as f32);
        assert!(
            (arena[NodeIndex(mid as u32)].prior - expected_prior).abs() < 1e-5,
            "Expected prior ~{}, got {}",
            expected_prior,
            arena[NodeIndex(mid as u32)].prior
        );
        let last = n - 1;
        let expected_last = (last as f32) / (n as f32);
        assert!(
            (arena[NodeIndex(last as u32)].prior - expected_last).abs() < 1e-5,
            "Expected prior ~{}, got {}",
            expected_last,
            arena[NodeIndex(last as u32)].prior
        );
    }

    #[test]
    fn mutable_access_modifies_node() {
        let mut arena = Arena::new(8);
        let idx = arena.alloc(Node::root());

        assert_eq!(arena[idx].visit_count, 0);
        assert_eq!(arena[idx].total_value, 0.0);

        // Mutate via get_mut()
        arena.get_mut(idx).visit_count = 42;
        arena.get_mut(idx).total_value = 31.5;
        assert_eq!(arena[idx].visit_count, 42);
        assert_eq!(arena[idx].total_value, 31.5);

        // Mutate via IndexMut trait (bracket syntax)
        arena[idx].is_expanded = true;
        assert!(arena[idx].is_expanded);
    }

    #[test]
    fn new_arena_is_empty() {
        let arena = Arena::new(100);
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
        assert!(arena.capacity() >= 100);
    }

    #[test]
    #[should_panic(expected = "NULL_NODE")]
    fn get_null_node_panics() {
        let arena = Arena::new(8);
        let _ = arena.get(NULL_NODE);
    }

    #[test]
    #[should_panic(expected = "NULL_NODE")]
    fn get_mut_null_node_panics() {
        let mut arena = Arena::new(8);
        let _ = arena.get_mut(NULL_NODE);
    }

    #[test]
    fn clear_and_reuse() {
        let mut arena = Arena::new(16);
        let _idx0 = arena.alloc(Node::root());
        let _idx1 = arena.alloc(Node::new(Move::new(Square::E2, Square::E4), 0.5));
        assert_eq!(arena.len(), 2);

        arena.clear();
        assert_eq!(arena.len(), 0);

        // Allocate again — indices restart from 0
        let idx_new = arena.alloc(Node::root());
        assert_eq!(idx_new, NodeIndex(0));
        assert_eq!(arena.len(), 1);
    }
}
