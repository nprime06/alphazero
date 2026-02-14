//! # MCTS (Monte Carlo Tree Search)
//!
//! This crate implements the tree data structures for AlphaZero-style MCTS.
//!
//! The core design principles are:
//! - **Cache-friendly layout**: Nodes are stored contiguously in an arena allocator,
//!   and each node is kept small (~28 bytes) so two fit per 64-byte cache line.
//! - **Index-based references**: Instead of pointers or `Rc`, we use `NodeIndex`
//!   (a u32 newtype) to refer to nodes. This avoids lifetime issues during tree
//!   mutations and keeps each reference at 4 bytes.
//! - **Sibling-linked children**: Rather than storing a `Vec<NodeIndex>` per node
//!   (24 bytes of overhead each), children form a linked list via `next_sibling`
//!   fields, costing only 4 bytes per node.

pub mod arena;
pub mod backup;
pub mod batch;
pub mod config;
pub mod expand;
pub mod nn;
pub mod node;
pub mod reuse;
pub mod search;
pub mod select;
pub mod transposition;
