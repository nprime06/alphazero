//! PUCT selection with First Play Urgency (FPU).
//!
//! During the selection phase of MCTS, we walk down the tree by repeatedly
//! choosing the child that maximises the PUCT score:
//!
//! ```text
//! UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(s,a))
//! ```
//!
//! For **unvisited** children (visit_count == 0) we apply First Play Urgency:
//! instead of using Q = 0 (which is over-optimistic in winning positions), we
//! set `Q_fpu = parent_Q - fpu_reduction`. This makes unvisited moves start
//! with a pessimistic estimate relative to the parent, so the search focuses
//! on the most promising moves first.

use crate::arena::Arena;
use crate::config::MctsConfig;
use crate::node::{NodeIndex, NULL_NODE};

// =============================================================================
// Public API
// =============================================================================

/// Calculate the PUCT score for a single child.
///
/// ```text
/// UCB = Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
/// ```
///
/// This is a pure function exposed for testing. It does **not** apply FPU —
/// the caller is responsible for passing the correct `child_q` (which may be
/// the FPU value for unvisited nodes).
#[inline]
pub fn puct_score(
    child_q: f32,
    child_prior: f32,
    child_visits: u32,
    parent_visits: u32,
    c_puct: f32,
) -> f32 {
    let exploration = c_puct * child_prior * (parent_visits as f32).sqrt() / (1 + child_visits) as f32;
    child_q + exploration
}

/// Select the child of `parent` with the highest PUCT score.
///
/// Returns the [`NodeIndex`] of the best child. For unvisited children, the
/// First Play Urgency heuristic is applied: their Q value is set to
/// `parent_Q - fpu_reduction` instead of the default 0.
///
/// When scores are tied, the first child encountered in the linked-list
/// iteration order wins (i.e. the last child that was added via
/// [`Arena::add_child`], since children are prepended).
///
/// # Panics
///
/// Panics if `parent` has no children (i.e. `first_child == NULL_NODE`).
pub fn select_child(
    parent: NodeIndex,
    arena: &Arena,
    config: &MctsConfig,
) -> NodeIndex {
    let parent_node = &arena[parent];
    assert_ne!(
        parent_node.first_child, NULL_NODE,
        "select_child called on a node with no children"
    );

    let parent_visits = parent_node.visit_count;
    let parent_q = parent_node.q_value();
    let fpu_value = parent_q - config.fpu_reduction;

    let mut best_index = NULL_NODE;
    let mut best_score = f32::NEG_INFINITY;

    for child_idx in parent_node.children(arena) {
        let child = &arena[child_idx];

        // Apply FPU: unvisited children use the pessimistic parent-based estimate
        let child_q = if child.visit_count == 0 {
            fpu_value
        } else {
            child.q_value()
        };

        let score = puct_score(
            child_q,
            child.prior,
            child.visit_count,
            parent_visits,
            config.c_puct,
        );

        // Strictly greater than: ties go to the first child encountered
        if score > best_score {
            best_score = score;
            best_index = child_idx;
        }
    }

    debug_assert_ne!(best_index, NULL_NODE, "no best child found despite children existing");
    best_index
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Arena;
    use crate::node::Node;
    use chess_engine::moves::Move;
    use chess_engine::types::Square;

    /// Helper: create a dummy move (doesn't matter for selection tests).
    fn dummy_move() -> Move {
        Move::new(Square::E2, Square::E4)
    }

    /// Helper: build a parent with children and set up the arena.
    /// Returns (arena, parent_idx, vec_of_child_indices).
    /// Children are added in order, so due to prepending, iteration order
    /// is reversed: children[last] is iterated first.
    fn setup_parent_with_children(
        parent_visits: u32,
        parent_total_value: f32,
        children: &[(f32, u32, f32)], // (prior, visits, total_value) per child
    ) -> (Arena, NodeIndex, Vec<NodeIndex>) {
        let mut arena = Arena::new(64);

        let mut parent = Node::root();
        parent.visit_count = parent_visits;
        parent.total_value = parent_total_value;
        parent.is_expanded = true;
        let parent_idx = arena.alloc(parent);

        let mut child_indices = Vec::new();
        for &(prior, visits, total_value) in children {
            let mut child = Node::new(dummy_move(), prior);
            child.visit_count = visits;
            child.total_value = total_value;
            let child_idx = arena.alloc(child);
            child_indices.push(child_idx);
        }

        // Add children to parent (prepend order means iteration reverses)
        for &idx in &child_indices {
            arena.add_child(parent_idx, idx);
        }

        (arena, parent_idx, child_indices)
    }

    // =========================================================================
    // puct_score tests
    // =========================================================================

    #[test]
    fn puct_score_basic_manual_calculation() {
        // Q=0.5, prior=0.3, child_visits=10, parent_visits=100, c_puct=2.5
        // exploration = 2.5 * 0.3 * sqrt(100) / (1 + 10) = 2.5 * 0.3 * 10 / 11 = 7.5 / 11 ≈ 0.6818
        // score = 0.5 + 0.6818 ≈ 1.1818
        let score = puct_score(0.5, 0.3, 10, 100, 2.5);
        let expected = 0.5 + 2.5 * 0.3 * 10.0 / 11.0;
        assert!(
            (score - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            score
        );
    }

    #[test]
    fn puct_score_with_zero_child_visits() {
        // Q=0.0, prior=0.4, child_visits=0, parent_visits=50, c_puct=2.5
        // exploration = 2.5 * 0.4 * sqrt(50) / (1 + 0) = 1.0 * sqrt(50) ≈ 7.071
        // score = 0.0 + 7.071 ≈ 7.071
        let score = puct_score(0.0, 0.4, 0, 50, 2.5);
        let expected = 2.5 * 0.4 * (50.0_f32).sqrt();
        assert!(
            (score - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            score
        );
    }

    #[test]
    fn puct_score_with_zero_parent_visits() {
        // When parent has 0 visits, sqrt(0) = 0, so exploration term = 0
        // Score = Q only
        let score = puct_score(0.7, 0.5, 3, 0, 2.5);
        assert!(
            (score - 0.7).abs() < 1e-6,
            "With zero parent visits, score should equal Q, got {}",
            score
        );
    }

    #[test]
    fn puct_score_with_zero_c_puct() {
        // With c_puct = 0, exploration = 0, score = Q
        let score = puct_score(0.6, 0.8, 5, 100, 0.0);
        assert!(
            (score - 0.6).abs() < 1e-6,
            "With c_puct=0, score should equal Q, got {}",
            score
        );
    }

    #[test]
    fn puct_score_negative_q() {
        // Losing position: Q = -0.8
        let score = puct_score(-0.8, 0.2, 5, 100, 2.5);
        let expected = -0.8 + 2.5 * 0.2 * 10.0 / 6.0;
        assert!(
            (score - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            score
        );
    }

    // =========================================================================
    // select_child tests
    // =========================================================================

    #[test]
    fn select_single_child() {
        // With only one child, it must be selected regardless of scores
        let (arena, parent_idx, child_indices) = setup_parent_with_children(
            10,  // parent visits
            5.0, // parent total_value => Q = 0.5
            &[(0.5, 3, 0.9)], // one child: prior=0.5, visits=3, total=0.9
        );
        let config = MctsConfig::default();
        let selected = select_child(parent_idx, &arena, &config);
        assert_eq!(selected, child_indices[0]);
    }

    #[test]
    #[should_panic(expected = "no children")]
    fn select_child_panics_on_no_children() {
        let mut arena = Arena::new(8);
        let parent_idx = arena.alloc(Node::root());
        let config = MctsConfig::default();
        let _ = select_child(parent_idx, &arena, &config);
    }

    #[test]
    fn uniform_priors_no_visits_selects_first_in_iteration() {
        // All children: same prior, zero visits, zero value.
        // Scores are all equal. Ties broken by iteration order (first encountered wins).
        // Children are prepended, so the last added child iterates first.
        let (arena, parent_idx, child_indices) = setup_parent_with_children(
            0,   // parent visits (root before any simulation)
            0.0, // parent total_value
            &[
                (0.25, 0, 0.0), // child 0 (added first -> iterated last)
                (0.25, 0, 0.0), // child 1
                (0.25, 0, 0.0), // child 2
                (0.25, 0, 0.0), // child 3 (added last -> iterated first)
            ],
        );
        let config = MctsConfig::default();
        let selected = select_child(parent_idx, &arena, &config);

        // Due to prepend, child_indices[3] is first in iteration order.
        // With parent_visits=0, sqrt(0)=0, exploration=0 for all.
        // Q_fpu = 0.0 - 0.25 = -0.25 for all. All scores equal => first wins.
        assert_eq!(selected, child_indices[3]);
    }

    #[test]
    fn after_visits_exploitation_vs_exploration() {
        // Parent has 100 visits.
        // Child A: high Q (0.8), many visits (80) -- exploitation favourite
        // Child B: low Q (0.2), few visits (5), high prior (0.6) -- exploration favourite
        // With default c_puct=2.5:
        //   A: Q=0.8, exploration = 2.5 * 0.3 * 10 / 81 = 0.0926 => 0.8926
        //   B: Q=0.2, exploration = 2.5 * 0.6 * 10 / 6 = 2.5 => 2.7
        // B should win because it's underexplored with high prior.
        let (arena, parent_idx, child_indices) = setup_parent_with_children(
            100,
            80.0, // parent Q = 0.8
            &[
                (0.3, 80, 64.0), // child A: Q = 64/80 = 0.8
                (0.6, 5, 1.0),   // child B: Q = 1/5 = 0.2
            ],
        );
        let config = MctsConfig::default();
        let selected = select_child(parent_idx, &arena, &config);
        assert_eq!(
            selected, child_indices[1],
            "Underexplored child with high prior should be selected"
        );
    }

    #[test]
    fn fpu_discourages_unvisited_when_parent_is_winning() {
        // Parent Q = 0.8 (winning), fpu_reduction = 0.25.
        // Unvisited child gets Q_fpu = 0.8 - 0.25 = 0.55
        // Visited child A has Q = 0.7, visits = 10
        // Visited child B (unvisited) has Q_fpu = 0.55, visits = 0
        //
        // With parent_visits=20, c_puct=2.5:
        //   A: Q=0.7, exploration = 2.5 * 0.5 * sqrt(20) / 11 = 2.5*0.5*4.472/11 ≈ 0.508 => total ≈ 1.208
        //   B: Q_fpu=0.55, exploration = 2.5 * 0.5 * sqrt(20) / 1 = 2.5*0.5*4.472 ≈ 5.590 => total ≈ 6.140
        //
        // B still wins due to huge exploration bonus (zero visits), despite FPU penalty.
        // That's correct — FPU doesn't prevent exploration, just makes it less aggressive.
        let (arena, parent_idx, child_indices) = setup_parent_with_children(
            20,
            16.0, // Q = 0.8
            &[
                (0.5, 10, 7.0),  // child A: Q = 0.7
                (0.5, 0, 0.0),   // child B: unvisited, will get Q_fpu = 0.55
            ],
        );
        let config = MctsConfig::default();
        let selected = select_child(parent_idx, &arena, &config);
        // Unvisited B still wins, but let's verify the FPU value is being used
        // by checking a scenario where FPU actually matters:
        assert_eq!(selected, child_indices[1]);

        // Now test where FPU actually tips the balance:
        // Visited child with Q=0.6 vs unvisited with tiny prior
        // Parent Q = 0.8, fpu = 0.55
        let (arena2, parent_idx2, child_indices2) = setup_parent_with_children(
            100,
            80.0, // Q = 0.8
            &[
                (0.4, 40, 24.0),   // child A: Q = 0.6, well-explored
                (0.01, 0, 0.0),    // child B: unvisited, tiny prior
            ],
        );
        let selected2 = select_child(parent_idx2, &arena2, &config);
        // A: Q=0.6, exploration = 2.5 * 0.4 * 10 / 41 ≈ 0.2439 => 0.8439
        // B: Q_fpu=0.55, exploration = 2.5 * 0.01 * 10 / 1 = 0.25 => 0.80
        // A wins because B has tiny prior and FPU penalty
        assert_eq!(
            selected2, child_indices2[0],
            "Visited child should beat unvisited with tiny prior thanks to FPU"
        );
    }

    #[test]
    fn c_puct_zero_pure_exploitation() {
        // With c_puct=0, exploration term vanishes. Should always pick highest Q.
        let (arena, parent_idx, child_indices) = setup_parent_with_children(
            50,
            25.0, // Q = 0.5
            &[
                (0.1, 10, 3.0),  // child A: Q = 0.3
                (0.8, 10, 7.0),  // child B: Q = 0.7  <-- highest Q
                (0.1, 10, 5.0),  // child C: Q = 0.5
            ],
        );
        let config = MctsConfig {
            c_puct: 0.0,
            ..MctsConfig::default()
        };
        let selected = select_child(parent_idx, &arena, &config);
        assert_eq!(
            selected, child_indices[1],
            "With c_puct=0, should select highest Q child"
        );
    }

    #[test]
    fn high_c_puct_prefers_high_prior_unvisited() {
        // Very high c_puct means exploration dominates.
        // Unvisited child with high prior should be preferred over visited children.
        let (arena, parent_idx, child_indices) = setup_parent_with_children(
            100,
            80.0, // Q = 0.8
            &[
                (0.1, 50, 45.0),  // child A: Q = 0.9, well-visited, low prior
                (0.8, 0, 0.0),    // child B: unvisited, high prior
                (0.1, 40, 30.0),  // child C: Q = 0.75, well-visited, low prior
            ],
        );
        let config = MctsConfig {
            c_puct: 100.0, // very high exploration
            ..MctsConfig::default()
        };
        let selected = select_child(parent_idx, &arena, &config);
        assert_eq!(
            selected, child_indices[1],
            "High c_puct should prefer unvisited child with high prior"
        );
    }

    #[test]
    fn fpu_value_is_relative_to_parent_q() {
        // Test that FPU is computed from the parent's Q, not from 0.
        // Parent Q = -0.5 (losing), fpu_reduction = 0.25.
        // Unvisited children get Q_fpu = -0.5 - 0.25 = -0.75
        //
        // Compare: if FPU were Q=0 (no FPU), unvisited would get Q=0 > -0.5,
        // which over-estimates them.
        let (arena, parent_idx, child_indices) = setup_parent_with_children(
            100,
            -50.0, // Q = -0.5
            &[
                (0.5, 50, -20.0), // child A: Q = -0.4 (slightly better than parent)
                (0.5, 0, 0.0),    // child B: unvisited, Q_fpu = -0.75
            ],
        );
        let config = MctsConfig {
            c_puct: 0.0, // pure exploitation to isolate FPU effect
            ..MctsConfig::default()
        };
        let selected = select_child(parent_idx, &arena, &config);
        // With c_puct=0: A has Q=-0.4, B has Q_fpu=-0.75
        // A should win because -0.4 > -0.75
        assert_eq!(
            selected, child_indices[0],
            "FPU should use parent Q, making unvisited child pessimistic"
        );
    }

    #[test]
    fn fpu_with_zero_parent_visits() {
        // Edge case: parent has 0 visits (fresh root).
        // Parent Q = 0.0, fpu_reduction = 0.25 => Q_fpu = -0.25
        // With parent_visits=0, sqrt(0)=0, so exploration = 0 for all.
        // All unvisited children get score = Q_fpu = -0.25, tie broken by iteration.
        let (arena, parent_idx, child_indices) = setup_parent_with_children(
            0,
            0.0,
            &[
                (0.5, 0, 0.0), // child 0
                (0.3, 0, 0.0), // child 1
                (0.2, 0, 0.0), // child 2 (iterated first due to prepend)
            ],
        );
        let config = MctsConfig::default();
        let selected = select_child(parent_idx, &arena, &config);
        // All scores equal at -0.25. First in iteration = child_indices[2]
        assert_eq!(selected, child_indices[2]);
    }

    #[test]
    fn exploration_decreases_with_visits() {
        // Verify that as a child gets more visits, its exploration bonus shrinks
        // and the score converges towards Q.
        let q = 0.5_f32;
        let prior = 0.3;
        let parent_visits = 100;
        let c_puct = 2.5;

        let score_1 = puct_score(q, prior, 1, parent_visits, c_puct);
        let score_10 = puct_score(q, prior, 10, parent_visits, c_puct);
        let score_100 = puct_score(q, prior, 100, parent_visits, c_puct);

        // More visits => lower score (exploration bonus shrinks)
        assert!(
            score_1 > score_10,
            "Score with 1 visit ({}) should be > score with 10 ({}).",
            score_1,
            score_10
        );
        assert!(
            score_10 > score_100,
            "Score with 10 visits ({}) should be > score with 100 ({}).",
            score_10,
            score_100
        );

        // As visits grow, score approaches Q
        assert!(
            (score_100 - q).abs() < 0.1,
            "With many visits, score ({}) should be close to Q ({})",
            score_100,
            q
        );
    }

    #[test]
    fn higher_prior_gives_higher_score() {
        // With same Q and visits, higher prior should give higher PUCT score
        let score_low = puct_score(0.5, 0.1, 5, 100, 2.5);
        let score_high = puct_score(0.5, 0.9, 5, 100, 2.5);
        assert!(
            score_high > score_low,
            "Higher prior ({}) should give higher score than lower prior ({})",
            score_high,
            score_low
        );
    }
}
