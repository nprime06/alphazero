//! Search configuration for MCTS.
//!
//! [`MctsConfig`] groups all the tunable hyperparameters that control how the
//! search explores the game tree. Reasonable defaults matching the AlphaZero
//! paper are provided via [`MctsConfig::default()`].

// =============================================================================
// MctsConfig
// =============================================================================

/// Configuration for MCTS search parameters.
///
/// These hyperparameters control the exploration/exploitation trade-off,
/// root noise injection, and move selection temperature.
#[derive(Clone, Debug)]
pub struct MctsConfig {
    /// Exploration constant in PUCT formula.
    /// Higher = more exploration, lower = more exploitation.
    /// AlphaZero uses 2.5 with policy prior scaling.
    pub c_puct: f32,

    /// First Play Urgency reduction.
    /// For unvisited children, Q_fpu = parent_Q - fpu_reduction.
    /// This discourages over-exploring unvisited nodes when the parent
    /// already has a good value estimate.
    /// Default: 0.25
    pub fpu_reduction: f32,

    /// Number of simulations per search.
    pub num_simulations: u32,

    /// Dirichlet noise alpha for root exploration.
    /// 0.3 for chess (AlphaZero paper).
    pub dirichlet_alpha: f32,

    /// Weight for Dirichlet noise at root.
    /// prior = (1 - eps) * prior + eps * noise
    /// Default: 0.25
    pub dirichlet_epsilon: f32,

    /// Temperature for move selection.
    /// T=1: proportional to visit counts (exploratory)
    /// T->0: pick most visited (greedy)
    pub temperature: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            c_puct: 2.5,
            fpu_reduction: 0.25,
            num_simulations: 800,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            temperature: 1.0,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values_are_reasonable() {
        let config = MctsConfig::default();

        // c_puct should be positive
        assert!(config.c_puct > 0.0, "c_puct must be positive");

        // fpu_reduction should be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&config.fpu_reduction),
            "fpu_reduction should be in [0, 1], got {}",
            config.fpu_reduction
        );

        // num_simulations should be positive
        assert!(
            config.num_simulations > 0,
            "num_simulations must be positive"
        );

        // dirichlet_alpha should be positive
        assert!(
            config.dirichlet_alpha > 0.0,
            "dirichlet_alpha must be positive"
        );

        // dirichlet_epsilon should be in (0, 1)
        assert!(
            (0.0..=1.0).contains(&config.dirichlet_epsilon),
            "dirichlet_epsilon should be in [0, 1], got {}",
            config.dirichlet_epsilon
        );

        // temperature should be non-negative
        assert!(
            config.temperature >= 0.0,
            "temperature must be non-negative"
        );
    }

    #[test]
    fn default_matches_alphazero_paper() {
        let config = MctsConfig::default();
        assert_eq!(config.c_puct, 2.5);
        assert_eq!(config.num_simulations, 800);
        assert_eq!(config.dirichlet_alpha, 0.3);
        assert_eq!(config.dirichlet_epsilon, 0.25);
        assert_eq!(config.temperature, 1.0);
    }

    #[test]
    fn config_is_clone() {
        let config = MctsConfig::default();
        let config2 = config.clone();
        assert_eq!(config.c_puct, config2.c_puct);
        assert_eq!(config.fpu_reduction, config2.fpu_reduction);
        assert_eq!(config.num_simulations, config2.num_simulations);
    }
}
