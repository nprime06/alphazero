"""Model evaluation for AlphaZero training.

Plays games between two neural network models to measure strength progression.
Uses a simple pure-Python MCTS implementation since the Rust MCTS bindings
aren't available yet (Phase 7).

The evaluation flow:
1. Load two TorchScript models (current vs previous)
2. Play N games (alternating colors)
3. Report win/draw/loss statistics
4. Estimate ELO difference

Dependencies:
    - ``python-chess`` for legal move generation and game simulation
    - ``neural.encoding`` for converting board positions to network input tensors
    - ``neural.moves`` for mapping between chess moves and policy vector indices

Usage::

    from orchestrator.evaluate import evaluate_models

    results = evaluate_models(
        model_a_path="weights/model_v000002.pt",
        model_b_path="weights/model_v000001.pt",
        num_games=20,
        simulations=100,
    )
    print(results.summary())
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from neural.encoding import BoardState, encode_board
from neural.moves import move_to_index, index_to_move, POLICY_SIZE

try:
    import chess

    HAS_PYTHON_CHESS = True
except ImportError:
    HAS_PYTHON_CHESS = False


# =============================================================================
# Result Data Structure
# =============================================================================


@dataclass
class EvalResult:
    """Results from an evaluation match between two models.

    All statistics are reported from Model A's perspective: ``a_wins``
    counts how many games Model A won, ``b_wins`` how many it lost, etc.

    Attributes:
        a_wins: Number of games won by Model A.
        b_wins: Number of games won by Model B.
        draws: Number of drawn games.
        total_games: Total games played (should equal a_wins + b_wins + draws).
        game_lengths: List of game lengths in half-moves (plies), one per game.
    """

    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    total_games: int = 0
    game_lengths: list[int] = field(default_factory=list)

    @property
    def a_win_rate(self) -> float:
        """Win rate of Model A (0.0 to 1.0).

        Draws count as half a win for each side, following the standard
        convention in chess rating calculations.
        """
        if self.total_games == 0:
            return 0.0
        return (self.a_wins + 0.5 * self.draws) / self.total_games

    def elo_difference(self) -> float:
        """Estimated ELO difference (A relative to B).

        Uses the logistic ELO formula:
            ELO_diff = -400 * log10(1/score - 1)

        where score is the win rate (with draws counting as 0.5).

        When one side wins all games (score = 0 or 1), the ELO difference
        is theoretically infinite, so we return +/- inf.

        When both sides have wins, we can also use the simpler formula:
            ELO_diff = 400 * log10(W/L)
        but the logistic formula handles draws more gracefully.

        Returns:
            Positive if A is stronger, negative if B is stronger, 0 if equal.
        """
        if self.a_wins == 0 and self.b_wins == 0:
            # All draws or no games
            return 0.0

        score = self.a_win_rate
        if score <= 0.0:
            return float("-inf")
        if score >= 1.0:
            return float("inf")
        return -400.0 * math.log10(1.0 / score - 1.0)

    def summary(self) -> str:
        """Human-readable summary of the evaluation results."""
        avg_len = (
            sum(self.game_lengths) / len(self.game_lengths)
            if self.game_lengths
            else 0
        )
        elo = self.elo_difference()
        return (
            f"Model A vs Model B: {self.a_wins}W / {self.draws}D / {self.b_wins}L "
            f"({self.total_games} games)\n"
            f"Win rate: {self.a_win_rate:.1%}\n"
            f"ELO difference: {elo:+.0f}\n"
            f"Avg game length: {avg_len:.0f} moves"
        )


# =============================================================================
# Move Conversion Helpers
# =============================================================================


def _require_python_chess() -> None:
    """Raise ImportError with a helpful message if python-chess is missing."""
    if not HAS_PYTHON_CHESS:
        raise ImportError(
            "The 'python-chess' package is required for model evaluation. "
            "Install it with: pip install python-chess"
        )


# Promotion piece mapping: python-chess piece type -> our string name
_PROMOTION_MAP = {}
if HAS_PYTHON_CHESS:
    _PROMOTION_MAP = {
        chess.QUEEN: "queen",
        chess.ROOK: "rook",
        chess.BISHOP: "bishop",
        chess.KNIGHT: "knight",
    }


def chess_move_to_policy_index(move: "chess.Move", is_black: bool) -> int:
    """Convert a python-chess Move to our policy vector index.

    Our move encoding uses (from_rank, from_file, to_rank, to_file) where
    rank 0 = rank 1 (white's back rank) and file 0 = a-file. This matches
    python-chess's square_rank() and square_file() exactly.

    When it is black's turn, the board encoding flips the board vertically,
    so the move encoding must also be flipped (flip_for_black=True).

    Args:
        move: A python-chess Move object.
        is_black: True if it's black's turn (applies rank flipping).

    Returns:
        Policy index in [0, 4671].
    """
    from_rank = chess.square_rank(move.from_square)
    from_file = chess.square_file(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    to_file = chess.square_file(move.to_square)

    promotion: Optional[str] = None
    if move.promotion is not None:
        promotion = _PROMOTION_MAP.get(move.promotion)

    return move_to_index(
        from_rank,
        from_file,
        to_rank,
        to_file,
        promotion=promotion,
        flip_for_black=is_black,
    )


def _board_to_board_state(board: "chess.Board") -> BoardState:
    """Convert a python-chess Board to our BoardState for neural network encoding.

    This creates a BoardState from the board's FEN string, which captures
    piece positions, side to move, castling rights, en passant square,
    and move counters. History is not tracked (all history planes will be
    zero), which is acceptable for evaluation purposes.

    Args:
        board: A python-chess Board object.

    Returns:
        A BoardState suitable for ``encode_board()``.
    """
    return BoardState.from_fen_piece_placement(board.fen())


# =============================================================================
# Simple Python MCTS (for evaluation only)
# =============================================================================


@dataclass
class MCTSNode:
    """A node in the MCTS tree.

    This is a minimal MCTS node for evaluation purposes. It tracks visit
    counts and accumulated value for PUCT-based selection, plus the prior
    probability from the neural network policy.
    """

    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    children: dict = field(default_factory=dict)  # move_uci -> MCTSNode

    @property
    def q_value(self) -> float:
        """Mean action value Q(s,a) = W(s,a) / N(s,a)."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


def _evaluate_position(
    board: "chess.Board",
    model: torch.jit.ScriptModule,
    device: str = "cpu",
) -> tuple[dict[str, float], float]:
    """Run the neural network on a position to get policy and value.

    Args:
        board: Current board position.
        model: TorchScript model accepting (1, 119, 8, 8) input.
        device: Device for inference.

    Returns:
        Tuple of (policy_dict, value) where policy_dict maps UCI move
        strings to probabilities (only legal moves, sums to ~1.0) and
        value is the position evaluation from the current player's
        perspective in [-1, 1].
    """
    state = _board_to_board_state(board)
    tensor = encode_board(state).unsqueeze(0).to(device)
    is_black = board.turn == chess.BLACK

    with torch.no_grad():
        policy_logits, value_tensor = model(tensor)

    # Extract value (scalar from current player's perspective)
    value = value_tensor.item()

    # Build legal move mask and extract policy probabilities
    policy_logits = policy_logits.squeeze(0)  # (4672,)
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return {}, value

    # Create mask over legal moves
    legal_indices = []
    legal_uci = []
    for move in legal_moves:
        idx = chess_move_to_policy_index(move, is_black)
        legal_indices.append(idx)
        legal_uci.append(move.uci())

    # Extract logits for legal moves and apply softmax
    legal_logits = policy_logits[legal_indices]
    probs = F.softmax(legal_logits, dim=0).cpu().numpy()

    policy_dict = {}
    for uci, prob in zip(legal_uci, probs):
        policy_dict[uci] = float(prob)

    return policy_dict, value


def mcts_search(
    board: "chess.Board",
    model: torch.jit.ScriptModule,
    simulations: int = 100,
    c_puct: float = 1.5,
    device: str = "cpu",
) -> dict[str, float]:
    """Run MCTS search and return visit count distribution.

    This is a simplified MCTS for evaluation only. It uses the neural network
    for both policy prior and value estimation at leaf nodes. The search tree
    is built incrementally: each simulation traverses from root to a leaf,
    expands the leaf, and backpropagates the value.

    Args:
        board: Current board position.
        model: TorchScript model that takes (1, 119, 8, 8) input.
        simulations: Number of MCTS simulations to run.
        c_puct: Exploration constant for PUCT formula.
        device: Device for inference.

    Returns:
        Dict mapping move UCI strings to visit fractions (sums to ~1.0).
    """
    # Get initial policy and value for root
    root_policy, _ = _evaluate_position(board, model, device)

    if not root_policy:
        return {}

    root = MCTSNode(visits=1)
    for uci, prior in root_policy.items():
        root.children[uci] = MCTSNode(prior=prior)

    for _ in range(simulations):
        # --- Selection: walk down tree using PUCT ---
        node = root
        sim_board = board.copy()
        path: list[tuple[MCTSNode, str]] = []

        while node.children:
            # Find the best child according to PUCT
            best_uci = None
            best_score = float("-inf")
            sqrt_parent = math.sqrt(node.visits)

            for uci, child in node.children.items():
                # PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
                q = child.q_value
                exploration = c_puct * child.prior * sqrt_parent / (1 + child.visits)
                score = q + exploration
                if score > best_score:
                    best_score = score
                    best_uci = uci

            assert best_uci is not None
            path.append((node, best_uci))
            node = node.children[best_uci]
            sim_board.push(chess.Move.from_uci(best_uci))

            # If this node hasn't been expanded yet, expand it
            if node.visits == 0 and not node.children:
                break

        # --- Expansion & Evaluation ---
        if sim_board.is_game_over():
            # Terminal node: use game result as value
            result = sim_board.result()
            if result == "1-0":
                # White won. Value from perspective of side that just moved.
                # sim_board.turn is the side TO move (loser, since game is over
                # and last move ended it). The side that just moved is the winner.
                leaf_value = 1.0
            elif result == "0-1":
                leaf_value = 1.0
            else:
                leaf_value = 0.0
        else:
            # Expand the leaf node
            policy, leaf_value = _evaluate_position(sim_board, model, device)
            for uci, prior in policy.items():
                if uci not in node.children:
                    node.children[uci] = MCTSNode(prior=prior)
            # Negate value because _evaluate_position returns value from the
            # perspective of the player to move at the leaf, but we need it
            # from the perspective of the player who chose the move leading here
            leaf_value = -leaf_value

        # --- Backpropagation ---
        # Walk back up the path, alternating the sign of the value
        # (what's good for one player is bad for the other)
        current_value = leaf_value
        for parent_node, uci in reversed(path):
            child = parent_node.children[uci]
            child.visits += 1
            child.value_sum += current_value
            parent_node.visits += 1
            current_value = -current_value

        # If path was empty (shouldn't happen normally), update root
        if not path:
            node.visits += 1
            node.value_sum += leaf_value

    # --- Return visit distribution ---
    total_visits = sum(c.visits for c in root.children.values())
    if total_visits == 0:
        # Uniform fallback
        n = len(root.children)
        return {uci: 1.0 / n for uci in root.children}

    return {
        uci: child.visits / total_visits for uci, child in root.children.items()
    }


# =============================================================================
# Game Playing
# =============================================================================


def _select_move(
    visit_distribution: dict[str, float], temperature: float = 0.1
) -> str:
    """Select a move from a visit distribution.

    With temperature near 0, this is almost deterministic (picks the most
    visited move). With temperature = 1.0, sampling is proportional to
    visit counts.

    Args:
        visit_distribution: Dict mapping UCI move strings to visit fractions.
        temperature: Controls randomness. 0 = greedy, 1 = proportional.

    Returns:
        Selected move as a UCI string.
    """
    moves = list(visit_distribution.keys())
    visits = list(visit_distribution.values())

    if temperature < 1e-6:
        # Greedy: pick the move with the highest visit fraction
        return moves[max(range(len(visits)), key=lambda i: visits[i])]

    # Apply temperature: raise visit counts to power of 1/temperature,
    # then normalize
    adjusted = [v ** (1.0 / temperature) for v in visits]
    total = sum(adjusted)
    if total == 0:
        return random.choice(moves)
    probs = [a / total for a in adjusted]

    return random.choices(moves, weights=probs, k=1)[0]


def _select_move_policy_only(
    policy_dict: dict[str, float], temperature: float = 0.1
) -> str:
    """Select a move directly from policy probabilities (no MCTS).

    Args:
        policy_dict: Dict mapping UCI move strings to probabilities.
        temperature: Controls randomness.

    Returns:
        Selected move as a UCI string.
    """
    return _select_move(policy_dict, temperature)


def play_game(
    model_white: torch.jit.ScriptModule,
    model_black: torch.jit.ScriptModule,
    max_moves: int = 512,
    simulations: int = 0,
    temperature: float = 0.1,
    device: str = "cpu",
) -> tuple[str, int]:
    """Play a single game between two models.

    The game proceeds until a terminal state (checkmate, stalemate, draw
    by repetition/50-move rule/insufficient material) or the maximum number
    of moves is reached.

    Args:
        model_white: TorchScript model playing as white.
        model_black: TorchScript model playing as black.
        max_moves: Maximum number of half-moves (plies) before declaring a draw.
        simulations: Number of MCTS simulations per move. 0 means policy-only
            (pick the move with highest policy probability -- no search).
        temperature: Temperature for move selection (0 = greedy, 1 = proportional).
        device: Device for neural network inference.

    Returns:
        Tuple of (result, num_moves) where result is "white", "black", or
        "draw", and num_moves is the number of half-moves played.
    """
    _require_python_chess()

    board = chess.Board()
    num_moves = 0

    while not board.is_game_over() and num_moves < max_moves:
        # Pick the model for the current side
        model = model_white if board.turn == chess.WHITE else model_black

        if simulations > 0:
            # MCTS mode: run search and select from visit distribution
            visit_dist = mcts_search(
                board, model, simulations=simulations, device=device
            )
            if not visit_dist:
                break
            move_uci = _select_move(visit_dist, temperature)
        else:
            # Policy-only mode: select directly from network policy
            policy_dict, _ = _evaluate_position(board, model, device)
            if not policy_dict:
                break
            move_uci = _select_move_policy_only(policy_dict, temperature)

        board.push(chess.Move.from_uci(move_uci))
        num_moves += 1

    # Determine game result
    if board.is_game_over():
        result_str = board.result()
        if result_str == "1-0":
            return "white", num_moves
        elif result_str == "0-1":
            return "black", num_moves
        else:
            return "draw", num_moves
    else:
        # Max moves reached
        return "draw", num_moves


# =============================================================================
# Match Evaluation
# =============================================================================


def evaluate_models(
    model_a_path: str,
    model_b_path: str,
    num_games: int = 20,
    simulations: int = 0,
    temperature: float = 0.1,
    device: str = "cpu",
    verbose: bool = True,
) -> EvalResult:
    """Play a match between two models and return evaluation results.

    Model A and B alternate colors each game so that any first-move advantage
    is balanced. In even-numbered games (0, 2, 4, ...) Model A plays white;
    in odd-numbered games (1, 3, 5, ...) Model A plays black.

    Results are reported from Model A's perspective.

    Args:
        model_a_path: Path to Model A's TorchScript file.
        model_b_path: Path to Model B's TorchScript file.
        num_games: Number of games to play.
        simulations: MCTS simulations per move (0 = policy-only).
        temperature: Temperature for move selection.
        device: Device for inference ("cpu" or "cuda").
        verbose: If True, print progress after each game.

    Returns:
        EvalResult with match statistics from Model A's perspective.
    """
    _require_python_chess()

    model_a = torch.jit.load(model_a_path, map_location=device)
    model_b = torch.jit.load(model_b_path, map_location=device)

    result = EvalResult()

    for game_idx in range(num_games):
        # Alternate colors: even games A=white, odd games A=black
        if game_idx % 2 == 0:
            model_white = model_a
            model_black = model_b
            a_is_white = True
        else:
            model_white = model_b
            model_black = model_a
            a_is_white = False

        outcome, length = play_game(
            model_white=model_white,
            model_black=model_black,
            simulations=simulations,
            temperature=temperature,
            device=device,
        )

        result.total_games += 1
        result.game_lengths.append(length)

        # Translate outcome to Model A's perspective
        if outcome == "draw":
            result.draws += 1
        elif (outcome == "white" and a_is_white) or (
            outcome == "black" and not a_is_white
        ):
            result.a_wins += 1
        else:
            result.b_wins += 1

        if verbose:
            a_color = "white" if a_is_white else "black"
            print(
                f"Game {game_idx + 1}/{num_games}: "
                f"A ({a_color}) vs B -- {outcome} in {length} moves "
                f"[A: {result.a_wins}W/{result.draws}D/{result.b_wins}L]"
            )

    if verbose:
        print()
        print(result.summary())

    return result
