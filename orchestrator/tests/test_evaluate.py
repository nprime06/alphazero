"""Tests for the model evaluation module.

These tests run on CPU without requiring a GPU. Tests that need python-chess
are skipped if it's not installed. Tests that need actual neural network
models create tiny networks on the fly.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pytest
import torch

from orchestrator.evaluate import EvalResult

# Skip python-chess-dependent tests if not installed
try:
    import chess

    HAS_PYTHON_CHESS = True
except ImportError:
    HAS_PYTHON_CHESS = False

requires_python_chess = pytest.mark.skipif(
    not HAS_PYTHON_CHESS,
    reason="python-chess not installed",
)


# =============================================================================
# Helpers: create tiny model for testing
# =============================================================================


def _create_test_model(tmp_dir: Path, name: str = "model.pt") -> Path:
    """Create a tiny TorchScript model and return its path."""
    from neural.config import NetworkConfig
    from neural.network import AlphaZeroNetwork
    from neural.export import export_torchscript

    config = NetworkConfig.tiny()
    model = AlphaZeroNetwork(config)
    model_path = tmp_dir / name
    export_torchscript(model, model_path)
    return model_path


# =============================================================================
# TestEvalResult
# =============================================================================


class TestEvalResult:
    """Tests for the EvalResult dataclass."""

    def test_defaults(self) -> None:
        """All fields should be zero/empty initially."""
        r = EvalResult()
        assert r.a_wins == 0
        assert r.b_wins == 0
        assert r.draws == 0
        assert r.total_games == 0
        assert r.game_lengths == []

    def test_win_rate_no_games(self) -> None:
        """Win rate should be 0 when no games have been played."""
        r = EvalResult()
        assert r.a_win_rate == 0.0

    def test_win_rate_all_a_wins(self) -> None:
        """100% win rate when A wins every game."""
        r = EvalResult(a_wins=10, b_wins=0, draws=0, total_games=10)
        assert r.a_win_rate == 1.0

    def test_win_rate_all_b_wins(self) -> None:
        """0% win rate when B wins every game."""
        r = EvalResult(a_wins=0, b_wins=10, draws=0, total_games=10)
        assert r.a_win_rate == 0.0

    def test_win_rate_even(self) -> None:
        """50% win rate when wins are split evenly (no draws)."""
        r = EvalResult(a_wins=5, b_wins=5, draws=0, total_games=10)
        assert r.a_win_rate == 0.5

    def test_win_rate_with_draws(self) -> None:
        """Draws count as 0.5 wins: 3W + 4D + 3L = (3 + 2) / 10 = 0.5."""
        r = EvalResult(a_wins=3, b_wins=3, draws=4, total_games=10)
        assert r.a_win_rate == pytest.approx(0.5)

    def test_elo_difference_even(self) -> None:
        """Equal wins should give ~0 ELO difference."""
        r = EvalResult(a_wins=5, b_wins=5, draws=0, total_games=10)
        assert r.elo_difference() == pytest.approx(0.0, abs=1.0)

    def test_elo_difference_strong_a(self) -> None:
        """More wins for A should give positive ELO difference."""
        r = EvalResult(a_wins=8, b_wins=2, draws=0, total_games=10)
        elo = r.elo_difference()
        assert elo > 0
        # 80% win rate should be roughly +240 ELO
        assert 150 < elo < 350

    def test_elo_difference_strong_b(self) -> None:
        """More wins for B should give negative ELO difference."""
        r = EvalResult(a_wins=2, b_wins=8, draws=0, total_games=10)
        elo = r.elo_difference()
        assert elo < 0

    def test_elo_difference_all_a_wins(self) -> None:
        """All wins for A should give +inf ELO difference."""
        r = EvalResult(a_wins=10, b_wins=0, draws=0, total_games=10)
        assert r.elo_difference() == float("inf")

    def test_elo_difference_all_b_wins(self) -> None:
        """All wins for B should give -inf ELO difference."""
        r = EvalResult(a_wins=0, b_wins=10, draws=0, total_games=10)
        assert r.elo_difference() == float("-inf")

    def test_elo_difference_all_draws(self) -> None:
        """All draws should give 0 ELO difference."""
        r = EvalResult(a_wins=0, b_wins=0, draws=10, total_games=10)
        assert r.elo_difference() == pytest.approx(0.0, abs=1.0)

    def test_summary_format(self) -> None:
        """summary() should return a string with expected substrings."""
        r = EvalResult(
            a_wins=3,
            b_wins=2,
            draws=1,
            total_games=6,
            game_lengths=[40, 50, 60, 30, 45, 55],
        )
        s = r.summary()
        assert "3W" in s
        assert "2L" in s
        assert "1D" in s
        assert "6 games" in s
        assert "Win rate" in s
        assert "ELO" in s
        assert "Avg game length" in s


# =============================================================================
# TestChessMoveConversion
# =============================================================================


@requires_python_chess
class TestChessMoveConversion:
    """Tests for converting between python-chess moves and policy indices."""

    def test_e2e4_white(self) -> None:
        """White pawn e2-e4 should map to a valid policy index."""
        from orchestrator.evaluate import chess_move_to_policy_index

        move = chess.Move.from_uci("e2e4")
        idx = chess_move_to_policy_index(move, is_black=False)
        assert 0 <= idx < 4672

    def test_e7e5_black(self) -> None:
        """Black pawn e7-e5 should map to a valid policy index (with flip)."""
        from orchestrator.evaluate import chess_move_to_policy_index

        move = chess.Move.from_uci("e7e5")
        idx = chess_move_to_policy_index(move, is_black=True)
        assert 0 <= idx < 4672

    def test_knight_move(self) -> None:
        """Knight g1-f3 should map to a valid policy index."""
        from orchestrator.evaluate import chess_move_to_policy_index

        move = chess.Move.from_uci("g1f3")
        idx = chess_move_to_policy_index(move, is_black=False)
        assert 0 <= idx < 4672

    def test_promotion_queen(self) -> None:
        """Queen promotion should map to a valid policy index."""
        from orchestrator.evaluate import chess_move_to_policy_index

        move = chess.Move.from_uci("a7a8q")
        idx = chess_move_to_policy_index(move, is_black=False)
        assert 0 <= idx < 4672

    def test_promotion_knight(self) -> None:
        """Knight underpromotion should map to a valid policy index."""
        from orchestrator.evaluate import chess_move_to_policy_index

        move = chess.Move.from_uci("a7a8n")
        idx = chess_move_to_policy_index(move, is_black=False)
        assert 0 <= idx < 4672

    def test_different_moves_different_indices(self) -> None:
        """Different moves from the starting position should give different indices."""
        from orchestrator.evaluate import chess_move_to_policy_index

        move1 = chess.Move.from_uci("e2e4")
        move2 = chess.Move.from_uci("d2d4")
        idx1 = chess_move_to_policy_index(move1, is_black=False)
        idx2 = chess_move_to_policy_index(move2, is_black=False)
        assert idx1 != idx2


# =============================================================================
# TestBoardConversion
# =============================================================================


@requires_python_chess
class TestBoardConversion:
    """Tests for converting python-chess boards to our BoardState."""

    def test_initial_position(self) -> None:
        """Starting position should encode to a (119, 8, 8) tensor."""
        from orchestrator.evaluate import _board_to_board_state
        from neural.encoding import encode_board

        board = chess.Board()
        state = _board_to_board_state(board)
        tensor = encode_board(state)
        assert tensor.shape == (119, 8, 8)

    def test_after_e4(self) -> None:
        """Position after 1.e4 should encode correctly with black to move."""
        from orchestrator.evaluate import _board_to_board_state
        from neural.encoding import encode_board, Color

        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        state = _board_to_board_state(board)
        assert state.side_to_move == Color.BLACK
        tensor = encode_board(state)
        assert tensor.shape == (119, 8, 8)


# =============================================================================
# TestPlayGame
# =============================================================================


@requires_python_chess
class TestPlayGame:
    """Tests for playing games between models.

    These tests create tiny models on the fly. Games played by random
    (untrained) networks are essentially random, but they should still
    complete without errors and produce valid results.
    """

    @pytest.fixture
    def model_path(self, tmp_path: Path) -> Path:
        """Create a tiny test model."""
        return _create_test_model(tmp_path)

    def test_play_game_completes(self, model_path: Path) -> None:
        """A game should complete without errors."""
        from orchestrator.evaluate import play_game

        model = torch.jit.load(str(model_path), map_location="cpu")
        result, length = play_game(
            model_white=model,
            model_black=model,
            max_moves=50,
            simulations=0,
            temperature=0.1,
            device="cpu",
        )
        assert result in ("white", "black", "draw")
        assert length > 0

    def test_game_result_valid(self, model_path: Path) -> None:
        """Game result should be one of the expected values."""
        from orchestrator.evaluate import play_game

        model = torch.jit.load(str(model_path), map_location="cpu")
        result, _ = play_game(
            model_white=model,
            model_black=model,
            max_moves=30,
            simulations=0,
            device="cpu",
        )
        assert result in ("white", "black", "draw")

    def test_game_length_reasonable(self, model_path: Path) -> None:
        """Game length should be positive and within max_moves."""
        from orchestrator.evaluate import play_game

        max_mv = 100
        model = torch.jit.load(str(model_path), map_location="cpu")
        _, length = play_game(
            model_white=model,
            model_black=model,
            max_moves=max_mv,
            simulations=0,
            device="cpu",
        )
        assert 0 < length <= max_mv

    def test_play_game_with_mcts(self, model_path: Path) -> None:
        """Game with MCTS search should complete without errors."""
        from orchestrator.evaluate import play_game

        model = torch.jit.load(str(model_path), map_location="cpu")
        result, length = play_game(
            model_white=model,
            model_black=model,
            max_moves=10,  # Very short to keep test fast
            simulations=5,  # Minimal search
            device="cpu",
        )
        assert result in ("white", "black", "draw")
        assert length > 0


# =============================================================================
# TestMCTSSearch
# =============================================================================


@requires_python_chess
class TestMCTSSearch:
    """Tests for the pure-Python MCTS implementation."""

    @pytest.fixture
    def model(self, tmp_path: Path) -> torch.jit.ScriptModule:
        """Create and load a tiny test model."""
        model_path = _create_test_model(tmp_path)
        return torch.jit.load(str(model_path), map_location="cpu")

    def test_mcts_returns_distribution(self, model: torch.jit.ScriptModule) -> None:
        """MCTS should return a distribution over legal moves."""
        from orchestrator.evaluate import mcts_search

        board = chess.Board()
        dist = mcts_search(board, model, simulations=10, device="cpu")
        assert len(dist) > 0
        # All legal moves from starting position
        assert len(dist) == len(list(board.legal_moves))
        # Should sum to approximately 1.0
        assert sum(dist.values()) == pytest.approx(1.0, abs=0.01)

    def test_mcts_all_valid_moves(self, model: torch.jit.ScriptModule) -> None:
        """All moves returned by MCTS should be legal."""
        from orchestrator.evaluate import mcts_search

        board = chess.Board()
        dist = mcts_search(board, model, simulations=10, device="cpu")
        legal_ucis = {m.uci() for m in board.legal_moves}
        for uci in dist:
            assert uci in legal_ucis


# =============================================================================
# TestEvaluateModels
# =============================================================================


@requires_python_chess
class TestEvaluateModels:
    """Tests for the full evaluation pipeline."""

    @pytest.fixture
    def model_paths(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create two tiny test models."""
        path_a = _create_test_model(tmp_path, "model_a.pt")
        path_b = _create_test_model(tmp_path, "model_b.pt")
        return path_a, path_b

    def test_evaluate_two_games(self, model_paths: tuple[Path, Path]) -> None:
        """evaluate_models with 2 games should produce valid results."""
        from orchestrator.evaluate import evaluate_models

        path_a, path_b = model_paths
        result = evaluate_models(
            model_a_path=str(path_a),
            model_b_path=str(path_b),
            num_games=2,
            simulations=0,
            temperature=0.1,
            device="cpu",
            verbose=False,
        )
        assert result.total_games == 2
        assert result.a_wins + result.b_wins + result.draws == 2
        assert len(result.game_lengths) == 2
        assert all(length > 0 for length in result.game_lengths)

    def test_same_model_runs(self, model_paths: tuple[Path, Path]) -> None:
        """Same model vs itself should complete without errors."""
        from orchestrator.evaluate import evaluate_models

        path_a, _ = model_paths
        result = evaluate_models(
            model_a_path=str(path_a),
            model_b_path=str(path_a),  # Same model
            num_games=2,
            simulations=0,
            device="cpu",
            verbose=False,
        )
        assert result.total_games == 2
        assert result.a_wins + result.b_wins + result.draws == 2

    def test_color_alternation(self, model_paths: tuple[Path, Path]) -> None:
        """Model A should alternate colors across games.

        We verify this indirectly: with 2 games, A plays white once and
        black once. The result should still be valid.
        """
        from orchestrator.evaluate import evaluate_models

        path_a, path_b = model_paths
        result = evaluate_models(
            model_a_path=str(path_a),
            model_b_path=str(path_b),
            num_games=2,
            simulations=0,
            device="cpu",
            verbose=False,
        )
        # Just checking it completes with correct count
        assert result.total_games == 2

    def test_summary_after_evaluation(self, model_paths: tuple[Path, Path]) -> None:
        """summary() should work after an actual evaluation."""
        from orchestrator.evaluate import evaluate_models

        path_a, path_b = model_paths
        result = evaluate_models(
            model_a_path=str(path_a),
            model_b_path=str(path_b),
            num_games=2,
            simulations=0,
            device="cpu",
            verbose=False,
        )
        s = result.summary()
        assert isinstance(s, str)
        assert "2 games" in s


# =============================================================================
# TestEvaluatePosition
# =============================================================================


@requires_python_chess
class TestEvaluatePosition:
    """Tests for the neural network position evaluation helper."""

    @pytest.fixture
    def model(self, tmp_path: Path) -> torch.jit.ScriptModule:
        """Create and load a tiny test model."""
        model_path = _create_test_model(tmp_path)
        return torch.jit.load(str(model_path), map_location="cpu")

    def test_evaluate_starting_position(
        self, model: torch.jit.ScriptModule
    ) -> None:
        """Evaluating the starting position should return policy and value."""
        from orchestrator.evaluate import _evaluate_position

        board = chess.Board()
        policy, value = _evaluate_position(board, model, device="cpu")

        # Policy should cover all 20 legal moves in starting position
        assert len(policy) == 20
        # Probabilities should sum to ~1
        assert sum(policy.values()) == pytest.approx(1.0, abs=0.01)
        # Value should be in [-1, 1]
        assert -1.0 <= value <= 1.0

    def test_evaluate_after_moves(self, model: torch.jit.ScriptModule) -> None:
        """Evaluation should work after some moves have been played."""
        from orchestrator.evaluate import _evaluate_position

        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        board.push(chess.Move.from_uci("e7e5"))

        policy, value = _evaluate_position(board, model, device="cpu")
        assert len(policy) > 0
        assert -1.0 <= value <= 1.0
