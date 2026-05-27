"""Tests for the AlphaZero CLI.

Verifies argument parsing, defaults, and --help for every subcommand.
Does NOT test actual execution of training, evaluation, etc. (those are
covered by each module's own test suite).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

import alphazero.cli as cli
from alphazero.cli import build_parser, main


# ============================================================================
# Helpers
# ============================================================================

@pytest.fixture
def parser():
    """Return a fresh parser instance for each test."""
    return build_parser()


# ============================================================================
# --help smoke tests (each subcommand should accept --help without error)
# ============================================================================

@pytest.mark.parametrize(
    "subcmd",
    ["train", "self-play", "evaluate", "play", "analyze", "pipeline", "export"],
)
def test_subcommand_help(subcmd: str, capsys):
    """``alphazero <subcmd> --help`` should exit(0) with usage text."""
    with pytest.raises(SystemExit) as exc_info:
        main([subcmd, "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert subcmd in captured.out or "usage" in captured.out.lower()


def test_top_level_help(capsys):
    """``alphazero --help`` should exit(0) with usage text."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "alphazero" in captured.out.lower()


# ============================================================================
# Main entry point (no subcommand)
# ============================================================================

def test_no_subcommand_prints_help(capsys):
    """Running ``alphazero`` with no args should print help and exit(0)."""
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "alphazero" in captured.out.lower()


# ============================================================================
# train: argument defaults
# ============================================================================

class TestTrainDefaults:
    """Verify default values for the ``train`` subcommand."""

    def test_defaults(self, parser):
        args = parser.parse_args(["train"])
        assert args.command == "train"
        assert args.data_dir == "./data"
        assert args.batch_size == 4096
        assert args.steps == 700000
        assert args.lr == 0.2
        assert args.network == "full"
        assert args.no_compile is False
        assert args.no_amp is False
        assert args.log_dir == "./runs"
        assert args.checkpoint_dir == "./checkpoints"
        assert args.checkpoint_interval == 1000
        assert args.resume is None
        assert args.dummy_data is False

    def test_custom_values(self, parser):
        args = parser.parse_args([
            "train",
            "--data-dir", "/tmp/data",
            "--batch-size", "256",
            "--steps", "1000",
            "--lr", "0.01",
            "--network", "tiny",
            "--no-compile",
            "--no-amp",
            "--log-dir", "/tmp/logs",
            "--checkpoint-dir", "/tmp/ckpt",
            "--checkpoint-interval", "500",
            "--resume", "/tmp/ckpt/checkpoint.pt",
            "--dummy-data",
        ])
        assert args.data_dir == "/tmp/data"
        assert args.batch_size == 256
        assert args.steps == 1000
        assert args.lr == 0.01
        assert args.network == "tiny"
        assert args.no_compile is True
        assert args.no_amp is True
        assert args.log_dir == "/tmp/logs"
        assert args.checkpoint_dir == "/tmp/ckpt"
        assert args.checkpoint_interval == 500
        assert args.resume == "/tmp/ckpt/checkpoint.pt"
        assert args.dummy_data is True


# ============================================================================
# self-play: argument parsing
# ============================================================================

class TestSelfPlayArgs:
    """Verify argument parsing for the ``self-play`` subcommand."""

    def test_defaults_with_required(self, parser):
        args = parser.parse_args(["self-play", "--model", "model.pt"])
        assert args.command == "self-play"
        assert args.model == "model.pt"
        assert args.output == "./games"
        assert args.games == 100
        assert args.sims == 800
        assert args.threads == 1
        assert args.parallel_games == 1
        assert args.batch_size == 8
        assert args.max_moves == 512
        assert args.no_noise is False
        assert args.c_puct == 2.5

    def test_custom_values(self, parser):
        args = parser.parse_args([
            "self-play",
            "--model", "model.pt",
            "--output", "/tmp/games",
            "--games", "12",
            "--sims", "64",
            "--threads", "2",
            "--parallel-games", "3",
            "--batch-size", "6",
            "--max-moves", "200",
            "--no-noise",
            "--c-puct", "1.75",
        ])
        assert args.output == "/tmp/games"
        assert args.games == 12
        assert args.sims == 64
        assert args.threads == 2
        assert args.parallel_games == 3
        assert args.batch_size == 6
        assert args.max_moves == 200
        assert args.no_noise is True
        assert args.c_puct == 1.75

    def test_missing_model_exits(self, parser):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["self-play"])
        assert exc_info.value.code != 0


# ============================================================================
# evaluate: argument parsing
# ============================================================================

class TestEvaluateArgs:
    """Verify argument parsing for the ``evaluate`` subcommand."""

    def test_defaults_with_required(self, parser):
        args = parser.parse_args([
            "evaluate", "--model-a", "a.pt", "--model-b", "b.pt",
        ])
        assert args.command == "evaluate"
        assert args.model_a == "a.pt"
        assert args.model_b == "b.pt"
        assert args.num_games == 100
        assert args.simulations == 100
        assert args.device == "cpu"
        assert args.backend == "auto"
        assert args.max_moves == 512

    def test_custom_backend(self, parser):
        args = parser.parse_args([
            "evaluate",
            "--model-a", "a.pt",
            "--model-b", "b.pt",
            "--backend", "rust",
            "--max-moves", "64",
        ])
        assert args.backend == "rust"
        assert args.max_moves == 64

    def test_missing_models_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["evaluate"])


# ============================================================================
# play: argument parsing
# ============================================================================

class TestPlayArgs:
    """Verify argument parsing for the ``play`` subcommand."""

    def test_defaults(self, parser):
        args = parser.parse_args(["play"])
        assert args.command == "play"
        assert args.model is None
        assert args.simulations == 800
        assert args.device == "cpu"
        assert args.color == "white"

    def test_custom_values(self, parser):
        args = parser.parse_args([
            "play", "--model", "model.pt", "--simulations", "400",
            "--device", "cuda", "--color", "black",
        ])
        assert args.model == "model.pt"
        assert args.simulations == 400
        assert args.device == "cuda"
        assert args.color == "black"

    def test_invalid_color_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["play", "--color", "red"])


# ============================================================================
# analyze: argument parsing
# ============================================================================

class TestAnalyzeArgs:
    """Verify argument parsing for the ``analyze`` subcommand."""

    START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    def test_defaults_with_required(self, parser):
        args = parser.parse_args(["analyze", "--fen", self.START_FEN])
        assert args.command == "analyze"
        assert args.fen == self.START_FEN
        assert args.model is None
        assert args.simulations == 800
        assert args.device == "cpu"

    def test_missing_fen_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["analyze"])


# ============================================================================
# pipeline: argument parsing
# ============================================================================

class TestPipelineArgs:
    """Verify argument parsing for the ``pipeline`` subcommand."""

    def test_defaults_with_required(self, parser):
        args = parser.parse_args(["pipeline", "--config", "config.yaml"])
        assert args.command == "pipeline"
        assert args.config == "config.yaml"
        assert args.run_dir is None
        assert args.project_dir is None
        assert args.iterations is None
        assert args.network is None
        assert args.gpus is None
        assert args.eval_backend is None
        assert args.eval_max_moves is None
        assert args.dry_run is False

    def test_overrides(self, parser):
        args = parser.parse_args([
            "pipeline",
            "--config", "config.yaml",
            "--run-dir", "/tmp/run",
            "--project-dir", "/tmp/project",
            "--iterations", "3",
            "--network", "tiny",
            "--gpus", "2",
            "--eval-backend", "rust",
            "--eval-max-moves", "80",
            "--dry-run",
        ])
        assert args.run_dir == "/tmp/run"
        assert args.project_dir == "/tmp/project"
        assert args.iterations == 3
        assert args.network == "tiny"
        assert args.gpus == 2
        assert args.eval_backend == "rust"
        assert args.eval_max_moves == 80
        assert args.dry_run is True

    def test_config_is_optional(self, parser):
        args = parser.parse_args(["pipeline", "--dry-run"])
        assert args.config is None
        assert args.dry_run is True


# ============================================================================
# export: argument parsing
# ============================================================================

class TestExportArgs:
    """Verify argument parsing for the ``export`` subcommand."""

    def test_defaults_with_required(self, parser):
        args = parser.parse_args([
            "export", "--checkpoint", "ckpt.pt", "--output", "model.pt",
        ])
        assert args.command == "export"
        assert args.checkpoint == "ckpt.pt"
        assert args.output == "model.pt"
        assert args.network == "full"

    def test_custom_network(self, parser):
        args = parser.parse_args([
            "export", "--checkpoint", "ckpt.pt", "--output", "model.pt",
            "--network", "tiny",
        ])
        assert args.network == "tiny"

    def test_missing_checkpoint_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["export"])

    def test_missing_output_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["export", "--checkpoint", "ckpt.pt"])


# ============================================================================
# Executing wrapper subcommands: self-play, play, analyze
# ============================================================================

class TestExecutingSubcommands:
    """Verify command construction without launching real Rust binaries."""

    def test_selfplay_invokes_rust_binary_with_torch_env(
        self, tmp_path, monkeypatch, capsys
    ):
        binary = tmp_path / "self-play"
        binary.write_text("")
        torch_lib = tmp_path / "torch" / "lib"
        torch_lib.mkdir(parents=True)
        calls = {}

        def fake_run(cmd, env=None):
            calls["cmd"] = cmd
            calls["env"] = env
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(cli, "_release_binary", lambda name: binary)
        monkeypatch.setattr(cli, "_find_torch_lib_dir", lambda: torch_lib)
        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        monkeypatch.setenv("DYLD_LIBRARY_PATH", "/existing")

        main([
            "self-play",
            "--model", "model.pt",
            "--output", "/tmp/games",
            "--games", "12",
            "--sims", "64",
            "--threads", "2",
            "--parallel-games", "3",
            "--batch-size", "6",
            "--max-moves", "200",
            "--no-noise",
            "--c-puct", "1.75",
        ])

        assert calls["cmd"] == [
            str(binary),
            "--model", "model.pt",
            "--games", "12",
            "--output", "/tmp/games",
            "--sims", "64",
            "--threads", "2",
            "--parallel-games", "3",
            "--batch-size", "6",
            "--max-moves", "200",
            "--no-noise",
            "--c-puct", "1.75",
        ]
        assert calls["env"]["DYLD_LIBRARY_PATH"].split(os.pathsep)[0] == str(torch_lib)
        assert calls["env"]["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(torch_lib)
        assert "Running self-play" in capsys.readouterr().out

    def test_selfplay_missing_binary_exits(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "_release_binary", lambda name: tmp_path / "missing")

        with pytest.raises(SystemExit) as exc_info:
            main(["self-play", "--model", "model.pt"])

        assert exc_info.value.code == 1
        assert "cargo build --release -p self-play" in capsys.readouterr().err

    def test_play_invokes_interactive_binary(self, tmp_path, monkeypatch, capsys):
        binary = tmp_path / "play"
        binary.write_text("")
        calls = {}

        def fake_run(cmd):
            calls["cmd"] = cmd
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(cli, "_release_binary", lambda name: binary)
        monkeypatch.setattr(cli.subprocess, "run", fake_run)

        main(["play"])

        assert calls["cmd"] == [str(binary)]
        assert "Launching interactive board" in capsys.readouterr().out

    def test_play_rejects_model_options(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["play", "--model", "model.pt"])

        assert exc_info.value.code == 2
        assert "model-vs-human play is not wired" in capsys.readouterr().err

    def test_analyze_uses_uniform_search(self, monkeypatch, capsys):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        calls = {}

        class FakeBoard:
            @staticmethod
            def from_fen(value):
                calls["fen"] = value
                return "board"

        def fake_search_uniform(board, simulations, temperature, c_puct):
            calls["search"] = (board, simulations, temperature, c_puct)
            return SimpleNamespace(
                moves=[("d2d4", 7), ("e2e4", 9)],
                total_simulations=16,
                root_value=0.25,
                best_move="e2e4",
            )

        fake_module = SimpleNamespace(
            Board=FakeBoard,
            search_uniform=fake_search_uniform,
        )
        monkeypatch.setitem(sys.modules, "alphazero_py", fake_module)
        monkeypatch.setattr(cli, "_install_torch_lib_env", lambda: None)

        main(["analyze", "--fen", fen, "--simulations", "16"])

        assert calls["fen"] == fen
        assert calls["search"] == ("board", 16, 1.0, 2.5)
        out = capsys.readouterr().out
        assert "Best move: e2e4" in out
        assert "Root value: 0.2500" in out
        assert "e2e4: 9 visits" in out

    def test_analyze_uses_model_search(self, monkeypatch, capsys):
        fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
        calls = {}

        class FakeBoard:
            @staticmethod
            def from_fen(value):
                return "board"

        def fake_search_with_model(
            board, model, simulations, temperature, c_puct, device
        ):
            calls["search"] = (
                board,
                model,
                simulations,
                temperature,
                c_puct,
                device,
            )
            return SimpleNamespace(
                moves=[("a1a2", 4)],
                total_simulations=4,
                root_value=-0.5,
                best_move="a1a2",
            )

        fake_module = SimpleNamespace(
            Board=FakeBoard,
            search_with_model=fake_search_with_model,
        )
        monkeypatch.setitem(sys.modules, "alphazero_py", fake_module)
        monkeypatch.setattr(cli, "_install_torch_lib_env", lambda: None)

        main([
            "analyze",
            "--fen", fen,
            "--model", "model.pt",
            "--simulations", "4",
            "--device", "cuda",
        ])

        assert calls["search"] == ("board", "model.pt", 4, 1.0, 2.5, "cuda")
        assert "Model: model.pt" in capsys.readouterr().out

    def test_analyze_missing_bindings_exits(self, monkeypatch, capsys):
        monkeypatch.setitem(sys.modules, "alphazero_py", None)
        monkeypatch.setattr(cli, "_install_torch_lib_env", lambda: None)

        with pytest.raises(SystemExit) as exc_info:
            main(["analyze", "--fen", "8/8/8/8/8/8/8/K6k w - - 0 1"])

        assert exc_info.value.code == 1
        assert "maturin develop --manifest-path alphazero-py/Cargo.toml" in (
            capsys.readouterr().err
        )


# ============================================================================
# Verbose flag
# ============================================================================

def test_verbose_flag(parser):
    """The -v/--verbose flag should be accepted at the top level."""
    args = parser.parse_args(["-v", "train"])
    assert args.verbose is True
    assert args.command == "train"
