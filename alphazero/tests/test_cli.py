"""Tests for the AlphaZero CLI.

Verifies argument parsing, defaults, and --help for every subcommand.
Does NOT test actual execution of training, evaluation, etc. (those are
covered by each module's own test suite).
"""

from __future__ import annotations

import pytest

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
        assert args.output_dir == "./games"
        assert args.num_games == 100
        assert args.simulations == 800
        assert args.temperature == 1.0
        assert args.device == "cuda"

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
        assert args.dry_run is False

    def test_dry_run(self, parser):
        args = parser.parse_args([
            "pipeline", "--config", "config.yaml", "--dry-run",
        ])
        assert args.dry_run is True

    def test_missing_config_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["pipeline"])


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
# Placeholder subcommands: self-play, play, analyze produce expected output
# ============================================================================

class TestPlaceholderOutput:
    """Verify that placeholder subcommands print informative messages."""

    def test_selfplay_prints_rust_hint(self, capsys):
        main(["self-play", "--model", "model.pt"])
        captured = capsys.readouterr()
        assert "cargo run" in captured.out
        assert "self-play" in captured.out

    def test_play_prints_rust_hint(self, capsys):
        main(["play"])
        captured = capsys.readouterr()
        assert "cargo run" in captured.out
        assert "chess-engine" in captured.out

    def test_analyze_prints_pyo3_hint(self, capsys):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        main(["analyze", "--fen", fen])
        captured = capsys.readouterr()
        assert fen in captured.out
        assert "PyO3" in captured.out


# ============================================================================
# Verbose flag
# ============================================================================

def test_verbose_flag(parser):
    """The -v/--verbose flag should be accepted at the top level."""
    args = parser.parse_args(["-v", "train"])
    assert args.verbose is True
    assert args.command == "train"
