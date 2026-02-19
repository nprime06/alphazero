"""Tests for the AlphaZero pipeline coordinator.

Verifies configuration, state persistence, and coordinator logic
without actually running self-play or training.
"""

from __future__ import annotations

import yaml
import pytest
from pathlib import Path

from orchestrator.coordinator import PipelineConfig, PipelineState, Coordinator


# ============================================================================
# PipelineConfig
# ============================================================================


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_defaults(self):
        """Default config has sensible values."""
        config = PipelineConfig()

        assert config.project_dir == "/home/willzhao/alphazero"
        assert config.data_dir == "data"
        assert config.weights_dir == "weights"
        assert config.checkpoint_dir == "checkpoints"

        # Self-play defaults
        assert config.selfplay_games_per_iteration == 250
        assert config.selfplay_simulations == 400
        assert config.selfplay_max_moves == 300
        assert config.selfplay_parallel_games == 16
        assert config.selfplay_threads == 1
        assert config.selfplay_batch_size == 32
        assert config.selfplay_fp16 is True

        # Training defaults
        assert config.train_steps_per_iteration == 500
        assert config.train_batch_size == 256
        assert config.train_network == "full"
        assert config.train_gpus == 1

        # Evaluation defaults
        assert config.eval_games == 40
        assert config.eval_simulations == 400
        assert config.eval_win_threshold == 0.55

        # Pipeline defaults
        assert config.max_iterations == 0
        assert config.min_games_before_training == 50
        assert config.weights_keep_n == 10

        # Slurm defaults
        assert config.slurm_partition == "mit_normal_gpu"
        assert config.slurm_time_selfplay == "6:00:00"
        assert config.slurm_time_train == "6:00:00"

        # Runtime
        assert config.dry_run is False

    def test_from_yaml(self, tmp_path):
        """Config can be loaded from a YAML file."""
        config_data = {
            "project_dir": "/tmp/test_project",
            "data_dir": "my_data",
            "selfplay_games_per_iteration": 100,
            "train_steps_per_iteration": 500,
            "eval_win_threshold": 0.60,
            "max_iterations": 10,
            "slurm_partition": "test_partition",
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = PipelineConfig.from_yaml(str(config_file))

        assert config.project_dir == "/tmp/test_project"
        assert config.data_dir == "my_data"
        assert config.selfplay_games_per_iteration == 100
        assert config.train_steps_per_iteration == 500
        assert config.eval_win_threshold == 0.60
        assert config.max_iterations == 10
        assert config.slurm_partition == "test_partition"

        # Non-specified values should keep their defaults
        assert config.weights_dir == "weights"
        assert config.train_batch_size == 256

    def test_from_yaml_ignores_unknown_keys(self, tmp_path):
        """Unknown keys in the YAML file are silently ignored."""
        config_data = {
            "project_dir": "/tmp/test",
            "unknown_key": "should_be_ignored",
            "another_unknown": 42,
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = PipelineConfig.from_yaml(str(config_file))
        assert config.project_dir == "/tmp/test"
        # Should not raise, unknown keys are silently dropped

    def test_from_yaml_empty_file(self, tmp_path):
        """An empty YAML file returns all defaults."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config = PipelineConfig.from_yaml(str(config_file))
        assert config.project_dir == "/home/willzhao/alphazero"
        assert config.selfplay_games_per_iteration == 250

    def test_resolve_path_relative(self):
        """Relative paths are resolved against project_dir."""
        config = PipelineConfig(project_dir="/home/user/project")
        resolved = config.resolve_path("data")
        assert resolved == Path("/home/user/project/data")

    def test_resolve_path_absolute(self):
        """Absolute paths are returned as-is."""
        config = PipelineConfig(project_dir="/home/user/project")
        resolved = config.resolve_path("/tmp/absolute/path")
        assert resolved == Path("/tmp/absolute/path")


# ============================================================================
# PipelineState
# ============================================================================


class TestPipelineState:
    """Tests for PipelineState dataclass."""

    def test_defaults(self):
        """Default state has all zeros."""
        state = PipelineState()
        assert state.iteration == 0
        assert state.best_model_version == 0
        assert state.total_games == 0
        assert state.total_train_steps == 0

    def test_save_load_roundtrip(self, tmp_path):
        """State can be saved and loaded back with all values preserved."""
        state = PipelineState(
            iteration=5,
            best_model_version=3,
            total_games=2500,
            total_train_steps=5000,
        )
        state_file = tmp_path / "state.yaml"
        state.save(str(state_file))

        loaded = PipelineState.load(str(state_file))
        assert loaded.iteration == 5
        assert loaded.best_model_version == 3
        assert loaded.total_games == 2500
        assert loaded.total_train_steps == 5000

    def test_load_nonexistent_returns_default(self, tmp_path):
        """Loading from a nonexistent path returns default state."""
        state = PipelineState.load(str(tmp_path / "nonexistent.yaml"))
        assert state.iteration == 0
        assert state.best_model_version == 0
        assert state.total_games == 0
        assert state.total_train_steps == 0

    def test_save_creates_parent_dirs(self, tmp_path):
        """Save creates parent directories if they don't exist."""
        state = PipelineState(iteration=1)
        nested_path = tmp_path / "a" / "b" / "state.yaml"
        state.save(str(nested_path))

        assert nested_path.exists()
        loaded = PipelineState.load(str(nested_path))
        assert loaded.iteration == 1

    def test_save_is_yaml_readable(self, tmp_path):
        """Saved state file is valid, human-readable YAML."""
        state = PipelineState(iteration=3, total_games=100)
        state_file = tmp_path / "state.yaml"
        state.save(str(state_file))

        with open(state_file) as f:
            data = yaml.safe_load(f)

        assert data["iteration"] == 3
        assert data["total_games"] == 100
        assert data["best_model_version"] == 0
        assert data["total_train_steps"] == 0


# ============================================================================
# Coordinator
# ============================================================================


class TestCoordinator:
    """Tests for the Coordinator class."""

    def _make_config(self, tmp_path, **overrides):
        """Helper to create a config pointing at tmp directories."""
        defaults = dict(
            project_dir=str(tmp_path),
            data_dir="data",
            weights_dir="weights",
            checkpoint_dir="checkpoints",
            max_iterations=1,
            dry_run=True,
        )
        defaults.update(overrides)
        return PipelineConfig(**defaults)

    def test_init(self, tmp_path):
        """Coordinator initializes without error and creates directories."""
        config = self._make_config(tmp_path)
        coordinator = Coordinator(config)

        assert coordinator.state.iteration == 0
        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "weights").is_dir()
        assert (tmp_path / "checkpoints").is_dir()

    def test_init_loads_existing_state(self, tmp_path):
        """Coordinator loads existing state on initialization."""
        # Pre-create state file
        state = PipelineState(iteration=3, best_model_version=2)
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        state.save(str(ckpt_dir / "pipeline_state.yaml"))

        config = self._make_config(tmp_path)
        coordinator = Coordinator(config)

        assert coordinator.state.iteration == 3
        assert coordinator.state.best_model_version == 2

    def test_count_games_empty(self, tmp_path):
        """No games in an empty data directory."""
        config = self._make_config(tmp_path)
        coordinator = Coordinator(config)

        assert coordinator._count_games() == 0

    def test_count_games_with_files(self, tmp_path):
        """Correctly counts .msgpack game files."""
        config = self._make_config(tmp_path)
        coordinator = Coordinator(config)

        data_dir = tmp_path / "data"
        # Create some game files
        (data_dir / "games_iter0001.msgpack").write_bytes(b"data1")
        (data_dir / "games_iter0002.msgpack").write_bytes(b"data2")
        (data_dir / "games_iter0003.msgpack").write_bytes(b"data3")
        # Create a non-game file (should not be counted)
        (data_dir / "README.txt").write_text("not a game file")

        assert coordinator._count_games() == 3

    def test_count_games_nested(self, tmp_path):
        """Counts .msgpack files in subdirectories too."""
        config = self._make_config(tmp_path)
        coordinator = Coordinator(config)

        data_dir = tmp_path / "data"
        sub_dir = data_dir / "worker_1"
        sub_dir.mkdir()
        (data_dir / "top.msgpack").write_bytes(b"data")
        (sub_dir / "nested.msgpack").write_bytes(b"data")

        assert coordinator._count_games() == 2

    def test_dry_run(self, tmp_path):
        """Dry-run mode completes an iteration without launching anything."""
        config = self._make_config(tmp_path, dry_run=True, max_iterations=1)
        coordinator = Coordinator(config)

        # Should complete without error and without creating game files
        coordinator.run()

        # State should show iteration 1
        assert coordinator.state.iteration == 1

    def test_dry_run_no_games_skips_training(self, tmp_path):
        """Dry-run with no games skips training due to min_games check."""
        config = self._make_config(
            tmp_path,
            dry_run=True,
            max_iterations=1,
            min_games_before_training=100,
        )
        coordinator = Coordinator(config)
        coordinator.run()

        # Training was skipped, so total_train_steps stays at 0
        assert coordinator.state.total_train_steps == 0

    def test_state_persisted_after_run(self, tmp_path):
        """Pipeline state is saved to disk after each iteration."""
        config = self._make_config(tmp_path, dry_run=True, max_iterations=1)
        coordinator = Coordinator(config)
        coordinator.run()

        # Load the persisted state
        state_path = tmp_path / "checkpoints" / "pipeline_state.yaml"
        assert state_path.exists()

        loaded = PipelineState.load(str(state_path))
        assert loaded.iteration == 1

    def test_max_iterations_respected(self, tmp_path):
        """Pipeline stops after max_iterations."""
        config = self._make_config(tmp_path, dry_run=True, max_iterations=3)
        coordinator = Coordinator(config)
        coordinator.run()

        assert coordinator.state.iteration == 3

    def test_get_best_model_path_none(self, tmp_path):
        """Returns None when no model exists."""
        config = self._make_config(tmp_path)
        coordinator = Coordinator(config)

        assert coordinator._get_best_model_path() is None

    def test_get_best_model_path_from_state(self, tmp_path):
        """Returns the model pointed to by best_model_version in state."""
        config = self._make_config(tmp_path)
        coordinator = Coordinator(config)

        # Create a model file matching the state
        coordinator.state.best_model_version = 2
        model_path = tmp_path / "weights" / "model_v000002.pt"
        model_path.write_bytes(b"model_data")

        result = coordinator._get_best_model_path()
        assert result == model_path

    def test_get_best_model_path_fallback_to_latest_txt(self, tmp_path):
        """Falls back to latest.txt when state version file is missing."""
        config = self._make_config(tmp_path)
        coordinator = Coordinator(config)

        # State points to version 0 (no best), but latest.txt exists
        weights_dir = tmp_path / "weights"
        (weights_dir / "latest.txt").write_text("3")
        (weights_dir / "model_v000003.pt").write_bytes(b"model_data")

        result = coordinator._get_best_model_path()
        assert result == weights_dir / "model_v000003.pt"

    def test_get_latest_weight_version(self, tmp_path):
        """Reads version from latest.txt correctly."""
        config = self._make_config(tmp_path)
        coordinator = Coordinator(config)

        # No latest.txt -> version 0
        assert coordinator._get_latest_weight_version() == 0

        # Write a version
        (tmp_path / "weights" / "latest.txt").write_text("5")
        assert coordinator._get_latest_weight_version() == 5

    def test_promote_model(self, tmp_path):
        """Promote updates best_model_version in state."""
        config = self._make_config(tmp_path)
        coordinator = Coordinator(config)

        # Simulate a published model
        (tmp_path / "weights" / "latest.txt").write_text("4")
        (tmp_path / "weights" / "model_v000004.pt").write_bytes(b"data")

        coordinator._promote_model()

        assert coordinator.state.best_model_version == 4
