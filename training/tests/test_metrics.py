"""Tests for the training metrics and monitoring module."""

from __future__ import annotations

import glob
import os

import torch
import pytest

from training.metrics import MetricsLogger, compute_policy_accuracy


# ============================================================================
# Helpers
# ============================================================================


def _make_metrics(total=1.5, policy=1.0, value=0.5, lr=0.01):
    return {
        "total_loss": total,
        "policy_loss": policy,
        "value_loss": value,
        "learning_rate": lr,
    }


# ============================================================================
# MetricsLogger tests
# ============================================================================


class TestMetricsLogger:
    """Tests for the MetricsLogger class."""

    def test_init_creates_log_dir(self, tmp_path):
        """MetricsLogger creates the log directory."""
        log_dir = str(tmp_path / "logs")
        logger = MetricsLogger(log_dir=log_dir)
        assert os.path.isdir(log_dir)
        logger.close()

    def test_log_step_no_errors(self, tmp_path):
        """Log a single step with a valid metrics dict, no exceptions."""
        logger = MetricsLogger(log_dir=str(tmp_path / "logs"))
        logger.log_step(1, _make_metrics(), batch_size=32)
        logger.close()

    def test_log_step_multiple(self, tmp_path):
        """Log 10 steps sequentially, no exceptions."""
        logger = MetricsLogger(log_dir=str(tmp_path / "logs"))
        for step in range(1, 11):
            logger.log_step(step, _make_metrics(), batch_size=32)
        logger.close()

    def test_tensorboard_files_created(self, tmp_path):
        """After logging and closing, TensorBoard event files exist in the log dir."""
        log_dir = str(tmp_path / "logs")
        logger = MetricsLogger(log_dir=log_dir)
        logger.log_step(1, _make_metrics())
        logger.close()

        events = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
        assert len(events) > 0, "Expected TensorBoard event files in log dir"

    def test_close_no_errors(self, tmp_path):
        """Calling close() after logging doesn't raise."""
        logger = MetricsLogger(log_dir=str(tmp_path / "logs"))
        logger.log_step(1, _make_metrics())
        logger.close()

    def test_log_checkpoint(self, tmp_path):
        """log_checkpoint(step, path) doesn't raise."""
        logger = MetricsLogger(log_dir=str(tmp_path / "logs"))
        logger.log_checkpoint(10, "/tmp/checkpoint_10.pt")
        logger.close()

    def test_log_policy_accuracy(self, tmp_path):
        """log_policy_accuracy(step, 0.5) doesn't raise."""
        logger = MetricsLogger(log_dir=str(tmp_path / "logs"))
        logger.log_policy_accuracy(5, 0.5)
        logger.close()

    def test_log_value_mse(self, tmp_path):
        """log_value_mse(step, 0.1) doesn't raise."""
        logger = MetricsLogger(log_dir=str(tmp_path / "logs"))
        logger.log_value_mse(5, 0.1)
        logger.close()

    def test_console_interval(self, tmp_path, capsys):
        """Log steps 1-5 with console_interval=3; step 3 printed, not 1 or 2."""
        logger = MetricsLogger(
            log_dir=str(tmp_path / "logs"),
            console_interval=3,
        )
        for step in range(1, 6):
            logger.log_step(step, _make_metrics(), batch_size=32)
        logger.close()

        captured = capsys.readouterr().out
        assert "[Step       3]" in captured, (
            f"Expected step 3 in console output, got: {captured!r}"
        )
        assert "[Step       1]" not in captured
        assert "[Step       2]" not in captured

    def test_wandb_disabled_by_default(self, tmp_path):
        """MetricsLogger(use_wandb=False) has _wandb_active == False."""
        logger = MetricsLogger(
            log_dir=str(tmp_path / "logs"),
            use_wandb=False,
        )
        assert logger._wandb_active is False
        logger.close()

    def test_throughput_with_batch_size(self, tmp_path):
        """Log 2+ steps with batch_size > 0, no errors (throughput from 2nd step)."""
        logger = MetricsLogger(log_dir=str(tmp_path / "logs"))
        logger.log_step(1, _make_metrics(), batch_size=128)
        logger.log_step(2, _make_metrics(), batch_size=128)
        logger.log_step(3, _make_metrics(), batch_size=128)
        logger.close()


# ============================================================================
# compute_policy_accuracy tests
# ============================================================================


class TestComputePolicyAccuracy:
    """Tests for the compute_policy_accuracy helper function."""

    def test_perfect_accuracy(self):
        """Identical argmax for all positions gives 1.0."""
        # Same logits → same argmax
        logits = torch.tensor([[0.0, 0.0, 1.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0]])
        targets = torch.tensor([[0.0, 0.0, 1.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0]])
        assert compute_policy_accuracy(logits, targets) == 1.0

    def test_zero_accuracy(self):
        """Completely different argmax gives 0.0."""
        logits = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0]])
        targets = torch.tensor([[0.0, 0.0, 0.0, 1.0],
                                [0.0, 0.0, 0.0, 1.0]])
        assert compute_policy_accuracy(logits, targets) == 0.0

    def test_partial_accuracy(self):
        """2 out of 4 match gives 0.5."""
        logits = torch.tensor([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [1.0, 0.0, 0.0]])
        targets = torch.tensor([[1.0, 0.0, 0.0],   # match
                                [0.0, 1.0, 0.0],   # match
                                [1.0, 0.0, 0.0],   # mismatch
                                [0.0, 1.0, 0.0]])   # mismatch
        assert compute_policy_accuracy(logits, targets) == pytest.approx(0.5)

    def test_batch_of_one(self):
        """Works with batch_size=1."""
        logits = torch.tensor([[0.0, 0.0, 1.0]])
        targets = torch.tensor([[0.0, 0.0, 1.0]])
        assert compute_policy_accuracy(logits, targets) == 1.0

    def test_returns_float(self):
        """Return type is float, not tensor."""
        logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        result = compute_policy_accuracy(logits, targets)
        assert isinstance(result, float)


# ============================================================================
# Integration tests
# ============================================================================


class TestMetricsLoggerIntegration:
    """Integration tests that combine MetricsLogger with the real Trainer."""

    def test_with_real_trainer_metrics(self, tmp_path):
        """Create a Trainer, run one train_step, pass metrics to logger."""
        from training.train import TrainConfig, Trainer
        from training.dataloader import DummyDataset
        from torch.utils.data import DataLoader

        config = TrainConfig(
            batch_size=32,
            learning_rate=0.01,
            network_config="tiny",
            use_compile=False,
            total_steps=100,
            log_interval=10_000,
            use_amp=False,
        )
        device = torch.device("cpu")
        trainer = Trainer(config, device)

        ds = DummyDataset(size=64)
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        batch = next(iter(loader))

        metrics = trainer.train_step(batch)

        logger = MetricsLogger(log_dir=str(tmp_path / "logs"))
        logger.log_step(step=1, metrics=metrics, batch_size=32)
        logger.close()
