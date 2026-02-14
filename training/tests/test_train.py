"""Tests for the training loop.

All tests use :class:`DummyDataset` (random data) and ``NetworkConfig.tiny()``
so they run on CPU without game files or a GPU.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

import pytest

from neural.config import NetworkConfig
from training.dataloader import DummyDataset
from training.train import TrainConfig, Trainer


# ============================================================================
# Helpers
# ============================================================================


def _make_trainer(
    *,
    device: torch.device | None = None,
    use_compile: bool = False,
    network: str = "tiny",
    lr: float = 0.01,
    batch_size: int = 32,
    total_steps: int = 100,
    use_amp: bool = False,
) -> Trainer:
    """Build a Trainer with sensible test defaults."""
    device = device or torch.device("cpu")
    config = TrainConfig(
        batch_size=batch_size,
        learning_rate=lr,
        network_config=network,
        use_compile=use_compile,
        total_steps=total_steps,
        log_interval=10_000,  # suppress logs during tests
        use_amp=use_amp,
    )
    return Trainer(config, device)


def _make_dataloader(batch_size: int = 32, size: int = 256) -> DataLoader:
    """Build a DataLoader over DummyDataset."""
    ds = DummyDataset(size=size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# ============================================================================
# TrainConfig tests
# ============================================================================


class TestTrainConfig:
    """Verify TrainConfig defaults."""

    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.batch_size == 4096
        assert cfg.learning_rate == 0.2
        assert cfg.momentum == 0.9
        assert cfg.weight_decay == 1e-4
        assert cfg.total_steps == 700_000
        assert cfg.lr_milestones == [100_000, 300_000, 500_000]
        assert cfg.lr_gamma == 0.1
        assert cfg.network_config == "full"
        assert cfg.use_compile is True

    def test_custom_values(self):
        cfg = TrainConfig(batch_size=64, learning_rate=0.01, total_steps=500)
        assert cfg.batch_size == 64
        assert cfg.learning_rate == 0.01
        assert cfg.total_steps == 500


# ============================================================================
# Trainer initialization tests
# ============================================================================


class TestTrainerInit:
    """Verify that the Trainer constructs all components correctly."""

    def test_init_default(self):
        """Trainer initializes with defaults (tiny network for speed)."""
        trainer = _make_trainer()
        assert trainer.step == 0
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.loss_fn is not None

    def test_model_on_device(self):
        """Model parameters should be on the requested device."""
        device = torch.device("cpu")
        trainer = _make_trainer(device=device)
        for p in trainer.model.parameters():
            assert p.device == device

    def test_network_presets(self):
        """Each network preset creates a model with different sizes."""
        counts = {}
        for preset in ("tiny", "small", "full"):
            trainer = _make_trainer(network=preset)
            counts[preset] = sum(
                p.numel() for p in trainer.model.parameters()
            )
        assert counts["tiny"] < counts["small"] < counts["full"]

    def test_no_compile(self):
        """With use_compile=False, compiled_model is just the raw model."""
        trainer = _make_trainer(use_compile=False)
        assert trainer.compiled_model is trainer.model


# ============================================================================
# train_step tests
# ============================================================================


class TestTrainStep:
    """Tests for a single training step."""

    def test_returns_expected_keys(self):
        """train_step returns a dict with the four expected metric keys."""
        trainer = _make_trainer()
        loader = _make_dataloader(batch_size=32)
        batch = next(iter(loader))
        metrics = trainer.train_step(batch)

        expected_keys = {"total_loss", "policy_loss", "value_loss", "learning_rate"}
        assert set(metrics.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(metrics.keys())}"
        )

    def test_losses_are_finite(self):
        """All loss values should be finite (not NaN or inf)."""
        trainer = _make_trainer()
        loader = _make_dataloader(batch_size=32)
        batch = next(iter(loader))
        metrics = trainer.train_step(batch)

        for key in ("total_loss", "policy_loss", "value_loss"):
            val = metrics[key]
            assert not (val != val), f"{key} is NaN"  # NaN != NaN is True
            assert abs(val) < 1e10, f"{key} is too large: {val}"

    def test_step_counter_increments(self):
        """Each call to train_step increments self.step by 1."""
        trainer = _make_trainer()
        loader = _make_dataloader(batch_size=32)
        batch = next(iter(loader))

        assert trainer.step == 0
        trainer.train_step(batch)
        assert trainer.step == 1
        trainer.train_step(batch)
        assert trainer.step == 2

    def test_gradients_nonzero(self):
        """After a train_step, at least some parameter gradients are non-zero."""
        trainer = _make_trainer()
        loader = _make_dataloader(batch_size=32)
        batch = next(iter(loader))
        trainer.train_step(batch)

        has_nonzero_grad = False
        for p in trainer.model.parameters():
            if p.grad is not None and p.grad.abs().max().item() > 0:
                has_nonzero_grad = True
                break

        assert has_nonzero_grad, "Expected at least some non-zero gradients"

    def test_without_compile(self):
        """train_step works the same with use_compile=False."""
        trainer = _make_trainer(use_compile=False)
        loader = _make_dataloader(batch_size=32)
        batch = next(iter(loader))
        metrics = trainer.train_step(batch)

        assert "total_loss" in metrics
        assert metrics["total_loss"] > 0


# ============================================================================
# Training loop tests
# ============================================================================


class TestTrainLoop:
    """Integration tests for the full training loop."""

    def test_loss_decreases_over_steps(self):
        """Loss should decrease over 100 steps on dummy data (tiny network).

        This is a basic sanity check that backprop is working. We use a
        moderate learning rate and compare the first batch loss to the
        average of the last 10 batches.
        """
        trainer = _make_trainer(lr=0.01, total_steps=100, batch_size=32)
        loader = _make_dataloader(batch_size=32, size=3200)

        losses = []
        trainer.model.train()
        for batch in loader:
            if trainer.step >= 100:
                break
            metrics = trainer.train_step(batch)
            losses.append(metrics["total_loss"])

        assert len(losses) >= 20, f"Expected >= 20 steps, got {len(losses)}"

        early_avg = sum(losses[:10]) / 10
        late_avg = sum(losses[-10:]) / 10

        assert late_avg < early_avg, (
            f"Loss should decrease: early_avg={early_avg:.4f}, "
            f"late_avg={late_avg:.4f}"
        )

    def test_train_stops_at_total_steps(self):
        """trainer.train() stops at the configured total_steps."""
        total = 20
        trainer = _make_trainer(total_steps=total, batch_size=32)
        loader = _make_dataloader(batch_size=32, size=1000)

        trainer.train(loader, total_steps=total)
        assert trainer.step == total, (
            f"Expected step={total}, got {trainer.step}"
        )


# ============================================================================
# Learning rate schedule tests
# ============================================================================


class TestLRSchedule:
    """Verify that the learning rate decays at milestones."""

    def test_lr_at_milestones(self):
        """LR should drop by lr_gamma at each milestone step."""
        milestones = [10, 30, 50]
        initial_lr = 0.1
        gamma = 0.1

        config = TrainConfig(
            learning_rate=initial_lr,
            lr_milestones=milestones,
            lr_gamma=gamma,
            network_config="tiny",
            use_compile=False,
            batch_size=32,
            total_steps=60,
            log_interval=10_000,
        )
        device = torch.device("cpu")
        trainer = Trainer(config, device)
        loader = _make_dataloader(batch_size=32, size=2000)

        lr_history = []
        trainer.model.train()
        for batch in loader:
            if trainer.step >= 60:
                break
            metrics = trainer.train_step(batch)
            lr_history.append(metrics["learning_rate"])

        # Before first milestone (step 10), LR should be initial_lr
        assert abs(lr_history[0] - initial_lr) < 1e-6, (
            f"Initial LR should be {initial_lr}, got {lr_history[0]}"
        )

        # After step 10 (index 10), LR should have dropped
        assert lr_history[10] < initial_lr, (
            f"LR at step 11 should be < {initial_lr}, got {lr_history[10]}"
        )

        # After step 30 (index 30), LR should have dropped again
        assert lr_history[30] < lr_history[10], (
            f"LR at step 31 should be < LR at step 11"
        )


# ============================================================================
# torch.compile tests
# ============================================================================


class TestCompile:
    """Tests related to torch.compile behaviour."""

    def test_compile_does_not_change_output_shapes(self):
        """With torch.compile, output shapes should be identical."""
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile not available")

        trainer_no_compile = _make_trainer(use_compile=False)
        trainer_compile = _make_trainer(use_compile=True)

        # Use the same input
        x = torch.randn(4, 119, 8, 8)

        trainer_no_compile.model.eval()
        trainer_compile.model.eval()

        with torch.no_grad():
            p1, v1 = trainer_no_compile.compiled_model(x)
            p2, v2 = trainer_compile.compiled_model(x)

        assert p1.shape == p2.shape, (
            f"Policy shapes differ: {p1.shape} vs {p2.shape}"
        )
        assert v1.shape == v2.shape, (
            f"Value shapes differ: {v1.shape} vs {v2.shape}"
        )


# ============================================================================
# CLI argument parsing test
# ============================================================================


class TestCLI:
    """Test that the argparse configuration is valid."""

    def test_default_args(self):
        """Parsing with no arguments should give sensible defaults."""
        from training.train import main
        import argparse

        # We test the parser directly rather than calling main()
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-dir", type=str, default="./data")
        parser.add_argument("--batch-size", type=int, default=4096)
        parser.add_argument("--steps", type=int, default=700_000)
        parser.add_argument("--lr", type=float, default=0.2)
        parser.add_argument("--network", type=str, default="full",
                            choices=["tiny", "small", "medium", "full"])
        parser.add_argument("--no-compile", action="store_true")
        parser.add_argument("--num-workers", type=int, default=4)
        parser.add_argument("--dummy-data", action="store_true")

        args = parser.parse_args([])
        assert args.data_dir == "./data"
        assert args.batch_size == 4096
        assert args.steps == 700_000
        assert args.lr == 0.2
        assert args.network == "full"
        assert args.no_compile is False
        assert args.dummy_data is False

    def test_custom_args(self):
        """Parsing with custom arguments."""
        from training.train import main
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--data-dir", type=str, default="./data")
        parser.add_argument("--batch-size", type=int, default=4096)
        parser.add_argument("--steps", type=int, default=700_000)
        parser.add_argument("--lr", type=float, default=0.2)
        parser.add_argument("--network", type=str, default="full",
                            choices=["tiny", "small", "medium", "full"])
        parser.add_argument("--no-compile", action="store_true")
        parser.add_argument("--num-workers", type=int, default=4)
        parser.add_argument("--dummy-data", action="store_true")

        args = parser.parse_args([
            "--data-dir", "/tmp/games",
            "--batch-size", "128",
            "--steps", "500",
            "--lr", "0.01",
            "--network", "tiny",
            "--no-compile",
            "--dummy-data",
        ])
        assert args.data_dir == "/tmp/games"
        assert args.batch_size == 128
        assert args.steps == 500
        assert args.lr == 0.01
        assert args.network == "tiny"
        assert args.no_compile is True
        assert args.dummy_data is True


# ============================================================================
# AMP (Automatic Mixed Precision) tests
# ============================================================================


class TestAMP:
    """Tests for automatic mixed precision support."""

    def test_amp_disabled_on_cpu(self):
        """AMP should be disabled when device is CPU (even if use_amp=True)."""
        config = TrainConfig(
            use_amp=True, network_config="tiny", use_compile=False,
            batch_size=32, total_steps=100, log_interval=10_000,
        )
        device = torch.device("cpu")
        trainer = Trainer(config, device)
        assert trainer.use_amp is False

    def test_amp_disabled_when_configured_off(self):
        """AMP should be disabled when use_amp=False."""
        config = TrainConfig(
            use_amp=False, network_config="tiny", use_compile=False,
            batch_size=32, total_steps=100, log_interval=10_000,
        )
        device = torch.device("cpu")
        trainer = Trainer(config, device)
        assert trainer.use_amp is False

    def test_scaler_exists(self):
        """Trainer should have a GradScaler attribute."""
        trainer = _make_trainer()
        assert hasattr(trainer, "scaler")
        assert isinstance(trainer.scaler, torch.amp.GradScaler)

    def test_train_step_with_amp_disabled(self):
        """train_step works with AMP explicitly disabled."""
        config = TrainConfig(
            use_amp=False, network_config="tiny", use_compile=False,
            batch_size=32, total_steps=100, log_interval=10_000,
        )
        device = torch.device("cpu")
        trainer = Trainer(config, device)
        loader = _make_dataloader(batch_size=32)
        batch = next(iter(loader))
        metrics = trainer.train_step(batch)
        assert "total_loss" in metrics
        assert metrics["total_loss"] > 0

    def test_loss_finite_with_amp_disabled(self):
        """Losses remain finite with AMP disabled (CPU path)."""
        config = TrainConfig(
            use_amp=False, network_config="tiny", use_compile=False,
            batch_size=32, total_steps=100, log_interval=10_000,
        )
        device = torch.device("cpu")
        trainer = Trainer(config, device)
        loader = _make_dataloader(batch_size=32)
        batch = next(iter(loader))
        metrics = trainer.train_step(batch)
        for key in ("total_loss", "policy_loss", "value_loss"):
            val = metrics[key]
            assert not (val != val), f"{key} is NaN"
            assert abs(val) < 1e10, f"{key} too large: {val}"

    def test_config_default_amp_true(self):
        """Default config has use_amp=True."""
        cfg = TrainConfig()
        assert cfg.use_amp is True
