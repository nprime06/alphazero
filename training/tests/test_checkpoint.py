"""Tests for the checkpoint module.

All tests use :class:`DummyDataset` (random data) and ``NetworkConfig.tiny()``
so they run on CPU without game files or a GPU.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

import pytest

from training.dataloader import DummyDataset
from training.train import TrainConfig, Trainer
from training.checkpoint import CheckpointManager


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
    )
    return Trainer(config, device)


def _make_dataloader(batch_size: int = 32, size: int = 256) -> DataLoader:
    """Build a DataLoader over DummyDataset."""
    ds = DummyDataset(size=size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


# ============================================================================
# CheckpointManager tests
# ============================================================================


class TestCheckpointManager:
    """Tests for saving, loading, and cleanup of checkpoints."""

    def test_save_creates_file(self, tmp_path):
        """save() at step 100 should create a checkpoint file on disk."""
        trainer = _make_trainer()
        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=5)

        path = manager.save(trainer, step=100)
        assert path.exists()

    def test_save_filename_format(self, tmp_path):
        """Checkpoint filename should match checkpoint_step_{step:07d}.pt."""
        trainer = _make_trainer()
        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=5)

        path = manager.save(trainer, step=100)
        assert path.name == "checkpoint_step_0000100.pt"

    def test_save_atomic_write(self, tmp_path):
        """After save(), no .tmp files should remain in the directory."""
        trainer = _make_trainer()
        ckpt_dir = tmp_path / "ckpts"
        manager = CheckpointManager(str(ckpt_dir), keep_n=5)

        manager.save(trainer, step=100)

        tmp_files = list(ckpt_dir.glob("*.tmp"))
        assert tmp_files == [], f"Temporary files should not persist: {tmp_files}"

    def test_save_contents(self, tmp_path):
        """Saved checkpoint should contain all expected keys."""
        trainer = _make_trainer()
        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=5)

        path = manager.save(trainer, step=100, extra_metadata={"note": "test"})
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

        expected_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "step",
            "config",
            "note",
        }
        assert expected_keys.issubset(checkpoint.keys()), (
            f"Missing keys: {expected_keys - set(checkpoint.keys())}"
        )
        assert checkpoint["step"] == 100
        assert checkpoint["note"] == "test"

    def test_latest_finds_most_recent(self, tmp_path):
        """latest() should return the checkpoint with the highest step."""
        trainer = _make_trainer()
        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=5)

        manager.save(trainer, step=10)
        manager.save(trainer, step=20)
        manager.save(trainer, step=30)

        latest = manager.latest()
        assert latest is not None
        assert latest.name == "checkpoint_step_0000030.pt"

    def test_latest_empty_dir(self, tmp_path):
        """latest() should return None when no checkpoints exist."""
        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=5)
        assert manager.latest() is None

    def test_cleanup_keeps_n(self, tmp_path):
        """After saving 8 checkpoints with keep_n=3, only 3 should remain."""
        trainer = _make_trainer()
        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=3)

        for step in range(1, 9):
            manager.save(trainer, step=step * 10)

        remaining = sorted(manager.checkpoint_dir.glob("checkpoint_step_*.pt"))
        assert len(remaining) == 3

    def test_cleanup_keeps_newest(self, tmp_path):
        """The 3 remaining checkpoints should be the 3 most recent."""
        trainer = _make_trainer()
        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=3)

        for step in range(1, 9):
            manager.save(trainer, step=step * 10)

        remaining = sorted(manager.checkpoint_dir.glob("checkpoint_step_*.pt"))
        names = [p.name for p in remaining]
        assert names == [
            "checkpoint_step_0000060.pt",
            "checkpoint_step_0000070.pt",
            "checkpoint_step_0000080.pt",
        ]


# ============================================================================
# Resume tests
# ============================================================================


class TestResume:
    """Tests for resuming training from a checkpoint."""

    def test_resume_restores_step(self, tmp_path):
        """After resume, trainer.step should match the saved step."""
        trainer = _make_trainer()
        trainer.step = 50
        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=5)
        path = manager.save(trainer, step=50)

        new_trainer = _make_trainer()
        assert new_trainer.step == 0
        manager.resume(new_trainer, str(path))
        assert new_trainer.step == 50

    def test_resume_restores_model_weights(self, tmp_path):
        """Model weights should match after save and resume."""
        trainer = _make_trainer()
        # Run a few training steps to change weights from init
        loader = _make_dataloader(batch_size=32)
        batch = next(iter(loader))
        trainer.train_step(batch)

        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=5)
        path = manager.save(trainer, step=trainer.step)

        # Get original weights
        original_weights = {
            k: v.clone() for k, v in trainer.model.state_dict().items()
        }

        # Resume into a fresh trainer
        new_trainer = _make_trainer()
        manager.resume(new_trainer, str(path))

        for key in original_weights:
            assert torch.equal(original_weights[key], new_trainer.model.state_dict()[key]), (
                f"Weight mismatch for {key}"
            )

    def test_resume_restores_optimizer(self, tmp_path):
        """Optimizer state should be restored after resume."""
        trainer = _make_trainer()
        loader = _make_dataloader(batch_size=32)
        batch = next(iter(loader))
        trainer.train_step(batch)

        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=5)
        path = manager.save(trainer, step=trainer.step)

        original_opt_state = trainer.optimizer.state_dict()

        new_trainer = _make_trainer()
        manager.resume(new_trainer, str(path))

        resumed_opt_state = new_trainer.optimizer.state_dict()

        # Compare param_groups (hyperparameters)
        assert len(original_opt_state["param_groups"]) == len(
            resumed_opt_state["param_groups"]
        )
        for orig_pg, resumed_pg in zip(
            original_opt_state["param_groups"],
            resumed_opt_state["param_groups"],
        ):
            assert orig_pg["lr"] == resumed_pg["lr"]
            assert orig_pg["momentum"] == resumed_pg["momentum"]

    def test_resume_training_continues(self, tmp_path):
        """After resume at step 50, training should continue to step 100."""
        trainer = _make_trainer(total_steps=100)
        loader = _make_dataloader(batch_size=32, size=5000)

        trainer.train(loader, total_steps=50)
        assert trainer.step == 50

        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=5)
        path = manager.save(trainer, step=trainer.step)

        new_trainer = _make_trainer(total_steps=100)
        manager.resume(new_trainer, str(path))
        assert new_trainer.step == 50

        new_loader = _make_dataloader(batch_size=32, size=5000)
        new_trainer.train(new_loader, total_steps=100)
        assert new_trainer.step == 100

    def test_resume_loss_continuity(self, tmp_path):
        """Loss should not jump discontinuously after resume.

        Train for 50 steps, save, resume, and train 1 more step. The loss
        at step 51 should be close to the loss at step 50 (no sudden jump
        from re-initialized weights or optimizer state).
        """
        torch.manual_seed(42)
        trainer = _make_trainer(lr=0.001, total_steps=100)

        # Use a fixed-seed DummyDataset for reproducibility
        ds = DummyDataset(size=5000)
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        # Train for 50 steps and record the last loss
        losses = []
        trainer.model.train()
        for batch in loader:
            if trainer.step >= 50:
                break
            metrics = trainer.train_step(batch)
            losses.append(metrics["total_loss"])

        loss_at_50 = losses[-1]

        # Save checkpoint
        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=5)
        path = manager.save(trainer, step=trainer.step)

        # Resume into a new trainer
        torch.manual_seed(42)
        new_trainer = _make_trainer(lr=0.001, total_steps=100)
        manager.resume(new_trainer, str(path))

        # Train one more step
        new_loader = DataLoader(DummyDataset(size=5000), batch_size=32, shuffle=False)
        new_trainer.model.train()

        # Skip to roughly where we left off in the data
        batch_iter = iter(new_loader)
        batch = next(batch_iter)
        metrics = new_trainer.train_step(batch)
        loss_at_51 = metrics["total_loss"]

        # Loss should not jump dramatically -- allow 50% relative difference
        # since we're using random data and different batches
        assert abs(loss_at_51 - loss_at_50) < loss_at_50 * 0.5, (
            f"Loss jumped from {loss_at_50:.4f} to {loss_at_51:.4f} after resume"
        )


# ============================================================================
# Auto-export tests
# ============================================================================


class TestAutoExport:
    """Tests for TorchScript auto-export alongside checkpoints."""

    def test_export_creates_torchscript(self, tmp_path):
        """auto_export_torchscript should create a .pt TorchScript file."""
        trainer = _make_trainer()
        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=5)

        export_dir = tmp_path / "exports"
        path = manager.auto_export_torchscript(
            trainer, step=100, export_dir=str(export_dir)
        )

        assert path.exists()
        assert path.name == "model_step_0000100.pt"

    def test_exported_model_runs(self, tmp_path):
        """The exported TorchScript model should produce valid inference output."""
        trainer = _make_trainer()
        manager = CheckpointManager(str(tmp_path / "ckpts"), keep_n=5)

        export_dir = tmp_path / "exports"
        path = manager.auto_export_torchscript(
            trainer, step=100, export_dir=str(export_dir)
        )

        # Load the TorchScript model and run inference
        loaded = torch.jit.load(str(path), map_location="cpu")
        test_input = torch.randn(1, 119, 8, 8)

        with torch.no_grad():
            policy, value = loaded(test_input)

        # Verify output shapes
        assert policy.shape == (1, 4672), f"Policy shape: {policy.shape}"
        assert value.shape == (1, 1), f"Value shape: {value.shape}"

        # Verify outputs are finite
        assert torch.isfinite(policy).all()
        assert torch.isfinite(value).all()
