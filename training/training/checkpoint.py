"""Checkpoint save/resume and TorchScript export.

Manages training checkpoints with atomic writes, automatic cleanup of older
checkpoints, and optional TorchScript export for inference.

Checkpoints are saved as ``checkpoint_step_{step:07d}.pt`` files containing:

    - ``model_state_dict``: Raw model weights (never the compiled version).
    - ``optimizer_state_dict``: Optimizer state for resuming training.
    - ``scheduler_state_dict``: Learning rate scheduler state.
    - ``step``: Current training step counter.
    - ``config``: :class:`~training.train.TrainConfig` serialized as a dict.
    - Any extra metadata passed by the caller.

Atomic writes are used to prevent partial checkpoint files: data is first
written to a ``.tmp`` file and then atomically renamed to the final path
via ``os.replace()``.

Usage::

    from training.checkpoint import CheckpointManager
    from training.train import TrainConfig, Trainer

    trainer = Trainer(TrainConfig(), device)
    manager = CheckpointManager("./checkpoints", keep_n=5)

    # Save a checkpoint
    path = manager.save(trainer, step=1000)

    # Resume from the latest checkpoint
    latest = manager.latest()
    if latest:
        manager.resume(trainer, str(latest))

    # Export a TorchScript model alongside the checkpoint
    manager.auto_export_torchscript(trainer, step=1000)
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch


class CheckpointManager:
    """Manages saving, loading, and cleaning up training checkpoints.

    Args:
        checkpoint_dir: Directory where checkpoint files are stored.
            Created automatically if it does not exist.
        keep_n: Maximum number of checkpoints to keep on disk.  After each
            save, older checkpoints beyond this limit are deleted.
    """

    def __init__(self, checkpoint_dir: str, keep_n: int = 5) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_n = keep_n
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        trainer,
        step: int,
        extra_metadata: dict | None = None,
    ) -> Path:
        """Save a training checkpoint with atomic write.

        Writes the checkpoint to a temporary file first, then atomically
        renames it to the final path to avoid leaving partial files on disk
        in case of a crash or interruption.

        Args:
            trainer: A :class:`~training.train.Trainer` instance.
            step: Current training step number.
            extra_metadata: Optional dict of additional metadata to include
                in the checkpoint.

        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint = {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "step": step,
            "config": asdict(trainer.config),
        }

        if extra_metadata is not None:
            checkpoint.update(extra_metadata)

        filename = f"checkpoint_step_{step:07d}.pt"
        final_path = self.checkpoint_dir / filename
        tmp_path = self.checkpoint_dir / f"{filename}.tmp"

        torch.save(checkpoint, str(tmp_path))
        os.replace(str(tmp_path), str(final_path))

        self._cleanup_old()

        return final_path

    def _cleanup_old(self) -> None:
        """Delete older checkpoints, keeping only the ``keep_n`` most recent.

        Checkpoints are sorted by filename (lexicographic order), which works
        correctly because the step number is zero-padded to 7 digits.
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > self.keep_n:
            for old in checkpoints[: len(checkpoints) - self.keep_n]:
                old.unlink()

    def load(self, path: str) -> dict:
        """Load a checkpoint dict from disk.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The checkpoint dict with keys ``model_state_dict``,
            ``optimizer_state_dict``, ``scheduler_state_dict``, ``step``,
            ``config``, and any extra metadata.
        """
        return torch.load(path, map_location="cpu", weights_only=False)

    def latest(self) -> Optional[Path]:
        """Find the most recent checkpoint in the directory.

        Returns:
            Path to the most recent checkpoint, or ``None`` if no checkpoints
            exist.
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if checkpoints:
            return checkpoints[-1]
        return None

    def resume(self, trainer, path: str) -> None:
        """Load a checkpoint and restore trainer state.

        Restores the model weights, optimizer state, scheduler state, and
        step counter from a previously saved checkpoint.

        Args:
            trainer: A :class:`~training.train.Trainer` instance to restore
                state into.
            path: Path to the checkpoint file.
        """
        checkpoint = self.load(path)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.step = checkpoint["step"]

    def auto_export_torchscript(
        self,
        trainer,
        step: int,
        export_dir: str | None = None,
    ) -> Path:
        """Export the model as TorchScript after saving a checkpoint.

        Uses :func:`neural.export.export_torchscript` to trace the model and
        save it in a format loadable from C++ or Rust without Python.

        Args:
            trainer: A :class:`~training.train.Trainer` instance.
            step: Current training step (used in the filename).
            export_dir: Directory to save the exported model.  Defaults to
                the checkpoint directory if not specified.

        Returns:
            Path to the exported TorchScript model.
        """
        from neural.export import export_torchscript

        if export_dir is None:
            out_dir = self.checkpoint_dir
        else:
            out_dir = Path(export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        export_path = out_dir / f"model_step_{step:07d}.pt"
        export_torchscript(trainer.model, export_path)

        return export_path
