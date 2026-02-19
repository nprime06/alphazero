"""Weight distribution for AlphaZero self-play workers.

Manages exporting trained model weights as TorchScript files and tracking
versions so self-play workers can detect and hot-swap to newer models.

The weight distribution system uses a shared filesystem directory:

    weights_dir/
    ├── latest.txt          # Contains the version number of the latest model
    ├── model_v000001.pt    # TorchScript model version 1
    ├── model_v000002.pt    # TorchScript model version 2
    └── ...

The coordinator exports new weights after each training cycle. Self-play
workers poll ``latest.txt`` and load the new model when the version changes.

Usage (coordinator side)::

    publisher = WeightPublisher("./weights")
    publisher.publish(trainer.model, step=10000)

Usage (worker side)::

    watcher = WeightWatcher("./weights")
    while True:
        if watcher.has_new_version():
            model = watcher.load_latest()
            # ... use model for self-play ...
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn


class WeightPublisher:
    """Publishes trained model weights for self-play workers.

    Exports the current model as a TorchScript file and maintains a
    ``latest.txt`` pointer so that workers can detect new versions without
    scanning the directory. Old weight files are automatically cleaned up
    to prevent unbounded disk usage.

    Args:
        weights_dir: Directory for storing weight files.
        keep_n: Number of weight versions to keep. Older versions are deleted.
    """

    def __init__(self, weights_dir: str, keep_n: int = 5, fp16: bool = False) -> None:
        self._weights_dir = Path(weights_dir)
        self._weights_dir.mkdir(parents=True, exist_ok=True)
        self._keep_n = keep_n
        self._fp16 = fp16

        # Recover version counter from latest.txt if it exists (allows
        # restarting the coordinator without resetting the version sequence).
        self._version = 0
        latest_file = self._weights_dir / "latest.txt"
        if latest_file.exists():
            text = latest_file.read_text().strip()
            if text:
                self._version = int(text)

    def publish(self, model: nn.Module, step: int = 0) -> Path:
        """Export model as TorchScript and publish as new version.

        Steps:
        1. Increment version counter
        2. Export model via torch.jit.trace to model_v{version:06d}.pt
        3. Write version number to latest.txt (atomic write via temp file +
           rename)
        4. Clean up old versions beyond keep_n

        Args:
            model: The trained model (nn.Module). Set to eval mode before
                tracing.
            step: Training step (embedded as context for reference, not used
                in the filename).

        Returns:
            Path to the exported model file.
        """
        self._version += 1
        version = self._version

        model_path = self._weights_dir / f"model_v{version:06d}.pt"

        if self._fp16:
            from neural.export import export_fp16
            export_fp16(model, str(model_path))
        else:
            # Set to eval mode for deterministic BatchNorm behavior during tracing
            was_training = model.training
            model.eval()

            # Create dummy input for tracing -- shape (1, 119, 8, 8) matches the
            # standard AlphaZero chess encoding
            device = next(model.parameters()).device
            dummy_input = torch.randn(1, 119, 8, 8, device=device)

            with torch.no_grad():
                traced = torch.jit.trace(model, dummy_input)
            traced.save(str(model_path))

            # Restore original training mode
            if was_training:
                model.train()

        # Atomically update latest.txt: write to a temp file in the same
        # directory, then rename. os.replace() is atomic on POSIX and
        # guarantees that workers never read a partial version number.
        latest_file = self._weights_dir / "latest.txt"
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._weights_dir), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(str(version))
            os.replace(tmp_path, str(latest_file))
        except BaseException:
            # Clean up the temp file on failure
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        self._cleanup_old()

        return model_path

    def _cleanup_old(self) -> None:
        """Delete old weight files, keeping only the most recent keep_n."""
        model_files = sorted(self._weights_dir.glob("model_v*.pt"))
        if len(model_files) > self._keep_n:
            for old in model_files[: len(model_files) - self._keep_n]:
                old.unlink()

    @property
    def current_version(self) -> int:
        """Return the current published version number."""
        return self._version


class WeightWatcher:
    """Watches for new model weights published by the coordinator.

    Designed for self-play workers that need to detect and load updated
    models during training. Workers poll :meth:`has_new_version` and call
    :meth:`load_latest` when a new version is available.

    Args:
        weights_dir: Directory where WeightPublisher writes models.
    """

    def __init__(self, weights_dir: str) -> None:
        self._weights_dir = Path(weights_dir)
        self._last_version = 0

    def get_latest_version(self) -> int:
        """Read the latest version number from latest.txt.

        Returns 0 if no model has been published yet (i.e., latest.txt does
        not exist or is empty).
        """
        latest_file = self._weights_dir / "latest.txt"
        if not latest_file.exists():
            return 0
        text = latest_file.read_text().strip()
        if not text:
            return 0
        return int(text)

    def has_new_version(self) -> bool:
        """Check if a newer version is available than what we last loaded."""
        return self.get_latest_version() > self._last_version

    def load_latest(self, device: str = "cpu") -> torch.jit.ScriptModule:
        """Load the latest published TorchScript model.

        Updates internal version tracker so :meth:`has_new_version` returns
        False until a new version appears.

        Args:
            device: Device to load model onto ("cpu" or "cuda").

        Returns:
            The loaded TorchScript model.

        Raises:
            FileNotFoundError: If no model has been published yet.
        """
        version = self.get_latest_version()
        if version == 0:
            raise FileNotFoundError(
                f"No model has been published yet in {self._weights_dir}"
            )

        path = self.model_path(version)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        model = torch.jit.load(str(path), map_location=device)
        self._last_version = version
        return model

    def model_path(self, version: int) -> Path:
        """Return the path to a specific model version.

        Args:
            version: The version number.

        Returns:
            Path to the model file (may or may not exist on disk).
        """
        return self._weights_dir / f"model_v{version:06d}.pt"
