"""TensorBoard and W&B monitoring for AlphaZero training.

Provides a :class:`MetricsLogger` that logs training metrics (losses, learning
rate, throughput) to TensorBoard and optionally to Weights & Biases.  Console
output is also generated at configurable intervals.

The logger is intentionally device-agnostic -- it receives plain Python floats
from the training loop and writes them to the configured backends.

Usage::

    from training.metrics import MetricsLogger, compute_policy_accuracy

    logger = MetricsLogger(log_dir="runs/experiment_01")

    for step, batch in enumerate(dataloader):
        metrics = trainer.train_step(batch)
        logger.log_step(step, metrics, batch_size=4096)

    logger.close()
"""

from __future__ import annotations

import time

import torch
from torch.utils.tensorboard import SummaryWriter

# wandb is optional -- gracefully degrade if not installed.
try:
    import wandb as _wandb
except ImportError:  # pragma: no cover
    _wandb = None  # type: ignore[assignment]


# ============================================================================
# MetricsLogger
# ============================================================================


class MetricsLogger:
    """Training metrics logger with TensorBoard and optional W&B support.

    Logs training metrics (losses, learning rate, throughput) to TensorBoard
    and optionally to Weights & Biases. Also logs to console at configurable
    intervals.

    Args:
        log_dir: Directory for TensorBoard logs. Created if it doesn't exist.
        use_wandb: Whether to log to Weights & Biases (requires wandb package).
        wandb_project: W&B project name (only used if use_wandb=True).
        wandb_run_name: W&B run name (only used if use_wandb=True).
        console_interval: Log to console every N steps.
    """

    def __init__(
        self,
        log_dir: str = "runs",
        use_wandb: bool = False,
        wandb_project: str = "alphazero-chess",
        wandb_run_name: str | None = None,
        console_interval: int = 100,
    ) -> None:
        self.console_interval = console_interval

        # -- TensorBoard --
        self.writer = SummaryWriter(log_dir=log_dir)

        # -- Weights & Biases (optional) --
        self._wandb_active = False
        if use_wandb:
            if _wandb is None:
                print(
                    "WARNING: use_wandb=True but wandb is not installed. "
                    "Falling back to TensorBoard only."
                )
            else:
                _wandb.init(project=wandb_project, name=wandb_run_name)
                self._wandb_active = True

        # -- Throughput tracking --
        self._last_step_time: float | None = None

    # ------------------------------------------------------------------ #
    # Logging methods
    # ------------------------------------------------------------------ #

    def log_step(self, step: int, metrics: dict, batch_size: int = 0) -> None:
        """Log metrics for a single training step.

        Expected metrics dict keys (from Trainer.train_step):
            - total_loss (float)
            - policy_loss (float)
            - value_loss (float)
            - learning_rate (float)

        Additional computed metrics:
            - samples_per_sec (if batch_size > 0)
            - step_time_ms

        Args:
            step: Current training step number.
            metrics: Dict of metric name to value.
            batch_size: Batch size for throughput calculation.  If 0,
                throughput metrics are not logged.
        """
        now = time.time()

        # -- Compute step timing --
        step_time_ms: float | None = None
        samples_per_sec: float | None = None

        if self._last_step_time is not None:
            elapsed = now - self._last_step_time
            step_time_ms = elapsed * 1000.0
            if batch_size > 0 and elapsed > 0:
                samples_per_sec = batch_size / elapsed

        self._last_step_time = now

        # -- TensorBoard scalars --
        self.writer.add_scalar("loss/total", metrics["total_loss"], step)
        self.writer.add_scalar("loss/policy", metrics["policy_loss"], step)
        self.writer.add_scalar("loss/value", metrics["value_loss"], step)
        self.writer.add_scalar("learning_rate", metrics["learning_rate"], step)

        if step_time_ms is not None:
            self.writer.add_scalar("throughput/step_time_ms", step_time_ms, step)
        if samples_per_sec is not None:
            self.writer.add_scalar("throughput/samples_per_sec", samples_per_sec, step)

        # -- Weights & Biases --
        if self._wandb_active:
            wandb_log: dict = {
                "loss/total": metrics["total_loss"],
                "loss/policy": metrics["policy_loss"],
                "loss/value": metrics["value_loss"],
                "learning_rate": metrics["learning_rate"],
            }
            if step_time_ms is not None:
                wandb_log["throughput/step_time_ms"] = step_time_ms
            if samples_per_sec is not None:
                wandb_log["throughput/samples_per_sec"] = samples_per_sec
            _wandb.log(wandb_log, step=step)  # type: ignore[union-attr]

        # -- Console output --
        if step > 0 and step % self.console_interval == 0:
            throughput_str = ""
            if samples_per_sec is not None:
                throughput_str = f" samples/s={samples_per_sec:.0f}"
            print(
                f"[Step {step:>7d}] "
                f"loss={metrics['total_loss']:.4f} "
                f"(policy={metrics['policy_loss']:.4f} "
                f"value={metrics['value_loss']:.4f}) "
                f"lr={metrics['learning_rate']:.6f}"
                f"{throughput_str}"
            )

    def log_policy_accuracy(self, step: int, accuracy: float) -> None:
        """Log policy top-1 accuracy (fraction of moves matching MCTS target).

        Args:
            step: Current training step number.
            accuracy: Accuracy as a float in [0, 1].
        """
        self.writer.add_scalar("accuracy/policy_top1", accuracy, step)
        if self._wandb_active:
            _wandb.log({"accuracy/policy_top1": accuracy}, step=step)  # type: ignore[union-attr]

    def log_value_mse(self, step: int, mse: float) -> None:
        """Log value head MSE against game outcomes.

        Args:
            step: Current training step number.
            mse: Mean squared error between predicted and actual values.
        """
        self.writer.add_scalar("accuracy/value_mse", mse, step)
        if self._wandb_active:
            _wandb.log({"accuracy/value_mse": mse}, step=step)  # type: ignore[union-attr]

    def log_checkpoint(self, step: int, path: str) -> None:
        """Log that a checkpoint was saved (text event in TensorBoard).

        Args:
            step: Training step at which the checkpoint was saved.
            path: Filesystem path to the saved checkpoint file.
        """
        self.writer.add_text("checkpoints", f"Saved checkpoint: {path}", step)
        if self._wandb_active:
            _wandb.log({"checkpoint_path": path}, step=step)  # type: ignore[union-attr]

    def close(self) -> None:
        """Flush and close all writers."""
        self.writer.close()
        if self._wandb_active:
            _wandb.finish()  # type: ignore[union-attr]
            self._wandb_active = False


# ============================================================================
# Helper functions
# ============================================================================


def compute_policy_accuracy(
    policy_logits: torch.Tensor,
    target_policy: torch.Tensor,
) -> float:
    """Compute top-1 policy accuracy.

    Checks what fraction of positions have the same argmax (highest probability
    move) in both the network output and the MCTS target distribution.

    Args:
        policy_logits: Raw logits from policy head, shape (B, 4672).
        target_policy: MCTS target distribution, shape (B, 4672).

    Returns:
        Accuracy as a float in [0, 1].
    """
    pred_moves = policy_logits.argmax(dim=1)
    target_moves = target_policy.argmax(dim=1)
    return (pred_moves == target_moves).float().mean().item()
