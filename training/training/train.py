"""Single-GPU training loop for AlphaZero.

Implements the core training loop with SGD + momentum, multi-step learning rate
decay, and optional ``torch.compile`` acceleration.  Designed to run on a single
NVIDIA H200 GPU.

The training loop follows the original AlphaZero paper:

    - **Optimizer**: SGD with momentum 0.9 and weight decay 1e-4.
    - **Learning rate schedule**: starts at 0.2, decays by 10x at steps
      100k, 300k, and 500k.
    - **Batch size**: 4096 positions per step.
    - **Total steps**: 700k (the paper trains for ~700k mini-batch updates).

Usage::

    # Train on real self-play data
    python -m training.train --data-dir ./data --steps 100000

    # Quick smoke test with random data
    python -m training.train --dummy-data --network tiny --steps 100 --batch-size 32

    # Full training run
    python -m training.train --data-dir ./data --network full --steps 700000
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural.config import NetworkConfig
from neural.losses import AlphaZeroLoss
from neural.network import AlphaZeroNetwork
from training.buffer import ReplayBuffer
from training.dataloader import DummyDataset, create_dataloader


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Defaults match the original AlphaZero paper where applicable.

    Attributes:
        data_dir: Path to directory containing ``.msgpack`` game files.
        batch_size: Number of positions per mini-batch.  4096 matches the
            paper and saturates a single H200 GPU.
        num_workers: Number of DataLoader worker processes.
        samples_per_epoch: Number of samples per epoch (relevant only for
            the ``ReplayDataset`` iterable dataset).
        learning_rate: Initial learning rate.  0.2 is the paper's default.
        momentum: SGD momentum.
        weight_decay: L2 regularization coefficient.
        lr_milestones: Steps at which the learning rate is decayed by
            ``lr_gamma``.
        lr_gamma: Multiplicative decay factor at each milestone.
        total_steps: Total number of training steps (mini-batch updates).
        log_interval: Print metrics every N steps.
        network_config: Which network preset to use (``"tiny"``,
            ``"small"``, ``"medium"``, or ``"full"``).
        use_compile: Whether to use ``torch.compile`` for a potential
            speed-up.  Requires PyTorch >= 2.0.
    """

    # Data
    data_dir: str = "./data"
    batch_size: int = 4096
    num_workers: int = 4
    samples_per_epoch: int = 200_000

    # Optimizer
    learning_rate: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # LR schedule: multi-step decay
    lr_milestones: list = field(default_factory=lambda: [100_000, 300_000, 500_000])
    lr_gamma: float = 0.1

    # Training
    total_steps: int = 700_000
    log_interval: int = 100

    # Network
    network_config: str = "full"

    # torch.compile
    use_compile: bool = True

    # Mixed precision
    use_amp: bool = True
    """Whether to use automatic mixed precision (AMP) for ~2x training speedup on GPU. Only effective on CUDA devices."""


# ============================================================================
# Trainer
# ============================================================================


class Trainer:
    """AlphaZero training manager.

    Owns the model, optimizer, LR scheduler, and loss function.  Provides
    :meth:`train_step` for a single gradient update and :meth:`train` for
    the full training loop.

    Args:
        config: Training hyperparameters.
        device: The torch device to train on (should be ``"cuda"``).
    """

    def __init__(self, config: TrainConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.step: int = 0

        # -- Build the network --
        net_config = self._get_network_config()
        self.model = AlphaZeroNetwork(net_config).to(device)

        # -- Optional torch.compile --
        if config.use_compile and hasattr(torch, "compile"):
            self.compiled_model = torch.compile(self.model)
        else:
            self.compiled_model = self.model

        # -- Loss function --
        self.loss_fn = AlphaZeroLoss().to(device)

        # -- Optimizer (SGD + momentum, as in the paper) --
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        # -- LR scheduler: step once per training step (not per epoch) --
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.lr_milestones,
            gamma=config.lr_gamma,
        )

        # -- AMP GradScaler for mixed precision training --
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
        self.use_amp = config.use_amp and device.type == "cuda"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_network_config(self) -> NetworkConfig:
        """Resolve the string network preset to a :class:`NetworkConfig`."""
        configs = {
            "tiny": NetworkConfig.tiny,
            "small": NetworkConfig.small,
            "medium": NetworkConfig.medium,
            "full": NetworkConfig.full,
        }
        factory = configs.get(self.config.network_config, NetworkConfig.full)
        return factory()

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train_step(self, batch) -> dict:
        """Execute one training step (forward + backward + optimizer step).

        Args:
            batch: A tuple ``(boards, target_policies, target_values)`` from
                the DataLoader.

        Returns:
            A dict with keys ``total_loss``, ``policy_loss``, ``value_loss``,
            and ``learning_rate`` for logging.
        """
        boards, target_policies, target_values = batch
        boards = boards.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)

        # Forward pass with optional AMP autocast
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            policy_logits, value = self.compiled_model(boards)
            loss_result = self.loss_fn(
                policy_logits, value, target_policies, target_values
            )

        # Backward pass with GradScaler
        self.optimizer.zero_grad()
        self.scaler.scale(loss_result.total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self.step += 1

        return {
            "total_loss": loss_result.total_loss.item(),
            "policy_loss": loss_result.policy_loss.item(),
            "value_loss": loss_result.value_loss.item(),
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

    def train(
        self,
        dataloader: DataLoader,
        total_steps: Optional[int] = None,
    ) -> None:
        """Run the training loop.

        Iterates over ``dataloader`` batches, calling :meth:`train_step` for
        each one, until ``total_steps`` have been completed.  Logs metrics to
        stdout at the interval specified by ``config.log_interval``.

        Args:
            dataloader: A PyTorch DataLoader yielding
                ``(boards, policies, values)`` tuples.
            total_steps: Override ``config.total_steps`` if provided.
        """
        total_steps = total_steps or self.config.total_steps
        self.model.train()

        step_times: list[float] = []

        for batch in dataloader:
            if self.step >= total_steps:
                break

            start = time.time()
            metrics = self.train_step(batch)
            elapsed = time.time() - start
            step_times.append(elapsed)

            if self.step % self.config.log_interval == 0:
                avg_time = sum(step_times[-100:]) / len(step_times[-100:])
                samples_per_sec = self.config.batch_size / avg_time
                print(
                    f"[Step {self.step:>7d}/{total_steps}] "
                    f"loss={metrics['total_loss']:.4f} "
                    f"(policy={metrics['policy_loss']:.4f} "
                    f"value={metrics['value_loss']:.4f}) "
                    f"lr={metrics['learning_rate']:.6f} "
                    f"samples/s={samples_per_sec:.0f}"
                )


# ============================================================================
# CLI entry point
# ============================================================================


def main() -> None:
    """Parse command-line arguments and run training."""
    parser = argparse.ArgumentParser(description="AlphaZero Training")

    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory for checkpoints, logs, and TensorBoard. "
        "Auto-resumes from latest checkpoint if one exists. "
        "Created with timestamp if omitted.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing .msgpack game files (default: ./data)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size (default: 4096)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=700_000,
        help="Total training steps (default: 700000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.2,
        help="Initial learning rate (default: 0.2)",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="full",
        choices=["tiny", "small", "medium", "full"],
        help="Network preset (default: full)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4)",
    )
    parser.add_argument(
        "--dummy-data",
        action="store_true",
        help="Use random data for testing (no game files needed)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (default: 1000)",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=5,
        help="Number of recent checkpoints to keep (default: 5)",
    )
    args = parser.parse_args()

    # -- Run directory: everything for this run lives here --
    if args.run_dir is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(f"runs/train_{timestamp}")
    else:
        run_dir = Path(args.run_dir)

    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # -- Device selection: CUDA only, fall back to CPU for testing --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        total_steps=args.steps,
        learning_rate=args.lr,
        network_config=args.network,
        use_compile=not args.no_compile,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
    )

    trainer = Trainer(config, device)

    # -- Checkpoint manager --
    from training.checkpoint import CheckpointManager
    ckpt_manager = CheckpointManager(
        str(checkpoint_dir), keep_n=args.keep_checkpoints
    )

    # -- Auto-resume from latest checkpoint --
    latest_ckpt = ckpt_manager.latest()
    if latest_ckpt is not None:
        print(f"Resuming from checkpoint: {latest_ckpt}")
        ckpt_manager.resume(trainer, str(latest_ckpt))
        print(f"  Resumed at step {trainer.step}")
    else:
        print("Starting training from scratch")

    # -- Metrics logger --
    from training.metrics import MetricsLogger
    logger = MetricsLogger(log_dir=str(tensorboard_dir))

    print(f"Run directory: {run_dir}")
    print(f"  checkpoints: {checkpoint_dir}")
    print(f"  tensorboard: {tensorboard_dir}")
    print(f"  device: {device}")
    print(f"  network: {config.network_config}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  total_steps: {config.total_steps}")
    print(f"  checkpoint_interval: {args.checkpoint_interval}")

    # -- Data --
    if args.dummy_data:
        dataset = DummyDataset(
            size=max(config.total_steps * config.batch_size, 10_000)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=config.num_workers,
        )
    else:
        buffer = ReplayBuffer(config.data_dir)
        num_games = buffer.scan()
        print(f"Found {num_games} games in replay buffer")
        dataloader = create_dataloader(
            buffer,
            config.batch_size,
            config.samples_per_epoch,
            config.num_workers,
        )

    # -- Training loop with checkpointing and logging --
    total_steps = config.total_steps
    trainer.model.train()

    step_times: list[float] = []

    for batch in dataloader:
        if trainer.step >= total_steps:
            break

        start = time.time()
        metrics = trainer.train_step(batch)
        elapsed = time.time() - start
        step_times.append(elapsed)

        # Log metrics
        logger.log_step(trainer.step, metrics, batch_size=config.batch_size)

        # Save checkpoint at interval
        if trainer.step % args.checkpoint_interval == 0 and trainer.step > 0:
            path = ckpt_manager.save(trainer, trainer.step)
            ckpt_manager.auto_export_torchscript(trainer, trainer.step)
            logger.log_checkpoint(trainer.step, str(path))
            print(f"  [Checkpoint saved: {path}]")

    # -- Final checkpoint --
    if trainer.step > 0:
        path = ckpt_manager.save(trainer, trainer.step)
        ckpt_manager.auto_export_torchscript(trainer, trainer.step)
        logger.log_checkpoint(trainer.step, str(path))
        print(f"  [Final checkpoint: {path}]")

    logger.close()
    print(f"Training complete at step {trainer.step}. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
