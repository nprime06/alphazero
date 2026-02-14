"""AlphaZero training pipeline coordinator.

Orchestrates the full training loop:
1. Launch self-play workers to generate games
2. Wait for enough new data in the replay buffer
3. Launch training job
4. Export new weights after training
5. Evaluate new model vs current best
6. If new model wins >55%: promote to best model
7. Repeat

The coordinator runs on a login node and submits Slurm jobs for
self-play and training. It uses the filesystem for coordination:
- Replay buffer directory for game data
- Weights directory for model versions
- Checkpoint directory for training state

Usage::

    python -m orchestrator.coordinator --config config.yaml

    # Or with defaults:
    python -m orchestrator.coordinator --data-dir ./data --weights-dir ./weights
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import time
import yaml
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class PipelineConfig:
    """Configuration for the AlphaZero training pipeline.

    Attributes:
        project_dir: Root project directory.
        data_dir: Directory for self-play game data (replay buffer).
        weights_dir: Directory for published model weights.
        checkpoint_dir: Directory for training checkpoints.

        selfplay_games_per_iteration: Number of games per self-play round.
        selfplay_simulations: MCTS simulations per move.
        selfplay_threads: CPU threads per self-play worker.
        selfplay_batch_size: Inference batch size.

        train_steps_per_iteration: Training steps per iteration.
        train_batch_size: Training batch size.
        train_network: Network preset ("tiny", "small", "medium", "full").
        train_gpus: Number of GPUs for training.

        eval_games: Number of games for model evaluation.
        eval_simulations: MCTS simulations for evaluation games.
        eval_win_threshold: Win rate threshold to promote new model (default 0.55).

        max_iterations: Maximum number of training iterations (0 = infinite).
        min_games_before_training: Minimum games in buffer before first training.
        weights_keep_n: Number of weight versions to keep.

        slurm_partition: Slurm partition name.
        slurm_time_selfplay: Wall time for self-play jobs.
        slurm_time_train: Wall time for training jobs.

        dry_run: If True, log what would be done without executing.
    """

    project_dir: str = "/home/willzhao/alphazero"
    data_dir: str = "data"
    weights_dir: str = "weights"
    checkpoint_dir: str = "checkpoints"

    # Self-play
    selfplay_games_per_iteration: int = 500
    selfplay_simulations: int = 800
    selfplay_threads: int = 8
    selfplay_batch_size: int = 8

    # Training
    train_steps_per_iteration: int = 1000
    train_batch_size: int = 4096
    train_network: str = "full"
    train_gpus: int = 1

    # Evaluation
    eval_games: int = 40
    eval_simulations: int = 400
    eval_win_threshold: float = 0.55

    # Pipeline
    max_iterations: int = 0
    min_games_before_training: int = 100
    weights_keep_n: int = 10

    # Slurm
    slurm_partition: str = "mit_normal_gpu"
    slurm_time_selfplay: str = "6:00:00"
    slurm_time_train: str = "12:00:00"

    # Runtime
    dry_run: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> PipelineConfig:
        """Load configuration from a YAML file.

        Unknown keys in the YAML file are silently ignored so that the
        config file can contain comments or future fields without breaking
        older code.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A PipelineConfig populated from the file.
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Only pass keys that are valid fields on the dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path relative to the project directory.

        If ``relative`` is already absolute, returns it as-is. Otherwise,
        joins it with ``project_dir``.

        Args:
            relative: A relative or absolute path string.

        Returns:
            Resolved absolute Path.
        """
        p = Path(relative)
        if p.is_absolute():
            return p
        return Path(self.project_dir) / p


# ============================================================================
# Pipeline State
# ============================================================================


@dataclass
class PipelineState:
    """Tracks the current state of the pipeline.

    Persisted to disk as a YAML file so the coordinator can resume
    after interruption. All fields default to zero for a fresh start.

    Attributes:
        iteration: Current iteration number (0-based).
        best_model_version: Version number of the current best model.
        total_games: Cumulative number of self-play games generated.
        total_train_steps: Cumulative number of training steps completed.
    """

    iteration: int = 0
    best_model_version: int = 0
    total_games: int = 0
    total_train_steps: int = 0

    def save(self, path: str) -> None:
        """Save state to a YAML file.

        Uses atomic write (write to temp, then rename) to avoid
        corruption from partial writes.

        Args:
            path: File path to save to.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = p.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
        tmp_path.replace(p)

    @classmethod
    def load(cls, path: str) -> PipelineState:
        """Load state from a YAML file.

        Returns a default (all-zeros) state if the file does not exist,
        allowing the coordinator to start fresh transparently.

        Args:
            path: File path to load from.

        Returns:
            Loaded pipeline state, or default state if file is missing.
        """
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


# ============================================================================
# Coordinator
# ============================================================================


class Coordinator:
    """Orchestrates the AlphaZero training pipeline.

    Each iteration of the pipeline:

    1. **Self-play**: generate games using the current best model.
    2. **Train**: update network weights on the accumulated game data.
    3. **Evaluate**: play a match between the new model and the current best.
    4. **Promote**: if the new model exceeds the win-rate threshold, it
       becomes the new best model.

    The coordinator is designed with two execution modes in mind:

    - **Local mode** (current): runs self-play as a subprocess, training
      via direct Python import, and evaluation via the orchestrator's
      ``evaluate_models`` function.
    - **Slurm mode** (future): submits jobs to a Slurm cluster. The
      ``_run_selfplay`` and ``_run_training`` methods are structured to
      be overridden or swapped for Slurm-based implementations.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        # Resolve directories
        self._data_dir = config.resolve_path(config.data_dir)
        self._weights_dir = config.resolve_path(config.weights_dir)
        self._checkpoint_dir = config.resolve_path(config.checkpoint_dir)
        self._project_dir = Path(config.project_dir)

        # Ensure directories exist
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._weights_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize pipeline state
        self._state_path = self._checkpoint_dir / "pipeline_state.yaml"
        self.state = PipelineState.load(str(self._state_path))

        logger.info(
            "Coordinator initialized: iteration=%d, best_model=v%d, "
            "total_games=%d, total_steps=%d",
            self.state.iteration,
            self.state.best_model_version,
            self.state.total_games,
            self.state.total_train_steps,
        )

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Run the training pipeline loop.

        Each iteration:
        1. Self-play: generate games using current best model
        2. Train: update network weights
        3. Evaluate: compare new model vs current best
        4. Maybe promote: if new model is stronger

        Runs until ``max_iterations`` is reached (if nonzero) or
        indefinitely (if ``max_iterations == 0``).
        """
        logger.info("Starting AlphaZero pipeline")
        logger.info(
            "Config: %d games/iter, %d train steps/iter, "
            "eval threshold=%.0f%%",
            self.config.selfplay_games_per_iteration,
            self.config.train_steps_per_iteration,
            self.config.eval_win_threshold * 100,
        )

        while True:
            # Check iteration limit
            if (
                self.config.max_iterations > 0
                and self.state.iteration >= self.config.max_iterations
            ):
                logger.info(
                    "Reached max iterations (%d). Stopping.",
                    self.config.max_iterations,
                )
                break

            self.state.iteration += 1
            iter_start = time.time()
            logger.info(
                "=== Iteration %d ===", self.state.iteration
            )

            # Step 1: Self-play
            self._run_selfplay()
            game_count = self._count_games()
            self.state.total_games = game_count
            logger.info("Replay buffer: %d games", game_count)

            # Wait for minimum games before first training
            if game_count < self.config.min_games_before_training:
                logger.info(
                    "Not enough games yet (%d < %d). Skipping training.",
                    game_count,
                    self.config.min_games_before_training,
                )
                self._save_state()
                continue

            # Step 2: Train
            self._run_training()

            # Step 3: Evaluate
            promoted = self._run_evaluation()

            # Step 4: Maybe promote
            if promoted:
                self._promote_model()

            # Save state and log summary
            self._save_state()
            elapsed = time.time() - iter_start
            self._log_iteration_summary(elapsed)

    # ------------------------------------------------------------------ #
    # Game counting
    # ------------------------------------------------------------------ #

    def _count_games(self) -> int:
        """Count .msgpack game files in the data directory.

        Recursively searches the data directory for files with the
        ``.msgpack`` extension.

        Returns:
            Number of game files found.
        """
        return len(list(self._data_dir.glob("**/*.msgpack")))

    # ------------------------------------------------------------------ #
    # Self-play (local mode)
    # ------------------------------------------------------------------ #

    def _run_selfplay(self) -> None:
        """Launch self-play to generate games.

        In local mode, calls the self-play binary directly as a
        subprocess. The binary reads the current best model from the
        weights directory and writes game files to the data directory.

        On the cluster, this method would be replaced with a Slurm
        job submission.
        """
        if self.config.dry_run:
            logger.info("[DRY RUN] Would run self-play: %d games",
                       self.config.selfplay_games_per_iteration)
            return

        # Find the model to use for self-play
        model_path = self._get_best_model_path()
        if model_path is None:
            logger.warning(
                "No model found for self-play. Skipping. "
                "Publish an initial model to %s first.",
                self._weights_dir,
            )
            return

        selfplay_binary = self._project_dir / "target" / "release" / "self-play"
        if not selfplay_binary.exists():
            logger.error(
                "Self-play binary not found at %s. "
                "Build with: cargo build --release -p self-play",
                selfplay_binary,
            )
            return

        # Output file for this iteration's games
        output_path = (
            self._data_dir
            / f"games_iter{self.state.iteration:04d}.msgpack"
        )

        cmd = [
            str(selfplay_binary),
            "--model", str(model_path),
            "--games", str(self.config.selfplay_games_per_iteration),
            "--output", str(output_path),
            "--sims", str(self.config.selfplay_simulations),
            "--threads", str(self.config.selfplay_threads),
            "--batch-size", str(self.config.selfplay_batch_size),
        ]

        logger.info("Running self-play: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(
                "Self-play failed (exit code %d):\n%s",
                result.returncode,
                result.stderr,
            )
        else:
            logger.info("Self-play completed successfully")
            if result.stdout:
                logger.info("Self-play output:\n%s", result.stdout)

    # ------------------------------------------------------------------ #
    # Training (local mode)
    # ------------------------------------------------------------------ #

    def _run_training(self) -> None:
        """Launch training to update network weights.

        In local mode, imports the Trainer and runs training steps
        directly. After training, exports the model via WeightPublisher
        so it can be used by self-play and evaluation.

        On the cluster, this method would be replaced with a Slurm
        job submission.
        """
        if self.config.dry_run:
            logger.info(
                "[DRY RUN] Would train for %d steps",
                self.config.train_steps_per_iteration,
            )
            return

        import torch
        from torch.utils.data import DataLoader

        from training.train import TrainConfig, Trainer
        from training.buffer import ReplayBuffer
        from training.dataloader import create_dataloader
        from training.checkpoint import CheckpointManager
        from orchestrator.weights import WeightPublisher

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training on device: %s", device)

        # Configure training
        train_config = TrainConfig(
            data_dir=str(self._data_dir),
            batch_size=self.config.train_batch_size,
            total_steps=self.config.train_steps_per_iteration,
            network_config=self.config.train_network,
        )

        # Create trainer
        trainer = Trainer(train_config, device)

        # Resume from checkpoint if available
        ckpt_manager = CheckpointManager(
            str(self._checkpoint_dir), keep_n=5
        )
        latest_ckpt = ckpt_manager.latest()
        if latest_ckpt:
            logger.info("Resuming from checkpoint: %s", latest_ckpt)
            ckpt_manager.resume(trainer, str(latest_ckpt))

        # Load replay buffer
        buffer = ReplayBuffer(str(self._data_dir))
        num_games = buffer.scan()
        logger.info("Training on %d games from replay buffer", num_games)

        if num_games == 0:
            logger.warning("No games in replay buffer. Skipping training.")
            return

        dataloader = create_dataloader(
            buffer,
            train_config.batch_size,
            train_config.samples_per_epoch,
            train_config.num_workers,
        )

        # Train
        start_step = trainer.step
        trainer.train(dataloader, total_steps=start_step + self.config.train_steps_per_iteration)
        self.state.total_train_steps = trainer.step
        logger.info(
            "Training complete: steps %d -> %d",
            start_step,
            trainer.step,
        )

        # Save checkpoint
        ckpt_path = ckpt_manager.save(trainer, trainer.step)
        logger.info("Checkpoint saved: %s", ckpt_path)

        # Publish weights for self-play and evaluation
        publisher = WeightPublisher(
            str(self._weights_dir),
            keep_n=self.config.weights_keep_n,
        )
        weight_path = publisher.publish(trainer.model, step=trainer.step)
        logger.info("Weights published: %s (v%d)", weight_path, publisher.current_version)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def _run_evaluation(self) -> bool:
        """Evaluate the new model against the current best.

        Plays a match between the latest published model and the
        current best model. If there is no previous best model (first
        iteration), the new model is automatically promoted.

        Returns:
            True if the new model should be promoted to best.
        """
        if self.config.dry_run:
            logger.info(
                "[DRY RUN] Would evaluate: %d games, threshold=%.0f%%",
                self.config.eval_games,
                self.config.eval_win_threshold * 100,
            )
            return False

        # Find the latest model (candidate)
        latest_version = self._get_latest_weight_version()
        if latest_version == 0:
            logger.warning("No model to evaluate. Skipping evaluation.")
            return False

        candidate_path = (
            self._weights_dir / f"model_v{latest_version:06d}.pt"
        )

        # Find the best model
        if self.state.best_model_version == 0:
            # No previous best -- auto-promote the first model
            logger.info(
                "No previous best model. Auto-promoting v%d.",
                latest_version,
            )
            return True

        if latest_version == self.state.best_model_version:
            logger.info(
                "Latest model (v%d) is already the best. Skipping evaluation.",
                latest_version,
            )
            return False

        best_path = (
            self._weights_dir
            / f"model_v{self.state.best_model_version:06d}.pt"
        )

        if not best_path.exists():
            logger.warning(
                "Best model file missing (%s). Auto-promoting v%d.",
                best_path,
                latest_version,
            )
            return True

        # Run the evaluation match
        from orchestrator.evaluate import evaluate_models

        logger.info(
            "Evaluating: v%d (candidate) vs v%d (best) -- %d games",
            latest_version,
            self.state.best_model_version,
            self.config.eval_games,
        )

        result = evaluate_models(
            model_a_path=str(candidate_path),
            model_b_path=str(best_path),
            num_games=self.config.eval_games,
            simulations=self.config.eval_simulations,
            device="cuda" if self._has_cuda() else "cpu",
            verbose=True,
        )

        logger.info("Evaluation result:\n%s", result.summary())

        if result.a_win_rate >= self.config.eval_win_threshold:
            logger.info(
                "New model wins %.1f%% >= %.1f%% threshold. Promoting!",
                result.a_win_rate * 100,
                self.config.eval_win_threshold * 100,
            )
            return True
        else:
            logger.info(
                "New model wins %.1f%% < %.1f%% threshold. Keeping current best.",
                result.a_win_rate * 100,
                self.config.eval_win_threshold * 100,
            )
            return False

    # ------------------------------------------------------------------ #
    # Model promotion
    # ------------------------------------------------------------------ #

    def _promote_model(self) -> None:
        """Promote the latest trained model to be the best model.

        Updates the pipeline state to record the new best model version.
        The model file itself remains in the weights directory; only the
        state pointer changes.
        """
        latest_version = self._get_latest_weight_version()
        if latest_version == 0:
            logger.warning("No model to promote.")
            return

        old_best = self.state.best_model_version
        self.state.best_model_version = latest_version
        logger.info(
            "Model promoted: v%d -> v%d",
            old_best,
            latest_version,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_best_model_path(self) -> Optional[Path]:
        """Get the file path of the current best model.

        Returns:
            Path to the best model, or None if no model exists.
        """
        if self.state.best_model_version > 0:
            path = (
                self._weights_dir
                / f"model_v{self.state.best_model_version:06d}.pt"
            )
            if path.exists():
                return path

        # Fall back to whatever latest.txt points to
        latest_file = self._weights_dir / "latest.txt"
        if latest_file.exists():
            text = latest_file.read_text().strip()
            if text:
                version = int(text)
                path = self._weights_dir / f"model_v{version:06d}.pt"
                if path.exists():
                    return path

        return None

    def _get_latest_weight_version(self) -> int:
        """Read the latest weight version from latest.txt.

        Returns:
            Version number, or 0 if no weights have been published.
        """
        latest_file = self._weights_dir / "latest.txt"
        if not latest_file.exists():
            return 0
        text = latest_file.read_text().strip()
        if not text:
            return 0
        return int(text)

    def _has_cuda(self) -> bool:
        """Check if CUDA is available (lazy import to avoid torch at module level)."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _save_state(self) -> None:
        """Persist the current pipeline state to disk."""
        self.state.save(str(self._state_path))

    def _log_iteration_summary(self, elapsed: float) -> None:
        """Log a summary of the completed iteration.

        Args:
            elapsed: Wall-clock time for the iteration in seconds.
        """
        logger.info(
            "Iteration %d complete in %.1fs | "
            "best_model=v%d | games=%d | train_steps=%d",
            self.state.iteration,
            elapsed,
            self.state.best_model_version,
            self.state.total_games,
            self.state.total_train_steps,
        )


# ============================================================================
# CLI entry point
# ============================================================================


def main() -> None:
    """CLI entry point for the coordinator."""
    parser = argparse.ArgumentParser(
        description="AlphaZero Pipeline Coordinator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Run with a YAML config file:
  python -m orchestrator.coordinator --config config.yaml

  # Run with command-line overrides:
  python -m orchestrator.coordinator --project-dir . --data-dir ./data

  # Dry run (shows what would happen):
  python -m orchestrator.coordinator --config config.yaml --dry-run
""",
    )
    parser.add_argument(
        "--config", type=str, help="YAML config file"
    )
    parser.add_argument(
        "--project-dir", type=str, default=".",
        help="Root project directory (default: .)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Self-play game data directory (default: data)",
    )
    parser.add_argument(
        "--weights-dir", type=str, default="weights",
        help="Model weights directory (default: weights)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Training checkpoint directory (default: checkpoints)",
    )
    parser.add_argument(
        "--iterations", type=int, default=0,
        help="Max iterations, 0=infinite (default: 0)",
    )
    parser.add_argument(
        "--network", type=str, default="full",
        choices=["tiny", "small", "medium", "full"],
        help="Network preset (default: full)",
    )
    parser.add_argument(
        "--gpus", type=int, default=1,
        help="Number of GPUs for training (default: 1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without executing",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Build config
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
        # CLI args override YAML values
        if args.project_dir != ".":
            config.project_dir = args.project_dir
        if args.data_dir != "data":
            config.data_dir = args.data_dir
        if args.weights_dir != "weights":
            config.weights_dir = args.weights_dir
        if args.checkpoint_dir != "checkpoints":
            config.checkpoint_dir = args.checkpoint_dir
        if args.iterations != 0:
            config.max_iterations = args.iterations
        if args.network != "full":
            config.train_network = args.network
        if args.gpus != 1:
            config.train_gpus = args.gpus
    else:
        config = PipelineConfig(
            project_dir=args.project_dir,
            data_dir=args.data_dir,
            weights_dir=args.weights_dir,
            checkpoint_dir=args.checkpoint_dir,
            max_iterations=args.iterations,
            train_network=args.network,
            train_gpus=args.gpus,
        )

    config.dry_run = args.dry_run

    # Run the pipeline
    coordinator = Coordinator(config)
    coordinator.run()


if __name__ == "__main__":
    main()
