"""AlphaZero Chess CLI.

Unified command-line interface for the AlphaZero chess project.  Each
subcommand is a thin wrapper that delegates to the appropriate module
(training, orchestrator, neural, etc.).

Usage::

    alphazero train --dummy-data --network tiny --steps 100
    alphazero evaluate --model-a best.pt --model-b prev.pt --num-games 20
    alphazero pipeline --config pipeline.yaml
    alphazero export --checkpoint ckpt.pt --output model.pt
    alphazero self-play --model model.pt --num-games 100
    alphazero play --model model.pt --color white
    alphazero analyze --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("alphazero")


# ============================================================================
# Subcommand: train
# ============================================================================


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``train`` subcommand."""
    p = subparsers.add_parser("train", help="Launch training")
    p.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory containing .msgpack game files (default: ./data)",
    )
    p.add_argument(
        "--batch-size", type=int, default=4096,
        help="Batch size (default: 4096)",
    )
    p.add_argument(
        "--steps", type=int, default=700000,
        help="Total training steps (default: 700000)",
    )
    p.add_argument(
        "--lr", type=float, default=0.2,
        help="Initial learning rate (default: 0.2)",
    )
    p.add_argument(
        "--network", type=str, default="full",
        choices=["tiny", "small", "medium", "full"],
        help="Network preset (default: full)",
    )
    p.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile",
    )
    p.add_argument(
        "--no-amp", action="store_true",
        help="Disable automatic mixed precision (AMP)",
    )
    p.add_argument(
        "--log-dir", type=str, default="./runs",
        help="Directory for TensorBoard logs (default: ./runs)",
    )
    p.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints",
        help="Directory for training checkpoints (default: ./checkpoints)",
    )
    p.add_argument(
        "--checkpoint-interval", type=int, default=1000,
        help="Save checkpoint every N steps (default: 1000)",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    p.add_argument(
        "--dummy-data", action="store_true",
        help="Use random data for testing (no game files needed)",
    )
    p.set_defaults(func=_cmd_train)


def _cmd_train(args: argparse.Namespace) -> None:
    """Execute the ``train`` subcommand."""
    try:
        import torch
        from torch.utils.data import DataLoader

        from training.train import TrainConfig, Trainer
        from training.checkpoint import CheckpointManager
    except ImportError as exc:
        print(
            f"Error: required package not available: {exc}\n"
            "Install with: pip install -e training/ -e neural/",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        total_steps=args.steps,
        learning_rate=args.lr,
        network_config=args.network,
        use_compile=not args.no_compile,
        use_amp=not args.no_amp,
    )

    trainer = Trainer(config, device)

    # Resume from checkpoint if requested
    if args.resume:
        ckpt_mgr = CheckpointManager(args.checkpoint_dir)
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt_mgr.resume(trainer, args.resume)

    if args.dummy_data:
        from training.dataloader import DummyDataset

        dataset = DummyDataset(
            size=max(config.total_steps * config.batch_size, 10_000)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
    else:
        from training.buffer import ReplayBuffer
        from training.dataloader import create_dataloader

        buffer = ReplayBuffer(config.data_dir)
        num_games = buffer.scan()
        print(f"Found {num_games} games in replay buffer")
        dataloader = create_dataloader(
            buffer,
            config.batch_size,
            config.samples_per_epoch,
            0,
        )

    trainer.train(dataloader)


# ============================================================================
# Subcommand: self-play
# ============================================================================


def _add_selfplay_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``self-play`` subcommand."""
    p = subparsers.add_parser("self-play", help="Run self-play workers")
    p.add_argument(
        "--model", type=str, required=True,
        help="Path to TorchScript model",
    )
    p.add_argument(
        "--output-dir", type=str, default="./games",
        help="Directory to write game files (default: ./games)",
    )
    p.add_argument(
        "--num-games", type=int, default=100,
        help="Number of games to generate (default: 100)",
    )
    p.add_argument(
        "--simulations", type=int, default=800,
        help="MCTS simulations per move (default: 800)",
    )
    p.add_argument(
        "--temperature", type=float, default=1.0,
        help="Temperature for move selection (default: 1.0)",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="Device for inference (default: cuda)",
    )
    p.set_defaults(func=_cmd_selfplay)


def _cmd_selfplay(args: argparse.Namespace) -> None:
    """Execute the ``self-play`` subcommand."""
    print("Self-play uses the Rust binary for maximum performance.")
    print()
    print("Run:")
    print(
        "  cargo run -p self-play --release -- "
        f"--model {args.model} "
        f"--output-dir {args.output_dir} "
        f"--num-games {args.num_games} "
        f"--simulations {args.simulations} "
        f"--temperature {args.temperature} "
        f"--device {args.device}"
    )


# ============================================================================
# Subcommand: evaluate
# ============================================================================


def _add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``evaluate`` subcommand."""
    p = subparsers.add_parser("evaluate", help="Evaluate two models")
    p.add_argument(
        "--model-a", type=str, required=True,
        help="Path to first TorchScript model",
    )
    p.add_argument(
        "--model-b", type=str, required=True,
        help="Path to second TorchScript model",
    )
    p.add_argument(
        "--num-games", type=int, default=100,
        help="Number of evaluation games (default: 100)",
    )
    p.add_argument(
        "--simulations", type=int, default=100,
        help="MCTS simulations per move (default: 100)",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Device for inference (default: cpu)",
    )
    p.set_defaults(func=_cmd_evaluate)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Execute the ``evaluate`` subcommand."""
    try:
        from orchestrator.evaluate import evaluate_models
    except ImportError as exc:
        print(
            f"Error: required package not available: {exc}\n"
            "Install with: pip install -e orchestrator/ -e neural/",
            file=sys.stderr,
        )
        sys.exit(1)

    results = evaluate_models(
        model_a_path=args.model_a,
        model_b_path=args.model_b,
        num_games=args.num_games,
        simulations=args.simulations,
        device=args.device,
    )
    print(results.summary())


# ============================================================================
# Subcommand: play
# ============================================================================


def _add_play_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``play`` subcommand."""
    p = subparsers.add_parser("play", help="Interactive play vs engine")
    p.add_argument(
        "--model", type=str, default=None,
        help="Path to TorchScript model (omit for random play)",
    )
    p.add_argument(
        "--simulations", type=int, default=800,
        help="MCTS simulations per move (default: 800)",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Device for inference (default: cpu)",
    )
    p.add_argument(
        "--color", type=str, default="white",
        choices=["white", "black"],
        help="Side for the human player (default: white)",
    )
    p.set_defaults(func=_cmd_play)


def _cmd_play(args: argparse.Namespace) -> None:
    """Execute the ``play`` subcommand."""
    print("Interactive play uses the Rust binary.")
    print()
    print("Run:")
    print(
        "  cargo run -p chess-engine --bin play --release"
    )
    print()
    print(
        "This will be enhanced with a Python interface once the "
        "PyO3 bindings (alphazero-py) are ready."
    )


# ============================================================================
# Subcommand: analyze
# ============================================================================


def _add_analyze_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``analyze`` subcommand."""
    p = subparsers.add_parser("analyze", help="Analyze a position")
    p.add_argument(
        "--fen", type=str, required=True,
        help="FEN string of the position to analyze",
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="Path to TorchScript model",
    )
    p.add_argument(
        "--simulations", type=int, default=800,
        help="MCTS simulations per move (default: 800)",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Device for inference (default: cpu)",
    )
    p.set_defaults(func=_cmd_analyze)


def _cmd_analyze(args: argparse.Namespace) -> None:
    """Execute the ``analyze`` subcommand."""
    print(f"Position: {args.fen}")
    print()
    print(
        "Position analysis requires the PyO3 bindings (alphazero-py) "
        "for Rust MCTS integration."
    )
    print(
        "This feature will be available once the bindings are built "
        "(Phase 7)."
    )
    if args.model:
        print(f"Model: {args.model}")
    print(f"Simulations: {args.simulations}")
    print(f"Device: {args.device}")


# ============================================================================
# Subcommand: pipeline
# ============================================================================


def _add_pipeline_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``pipeline`` subcommand."""
    p = subparsers.add_parser("pipeline", help="Run the full training pipeline")
    p.add_argument(
        "--config", type=str, required=True,
        help="Path to pipeline YAML config",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Log what would be done without executing",
    )
    p.set_defaults(func=_cmd_pipeline)


def _cmd_pipeline(args: argparse.Namespace) -> None:
    """Execute the ``pipeline`` subcommand."""
    try:
        from orchestrator.coordinator import Coordinator, PipelineConfig
    except ImportError as exc:
        print(
            f"Error: required package not available: {exc}\n"
            "Install with: pip install -e orchestrator/ -e neural/",
            file=sys.stderr,
        )
        sys.exit(1)

    config = PipelineConfig.from_yaml(args.config)
    if args.dry_run:
        config.dry_run = True
        print("Dry-run mode enabled.")

    coordinator = Coordinator(config)
    coordinator.run()


# ============================================================================
# Subcommand: export
# ============================================================================


def _add_export_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``export`` subcommand."""
    p = subparsers.add_parser("export", help="Export model to TorchScript")
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to training checkpoint",
    )
    p.add_argument(
        "--output", type=str, required=True,
        help="Output .pt path for the TorchScript model",
    )
    p.add_argument(
        "--network", type=str, default="full",
        choices=["tiny", "small", "medium", "full"],
        help="Network preset (default: full)",
    )
    p.set_defaults(func=_cmd_export)


def _cmd_export(args: argparse.Namespace) -> None:
    """Execute the ``export`` subcommand."""
    try:
        import torch
        from neural.config import NetworkConfig
        from neural.network import AlphaZeroNetwork
        from neural.export import export_torchscript
        from training.checkpoint import CheckpointManager
    except ImportError as exc:
        print(
            f"Error: required package not available: {exc}\n"
            "Install with: pip install -e training/ -e neural/",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load checkpoint
    ckpt_mgr = CheckpointManager(".")  # dir doesn't matter for load()
    checkpoint = ckpt_mgr.load(args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint} (step {checkpoint.get('step', '?')})")

    # Build model from the network preset
    config_map = {
        "tiny": NetworkConfig.tiny,
        "small": NetworkConfig.small,
        "medium": NetworkConfig.medium,
        "full": NetworkConfig.full,
    }
    net_config = config_map[args.network]()
    model = AlphaZeroNetwork.from_config(net_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded weights into {args.network} network")

    # Export
    output_path = export_torchscript(model, args.output)
    print(f"Exported TorchScript model to {output_path}")


# ============================================================================
# Main entry point
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser.

    Separated from :func:`main` so that tests can introspect the parser
    without triggering side effects.
    """
    parser = argparse.ArgumentParser(
        prog="alphazero",
        description="AlphaZero Chess -- unified CLI",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    subparsers = parser.add_subparsers(dest="command")

    _add_train_parser(subparsers)
    _add_selfplay_parser(subparsers)
    _add_evaluate_parser(subparsers)
    _add_play_parser(subparsers)
    _add_analyze_parser(subparsers)
    _add_pipeline_parser(subparsers)
    _add_export_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Parameters
    ----------
    argv:
        Argument list to parse.  Defaults to ``sys.argv[1:]`` when
        *None* (the normal case when invoked from the console script).
        Passing an explicit list is useful for testing.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)
