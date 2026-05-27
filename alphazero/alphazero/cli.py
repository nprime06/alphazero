"""AlphaZero Chess CLI.

Unified command-line interface for the AlphaZero chess project.  Each
subcommand is a thin wrapper that delegates to the appropriate module
(training, orchestrator, neural, etc.).

Usage::

    alphazero train --dummy-data --network tiny --steps 100
    alphazero evaluate --model-a best.pt --model-b prev.pt --num-games 20
    alphazero pipeline --config pipeline.yaml --iterations 5
    alphazero export --checkpoint ckpt.pt --output model.pt
    alphazero self-play --model model.pt --games 100 --output ./data
    alphazero play
    alphazero analyze --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

logger = logging.getLogger("alphazero")


# ============================================================================
# Runtime helpers
# ============================================================================


def _project_root() -> Path:
    """Return the repository root for an editable checkout."""
    return Path(__file__).resolve().parents[2]


def _release_binary(name: str) -> Path:
    """Return the expected path to a Rust release binary."""
    return _project_root() / "target" / "release" / name


def _prepend_path(existing: str | None, path: Path) -> str:
    """Prepend a directory to a path-like environment variable."""
    path_str = str(path)
    parts = [p for p in (existing or "").split(os.pathsep) if p]
    parts = [p for p in parts if p != path_str]
    return os.pathsep.join([path_str, *parts])


def _find_torch_lib_dir() -> Path | None:
    """Locate the active Python environment's torch/lib directory."""
    try:
        import torch
    except ImportError:
        return None

    torch_paths = getattr(torch, "__path__", None)
    if torch_paths:
        lib_dir = Path(next(iter(torch_paths))) / "lib"
    else:
        torch_file = getattr(torch, "__file__", None)
        if torch_file is None:
            return None
        lib_dir = Path(torch_file).resolve().parent / "lib"

    return lib_dir if lib_dir.is_dir() else None


def _env_with_torch_libs(
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment with torch/lib first in dylib search paths."""
    env = dict(os.environ if base_env is None else base_env)
    torch_lib = _find_torch_lib_dir()
    if torch_lib is None:
        return env

    for key in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        env[key] = _prepend_path(env.get(key), torch_lib)
    return env


def _install_torch_lib_env() -> None:
    """Apply torch/lib search paths to the current Python process env."""
    os.environ.update(_env_with_torch_libs())


def _format_command(cmd: list[str]) -> str:
    """Format a command for display without shell-escaping requirements."""
    return " ".join(cmd)


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
        "--output", "--output-dir", dest="output", type=str, default="./games",
        help="Directory to write game files (default: ./games)",
    )
    p.add_argument(
        "--games", "--num-games", dest="games", type=int, default=100,
        help="Number of games to generate (default: 100)",
    )
    p.add_argument(
        "--sims", "--simulations", dest="sims", type=int, default=800,
        help="MCTS simulations per move (default: 800)",
    )
    p.add_argument(
        "--threads", type=int, default=1,
        help="Search threads per game (default: 1)",
    )
    p.add_argument(
        "--parallel-games", type=int, default=1,
        help="Number of games to run concurrently (default: 1)",
    )
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Neural network inference batch size (default: 8)",
    )
    p.add_argument(
        "--max-moves", type=int, default=512,
        help="Max moves per game before a forced draw (default: 512)",
    )
    p.add_argument(
        "--no-noise", action="store_true",
        help="Disable Dirichlet root noise",
    )
    p.add_argument(
        "--c-puct", type=float, default=2.5,
        help="MCTS exploration constant (default: 2.5)",
    )
    p.set_defaults(func=_cmd_selfplay)


def _cmd_selfplay(args: argparse.Namespace) -> None:
    """Execute the ``self-play`` subcommand."""
    binary = _release_binary("self-play")
    if not binary.exists():
        print(
            f"Error: self-play binary not found at {binary}\n"
            "Build it first with: cargo build --release -p self-play",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [
        str(binary),
        "--model", args.model,
        "--games", str(args.games),
        "--output", args.output,
        "--sims", str(args.sims),
        "--threads", str(args.threads),
        "--parallel-games", str(args.parallel_games),
        "--batch-size", str(args.batch_size),
        "--max-moves", str(args.max_moves),
    ]
    if args.no_noise:
        cmd.append("--no-noise")
    cmd.extend(["--c-puct", str(args.c_puct)])

    print(f"Running self-play: {_format_command(cmd)}")
    result = subprocess.run(cmd, env=_env_with_torch_libs())
    if result.returncode != 0:
        sys.exit(result.returncode)


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
    p.add_argument(
        "--backend", type=str, default="auto",
        choices=["auto", "python", "rust"],
        help="Evaluation backend (default: auto)",
    )
    p.add_argument(
        "--max-moves", type=int, default=512,
        help="Max half-moves per evaluation game (default: 512)",
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
        backend=args.backend,
        max_moves=args.max_moves,
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
    unsupported = []
    if args.model is not None:
        unsupported.append("--model")
    if args.simulations != 800:
        unsupported.append("--simulations")
    if args.device != "cpu":
        unsupported.append("--device")
    if args.color != "white":
        unsupported.append("--color")

    if unsupported:
        print(
            "Error: model-vs-human play is not wired into the Python CLI yet. "
            "Unsupported options in this focused pass: "
            f"{', '.join(unsupported)}.",
            file=sys.stderr,
        )
        print(
            "Use plain `alphazero play` to launch the Rust interactive board.",
            file=sys.stderr,
        )
        sys.exit(2)

    binary = _release_binary("play")
    if not binary.exists():
        print(
            f"Error: interactive play binary not found at {binary}\n"
            "Build it first with: cargo build --release -p chess-engine --bin play",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [str(binary)]
    print(f"Launching interactive board: {_format_command(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


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
    _install_torch_lib_env()
    try:
        import alphazero_py
    except ImportError as exc:
        print(
            f"Error: alphazero_py is not available: {exc}\n"
            "Build/install it with: "
            "maturin develop --manifest-path alphazero-py/Cargo.toml",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        board = alphazero_py.Board.from_fen(args.fen)
    except Exception as exc:
        print(f"Error: invalid FEN: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.model:
            result = alphazero_py.search_with_model(
                board,
                args.model,
                args.simulations,
                1.0,
                2.5,
                args.device,
            )
        else:
            result = alphazero_py.search_uniform(
                board,
                args.simulations,
                1.0,
                2.5,
            )
    except Exception as exc:
        print(f"Error: search failed: {exc}", file=sys.stderr)
        sys.exit(1)

    moves = sorted(
        list(getattr(result, "moves", [])),
        key=lambda item: item[1],
        reverse=True,
    )
    best_move = getattr(result, "best_move", None)
    if best_move is None and moves:
        best_move = moves[0][0]
    root_value = getattr(result, "root_value", None)
    total_simulations = getattr(result, "total_simulations", args.simulations)

    print(f"Position: {args.fen}")
    print(f"Model: {args.model if args.model else 'uniform evaluator'}")
    print(f"Simulations: {total_simulations}")
    print(f"Device: {args.device}")
    print(f"Best move: {best_move if best_move else '(none)'}")
    if isinstance(root_value, (float, int)):
        print(f"Root value: {root_value:.4f}")
    else:
        print(f"Root value: {root_value}")
    print("Top moves:")
    for move, visits in moves[:10]:
        print(f"  {move}: {visits} visits")


# ============================================================================
# Subcommand: pipeline
# ============================================================================


def _add_pipeline_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``pipeline`` subcommand."""
    p = subparsers.add_parser("pipeline", help="Run the full training pipeline")
    p.add_argument(
        "--config", type=str,
        help="Path to pipeline YAML config",
    )
    p.add_argument(
        "--run-dir", type=str, default=None,
        help="Run directory. Resumes if it exists, creates if new.",
    )
    p.add_argument(
        "--project-dir", type=str, default=None,
        help="Root project directory (default: from config or current checkout)",
    )
    p.add_argument(
        "--iterations", type=int, default=None,
        help="Max iterations, 0=infinite (default: from config or 0)",
    )
    p.add_argument(
        "--network", type=str, default=None,
        choices=["tiny", "small", "medium", "full"],
        help="Network preset (default: from config or full)",
    )
    p.add_argument(
        "--gpus", type=int, default=None,
        help="Number of GPUs for training (default: from config or 1)",
    )
    p.add_argument(
        "--eval-backend", type=str, default=None,
        choices=["auto", "python", "rust"],
        help="Evaluation backend (default: from config or auto)",
    )
    p.add_argument(
        "--eval-max-moves", type=int, default=None,
        help="Max half-moves per evaluation game (default: from config or 512)",
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

    config = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()

    if args.project_dir is not None:
        config.project_dir = args.project_dir
    elif args.config is None:
        config.project_dir = str(_project_root())
    if args.iterations is not None:
        config.max_iterations = args.iterations
    if args.network is not None:
        config.train_network = args.network
    if args.gpus is not None:
        config.train_gpus = args.gpus
    if args.eval_backend is not None:
        config.eval_backend = args.eval_backend
    if args.eval_max_moves is not None:
        config.eval_max_moves = args.eval_max_moves
    if args.dry_run:
        config.dry_run = True
        print("Dry-run mode enabled.")

    if args.run_dir is not None:
        config.run_dir = args.run_dir
    elif config.run_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.run_dir = str(
            Path(config.project_dir) / "runs" / f"coord_{timestamp}"
        )

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
