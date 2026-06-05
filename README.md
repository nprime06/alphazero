# AlphaZero Chess

A from-scratch implementation of [DeepMind's AlphaZero](https://arxiv.org/abs/1712.01815) for chess. The system learns to play chess entirely through self-play reinforcement learning, with no human game data or hand-crafted evaluation functions.

**Rust** handles the performance-critical components (chess engine, MCTS, self-play), while **Python/PyTorch** handles neural network training and orchestration. The two interoperate via TorchScript models and MessagePack data files.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Pipeline Coordinator                       │
│                    (Slurm Job Management)                        │
└─────────────────────────────────────────────────────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  Self-Play    │    │  Training Loop    │    │   Evaluation      │
│  Workers      │    │  (Single-process) │    │    Workers        │
│  (Rust)       │    │  (PyTorch)        │    │  (Python/Rust MCTS)│
└───────────────┘    └───────────────────┘    └───────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Shared Components                         │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐ │
│  │Chess Engine │  │    MCTS     │  │   Neural Network         │ │
│  │  (Rust)     │  │   (Rust)    │  │  (PyTorch + TorchScript) │ │
│  └─────────────┘  └─────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

The training loop is:
1. **Self-play** workers generate games using MCTS + the current neural network
2. **Training** updates the neural network on the generated game data
3. **Evaluation** plays the new network against the previous best
4. If the new network wins, it becomes the new best; repeat from step 1

## Project Structure

```
alphazero/
├── Cargo.toml                 # Rust workspace root
├── README.md
├── PLAN.md                    # Detailed implementation plan
│
├── chess-engine/              # Phase 1: Bitboard chess engine (Rust)
├── neural/                    # Phase 2: Neural network (Python/PyTorch)
├── mcts/                      # Phase 3: Monte Carlo Tree Search (Rust)
├── self-play/                 # Phase 4: Self-play data generation (Rust)
├── training/                  # Phase 5: Training loop and DDP helpers (Python)
├── orchestrator/              # Phase 6: Pipeline coordination (Python)
├── alphazero-py/              # Phase 7a: PyO3 Python bindings (Rust)
└── alphazero/                 # Phase 7b: Unified CLI (Python)
```

---

## `chess-engine/` — Bitboard Chess Engine

A fast legal move generator using magic bitboards. Handles all chess rules including castling, en passant, promotions, pins, check evasion, and draw detection.

```
chess-engine/
├── Cargo.toml
└── src/
    ├── lib.rs                 # Crate root, re-exports all modules
    ├── types.rs               # Core types: Square (0-63), Piece (P/N/B/R/Q/K), Color (W/B)
    ├── bitboard.rs            # 64-bit bitboard type with set/clear/iterate operations
    ├── board.rs               # Board state: 12 piece bitboards, castling rights, en passant, clocks
    ├── fen.rs                 # FEN string parsing and generation
    ├── magic.rs               # Magic bitboard tables for sliding piece attack generation
    ├── attacks.rs             # Attack/check detection for all piece types
    ├── movegen.rs             # Legal move generation with pin-aware filtering
    ├── moves.rs               # Move representation: from/to squares, promotion, flags
    ├── makemove.rs            # Apply/undo moves on the board (with UndoInfo for reversal)
    ├── zobrist.rs             # Zobrist hashing for fast position comparison
    ├── game.rs                # Game wrapper: tracks position history, detects checkmate/stalemate/draws
    ├── perft.rs               # Perft testing (counts leaf nodes at depth N for correctness validation)
    └── bin/
        └── play.rs            # Interactive CLI: play chess with SAN notation (e.g. "e4", "Nf3", "O-O")
```

**492 tests** covering every rule: castling through check, en passant edge cases, pin-aware move generation, perft suite validation against known counts.

---

## `neural/` — Neural Network

The AlphaZero dual-headed neural network: a shared ResNet trunk feeding into a policy head (move probabilities) and a value head (position evaluation).

```
neural/
├── pyproject.toml
├── neural/
│   ├── __init__.py
│   ├── config.py              # NetworkConfig: presets (tiny/small/medium/full) for depth & width
│   ├── blocks.py              # Residual block: Conv → BatchNorm → ReLU → Conv → BatchNorm + skip
│   ├── network.py             # AlphaZeroNetwork: input conv → N residual blocks → policy + value heads
│   ├── encoding.py            # Board → 119×8×8 tensor (piece planes, history, castling, clocks)
│   ├── moves.py               # Move ↔ policy index mapping (4672 possible moves per position)
│   ├── losses.py              # AlphaZeroLoss: cross-entropy (policy) + MSE (value)
│   └── export.py              # TorchScript export for use in Rust inference
└── tests/
    ├── test_blocks.py
    ├── test_config.py
    ├── test_encoding.py       # Cross-validated against Rust encoding in mcts/src/nn.rs
    ├── test_export.py
    ├── test_losses.py
    ├── test_moves.py          # Cross-validated against Rust move encoding
    └── test_network.py
```

**Network sizes:**
| Preset | Blocks | Filters | Parameters |
|--------|--------|---------|------------|
| tiny   | 5      | 64      | ~1.1M      |
| small  | 10     | 128     | ~3.7M      |
| medium | 15     | 192     | ~10.8M     |
| full   | 19     | 256     | ~23.3M     |

**Input encoding** (119 planes of 8×8):
- 112 history planes: 8 time steps × 14 planes (6 own pieces + 6 opponent pieces + 2 repetition counts)
- 7 auxiliary planes: side to move, move count, 4 castling rights, halfmove clock

Current self-play replay files store the current FEN plus up to seven previous
FENs, most recent first. Training reconstructs history and repetition planes
from those FENs. Legacy v1 replay files without history remain readable and
encode with empty history planes.

**Policy output**: 4672 logits = 8×8 source squares × 73 move types (56 queen-type + 8 knight + 9 underpromotions)

**358 tests**, 1 skipped.

---

## `mcts/` — Monte Carlo Tree Search

AlphaZero-style PUCT search with neural network evaluation, Dirichlet noise for exploration, and batched GPU inference.

```
mcts/
├── Cargo.toml
└── src/
    ├── lib.rs                 # Crate root
    ├── config.rs              # MctsConfig: c_puct, simulations, Dirichlet params, temperature
    ├── node.rs                # Tree node: visit count, total value, prior, sibling-linked children
    ├── arena.rs               # Arena allocator for cache-friendly contiguous node storage
    ├── select.rs              # PUCT child selection: Q + c * P * sqrt(N_parent) / (1 + N_child)
    ├── expand.rs              # Leaf expansion: create child nodes from legal moves with policy priors
    ├── backup.rs              # Value backup: propagate leaf values up the path, negating at each level
    ├── search.rs              # Complete search loop: select → expand → evaluate → backup
    ├── nn.rs                  # TorchScript model loading, board encoding, move encoding (matches Python)
    ├── batch.rs               # InferenceServer: async batched NN evaluation across worker threads
    ├── reuse.rs               # Tree reuse helper, implemented but not wired into self-play
    └── transposition.rs       # Transposition table helper, implemented but not wired into search
```

**Key design decisions:**
- Nodes are ~28 bytes each (two fit per cache line)
- Index-based references (`NodeIndex = u32`) instead of pointers — avoids lifetime issues during mutations
- Children stored as sibling-linked lists (4 bytes per node vs 24 bytes for `Vec`)
- Batched inference collects multiple leaf positions for a single GPU forward pass

**168 tests.**

---

## `self-play/` — Self-Play Data Generation

Rust binary that plays games against itself using MCTS, producing training data for the neural network.

```
self-play/
├── Cargo.toml
└── src/
    ├── lib.rs                 # Crate root
    ├── main.rs                # CLI binary: --model, --games, --output, --sims, --threads
    ├── game.rs                # Game loop: MCTS search → select move → record position → repeat
    ├── data.rs                # GameRecord → TrainingSample conversion (assigns game outcomes)
    ├── buffer.rs              # Disk-based replay buffer with capacity limits and oldest-first eviction
    └── serialize.rs           # MessagePack serialization: FEN history + sparse policy + value per sample
```

**Data format**: Each game file is a MessagePack blob containing:
- Header: format version, sample count
- Samples: FEN string, optional `history_fens`, sparse policy `[(index, probability), ...]`, value `{-1, 0, +1}`

**65 tests.**

---

## `training/` — Training Pipeline

PyTorch training loop with mixed precision, checkpointing, Slurm scripts, and
DDP utility scaffolding. The main `training.train` entrypoint is currently a
single-process trainer; the DDP helpers and `torchrun` script are not fully
wired into the training loop yet.

```
training/
├── pyproject.toml
├── training/
│   ├── __init__.py
│   ├── train.py               # TrainConfig + Trainer: SGD with momentum, MultiStepLR schedule
│   ├── dataloader.py          # ReplayDataset (live buffer sampling) + DummyDataset (testing)
│   ├── buffer.py              # Python replay buffer reader: loads .msgpack files from self-play
│   ├── checkpoint.py          # CheckpointManager: atomic saves, keep-N cleanup, auto TorchScript export
│   ├── distributed.py         # DDP helpers: setup/cleanup, rank detection, model wrapping
│   └── metrics.py             # MetricsLogger: TensorBoard + optional Weights & Biases
├── scripts/
│   ├── train.sh               # Slurm job script: torchrun with NCCL backend
│   ├── submit_train.sh        # Submit wrapper: configures GPUs, memory, wall time
│   ├── selfplay.sh            # Slurm job script for self-play workers
│   └── submit_selfplay.sh     # Submit wrapper for self-play
└── tests/
    ├── test_train.py          # Training loop, loss decrease, LR schedule, torch.compile, AMP
    ├── test_dataloader.py     # Dataset shapes, dtypes, DataLoader batching
    ├── test_buffer.py         # MessagePack reading, cross-language compat, sampling
    ├── test_checkpoint.py     # Save/load, resume, atomic writes, TorchScript export
    ├── test_distributed.py    # DDP helpers in single-process mode
    └── test_metrics.py        # TensorBoard logging, throughput, policy accuracy
```

**Training hyperparameters** (matching AlphaZero paper):
- Batch size: 4096
- Optimizer: SGD (momentum=0.9, weight_decay=1e-4)
- Learning rate: 0.2 → 0.02 → 0.002 (step decay at 100K, 300K, 500K)
- Mixed precision (AMP) on CUDA

**105 tests.**

---

## `orchestrator/` — Pipeline Coordination

Manages the full training pipeline: weight distribution, model evaluation, and the self-play → train → evaluate loop.
Evaluation supports `backend="auto"`, `backend="python"`, and
`backend="rust"`. The Rust backend calls `alphazero_py.search_with_model`
when the PyO3 extension is installed and evaluation simulations are positive.

```
orchestrator/
├── pyproject.toml
├── orchestrator/
│   ├── __init__.py
│   ├── weights.py             # WeightPublisher: version + export TorchScript; WeightWatcher: detect updates
│   ├── evaluate.py            # Model evaluation: Python or Rust/PyO3 MCTS backends
│   ├── coordinator.py         # PipelineCoordinator: YAML config, persistent state, iteration loop
│   ├── doctor.py              # Run integrity checks before cluster resume
│   └── config.yaml            # Example pipeline configuration
└── tests/
    ├── test_weights.py        # Publish/watch cycle, versioning, cleanup
    ├── test_evaluate.py       # ELO calculation, game playing, move conversion
    ├── test_coordinator.py    # Config loading, state persistence, dry run, iteration control
    └── test_doctor.py         # Run-resume integrity and promotion-ledger checks
```

**80 tests.**

---

## `alphazero-py/` — Python Bindings (PyO3)

Native Python extension exposing the Rust chess engine and MCTS to Python.

```
alphazero-py/
├── Cargo.toml
└── src/
    └── lib.rs                 # PyO3 bindings: Board, SearchResult, search functions
```

**Python API:**
```python
import alphazero_py

# Chess board (python-chess-like API)
board = alphazero_py.Board()                # Starting position
board = alphazero_py.Board.from_fen(fen)    # From FEN
board.legal_moves()                         # ["e2e4", "d2d4", ...]
board.push("e2e4")                          # Make move (UCI notation)
board.pop()                                 # Undo last move
board.is_game_over()                        # True if checkmate/stalemate/draw
board.fen                                   # Current FEN string
board.turn                                  # "white" or "black"
board.is_check()                            # True if side to move is in check

# MCTS search (uniform evaluator)
result = alphazero_py.search_uniform(board, num_simulations=800)

# MCTS search (neural network)
result = alphazero_py.search_with_model(board, "model.pt", num_simulations=800, device="cuda")

result.best_move       # "e2e4"
result.moves           # [("e2e4", 342), ("d2d4", 215), ...]
result.root_value      # 0.15
```

---

## `alphazero/` — Unified CLI

Command-line interface wrapping all components.

```
alphazero/
├── pyproject.toml
├── alphazero/
│   ├── __init__.py
│   └── cli.py                 # Subcommands: train, self-play, evaluate, play, analyze, pipeline, export
└── tests/
    └── test_cli.py            # Argument parsing, wrappers, help text
```

**37 tests.**

---

## Usage

### Prerequisites

- **Rust** (stable toolchain)
- **Python 3.10+** with PyTorch 2.0+
- **NVIDIA GPU** with CUDA (for training; CPU works for testing)

### Setup

```bash
# Clone and enter project
cd alphazero

# Create Python venv and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install torch tensorboard pyyaml python-chess msgpack

# Install Python packages in development mode
pip install -e neural/
pip install -e training/
pip install -e orchestrator/
pip install -e alphazero/

# Set up Rust build/test environment for tch-rs
export LIBTORCH_USE_PYTORCH=1
export TORCH_LIB=$(python -c "import torch; print(torch.__path__[0] + '/lib')")
export DYLD_LIBRARY_PATH=$TORCH_LIB:${DYLD_LIBRARY_PATH:-}  # macOS
export LD_LIBRARY_PATH=$TORCH_LIB:${LD_LIBRARY_PATH:-}      # Linux

# Build Rust crates
cargo build --release

# Optional: install the PyO3 bindings used by `alphazero analyze`
pip install maturin
maturin develop --manifest-path alphazero-py/Cargo.toml
```

### Play Chess Interactively

```bash
alphazero play
```

Uses standard algebraic notation:
```
=== AlphaZero Chess Engine — Interactive Play ===
Type moves in SAN (e.g. e4, Nf3, O-O) or a command:
  moves, fen, fen <FEN>, board, undo, new, quit

  1. e4
  1... e5
  2. Nf3
```

### Run Self-Play

```bash
alphazero self-play \
    --model model.pt \
    --games 100 \
    --output ./data/ \
    --sims 800
```

### Train the Network

Training is intended to run on the compute cluster, not on a laptop or local
desktop. Local training commands are only for tiny smoke tests that verify the
pipeline can import packages, decode replay, and write checkpoints.

```bash
# Local smoke only
python -m training.train --dummy-data --network tiny --steps 100
```

### Evaluate Two Models

```bash
python -c "
from orchestrator.evaluate import evaluate_models
result = evaluate_models('model_a.pt', 'model_b.pt', num_games=100)
print(result.summary())
"

# Rust-backed MCTS evaluation when alphazero_py is installed
alphazero evaluate --model-a model_a.pt --model-b model_b.pt --simulations 100 --backend rust
```

### Run on Slurm Cluster

The Slurm wrappers are the supported entry point for actual training work. They
derive the project root from the checked-out repo path, install the local Python
packages into the active conda environment, and set the PyTorch lib path needed
by Rust/tch binaries.

Before submitting jobs on a fresh cluster checkout:

```bash
# One-time cluster build/setup
module load miniforge
conda create -y -p .conda/env python=3.13
conda activate .conda/env
python -m pip install -r requirements.txt
python -m pip install -e neural -e training -e orchestrator -e alphazero

export LIBTORCH_USE_PYTORCH=1
export TORCH_LIB=$(python -c "from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / 'lib')")
export LD_LIBRARY_PATH=$TORCH_LIB:${LD_LIBRARY_PATH:-}

cargo build --release -p self-play
python -m pip install maturin
python -m maturin develop --manifest-path alphazero-py/Cargo.toml
```

Submit cluster jobs through the wrappers:

```bash
# Tiny one-GPU cluster smoke; writes a self-contained run under runs/
bash training/scripts/submit_cluster_smoke.sh

# Standalone one-GPU training job. DDP is not wired yet; --gpus must stay 1.
bash training/scripts/submit_train.sh \
    --gpus 1 \
    --data-dir /path/to/replay \
    --network tiny \
    --steps 1000

# Standalone self-play job
bash training/scripts/submit_selfplay.sh \
    --gpus 1 \
    --model /path/to/model.pt \
    --games 1000 \
    --output /path/to/data
```

Before resuming a coordinator run, inspect its artifact/state consistency:

```bash
alphazero doctor --run-dir runs/coord_YYYYMMDD_HHMMSS
```

The doctor fails nonzero for missing best/latest weights, unprovable
best-model lineage, malformed promotion ledgers, or mismatches between the
promotion ledger and `pipeline_state.yaml`.
`submit_coordinator.sh --run-dir ...` runs this check inside the Slurm
allocation before the coordinator starts. Use `--skip-resume-doctor` only for
manual recovery work where you intentionally need to inspect or repair an
unsafe run without continuing it.
Coordinator submissions also write `runs/slurm_setup/latest_orcd_jobs.txt`
with the Slurm job id, run directory, source commit, requested resources, and
previous submitted coordinator job so monitors can recover state without
scraping terminal output.

Current cluster limitations:
- `training.train` is single-process; DDP helpers exist, but Slurm wrappers
  intentionally reject `--gpus > 1`.
- The coordinator currently runs as one Slurm job that invokes local self-play,
  training, and evaluation phases. It does not yet submit separate job arrays
  for self-play workers or distributed training.
- Serious runs should use fresh replay generated by the current v2 serializer;
  legacy v1 replay remains readable but has empty history planes.

Local `alphazero self-play`, `alphazero play`, and coordinator self-play runs
expect the relevant release binaries to already exist. Build them with:

```bash
cargo build --release -p self-play
cargo build --release -p chess-engine --bin play
```

`alphazero analyze --fen ... [--model model.pt]` uses the `alphazero_py`
bindings when installed. Without `--model`, it runs Rust MCTS with a uniform
evaluator.

### Export a Checkpoint to TorchScript

```bash
alphazero export --checkpoint checkpoints/checkpoint_step_0001000.pt --output model.pt --network full
```

### Run Tests

```bash
# Rust tests (chess engine)
cargo test -p chess-engine

# Rust tests that use tch/libtorch
LIBTORCH_USE_PYTORCH=1 DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/torch/lib" cargo test -p mcts
LIBTORCH_USE_PYTORCH=1 DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/torch/lib" cargo test -p self-play
LIBTORCH_USE_PYTORCH=1 DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/torch/lib" cargo test -p alphazero-py

# Python tests: run package suites separately to avoid tests/ import collisions
.venv/bin/python -m pytest neural/tests -q
.venv/bin/python -m pytest training/tests -q
.venv/bin/python -m pytest orchestrator/tests -q
.venv/bin/python -m pytest alphazero/tests -q
```

On Linux, use `LD_LIBRARY_PATH` instead of or in addition to
`DYLD_LIBRARY_PATH` for the Rust `tch` tests and runtime commands.

## Test Summary

| Package | Tests | Notes |
|---------|-------|-------|
| chess-engine | 484 passed, 8 ignored | Perft-validated against Stockfish |
| neural | 358 passed, 1 skipped | Cross-validated Rust ↔ Python encoding |
| mcts | 168 passed | Requires libtorch at runtime |
| self-play | 65 passed | Requires libtorch at runtime |
| training | 105 passed | CPU-only tests with tiny network |
| orchestrator | 80 passed | Python and Rust/PyO3 evaluation paths |
| alphazero (CLI) | 37 passed | Argument parsing, wrappers, and help text |
| **Total** | **~1,297** | |

## Hardware Target

Designed for Slurm clusters with NVIDIA H200 GPUs (141GB HBM3). The large GPU memory enables:
- Batch sizes of 4096+ on a single GPU
- Concurrent self-play and training on separate GPUs
- Mixed precision training for ~2x speedup

## References

- Silver, D. et al. (2018). [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://doi.org/10.1126/science.aar6404). *Science*, 362(6419).
- Silver, D. et al. (2017). [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815). *arXiv*.
- He, K. et al. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). *arXiv*.
