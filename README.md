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
│  Workers      │    │  (Distributed)    │    │    Workers        │
│  (Rust)       │    │  (PyTorch DDP)    │    │  (Python)         │
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
├── training/                  # Phase 5: Distributed training (Python)
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
| tiny   | 5      | 64      | ~600K      |
| small  | 10     | 128     | ~5M        |
| medium | 15     | 192     | ~16M       |
| full   | 19     | 256     | ~40M       |

**Input encoding** (119 planes of 8×8):
- 112 history planes: 8 time steps × 14 planes (6 own pieces + 6 opponent pieces + 2 repetition counts)
- 7 auxiliary planes: side to move, move count, 4 castling rights, halfmove clock

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
    ├── reuse.rs               # Tree reuse: carry over subtree after making a move
    └── transposition.rs       # Transposition table: cache (policy, value) by Zobrist hash
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
    └── serialize.rs           # MessagePack serialization: FEN + sparse policy + value per sample
```

**Data format**: Each game file is a MessagePack blob containing:
- Header: format version, sample count
- Samples: FEN string, sparse policy `[(index, probability), ...]`, value `{-1, 0, +1}`

**62 tests.**

---

## `training/` — Distributed Training Pipeline

PyTorch training loop with DDP multi-GPU support, mixed precision, checkpointing, and Slurm integration.

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

**98 tests.**

---

## `orchestrator/` — Pipeline Coordination

Manages the full training pipeline: weight distribution, model evaluation, and the self-play → train → evaluate loop.

```
orchestrator/
├── pyproject.toml
├── orchestrator/
│   ├── __init__.py
│   ├── weights.py             # WeightPublisher: version + export TorchScript; WeightWatcher: detect updates
│   ├── evaluate.py            # Model evaluation: play matches, compute ELO differences
│   ├── coordinator.py         # PipelineCoordinator: YAML config, persistent state, iteration loop
│   └── config.yaml            # Example pipeline configuration
└── tests/
    ├── test_weights.py        # Publish/watch cycle, versioning, cleanup
    ├── test_evaluate.py       # ELO calculation, game playing, move conversion
    └── test_coordinator.py    # Config loading, state persistence, dry run, iteration control
```

**74 tests.**

---

## `alphazero-py/` — Python Bindings (PyO3)

Native Python extension exposing the Rust chess engine and MCTS to Python.

```
alphazero-py/
├── Cargo.toml
└── src/
    └── lib.rs                 # PyO3 bindings: PyBoard, PySearchResult, search functions
```

**Python API:**
```python
import alphazero_py

# Chess board (python-chess-like API)
board = alphazero_py.PyBoard()              # Starting position
board = alphazero_py.PyBoard.from_fen(fen)  # From FEN
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
    └── test_cli.py            # Argument parsing, help text, placeholder output
```

**31 tests.**

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

# Set up Rust environment for tch-rs
export LIBTORCH=$(python -c "import torch; print(torch.__path__[0])")
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# Build Rust crates
cargo build --release
```

### Play Chess Interactively

```bash
cargo run -p chess-engine --bin play --release
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
cargo run -p self-play --release -- \
    --model model.pt \
    --games 100 \
    --output ./data/ \
    --sims 800
```

### Train the Network

```bash
# With real self-play data
python -m training.train --data-dir ./data/ --network full --steps 700000

# Quick test with random data
python -m training.train --dummy-data --network tiny --steps 100
```

### Evaluate Two Models

```bash
python -c "
from orchestrator.evaluate import evaluate_models
result = evaluate_models('model_a.pt', 'model_b.pt', num_games=100)
print(result.summary())
"
```

### Run on Slurm Cluster

```bash
# Submit a training job (1 GPU, 12 hours)
bash training/scripts/submit_train.sh --gpus 1 --dummy-data --network tiny --steps 1000

# Submit a self-play job
bash training/scripts/submit_selfplay.sh --gpus 1

# Run the full pipeline
alphazero pipeline --config orchestrator/orchestrator/config.yaml
```

### Export a Checkpoint to TorchScript

```bash
alphazero export --checkpoint checkpoints/checkpoint_step_0001000.pt --output model.pt --network full
```

### Run Tests

```bash
# Rust tests (chess engine)
cargo test -p chess-engine

# All Python tests
PYTHONPATH=neural:training:orchestrator python -m pytest neural/tests/ training/tests/ orchestrator/tests/ alphazero/tests/ -v

# Full Rust workspace (requires CUDA for mcts/self-play tests)
cargo test --workspace
```

## Test Summary

| Package | Tests | Notes |
|---------|-------|-------|
| chess-engine | 484 passed, 8 ignored | Perft-validated against Stockfish |
| neural | 358 passed, 1 skipped | Cross-validated Rust ↔ Python encoding |
| mcts | 168 passed | Requires libtorch at runtime |
| self-play | 62 passed | Requires libtorch at runtime |
| training | 98 passed | CPU-only tests with tiny network |
| orchestrator | 74 passed | Pure Python MCTS for evaluation |
| alphazero (CLI) | 31 passed | Argument parsing and help text |
| **Total** | **~1,275** | |

## Hardware Target

Designed for Slurm clusters with NVIDIA H200 GPUs (141GB HBM3). The large GPU memory enables:
- Batch sizes of 4096+ on a single GPU
- Concurrent self-play and training on separate GPUs
- Mixed precision training for ~2x speedup

## References

- Silver, D. et al. (2018). [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://doi.org/10.1126/science.aar6404). *Science*, 362(6419).
- Silver, D. et al. (2017). [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815). *arXiv*.
- He, K. et al. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). *arXiv*.
