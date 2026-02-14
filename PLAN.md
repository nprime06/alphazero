# AlphaZero Chess Implementation Plan

## Overview

Recreate AlphaZero for chess with:
- Super-fast chess engine (bitboard-based, Rust)
- Super-fast MCTS with neural network guidance (Rust)
- Distributed training on Slurm clusters (Python/PyTorch)

## Hardware Target

- **GPUs**: 1-2x NVIDIA H200 (141GB HBM3 each)
- **RAM**: ~200GB system memory
- **CPUs**: ~10 cores

## Design Principles

- **Educational first**: Code should be readable and well-documented
- **Clear module boundaries**: Each file does one thing
- **No magic numbers**: Named constants with comments explaining *why*
- **Readable over clever**: Explicit match arms over bit-trick one-liners, unless benchmarks demand it
- **Small functions**: If a section needs a comment, it becomes its own function
- **Configurable**: Network size, search params, training hyperparams all adjustable

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Orchestrator                       │
│                    (Slurm Job Management)                        │
└─────────────────────────────────────────────────────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  Self-Play    │    │  Training Loop    │    │   Evaluation      │
│   Workers     │    │  (Distributed)    │    │    Workers        │
└───────────────┘    └───────────────────┘    └───────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Shared Components                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │Chess Engine │  │    MCTS     │  │   Neural Network        │  │
│  │  (Rust)     │  │   (Rust)    │  │   (PyTorch + TorchScript)│ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Chess Engine | Rust | Speed, safety, no GC pauses |
| MCTS | Rust | Speed, parallel search |
| Neural Network | PyTorch | Ecosystem, flexibility |
| Inference in Rust | tch-rs (libtorch) | Avoid Python overhead in search |
| Python Bindings | PyO3 | Mature, fast |
| Distributed Training | PyTorch DDP | Battle-tested, Slurm-friendly |
| Cluster | Slurm | Standard HPC scheduler |
| Serialization | MessagePack | Fast, compact |

---

## Performance Strategy

### The Bottleneck: Neural Network Inference in MCTS

MCTS speed is gated almost entirely by NN inference. At 800 simulations per
move, every leaf expansion needs a forward pass. The single most impactful
optimization is making inference fast and batching it efficiently.

```
Time per move ≈ 800 × (inference_latency / batch_size) + tree_overhead
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                This dominates everything
```

### Inference Optimization Stack (integrated into plan)

| Approach | Effort | Speedup | Where in Plan |
|----------|--------|---------|---------------|
| FP16 inference | Trivial | ~2x | Chunk 2.7 |
| `torch.compile` | One line | 1.5-3x | Chunk 2.7 |
| Batched async inference | Medium | 5-10x vs serial | Chunk 3.9a |
| TensorRT export | Medium | 2-5x | Chunk 2.8 |
| FP8 inference (H200) | Low | ~3-4x | Chunk 2.8 |

### MCTS Tree Performance Tricks (integrated into plan)

| Trick | Impact | Where in Plan |
|-------|--------|---------------|
| Arena allocation | Avoid heap fragmentation, cache-friendly | Chunk 3.2 |
| Cache-line-aware node layout | Keep hot fields together | Chunk 3.1 |
| Tree reuse between moves | Reuse subtree instead of rebuilding | Chunk 3.10 |
| First Play Urgency (FPU) | Better handling of unvisited nodes | Chunk 3.3 |
| Lock-free atomics | Parallel scaling without mutex overhead | Chunk 3.9b |
| Transposition table | Share evals for same position via different paths | Chunk 3.10 |

### Training Performance (integrated into plan)

| Approach | Effort | Speedup | Where in Plan |
|----------|--------|---------|---------------|
| Mixed precision (AMP) | Low | ~2x | Chunk 5.4 |
| `torch.compile` on training | One line | 1.5-2x | Chunk 5.2 |
| Large batch (H200 VRAM) | Config | Better GPU util | Chunk 5.2 |
| Pinned memory + prefetch | Low | Reduce data stall | Chunk 5.1 |

### Network Size Presets (all fully configurable)

| Preset | Blocks | Filters | Params | Use Case |
|--------|--------|---------|--------|----------|
| Tiny | 5 | 64 | ~200K | Debugging, learning, fast experiments |
| Small | 10 | 128 | ~3M | Overnight training runs |
| Medium | 15 | 192 | ~7M | Multi-day training |
| Full | 19 | 256 | ~12M | The real deal, matches original paper |

All presets use the same code with different config values. Block count, filter
count, policy head size, and value head size are all configurable.

---

# Implementation Chunks

## Phase 1: Chess Engine (Rust)

### Chunk 1.1: Project Setup & Basic Types
**Goal**: Set up Rust workspace with basic type definitions.

**Deliverables**:
- Cargo workspace setup (will hold chess-engine, mcts, self-play crates)
- `Square` enum (A1-H8, 64 squares)
- `Piece` enum (Pawn, Knight, Bishop, Rook, Queen, King)
- `Color` enum (White, Black)
- `Move` struct (from, to, promotion, flags for castling/en passant/etc.)

**Files**:
```
Cargo.toml                # workspace root
chess-engine/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── types.rs          # Square, Piece, Color
    └── moves.rs          # Move struct
```

**Test**: Unit tests for type conversions (square index <-> rank/file).

---

### Chunk 1.2: Bitboard Fundamentals
**Goal**: Implement bitboard type with essential operations.

**Deliverables**:
- `Bitboard` struct (wrapper around u64)
- Bit manipulation: set/clear/test bit, pop_lsb, count
- Display/debug formatting (8x8 grid)
- Precomputed constants: FILES, RANKS, DIAGONALS

**Files**:
```
src/
└── bitboard.rs
```

**Test**:
- Verify bit operations
- Test `pop_lsb` iteration
- Visual verification of constants

---

### Chunk 1.3: Board Representation
**Goal**: Implement the board state structure.

**Deliverables**:
- `Board` struct with 12 piece bitboards
- Occupancy helpers (white_pieces, black_pieces, all_pieces)
- `piece_at(square)` and `put_piece(square, piece, color)`
- Starting position initialization

**Files**:
```
src/
└── board.rs
```

**Test**:
- Initialize starting position
- Verify piece placement
- Test occupancy bitboards

---

### Chunk 1.4: FEN Parsing
**Goal**: Parse and generate FEN strings.

**Deliverables**:
- Parse FEN into Board (piece placement, castling, en passant, clocks)
- Generate FEN from Board
- Standard position constants (starting, common test positions)

**Files**:
```
src/
└── fen.rs
```

**Test**:
- Roundtrip: FEN -> Board -> FEN
- Parse known positions (starting, Sicilian, etc.)
- Error handling for invalid FEN

---

### Chunk 1.5: Non-Sliding Piece Attacks
**Goal**: Precompute knight and king attack tables.

**Deliverables**:
- `KNIGHT_ATTACKS[64]` lookup table
- `KING_ATTACKS[64]` lookup table
- Pawn attack tables (white and black)
- Generation functions for tables (computed at compile time or startup)

**Files**:
```
src/
└── attacks.rs
```

**Test**:
- Verify knight attacks from corners, edges, center
- Verify king attacks
- Verify pawn attacks

---

### Chunk 1.6: Magic Bitboards for Sliding Pieces
**Goal**: Implement magic bitboards for bishops and rooks.

**Deliverables**:
- Occupancy mask generation (relevant blockers for each square)
- Magic number tables (use known good magics, not search for them)
- `bishop_attacks(square, occupancy)` function
- `rook_attacks(square, occupancy)` function
- Queen attacks = bishop_attacks | rook_attacks

**Files**:
```
src/
└── magic.rs
```

**Test**:
- Compare against naive ray-based generation for many positions
- Test blocked vs unblocked scenarios
- Benchmark: target <10ns per lookup

---

### Chunk 1.7: Pseudo-Legal Move Generation
**Goal**: Generate all pseudo-legal moves (ignoring check legality).

**Deliverables**:
- Generate pawn moves (pushes, double pushes, captures, en passant, promotions)
- Generate knight moves
- Generate bishop/rook/queen moves (using magic bitboards)
- Generate king moves (non-castling)
- `generate_pseudo_legal_moves(board) -> Vec<Move>`

**Files**:
```
src/
└── movegen.rs
```

**Test**:
- Count moves from starting position (20)
- Count moves from known mid-game positions
- Verify each piece type generates correctly

---

### Chunk 1.8: Make/Unmake Move
**Goal**: Apply and revert moves on the board.

**Deliverables**:
- `board.make_move(mv)` - apply move, return undo info
- Handle special moves: castling, en passant, promotion
- Update castling rights, en passant square, halfmove/fullmove clocks
- Copy-make variant for MCTS: `board.clone().make_move(mv)`

**Files**:
```
src/
└── makemove.rs
```

**Test**:
- Make/unmake preserves board state exactly
- Castling moves both king and rook
- En passant removes the captured pawn
- Promotion replaces pawn with correct piece

---

### Chunk 1.9: Check Detection & Legal Move Filtering
**Goal**: Detect checks and filter to only legal moves.

**Deliverables**:
- `is_square_attacked(square, by_color, board)` function
- `is_in_check(board, color)` function
- Filter pseudo-legal moves: make move, check if own king attacked, unmake
- Castling legality (king not in check, doesn't pass through or land in check)

**Files**:
```
src/
├── attacks.rs  # (extend with is_square_attacked)
└── movegen.rs  # (extend with legal filtering)
```

**Test**:
- Detect known check positions
- Legal move count matches known positions
- Can't castle through check, out of check, or when rook has moved

---

### Chunk 1.10: Zobrist Hashing
**Goal**: Implement incremental position hashing for transposition detection.

**Deliverables**:
- Random 64-bit keys for each (piece, color, square) combination
- Keys for castling rights, en passant file, side to move
- `board.hash()` computed incrementally during make_move
- XOR-based update (hash ^= old_key ^ new_key)

**Files**:
```
src/
└── zobrist.rs
```

**Test**:
- Same position via different move orders = same hash
- Different positions = different hash (test many)
- Incremental matches from-scratch computation

---

### Chunk 1.11: Perft Testing & Benchmarks
**Goal**: Validate correctness with perft, benchmark speed.

**Deliverables**:
- `perft(board, depth)` - count leaf nodes at given depth
- `divide(board, depth)` - perft broken down per root move (for debugging)
- Test suite with known perft values from multiple positions
- Criterion benchmark harness

**Files**:
```
src/
└── perft.rs
benches/
└── perft.rs
```

**Test**:
- Starting position: perft(1)=20, perft(2)=400, perft(3)=8902, perft(4)=197281, perft(5)=4865609
- Kiwipete position and other tricky positions
- **Performance target**: >100M nodes/sec

---

### Chunk 1.12: Game State & Termination
**Goal**: Detect all game-ending conditions.

**Deliverables**:
- Checkmate detection (in check + no legal moves)
- Stalemate detection (not in check + no legal moves)
- Threefold repetition (needs position history with Zobrist hashes)
- 50-move rule (halfmove clock >= 100)
- Insufficient material (K vs K, K+B vs K, K+N vs K)
- `GameResult` enum: WhiteWins, BlackWins, Draw, Ongoing

**Files**:
```
src/
└── game.rs
```

**Test**:
- Known checkmate positions (scholar's mate, back rank, etc.)
- Known stalemate positions
- Threefold repetition in a real game sequence
- 50-move draw scenario

---

## Phase 2: Neural Network (Python/PyTorch)

### Chunk 2.1: Project Setup & Config
**Goal**: Set up Python project with fully configurable network parameters.

**Deliverables**:
- pyproject.toml with dependencies (torch, numpy)
- `NetworkConfig` dataclass:
  - `num_blocks`: int (default 19, range 1-40)
  - `num_filters`: int (default 256, range 32-512)
  - `input_planes`: int (default 119)
  - `policy_output_size`: int (default 4672)
  - `value_hidden_size`: int (default 256)
- Preset configs: TINY, SMALL, MEDIUM, FULL
- Device handling (CPU/CUDA auto-detect)

**Files**:
```
neural/
├── pyproject.toml
└── neural/
    ├── __init__.py
    └── config.py
```

**Test**: Import and instantiate each preset. Verify parameter counts.

---

### Chunk 2.2: Board Encoding
**Goal**: Convert board state to neural network input tensor.

**Deliverables**:
- Encode single position to tensor (8x8xP planes)
- Plane layout (document clearly):
  - 12 planes per history position (P1 pawns, P1 knights, ..., P2 pawns, ...) x T positions
  - Repetition count planes
  - Castling rights (4 planes)
  - Side to move (1 plane)
  - Move count planes
- Batch encoding for multiple positions
- History stacking (T=8 past positions)

**Files**:
```
neural/
└── neural/
    └── encoding.py
```

**Test**:
- Encode starting position, verify piece planes are correct
- Encode mid-game positions, verify by hand
- Benchmark encoding speed (should not bottleneck training)

---

### Chunk 2.3: Move Encoding/Decoding
**Goal**: Bidirectional mapping between chess moves and policy vector indices.

**Deliverables**:
- Policy output = 8x8x73 = 4672 outputs
  - 56 "queen moves" (7 distances x 8 directions)
  - 8 knight moves
  - 9 underpromotions (3 piece types x 3 directions)
- `move_to_index(from_sq, to_sq, promotion)` -> int
- `index_to_move(index)` -> (from_sq, to_sq, promotion)
- `mask_legal_moves(legal_moves)` -> bool tensor of size 4672

**Files**:
```
neural/
└── neural/
    └── moves.py
```

**Test**:
- Roundtrip all 4672 indices
- Verify queen moves along each direction
- Verify knight move encoding
- Verify underpromotion encoding

---

### Chunk 2.4: Residual Block
**Goal**: Implement a single residual block (the building block).

**Deliverables**:
- `ResidualBlock(nn.Module)`:
  - Conv2d(filters, filters, 3, padding=1) -> BatchNorm2d -> ReLU
  - Conv2d(filters, filters, 3, padding=1) -> BatchNorm2d
  - Skip connection (add input)
  - Final ReLU
- He initialization for conv weights
- Configurable filter count

**Files**:
```
neural/
└── neural/
    └── blocks.py
```

**Test**:
- Forward pass: input shape = output shape
- Gradient flows through skip connection (gradient norm > 0)
- Parameter count: 2 * (filters^2 * 9 + filters) for convs + BN params

---

### Chunk 2.5: Full Network Architecture
**Goal**: Assemble complete dual-headed network, fully configurable.

**Deliverables**:
- `AlphaZeroNetwork(nn.Module)`:
  - Input conv: Conv2d(input_planes, num_filters, 3, padding=1) -> BN -> ReLU
  - Body: N x ResidualBlock (N = config.num_blocks)
  - Policy head: Conv2d(num_filters, 2, 1) -> BN -> ReLU -> flatten -> Linear(2*64, 4672)
  - Value head: Conv2d(num_filters, 1, 1) -> BN -> ReLU -> flatten -> Linear(64, value_hidden) -> ReLU -> Linear(value_hidden, 1) -> tanh
- All sizes derived from NetworkConfig
- `from_config(config)` class method

**Files**:
```
neural/
└── neural/
    └── network.py
```

**Test**:
- Forward pass with batch of 8 positions using TINY config
- Output shapes: policy=(B, 4672), value=(B, 1)
- Value output in [-1, 1] (tanh)
- Test each preset: verify param counts match expectations
- Forward pass with FULL config (verify it runs, check shapes)

---

### Chunk 2.6: Loss Functions
**Goal**: Implement AlphaZero combined loss.

**Deliverables**:
- Policy loss: cross-entropy between predicted policy and MCTS visit distribution
  - `loss_policy = -sum(pi * log(p))` where pi = MCTS target, p = network output
- Value loss: MSE between predicted value and game outcome
  - `loss_value = (z - v)^2` where z = {-1, 0, 1}, v = network output
- Combined: `loss = loss_value + loss_policy + c * ||theta||^2`
  - c = weight decay (1e-4), applied via optimizer, not manually
- Return individual losses for logging

**Files**:
```
neural/
└── neural/
    └── losses.py
```

**Test**:
- Loss on random data is high
- Loss on "perfect" data (matching targets) is near zero
- Gradients flow correctly

---

### Chunk 2.7: Model Export with Optimizations
**Goal**: Export model for fast Rust inference.

**Deliverables**:
- TorchScript export via `torch.jit.trace`
- `torch.compile` wrapper for optimized inference (when used in Python)
- FP16 model variant export
- Verify exported model matches original (within FP tolerance)
- Benchmark: compare FP32 vs FP16 inference speed
- Document exact input/output tensor shapes and dtypes

**Files**:
```
neural/
└── neural/
    └── export.py
```

**Test**:
- Export each preset, reload, compare outputs (atol=1e-5 for FP32, 1e-3 for FP16)
- Benchmark inference: FP32 vs FP16 on GPU
- Verify TorchScript model is portable (load on different machine)

---

### Chunk 2.8: TensorRT Export (Optional Performance)
**Goal**: Export optimized TensorRT engine for maximum inference speed.

**Deliverables**:
- TensorRT export via torch-tensorrt or trtexec
- Fixed input shapes for MCTS batch sizes (32, 64, 128)
- FP16 and FP8 (H200) precision modes
- Fallback to TorchScript if TensorRT unavailable

**Files**:
```
neural/
└── neural/
    └── tensorrt_export.py
```

**Test**:
- Benchmark TensorRT vs TorchScript: expect 2-5x speedup
- Output accuracy within tolerance
- Verify FP8 doesn't degrade play strength significantly

---

## Phase 3: MCTS (Rust)

### Chunk 3.1: Tree Node Structure
**Goal**: Define MCTS tree node with cache-friendly layout.

**Deliverables**:
- `Node` struct with hot fields packed for cache line (64 bytes):
  - `visit_count: u32` (4 bytes)
  - `total_value: f32` (4 bytes) — mean Q = total_value / visit_count
  - `prior: f32` (4 bytes) — P(s,a) from neural network
  - `move: Move` (4 bytes)
  - `first_child: NodeIndex` (4 bytes)
  - `next_sibling: NodeIndex` (4 bytes)
  - `num_children: u16` (2 bytes)
  - `is_expanded: bool` (1 byte)
- `NodeIndex` type (u32 index into arena)
- Sibling-linked list for children (avoids Vec allocation per node)
- Document why this layout: hot fields (visit_count, total_value, prior) are
  accessed every selection step and should be in same cache line

**Files**:
```
mcts/
├── Cargo.toml
└── src/
    ├── lib.rs
    └── node.rs
```

**Test**:
- `assert!(std::mem::size_of::<Node>() <= 32)` — fits two nodes per cache line
- Create nodes, add children via sibling links, traverse

---

### Chunk 3.2: Arena Allocator
**Goal**: Fast, cache-friendly bulk node allocation.

**Deliverables**:
- `Arena<Node>` backed by `Vec<Node>` (contiguous memory)
- `arena.alloc() -> NodeIndex` — returns index, not pointer
- `arena[index]` — index-based access
- `arena.clear()` — reset for new search (no deallocation, just reset counter)
- Pre-allocate capacity based on expected search size

**Files**:
```
mcts/
└── src/
    └── arena.rs
```

**Test**:
- Allocate 1M nodes, verify indices sequential
- Clear and re-allocate, verify reuse
- Benchmark: alloc should be <10ns per node

---

### Chunk 3.3: PUCT Selection with FPU
**Goal**: Implement UCB selection formula with First Play Urgency.

**Deliverables**:
- PUCT formula:
  ```
  UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
  ```
- First Play Urgency: for unvisited children, use:
  ```
  Q_fpu = parent_Q - fpu_reduction
  ```
  instead of Q=0. This avoids over-optimistic exploration of unvisited nodes.
  Default `fpu_reduction = 0.25` (configurable).
- `select_child(node, arena, config) -> NodeIndex`
- Config struct with c_puct, fpu_reduction

**Files**:
```
mcts/
└── src/
    ├── select.rs
    └── config.rs
```

**Test**:
- With uniform priors and no visits, selects first child
- After visits accumulate, balances exploration vs exploitation
- FPU: unvisited node score < parent Q when parent Q > fpu_reduction
- c_puct=0 should always select highest Q child

---

### Chunk 3.4: Tree Expansion
**Goal**: Expand leaf nodes with legal moves and policy priors.

**Deliverables**:
- `expand(node, board, policy, arena)`:
  - Generate legal moves from board
  - Create child node for each move
  - Set prior = policy[move] (from NN) normalized to sum to 1
  - Link children via sibling list
- Handle terminal nodes (no expansion, return game result as value)

**Files**:
```
mcts/
└── src/
    └── expand.rs
```

**Test**:
- Expand from starting position: 20 children
- Priors sum to 1.0
- Terminal positions: no children, returns correct value

---

### Chunk 3.5: Backpropagation
**Goal**: Update tree statistics after leaf evaluation.

**Deliverables**:
- `backup(path, value)`:
  - Walk path from leaf to root
  - At each node: visit_count += 1, total_value += value
  - Negate value at each level (what's good for me is bad for opponent)
- Path stored as Vec<NodeIndex> during selection

**Files**:
```
mcts/
└── src/
    └── backup.rs
```

**Test**:
- Single path backup: verify counts and values
- Two paths sharing prefix: verify correct accumulation
- Verify Q = total_value / visit_count after backup

---

### Chunk 3.6: Single-Threaded Search (No NN)
**Goal**: Complete MCTS loop working end-to-end with uniform random policy.

**Deliverables**:
- Full search loop for N simulations:
  1. Select: walk tree using PUCT until leaf
  2. Expand: add children with uniform priors (1/num_moves)
  3. Evaluate: random value in [-1, 1] (placeholder for NN)
  4. Backup: propagate value up
- `search(board, num_simulations, config) -> Vec<(Move, visit_count)>`
- Temperature-based move selection:
  - T=1: sample proportional to visit_count
  - T->0: pick most visited
- Dirichlet noise at root: `prior = (1-eps)*prior + eps*Dir(alpha)`
  - alpha=0.3 for chess, eps=0.25

**Files**:
```
mcts/
└── src/
    └── search.rs
```

**Test**:
- 800 simulations completes in <100ms (no NN overhead)
- Play 100 games: MCTS vs pure random. MCTS should win >90%
- Verify root visit counts: most-visited move is reasonable

---

### Chunk 3.7: TorchScript Integration
**Goal**: Load and run neural network from Rust via tch-rs.

**Deliverables**:
- Load TorchScript model file (`.pt`)
- Encode `Board` -> tensor matching Python encoding exactly
- Single position inference: `(policy, value)` = forward(encoded_board)
- Batched inference: `Vec<(policy, value)>` = forward(batch_of_boards)
- GPU device selection

**Files**:
```
mcts/
└── src/
    └── nn.rs
```

**Test**:
- Load exported model from Chunk 2.7
- Compare Rust inference output vs Python for same positions (within tolerance)
- Benchmark single inference latency
- Benchmark batched inference throughput (batch sizes 32, 64, 128)

---

### Chunk 3.8: NN-Guided Search
**Goal**: Replace random policy/value with neural network evaluation.

**Deliverables**:
- During expansion: get policy from NN, use as priors
- During evaluation: use NN value instead of random
- Leaf evaluation: encode position, run NN, extract policy+value
- Still single-threaded (batching comes next)

**Files**:
```
mcts/
└── src/
    └── search.rs  # (extend)
```

**Test**:
- Search with NN produces different (better) move ordering than random
- Compare move selection: NN-guided should prefer more natural moves
- Benchmark: expect slower than random due to NN calls (this is expected)

---

### Chunk 3.9a: Batched Async Inference Pipeline
**Goal**: Batch leaf evaluations for efficient GPU use.

This is the single most important performance optimization. Without batching,
each MCTS simulation makes one GPU call. With batching, we collect multiple
leaves and evaluate them in a single forward pass.

**Deliverables**:
- `InferenceServer`:
  - Receives position evaluation requests via channel
  - Collects requests until batch is full or timeout
  - Runs batched inference on GPU
  - Dispatches results back to waiting threads
- Configurable batch size (32, 64, 128)
- Configurable max wait time (prevents starvation)
- Stats tracking: batch fill rate, GPU utilization

**Files**:
```
mcts/
└── src/
    └── batch.rs
```

**Test**:
- Submit 128 requests, verify all get results
- Verify batching: GPU called once, not 128 times
- Benchmark: throughput should be ~batch_size x single inference

---

### Chunk 3.9b: Parallel Search with Lock-Free Atomics
**Goal**: Multi-threaded MCTS with minimal synchronization.

**Deliverables**:
- Virtual loss: when thread selects a node, temporarily add a "virtual loss"
  to discourage other threads from picking the same path
  - `virtual_loss = 1` added to visit_count, subtracted from value
  - Removed during backup
- Lock-free stats updates using atomics:
  - `visit_count: AtomicU32`
  - `total_value: AtomicU32` (store f32 bits as u32, use CAS for add)
- Multiple worker threads: each runs select->expand->evaluate->backup
- Workers submit leaf positions to InferenceServer (from 3.9a), block until result
- Configurable thread count

**Files**:
```
mcts/
└── src/
    ├── search.rs   # (extend with parallel search)
    └── node.rs     # (add atomic variants)
```

**Test**:
- Run with 1, 2, 4, 8 threads
- Verify total visit count = num_simulations regardless of thread count
- Benchmark scaling: expect near-linear up to ~4 threads, then GPU-bound
- No data races: run under `cargo test` with thread sanitizer

---

### Chunk 3.10: Tree Reuse & Transposition Table
**Goal**: Avoid redundant work across moves and across transpositions.

**Deliverables**:
- **Tree reuse**: After selecting a move, keep the subtree rooted at that move.
  Discard everything else. The next search starts with pre-existing visit counts.
  - `reuse_subtree(root, chosen_move, arena) -> new_root`
- **Transposition table**: Hash map from Zobrist hash -> NodeIndex.
  When expanding a position that's already in the table, share the evaluation
  (policy + value) instead of calling NN again.
  - Use a fixed-size hash table (power of 2, replace-oldest strategy)
  - Only share evaluations, not subtrees (sharing subtrees creates DAGs which
    complicate backup — keep it simple)

**Files**:
```
mcts/
└── src/
    └── transposition.rs
```

**Test**:
- Tree reuse: after move, root has previous visits
- Transposition: same position via different move orders hits cache
- Memory: transposition table stays within size limit
- Benchmark: measure NN calls saved by transposition table

---

## Phase 4: Self-Play

### Chunk 4.1: Game Loop
**Goal**: Play a complete self-play game using MCTS.

**Deliverables**:
- Initialize board at starting position
- At each position:
  - Run MCTS for N simulations (configurable, default 800)
  - Select move using temperature schedule:
    - Moves 1-30: T=1 (proportional to visit counts, for diversity)
    - Moves 31+: T->0 (pick most visited, for strength)
  - Apply move, record position
- Detect game end (checkmate, stalemate, draw)
- Return game record with result

**Files**:
```
self-play/
├── Cargo.toml
└── src/
    ├── main.rs
    └── game.rs
```

**Test**:
- Play single game to completion (with tiny/random network)
- All moves are legal
- Game ends with valid result
- Average game length is reasonable (40-100 moves)

---

### Chunk 4.2: Training Data Collection
**Goal**: Extract training data from self-play games.

**Deliverables**:
- At each move, store a training sample:
  - `position`: board state (encoded for NN input)
  - `policy`: normalized MCTS visit counts (target for policy head)
  - `result`: placeholder, filled after game ends
- After game: assign result to all positions:
  - +1 for winning side's positions
  - -1 for losing side's positions
  - 0 for draws
- `TrainingSample { position, policy, value }` struct

**Files**:
```
self-play/
└── src/
    └── data.rs
```

**Test**:
- Play game, extract samples
- Policy sums to 1.0 for each sample
- Values are exactly {-1, 0, +1}
- Sample count = number of moves in game

---

### Chunk 4.3: Data Serialization
**Goal**: Efficiently serialize training data to disk.

**Deliverables**:
- Binary format using MessagePack (serde + rmp-serde):
  - Compact: ~2KB per sample
  - Fast: >100K samples/sec write speed
- File format: one file per game, or batched files
- Include metadata header (format version, network config used)
- Read function for verification

**Files**:
```
self-play/
└── src/
    └── serialize.rs
```

**Test**:
- Roundtrip: serialize -> deserialize -> compare
- File size matches expectations
- Benchmark write throughput

---

### Chunk 4.4: Replay Buffer
**Goal**: Manage accumulated training data with ring buffer semantics.

**Deliverables**:
- Rust side: write game files to output directory
- Python side: `ReplayBuffer` class that:
  - Scans directory for game files
  - Maintains window of most recent N games (ring buffer)
  - Random sampling of individual positions for training
  - Removes old files when buffer is full
- Configurable capacity (number of games)

**Files**:
```
self-play/
└── src/
    └── buffer.rs
training/
└── training/
    └── buffer.py  # Python reader
```

**Test**:
- Add more games than capacity, oldest are evicted
- Sampling is approximately uniform
- Python reads what Rust wrote (cross-language compatibility)

---

### Chunk 4.5: Self-Play Worker Binary
**Goal**: Standalone binary for generating self-play games.

**Deliverables**:
- CLI: `self-play --model model.pt --games 100 --output ./data/ --sims 800`
- Generate N games with given model
- Progress bar and statistics:
  - Games completed / total
  - Average game length
  - Win/Draw/Loss distribution
  - Games per second
  - NN evaluations per second
- Signal handling: save progress on SIGTERM

**Files**:
```
self-play/
└── src/
    └── main.rs  # (extend with CLI and stats)
```

**Test**:
- Run worker, verify output files are created and valid
- Statistics output is reasonable
- Monitor GPU utilization (should be high)

---

## Phase 5: Training

### Chunk 5.1: Data Loading
**Goal**: Efficient PyTorch DataLoader for replay buffer data.

**Deliverables**:
- `ReplayDataset(Dataset)`: reads from replay buffer
- Decodes binary samples to tensors
- Pin memory for fast GPU transfer
- Prefetch with num_workers > 0
- Shuffling across the buffer

**Files**:
```
training/
├── pyproject.toml
└── training/
    ├── __init__.py
    └── dataloader.py
```

**Test**:
- Load data, verify tensor shapes match network expectations
- Benchmark: data loading should not be the bottleneck
- Memory usage is stable (no leaks over many batches)

---

### Chunk 5.2: Training Loop (Single GPU)
**Goal**: Basic training loop with essential optimizations.

**Deliverables**:
- Training step: forward pass -> combined loss -> backward -> optimizer step
- Optimizer: SGD with momentum (0.9) and weight decay (1e-4)
- Learning rate schedule: multi-step decay (0.2 -> 0.02 -> 0.002)
- `torch.compile(model)` for fused kernels (easy 1.5-2x speedup)
- Logging per step: policy_loss, value_loss, total_loss, learning_rate
- Configurable batch size (default 4096, fits easily on H200)

**Files**:
```
training/
└── training/
    └── train.py
```

**Test**:
- Train 100 steps on dummy data: loss decreases
- Gradients are non-zero for all parameters
- Learning rate updates at correct steps
- Verify `torch.compile` doesn't change outputs (within tolerance)

---

### Chunk 5.3: Checkpointing
**Goal**: Robust save/resume for long training runs.

**Deliverables**:
- Save: model state, optimizer state, scheduler state, step count, config
- Load: resume training from any checkpoint
- Keep N most recent checkpoints (default 5), delete older ones
- Auto-export TorchScript model alongside each checkpoint (for self-play)
- Atomic writes (write to temp file, then rename) to prevent corruption

**Files**:
```
training/
└── training/
    └── checkpoint.py
```

**Test**:
- Save at step 100, resume, train to step 200
- Loss trajectory is continuous (no jump at resume)
- Old checkpoints are cleaned up

---

### Chunk 5.4: Mixed Precision Training
**Goal**: Use AMP for ~2x training speedup.

**Deliverables**:
- `torch.amp.autocast('cuda')` context for forward pass
- `GradScaler` for safe FP16 backward pass
- Skip steps with NaN/Inf gradients (GradScaler handles this)
- Compare: FP32 vs FP16 loss curves should be nearly identical
- Flag to enable/disable AMP

**Files**:
```
training/
└── training/
    └── train.py  # (extend with AMP support)
```

**Test**:
- Train 500 steps FP32 vs FP16: final loss within 5%
- No NaN/Inf in any parameters after training
- Benchmark: expect ~1.5-2x wall-clock speedup on H200

---

### Chunk 5.5: Distributed Training (DDP)
**Goal**: Scale training across multiple GPUs.

**Deliverables**:
- `DistributedDataParallel` wrapper
- Process group init via `init_process_group(backend='nccl')`
- `DistributedSampler` for data loading (no duplicates across ranks)
- Gradient synchronization (automatic with DDP)
- Only rank 0 saves checkpoints and logs

**Files**:
```
training/
└── training/
    └── distributed.py
```

**Test**:
- Train on 2 GPUs: `torchrun --nproc_per_node=2 train.py`
- Verify all ranks have identical model weights after each step
- Loss curve matches single-GPU (with same total batch size)

---

### Chunk 5.6: Slurm Integration
**Goal**: Job scripts for cluster training.

**Deliverables**:
- Single-node SBATCH script (1 node, N GPUs)
- Multi-node SBATCH script (M nodes, N GPUs each)
- Auto-detect SLURM_PROCID, SLURM_NTASKS for DDP world setup
- Fault tolerance: auto-restart from latest checkpoint
- Resource requests tuned for H200 (memory, time limits)

**Files**:
```
training/
└── scripts/
    ├── train.sbatch
    ├── train_multi.sbatch
    └── launch.py           # Helper to submit with correct args
```

**Test**:
- Submit single-node job, training completes
- Submit multi-node job, training completes
- Kill job mid-training, resubmit, training resumes

---

### Chunk 5.7: Monitoring & Metrics
**Goal**: Comprehensive training visibility.

**Deliverables**:
- TensorBoard integration:
  - Scalar: policy_loss, value_loss, total_loss, learning_rate
  - Scalar: policy_accuracy (top-1 match with MCTS target)
  - Scalar: value_MSE
  - Scalar: samples/second throughput
- Optional Weights & Biases integration
- Log to both console and file

**Files**:
```
training/
└── training/
    └── metrics.py
```

**Test**:
- Run TensorBoard, verify all metrics appear
- Metrics update at correct frequency

---

## Phase 6: Orchestration

### Chunk 6.1: Weight Distribution
**Goal**: Sync trained model weights to self-play workers.

**Deliverables**:
- After N training steps, export latest model as TorchScript
- Write to shared filesystem path with version number
- Self-play workers poll for new weights and hot-swap
- Version tracking: workers log which model version they're using

**Files**:
```
orchestrator/
├── pyproject.toml
└── orchestrator/
    └── weights.py
```

**Test**:
- Export weights, worker detects and loads new version
- Self-play continues without interruption during swap

---

### Chunk 6.2: Model Evaluation
**Goal**: Measure strength progression.

**Deliverables**:
- Play N games between two checkpoints (e.g. current vs previous)
- Win/Draw/Loss statistics
- ELO estimation using BayesElo or simple formula
- Optional: play against Stockfish at fixed depth for absolute strength

**Files**:
```
orchestrator/
└── orchestrator/
    └── evaluate.py
```

**Test**:
- Evaluate random network vs random network: ~50% win rate
- After training: new model beats old model

---

### Chunk 6.3: Pipeline Coordinator
**Goal**: Orchestrate the full AlphaZero training loop.

**Deliverables**:
- Coordinator loop:
  1. Launch self-play workers (Slurm job array)
  2. Wait until replay buffer has enough new data
  3. Launch training job
  4. After training: export new weights
  5. Launch evaluation job
  6. If new model wins >55%: promote to best model
  7. Repeat
- YAML config for all pipeline parameters
- Logging: full pipeline state, timing per phase

**Files**:
```
orchestrator/
└── orchestrator/
    ├── coordinator.py
    └── config.yaml
```

**Test**:
- Run one full iteration end-to-end
- Pipeline recovers from any phase failing

---

## Phase 7: Python Bindings & CLI

### Chunk 7.1: PyO3 Chess Engine Bindings ✅ DONE
**Goal**: Expose chess engine to Python for testing and integration.

**Deliverables**:
- `PyBoard` class: FEN parsing, legal moves, make move, game state
- `PyMove` class: from/to/promotion
- Compatible with training code (can use for encoding verification)

**Files**:
```
alphazero-py/
├── Cargo.toml
└── src/
    └── lib.rs
```

**Test**:
- Use from Python
- Compare legal move generation with python-chess library

---

### Chunk 7.2: PyO3 MCTS Bindings ✅ DONE
**Goal**: Expose MCTS to Python for analysis and debugging.

**Deliverables**:
- `search(board, model_path, num_sims, config) -> list[(move, visits, Q)]`
- Configurable search parameters from Python
- Useful for debugging: inspect what MCTS is thinking

**Files**:
```
alphazero-py/
└── src/
    └── lib.rs  # (extend)
```

**Test**:
- Run search from Python, verify reasonable output
- Compare with Rust-only results

---

### Chunk 7.3: CLI Tool ✅ DONE
**Goal**: Unified command-line interface.

**Deliverables**:
- `alphazero train` — launch training pipeline
- `alphazero self-play` — run self-play worker
- `alphazero evaluate` — evaluate two models
- `alphazero play` — interactive play vs engine in terminal
- `alphazero analyze` — analyze a FEN position (show top moves, values)

**Files**:
```
alphazero/
├── pyproject.toml
└── alphazero/
    └── cli.py
```

**Test**:
- Each subcommand shows help
- `alphazero play` works interactively

---

# Dependency Graph

```
Phase 1 (Chess Engine):
1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7 → 1.8 → 1.9 → 1.10 → 1.11 → 1.12

Phase 2 (Neural Network):                          [can run parallel with Phase 1]
2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6 → 2.7 → 2.8(optional)

Phase 3 (MCTS):                                    [needs Phase 1 + 2]
3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7 → 3.8 → 3.9a → 3.9b → 3.10

Phase 4 (Self-Play):                               [needs Phase 3]
4.1 → 4.2 → 4.3 → 4.4 → 4.5

Phase 5 (Training):                                [needs Phase 2 + 4]
5.1 → 5.2 → 5.3 → 5.4 → 5.5 → 5.6 → 5.7

Phase 6 (Orchestration):                           [needs Phase 4 + 5]
6.1 → 6.2 → 6.3

Phase 7 (Bindings & CLI):                          [can start after Phase 1]
7.1 → 7.2 → 7.3
```

**Parallel tracks**:
- Phase 1 and Phase 2 have zero dependencies on each other
- Phase 7.1 can start as soon as Phase 1 is done

---

# Quick Reference: Chunk Summaries

| Chunk | Description | Test Command |
|-------|-------------|--------------|
| 1.1 | Types (Square, Piece, Move) | `cargo test -p chess-engine` | DONE (41 tests) |
| 1.2 | Bitboard operations | `cargo test bitboard` | DONE (44 tests) |
| 1.3 | Board representation | `cargo test board` | DONE (42 tests) |
| 1.4 | FEN parsing | `cargo test fen` | DONE (55 tests) |
| 1.5 | Knight/King attacks | `cargo test attacks` | DONE (34 tests) |
| 1.6 | Magic bitboards | `cargo test magic && cargo bench` | DONE (44 tests) |
| 1.7 | Pseudo-legal movegen | `cargo test movegen` | DONE (40 tests) |
| 1.8 | Make/unmake move | `cargo test makemove` | DONE (60 tests) |
| 1.9 | Legal move filtering | `cargo test legal` | DONE (33 tests) |
| 1.10 | Zobrist hashing | `cargo test zobrist` | DONE (30 tests) |
| 1.11 | Perft validation | `cargo test perft && cargo bench perft` | DONE (24+8 tests) |
| 1.12 | Game termination | `cargo test game` | DONE (37 tests) |
| 2.1 | Config + presets | `pytest neural/` | DONE (36 tests) |
| 2.2 | Board encoding | `pytest tests/test_encoding.py` | DONE (71 tests) |
| 2.3 | Move encoding | `pytest tests/test_moves.py` | DONE (78 tests) |
| 2.4 | Residual block | `pytest tests/test_blocks.py` | DONE (37 tests) |
| 2.5 | Full network (configurable) | `pytest tests/test_network.py` | DONE (64 tests) |
| 2.6 | Loss functions | `pytest tests/test_losses.py` | DONE (36 tests) |
| 2.7 | Export (TorchScript + FP16) | `python -m neural.export && pytest` | DONE (36 tests) |
| 2.8 | TensorRT export (optional) | `python -m neural.tensorrt_export` |
| 3.1 | Cache-friendly tree nodes | `cargo test -p mcts node` | DONE (13 tests) |
| 3.2 | Arena allocator | `cargo test arena` | DONE (10 tests) |
| 3.3 | PUCT selection + FPU | `cargo test select` | DONE (17 tests) |
| 3.4 | Tree expansion | `cargo test expand` | DONE (15 tests) |
| 3.5 | Backpropagation | `cargo test backup` | DONE (9 tests) |
| 3.6 | Single-threaded search | `cargo test search` | DONE (18 tests) |
| 3.7 | TorchScript in Rust | `cargo test nn` | DONE (35 tests) |
| 3.8 | NN-guided search | `cargo test guided` | DONE (12 tests) |
| 3.9a | Batched async inference | `cargo test batch && cargo bench` | DONE (10 tests) |
| 3.9b | Parallel search (atomics) | `cargo test parallel && cargo bench` | DONE (9 tests) |
| 3.10 | Tree reuse + transpositions | `cargo test reuse && cargo test transposition` | DONE (18 tests) |
| 4.1 | Game loop | `cargo run -p self-play -- --games 1` | DONE (14 tests) |
| 4.2 | Data collection | `cargo test data` | DONE (13 tests) |
| 4.3 | Serialization | `cargo test serialize` | DONE (11 tests) |
| 4.4 | Replay buffer | `cargo test buffer` | DONE (11 tests + 17 Python) |
| 4.5 | Self-play worker | `cargo run -p self-play -- --games 10` | DONE (8 tests + 2 parallel) |
| 5.1 | Data loading | `pytest tests/test_dataloader.py` | DONE (13 tests) |
| 5.2 | Training loop + torch.compile | `python -m training.train --steps 100` | DONE (16 tests) |
| 5.3 | Checkpointing | `pytest tests/test_checkpoint.py` | DONE (16 tests) |
| 5.4 | Mixed precision (AMP) | `python -m training.train --amp` | DONE (6 tests) |
| 5.5 | Distributed training (DDP) | `torchrun --nproc_per_node=2 ...` | DONE (10 tests) |
| 5.6 | Slurm integration | `sbatch scripts/train.sbatch` | DONE (4 scripts) |
| 5.7 | Monitoring (TensorBoard) | `tensorboard --logdir runs/` | DONE (17 tests) |
| 6.1 | Weight distribution | `pytest tests/test_weights.py` | DONE (16 tests) |
| 6.2 | Model evaluation | `python -m orchestrator.evaluate` | DONE (33 tests) |
| 6.3 | Pipeline coordinator | `python -m orchestrator.coordinator` | DONE (25 tests) |
| 7.1 | Chess engine bindings | `pytest tests/test_bindings.py` |
| 7.2 | MCTS bindings | `pytest tests/test_mcts_bindings.py` |
| 7.3 | CLI tool | `alphazero --help` |
