"""Data loading utilities for AlphaZero training.

Provides a PyTorch-compatible Dataset and DataLoader factory for sampling
training positions from the replay buffer.

Two dataset types are provided:

    :class:`ReplayDataset`
        An ``IterableDataset`` that continuously samples from a live replay
        buffer.  Uses the iterable style (rather than map-style) because the
        buffer changes dynamically as self-play generates new games -- there is
        no fixed-length index to map over.  Each worker independently samples
        random positions, so multi-worker loading works out of the box with no
        custom sampler.

    :class:`DummyDataset`
        A map-style ``Dataset`` that generates random tensors matching the
        expected shapes.  Useful for benchmarking the training loop and running
        tests without real game data.

Both datasets yield ``(board, policy, value)`` tuples with shapes
``(119, 8, 8)``, ``(4672,)``, ``(1,)`` respectively, all ``float32``.

Usage::

    from training.buffer import ReplayBuffer
    from training.dataloader import create_dataloader, DummyDataset

    # Real data
    buf = ReplayBuffer("/path/to/replay/data")
    buf.scan()
    loader = create_dataloader(buf, batch_size=4096)

    # Dummy data for testing
    from torch.utils.data import DataLoader
    loader = DataLoader(DummyDataset(size=10000), batch_size=32, shuffle=True)
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from training.buffer import ReplayBuffer

# ---------------------------------------------------------------------------
# Constants (duplicated from neural.config for import-free access)
# ---------------------------------------------------------------------------

POLICY_SIZE: int = 4672
"""Number of possible moves in the AlphaZero policy encoding (8*8*73)."""

INPUT_PLANES: int = 119
"""Number of input feature planes for the neural network."""


# ============================================================================
# ReplayDataset -- live sampling from the replay buffer
# ============================================================================


class ReplayDataset(IterableDataset):
    """An iterable dataset that continuously samples from the replay buffer.

    Uses ``IterableDataset`` (not map-style) because the replay buffer changes
    dynamically as self-play generates new games.  Each worker independently
    samples random positions, so there is no need for a custom sampler or
    index-based access.

    Each yielded sample is a tuple of:
        - ``board``:  ``(119, 8, 8)`` float32 tensor
        - ``policy``: ``(4672,)``    float32 tensor (sparse -> dense)
        - ``value``:  ``(1,)``       float32 tensor

    Args:
        buffer: :class:`~training.buffer.ReplayBuffer` to sample from.
            Must have been scanned (``buffer.scan()``) before iteration.
        samples_per_epoch: Number of samples per "epoch".  Since the buffer
            is dynamic, an epoch is simply a fixed number of samples rather
            than a full pass over the data.
    """

    def __init__(self, buffer: ReplayBuffer, samples_per_epoch: int = 100_000) -> None:
        self.buffer = buffer
        self.samples_per_epoch = samples_per_epoch

    def __iter__(self):
        """Yield encoded training samples from the replay buffer."""
        # Defer the import so that workers that only use DummyDataset do not
        # need the neural package installed.
        from neural.encoding import BoardState, encode_board

        for _ in range(self.samples_per_epoch):
            positions = self.buffer.sample_positions(1)
            if not positions:
                continue
            pos = positions[0]

            # Encode the board position into a (119, 8, 8) tensor
            state = BoardState.from_fen_piece_placement(pos["fen"])
            board_tensor = encode_board(state)  # (119, 8, 8)

            # Convert sparse policy to dense (4672,) tensor
            policy = torch.zeros(POLICY_SIZE, dtype=torch.float32)
            for idx, prob in pos["policy"]:
                policy[int(idx)] = float(prob)

            # Value target
            value = torch.tensor([pos["value"]], dtype=torch.float32)

            yield board_tensor, policy, value

    def __len__(self) -> int:
        """Return the nominal epoch length.

        This is approximate -- some samples may be skipped if the buffer
        fails to return data for a position.
        """
        return self.samples_per_epoch


# ============================================================================
# DummyDataset -- random data for testing
# ============================================================================


class DummyDataset(Dataset):
    """Generates random training data for testing the training loop.

    Produces random boards, softmax-normalized policy targets, and random
    values in [-1, 1].  Shapes match the real data exactly so the training
    loop can be exercised without any game files.

    Args:
        size: Number of samples in the dataset.
    """

    def __init__(self, size: int = 10_000) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        board = torch.randn(INPUT_PLANES, 8, 8)
        policy = torch.softmax(torch.randn(POLICY_SIZE), dim=0)
        value = torch.tensor([random.uniform(-1, 1)], dtype=torch.float32)
        return board, policy, value


# ============================================================================
# DataLoader factory
# ============================================================================


def create_dataloader(
    buffer: ReplayBuffer,
    batch_size: int = 4096,
    samples_per_epoch: int = 100_000,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader that samples from the replay buffer.

    This is the recommended way to feed data into the training loop.  The
    returned loader yields batches of ``(boards, policies, values)`` tensors
    with shapes ``(B, 119, 8, 8)``, ``(B, 4672)``, ``(B, 1)``.

    Args:
        buffer: :class:`~training.buffer.ReplayBuffer` to sample from.
        batch_size: Number of positions per batch.
        samples_per_epoch: Total samples per epoch.
        num_workers: Number of data-loading worker processes.  ``0`` means
            loading happens in the main process (simplest, fine for small
            buffers).
        pin_memory: Whether to copy tensors into CUDA pinned memory before
            returning them.  ``True`` gives a small speedup when training on
            GPU.

    Returns:
        A :class:`~torch.utils.data.DataLoader` instance.
    """
    dataset = ReplayDataset(buffer, samples_per_epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
