"""Replay buffer reader for loading self-play training data in Python.

Reads MessagePack game files written by the Rust self-play worker. Each game
file contains a header and a list of serialized training samples in the format:

    {
        "header": {"version": 1, "num_samples": N},
        "samples": [
            {"fen": "...", "policy": [(index, prob), ...], "value": float},
            ...
        ]
    }

The buffer scans a directory of ``.msgpack`` game files and provides random
sampling of training positions for neural network training.

Usage::

    from training.buffer import ReplayBuffer

    buf = ReplayBuffer("/path/to/replay/data")
    buf.scan()
    boards, policies, values = buf.sample_batch(256)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import msgpack
import numpy as np


# Policy vector size: 8 * 8 * 73 = 4672
POLICY_SIZE: int = 4672


class ReplayBuffer:
    """Python reader for the self-play replay buffer.

    Scans a directory of .msgpack game files and provides random sampling
    of training positions for neural network training.

    Attributes:
        data_dir: Path to the directory containing .msgpack game files.
        capacity: Maximum number of game files to track (oldest evicted).
    """

    def __init__(self, data_dir: str, capacity: int = 500_000) -> None:
        """Initialize the replay buffer.

        Args:
            data_dir: Path to directory containing .msgpack game files.
            capacity: Max number of game files to keep track of. Oldest
                files (by filename sort order) are dropped when this limit
                is exceeded.
        """
        self.data_dir = Path(data_dir)
        self.capacity = capacity
        self._games_cache: List[Path] = []

    def scan(self) -> int:
        """Rescan the directory and return the number of game files found.

        Updates the internal cache of game file paths. Files are sorted
        by name (which includes timestamps, so oldest first). If the
        number of files exceeds capacity, only the newest ``capacity``
        files are kept in the cache.

        Returns:
            Number of game files found (after capacity trimming).
        """
        if not self.data_dir.exists():
            self._games_cache = []
            return 0

        files = sorted(
            p for p in self.data_dir.iterdir()
            if p.suffix == ".msgpack" and p.is_file()
        )

        # Keep only the newest files if over capacity
        if len(files) > self.capacity:
            files = files[-self.capacity:]

        self._games_cache = files
        return len(self._games_cache)

    def sample_positions(self, n: int) -> List[Dict]:
        """Sample n random training positions from the buffer.

        Each position is drawn by selecting a random game file and then
        a random position within that game. Games are selected with
        replacement.

        Returns:
            List of dicts with keys:
                - ``fen`` (str): Board position as a FEN string.
                - ``policy`` (list): List of ``(policy_index, probability)``
                  tuples.
                - ``value`` (float): Value target from side-to-move's
                  perspective.

        Raises:
            RuntimeError: If the buffer is empty (call :meth:`scan` first).
        """
        if not self._games_cache:
            raise RuntimeError(
                "No game files in buffer. Call scan() first or check data_dir."
            )

        results: List[Dict] = []
        max_attempts = n * 3
        attempts = 0

        while len(results) < n and attempts < max_attempts:
            attempts += 1

            # Pick a random game file
            game_path = random.choice(self._games_cache)

            try:
                samples = _read_game_file(game_path)
            except Exception:
                continue

            if not samples:
                continue

            # Pick a random position
            sample = random.choice(samples)
            results.append(sample)

        return results

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a training batch ready for PyTorch.

        Samples random positions and converts them into numpy arrays
        suitable for creating PyTorch tensors.

        Uses the ``neural.encoding`` module to convert FEN strings to
        the 119x8x8 tensor format expected by the neural network.

        Args:
            batch_size: Number of positions to sample.

        Returns:
            Tuple of:
                - ``boards``: ``(batch_size, 119, 8, 8)`` float32 array
                - ``policies``: ``(batch_size, 4672)`` float32 array
                - ``values``: ``(batch_size, 1)`` float32 array
        """
        from neural.encoding import BoardState, encode_board

        positions = self.sample_positions(batch_size)

        boards = np.zeros((len(positions), 119, 8, 8), dtype=np.float32)
        policies = np.zeros((len(positions), POLICY_SIZE), dtype=np.float32)
        values = np.zeros((len(positions), 1), dtype=np.float32)

        for i, pos in enumerate(positions):
            # Convert FEN to BoardState and encode
            state = BoardState.from_fen_piece_placement(pos["fen"])
            tensor = encode_board(state)
            boards[i] = tensor.numpy()

            # Fill the sparse policy into a dense vector
            for idx, prob in pos["policy"]:
                if 0 <= idx < POLICY_SIZE:
                    policies[i, idx] = prob

            values[i, 0] = pos["value"]

        return boards, policies, values


def _read_game_file(path: Path) -> List[Dict]:
    """Read a single MessagePack game file and return its samples.

    Args:
        path: Path to the .msgpack file.

    Returns:
        List of sample dicts with keys 'fen', 'policy', 'value'.

    Raises:
        ValueError: If the file format version is unsupported.
    """
    with open(path, "rb") as f:
        data = msgpack.unpackb(f.read(), raw=False)

    # The file is a map with "header" and "samples" keys
    header = data["header"]
    if header["version"] != 1:
        raise ValueError(
            f"Unsupported format version: {header['version']} (expected 1)"
        )

    samples = []
    for raw_sample in data["samples"]:
        samples.append({
            "fen": raw_sample["fen"],
            "policy": [
                (int(idx), float(prob))
                for idx, prob in raw_sample["policy"]
            ],
            "value": float(raw_sample["value"]),
        })

    return samples
