"""Replay buffer reader for loading self-play training data in Python.

Reads MessagePack game files written by the Rust self-play worker. Each game
file contains a header and a list of serialized training samples in the format:

    {
        "header": {"version": 1 or 2, "num_samples": N},
        "samples": [
            {
                "fen": "...",
                "history_fens": ["...", ...],  # v2, optional
                "policy": [(index, prob), ...],
                "value": float,
            },
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
SUPPORTED_FORMAT_VERSIONS = {1, 2}
MAX_HISTORY_FENS: int = 7


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
        from neural.encoding import encode_board

        positions = self.sample_positions(batch_size)

        boards = np.zeros((len(positions), 119, 8, 8), dtype=np.float32)
        policies = np.zeros((len(positions), POLICY_SIZE), dtype=np.float32)
        values = np.zeros((len(positions), 1), dtype=np.float32)

        for i, pos in enumerate(positions):
            # Convert FEN to BoardState and encode
            state = board_state_from_sample(pos)
            tensor = encode_board(state)
            boards[i] = tensor.numpy()

            # Fill the sparse policy into a dense vector
            for idx, prob in pos["policy"]:
                if 0 <= idx < POLICY_SIZE:
                    policies[i, idx] = prob

            values[i, 0] = pos["value"]

        return boards, policies, values


def board_state_from_sample(sample: Dict):
    """Build a neural ``BoardState`` from a replay sample.

    Version 2 replay samples include up to seven previous positions as FEN
    strings, most recent first. Older version 1 samples omit this field and
    therefore encode with empty history, preserving backward compatibility.
    Repetition counts are reconstructed from the available FEN history.
    """
    from neural.encoding import BoardState

    history_fens = list(sample.get("history_fens", []))[:MAX_HISTORY_FENS]

    history_states = []
    for idx, fen in enumerate(history_fens):
        repetition_count = _count_repetitions(fen, history_fens[idx + 1:])
        history_states.append(
            BoardState.from_fen_piece_placement(
                fen,
                repetition_count=repetition_count,
            )
        )

    current_repetition_count = _count_repetitions(sample["fen"], history_fens)
    return BoardState.from_fen_piece_placement(
        sample["fen"],
        repetition_count=current_repetition_count,
        history=history_states,
    )


def _repetition_key(fen: str) -> str:
    """Return the FEN fields that define repetition identity."""
    parts = fen.strip().split()
    if len(parts) >= 4:
        return " ".join(parts[:4])
    return parts[0] if parts else ""


def _count_repetitions(fen: str, previous_fens: List[str]) -> int:
    """Count prior occurrences of ``fen`` in a most-recent-first history."""
    key = _repetition_key(fen)
    return sum(1 for previous in previous_fens if _repetition_key(previous) == key)


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

    header, raw_samples = _decode_game_file(data)
    version = int(header["version"])
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported format version: {version} "
            f"(supported: {sorted(SUPPORTED_FORMAT_VERSIONS)})"
        )

    samples = []
    for raw_sample in raw_samples:
        sample = _decode_sample(raw_sample)
        samples.append({
            "fen": sample["fen"],
            "history_fens": [
                str(fen)
                for fen in sample.get("history_fens", [])[:MAX_HISTORY_FENS]
            ],
            "policy": [
                (int(idx), float(prob))
                for idx, prob in sample["policy"]
            ],
            "value": float(sample["value"]),
        })

    return samples


def _decode_game_file(data) -> Tuple[Dict, List]:
    """Decode named-map or tuple-shaped Rust MessagePack game files."""
    if isinstance(data, dict):
        return data["header"], data["samples"]
    if isinstance(data, (list, tuple)) and len(data) == 2:
        header = _decode_header(data[0])
        samples = list(data[1])
        return header, samples
    raise ValueError("Malformed replay file: expected game file map or tuple")


def _decode_header(raw_header) -> Dict:
    """Decode a GameFileHeader from map or tuple representation."""
    if isinstance(raw_header, dict):
        return {
            "version": raw_header["version"],
            "num_samples": raw_header["num_samples"],
        }
    if isinstance(raw_header, (list, tuple)) and len(raw_header) == 2:
        return {
            "version": raw_header[0],
            "num_samples": raw_header[1],
        }
    raise ValueError("Malformed replay file: expected header map or tuple")


def _decode_sample(raw_sample) -> Dict:
    """Decode a SerializedSample from map, v2 tuple, or legacy v1 tuple."""
    if isinstance(raw_sample, dict):
        return raw_sample
    if isinstance(raw_sample, (list, tuple)):
        if len(raw_sample) == 4:
            fen, history_fens, policy, value = raw_sample
            return {
                "fen": fen,
                "history_fens": history_fens,
                "policy": policy,
                "value": value,
            }
        if len(raw_sample) == 3:
            fen, policy, value = raw_sample
            return {
                "fen": fen,
                "history_fens": [],
                "policy": policy,
                "value": value,
            }
    raise ValueError("Malformed replay file: expected sample map or tuple")
