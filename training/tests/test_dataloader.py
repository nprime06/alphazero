"""Tests for the data loading utilities.

Covers both :class:`DummyDataset` (always available) and
:class:`ReplayDataset` (requires game data and neural encoding).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

import pytest

from training.dataloader import (
    DummyDataset,
    ReplayDataset,
    create_dataloader,
    INPUT_PLANES,
    POLICY_SIZE,
)


# ============================================================================
# DummyDataset tests
# ============================================================================


class TestDummyDataset:
    """Tests for DummyDataset (random data generator)."""

    def test_shapes(self):
        """Each sample has the correct shapes: (119,8,8), (4672,), (1,)."""
        ds = DummyDataset(size=5)
        board, policy, value = ds[0]

        assert board.shape == (INPUT_PLANES, 8, 8), (
            f"Board shape should be ({INPUT_PLANES}, 8, 8), got {board.shape}"
        )
        assert policy.shape == (POLICY_SIZE,), (
            f"Policy shape should be ({POLICY_SIZE},), got {policy.shape}"
        )
        assert value.shape == (1,), (
            f"Value shape should be (1,), got {value.shape}"
        )

    def test_dtypes(self):
        """All tensors are float32."""
        ds = DummyDataset(size=5)
        board, policy, value = ds[0]

        assert board.dtype == torch.float32
        assert policy.dtype == torch.float32
        assert value.dtype == torch.float32

    def test_policy_sums_to_one(self):
        """Policy is softmax-normalized, so it should sum to ~1.0."""
        ds = DummyDataset(size=10)
        for i in range(min(10, len(ds))):
            _, policy, _ = ds[i]
            policy_sum = policy.sum().item()
            assert abs(policy_sum - 1.0) < 1e-5, (
                f"Policy sum should be ~1.0, got {policy_sum}"
            )

    def test_values_in_range(self):
        """Values should be in [-1, 1]."""
        ds = DummyDataset(size=50)
        for i in range(len(ds)):
            _, _, value = ds[i]
            v = value.item()
            assert -1.0 <= v <= 1.0, f"Value {v} out of range [-1, 1]"

    def test_len_matches_size(self):
        """__len__ returns the configured size."""
        for size in [1, 100, 9999]:
            ds = DummyDataset(size=size)
            assert len(ds) == size, f"Expected len={size}, got {len(ds)}"

    def test_default_size(self):
        """Default size is 10000."""
        ds = DummyDataset()
        assert len(ds) == 10_000

    def test_dataloader_batches(self):
        """DataLoader produces batches with correct shapes."""
        batch_size = 16
        ds = DummyDataset(size=64)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        batch = next(iter(loader))
        boards, policies, values = batch

        assert boards.shape == (batch_size, INPUT_PLANES, 8, 8), (
            f"Batch boards shape: {boards.shape}"
        )
        assert policies.shape == (batch_size, POLICY_SIZE), (
            f"Batch policies shape: {policies.shape}"
        )
        assert values.shape == (batch_size, 1), (
            f"Batch values shape: {values.shape}"
        )


# ============================================================================
# ReplayDataset tests (require game data + neural encoding)
# ============================================================================


class TestReplayDataset:
    """Tests for ReplayDataset with a real replay buffer.

    These tests create a small buffer with synthetic game files and verify
    that the dataset yields correctly shaped tensors.
    """

    @pytest.fixture(autouse=True)
    def _check_neural(self):
        """Skip if neural.encoding is not importable."""
        try:
            from neural.encoding import BoardState, encode_board  # noqa: F401
        except ImportError:
            pytest.skip("neural.encoding not available")

    @pytest.fixture
    def buffer_with_data(self, tmp_path):
        """Create a ReplayBuffer with a few synthetic game files."""
        import msgpack
        from training.buffer import ReplayBuffer

        data_dir = tmp_path / "replay"
        data_dir.mkdir()

        starting_fen = (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )

        for i in range(3):
            game_file = {
                "header": {"version": 1, "num_samples": 2},
                "samples": [
                    {
                        "fen": starting_fen,
                        "policy": [(877, 0.5), (1679, 0.3), (495, 0.2)],
                        "value": 1.0 if i % 2 == 0 else -1.0,
                    },
                    {
                        "fen": starting_fen,
                        "policy": [(877, 0.6), (1679, 0.4)],
                        "value": -1.0 if i % 2 == 0 else 1.0,
                    },
                ],
            }
            path = data_dir / f"game_{i:020d}.msgpack"
            with open(path, "wb") as f:
                f.write(msgpack.packb(game_file, use_bin_type=True))

        buf = ReplayBuffer(str(data_dir))
        buf.scan()
        return buf

    def test_yields_correct_shapes(self, buffer_with_data):
        """Each sample from the dataset has the expected shapes."""
        ds = ReplayDataset(buffer_with_data, samples_per_epoch=5)
        sample = next(iter(ds))
        board, policy, value = sample

        assert board.shape == (INPUT_PLANES, 8, 8)
        assert policy.shape == (POLICY_SIZE,)
        assert value.shape == (1,)

    def test_yields_correct_dtypes(self, buffer_with_data):
        """Tensors are float32."""
        ds = ReplayDataset(buffer_with_data, samples_per_epoch=3)
        board, policy, value = next(iter(ds))

        assert board.dtype == torch.float32
        assert policy.dtype == torch.float32
        assert value.dtype == torch.float32

    def test_len(self, buffer_with_data):
        """__len__ returns samples_per_epoch."""
        ds = ReplayDataset(buffer_with_data, samples_per_epoch=42)
        assert len(ds) == 42


# ============================================================================
# create_dataloader tests
# ============================================================================


class TestCreateDataloader:
    """Tests for the create_dataloader factory function."""

    @pytest.fixture(autouse=True)
    def _check_neural(self):
        """Skip if neural.encoding is not importable."""
        try:
            from neural.encoding import BoardState, encode_board  # noqa: F401
        except ImportError:
            pytest.skip("neural.encoding not available")

    @pytest.fixture
    def buffer_with_data(self, tmp_path):
        """Create a ReplayBuffer with synthetic game files."""
        import msgpack
        from training.buffer import ReplayBuffer

        data_dir = tmp_path / "replay"
        data_dir.mkdir()

        starting_fen = (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )

        for i in range(3):
            game_file = {
                "header": {"version": 1, "num_samples": 1},
                "samples": [
                    {
                        "fen": starting_fen,
                        "policy": [(877, 0.5), (1679, 0.3), (495, 0.2)],
                        "value": 1.0,
                    },
                ],
            }
            path = data_dir / f"game_{i:020d}.msgpack"
            with open(path, "wb") as f:
                f.write(msgpack.packb(game_file, use_bin_type=True))

        buf = ReplayBuffer(str(data_dir))
        buf.scan()
        return buf

    def test_returns_dataloader(self, buffer_with_data):
        """create_dataloader returns a DataLoader instance."""
        loader = create_dataloader(
            buffer_with_data,
            batch_size=4,
            samples_per_epoch=20,
            num_workers=0,
        )
        assert isinstance(loader, DataLoader)

    def test_batch_shapes(self, buffer_with_data):
        """Batches from create_dataloader have correct shapes."""
        batch_size = 4
        loader = create_dataloader(
            buffer_with_data,
            batch_size=batch_size,
            samples_per_epoch=20,
            num_workers=0,
        )
        batch = next(iter(loader))
        boards, policies, values = batch

        assert boards.shape == (batch_size, INPUT_PLANES, 8, 8)
        assert policies.shape == (batch_size, POLICY_SIZE)
        assert values.shape == (batch_size, 1)
