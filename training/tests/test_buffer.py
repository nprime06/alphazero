"""Tests for the training replay buffer.

These tests verify:
1. Cross-language compatibility (Python can read Rust-written MessagePack)
2. sample_positions returns correct structure
3. ReplayBuffer.scan() counts files correctly
"""

import os
import tempfile
from pathlib import Path

import msgpack
import numpy as np
import pytest

from training.buffer import ReplayBuffer, _read_game_file, POLICY_SIZE


# =============================================================================
# Fixtures
# =============================================================================

# Starting position FEN
STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# A FEN after 1. e4
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def _make_sample(fen: str, policy_indices: list, value: float) -> dict:
    """Create a raw sample dict matching the Rust serialization format."""
    return {
        "fen": fen,
        "policy": [(idx, prob) for idx, prob in policy_indices],
        "value": value,
    }


def _write_msgpack_game_file(path: Path, samples: list, version: int = 1) -> None:
    """Write a game file in the same format as the Rust serializer.

    This manually creates the MessagePack format to test cross-language
    compatibility without needing the Rust binary.
    """
    game_file = {
        "header": {
            "version": version,
            "num_samples": len(samples),
        },
        "samples": samples,
    }
    with open(path, "wb") as f:
        f.write(msgpack.packb(game_file, use_bin_type=True))


@pytest.fixture
def buffer_dir(tmp_path):
    """Create a temporary directory with some game files for testing."""
    data_dir = tmp_path / "replay_data"
    data_dir.mkdir()

    # Write 3 game files
    for i in range(3):
        samples = [
            _make_sample(
                STARTING_FEN,
                [(877, 0.5), (1679, 0.3), (495, 0.2)],  # e2e4, d2d4, g1f3 indices
                1.0 if i % 2 == 0 else -1.0,
            ),
            _make_sample(
                AFTER_E4_FEN,
                [(877, 0.6), (1679, 0.4)],
                -1.0 if i % 2 == 0 else 1.0,
            ),
        ]
        path = data_dir / f"game_{1000000 + i:020d}_{i:08x}.msgpack"
        _write_msgpack_game_file(path, samples)

    return data_dir


@pytest.fixture
def empty_buffer_dir(tmp_path):
    """Create an empty temporary directory for testing."""
    data_dir = tmp_path / "empty_replay"
    data_dir.mkdir()
    return data_dir


# =============================================================================
# Test 1: Cross-language compatibility
# =============================================================================


class TestCrossLanguageCompat:
    """Test that Python can read the MessagePack format written by Rust."""

    def test_read_msgpack_game_file(self, tmp_path):
        """Write a msgpack file in the Rust format and verify Python reads it."""
        samples = [
            _make_sample(
                STARTING_FEN,
                [(877, 0.5), (1679, 0.3), (495, 0.2)],
                1.0,
            ),
            _make_sample(
                AFTER_E4_FEN,
                [(877, 0.7), (1679, 0.3)],
                -1.0,
            ),
        ]

        path = tmp_path / "test_game.msgpack"
        _write_msgpack_game_file(path, samples)

        # Read it back
        loaded = _read_game_file(path)

        assert len(loaded) == 2, "Should load 2 samples"

        # First sample
        assert loaded[0]["fen"] == STARTING_FEN
        assert len(loaded[0]["policy"]) == 3
        assert loaded[0]["policy"][0] == (877, 0.5)
        assert loaded[0]["policy"][1] == (1679, 0.3)
        assert loaded[0]["policy"][2] == (495, 0.2)
        assert loaded[0]["value"] == 1.0

        # Second sample
        assert loaded[1]["fen"] == AFTER_E4_FEN
        assert len(loaded[1]["policy"]) == 2
        assert loaded[1]["value"] == -1.0

    def test_read_msgpack_preserves_fen(self, tmp_path):
        """FEN strings are preserved exactly through serialization."""
        fens = [
            STARTING_FEN,
            AFTER_E4_FEN,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        ]
        for fen in fens:
            samples = [_make_sample(fen, [(0, 1.0)], 0.0)]
            path = tmp_path / "fen_test.msgpack"
            _write_msgpack_game_file(path, samples)
            loaded = _read_game_file(path)
            assert loaded[0]["fen"] == fen, f"FEN should be preserved: {fen}"

    def test_read_rejects_wrong_version(self, tmp_path):
        """Files with wrong version are rejected."""
        samples = [_make_sample(STARTING_FEN, [(0, 1.0)], 0.0)]
        path = tmp_path / "bad_version.msgpack"
        _write_msgpack_game_file(path, samples, version=999)

        with pytest.raises(ValueError, match="Unsupported format version"):
            _read_game_file(path)

    def test_read_empty_samples(self, tmp_path):
        """A game file with zero samples is valid."""
        path = tmp_path / "empty_game.msgpack"
        _write_msgpack_game_file(path, [])
        loaded = _read_game_file(path)
        assert loaded == []

    def test_policy_index_types(self, tmp_path):
        """Policy indices and probabilities have correct types."""
        samples = [
            _make_sample(STARTING_FEN, [(42, 0.75), (100, 0.25)], 0.5)
        ]
        path = tmp_path / "types_test.msgpack"
        _write_msgpack_game_file(path, samples)
        loaded = _read_game_file(path)

        for idx, prob in loaded[0]["policy"]:
            assert isinstance(idx, int), f"Index should be int, got {type(idx)}"
            assert isinstance(prob, float), f"Prob should be float, got {type(prob)}"

        assert isinstance(loaded[0]["value"], float)


# =============================================================================
# Test 2: sample_positions returns correct structure
# =============================================================================


class TestSamplePositions:
    """Test the sample_positions method."""

    def test_returns_correct_count(self, buffer_dir):
        buf = ReplayBuffer(str(buffer_dir))
        buf.scan()
        positions = buf.sample_positions(5)
        assert len(positions) == 5, f"Expected 5 positions, got {len(positions)}"

    def test_sample_has_required_keys(self, buffer_dir):
        buf = ReplayBuffer(str(buffer_dir))
        buf.scan()
        positions = buf.sample_positions(3)
        for pos in positions:
            assert "fen" in pos, "Sample should have 'fen' key"
            assert "policy" in pos, "Sample should have 'policy' key"
            assert "value" in pos, "Sample should have 'value' key"

    def test_fen_is_string(self, buffer_dir):
        buf = ReplayBuffer(str(buffer_dir))
        buf.scan()
        positions = buf.sample_positions(3)
        for pos in positions:
            assert isinstance(pos["fen"], str), "FEN should be a string"
            assert len(pos["fen"]) > 0, "FEN should not be empty"

    def test_policy_is_list_of_tuples(self, buffer_dir):
        buf = ReplayBuffer(str(buffer_dir))
        buf.scan()
        positions = buf.sample_positions(3)
        for pos in positions:
            assert isinstance(pos["policy"], list), "Policy should be a list"
            assert len(pos["policy"]) > 0, "Policy should not be empty"
            for idx, prob in pos["policy"]:
                assert isinstance(idx, int), "Policy index should be int"
                assert isinstance(prob, float), "Policy prob should be float"
                assert 0 <= idx < POLICY_SIZE, (
                    f"Policy index {idx} out of range [0, {POLICY_SIZE})"
                )
                assert 0.0 <= prob <= 1.0, (
                    f"Policy prob {prob} out of range [0, 1]"
                )

    def test_value_is_float(self, buffer_dir):
        buf = ReplayBuffer(str(buffer_dir))
        buf.scan()
        positions = buf.sample_positions(3)
        for pos in positions:
            assert isinstance(pos["value"], float), "Value should be float"
            assert -1.0 <= pos["value"] <= 1.0, (
                f"Value {pos['value']} out of expected range [-1, 1]"
            )

    def test_empty_buffer_raises(self, empty_buffer_dir):
        buf = ReplayBuffer(str(empty_buffer_dir))
        buf.scan()
        with pytest.raises(RuntimeError, match="No game files"):
            buf.sample_positions(5)


# =============================================================================
# Test 3: ReplayBuffer.scan() counts files correctly
# =============================================================================


class TestScan:
    """Test the scan method."""

    def test_scan_counts_files(self, buffer_dir):
        buf = ReplayBuffer(str(buffer_dir))
        count = buf.scan()
        assert count == 3, f"Should find 3 game files, got {count}"

    def test_scan_empty_directory(self, empty_buffer_dir):
        buf = ReplayBuffer(str(empty_buffer_dir))
        count = buf.scan()
        assert count == 0, "Empty directory should have 0 files"

    def test_scan_nonexistent_directory(self, tmp_path):
        buf = ReplayBuffer(str(tmp_path / "nonexistent"))
        count = buf.scan()
        assert count == 0, "Nonexistent directory should report 0 files"

    def test_scan_ignores_non_msgpack_files(self, buffer_dir):
        # Add some non-msgpack files
        (buffer_dir / "readme.txt").write_text("not a game")
        (buffer_dir / "data.json").write_text("{}")
        (buffer_dir / "image.png").write_bytes(b"\x89PNG")

        buf = ReplayBuffer(str(buffer_dir))
        count = buf.scan()
        assert count == 3, f"Should ignore non-msgpack files, got {count}"

    def test_scan_respects_capacity(self, tmp_path):
        data_dir = tmp_path / "capacity_test"
        data_dir.mkdir()

        # Write 10 game files
        for i in range(10):
            samples = [_make_sample(STARTING_FEN, [(0, 1.0)], 0.0)]
            path = data_dir / f"game_{i:020d}_{0:08x}.msgpack"
            _write_msgpack_game_file(path, samples)

        buf = ReplayBuffer(str(data_dir), capacity=5)
        count = buf.scan()
        assert count == 5, f"Should respect capacity of 5, got {count}"

    def test_scan_updates_cache(self, buffer_dir):
        buf = ReplayBuffer(str(buffer_dir))

        # First scan
        count1 = buf.scan()
        assert count1 == 3

        # Add another file
        samples = [_make_sample(STARTING_FEN, [(0, 1.0)], 0.0)]
        path = buffer_dir / f"game_{9999999999:020d}_{0:08x}.msgpack"
        _write_msgpack_game_file(path, samples)

        # Second scan should find the new file
        count2 = buf.scan()
        assert count2 == 4, f"Should find 4 files after adding one, got {count2}"


# =============================================================================
# Test 4: sample_batch (if neural encoding is available)
# =============================================================================


class TestSampleBatch:
    """Test the sample_batch method.

    These tests require the neural encoding module to be importable.
    They are skipped if it is not available.
    """

    @pytest.fixture(autouse=True)
    def _check_neural(self):
        """Skip tests if neural.encoding is not available."""
        try:
            from neural.encoding import BoardState, encode_board  # noqa: F401
        except ImportError:
            pytest.skip("neural.encoding not available")

    def test_sample_batch_shapes(self, buffer_dir):
        buf = ReplayBuffer(str(buffer_dir))
        buf.scan()
        boards, policies, values = buf.sample_batch(4)

        assert boards.shape == (4, 119, 8, 8), f"boards shape: {boards.shape}"
        assert policies.shape == (4, POLICY_SIZE), f"policies shape: {policies.shape}"
        assert values.shape == (4, 1), f"values shape: {values.shape}"

    def test_sample_batch_dtypes(self, buffer_dir):
        buf = ReplayBuffer(str(buffer_dir))
        buf.scan()
        boards, policies, values = buf.sample_batch(2)

        assert boards.dtype == np.float32, f"boards dtype: {boards.dtype}"
        assert policies.dtype == np.float32, f"policies dtype: {policies.dtype}"
        assert values.dtype == np.float32, f"values dtype: {values.dtype}"

    def test_sample_batch_policy_sums(self, buffer_dir):
        """Policy vectors should approximately sum to 1.0."""
        buf = ReplayBuffer(str(buffer_dir))
        buf.scan()
        _, policies, _ = buf.sample_batch(3)

        for i in range(policies.shape[0]):
            policy_sum = policies[i].sum()
            assert abs(policy_sum - 1.0) < 0.01, (
                f"Policy {i} sum {policy_sum} should be ~1.0"
            )

    def test_sample_batch_values_in_range(self, buffer_dir):
        """Values should be in [-1, 1]."""
        buf = ReplayBuffer(str(buffer_dir))
        buf.scan()
        _, _, values = buf.sample_batch(5)

        assert np.all(values >= -1.0), "All values should be >= -1"
        assert np.all(values <= 1.0), "All values should be <= 1"
