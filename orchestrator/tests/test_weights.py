"""Tests for the weight distribution module."""

from __future__ import annotations

import torch

from neural.config import NetworkConfig
from neural.network import AlphaZeroNetwork
from orchestrator.weights import WeightPublisher, WeightWatcher


def _make_model() -> AlphaZeroNetwork:
    """Create a tiny AlphaZero network on CPU for testing."""
    config = NetworkConfig.tiny()
    return AlphaZeroNetwork(config)


# ---------------------------------------------------------------------------
# TestWeightPublisher
# ---------------------------------------------------------------------------


class TestWeightPublisher:
    def test_init_creates_dir(self, tmp_path):
        weights_dir = tmp_path / "weights"
        WeightPublisher(str(weights_dir))
        assert weights_dir.is_dir()

    def test_publish_creates_model_file(self, tmp_path):
        publisher = WeightPublisher(str(tmp_path))
        model = _make_model()
        path = publisher.publish(model, step=100)
        assert path.exists()
        assert path.name == "model_v000001.pt"

    def test_publish_creates_latest_txt(self, tmp_path):
        publisher = WeightPublisher(str(tmp_path))
        model = _make_model()
        publisher.publish(model, step=100)
        latest_file = tmp_path / "latest.txt"
        assert latest_file.exists()
        assert latest_file.read_text().strip() == "1"

    def test_publish_increments_version(self, tmp_path):
        publisher = WeightPublisher(str(tmp_path))
        model = _make_model()
        publisher.publish(model, step=100)
        publisher.publish(model, step=200)
        assert publisher.current_version == 2
        latest_file = tmp_path / "latest.txt"
        assert latest_file.read_text().strip() == "2"

    def test_publish_model_is_loadable(self, tmp_path):
        publisher = WeightPublisher(str(tmp_path))
        model = _make_model()
        path = publisher.publish(model, step=100)
        loaded = torch.jit.load(str(path))
        assert loaded is not None

    def test_publish_model_output_shapes(self, tmp_path):
        publisher = WeightPublisher(str(tmp_path))
        model = _make_model()
        path = publisher.publish(model, step=100)
        loaded = torch.jit.load(str(path))
        dummy = torch.randn(2, 119, 8, 8)
        policy, value = loaded(dummy)
        assert policy.shape == (2, 4672)
        assert value.shape == (2, 1)

    def test_cleanup_keeps_n(self, tmp_path):
        publisher = WeightPublisher(str(tmp_path), keep_n=3)
        model = _make_model()
        for i in range(7):
            publisher.publish(model, step=i * 100)
        model_files = sorted(tmp_path.glob("model_v*.pt"))
        assert len(model_files) == 3

    def test_cleanup_keeps_newest(self, tmp_path):
        publisher = WeightPublisher(str(tmp_path), keep_n=3)
        model = _make_model()
        for i in range(7):
            publisher.publish(model, step=i * 100)
        model_files = sorted(tmp_path.glob("model_v*.pt"))
        names = [f.name for f in model_files]
        assert names == [
            "model_v000005.pt",
            "model_v000006.pt",
            "model_v000007.pt",
        ]

    def test_version_survives_restart(self, tmp_path):
        model = _make_model()
        publisher1 = WeightPublisher(str(tmp_path))
        publisher1.publish(model, step=100)
        publisher1.publish(model, step=200)
        assert publisher1.current_version == 2

        # Create a brand-new publisher on the same directory
        publisher2 = WeightPublisher(str(tmp_path))
        assert publisher2.current_version == 2
        publisher2.publish(model, step=300)
        assert publisher2.current_version == 3


# ---------------------------------------------------------------------------
# TestWeightWatcher
# ---------------------------------------------------------------------------


class TestWeightWatcher:
    def test_no_model_version_zero(self, tmp_path):
        watcher = WeightWatcher(str(tmp_path))
        assert watcher.get_latest_version() == 0

    def test_detects_new_version(self, tmp_path):
        publisher = WeightPublisher(str(tmp_path))
        watcher = WeightWatcher(str(tmp_path))
        model = _make_model()
        publisher.publish(model, step=100)
        assert watcher.has_new_version() is True

    def test_load_latest(self, tmp_path):
        publisher = WeightPublisher(str(tmp_path))
        watcher = WeightWatcher(str(tmp_path))
        model = _make_model()
        publisher.publish(model, step=100)
        loaded = watcher.load_latest()
        assert isinstance(loaded, torch.jit.ScriptModule)
        # Verify it produces output
        dummy = torch.randn(1, 119, 8, 8)
        policy, value = loaded(dummy)
        assert policy.shape == (1, 4672)
        assert value.shape == (1, 1)

    def test_load_updates_tracker(self, tmp_path):
        publisher = WeightPublisher(str(tmp_path))
        watcher = WeightWatcher(str(tmp_path))
        model = _make_model()
        publisher.publish(model, step=100)
        assert watcher.has_new_version() is True
        watcher.load_latest()
        assert watcher.has_new_version() is False

    def test_detects_second_version(self, tmp_path):
        publisher = WeightPublisher(str(tmp_path))
        watcher = WeightWatcher(str(tmp_path))
        model = _make_model()

        publisher.publish(model, step=100)
        watcher.load_latest()
        assert watcher.has_new_version() is False

        publisher.publish(model, step=200)
        assert watcher.has_new_version() is True

    def test_model_path(self, tmp_path):
        watcher = WeightWatcher(str(tmp_path))
        expected = tmp_path / "model_v000003.pt"
        assert watcher.model_path(3) == expected


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_publish_and_watch_cycle(self, tmp_path):
        """Full cycle: publish v1, watcher detects, loads, publish v2, detects again."""
        publisher = WeightPublisher(str(tmp_path))
        watcher = WeightWatcher(str(tmp_path))
        model = _make_model()

        # No model yet
        assert watcher.has_new_version() is False
        assert watcher.get_latest_version() == 0

        # Publish v1
        path1 = publisher.publish(model, step=1000)
        assert publisher.current_version == 1
        assert watcher.has_new_version() is True
        assert watcher.get_latest_version() == 1

        # Load v1
        loaded1 = watcher.load_latest()
        assert watcher.has_new_version() is False
        dummy = torch.randn(1, 119, 8, 8)
        policy1, value1 = loaded1(dummy)
        assert policy1.shape == (1, 4672)
        assert value1.shape == (1, 1)

        # Publish v2
        path2 = publisher.publish(model, step=2000)
        assert publisher.current_version == 2
        assert watcher.has_new_version() is True
        assert watcher.get_latest_version() == 2

        # Load v2
        loaded2 = watcher.load_latest()
        assert watcher.has_new_version() is False
        policy2, value2 = loaded2(dummy)
        assert policy2.shape == (1, 4672)
        assert value2.shape == (1, 1)
