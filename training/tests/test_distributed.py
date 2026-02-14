"""Tests for the distributed training utilities.

These tests verify the DDP helper functions in a non-distributed context
(single process, CPU). Actual multi-GPU testing requires a cluster.
"""

from __future__ import annotations

import inspect

import torch
import torch.distributed as dist

import pytest

from training.distributed import (
    cleanup_ddp,
    get_device,
    get_local_rank,
    get_rank,
    get_world_size,
    is_main_process,
    setup_ddp,
    wrap_model_ddp,
)


# ============================================================================
# Helper function tests (non-distributed context)
# ============================================================================


class TestDistributedHelpers:
    """Test DDP helper functions when no process group is initialized."""

    def test_get_rank_without_init(self):
        """get_rank() returns 0 when dist is not initialized."""
        assert not dist.is_initialized()
        assert get_rank() == 0

    def test_get_world_size_without_init(self):
        """get_world_size() returns 1 when dist is not initialized."""
        assert not dist.is_initialized()
        assert get_world_size() == 1

    def test_is_main_process_without_init(self):
        """is_main_process() returns True when not distributed."""
        assert not dist.is_initialized()
        assert is_main_process() is True

    def test_get_local_rank_default(self, monkeypatch):
        """get_local_rank() returns 0 when LOCAL_RANK is not set."""
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert get_local_rank() == 0

    def test_get_local_rank_from_env(self, monkeypatch):
        """get_local_rank() reads LOCAL_RANK from the environment."""
        monkeypatch.setenv("LOCAL_RANK", "3")
        assert get_local_rank() == 3

    def test_get_device_cpu(self):
        """get_device() returns cpu when CUDA is not available (test machine)."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available; cannot test CPU fallback")
        device = get_device()
        assert device == torch.device("cpu")

    def test_cleanup_without_init(self):
        """cleanup_ddp() does not raise when no process group is initialized."""
        assert not dist.is_initialized()
        cleanup_ddp()  # should be a no-op

    def test_setup_is_callable(self):
        """setup_ddp exists and has the expected signature."""
        sig = inspect.signature(setup_ddp)
        assert "backend" in sig.parameters
        assert sig.parameters["backend"].default == "nccl"


# ============================================================================
# wrap_model_ddp tests
# ============================================================================


class TestWrapModelDDP:
    """Tests for the DDP model wrapper."""

    def test_wrap_model_signature(self):
        """wrap_model_ddp has the expected parameters."""
        sig = inspect.signature(wrap_model_ddp)
        assert "model" in sig.parameters
        assert "device_id" in sig.parameters

    def test_wrap_model_requires_init(self):
        """Wrapping a model without dist init raises an error."""
        assert not dist.is_initialized()
        model = torch.nn.Linear(10, 10)
        with pytest.raises((RuntimeError, ValueError)):
            wrap_model_ddp(model, device_id=0)
