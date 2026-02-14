"""Distributed Data Parallel (DDP) utilities for multi-GPU training.

Provides helpers to initialize PyTorch DDP, wrap models, and coordinate
multi-process training. Designed for use with ``torchrun`` launcher.

The key principle: **only rank 0 saves checkpoints and logs metrics**.
All ranks participate in forward/backward passes and gradient sync.

Usage::

    # Launched by torchrun:
    #   torchrun --standalone --nproc_per_node=2 -m training.train --data-dir ./data

    from training.distributed import setup_ddp, cleanup_ddp, is_main_process

    setup_ddp()
    # ... training ...
    cleanup_ddp()
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(backend: str = "nccl") -> None:
    """Initialize the distributed process group.

    Must be called before any DDP operations. Uses environment variables
    set by ``torchrun`` (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR,
    MASTER_PORT) to configure the process group.

    Args:
        backend: Communication backend. ``"nccl"`` for GPU training
            (recommended), ``"gloo"`` for CPU or fallback.
    """
    if dist.is_initialized():
        return
    dist.init_process_group(backend=backend)
    # Set CUDA device to local rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def cleanup_ddp() -> None:
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Return the global rank of this process (0 if not distributed)."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Return the total number of processes (1 if not distributed)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """Return the local rank (GPU index on this node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Return True if this is rank 0 (the main process).

    Only the main process should:
    - Save checkpoints
    - Log metrics to TensorBoard/W&B
    - Print to console
    """
    return get_rank() == 0


def wrap_model_ddp(
    model: torch.nn.Module,
    device_id: int | None = None,
) -> DDP:
    """Wrap a model with DistributedDataParallel.

    Args:
        model: The model to wrap. Must already be on the correct GPU.
        device_id: CUDA device ID for this process. If None, uses
            LOCAL_RANK from environment.

    Returns:
        The DDP-wrapped model.
    """
    if device_id is None:
        device_id = get_local_rank()
    return DDP(model, device_ids=[device_id])


def get_device() -> torch.device:
    """Return the CUDA device for this rank, or CPU if CUDA unavailable."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_local_rank()}")
    return torch.device("cpu")
