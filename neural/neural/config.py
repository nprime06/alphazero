"""
Network Configuration
=====================

Centralizes all configurable parameters for the AlphaZero neural network.

Why a dedicated config module?
    The AlphaZero network architecture is parameterized by several values that
    affect model capacity, training speed, and memory usage. Keeping them in a
    single dataclass makes it easy to:
    - Swap between presets (TINY for debugging, FULL for real training)
    - Serialize/deserialize configs alongside model checkpoints
    - Estimate resource requirements before committing to a training run

The defaults match the original DeepMind AlphaZero paper:
    - 19 residual blocks (sometimes cited as 20; 19 blocks + 1 input conv)
    - 256 filters per convolutional layer
    - 119 input planes (8 history steps x 14 planes + 7 auxiliary planes)
    - 4672 policy outputs (8x8 board x 73 move types)

Preset Configurations:
    +---------+--------+---------+---------+----------------------------------+
    | Preset  | Blocks | Filters | ~Params | Use Case                         |
    +---------+--------+---------+---------+----------------------------------+
    | TINY    |   5    |    64   |   ~1M   | Debugging, unit tests, learning  |
    | SMALL   |  10    |   128   |   ~4M   | Overnight training runs          |
    | MEDIUM  |  15    |   192   |  ~11M   | Multi-day training               |
    | FULL    |  19    |   256   |  ~23M   | Matches original paper           |
    +---------+--------+---------+---------+----------------------------------+
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(frozen=True)
class NetworkConfig:
    """Immutable configuration for the AlphaZero neural network.

    All network dimensions are derived from these parameters, so changing them
    here automatically propagates to every layer in the network. The dataclass
    is frozen (immutable) because a config should never change after the network
    is constructed -- doing so would silently break the correspondence between
    the config and the actual model weights.

    Attributes:
        num_blocks: Number of residual blocks in the network trunk. More blocks
            increase the network's capacity to represent complex positional
            features, but also increase compute and memory cost. The original
            AlphaZero paper uses 19 (sometimes cited as 20, counting the
            initial convolution). Valid range: 1-40.
        num_filters: Number of convolutional filters (channels) in each layer.
            This is the "width" of the network. Wider networks can represent
            more features per layer but are more expensive. The original paper
            uses 256. Valid range: 32-512.
        input_planes: Number of input feature planes. For standard AlphaZero
            chess encoding with 8 history steps: 8 * (6 piece types * 2 colors)
            + 2 repetition planes per step = 8 * 14 + 7 auxiliary = 119 planes.
            You generally should not change this unless you change the board
            encoding scheme.
        policy_output_size: Size of the policy head output vector. For chess
            this is 8*8*73 = 4672, representing all possible moves encoded as
            (source square) x (move type). The 73 move types are: 56 queen
            moves (7 distances x 8 directions), 8 knight moves, and 9
            underpromotions (3 piece types x 3 directions).
        value_hidden_size: Number of neurons in the value head's hidden layer.
            The value head is: conv1x1 -> flatten -> Linear(64, hidden) -> ReLU
            -> Linear(hidden, 1) -> tanh. The hidden size controls the capacity
            of the value prediction. Default 256 matches the original paper.
        device: The torch device to use for computation. If None (default),
            auto-detects CUDA availability and uses GPU if present.
    """

    num_blocks: int = 19
    num_filters: int = 256
    input_planes: int = 119
    policy_output_size: int = 4672
    value_hidden_size: int = 256
    device: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate all configuration parameters.

        Raises ValueError with a descriptive message if any parameter is out
        of its valid range. This catches configuration errors early, before
        the network is constructed.
        """
        if not (1 <= self.num_blocks <= 40):
            raise ValueError(
                f"num_blocks must be between 1 and 40, got {self.num_blocks}. "
                f"Values above 40 cause diminishing returns and training instability."
            )
        if not (32 <= self.num_filters <= 512):
            raise ValueError(
                f"num_filters must be between 32 and 512, got {self.num_filters}. "
                f"Values below 32 lack capacity; above 512 are impractically expensive."
            )
        if self.input_planes < 1:
            raise ValueError(
                f"input_planes must be positive, got {self.input_planes}."
            )
        if self.policy_output_size < 1:
            raise ValueError(
                f"policy_output_size must be positive, got {self.policy_output_size}."
            )
        if self.value_hidden_size < 1:
            raise ValueError(
                f"value_hidden_size must be positive, got {self.value_hidden_size}."
            )

    # -------------------------------------------------------------------------
    # Preset factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def tiny(cls) -> NetworkConfig:
        """Create a tiny network configuration for debugging and fast experiments.

        5 blocks, 64 filters, ~1M parameters. This is small enough to train
        on a CPU in minutes, making it ideal for:
        - Unit tests that need a real network but don't care about strength
        - Debugging the training pipeline end-to-end
        - Learning how the architecture works by inspecting activations
        """
        return cls(num_blocks=5, num_filters=64)

    @classmethod
    def small(cls) -> NetworkConfig:
        """Create a small network for overnight training runs.

        10 blocks, 128 filters, ~4M parameters. Large enough to learn basic
        chess patterns (material value, simple tactics) but trains in hours
        rather than days on a single GPU.
        """
        return cls(num_blocks=10, num_filters=128)

    @classmethod
    def medium(cls) -> NetworkConfig:
        """Create a medium network for multi-day training.

        15 blocks, 192 filters, ~11M parameters. A good balance between
        capacity and training cost. Suitable for serious experiments where
        you want meaningful strength but can't afford the full 19-block
        network's compute requirements.
        """
        return cls(num_blocks=15, num_filters=192)

    @classmethod
    def full(cls) -> NetworkConfig:
        """Create the full-size network matching the original AlphaZero paper.

        19 blocks, 256 filters, ~23M parameters. This is the architecture
        DeepMind used to achieve superhuman chess play. Expect multi-day
        training on multiple GPUs to reach strong play.
        """
        return cls(num_blocks=19, num_filters=256)

    # -------------------------------------------------------------------------
    # Device handling
    # -------------------------------------------------------------------------

    def get_device(self) -> torch.device:
        """Return the torch device to use, auto-detecting CUDA if not specified.

        Why auto-detect? During development you often switch between a laptop
        (CPU) and a training server (GPU). Auto-detection means the same config
        works in both environments without manual changes.

        Returns:
            torch.device: The resolved device (cuda:0 if available, else cpu).
        """
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # -------------------------------------------------------------------------
    # Parameter count estimation
    # -------------------------------------------------------------------------

    def estimate_param_count(self) -> int:
        """Estimate the total number of trainable parameters in the network.

        This is useful for:
        - Verifying that a config matches expectations before training
        - Estimating memory requirements (params * 4 bytes for FP32)
        - Comparing network sizes across presets

        The estimate counts parameters in each component:
        1. Input convolution: Conv2d(input_planes, F, 3) + BatchNorm(F)
        2. Residual blocks: each has 2x [Conv2d(F, F, 3) + BatchNorm(F)]
        3. Policy head: Conv2d(F, 2, 1) + BN(2) + Linear(2*64, policy_size)
        4. Value head: Conv2d(F, 1, 1) + BN(1) + Linear(64, hidden) + Linear(hidden, 1)

        where F = num_filters.

        Returns:
            int: Estimated total number of trainable parameters.
        """
        f = self.num_filters

        # --- Input convolution ---
        # Conv2d(input_planes, f, 3, padding=1): kernel has input_planes * f * 3 * 3 weights + f biases
        # BatchNorm2d(f): 2 * f parameters (gamma + beta)
        input_conv = self.input_planes * f * 9 + f  # conv weights + conv bias
        input_bn = 2 * f  # batch norm gamma + beta

        # --- Residual blocks ---
        # Each block has 2 convolutions: Conv2d(f, f, 3) + BatchNorm2d(f) each
        # Conv2d(f, f, 3): f * f * 9 weights + f biases
        # BatchNorm2d(f): 2 * f (gamma + beta)
        conv_params = f * f * 9 + f  # one conv layer
        bn_params = 2 * f  # one batch norm layer
        one_block = 2 * (conv_params + bn_params)  # two conv+bn per block
        residual_blocks = self.num_blocks * one_block

        # --- Policy head ---
        # Conv2d(f, 2, 1): f * 2 * 1 * 1 weights + 2 biases
        # BatchNorm2d(2): 2 * 2
        # Linear(2*64, policy_output_size): 128 * policy_output_size + policy_output_size
        policy_conv = f * 2 * 1 + 2
        policy_bn = 2 * 2
        policy_linear = 2 * 64 * self.policy_output_size + self.policy_output_size

        policy_head = policy_conv + policy_bn + policy_linear

        # --- Value head ---
        # Conv2d(f, 1, 1): f * 1 * 1 * 1 weights + 1 bias
        # BatchNorm2d(1): 2 * 1
        # Linear(64, value_hidden_size): 64 * hidden + hidden
        # Linear(value_hidden_size, 1): hidden * 1 + 1
        value_conv = f * 1 * 1 + 1
        value_bn = 2 * 1
        value_linear1 = 64 * self.value_hidden_size + self.value_hidden_size
        value_linear2 = self.value_hidden_size * 1 + 1

        value_head = value_conv + value_bn + value_linear1 + value_linear2

        total = input_conv + input_bn + residual_blocks + policy_head + value_head
        return total

    def estimate_memory_mb(self, dtype_bytes: int = 4) -> float:
        """Estimate model memory in megabytes for a given dtype.

        This estimates only the parameter memory, not activations or optimizer
        state. For training, multiply by ~4 (params + grads + 2 optimizer
        states for Adam) or ~3 for SGD with momentum.

        Args:
            dtype_bytes: Bytes per parameter. 4 for FP32, 2 for FP16/BF16.

        Returns:
            float: Estimated parameter memory in megabytes.
        """
        return self.estimate_param_count() * dtype_bytes / (1024 * 1024)

    def __str__(self) -> str:
        """Human-readable summary of the configuration."""
        device = self.get_device()
        params = self.estimate_param_count()
        memory_fp32 = self.estimate_memory_mb(dtype_bytes=4)

        return (
            f"NetworkConfig(\n"
            f"  blocks={self.num_blocks}, filters={self.num_filters},\n"
            f"  input_planes={self.input_planes}, "
            f"policy_size={self.policy_output_size},\n"
            f"  value_hidden={self.value_hidden_size},\n"
            f"  device={device},\n"
            f"  estimated_params={params:,},\n"
            f"  estimated_memory_fp32={memory_fp32:.1f} MB\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Module-level preset constants
# ---------------------------------------------------------------------------
# These are provided as a convenience so you can write:
#   from neural.config import TINY
# instead of:
#   config = NetworkConfig.tiny()
# Both approaches are equivalent.

TINY: NetworkConfig = NetworkConfig.tiny()
"""Tiny preset: 5 blocks, 64 filters, ~1M params. For debugging."""

SMALL: NetworkConfig = NetworkConfig.small()
"""Small preset: 10 blocks, 128 filters, ~4M params. For overnight runs."""

MEDIUM: NetworkConfig = NetworkConfig.medium()
"""Medium preset: 15 blocks, 192 filters, ~11M params. For multi-day training."""

FULL: NetworkConfig = NetworkConfig.full()
"""Full preset: 19 blocks, 256 filters, ~23M params. Matches original paper."""
