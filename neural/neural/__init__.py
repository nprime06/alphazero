"""
AlphaZero Neural Network Package
=================================

This package implements the neural network components for an AlphaZero chess engine.
The network uses a dual-headed architecture: a policy head that predicts move
probabilities and a value head that predicts the expected game outcome.

The design is fully configurable -- network depth, width, and head sizes are all
adjustable through the NetworkConfig dataclass. Preset configurations are provided
for common use cases ranging from quick debugging (TINY) to full-strength training
(FULL).

Quick Start:
    >>> from neural.config import NetworkConfig, TINY, FULL
    >>> config = NetworkConfig.tiny()   # 5 blocks, 64 filters -- fast experiments
    >>> config = NetworkConfig.full()   # 19 blocks, 256 filters -- matches the paper
    >>> config = NetworkConfig(num_blocks=12, num_filters=160)  # custom
"""

from neural.blocks import ResidualBlock
from neural.config import NetworkConfig
from neural.encoding import BoardState, CastlingRights, Color, PieceType, encode_batch, encode_board
from neural.export import (
    export_fp16,
    export_torchscript,
    export_with_metadata,
    load_metadata,
    load_torchscript,
    verify_export,
)
from neural.losses import AlphaZeroLoss, LossResult
from neural.network import AlphaZeroNetwork

__all__ = [
    "NetworkConfig",
    "ResidualBlock",
    "AlphaZeroNetwork",
    "AlphaZeroLoss",
    "LossResult",
    "BoardState",
    "CastlingRights",
    "Color",
    "PieceType",
    "encode_board",
    "encode_batch",
    "export_torchscript",
    "load_torchscript",
    "export_fp16",
    "verify_export",
    "export_with_metadata",
    "load_metadata",
]
