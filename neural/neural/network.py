"""
AlphaZero Network
=================

Assembles the complete dual-headed AlphaZero neural network from configurable
components. This is the "glue" module that connects the input convolution,
residual tower, policy head, and value head into a single ``nn.Module``.

Architecture overview::

    Input (B, 119, 8, 8)
        |
    [Input Convolution]
        Conv2d(119, F, 3, pad=1, bias=False) -> BatchNorm2d(F) -> ReLU
        |
    [Residual Tower]
        N x ResidualBlock(F)     (N = config.num_blocks)
        |
        +-------------------+
        |                   |
    [Policy Head]       [Value Head]
        |                   |
    Conv2d(F,2,1)       Conv2d(F,1,1)
    BN(2) -> ReLU       BN(1) -> ReLU
    Flatten             Flatten
    Linear(128, 4672)   Linear(64, H) -> ReLU -> Linear(H, 1) -> Tanh
        |                   |
    policy_logits       value
    (B, 4672)           (B, 1)

Why two heads?
    AlphaZero needs two different predictions from the same position:
    - **Policy**: a probability distribution over all legal moves, used by MCTS
      to guide search toward promising moves. The network outputs raw logits
      (before softmax) so that MCTS can mask illegal moves before normalizing.
    - **Value**: a scalar estimate of who is winning, in [-1, 1] where +1 means
      the current player is winning and -1 means they are losing. The tanh
      activation naturally bounds the output to this range.

    Sharing a common trunk (input conv + residual tower) between the two heads
    is efficient because both predictions depend on the same positional features.
    The heads then specialize: the policy head learns move-specific features,
    while the value head learns evaluation-specific features.

Why raw logits (no softmax) for policy?
    During MCTS, illegal moves must be masked out before computing probabilities.
    If we applied softmax inside the network, we'd need to re-normalize after
    masking. It's cleaner and numerically more stable to output raw logits, mask
    them (set illegal moves to -inf), and then apply softmax once.

Why He initialization everywhere?
    All convolutional and linear layers use ReLU activations (except the final
    value output which uses tanh). He initialization compensates for the variance
    reduction caused by ReLU's zeroing of negative values, keeping the signal
    magnitude stable across many layers. This is especially important for the
    deep residual tower (up to 19 blocks = 38 conv layers).

Configuration:
    All layer sizes are derived from ``NetworkConfig``. No dimensions are
    hardcoded, so the same code works for all presets (TINY through FULL).
    The ``from_config`` class method is the recommended way to construct a
    network, as it ensures the config is properly validated.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from neural.blocks import ResidualBlock
from neural.config import NetworkConfig


class AlphaZeroNetwork(nn.Module):
    """Complete AlphaZero dual-headed neural network.

    This module combines an input convolution, a tower of residual blocks,
    and two output heads (policy and value) into a single network. All
    dimensions are determined by the provided ``NetworkConfig``.

    Args:
        config: Network configuration specifying all architectural parameters.
            Use ``NetworkConfig.tiny()`` for debugging, ``NetworkConfig.full()``
            for the paper's architecture, or any custom configuration.

    Attributes:
        config: The configuration used to build this network (stored for
            serialization and inspection).

    Example::

        >>> config = NetworkConfig.tiny()  # 5 blocks, 64 filters
        >>> net = AlphaZeroNetwork(config)
        >>> x = torch.randn(8, 119, 8, 8)
        >>> policy_logits, value = net(x)
        >>> policy_logits.shape
        torch.Size([8, 4672])
        >>> value.shape
        torch.Size([8, 1])
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config

        # -------------------------------------------------------------------
        # Input convolution
        # -------------------------------------------------------------------
        # Transforms the raw input planes (e.g., 119 planes for standard chess
        # encoding) into the network's internal representation width (num_filters).
        # This is a single conv + batchnorm + relu, similar to the first layer
        # in a standard ResNet.
        #
        # bias=False because BatchNorm immediately follows (see blocks.py
        # module docstring for explanation).
        self.input_conv = nn.Conv2d(
            in_channels=config.input_planes,
            out_channels=config.num_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.input_bn = nn.BatchNorm2d(config.num_filters)
        self.input_relu = nn.ReLU(inplace=True)

        # -------------------------------------------------------------------
        # Residual tower (body)
        # -------------------------------------------------------------------
        # A stack of N residual blocks, where N = config.num_blocks. Each block
        # preserves the tensor shape (B, num_filters, 8, 8), so blocks can be
        # stacked to arbitrary depth. The skip connections in each block ensure
        # gradient flow even for deep networks (see blocks.py for details).
        #
        # We use nn.Sequential for clean forward pass delegation. ModuleList
        # would also work but requires an explicit loop in forward().
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(config.num_filters) for _ in range(config.num_blocks)]
        )

        # -------------------------------------------------------------------
        # Policy head
        # -------------------------------------------------------------------
        # Reduces the spatial representation to a flat policy vector over all
        # possible moves. The architecture follows the original AlphaZero paper:
        #
        # 1. 1x1 convolution reducing channels from num_filters to 2.
        #    Why 2 channels? This is a lightweight bottleneck that extracts
        #    move-relevant features while keeping compute low. The original
        #    paper uses 2; some implementations use 32 or more for stronger
        #    policy predictions at the cost of more parameters.
        #
        # 2. BatchNorm + ReLU for normalization and nonlinearity.
        #
        # 3. Flatten to a vector of size 2 * 8 * 8 = 128.
        #
        # 4. Linear layer mapping 128 -> policy_output_size (4672 for chess).
        #    This is where the network learns which spatial features correspond
        #    to which moves in the policy encoding.
        self.policy_conv = nn.Conv2d(
            in_channels=config.num_filters,
            out_channels=2,
            kernel_size=1,
            bias=False,
        )
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(
            in_features=2 * 8 * 8,  # 2 channels * 8 * 8 spatial = 128
            out_features=config.policy_output_size,
        )

        # -------------------------------------------------------------------
        # Value head
        # -------------------------------------------------------------------
        # Reduces the spatial representation to a single scalar value estimate
        # in [-1, 1]. The architecture follows the original AlphaZero paper:
        #
        # 1. 1x1 convolution reducing channels from num_filters to 1.
        #    A single channel is sufficient because the value is a scalar --
        #    we just need one "summary" feature map before flattening.
        #
        # 2. BatchNorm + ReLU.
        #
        # 3. Flatten to a vector of size 1 * 8 * 8 = 64.
        #
        # 4. Hidden linear layer: 64 -> value_hidden_size (default 256).
        #    This gives the value head enough capacity to learn complex
        #    evaluation functions (material balance, king safety, pawn
        #    structure, etc.).
        #
        # 5. ReLU activation on the hidden layer.
        #
        # 6. Output linear layer: value_hidden_size -> 1.
        #
        # 7. Tanh activation to bound the output to [-1, 1].
        #    This range matches the game outcome values: +1 (win), 0 (draw),
        #    -1 (loss). Tanh is preferred over sigmoid here because the
        #    symmetric range naturally represents the zero-sum nature of chess.
        self.value_conv = nn.Conv2d(
            in_channels=config.num_filters,
            out_channels=1,
            kernel_size=1,
            bias=False,
        )
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(
            in_features=1 * 8 * 8,  # 1 channel * 8 * 8 spatial = 64
            out_features=config.value_hidden_size,
        )
        self.value_fc1_relu = nn.ReLU(inplace=True)
        self.value_fc2 = nn.Linear(
            in_features=config.value_hidden_size,
            out_features=1,
        )
        self.value_tanh = nn.Tanh()

        # -------------------------------------------------------------------
        # Weight initialization
        # -------------------------------------------------------------------
        # Apply He initialization to all conv and linear layers in the heads
        # and input conv. The residual blocks handle their own initialization
        # internally (see ResidualBlock.__init__).
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply He (Kaiming) initialization to all conv and linear layers.

        He initialization sets weights to have std = sqrt(2 / fan_in) for
        layers followed by ReLU. This is applied to:
        - The input convolution
        - Policy head conv and linear layers
        - Value head conv and linear layers

        The residual blocks initialize their own weights in their __init__,
        so we skip them here to avoid double-initialization.

        BatchNorm parameters are set to the standard defaults (gamma=1,
        beta=0), which is already the PyTorch default but we set them
        explicitly for clarity.

        Note on the value head output layer (value_fc2):
            This layer is followed by tanh, not ReLU. Technically, He
            initialization is designed for ReLU. However, using He init here
            is a pragmatic choice: the layer has very few parameters (hidden+1),
            so the initialization has minimal impact. Xavier initialization
            would be theoretically more appropriate for tanh, but the difference
            is negligible in practice.
        """
        # Input conv + BN
        nn.init.kaiming_normal_(
            self.input_conv.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.ones_(self.input_bn.weight)
        nn.init.zeros_(self.input_bn.bias)

        # Policy head conv + BN + Linear
        nn.init.kaiming_normal_(
            self.policy_conv.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.ones_(self.policy_bn.weight)
        nn.init.zeros_(self.policy_bn.bias)
        nn.init.kaiming_normal_(
            self.policy_fc.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(self.policy_fc.bias)

        # Value head conv + BN + Linear layers
        nn.init.kaiming_normal_(
            self.value_conv.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.ones_(self.value_bn.weight)
        nn.init.zeros_(self.value_bn.bias)
        nn.init.kaiming_normal_(
            self.value_fc1.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(self.value_fc1.bias)
        nn.init.kaiming_normal_(
            self.value_fc2.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(self.value_fc2.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the complete AlphaZero network.

        Processes a batch of board positions through the shared trunk, then
        splits into policy and value heads for the two output predictions.

        Args:
            x: Input tensor of shape ``(B, input_planes, 8, 8)`` where B is
                the batch size and input_planes matches ``config.input_planes``
                (default 119 for standard chess encoding).

        Returns:
            A tuple ``(policy_logits, value)`` where:
            - ``policy_logits``: shape ``(B, policy_output_size)``, raw logits
              (NOT softmax-normalized). These should be masked for illegal moves
              and then softmax-normalized before use in MCTS.
            - ``value``: shape ``(B, 1)``, scalar value estimate in [-1, 1]
              thanks to tanh activation. +1 means the current player is winning,
              -1 means losing, 0 is a draw.
        """
        # --- Shared trunk ---

        # Input convolution: expand raw planes to the network's filter width
        s = self.input_conv(x)
        s = self.input_bn(s)
        s = self.input_relu(s)

        # Residual tower: deep feature extraction
        s = self.residual_tower(s)

        # --- Policy head ---
        p = self.policy_conv(s)
        p = self.policy_bn(p)
        p = self.policy_relu(p)
        p = p.flatten(start_dim=1)  # (B, 2, 8, 8) -> (B, 128)
        policy_logits = self.policy_fc(p)  # (B, 128) -> (B, policy_output_size)

        # --- Value head ---
        v = self.value_conv(s)
        v = self.value_bn(v)
        v = self.value_relu(v)
        v = v.flatten(start_dim=1)  # (B, 1, 8, 8) -> (B, 64)
        v = self.value_fc1(v)  # (B, 64) -> (B, value_hidden_size)
        v = self.value_fc1_relu(v)
        v = self.value_fc2(v)  # (B, value_hidden_size) -> (B, 1)
        value = self.value_tanh(v)  # bound to [-1, 1]

        return policy_logits, value

    @classmethod
    def from_config(cls, config: NetworkConfig) -> "AlphaZeroNetwork":
        """Construct an AlphaZeroNetwork from a NetworkConfig.

        This is the recommended way to create a network. It ensures the config
        is validated (via NetworkConfig's __post_init__) before the network is
        constructed, and optionally moves the network to the configured device.

        Args:
            config: Network configuration. Use presets like
                ``NetworkConfig.tiny()`` or custom configurations.

        Returns:
            A new AlphaZeroNetwork instance on the configured device.

        Example::

            >>> net = AlphaZeroNetwork.from_config(NetworkConfig.tiny())
            >>> net = AlphaZeroNetwork.from_config(NetworkConfig.full())
            >>> net = AlphaZeroNetwork.from_config(NetworkConfig(num_blocks=12, num_filters=160))
        """
        network = cls(config)
        device = config.get_device()
        network = network.to(device)
        return network

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters in the network.

        This counts actual parameters in the instantiated model, which can be
        compared against ``config.estimate_param_count()`` to verify correctness.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
