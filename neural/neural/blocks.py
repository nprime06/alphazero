"""
Residual Block
==============

Implements the residual block used as the core building block of the AlphaZero
network trunk. The full network stacks many of these blocks (5 to 19 depending
on the configuration) between the input convolution and the policy/value heads.

What is a residual block?
    A residual block passes its input through two convolutional layers and then
    *adds* the original input back to the result before the final activation.
    This "skip connection" (or "shortcut connection") was introduced in the
    ResNet paper (He et al., 2015) and is the key insight that enables training
    of very deep networks (50, 100, or even 1000+ layers).

Why do skip connections help?
    In a plain deep network, gradients must flow through every layer during
    backpropagation. As the network gets deeper, these gradients can vanish
    (shrink to near-zero) or explode (grow uncontrollably), making training
    impossible. The skip connection provides a direct path for gradients to
    flow from later layers to earlier layers without passing through any
    nonlinearities or weight matrices. Mathematically:

        output = F(x) + x

    The gradient of the output with respect to x is:

        d(output)/dx = d(F(x))/dx + 1

    That "+1" term means the gradient is *always* at least 1 along the skip
    path, regardless of what happens inside F(x). This prevents vanishing
    gradients and allows the network to be trained effectively at any depth.

Layer sequence (for filter count F):
    1. Conv2d(F, F, 3, padding=1, bias=False)   -- first 3x3 convolution
    2. BatchNorm2d(F)                            -- normalize activations
    3. ReLU                                      -- nonlinearity
    4. Conv2d(F, F, 3, padding=1, bias=False)   -- second 3x3 convolution
    5. BatchNorm2d(F)                            -- normalize activations
    6. Add input (skip connection)               -- residual addition
    7. ReLU                                      -- final nonlinearity

Why no bias on Conv2d layers?
    When a Conv2d layer is immediately followed by BatchNorm, the conv bias is
    redundant. BatchNorm subtracts the channel mean (which absorbs any constant
    bias) and then applies its own learnable shift (beta). So the conv bias
    would be subtracted out and replaced by the BN beta anyway. Omitting it
    saves parameters and avoids confusing the optimizer with two parameters
    that do the same thing.

Why He (Kaiming) initialization?
    ReLU activation zeros out all negative values, effectively halving the
    variance of each layer's output. He initialization compensates for this by
    scaling the initial weights with std = sqrt(2 / fan_in), where fan_in is
    the number of input connections per neuron. This keeps the signal variance
    roughly constant across layers at initialization, which is critical for
    stable training of deep networks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A single residual block for the AlphaZero network trunk.

    This block applies two 3x3 convolutions with batch normalization and ReLU
    activations, adding the input back via a skip connection before the final
    ReLU. The input and output have the same spatial dimensions (8x8) and the
    same number of channels (num_filters), so the skip connection is a simple
    identity addition with no projection needed.

    Architecture diagram::

        input (B, F, 8, 8)
          |
          +--------------------------+  (skip connection)
          |                          |
        Conv2d(F, F, 3, pad=1)      |
          |                          |
        BatchNorm2d(F)               |
          |                          |
        ReLU                         |
          |                          |
        Conv2d(F, F, 3, pad=1)      |
          |                          |
        BatchNorm2d(F)               |
          |                          |
        Add <------------------------+
          |
        ReLU
          |
        output (B, F, 8, 8)

    Args:
        num_filters: Number of convolutional filters (channels). This must
            match the channel dimension of the input tensor. In the AlphaZero
            network, this is the same for all residual blocks and is set by
            ``NetworkConfig.num_filters``.

    Example::

        >>> block = ResidualBlock(num_filters=256)
        >>> x = torch.randn(8, 256, 8, 8)  # batch of 8, 256 channels, 8x8 board
        >>> out = block(x)
        >>> out.shape
        torch.Size([8, 256, 8, 8])
    """

    def __init__(self, num_filters: int) -> None:
        super().__init__()

        self.num_filters = num_filters

        # --- First convolutional layer ---
        # 3x3 convolution preserving spatial dimensions (padding=1).
        # bias=False because BatchNorm immediately follows (see module docstring).
        self.conv1 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_filters)

        # --- Second convolutional layer ---
        # Same structure as the first. The skip connection is added after bn2
        # but before the final ReLU.
        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

        # --- Shared activation ---
        # A single ReLU module is reused for both activation points. Since ReLU
        # is stateless (no learnable parameters, no internal state), sharing a
        # single instance is safe and slightly cleaner than creating two.
        self.relu = nn.ReLU(inplace=True)

        # --- Weight initialization ---
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply He (Kaiming) initialization to convolutional weights.

        He initialization sets the initial weight standard deviation to
        sqrt(2 / fan_in), where fan_in = num_filters * kernel_height * kernel_width.
        For a 3x3 conv with F filters, fan_in = F * 9, so:

            std = sqrt(2 / (F * 9))

        This compensates for the variance reduction caused by ReLU (which zeros
        out ~half the activations) and keeps the signal magnitude stable as it
        passes through many layers.

        BatchNorm parameters are initialized to their standard defaults:
        gamma = 1 (scale), beta = 0 (shift). This is already the PyTorch default,
        but we set them explicitly for clarity.
        """
        for conv in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")

        for bn in [self.bn1, self.bn2]:
            nn.init.ones_(bn.weight)   # gamma = 1
            nn.init.zeros_(bn.bias)    # beta = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.

        Args:
            x: Input tensor of shape ``(batch_size, num_filters, height, width)``.
                Typically ``(B, F, 8, 8)`` for chess, where F = num_filters.

        Returns:
            Output tensor of the same shape as the input. The skip connection
            ensures that the output is at least as informative as the input --
            in the worst case, the network can learn F(x) = 0, making the block
            an identity function.
        """
        # Save the input for the skip connection
        identity = x

        # First conv -> batchnorm -> relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv -> batchnorm
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection: add the original input
        out = out + identity

        # Final activation
        out = self.relu(out)

        return out
