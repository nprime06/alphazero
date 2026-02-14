"""
Tests for ResidualBlock
========================

Verifies that the residual block implementation is correct by checking:
- Forward pass preserves tensor shape for various filter counts
- Gradient flows through the skip connection (non-zero gradients)
- Parameter count matches the expected formula (no unexpected biases)
- He (Kaiming) initialization produces the correct weight statistics
- Works with various batch sizes (1, 8, 32)
- The block is a valid nn.Module with proper structure

The expected parameter count for a ResidualBlock with F filters:
    2 conv layers: 2 * (F * F * 9) weights, NO biases (bias=False)
    2 batch norms: 2 * (F gamma + F beta) = 4 * F
    Total = 2 * F * F * 9 + 4 * F = 18 * F^2 + 4 * F
"""

import math

import pytest
import torch

from neural.blocks import ResidualBlock


# ---------------------------------------------------------------------------
# Forward pass shape preservation
# ---------------------------------------------------------------------------


class TestForwardPassShape:
    """The residual block should preserve the spatial and channel dimensions."""

    @pytest.mark.parametrize("num_filters", [64, 128, 256])
    def test_output_shape_matches_input(self, num_filters: int) -> None:
        """Output shape must exactly match input shape for any filter count."""
        block = ResidualBlock(num_filters=num_filters)
        x = torch.randn(4, num_filters, 8, 8)
        out = block(x)
        assert out.shape == x.shape, (
            f"Expected output shape {x.shape}, got {out.shape}"
        )

    @pytest.mark.parametrize("num_filters", [64, 128, 256])
    def test_output_dtype_matches_input(self, num_filters: int) -> None:
        """Output dtype should match input dtype."""
        block = ResidualBlock(num_filters=num_filters)
        x = torch.randn(4, num_filters, 8, 8)
        out = block(x)
        assert out.dtype == x.dtype

    def test_non_square_spatial_dims(self) -> None:
        """The block should work with any spatial dimensions, not just 8x8."""
        block = ResidualBlock(num_filters=64)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_single_pixel_spatial_eval_mode(self) -> None:
        """Edge case: 1x1 spatial dimensions work in eval mode.

        In training mode, BatchNorm requires more than 1 value per channel
        to compute variance, so batch_size=1 with 1x1 spatial dims fails.
        In eval mode, BatchNorm uses running statistics instead, so this
        edge case works fine. This is not relevant for chess (always 8x8).
        """
        block = ResidualBlock(num_filters=64)
        block.eval()  # use running stats instead of batch stats
        x = torch.randn(1, 64, 1, 1)
        out = block(x)
        assert out.shape == (1, 64, 1, 1)


# ---------------------------------------------------------------------------
# Batch size compatibility
# ---------------------------------------------------------------------------


class TestBatchSizes:
    """The block should handle various batch sizes correctly."""

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_various_batch_sizes(self, batch_size: int) -> None:
        """Forward pass should work with batch sizes 1, 8, and 32."""
        block = ResidualBlock(num_filters=64)
        x = torch.randn(batch_size, 64, 8, 8)
        out = block(x)
        assert out.shape == (batch_size, 64, 8, 8), (
            f"Failed for batch_size={batch_size}: got {out.shape}"
        )

    def test_output_values_are_non_negative(self) -> None:
        """Since the final activation is ReLU, all outputs should be >= 0."""
        block = ResidualBlock(num_filters=64)
        block.eval()  # deterministic batch norm
        x = torch.randn(8, 64, 8, 8)
        out = block(x)
        assert (out >= 0).all(), "ReLU output should be non-negative"


# ---------------------------------------------------------------------------
# Gradient flow through skip connection
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify that gradients flow through the skip connection.

    The skip connection is the key feature of residual blocks. If it works
    correctly, the gradient with respect to the input should always be
    non-zero, even if the convolutional layers have poor weights.
    """

    def test_gradient_is_nonzero(self) -> None:
        """The gradient through a residual block should be non-zero."""
        block = ResidualBlock(num_filters=64)
        x = torch.randn(4, 64, 8, 8, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Gradient should be computed"
        assert x.grad.abs().sum() > 0, (
            "Gradient should be non-zero due to skip connection"
        )

    def test_gradient_through_skip_path(self) -> None:
        """Even with zero conv weights, the skip connection carries gradient.

        We zero out all conv weights to isolate the skip connection path.
        The gradient should still be non-zero because:
            output = 0 + x (convs produce 0, skip adds x)
            d(output)/dx = 1 (from the skip path)

        Note: in practice, after zeroing conv weights, the output of the conv
        path is just the BN bias (beta). But the gradient with respect to x
        still includes the identity term from the skip connection.
        """
        block = ResidualBlock(num_filters=64)

        # Zero out all convolutional weights so only the skip path matters
        with torch.no_grad():
            block.conv1.weight.zero_()
            block.conv2.weight.zero_()

        x = torch.randn(2, 64, 8, 8, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        grad_norm = x.grad.norm().item()
        assert grad_norm > 0, (
            f"Gradient norm is {grad_norm}, expected > 0 from skip connection"
        )

    def test_all_parameters_receive_gradients(self) -> None:
        """Every trainable parameter should receive a non-zero gradient."""
        block = ResidualBlock(num_filters=64)
        x = torch.randn(4, 64, 8, 8)
        out = block(x)
        loss = out.sum()
        loss.backward()

        for name, param in block.named_parameters():
            assert param.grad is not None, f"Parameter '{name}' has no gradient"
            assert param.grad.abs().sum() > 0, (
                f"Parameter '{name}' has zero gradient"
            )


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


class TestParameterCount:
    """Verify the block has exactly the expected number of parameters.

    For a ResidualBlock with F filters:
        Conv weights:  2 * (F * F * 3 * 3) = 18 * F^2
        Conv biases:   0 (bias=False because BatchNorm follows)
        BN gamma:      2 * F
        BN beta:       2 * F
        Total:         18 * F^2 + 4 * F

    We also verify that there are no conv biases, which would indicate a
    misconfiguration (bias=True instead of bias=False).
    """

    @pytest.mark.parametrize("num_filters", [64, 128, 256])
    def test_total_parameter_count(self, num_filters: int) -> None:
        """Total parameters should match the formula 18*F^2 + 4*F."""
        block = ResidualBlock(num_filters=num_filters)
        expected = 18 * num_filters ** 2 + 4 * num_filters
        actual = sum(p.numel() for p in block.parameters())
        assert actual == expected, (
            f"For F={num_filters}: expected {expected} params, got {actual}. "
            f"Difference of {actual - expected} suggests unexpected biases."
        )

    @pytest.mark.parametrize("num_filters", [64, 128, 256])
    def test_no_conv_biases(self, num_filters: int) -> None:
        """Conv layers should have bias=False (bias is redundant with BatchNorm)."""
        block = ResidualBlock(num_filters=num_filters)
        assert block.conv1.bias is None, "conv1 should have no bias"
        assert block.conv2.bias is None, "conv2 should have no bias"

    def test_trainable_vs_total_parameters(self) -> None:
        """All parameters should be trainable (no frozen parameters)."""
        block = ResidualBlock(num_filters=64)
        total = sum(p.numel() for p in block.parameters())
        trainable = sum(p.numel() for p in block.parameters() if p.requires_grad)
        assert total == trainable, "All parameters should be trainable"

    def test_parameter_breakdown(self) -> None:
        """Verify the parameter count of each individual layer."""
        f = 64
        block = ResidualBlock(num_filters=f)

        # Conv weights: F_out * F_in * kernel_h * kernel_w
        assert block.conv1.weight.shape == (f, f, 3, 3)
        assert block.conv2.weight.shape == (f, f, 3, 3)

        # BN has weight (gamma) and bias (beta), each of size F
        assert block.bn1.weight.shape == (f,)
        assert block.bn1.bias.shape == (f,)
        assert block.bn2.weight.shape == (f,)
        assert block.bn2.bias.shape == (f,)


# ---------------------------------------------------------------------------
# He (Kaiming) initialization
# ---------------------------------------------------------------------------


class TestHeInitialization:
    """Verify that conv weights are initialized with He (Kaiming) normal.

    For He initialization with mode='fan_in' and ReLU nonlinearity:
        std = sqrt(2 / fan_in)
        fan_in = in_channels * kernel_h * kernel_w = F * 3 * 3 = 9F

    So the expected std = sqrt(2 / (9 * F)).

    We check that the actual std is within a reasonable tolerance of the
    expected value. The tolerance must account for the finite number of
    samples -- with F^2 * 9 weights, the sample std will deviate from
    the true std by approximately std / sqrt(2 * N) where N is the
    number of weights. We use a generous tolerance of 20% relative error
    since the exact value is stochastic.
    """

    @pytest.mark.parametrize("num_filters", [64, 128, 256])
    def test_conv_weight_std(self, num_filters: int) -> None:
        """Conv weight std should approximately match sqrt(2 / fan_in)."""
        block = ResidualBlock(num_filters=num_filters)

        fan_in = num_filters * 3 * 3  # in_channels * kernel_h * kernel_w
        expected_std = math.sqrt(2.0 / fan_in)

        for name, conv in [("conv1", block.conv1), ("conv2", block.conv2)]:
            actual_std = conv.weight.data.std().item()
            relative_error = abs(actual_std - expected_std) / expected_std
            assert relative_error < 0.20, (
                f"{name} (F={num_filters}): std={actual_std:.6f}, "
                f"expected={expected_std:.6f}, relative_error={relative_error:.3f}"
            )

    @pytest.mark.parametrize("num_filters", [64, 128, 256])
    def test_conv_weight_mean_near_zero(self, num_filters: int) -> None:
        """He initialization should produce weights with mean near zero."""
        block = ResidualBlock(num_filters=num_filters)

        for name, conv in [("conv1", block.conv1), ("conv2", block.conv2)]:
            actual_mean = conv.weight.data.mean().item()
            # Mean should be very close to 0 for a symmetric distribution.
            # Allow up to 0.01 absolute deviation.
            assert abs(actual_mean) < 0.01, (
                f"{name} (F={num_filters}): mean={actual_mean:.6f}, expected ~0"
            )

    def test_batchnorm_gamma_initialized_to_ones(self) -> None:
        """BatchNorm gamma (weight) should be initialized to 1."""
        block = ResidualBlock(num_filters=64)
        for name, bn in [("bn1", block.bn1), ("bn2", block.bn2)]:
            assert torch.allclose(bn.weight.data, torch.ones_like(bn.weight.data)), (
                f"{name} gamma should be all ones"
            )

    def test_batchnorm_beta_initialized_to_zeros(self) -> None:
        """BatchNorm beta (bias) should be initialized to 0."""
        block = ResidualBlock(num_filters=64)
        for name, bn in [("bn1", block.bn1), ("bn2", block.bn2)]:
            assert torch.allclose(bn.bias.data, torch.zeros_like(bn.bias.data)), (
                f"{name} beta should be all zeros"
            )


# ---------------------------------------------------------------------------
# Module structure
# ---------------------------------------------------------------------------


class TestModuleStructure:
    """Verify the block has the expected nn.Module structure."""

    def test_has_expected_submodules(self) -> None:
        """The block should contain conv1, bn1, conv2, bn2, and relu."""
        block = ResidualBlock(num_filters=64)
        child_names = {name for name, _ in block.named_children()}
        expected = {"conv1", "bn1", "conv2", "bn2", "relu"}
        assert child_names == expected, (
            f"Expected submodules {expected}, got {child_names}"
        )

    def test_conv_kernel_size(self) -> None:
        """Both convolutions should use 3x3 kernels."""
        block = ResidualBlock(num_filters=64)
        assert block.conv1.kernel_size == (3, 3)
        assert block.conv2.kernel_size == (3, 3)

    def test_conv_padding(self) -> None:
        """Both convolutions should use padding=1 to preserve spatial dims."""
        block = ResidualBlock(num_filters=64)
        assert block.conv1.padding == (1, 1)
        assert block.conv2.padding == (1, 1)

    def test_stores_num_filters(self) -> None:
        """The block should store num_filters for external inspection."""
        block = ResidualBlock(num_filters=128)
        assert block.num_filters == 128


# ---------------------------------------------------------------------------
# Eval mode behavior
# ---------------------------------------------------------------------------


class TestEvalMode:
    """Verify the block works correctly in eval mode (inference)."""

    def test_eval_mode_is_deterministic(self) -> None:
        """In eval mode, the same input should produce the same output."""
        block = ResidualBlock(num_filters=64)
        block.eval()
        x = torch.randn(4, 64, 8, 8)

        out1 = block(x)
        out2 = block(x)
        assert torch.allclose(out1, out2), (
            "Eval mode should be deterministic"
        )

    def test_train_vs_eval_differ(self) -> None:
        """Train and eval modes should produce different outputs.

        BatchNorm uses running statistics in eval mode but batch statistics
        in train mode, so the outputs should differ (except in degenerate
        cases). We run a forward pass in train mode first to accumulate
        running stats, then compare.
        """
        block = ResidualBlock(num_filters=64)
        x = torch.randn(8, 64, 8, 8)

        # Forward pass in train mode to update running stats
        block.train()
        out_train = block(x).detach()

        # Forward pass in eval mode uses running stats
        block.eval()
        out_eval = block(x).detach()

        # They should differ because batch stats != running stats (generally)
        assert not torch.allclose(out_train, out_eval, atol=1e-5), (
            "Train and eval mode outputs should generally differ due to BatchNorm"
        )
