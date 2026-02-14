"""
Tests for AlphaZeroNetwork
===========================

Verifies the complete dual-headed AlphaZero network implementation:
- Forward pass output shapes for policy and value heads
- Value output bounded to [-1, 1] by tanh activation
- Policy output is raw logits (can be negative, no softmax applied)
- Parameter count roughly matches config.estimate_param_count() (within 10%)
- Works with batch sizes 1, 8, 32
- from_config class method works for all presets (TINY, SMALL, MEDIUM, FULL)
- Gradient flows through both policy and value heads
- He (Kaiming) initialization on conv and linear weights
- No bias on Conv2d layers followed by BatchNorm

The tests use all four presets to ensure the network is truly configurable
and not accidentally hardcoded to any particular size.
"""

import math

import pytest
import torch

from neural.config import NetworkConfig, TINY, SMALL, MEDIUM, FULL
from neural.network import AlphaZeroNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_PRESETS = [
    pytest.param(NetworkConfig.tiny(), id="TINY"),
    pytest.param(NetworkConfig.small(), id="SMALL"),
    pytest.param(NetworkConfig.medium(), id="MEDIUM"),
    pytest.param(NetworkConfig.full(), id="FULL"),
]


def _make_input(config: NetworkConfig, batch_size: int = 4) -> torch.Tensor:
    """Create a random input tensor matching the config's input spec."""
    return torch.randn(batch_size, config.input_planes, 8, 8)


# ---------------------------------------------------------------------------
# Forward pass output shapes
# ---------------------------------------------------------------------------


class TestForwardPassShapes:
    """Verify that the network produces correctly shaped outputs."""

    @pytest.mark.parametrize("config", ALL_PRESETS)
    def test_policy_output_shape(self, config: NetworkConfig) -> None:
        """Policy output should be (B, policy_output_size) for each preset."""
        net = AlphaZeroNetwork(config)
        net.eval()
        x = _make_input(config, batch_size=8)
        policy, _ = net(x)
        assert policy.shape == (8, config.policy_output_size), (
            f"Expected policy shape (8, {config.policy_output_size}), got {policy.shape}"
        )

    @pytest.mark.parametrize("config", ALL_PRESETS)
    def test_value_output_shape(self, config: NetworkConfig) -> None:
        """Value output should be (B, 1) for each preset."""
        net = AlphaZeroNetwork(config)
        net.eval()
        x = _make_input(config, batch_size=8)
        _, value = net(x)
        assert value.shape == (8, 1), (
            f"Expected value shape (8, 1), got {value.shape}"
        )

    @pytest.mark.parametrize("config", ALL_PRESETS)
    def test_default_policy_size_is_4672(self, config: NetworkConfig) -> None:
        """All default presets use policy_output_size=4672 (chess standard)."""
        assert config.policy_output_size == 4672


# ---------------------------------------------------------------------------
# Value head: tanh bounds
# ---------------------------------------------------------------------------


class TestValueBounds:
    """The value head uses tanh, so output must be in [-1, 1]."""

    @pytest.mark.parametrize("config", ALL_PRESETS)
    def test_value_in_range(self, config: NetworkConfig) -> None:
        """Value output should be bounded to [-1, 1] by tanh."""
        net = AlphaZeroNetwork(config)
        net.eval()
        x = _make_input(config, batch_size=16)
        _, value = net(x)
        assert (value >= -1.0).all(), f"Value has elements < -1: min={value.min()}"
        assert (value <= 1.0).all(), f"Value has elements > 1: max={value.max()}"

    def test_value_with_extreme_input(self) -> None:
        """Even with large inputs, tanh should keep value in [-1, 1]."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        net.eval()
        # Large magnitude input to stress-test tanh saturation
        x = torch.randn(4, config.input_planes, 8, 8) * 100.0
        _, value = net(x)
        assert (value >= -1.0).all() and (value <= 1.0).all(), (
            f"Value out of range with extreme input: min={value.min()}, max={value.max()}"
        )


# ---------------------------------------------------------------------------
# Policy head: raw logits
# ---------------------------------------------------------------------------


class TestPolicyLogits:
    """The policy head should output raw logits, not softmax probabilities."""

    def test_policy_can_be_negative(self) -> None:
        """Raw logits can be negative (softmax would make everything positive)."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        net.eval()
        # Run several batches to increase likelihood of negative values
        found_negative = False
        for _ in range(10):
            x = _make_input(config, batch_size=32)
            policy, _ = net(x)
            if (policy < 0).any():
                found_negative = True
                break
        assert found_negative, (
            "Policy output should contain negative values (raw logits, not softmax)"
        )

    def test_policy_does_not_sum_to_one(self) -> None:
        """Raw logits should NOT sum to 1 (that would indicate softmax was applied)."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        net.eval()
        x = _make_input(config, batch_size=8)
        policy, _ = net(x)
        row_sums = policy.sum(dim=1)
        # If softmax were applied, all row sums would be exactly 1.0
        all_sum_to_one = torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01)
        assert not all_sum_to_one, (
            "Policy rows sum to ~1.0, suggesting softmax was applied. "
            "The network should output raw logits."
        )


# ---------------------------------------------------------------------------
# Batch size compatibility
# ---------------------------------------------------------------------------


class TestBatchSizes:
    """The network should handle various batch sizes correctly."""

    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_various_batch_sizes(self, batch_size: int) -> None:
        """Forward pass should work with batch sizes 1, 8, and 32."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        net.eval()
        x = _make_input(config, batch_size=batch_size)
        policy, value = net(x)
        assert policy.shape == (batch_size, config.policy_output_size), (
            f"Policy shape wrong for batch_size={batch_size}: {policy.shape}"
        )
        assert value.shape == (batch_size, 1), (
            f"Value shape wrong for batch_size={batch_size}: {value.shape}"
        )


# ---------------------------------------------------------------------------
# Parameter count estimation
# ---------------------------------------------------------------------------


class TestParameterCount:
    """Actual parameter count should roughly match config.estimate_param_count().

    The estimate in config.py includes conv biases (since it was written before
    the network implementation chose bias=False for conv layers followed by
    BatchNorm). The actual count will be slightly lower than the estimate.
    We require them to be within 10% of each other, which easily accommodates
    the small difference from omitted biases.
    """

    @pytest.mark.parametrize("config", ALL_PRESETS)
    def test_param_count_within_10_percent(self, config: NetworkConfig) -> None:
        """Actual parameters should be within 10% of estimated count."""
        net = AlphaZeroNetwork(config)
        actual = net.count_parameters()
        estimated = config.estimate_param_count()
        relative_error = abs(actual - estimated) / estimated
        assert relative_error < 0.10, (
            f"Parameter count mismatch: actual={actual:,}, estimated={estimated:,}, "
            f"relative_error={relative_error:.4f} (>{0.10})"
        )

    @pytest.mark.parametrize("config", ALL_PRESETS)
    def test_actual_less_than_or_equal_estimated(self, config: NetworkConfig) -> None:
        """Actual count should be <= estimated (estimate includes conv biases we omit)."""
        net = AlphaZeroNetwork(config)
        actual = net.count_parameters()
        estimated = config.estimate_param_count()
        assert actual <= estimated, (
            f"Actual params ({actual:,}) > estimated ({estimated:,}). "
            f"This suggests unexpected parameters in the network."
        )


# ---------------------------------------------------------------------------
# from_config class method
# ---------------------------------------------------------------------------


class TestFromConfig:
    """The from_config class method should work for all presets."""

    @pytest.mark.parametrize("config", ALL_PRESETS)
    def test_from_config_creates_network(self, config: NetworkConfig) -> None:
        """from_config should return a valid AlphaZeroNetwork."""
        # Force CPU to avoid issues on machines without GPU
        config_cpu = NetworkConfig(
            num_blocks=config.num_blocks,
            num_filters=config.num_filters,
            input_planes=config.input_planes,
            policy_output_size=config.policy_output_size,
            value_hidden_size=config.value_hidden_size,
            device="cpu",
        )
        net = AlphaZeroNetwork.from_config(config_cpu)
        assert isinstance(net, AlphaZeroNetwork)

    @pytest.mark.parametrize("config", ALL_PRESETS)
    def test_from_config_stores_config(self, config: NetworkConfig) -> None:
        """The constructed network should store its config for inspection."""
        net = AlphaZeroNetwork(config)
        assert net.config is config

    @pytest.mark.parametrize("config", ALL_PRESETS)
    def test_from_config_forward_pass(self, config: NetworkConfig) -> None:
        """A network created via from_config should produce valid outputs."""
        config_cpu = NetworkConfig(
            num_blocks=config.num_blocks,
            num_filters=config.num_filters,
            input_planes=config.input_planes,
            policy_output_size=config.policy_output_size,
            value_hidden_size=config.value_hidden_size,
            device="cpu",
        )
        net = AlphaZeroNetwork.from_config(config_cpu)
        net.eval()
        x = _make_input(config_cpu, batch_size=4)
        policy, value = net(x)
        assert policy.shape == (4, config.policy_output_size)
        assert value.shape == (4, 1)


# ---------------------------------------------------------------------------
# Gradient flow through both heads
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify gradients flow through both heads to all parameters."""

    def test_gradient_through_policy_head(self) -> None:
        """Policy head loss should produce gradients in the shared trunk."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        x = _make_input(config, batch_size=4)
        policy, _ = net(x)
        loss = policy.sum()
        loss.backward()

        # Check that the input conv received gradients (shared trunk)
        assert net.input_conv.weight.grad is not None, (
            "Input conv should receive gradient from policy head"
        )
        assert net.input_conv.weight.grad.abs().sum() > 0, (
            "Input conv gradient should be non-zero from policy head"
        )

    def test_gradient_through_value_head(self) -> None:
        """Value head loss should produce gradients in the shared trunk."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        x = _make_input(config, batch_size=4)
        _, value = net(x)
        loss = value.sum()
        loss.backward()

        # Check that the input conv received gradients (shared trunk)
        assert net.input_conv.weight.grad is not None, (
            "Input conv should receive gradient from value head"
        )
        assert net.input_conv.weight.grad.abs().sum() > 0, (
            "Input conv gradient should be non-zero from value head"
        )

    def test_gradient_through_combined_loss(self) -> None:
        """Combined policy + value loss should produce gradients everywhere."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        x = _make_input(config, batch_size=4)
        policy, value = net(x)
        loss = policy.sum() + value.sum()
        loss.backward()

        # Every trainable parameter should receive a gradient
        for name, param in net.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient"
            )
            assert param.grad.abs().sum() > 0, (
                f"Parameter '{name}' has zero gradient"
            )

    def test_gradient_flows_to_input(self) -> None:
        """Gradients should flow all the way back to the input tensor."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        x = torch.randn(4, config.input_planes, 8, 8, requires_grad=True)
        policy, value = net(x)
        loss = policy.sum() + value.sum()
        loss.backward()

        assert x.grad is not None, "Input should receive gradient"
        assert x.grad.abs().sum() > 0, "Input gradient should be non-zero"


# ---------------------------------------------------------------------------
# No bias on Conv2d layers followed by BatchNorm
# ---------------------------------------------------------------------------


class TestNoBiasOnConvBeforeBN:
    """Conv layers followed by BatchNorm should have bias=False.

    This is a design choice: BatchNorm subtracts the channel mean, which
    absorbs any constant bias from the preceding conv layer. So the conv
    bias is redundant and wastes parameters.
    """

    def test_input_conv_no_bias(self) -> None:
        """Input convolution (followed by BN) should have no bias."""
        net = AlphaZeroNetwork(NetworkConfig.tiny())
        assert net.input_conv.bias is None, "input_conv should have bias=False"

    def test_policy_conv_no_bias(self) -> None:
        """Policy head convolution (followed by BN) should have no bias."""
        net = AlphaZeroNetwork(NetworkConfig.tiny())
        assert net.policy_conv.bias is None, "policy_conv should have bias=False"

    def test_value_conv_no_bias(self) -> None:
        """Value head convolution (followed by BN) should have no bias."""
        net = AlphaZeroNetwork(NetworkConfig.tiny())
        assert net.value_conv.bias is None, "value_conv should have bias=False"


# ---------------------------------------------------------------------------
# He (Kaiming) initialization
# ---------------------------------------------------------------------------


class TestHeInitialization:
    """Verify that conv and linear weights use He (Kaiming) normal init.

    For He init with mode='fan_in' and ReLU:
        std = sqrt(2 / fan_in)

    We check that the actual std is within 20% of the expected value,
    which is generous enough to accommodate stochastic variation.
    """

    def test_input_conv_weight_std(self) -> None:
        """Input conv weight std should approximately match He init."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        fan_in = config.input_planes * 3 * 3
        expected_std = math.sqrt(2.0 / fan_in)
        actual_std = net.input_conv.weight.data.std().item()
        relative_error = abs(actual_std - expected_std) / expected_std
        assert relative_error < 0.20, (
            f"input_conv: std={actual_std:.6f}, expected={expected_std:.6f}, "
            f"relative_error={relative_error:.3f}"
        )

    def test_policy_conv_weight_std(self) -> None:
        """Policy conv weight std should approximately match He init."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        # Conv2d(num_filters, 2, 1): fan_in = num_filters * 1 * 1
        fan_in = config.num_filters * 1 * 1
        expected_std = math.sqrt(2.0 / fan_in)
        actual_std = net.policy_conv.weight.data.std().item()
        relative_error = abs(actual_std - expected_std) / expected_std
        assert relative_error < 0.30, (
            f"policy_conv: std={actual_std:.6f}, expected={expected_std:.6f}, "
            f"relative_error={relative_error:.3f}. "
            f"Note: wider tolerance because only 2*F weights (small sample)."
        )

    def test_value_conv_weight_std(self) -> None:
        """Value conv weight std should approximately match He init."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        # Conv2d(num_filters, 1, 1): fan_in = num_filters * 1 * 1
        fan_in = config.num_filters * 1 * 1
        expected_std = math.sqrt(2.0 / fan_in)
        actual_std = net.value_conv.weight.data.std().item()
        relative_error = abs(actual_std - expected_std) / expected_std
        assert relative_error < 0.50, (
            f"value_conv: std={actual_std:.6f}, expected={expected_std:.6f}, "
            f"relative_error={relative_error:.3f}. "
            f"Note: very wide tolerance because only F weights (tiny sample)."
        )

    def test_policy_fc_bias_is_zero(self) -> None:
        """Linear layer biases should be initialized to zero."""
        net = AlphaZeroNetwork(NetworkConfig.tiny())
        assert torch.allclose(
            net.policy_fc.bias.data,
            torch.zeros_like(net.policy_fc.bias.data),
        ), "policy_fc bias should be initialized to zero"

    def test_value_fc_biases_are_zero(self) -> None:
        """Value head linear layer biases should be initialized to zero."""
        net = AlphaZeroNetwork(NetworkConfig.tiny())
        assert torch.allclose(
            net.value_fc1.bias.data,
            torch.zeros_like(net.value_fc1.bias.data),
        ), "value_fc1 bias should be initialized to zero"
        assert torch.allclose(
            net.value_fc2.bias.data,
            torch.zeros_like(net.value_fc2.bias.data),
        ), "value_fc2 bias should be initialized to zero"


# ---------------------------------------------------------------------------
# Module structure
# ---------------------------------------------------------------------------


class TestModuleStructure:
    """Verify the network has the expected high-level module structure."""

    def test_has_input_conv_section(self) -> None:
        """Network should have input_conv, input_bn, input_relu."""
        net = AlphaZeroNetwork(NetworkConfig.tiny())
        assert hasattr(net, "input_conv")
        assert hasattr(net, "input_bn")
        assert hasattr(net, "input_relu")

    def test_has_residual_tower(self) -> None:
        """Network should have a residual_tower of the correct length."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        assert hasattr(net, "residual_tower")
        assert len(net.residual_tower) == config.num_blocks

    def test_has_policy_head(self) -> None:
        """Network should have policy_conv, policy_bn, policy_relu, policy_fc."""
        net = AlphaZeroNetwork(NetworkConfig.tiny())
        assert hasattr(net, "policy_conv")
        assert hasattr(net, "policy_bn")
        assert hasattr(net, "policy_relu")
        assert hasattr(net, "policy_fc")

    def test_has_value_head(self) -> None:
        """Network should have value_conv, value_bn, value_relu, value_fc1, value_fc2, value_tanh."""
        net = AlphaZeroNetwork(NetworkConfig.tiny())
        assert hasattr(net, "value_conv")
        assert hasattr(net, "value_bn")
        assert hasattr(net, "value_relu")
        assert hasattr(net, "value_fc1")
        assert hasattr(net, "value_fc1_relu")
        assert hasattr(net, "value_fc2")
        assert hasattr(net, "value_tanh")

    def test_residual_tower_block_count_matches_config(self) -> None:
        """Each preset should produce the correct number of residual blocks."""
        for preset_name, config in [
            ("TINY", NetworkConfig.tiny()),
            ("SMALL", NetworkConfig.small()),
            ("MEDIUM", NetworkConfig.medium()),
            ("FULL", NetworkConfig.full()),
        ]:
            net = AlphaZeroNetwork(config)
            actual = len(net.residual_tower)
            assert actual == config.num_blocks, (
                f"{preset_name}: expected {config.num_blocks} blocks, got {actual}"
            )


# ---------------------------------------------------------------------------
# count_parameters method
# ---------------------------------------------------------------------------


class TestCountParameters:
    """Verify the count_parameters() convenience method."""

    def test_count_parameters_matches_manual_count(self) -> None:
        """count_parameters() should match a manual parameter sum."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        manual_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
        assert net.count_parameters() == manual_count

    def test_all_parameters_are_trainable(self) -> None:
        """All parameters should be trainable (requires_grad=True)."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        total = sum(p.numel() for p in net.parameters())
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        assert total == trainable, "All parameters should be trainable"


# ---------------------------------------------------------------------------
# Eval mode determinism
# ---------------------------------------------------------------------------


class TestEvalMode:
    """Verify the network is deterministic in eval mode."""

    def test_eval_mode_is_deterministic(self) -> None:
        """Same input should produce identical outputs in eval mode."""
        config = NetworkConfig.tiny()
        net = AlphaZeroNetwork(config)
        net.eval()
        x = _make_input(config, batch_size=4)

        policy1, value1 = net(x)
        policy2, value2 = net(x)

        assert torch.allclose(policy1, policy2), "Policy should be deterministic in eval mode"
        assert torch.allclose(value1, value2), "Value should be deterministic in eval mode"


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------


class TestCustomConfig:
    """Verify the network works with non-preset custom configurations."""

    def test_custom_config(self) -> None:
        """Network should work with arbitrary valid configs."""
        config = NetworkConfig(num_blocks=3, num_filters=48)
        net = AlphaZeroNetwork(config)
        net.eval()
        x = _make_input(config, batch_size=2)
        policy, value = net(x)
        assert policy.shape == (2, 4672)
        assert value.shape == (2, 1)

    def test_minimal_config(self) -> None:
        """Network should work with minimum valid config (1 block, 32 filters)."""
        config = NetworkConfig(num_blocks=1, num_filters=32)
        net = AlphaZeroNetwork(config)
        net.eval()
        x = _make_input(config, batch_size=1)
        policy, value = net(x)
        assert policy.shape == (1, 4672)
        assert value.shape == (1, 1)
