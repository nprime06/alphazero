"""
Tests for NetworkConfig
========================

Verifies that:
- All presets can be instantiated with correct values
- Validation rejects out-of-range parameters
- Parameter count estimation produces reasonable numbers
- Device auto-detection works
- Module-level preset constants match factory methods
- Custom configurations work correctly
"""

import pytest
import torch

from neural.config import (
    FULL,
    MEDIUM,
    SMALL,
    TINY,
    NetworkConfig,
)


# ---------------------------------------------------------------------------
# Preset instantiation
# ---------------------------------------------------------------------------


class TestPresets:
    """Verify that each preset has the expected configuration values."""

    def test_tiny_preset(self) -> None:
        config = NetworkConfig.tiny()
        assert config.num_blocks == 5
        assert config.num_filters == 64
        assert config.input_planes == 119
        assert config.policy_output_size == 4672
        assert config.value_hidden_size == 256

    def test_small_preset(self) -> None:
        config = NetworkConfig.small()
        assert config.num_blocks == 10
        assert config.num_filters == 128

    def test_medium_preset(self) -> None:
        config = NetworkConfig.medium()
        assert config.num_blocks == 15
        assert config.num_filters == 192

    def test_full_preset(self) -> None:
        config = NetworkConfig.full()
        assert config.num_blocks == 19
        assert config.num_filters == 256

    def test_all_presets_share_defaults(self) -> None:
        """All presets should use the same input/output sizes (standard chess encoding)."""
        for factory in [NetworkConfig.tiny, NetworkConfig.small, NetworkConfig.medium, NetworkConfig.full]:
            config = factory()
            assert config.input_planes == 119, f"{factory.__name__} has wrong input_planes"
            assert config.policy_output_size == 4672, f"{factory.__name__} has wrong policy_output_size"
            assert config.value_hidden_size == 256, f"{factory.__name__} has wrong value_hidden_size"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleLevelConstants:
    """Verify that module-level constants match the factory methods."""

    def test_tiny_constant(self) -> None:
        assert TINY == NetworkConfig.tiny()

    def test_small_constant(self) -> None:
        assert SMALL == NetworkConfig.small()

    def test_medium_constant(self) -> None:
        assert MEDIUM == NetworkConfig.medium()

    def test_full_constant(self) -> None:
        assert FULL == NetworkConfig.full()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Verify that invalid configurations are rejected with clear errors."""

    def test_blocks_too_low(self) -> None:
        with pytest.raises(ValueError, match="num_blocks must be between 1 and 40"):
            NetworkConfig(num_blocks=0)

    def test_blocks_too_high(self) -> None:
        with pytest.raises(ValueError, match="num_blocks must be between 1 and 40"):
            NetworkConfig(num_blocks=41)

    def test_filters_too_low(self) -> None:
        with pytest.raises(ValueError, match="num_filters must be between 32 and 512"):
            NetworkConfig(num_filters=16)

    def test_filters_too_high(self) -> None:
        with pytest.raises(ValueError, match="num_filters must be between 32 and 512"):
            NetworkConfig(num_filters=1024)

    def test_input_planes_zero(self) -> None:
        with pytest.raises(ValueError, match="input_planes must be positive"):
            NetworkConfig(input_planes=0)

    def test_policy_output_size_zero(self) -> None:
        with pytest.raises(ValueError, match="policy_output_size must be positive"):
            NetworkConfig(policy_output_size=0)

    def test_value_hidden_size_zero(self) -> None:
        with pytest.raises(ValueError, match="value_hidden_size must be positive"):
            NetworkConfig(value_hidden_size=0)

    def test_boundary_values_are_valid(self) -> None:
        """Edge cases at the boundaries should be accepted."""
        NetworkConfig(num_blocks=1, num_filters=32)   # minimum valid
        NetworkConfig(num_blocks=40, num_filters=512)  # maximum valid


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    """Frozen dataclass should prevent accidental mutation."""

    def test_cannot_modify_num_blocks(self) -> None:
        config = NetworkConfig.tiny()
        with pytest.raises(AttributeError):
            config.num_blocks = 10  # type: ignore[misc]

    def test_cannot_modify_num_filters(self) -> None:
        config = NetworkConfig.tiny()
        with pytest.raises(AttributeError):
            config.num_filters = 128  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Parameter count estimation
# ---------------------------------------------------------------------------


class TestParamEstimation:
    """Verify parameter count estimates are in the right ballpark.

    Actual estimated values (policy head linear layer dominates at small sizes):
        TINY:   ~1M   (policy head alone is ~600K)
        SMALL:  ~4M
        MEDIUM: ~11M
        FULL:   ~23M

    These are approximate, so we use generous tolerance bands.
    """

    def test_tiny_param_count(self) -> None:
        params = TINY.estimate_param_count()
        assert 500_000 < params < 2_000_000, f"TINY params={params:,}, expected ~1M"

    def test_small_param_count(self) -> None:
        params = SMALL.estimate_param_count()
        assert 2_000_000 < params < 8_000_000, f"SMALL params={params:,}, expected ~4M"

    def test_medium_param_count(self) -> None:
        params = MEDIUM.estimate_param_count()
        assert 6_000_000 < params < 16_000_000, f"MEDIUM params={params:,}, expected ~11M"

    def test_full_param_count(self) -> None:
        params = FULL.estimate_param_count()
        assert 15_000_000 < params < 30_000_000, f"FULL params={params:,}, expected ~23M"

    def test_more_blocks_means_more_params(self) -> None:
        """Parameter count should increase monotonically with network size."""
        counts = [preset.estimate_param_count() for preset in [TINY, SMALL, MEDIUM, FULL]]
        for i in range(len(counts) - 1):
            assert counts[i] < counts[i + 1], (
                f"Expected monotonically increasing params, "
                f"but got {counts[i]:,} >= {counts[i+1]:,}"
            )

    def test_param_count_is_positive(self) -> None:
        config = NetworkConfig(num_blocks=1, num_filters=32)
        assert config.estimate_param_count() > 0


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------


class TestMemoryEstimation:
    """Verify memory estimation is consistent with parameter count."""

    def test_fp32_memory(self) -> None:
        params = FULL.estimate_param_count()
        memory_mb = FULL.estimate_memory_mb(dtype_bytes=4)
        expected_mb = params * 4 / (1024 * 1024)
        assert abs(memory_mb - expected_mb) < 0.01

    def test_fp16_is_half_fp32(self) -> None:
        fp32 = FULL.estimate_memory_mb(dtype_bytes=4)
        fp16 = FULL.estimate_memory_mb(dtype_bytes=2)
        assert abs(fp16 - fp32 / 2) < 0.01


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------


class TestDeviceHandling:
    """Verify device resolution logic."""

    def test_explicit_cpu(self) -> None:
        config = NetworkConfig(device="cpu")
        assert config.get_device() == torch.device("cpu")

    def test_explicit_cuda(self) -> None:
        """If CUDA is explicitly requested, get_device should return it.

        Note: This test just checks the string is parsed correctly; it does
        not require actual CUDA hardware.
        """
        config = NetworkConfig(device="cuda:0")
        assert config.get_device() == torch.device("cuda:0")

    def test_auto_detect_returns_valid_device(self) -> None:
        """Auto-detect should return either cpu or cuda, never crash."""
        config = NetworkConfig()  # device=None triggers auto-detect
        device = config.get_device()
        assert device.type in ("cpu", "cuda")

    def test_default_device_is_none(self) -> None:
        """Default config should have device=None (auto-detect)."""
        config = NetworkConfig()
        assert config.device is None


# ---------------------------------------------------------------------------
# Custom configuration
# ---------------------------------------------------------------------------


class TestCustomConfig:
    """Verify that custom configurations work correctly."""

    def test_custom_values(self) -> None:
        config = NetworkConfig(num_blocks=12, num_filters=160)
        assert config.num_blocks == 12
        assert config.num_filters == 160
        # Defaults should still be applied for unspecified fields
        assert config.input_planes == 119
        assert config.policy_output_size == 4672

    def test_custom_input_planes(self) -> None:
        """Support alternative board encoding schemes."""
        config = NetworkConfig(input_planes=20)
        assert config.input_planes == 20

    def test_custom_param_count_scales(self) -> None:
        """Doubling filters should roughly quadruple residual block params."""
        config_a = NetworkConfig(num_blocks=10, num_filters=64)
        config_b = NetworkConfig(num_blocks=10, num_filters=128)

        # The residual block params scale as num_filters^2, so doubling
        # filters should roughly 4x the block contribution. The heads
        # add a constant offset, so the total ratio won't be exactly 4.
        ratio = config_b.estimate_param_count() / config_a.estimate_param_count()
        assert 2.0 < ratio < 5.0, f"Filter doubling ratio={ratio:.2f}, expected ~3-4x"


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------


class TestStringRepresentation:
    """Verify __str__ produces useful output."""

    def test_str_contains_key_info(self) -> None:
        s = str(FULL)
        assert "blocks=19" in s
        assert "filters=256" in s
        assert "estimated_params" in s

    def test_str_does_not_crash_for_any_preset(self) -> None:
        for preset in [TINY, SMALL, MEDIUM, FULL]:
            s = str(preset)
            assert len(s) > 0
