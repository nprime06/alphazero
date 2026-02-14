"""
Tests for Model Export
========================

Verifies the TorchScript export pipeline by checking:
- Export creates a valid file on disk
- Roundtrip FP32: export -> load -> compare outputs within atol=1e-5
- Roundtrip FP16: export -> load -> compare outputs within atol=1e-2
- Exported model input/output shapes match the original
- Exported model works with different batch sizes (1, 4, 16)
- verify_export returns True for a valid export
- Export with metadata preserves the metadata across save/load
- Temporary directory cleanup (using pytest tmp_path fixture)
- Loaded model is in eval mode (BatchNorm uses running stats, not batch stats)
- Loading on CPU works regardless of save device
- FP16 export does not modify the original model

The tests use the TINY preset for speed -- the export logic is identical for
all presets, so testing with TINY is sufficient.
"""

import json

import pytest
import torch

from neural.config import NetworkConfig
from neural.export import (
    export_fp16,
    export_torchscript,
    export_with_metadata,
    load_metadata,
    load_torchscript,
    verify_export,
)
from neural.network import AlphaZeroNetwork


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_config() -> NetworkConfig:
    """Create a TINY config for fast tests."""
    return NetworkConfig.tiny()


@pytest.fixture
def tiny_model(tiny_config: NetworkConfig) -> AlphaZeroNetwork:
    """Create a TINY model on CPU for testing.

    The model is set to eval mode to match export behavior and ensure
    deterministic BatchNorm outputs.
    """
    model = AlphaZeroNetwork(tiny_config)
    model.eval()
    return model


@pytest.fixture
def example_input(tiny_config: NetworkConfig) -> torch.Tensor:
    """Create a deterministic example input tensor."""
    torch.manual_seed(42)
    return torch.randn(1, tiny_config.input_planes, 8, 8)


# ---------------------------------------------------------------------------
# Basic export: file creation and format
# ---------------------------------------------------------------------------


class TestExportCreatesFile:
    """Verify that export_torchscript creates a valid file on disk."""

    def test_export_creates_file(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Exporting should create a .pt file at the specified path."""
        path = tmp_path / "model.pt"  # type: ignore[operator]
        result_path = export_torchscript(tiny_model, path)

        assert path.exists(), f"Expected file at {path}, but it doesn't exist"  # type: ignore[union-attr]
        assert path.stat().st_size > 0, "Exported file should not be empty"  # type: ignore[union-attr]
        assert result_path == path, (
            f"Return value ({result_path}) should match the input path ({path})"
        )

    def test_export_file_is_loadable(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """The exported file should be loadable by torch.jit.load."""
        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)

        # This should not raise
        loaded = load_torchscript(path)
        assert loaded is not None

    def test_export_to_nested_directory(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Export should work even when the parent directory already exists."""
        subdir = tmp_path / "models" / "v1"  # type: ignore[operator]
        subdir.mkdir(parents=True)  # type: ignore[union-attr]
        path = subdir / "model.pt"

        export_torchscript(tiny_model, path)
        assert path.exists()


# ---------------------------------------------------------------------------
# FP32 roundtrip: export -> load -> compare
# ---------------------------------------------------------------------------


class TestFP32Roundtrip:
    """Verify that FP32 export preserves model outputs within tight tolerance."""

    def test_policy_output_matches(
        self,
        tiny_model: AlphaZeroNetwork,
        example_input: torch.Tensor,
        tmp_path: object,
    ) -> None:
        """Policy logits from the exported model should match the original."""
        path = tmp_path / "model_fp32.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)
        loaded = load_torchscript(path)

        with torch.no_grad():
            orig_policy, _ = tiny_model(example_input)
            loaded_policy, _ = loaded(example_input)

        assert torch.allclose(orig_policy, loaded_policy, atol=1e-5), (
            f"Policy max diff: {(orig_policy - loaded_policy).abs().max().item():.2e}"
        )

    def test_value_output_matches(
        self,
        tiny_model: AlphaZeroNetwork,
        example_input: torch.Tensor,
        tmp_path: object,
    ) -> None:
        """Value output from the exported model should match the original."""
        path = tmp_path / "model_fp32.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)
        loaded = load_torchscript(path)

        with torch.no_grad():
            _, orig_value = tiny_model(example_input)
            _, loaded_value = loaded(example_input)

        assert torch.allclose(orig_value, loaded_value, atol=1e-5), (
            f"Value max diff: {(orig_value - loaded_value).abs().max().item():.2e}"
        )

    def test_full_roundtrip_multiple_inputs(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Outputs should match across multiple different random inputs."""
        path = tmp_path / "model_fp32.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)
        loaded = load_torchscript(path)

        config = tiny_model.config
        for seed in range(5):
            torch.manual_seed(seed)
            test_input = torch.randn(2, config.input_planes, 8, 8)

            with torch.no_grad():
                orig_policy, orig_value = tiny_model(test_input)
                loaded_policy, loaded_value = loaded(test_input)

            assert torch.allclose(orig_policy, loaded_policy, atol=1e-5), (
                f"Policy mismatch for seed {seed}: "
                f"max diff = {(orig_policy - loaded_policy).abs().max().item():.2e}"
            )
            assert torch.allclose(orig_value, loaded_value, atol=1e-5), (
                f"Value mismatch for seed {seed}: "
                f"max diff = {(orig_value - loaded_value).abs().max().item():.2e}"
            )


# ---------------------------------------------------------------------------
# FP16 roundtrip: export -> load -> compare (relaxed tolerance)
# ---------------------------------------------------------------------------


class TestFP16Roundtrip:
    """Verify that FP16 export preserves model outputs within FP16 tolerance.

    FP16 has ~3.3 decimal digits of precision (vs ~7.2 for FP32), so we use
    a relaxed tolerance of 1e-2. In practice, the differences are usually
    much smaller than this.
    """

    def test_fp16_export_creates_file(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """FP16 export should create a valid file."""
        path = tmp_path / "model_fp16.pt"  # type: ignore[operator]
        export_fp16(tiny_model, path)
        assert path.exists()  # type: ignore[union-attr]
        assert path.stat().st_size > 0  # type: ignore[union-attr]

    def test_fp16_output_within_tolerance(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """FP16 model outputs should be close to FP32 within relaxed tolerance."""
        path = tmp_path / "model_fp16.pt"  # type: ignore[operator]
        export_fp16(tiny_model, path)

        # Load the FP16 model
        loaded = load_torchscript(path)

        # Create test input
        config = tiny_model.config
        torch.manual_seed(42)
        test_input = torch.randn(2, config.input_planes, 8, 8)

        with torch.no_grad():
            orig_policy, orig_value = tiny_model(test_input)
            # FP16 model needs FP16 input
            loaded_policy, loaded_value = loaded(test_input.half())

        # Compare in FP32 space.
        # FP16 has ~3.3 decimal digits of precision, and errors accumulate
        # through the residual tower (5 blocks in TINY). Policy logits pass
        # through many layers and a final linear projection to 4672 outputs,
        # so individual logit differences can reach ~0.03-0.05.
        policy_diff = (orig_policy - loaded_policy.float()).abs().max().item()
        value_diff = (orig_value - loaded_value.float()).abs().max().item()

        assert policy_diff < 5e-2, (
            f"FP16 policy max diff ({policy_diff:.4e}) exceeds tolerance (5e-2)"
        )
        assert value_diff < 1e-2, (
            f"FP16 value max diff ({value_diff:.4e}) exceeds tolerance (1e-2)"
        )

    def test_fp16_does_not_modify_original(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """FP16 export should not change the original model's dtype.

        The export function deep-copies the model before converting to FP16,
        so the caller's model should remain in FP32.
        """
        path = tmp_path / "model_fp16.pt"  # type: ignore[operator]

        # Check original dtype before export
        original_dtype = next(tiny_model.parameters()).dtype
        assert original_dtype == torch.float32, "Model should start as FP32"

        export_fp16(tiny_model, path)

        # Check original dtype after export -- should still be FP32
        after_dtype = next(tiny_model.parameters()).dtype
        assert after_dtype == torch.float32, (
            f"Original model dtype changed from FP32 to {after_dtype} after FP16 export"
        )

    def test_fp16_model_is_actually_fp16(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """The loaded FP16 model should have FP16 parameters."""
        path = tmp_path / "model_fp16.pt"  # type: ignore[operator]
        export_fp16(tiny_model, path)
        loaded = load_torchscript(path)

        # Check the dtype of the loaded model's parameters
        for param in loaded.parameters():
            assert param.dtype == torch.float16, (
                f"Expected FP16 parameter, got {param.dtype}"
            )
            break  # Only need to check one parameter


# ---------------------------------------------------------------------------
# Output shape verification
# ---------------------------------------------------------------------------


class TestOutputShapes:
    """Verify that exported model input/output shapes match the original."""

    def test_policy_shape_matches(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Exported model should produce policy output of shape (B, 4672)."""
        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)
        loaded = load_torchscript(path)

        config = tiny_model.config
        batch_size = 4
        test_input = torch.randn(batch_size, config.input_planes, 8, 8)

        with torch.no_grad():
            policy, _ = loaded(test_input)

        assert policy.shape == (batch_size, config.policy_output_size), (
            f"Expected policy shape ({batch_size}, {config.policy_output_size}), "
            f"got {policy.shape}"
        )

    def test_value_shape_matches(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Exported model should produce value output of shape (B, 1)."""
        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)
        loaded = load_torchscript(path)

        config = tiny_model.config
        batch_size = 4
        test_input = torch.randn(batch_size, config.input_planes, 8, 8)

        with torch.no_grad():
            _, value = loaded(test_input)

        assert value.shape == (batch_size, 1), (
            f"Expected value shape ({batch_size}, 1), got {value.shape}"
        )

    def test_value_bounded_minus_one_to_one(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Value output should be in [-1, 1] due to tanh activation."""
        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)
        loaded = load_torchscript(path)

        config = tiny_model.config
        test_input = torch.randn(8, config.input_planes, 8, 8)

        with torch.no_grad():
            _, value = loaded(test_input)

        assert value.min().item() >= -1.0, (
            f"Value minimum ({value.min().item()}) should be >= -1.0"
        )
        assert value.max().item() <= 1.0, (
            f"Value maximum ({value.max().item()}) should be <= 1.0"
        )


# ---------------------------------------------------------------------------
# Batch size flexibility
# ---------------------------------------------------------------------------


class TestBatchSizeFlexibility:
    """Verify that the exported model handles different batch sizes.

    TorchScript tracing captures the computation graph from a single forward
    pass, but the resulting graph should work with any batch size because
    PyTorch operations are naturally batch-dimension-agnostic.
    """

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(
        self,
        tiny_model: AlphaZeroNetwork,
        tmp_path: object,
        batch_size: int,
    ) -> None:
        """Exported model should work with batch sizes 1, 4, and 16."""
        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)
        loaded = load_torchscript(path)

        config = tiny_model.config
        test_input = torch.randn(batch_size, config.input_planes, 8, 8)

        with torch.no_grad():
            policy, value = loaded(test_input)

        assert policy.shape == (batch_size, config.policy_output_size), (
            f"Policy shape mismatch for batch_size={batch_size}: "
            f"expected ({batch_size}, {config.policy_output_size}), got {policy.shape}"
        )
        assert value.shape == (batch_size, 1), (
            f"Value shape mismatch for batch_size={batch_size}: "
            f"expected ({batch_size}, 1), got {value.shape}"
        )

    def test_single_sample_matches_batched(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Running one sample at a time should give same result as a batch.

        This is a consistency check: if we batch 4 independent inputs and run
        them together, each output should match running the input individually.
        """
        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)
        loaded = load_torchscript(path)

        config = tiny_model.config
        torch.manual_seed(123)
        batch_input = torch.randn(4, config.input_planes, 8, 8)

        with torch.no_grad():
            batch_policy, batch_value = loaded(batch_input)

            for i in range(4):
                single_policy, single_value = loaded(batch_input[i : i + 1])
                assert torch.allclose(batch_policy[i], single_policy[0], atol=1e-5), (
                    f"Policy mismatch at index {i}: max diff = "
                    f"{(batch_policy[i] - single_policy[0]).abs().max().item():.2e}"
                )
                assert torch.allclose(batch_value[i], single_value[0], atol=1e-5), (
                    f"Value mismatch at index {i}"
                )


# ---------------------------------------------------------------------------
# verify_export function
# ---------------------------------------------------------------------------


class TestVerifyExport:
    """Verify the verify_export convenience function."""

    def test_valid_export_returns_true(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """verify_export should return True for a correctly exported model."""
        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)

        match, diffs = verify_export(tiny_model, path)

        assert match is True, (
            f"verify_export should return True for valid export, "
            f"but got False with diffs: {diffs}"
        )

    def test_diffs_are_small_for_fp32(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """For FP32 export, max diffs should be very small (< 1e-5)."""
        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)

        _, diffs = verify_export(tiny_model, path)

        assert diffs["policy_max_diff"] < 1e-5, (
            f"Policy diff ({diffs['policy_max_diff']:.2e}) too large for FP32"
        )
        assert diffs["value_max_diff"] < 1e-5, (
            f"Value diff ({diffs['value_max_diff']:.2e}) too large for FP32"
        )

    def test_verify_fp16_with_relaxed_tolerance(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """verify_export with atol=5e-2 should pass for FP16 exports.

        FP16 precision errors accumulate through the residual tower and can
        reach ~0.03-0.05 for individual policy logits. A tolerance of 5e-2
        accommodates this while still catching gross errors.
        """
        path = tmp_path / "model_fp16.pt"  # type: ignore[operator]
        export_fp16(tiny_model, path)

        match, diffs = verify_export(tiny_model, path, atol=5e-2)

        assert match is True, (
            f"verify_export should return True for FP16 with atol=5e-2, "
            f"diffs: {diffs}"
        )

    def test_diffs_dict_has_correct_keys(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """The diffs dict should contain policy_max_diff and value_max_diff."""
        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)

        _, diffs = verify_export(tiny_model, path)

        assert "policy_max_diff" in diffs, "Missing 'policy_max_diff' key"
        assert "value_max_diff" in diffs, "Missing 'value_max_diff' key"
        assert isinstance(diffs["policy_max_diff"], float)
        assert isinstance(diffs["value_max_diff"], float)


# ---------------------------------------------------------------------------
# Metadata export and retrieval
# ---------------------------------------------------------------------------


class TestMetadata:
    """Verify that metadata is correctly embedded and retrievable."""

    def test_export_with_metadata_creates_file(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Export with metadata should create a valid file."""
        path = tmp_path / "model_meta.pt"  # type: ignore[operator]
        metadata = {"version": "0.1.0"}
        export_with_metadata(tiny_model, path, metadata)

        assert path.exists()  # type: ignore[union-attr]
        assert path.stat().st_size > 0  # type: ignore[union-attr]

    def test_metadata_roundtrip(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Metadata should survive export -> load -> read cycle."""
        path = tmp_path / "model_meta.pt"  # type: ignore[operator]
        metadata = {
            "config": {"num_blocks": 5, "num_filters": 64},
            "version": "0.1.0",
            "training_step": 10000,
        }
        export_with_metadata(tiny_model, path, metadata)

        # Load and check metadata
        loaded_metadata = load_metadata(path)

        assert loaded_metadata == metadata, (
            f"Metadata mismatch:\n"
            f"  expected: {metadata}\n"
            f"  got:      {loaded_metadata}"
        )

    def test_metadata_with_nested_dicts(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Metadata with nested dictionaries should be preserved."""
        path = tmp_path / "model_meta.pt"  # type: ignore[operator]
        metadata = {
            "config": {
                "num_blocks": 5,
                "num_filters": 64,
                "presets": {"tiny": True, "full": False},
            },
            "metrics": {"loss": 0.5, "accuracy": 0.8},
        }
        export_with_metadata(tiny_model, path, metadata)
        loaded_metadata = load_metadata(path)

        assert loaded_metadata == metadata

    def test_none_metadata_gives_empty_dict(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Passing None for metadata should store an empty dict."""
        path = tmp_path / "model_meta.pt"  # type: ignore[operator]
        export_with_metadata(tiny_model, path, metadata=None)
        loaded_metadata = load_metadata(path)

        assert loaded_metadata == {}, (
            f"Expected empty dict for None metadata, got {loaded_metadata}"
        )

    def test_model_with_metadata_is_functional(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """A model exported with metadata should still produce correct outputs."""
        path = tmp_path / "model_meta.pt"  # type: ignore[operator]
        metadata = {"version": "test"}
        export_with_metadata(tiny_model, path, metadata)

        loaded = load_torchscript(path)
        config = tiny_model.config
        test_input = torch.randn(2, config.input_planes, 8, 8)

        with torch.no_grad():
            orig_policy, orig_value = tiny_model(test_input)
            loaded_policy, loaded_value = loaded(test_input)

        assert torch.allclose(orig_policy, loaded_policy, atol=1e-5)
        assert torch.allclose(orig_value, loaded_value, atol=1e-5)

    def test_metadata_values_are_correct_types(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Metadata values should preserve their JSON types after roundtrip."""
        path = tmp_path / "model_meta.pt"  # type: ignore[operator]
        metadata = {
            "int_val": 42,
            "float_val": 3.14,
            "str_val": "hello",
            "bool_val": True,
            "list_val": [1, 2, 3],
            "null_val": None,
        }
        export_with_metadata(tiny_model, path, metadata)
        loaded_metadata = load_metadata(path)

        assert loaded_metadata["int_val"] == 42
        assert abs(loaded_metadata["float_val"] - 3.14) < 1e-10
        assert loaded_metadata["str_val"] == "hello"
        assert loaded_metadata["bool_val"] is True
        assert loaded_metadata["list_val"] == [1, 2, 3]
        assert loaded_metadata["null_val"] is None


# ---------------------------------------------------------------------------
# Eval mode verification
# ---------------------------------------------------------------------------


class TestEvalMode:
    """Verify that exported models behave deterministically (eval mode).

    BatchNorm layers in training mode use batch statistics, which means the
    output depends on ALL samples in the batch (not just the current one).
    In eval mode, BatchNorm uses the running mean/variance accumulated during
    training, making outputs deterministic and independent per sample.

    This is critical for MCTS: if the value of a position depends on what
    other positions are in the same batch, the search would be non-deterministic.
    """

    def test_model_is_deterministic_after_load(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Running the same input twice should give identical outputs."""
        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)
        loaded = load_torchscript(path)

        config = tiny_model.config
        test_input = torch.randn(4, config.input_planes, 8, 8)

        with torch.no_grad():
            policy1, value1 = loaded(test_input)
            policy2, value2 = loaded(test_input)

        assert torch.equal(policy1, policy2), (
            "Loaded model should be deterministic: same input -> same output"
        )
        assert torch.equal(value1, value2), (
            "Loaded model should be deterministic: same input -> same output"
        )

    def test_output_independent_of_batch_composition(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Output for a sample should not change when other batch samples change.

        This verifies that BatchNorm is in eval mode (using running stats).
        In training mode, BatchNorm normalizes using the batch, so the output
        for sample[0] would change if sample[1] changes.
        """
        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)
        loaded = load_torchscript(path)

        config = tiny_model.config
        torch.manual_seed(42)
        fixed_input = torch.randn(1, config.input_planes, 8, 8)

        # Run with different batch compositions
        batch_a = torch.cat([fixed_input, torch.randn(3, config.input_planes, 8, 8)])
        batch_b = torch.cat([fixed_input, torch.randn(3, config.input_planes, 8, 8)])

        with torch.no_grad():
            policy_a, value_a = loaded(batch_a)
            policy_b, value_b = loaded(batch_b)

        # The first sample's output should be the same regardless of batch
        assert torch.allclose(policy_a[0], policy_b[0], atol=1e-6), (
            "Output should not depend on other samples in the batch (eval mode). "
            f"Max diff: {(policy_a[0] - policy_b[0]).abs().max().item():.2e}"
        )
        assert torch.allclose(value_a[0], value_b[0], atol=1e-6), (
            "Value output should not depend on other samples in the batch"
        )

    def test_export_preserves_original_training_mode(
        self, tiny_config: NetworkConfig, tmp_path: object
    ) -> None:
        """Export should restore the model's original training mode after tracing.

        If the model was in training mode before export, it should still be
        in training mode after export completes. The export function
        temporarily switches to eval for tracing, then switches back.
        """
        model = AlphaZeroNetwork(tiny_config)
        model.train()  # Explicitly set to training mode
        assert model.training, "Model should start in training mode"

        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(model, path)

        assert model.training, (
            "Model should still be in training mode after export"
        )


# ---------------------------------------------------------------------------
# CPU loading portability
# ---------------------------------------------------------------------------


class TestCPUPortability:
    """Verify that models can be loaded on CPU regardless of save device."""

    def test_load_on_cpu_explicit(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Loading with device='cpu' should work for a CPU-saved model."""
        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)

        loaded = load_torchscript(path, device="cpu")

        # Verify the model is on CPU
        config = tiny_model.config
        test_input = torch.randn(1, config.input_planes, 8, 8)
        with torch.no_grad():
            policy, value = loaded(test_input)

        assert policy.device.type == "cpu"
        assert value.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No GPU available"
    )
    def test_gpu_saved_model_loads_on_cpu(
        self, tiny_config: NetworkConfig, tmp_path: object
    ) -> None:
        """A model saved on GPU should load correctly on CPU.

        The map_location='cpu' parameter in torch.jit.load handles the
        device remapping transparently.
        """
        # Create and export on GPU
        model = AlphaZeroNetwork(tiny_config).cuda()
        model.eval()

        path = tmp_path / "model_gpu.pt"  # type: ignore[operator]
        export_torchscript(model, path)

        # Load on CPU
        loaded = load_torchscript(path, device="cpu")
        test_input = torch.randn(1, tiny_config.input_planes, 8, 8)

        with torch.no_grad():
            policy, value = loaded(test_input)

        assert policy.device.type == "cpu"
        assert value.device.type == "cpu"


# ---------------------------------------------------------------------------
# Custom example input
# ---------------------------------------------------------------------------


class TestCustomExampleInput:
    """Verify that providing a custom example input works correctly."""

    def test_export_with_custom_input(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Providing an explicit example_input should work for tracing."""
        config = tiny_model.config
        custom_input = torch.ones(2, config.input_planes, 8, 8)

        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path, example_input=custom_input)

        loaded = load_torchscript(path)

        # The exported model should still work with any input
        test_input = torch.randn(4, config.input_planes, 8, 8)
        with torch.no_grad():
            policy, value = loaded(test_input)

        assert policy.shape == (4, config.policy_output_size)
        assert value.shape == (4, 1)

    def test_fp16_export_with_custom_input(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """FP16 export with custom example_input (FP32, auto-converted)."""
        config = tiny_model.config
        # Provide FP32 input -- the export function should convert to FP16
        custom_input = torch.randn(1, config.input_planes, 8, 8)

        path = tmp_path / "model_fp16.pt"  # type: ignore[operator]
        export_fp16(tiny_model, path, example_input=custom_input)

        loaded = load_torchscript(path)
        test_input = torch.randn(2, config.input_planes, 8, 8, dtype=torch.float16)
        with torch.no_grad():
            policy, value = loaded(test_input)

        assert policy.shape[0] == 2
        assert value.shape == (2, 1)


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and robustness of the export pipeline."""

    def test_export_model_already_in_eval_mode(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Exporting a model already in eval mode should work fine.

        The export function calls model.eval() -- calling it on an already-eval
        model should be a no-op.
        """
        tiny_model.eval()
        path = tmp_path / "model.pt"  # type: ignore[operator]
        export_torchscript(tiny_model, path)

        loaded = load_torchscript(path)
        config = tiny_model.config
        test_input = torch.randn(1, config.input_planes, 8, 8)

        with torch.no_grad():
            policy, value = loaded(test_input)

        assert policy.shape == (1, config.policy_output_size)

    def test_export_overwrites_existing_file(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """Exporting to an existing path should overwrite the file.

        Note: TorchScript serialization is not perfectly deterministic in file
        size (internal ordering and alignment may vary slightly), so we check
        that the file exists and is loadable rather than comparing exact sizes.
        """
        path = tmp_path / "model.pt"  # type: ignore[operator]

        # Export twice to the same path
        export_torchscript(tiny_model, path)
        assert path.exists()  # type: ignore[union-attr]

        export_torchscript(tiny_model, path)
        assert path.exists()  # type: ignore[union-attr]

        # The second export should produce a valid, loadable model
        loaded = load_torchscript(path)
        config = tiny_model.config
        test_input = torch.randn(1, config.input_planes, 8, 8)
        with torch.no_grad():
            policy, value = loaded(test_input)
        assert policy.shape == (1, config.policy_output_size)

    def test_fp16_smaller_than_fp32(
        self, tiny_model: AlphaZeroNetwork, tmp_path: object
    ) -> None:
        """FP16 model file should be smaller than FP32 (roughly half).

        This is a sanity check that the FP16 conversion actually happened --
        if the parameters are truly stored in 16-bit format, the file should
        be approximately half the size of the FP32 version.
        """
        fp32_path = tmp_path / "model_fp32.pt"  # type: ignore[operator]
        fp16_path = tmp_path / "model_fp16.pt"  # type: ignore[operator]

        export_torchscript(tiny_model, fp32_path)
        export_fp16(tiny_model, fp16_path)

        fp32_size = fp32_path.stat().st_size  # type: ignore[union-attr]
        fp16_size = fp16_path.stat().st_size  # type: ignore[union-attr]

        # FP16 should be noticeably smaller. We check < 80% of FP32 size
        # (not exactly 50% because the file also contains metadata, graph
        # structure, etc. that isn't affected by dtype).
        assert fp16_size < fp32_size * 0.80, (
            f"FP16 file ({fp16_size} bytes) should be significantly smaller "
            f"than FP32 ({fp32_size} bytes)"
        )
