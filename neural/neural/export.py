"""
Model Export for Inference
===========================

Exports the AlphaZero neural network for fast inference, primarily via TorchScript
for use in the Rust MCTS engine (via the ``tch-rs`` crate).

Why TorchScript?
    During self-play, the MCTS engine (written in Rust) needs to evaluate thousands
    of positions per second. Running Python for each evaluation would be far too
    slow due to the Python interpreter overhead and the GIL. TorchScript compiles
    the model into a standalone representation that can be loaded and executed from
    C++ or Rust without any Python dependency.

    There are two ways to create a TorchScript model:

    1. **torch.jit.trace** -- Records the operations performed during a single
       forward pass with example inputs. This is what we use here because our
       model has a fixed computation graph (no data-dependent control flow like
       if-statements that depend on tensor values). Tracing is more reliable and
       produces cleaner graphs for models with straightforward architectures.

    2. **torch.jit.script** -- Analyzes the Python source code and compiles it
       directly. This handles dynamic control flow but is more fragile and has
       many restrictions on what Python constructs are supported. We don't need
       it because our model's forward pass is the same for every input.

Why FP16 export?
    FP16 (half-precision floating point) uses 16 bits per number instead of 32,
    which provides several benefits for GPU inference:

    - **2x memory reduction**: The model takes half the VRAM, leaving more room
      for larger batch sizes during MCTS.
    - **~2x throughput on GPU**: Modern GPUs (especially NVIDIA Tensor Cores) can
      process FP16 operations roughly twice as fast as FP32.
    - **Minimal accuracy impact**: Neural network inference is remarkably robust
      to reduced precision. The policy and value outputs typically differ by less
      than 1% between FP32 and FP16.

    Important caveat: FP16 is primarily beneficial on GPU. On CPU, FP16 arithmetic
    is typically *slower* than FP32 because most CPUs lack native FP16 ALUs and
    must convert to FP32 internally.

Metadata storage:
    TorchScript archives support "extra files" -- arbitrary string data stored
    alongside the model. We use this to embed metadata (network config, version,
    training step, etc.) so that when a model is loaded later, you can inspect
    what configuration produced it without needing a separate metadata file.

Usage example::

    from neural.config import NetworkConfig
    from neural.network import AlphaZeroNetwork
    from neural.export import export_torchscript, load_torchscript, verify_export

    # Create and export a model
    config = NetworkConfig.tiny()
    model = AlphaZeroNetwork.from_config(config)
    export_torchscript(model, "model.pt")

    # Load and use the exported model
    traced = load_torchscript("model.pt")
    policy, value = traced(torch.randn(1, 119, 8, 8))

    # Verify the export matches the original
    match, diffs = verify_export(model, "model.pt")
    assert match  # True if outputs match within tolerance
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from neural.config import NetworkConfig


# ---------------------------------------------------------------------------
# TorchScript export (FP32)
# ---------------------------------------------------------------------------


def export_torchscript(
    model: nn.Module,
    path: Union[str, Path],
    example_input: Optional[torch.Tensor] = None,
) -> Path:
    """Export the model to TorchScript format via tracing.

    Tracing records the operations that the model performs on a concrete example
    input. The result is a self-contained file that can be loaded from C++, Rust
    (via tch-rs), or Python without the original model class definition.

    The model is automatically set to eval mode before tracing. This is critical
    because BatchNorm layers behave differently in train vs eval mode:
    - **Train mode**: normalizes using the current batch's mean/variance (noisy)
    - **Eval mode**: normalizes using the running mean/variance accumulated
      during training (deterministic)

    Using train mode would cause the exported model to produce different outputs
    depending on the batch composition, which would be a subtle and hard-to-debug
    correctness issue in MCTS.

    Args:
        model: The AlphaZeroNetwork (or any nn.Module) to export. Must have a
            ``config`` attribute with ``input_planes`` if no example_input is
            provided.
        path: File path to save the TorchScript model. Convention is ``.pt``
            extension.
        example_input: Optional tensor to use for tracing. If None, creates a
            dummy input of shape ``(1, config.input_planes, 8, 8)`` on the same
            device as the model. You may want to provide a custom input if your
            model uses a non-standard input shape.

    Returns:
        Path: The resolved path where the model was saved (useful when the input
        path was relative).

    Example::

        >>> config = NetworkConfig.tiny()
        >>> model = AlphaZeroNetwork(config)
        >>> saved_path = export_torchscript(model, "model_tiny.pt")
        >>> print(f"Model saved to {saved_path}")
    """
    path = Path(path)

    # Ensure eval mode for deterministic BatchNorm behavior.
    # We do NOT modify the original model's mode -- we work on a copy-like
    # state by switching and switching back if needed, but since tracing
    # captures a snapshot, this is safe.
    was_training = model.training
    model.eval()

    # Create example input on the same device as the model
    if example_input is None:
        # Infer device from model parameters
        device = next(model.parameters()).device
        config = model.config  # type: ignore[attr-defined]
        example_input = torch.randn(1, config.input_planes, 8, 8, device=device)

    # Trace the model -- this records every operation performed during
    # a single forward pass. The resulting graph is specialized to the
    # shapes and dtypes of the example input, but works for any batch size
    # because PyTorch traces dynamic dimensions correctly.
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    # Save the traced model to disk
    traced.save(str(path))

    # Restore original training mode if it was training before
    if was_training:
        model.train()

    return path


# ---------------------------------------------------------------------------
# TorchScript loading
# ---------------------------------------------------------------------------


def load_torchscript(
    path: Union[str, Path],
    device: str = "cpu",
) -> torch.jit.ScriptModule:
    """Load a TorchScript model from disk.

    The ``map_location`` parameter ensures the model is loaded onto the
    specified device regardless of where it was originally saved. This is
    important for portability: a model saved on GPU can be loaded on a
    CPU-only machine (and vice versa).

    Args:
        path: Path to the ``.pt`` TorchScript file.
        device: Device to load the model onto. Defaults to ``"cpu"`` for
            maximum portability. Use ``"cuda"`` or ``"cuda:0"`` for GPU
            inference.

    Returns:
        torch.jit.ScriptModule: The loaded model, ready for inference. The
        model is in eval mode by default (TorchScript models preserve the
        mode they were saved in, and we always save in eval mode).

    Example::

        >>> model = load_torchscript("model_tiny.pt", device="cpu")
        >>> policy, value = model(torch.randn(1, 119, 8, 8))
    """
    path = Path(path)
    loaded = torch.jit.load(str(path), map_location=device)
    return loaded


# ---------------------------------------------------------------------------
# FP16 export
# ---------------------------------------------------------------------------


def export_fp16(
    model: nn.Module,
    path: Union[str, Path],
    example_input: Optional[torch.Tensor] = None,
) -> Path:
    """Export the model in FP16 (half-precision) format.

    FP16 reduces memory usage by 2x and can provide ~2x inference speedup on
    GPUs with Tensor Cores (e.g., NVIDIA V100, A100, H100, H200). The precision
    reduction is generally negligible for neural network inference -- policy
    outputs typically agree to within 1%, and value outputs to within 0.01.

    Implementation details:
    - The model is deep-copied and converted to FP16 via ``.half()``
    - The example input is also converted to FP16
    - The original model is NOT modified (we work on a copy)

    Important caveat:
        FP16 inference on CPU is typically SLOWER than FP32 because most x86
        CPUs lack native FP16 compute units. The CPU must convert FP16 -> FP32
        -> compute -> FP32 -> FP16, adding overhead. Use FP16 only when
        deploying on GPU.

    Args:
        model: The network to export. Will not be modified (a copy is used).
        path: File path to save the FP16 TorchScript model.
        example_input: Optional FP16 tensor for tracing. If None, creates a
            dummy input and converts it to FP16.

    Returns:
        Path: The resolved path where the FP16 model was saved.

    Example::

        >>> model = AlphaZeroNetwork(NetworkConfig.tiny())
        >>> export_fp16(model, "model_tiny_fp16.pt")
    """
    import copy

    path = Path(path)

    # Deep copy the model so we don't modify the original. Converting to
    # half-precision changes all parameter tensors in-place, so without a
    # copy we'd corrupt the caller's model.
    model_fp16 = copy.deepcopy(model)
    model_fp16.eval()
    model_fp16.half()

    # Create or convert the example input to FP16
    if example_input is None:
        device = next(model_fp16.parameters()).device
        config = model_fp16.config  # type: ignore[attr-defined]
        example_input = torch.randn(
            1, config.input_planes, 8, 8, device=device, dtype=torch.float16
        )
    else:
        example_input = example_input.half()

    # Trace and save the FP16 model
    with torch.no_grad():
        traced = torch.jit.trace(model_fp16, example_input)

    traced.save(str(path))

    return path


# ---------------------------------------------------------------------------
# Export verification
# ---------------------------------------------------------------------------


def verify_export(
    original_model: nn.Module,
    exported_path: Union[str, Path],
    device: str = "cpu",
    atol: float = 1e-5,
) -> Tuple[bool, Dict[str, float]]:
    """Verify that an exported model produces the same outputs as the original.

    This is a critical safety check before deploying a model. Subtle issues in
    the export process (wrong eval mode, tracing errors, precision loss) could
    cause the exported model to behave differently from the original. This
    function catches such issues by comparing outputs on the same input.

    The comparison is done element-wise with an absolute tolerance. For FP32
    exports, a tolerance of 1e-5 is appropriate (differences arise from
    operation reordering during tracing). For FP16 exports, use a larger
    tolerance like 1e-2.

    Args:
        original_model: The original PyTorch model (in FP32).
        exported_path: Path to the exported TorchScript model file.
        device: Device to run the comparison on. Both models are moved to
            this device.
        atol: Absolute tolerance for comparing outputs. Use 1e-5 for FP32,
            1e-2 for FP16.

    Returns:
        A tuple ``(match, diffs)`` where:
        - ``match``: True if both policy and value outputs are within tolerance.
        - ``diffs``: Dict with ``"policy_max_diff"`` and ``"value_max_diff"``
          showing the maximum absolute difference for each head. Useful for
          debugging when the match fails.

    Example::

        >>> model = AlphaZeroNetwork(NetworkConfig.tiny())
        >>> export_torchscript(model, "model.pt")
        >>> match, diffs = verify_export(model, "model.pt")
        >>> print(f"Match: {match}, diffs: {diffs}")
        Match: True, diffs: {'policy_max_diff': 0.0, 'value_max_diff': 0.0}
    """
    exported_path = Path(exported_path)

    # Load the exported model
    loaded = load_torchscript(exported_path, device=device)

    # Ensure the original model is in eval mode and on the correct device
    original_model.eval()
    original_model = original_model.to(device)

    # Create a deterministic test input (fixed seed for reproducibility)
    config = original_model.config  # type: ignore[attr-defined]

    # Determine the dtype of the loaded model by checking its parameters
    loaded_dtype = torch.float32
    for param in loaded.parameters():
        loaded_dtype = param.dtype
        break

    test_input = torch.randn(
        2, config.input_planes, 8, 8, device=device, dtype=torch.float32
    )

    # Run both models
    with torch.no_grad():
        orig_policy, orig_value = original_model(test_input)

        # If the loaded model is FP16, we need FP16 input
        if loaded_dtype == torch.float16:
            loaded_policy, loaded_value = loaded(test_input.half())
            # Compare in FP32 space for meaningful absolute differences
            loaded_policy = loaded_policy.float()
            loaded_value = loaded_value.float()
        else:
            loaded_policy, loaded_value = loaded(test_input)

    # Compute max absolute differences
    policy_max_diff = (orig_policy - loaded_policy).abs().max().item()
    value_max_diff = (orig_value - loaded_value).abs().max().item()

    diffs = {
        "policy_max_diff": policy_max_diff,
        "value_max_diff": value_max_diff,
    }

    match = policy_max_diff <= atol and value_max_diff <= atol

    return match, diffs


# ---------------------------------------------------------------------------
# Export with metadata
# ---------------------------------------------------------------------------


def export_with_metadata(
    model: nn.Module,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export a TorchScript model with embedded metadata.

    TorchScript archives support "extra files" -- arbitrary string data stored
    alongside the serialized model. This is useful for embedding metadata like
    the network configuration, training step, version string, or any other
    information that helps identify and use the model later.

    The metadata is stored as a JSON string in an extra file named
    ``"metadata.json"`` inside the TorchScript archive. When loading the model
    later, you can retrieve this metadata using ``torch.jit.load`` with the
    ``_extra_files`` parameter.

    Why embed metadata?
        When managing many model checkpoints (e.g., during a long training run),
        it's easy to lose track of which configuration produced which model.
        Embedding the config directly in the model file means the model is
        self-describing -- you can always inspect it to find out its architecture,
        training history, and other relevant details.

    Args:
        model: The network to export.
        path: File path to save the model.
        metadata: Optional dictionary of metadata to embed. Common keys:
            - ``"config"``: Network configuration dict
            - ``"version"``: Model version string
            - ``"training_step"``: Training step when this checkpoint was saved
            - ``"timestamp"``: When the model was exported
            If None, no metadata is stored (but the extra file key is still
            created with an empty JSON object for consistency).

    Returns:
        Path: The resolved path where the model was saved.

    Example::

        >>> model = AlphaZeroNetwork(NetworkConfig.tiny())
        >>> metadata = {
        ...     "config": {"num_blocks": 5, "num_filters": 64},
        ...     "version": "0.1.0",
        ...     "training_step": 10000,
        ... }
        >>> export_with_metadata(model, "model_with_meta.pt", metadata)
    """
    path = Path(path)

    if metadata is None:
        metadata = {}

    # Serialize metadata to JSON string
    metadata_json = json.dumps(metadata, indent=2)

    # Set the model to eval mode for tracing
    was_training = model.training
    model.eval()

    # Create example input for tracing
    device = next(model.parameters()).device
    config = model.config  # type: ignore[attr-defined]
    example_input = torch.randn(1, config.input_planes, 8, 8, device=device)

    # Trace the model
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    # Save with extra files containing the metadata.
    # The extra_files dict maps filename -> content (as bytes or str).
    extra_files = {"metadata.json": metadata_json}
    traced.save(str(path), _extra_files=extra_files)

    # Restore original training mode
    if was_training:
        model.train()

    return path


def load_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Load metadata from a TorchScript model file.

    This is a convenience function for reading the metadata embedded by
    ``export_with_metadata`` without loading the full model. However, note
    that ``torch.jit.load`` still loads the model internally -- there is no
    way to read only the extra files without loading the model.

    Args:
        path: Path to the TorchScript model file.

    Returns:
        Dict: The metadata dictionary that was embedded during export.
            Returns an empty dict if no metadata was stored.

    Example::

        >>> metadata = load_metadata("model_with_meta.pt")
        >>> print(metadata["config"])
        {'num_blocks': 5, 'num_filters': 64}
    """
    path = Path(path)

    # Prepare the extra_files dict -- keys must exist, values are filled by load
    extra_files = {"metadata.json": ""}
    torch.jit.load(str(path), _extra_files=extra_files)

    metadata_json = extra_files["metadata.json"]
    if metadata_json:
        return json.loads(metadata_json)
    return {}
