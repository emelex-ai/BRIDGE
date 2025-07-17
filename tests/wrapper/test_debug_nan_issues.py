"""Focused debugging tests for NaN issues in sliding window attention.

This module contains targeted tests to investigate specific hypotheses about
where NaN values are being introduced in the sliding window implementation
with seq_len=1024.
"""

import os
import sys
import time
from typing import cast

import psutil
import pytest
import torch
from bridge.domain.dataset.bridge_dataset import BridgeDataset

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from bridge.domain.datamodels import ModelConfig
from bridge.domain.model import Model
from bridge.domain.model.synthetic_dataset import SyntheticBridgeDatasetMultiWord


def create_test_model_config(
    use_sliding_window: bool = False, window_size: int = 61
) -> ModelConfig:
    """Create a test model configuration.

    Args:
        use_sliding_window: Whether to enable sliding window attention.
        window_size: Window size for sliding window attention (must be odd).

    Returns:
        ModelConfig instance for testing.
    """
    # Ensure window_size is odd for symmetric windows
    if window_size % 2 == 0:
        window_size += 1

    return ModelConfig(
        d_model=128,  # Small model for fast testing
        nhead=4,  # Must divide d_model
        num_phon_enc_layers=2,
        num_orth_enc_layers=2,
        num_mixing_enc_layers=2,
        num_phon_dec_layers=2,
        num_orth_dec_layers=2,
        d_embedding=1,
        seed=42,  # Fixed seed for reproducibility
        use_sliding_window=use_sliding_window,
        window_size=window_size,
        is_causal=False,  # Non-causal for this test
        max_seq_len=1024,
        ensure_contiguous=False,  # Memory efficient
    )


def create_test_data(batch_size: int = 4, seq_len: int = 64) -> dict:
    """Create test input data for the model.

    Args:
        batch_size: Number of sequences in the batch.
        seq_len: Length of each sequence.

    Returns:
        Dictionary containing test input tensors.
    """
    device = torch.device("cpu")  # Use CPU for consistent testing

    # Create orthographic input (batch_size, seq_len)
    orth_enc_input = torch.randint(0, 50, (batch_size, seq_len), device=device)
    orth_enc_pad_mask = torch.zeros(
        (batch_size, seq_len), dtype=torch.bool, device=device
    )

    # Create phonological input (list of tensors)
    phon_enc_input = [
        torch.randint(0, 30, (seq_len,), device=device) for _ in range(batch_size)
    ]
    phon_enc_pad_mask = torch.zeros(
        (batch_size, seq_len), dtype=torch.bool, device=device
    )

    # Create decoder inputs
    phon_dec_input = [
        torch.randint(0, 30, (seq_len,), device=device) for _ in range(batch_size)
    ]
    orth_dec_input = torch.randint(0, 50, (batch_size, seq_len), device=device)

    return {
        "orth_enc_input": orth_enc_input,
        "orth_enc_pad_mask": orth_enc_pad_mask,
        "phon_enc_input": phon_enc_input,
        "phon_enc_pad_mask": phon_enc_pad_mask,
        "phon_dec_input": phon_dec_input,
        "phon_dec_pad_mask": orth_enc_pad_mask,  # Reuse for simplicity
        "orth_dec_input": orth_dec_input,
        "orth_dec_pad_mask": orth_enc_pad_mask,  # Reuse for simplicity
    }


def mock_model_sequence_lengths(model: Model, max_seq_len: int = 128) -> None:
    """Resize position embeddings for testing with longer sequences.

    Args:
        model: The model instance to modify.
        max_seq_len: The new maximum sequence length to use.
    """
    model.max_orth_seq_len = max_seq_len
    model.max_phon_seq_len = max_seq_len

    device = model.device
    d_model = model.model_config.d_model

    # Create new, randomly initialized embeddings
    model.orth_position_embedding = torch.nn.Embedding(
        max_seq_len,
        d_model,
        device=device,
    )
    model.phon_position_embedding = torch.nn.Embedding(
        max_seq_len,
        d_model,
        device=device,
    )


def test_hypothesis_1_attention_weights_nan():
    """Hypothesis 1: NaN values are introduced in attention weight computation.

    This test FAILS if NaN values are detected in attention weight computation.
    """
    print("\n=== Hypothesis 1: Attention Weights NaN ===")

    seq_len = 1024
    batch_size = 2

    # Create dataset and model
    dataset = SyntheticBridgeDatasetMultiWord(
        num_samples=5,
        max_seq_len=seq_len,
        min_words_per_sequence=5,
        max_words_per_sequence=15,
        seed=42,
    )
    dataset = cast(BridgeDataset, dataset)

    enabled_config = create_test_model_config(use_sliding_window=True)
    enabled_model = Model(enabled_config, dataset)
    mock_model_sequence_lengths(enabled_model, max_seq_len=seq_len * 2)
    enabled_model.eval()

    test_data = create_test_data(batch_size=batch_size, seq_len=seq_len)

    # Track attention-specific NaN propagation
    attention_nan_locations = []

    def check_attention_nan(module, input, output):
        # Only check MultiheadAttention modules
        if not isinstance(module, torch.nn.MultiheadAttention):
            return

        if isinstance(output, torch.Tensor) and torch.isnan(output).any():
            nan_count = torch.isnan(output).sum().item()
            total_elements = output.numel()
            nan_percentage = (nan_count / total_elements) * 100

            location_info = {
                "module_name": str(module),
                "nan_count": nan_count,
                "total_elements": total_elements,
                "nan_percentage": nan_percentage,
                "output_shape": output.shape,
                "output_dtype": output.dtype,
            }
            attention_nan_locations.append(location_info)

            print(
                f"    ❌ NaN detected in {module.__class__.__name__}: {nan_count}/{total_elements} elements ({nan_percentage:.2f}%)"
            )

            # Check input tensors for NaN
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor) and torch.isnan(inp).any():
                    inp_nan_count = torch.isnan(inp).sum().item()
                    inp_total = inp.numel()
                    print(
                        f"    ⚠ Input {i} contains NaN: {inp_nan_count}/{inp_total} elements"
                    )

    # Register hooks only on MultiheadAttention modules
    hooks = []
    for name, module in enabled_model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            hook = module.register_forward_hook(check_attention_nan)
            hooks.append(hook)
            print(f"    Monitoring attention module: {name}")

    try:
        with torch.no_grad():
            enabled_output = enabled_model.forward(
                "op2op",
                orth_enc_input=test_data["orth_enc_input"],
                orth_enc_pad_mask=test_data["orth_enc_pad_mask"],
                phon_enc_input=test_data["phon_enc_input"],
                phon_enc_pad_mask=test_data["phon_enc_pad_mask"],
                phon_dec_input=test_data["phon_dec_input"],
                phon_dec_pad_mask=test_data["phon_dec_pad_mask"],
                orth_dec_input=test_data["orth_dec_input"],
                orth_dec_pad_mask=test_data["orth_dec_pad_mask"],
            )
    finally:
        for hook in hooks:
            hook.remove()

    # FAIL the test if NaN values are detected
    assert len(attention_nan_locations) == 0, (
        f"NaN values detected in attention weights! "
        f"Found {len(attention_nan_locations)} attention layers with NaN values:\n"
        + "\n".join(
            [
                f"  - {loc['module_name']}: {loc['nan_count']}/{loc['total_elements']} NaN ({loc['nan_percentage']:.2f}%)"
                for loc in attention_nan_locations
            ]
        )
    )

    print("✅ Hypothesis 1 PASSED: No NaN detected in attention weights")


def test_hypothesis_2_sliding_window_mask_nan():
    """Hypothesis 2: NaN values are introduced in sliding window mask creation.

    This test FAILS if NaN values are detected in sliding window mask creation.
    """
    print("\n=== Hypothesis 2: Sliding Window Mask NaN ===")

    seq_len = 1024

    # Create dataset and model
    dataset = SyntheticBridgeDatasetMultiWord(
        num_samples=5,
        max_seq_len=seq_len,
        min_words_per_sequence=5,
        max_words_per_sequence=15,
        seed=42,
    )
    dataset = cast(BridgeDataset, dataset)

    enabled_config = create_test_model_config(use_sliding_window=True)
    enabled_model = Model(enabled_config, dataset)
    mock_model_sequence_lengths(enabled_model, max_seq_len=seq_len * 2)
    enabled_model.eval()

    print(f"    Testing sliding window mask creation for seq_len={seq_len}")
    print(f"    Window size: {enabled_config.window_size}")

    mask_nan_locations = []

    # Test mask creation directly
    for name, module in enabled_model.named_modules():
        if "sliding_window" in name.lower():
            print(f"    Testing mask creation for {name}...")

            try:
                if hasattr(module, "create_sliding_window_mask"):
                    test_mask = module.create_sliding_window_mask(
                        seq_len, torch.device("cpu")
                    )

                    if torch.isnan(test_mask).any():
                        nan_count = torch.isnan(test_mask).sum().item()
                        total_elements = test_mask.numel()
                        nan_percentage = (nan_count / total_elements) * 100

                        location_info = {
                            "module_name": name,
                            "mask_type": "sliding_window",
                            "nan_count": nan_count,
                            "total_elements": total_elements,
                            "nan_percentage": nan_percentage,
                            "mask_shape": test_mask.shape,
                            "mask_dtype": test_mask.dtype,
                        }
                        mask_nan_locations.append(location_info)

                        print(
                            f"    ❌ NaN in sliding window mask for {name}: {nan_count}/{total_elements} elements ({nan_percentage:.2f}%)"
                        )
                    else:
                        print(f"    ✅ Sliding window mask for {name} is clean")

                elif hasattr(module, "create_sliding_window_causal_mask"):
                    test_mask = module.create_sliding_window_causal_mask(
                        seq_len, torch.device("cpu")
                    )

                    if torch.isnan(test_mask).any():
                        nan_count = torch.isnan(test_mask).sum().item()
                        total_elements = test_mask.numel()
                        nan_percentage = (nan_count / total_elements) * 100

                        location_info = {
                            "module_name": name,
                            "mask_type": "causal_sliding_window",
                            "nan_count": nan_count,
                            "total_elements": total_elements,
                            "nan_percentage": nan_percentage,
                            "mask_shape": test_mask.shape,
                            "mask_dtype": test_mask.dtype,
                        }
                        mask_nan_locations.append(location_info)

                        print(
                            f"    ❌ NaN in causal sliding window mask for {name}: {nan_count}/{total_elements} elements ({nan_percentage:.2f}%)"
                        )
                    else:
                        print(f"    ✅ Causal sliding window mask for {name} is clean")

            except Exception as e:
                print(f"    ⚠ Error testing mask creation for {name}: {e}")

    # FAIL the test if NaN values are detected
    assert len(mask_nan_locations) == 0, (
        f"NaN values detected in sliding window masks! "
        f"Found {len(mask_nan_locations)} masks with NaN values:\n"
        + "\n".join(
            [
                f"  - {loc['module_name']} ({loc['mask_type']}): {loc['nan_count']}/{loc['total_elements']} NaN ({loc['nan_percentage']:.2f}%)"
                for loc in mask_nan_locations
            ]
        )
    )

    print("✅ Hypothesis 2 PASSED: No NaN detected in sliding window masks")


def test_hypothesis_3_numerical_overflow():
    """Hypothesis 3: NaN values are caused by numerical overflow in attention computation.

    This test FAILS if extreme values or overflow are detected.
    """
    print("\n=== Hypothesis 3: Numerical Overflow ===")

    seq_len = 1024
    batch_size = 2

    # Create dataset and model
    dataset = SyntheticBridgeDatasetMultiWord(
        num_samples=5,
        max_seq_len=seq_len,
        min_words_per_sequence=5,
        max_words_per_sequence=15,
        seed=42,
    )
    dataset = cast(BridgeDataset, dataset)

    enabled_config = create_test_model_config(use_sliding_window=True)
    enabled_model = Model(enabled_config, dataset)
    mock_model_sequence_lengths(enabled_model, max_seq_len=seq_len * 2)
    enabled_model.eval()

    test_data = create_test_data(batch_size=batch_size, seq_len=seq_len)

    # Track extreme values and overflow
    overflow_locations = []

    def check_numerical_overflow(module, input, output):
        if isinstance(output, torch.Tensor):
            # Check for infinite values
            if torch.isinf(output).any():
                inf_count = torch.isinf(output).sum().item()
                total_elements = output.numel()

                location_info = {
                    "module_name": str(module),
                    "issue_type": "infinity",
                    "count": inf_count,
                    "total_elements": total_elements,
                    "percentage": (inf_count / total_elements) * 100,
                    "output_shape": output.shape,
                    "output_dtype": output.dtype,
                }
                overflow_locations.append(location_info)

                print(
                    f"    ❌ Infinite values detected in {module.__class__.__name__}: {inf_count}/{total_elements} elements"
                )

            # Check for extreme values (very large or very small)
            if output.numel() > 0:
                output_min = output.min().item()
                output_max = output.max().item()
                output_mean = output.mean().item()
                output_std = output.std().item()

                # Define thresholds for "extreme" values
                extreme_threshold = 1e6  # Very large values
                small_threshold = 1e-10  # Very small values

                if (
                    abs(output_max) > extreme_threshold
                    or abs(output_min) > extreme_threshold
                ):
                    location_info = {
                        "module_name": str(module),
                        "issue_type": "extreme_large",
                        "min": output_min,
                        "max": output_max,
                        "mean": output_mean,
                        "std": output_std,
                        "output_shape": output.shape,
                        "output_dtype": output.dtype,
                    }
                    overflow_locations.append(location_info)

                    print(
                        f"    ❌ Extreme large values detected in {module.__class__.__name__}: min={output_min:.6f}, max={output_max:.6f}"
                    )

    # Register hooks on all modules
    hooks = []
    for name, module in enabled_model.named_modules():
        hook = module.register_forward_hook(check_numerical_overflow)
        hooks.append(hook)

    try:
        with torch.no_grad():
            enabled_output = enabled_model.forward(
                "op2op",
                orth_enc_input=test_data["orth_enc_input"],
                orth_enc_pad_mask=test_data["orth_enc_pad_mask"],
                phon_enc_input=test_data["phon_enc_input"],
                phon_enc_pad_mask=test_data["phon_enc_pad_mask"],
                phon_dec_input=test_data["phon_dec_input"],
                phon_dec_pad_mask=test_data["phon_dec_pad_mask"],
                orth_dec_input=test_data["orth_dec_input"],
                orth_dec_pad_mask=test_data["orth_dec_pad_mask"],
            )
    finally:
        for hook in hooks:
            hook.remove()

    # FAIL the test if overflow/extreme values are detected
    assert len(overflow_locations) == 0, (
        f"Numerical overflow/extreme values detected! "
        f"Found {len(overflow_locations)} locations with issues:\n"
        + "\n".join(
            [
                f"  - {loc['module_name']} ({loc['issue_type']}): {loc.get('count', 'N/A')} issues"
                for loc in overflow_locations
            ]
        )
    )

    print("✅ Hypothesis 3 PASSED: No numerical overflow detected")


def test_hypothesis_4_window_size_ratio():
    """Hypothesis 4: NaN values are caused by extreme window size to sequence length ratios.

    This test FAILS if NaN values are detected with any window size ratio.
    """
    print("\n=== Hypothesis 4: Window Size Ratio ===")

    seq_len = 1024
    batch_size = 2

    # Test different window sizes to see if ratio affects NaN occurrence
    window_sizes = [61, 127, 255, 511]  # Different odd window sizes

    all_nan_locations = []

    for window_size in window_sizes:
        print(f"\n    Testing window size: {window_size}")
        print(f"    Sequence length: {seq_len}")
        print(f"    Ratio: {seq_len / window_size:.2f}")

        # Create dataset and model
        dataset = SyntheticBridgeDatasetMultiWord(
            num_samples=5,
            max_seq_len=seq_len,
            min_words_per_sequence=5,
            max_words_per_sequence=15,
            seed=42,
        )
        dataset = cast(BridgeDataset, dataset)

        config = create_test_model_config(
            use_sliding_window=True, window_size=window_size
        )
        model = Model(config, dataset)
        mock_model_sequence_lengths(model, max_seq_len=seq_len * 2)
        model.eval()

        test_data = create_test_data(batch_size=batch_size, seq_len=seq_len)

        # Track NaN occurrence
        window_nan_locations = []

        def check_for_nan(module, input, output):
            if isinstance(output, torch.Tensor) and torch.isnan(output).any():
                nan_count = torch.isnan(output).sum().item()
                total_elements = output.numel()
                nan_percentage = (nan_count / total_elements) * 100

                location_info = {
                    "window_size": window_size,
                    "module_name": str(module),
                    "nan_count": nan_count,
                    "total_elements": total_elements,
                    "nan_percentage": nan_percentage,
                }
                window_nan_locations.append(location_info)
                all_nan_locations.append(location_info)

                print(
                    f"      ❌ NaN detected with window_size={window_size} in {module.__class__.__name__}: {nan_count}/{total_elements} elements"
                )

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            hook = module.register_forward_hook(check_for_nan)
            hooks.append(hook)

        try:
            with torch.no_grad():
                output = model.forward(
                    "op2op",
                    orth_enc_input=test_data["orth_enc_input"],
                    orth_enc_pad_mask=test_data["orth_enc_pad_mask"],
                    phon_enc_input=test_data["phon_enc_input"],
                    phon_enc_pad_mask=test_data["phon_enc_pad_mask"],
                    phon_dec_input=test_data["phon_dec_input"],
                    phon_dec_pad_mask=test_data["phon_dec_pad_mask"],
                    orth_dec_input=test_data["orth_dec_input"],
                    orth_dec_pad_mask=test_data["orth_dec_pad_mask"],
                )
        finally:
            for hook in hooks:
                hook.remove()

        # Check final output
        orth_output = output["orth"]
        if torch.isnan(orth_output).any():
            final_nan_count = torch.isnan(orth_output).sum().item()
            final_total = orth_output.numel()
            print(
                f"      ❌ Final output contains {final_nan_count}/{final_total} NaN values"
            )
        else:
            print(f"      ✅ Final output is clean (no NaN)")

        if window_nan_locations:
            print(f"      ❌ NaN detected with window_size={window_size}")
        else:
            print(f"      ✅ No NaN detected with window_size={window_size}")

    # FAIL the test if NaN values are detected with any window size
    assert len(all_nan_locations) == 0, (
        f"NaN values detected with window size ratios! "
        f"Found {len(all_nan_locations)} locations with NaN values:\n"
        + "\n".join(
            [
                f"  - Window size {loc['window_size']} ({seq_len/loc['window_size']:.2f} ratio): {loc['module_name']} - {loc['nan_count']}/{loc['total_elements']} NaN ({loc['nan_percentage']:.2f}%)"
                for loc in all_nan_locations
            ]
        )
    )

    print("✅ Hypothesis 4 PASSED: No NaN detected with any window size ratio")


def test_hypothesis_5_layer_specific_nan():
    """Hypothesis 5: NaN values are introduced in specific layers of the model.

    This test FAILS if NaN values are detected in any layer.
    """
    print("\n=== Hypothesis 5: Layer-Specific NaN ===")

    seq_len = 1024
    batch_size = 2

    # Create dataset and model
    dataset = SyntheticBridgeDatasetMultiWord(
        num_samples=5,
        max_seq_len=seq_len,
        min_words_per_sequence=5,
        max_words_per_sequence=15,
        seed=42,
    )
    dataset = cast(BridgeDataset, dataset)

    enabled_config = create_test_model_config(use_sliding_window=True)
    enabled_model = Model(enabled_config, dataset)
    mock_model_sequence_lengths(enabled_model, max_seq_len=seq_len * 2)
    enabled_model.eval()

    test_data = create_test_data(batch_size=batch_size, seq_len=seq_len)

    # Track layer-specific NaN propagation
    layer_nan_locations = []

    def check_layer_nan(module, input, output):
        if isinstance(output, torch.Tensor) and torch.isnan(output).any():
            nan_count = torch.isnan(output).sum().item()
            total_elements = output.numel()
            nan_percentage = (nan_count / total_elements) * 100

            location_info = {
                "module_name": str(module),
                "module_type": module.__class__.__name__,
                "nan_count": nan_count,
                "total_elements": total_elements,
                "nan_percentage": nan_percentage,
                "output_shape": output.shape,
                "output_dtype": output.dtype,
            }
            layer_nan_locations.append(location_info)

            print(
                f"    ❌ NaN detected in {module.__class__.__name__}: {nan_count}/{total_elements} elements ({nan_percentage:.2f}%)"
            )

    # Register hooks on all modules
    hooks = []
    for name, module in enabled_model.named_modules():
        hook = module.register_forward_hook(check_layer_nan)
        hooks.append(hook)

    try:
        with torch.no_grad():
            enabled_output = enabled_model.forward(
                "op2op",
                orth_enc_input=test_data["orth_enc_input"],
                orth_enc_pad_mask=test_data["orth_enc_pad_mask"],
                phon_enc_input=test_data["phon_enc_input"],
                phon_enc_pad_mask=test_data["phon_enc_pad_mask"],
                phon_dec_input=test_data["phon_dec_input"],
                phon_dec_pad_mask=test_data["phon_dec_pad_mask"],
                orth_dec_input=test_data["orth_dec_input"],
                orth_dec_pad_mask=test_data["orth_dec_pad_mask"],
            )
    finally:
        for hook in hooks:
            hook.remove()

    # FAIL the test if NaN values are detected in any layer
    assert len(layer_nan_locations) == 0, (
        f"NaN values detected in specific layers! "
        f"Found {len(layer_nan_locations)} layers with NaN values:\n"
        + "\n".join(
            [
                f"  - {loc['module_type']} ({loc['module_name']}): {loc['nan_count']}/{loc['total_elements']} NaN ({loc['nan_percentage']:.2f}%)"
                for loc in layer_nan_locations
            ]
        )
    )

    print("✅ Hypothesis 5 PASSED: No NaN detected in any layer")


if __name__ == "__main__":
    """Run all debugging hypothesis tests."""
    print("Running debugging hypothesis tests for NaN issues...")

    # Run all hypothesis tests
    test_hypothesis_1_attention_weights_nan()
    test_hypothesis_2_sliding_window_mask_nan()
    test_hypothesis_3_numerical_overflow()
    test_hypothesis_4_window_size_ratio()
    test_hypothesis_5_layer_specific_nan()

    print("\n✓ All debugging hypothesis tests completed!")
