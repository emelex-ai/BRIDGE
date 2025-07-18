"""Focused test to debug sliding window mask usage issues.

This test specifically investigates how sliding window masks are being used
and why they're causing NaN values in the model.
"""

import os
import sys
import time
from typing import cast

import pytest
import torch
from bridge.domain.dataset.bridge_dataset import BridgeDataset

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from bridge.domain.datamodels import ModelConfig
from bridge.domain.model import Model
from bridge.domain.model.synthetic_dataset import SyntheticBridgeDatasetMultiWord

from tests.wrapper.utils import create_test_data


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


'''
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
'''


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


def test_mask_creation_and_usage():
    """Test to investigate how sliding window masks are created and used.

    This test focuses on the specific issue where masks contain -inf values
    that might be causing NaN propagation in the model.
    """
    print("\n=== Test: Mask Creation and Usage ===")

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

    # Track mask creation and usage
    mask_creation_info = []
    mask_usage_info = []

    def track_mask_creation(module, input, output):
        # Track when masks are created
        if hasattr(module, "create_sliding_window_mask") or hasattr(
            module, "create_sliding_window_causal_mask"
        ):
            mask_creation_info.append(
                {
                    "module_name": str(module),
                    "module_type": module.__class__.__name__,
                    "has_sliding_window": "sliding_window" in str(module).lower(),
                }
            )

    def track_mask_usage(module, input, output):
        # Track how masks are used in attention
        if isinstance(module, torch.nn.MultiheadAttention):
            # Check if any input contains -inf values (from masks)
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    if torch.isinf(inp).any():
                        inf_count = torch.isinf(inp).sum().item()
                        total_elements = inp.numel()
                        inf_percentage = (inf_count / total_elements) * 100

                        mask_usage_info.append(
                            {
                                "module_name": str(module),
                                "input_index": i,
                                "inf_count": inf_count,
                                "total_elements": total_elements,
                                "inf_percentage": inf_percentage,
                                "input_shape": inp.shape,
                                "input_dtype": inp.dtype,
                            }
                        )

                        print(
                            f"    ⚠ MultiheadAttention input {i} contains {inf_count}/{total_elements} infinite values ({inf_percentage:.2f}%)"
                        )

                        # Check if these are -inf values (expected from masks)
                        if torch.isneginf(inp).any():
                            neg_inf_count = torch.isneginf(inp).sum().item()
                            print(
                                f"    ✓ {neg_inf_count} of these are -inf (expected from attention masks)"
                            )
                        else:
                            print(f"    ❌ These are NOT -inf values (unexpected!)")

    # Register hooks
    hooks = []
    for name, module in enabled_model.named_modules():
        # Hook for mask creation tracking
        hook1 = module.register_forward_hook(track_mask_creation)
        hooks.append(hook1)

        # Hook for mask usage tracking
        hook2 = module.register_forward_hook(track_mask_usage)
        hooks.append(hook2)

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

    # Report mask creation info
    print(f"\n  Mask Creation Info:")
    for info in mask_creation_info:
        print(f"    - {info['module_type']}: {info['module_name']}")

    # Report mask usage info
    print(f"\n  Mask Usage Info:")
    for info in mask_usage_info:
        print(
            f"    - {info['module_name']} input {info['input_index']}: {info['inf_count']}/{info['total_elements']} inf ({info['inf_percentage']:.2f}%)"
        )

    # Check final output for NaN
    orth_output = enabled_output["orth"]
    if torch.isnan(orth_output).any():
        nan_count = torch.isnan(orth_output).sum().item()
        total_elements = orth_output.numel()
        print(f"\n  ❌ Final output contains {nan_count}/{total_elements} NaN values")

        # This test should FAIL if NaN is detected
        assert (
            False
        ), f"NaN values detected in final output: {nan_count}/{total_elements} elements"
    else:
        print(f"\n  ✅ Final output is clean (no NaN)")

    print("✓ Mask creation and usage test completed")


def test_attention_mask_format():
    """Test to verify that attention masks are in the correct format.

    This test checks if the attention masks are being created and used
    in the correct format for PyTorch's attention mechanisms.
    """
    print("\n=== Test: Attention Mask Format ===")

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

    # Test mask creation directly
    print(f"    Testing attention mask format for seq_len={seq_len}")
    print(f"    Window size: {enabled_config.window_size}")

    mask_format_issues = []

    for name, module in enabled_model.named_modules():
        if "sliding_window" in name.lower():
            print(f"    Testing mask creation for {name}...")

            try:
                if hasattr(module, "create_sliding_window_mask"):
                    test_mask = module.create_sliding_window_mask(
                        seq_len, torch.device("cpu")
                    )

                    # Check mask format
                    unique_values = torch.unique(test_mask)
                    print(f"    ✓ Mask unique values: {unique_values}")

                    # Verify we have 0.0 and -inf values
                    expected_values = torch.tensor([float("-inf"), 0.0])
                    if not torch.allclose(unique_values, expected_values, atol=1e-6):
                        mask_format_issues.append(
                            {
                                "module_name": name,
                                "mask_type": "sliding_window",
                                "actual_values": unique_values.tolist(),
                                "expected_values": expected_values.tolist(),
                            }
                        )
                        print(f"    ❌ Unexpected mask values: {unique_values}")
                    else:
                        print(f"    ✅ Mask format is correct")

                elif hasattr(module, "create_sliding_window_causal_mask"):
                    test_mask = module.create_sliding_window_causal_mask(
                        seq_len, torch.device("cpu")
                    )

                    # Check mask format
                    unique_values = torch.unique(test_mask)
                    print(f"    ✓ Causal mask unique values: {unique_values}")

                    # Verify we have 0.0 and -inf values
                    expected_values = torch.tensor([float("-inf"), 0.0])
                    if not torch.allclose(unique_values, expected_values, atol=1e-6):
                        mask_format_issues.append(
                            {
                                "module_name": name,
                                "mask_type": "causal_sliding_window",
                                "actual_values": unique_values.tolist(),
                                "expected_values": expected_values.tolist(),
                            }
                        )
                        print(f"    ❌ Unexpected causal mask values: {unique_values}")
                    else:
                        print(f"    ✅ Causal mask format is correct")

            except Exception as e:
                print(f"    ⚠ Error testing mask creation for {name}: {e}")

    # FAIL the test if mask format issues are detected
    assert len(mask_format_issues) == 0, (
        f"Attention mask format issues detected! "
        f"Found {len(mask_format_issues)} masks with incorrect format:\n"
        + "\n".join(
            [
                f"  - {issue['module_name']} ({issue['mask_type']}): got {issue['actual_values']}, expected {issue['expected_values']}"
                for issue in mask_format_issues
            ]
        )
    )

    print("✅ Attention mask format test completed")


def test_numerical_stability():
    """Test to check for numerical stability issues in attention computation.

    This test specifically looks for issues that might cause NaN values
    when using sliding window attention masks.
    """
    print("\n=== Test: Numerical Stability ===")

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

    # Track numerical stability issues
    stability_issues = []

    def check_numerical_stability(module, input, output):
        if isinstance(output, torch.Tensor):
            # Check for NaN
            if torch.isnan(output).any():
                nan_count = torch.isnan(output).sum().item()
                total_elements = output.numel()

                stability_issues.append(
                    {
                        "module_name": str(module),
                        "module_type": module.__class__.__name__,
                        "issue_type": "nan",
                        "count": nan_count,
                        "total_elements": total_elements,
                        "percentage": (nan_count / total_elements) * 100,
                        "output_shape": output.shape,
                        "output_dtype": output.dtype,
                    }
                )

                print(
                    f"    ❌ NaN detected in {module.__class__.__name__}: {nan_count}/{total_elements} elements"
                )

            # Check for infinite values (except -inf which is expected in masks)
            if torch.isinf(output).any():
                inf_count = torch.isinf(output).sum().item()
                pos_inf_count = torch.isposinf(output).sum().item()
                neg_inf_count = torch.isneginf(output).sum().item()
                total_elements = output.numel()

                if pos_inf_count > 0:  # Positive infinity is problematic
                    stability_issues.append(
                        {
                            "module_name": str(module),
                            "module_type": module.__class__.__name__,
                            "issue_type": "positive_infinity",
                            "count": pos_inf_count,
                            "total_elements": total_elements,
                            "percentage": (pos_inf_count / total_elements) * 100,
                            "output_shape": output.shape,
                            "output_dtype": output.dtype,
                        }
                    )

                    print(
                        f"    ❌ Positive infinity detected in {module.__class__.__name__}: {pos_inf_count}/{total_elements} elements"
                    )

                if neg_inf_count > 0 and not isinstance(
                    module, torch.nn.MultiheadAttention
                ):
                    # Negative infinity is expected in attention masks but not elsewhere
                    stability_issues.append(
                        {
                            "module_name": str(module),
                            "module_type": module.__class__.__name__,
                            "issue_type": "negative_infinity",
                            "count": neg_inf_count,
                            "total_elements": total_elements,
                            "percentage": (neg_inf_count / total_elements) * 100,
                            "output_shape": output.shape,
                            "output_dtype": output.dtype,
                        }
                    )

                    print(
                        f"    ❌ Negative infinity detected in {module.__class__.__name__}: {neg_inf_count}/{total_elements} elements"
                    )

    # Register hooks on all modules
    hooks = []
    for name, module in enabled_model.named_modules():
        hook = module.register_forward_hook(check_numerical_stability)
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

    # FAIL the test if numerical stability issues are detected
    assert len(stability_issues) == 0, (
        f"Numerical stability issues detected! "
        f"Found {len(stability_issues)} issues:\n"
        + "\n".join(
            [
                f"  - {issue['module_type']} ({issue['module_name']}): {issue['issue_type']} - {issue['count']}/{issue['total_elements']} elements ({issue['percentage']:.2f}%)"
                for issue in stability_issues
            ]
        )
    )

    print("✅ Numerical stability test completed")


if __name__ == "__main__":
    """Run all mask debugging tests."""
    print("Running mask debugging tests...")

    # Run all tests
    test_mask_creation_and_usage()
    test_attention_mask_format()
    test_numerical_stability()

    print("\n✓ All mask debugging tests completed!")
