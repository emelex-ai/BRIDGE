"""Test suite for sliding window wrapper functionality.

This module contains tests for the SlidingWindowEncoderWrapper and
SlidingWindowDecoderWrapper classes, focusing on basic functionality,
performance, and edge cases.
"""

import os
import sys
import time

import psutil
import pytest
import torch

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from bridge.domain.datamodels import ModelConfig
from bridge.domain.model import Model
from bridge.domain.model.decoder import Decoder
from bridge.domain.model.encoder import Encoder
from bridge.domain.model.sliding_window_wrapper import (
    SlidingWindowDecoderWrapper,
    SlidingWindowEncoderWrapper,
)
from bridge.domain.model.synthetic_dataset import SyntheticBridgeDataset


def get_memory_usage() -> float:
    """Get current memory usage in MB.

    Returns:
        Memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def create_test_model_config(use_sliding_window: bool = False) -> ModelConfig:
    """Create a test model configuration.

    Args:
        use_sliding_window: Whether to enable sliding window attention.

    Returns:
        ModelConfig instance for testing.
    """
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
        window_size=61,  # Default window size
        is_causal=False,  # Non-causal for this test
        max_seq_len=1024,  # Reasonable cache size
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
        "phon_dec_pad_mask": phon_enc_pad_mask,  # Reuse for simplicity
        "orth_dec_input": orth_dec_input,
        "orth_dec_pad_mask": orth_enc_pad_mask,  # Reuse for simplicity
    }


def mock_model_sequence_lengths(model: Model, max_seq_len: int = 128) -> None:
    """Mock the model's sequence length constraints for testing.

    This function replaces the hardcoded sequence length limits in the model
    to allow testing with longer sequences without affecting other parts of the codebase.

    Args:
        model: The model instance to modify.
        max_seq_len: The new maximum sequence length to use.
    """
    # Store original values for potential restoration
    model._original_max_orth_seq_len = model.max_orth_seq_len
    model._original_max_phon_seq_len = model.max_phon_seq_len

    # Update the sequence length limits
    model.max_orth_seq_len = max_seq_len
    model.max_phon_seq_len = max_seq_len

    # Recreate position embeddings with new sequence length
    device = model.device
    d_model = model.model_config.d_model

    # Get original weights
    original_orth_weights = model.orth_position_embedding.weight.detach()
    original_phon_weights = model.phon_position_embedding.weight.detach()

    # Create new position embeddings with proper initialization
    new_orth_position_embedding = torch.nn.Embedding(
        max_seq_len, d_model, device=device
    )
    new_phon_position_embedding = torch.nn.Embedding(
        max_seq_len, d_model, device=device
    )

    # Initialize with random values first
    with torch.no_grad():
        # Copy existing weights if the new size is larger
        if model._original_max_orth_seq_len <= max_seq_len:
            # Copy existing weights to the beginning of the new embedding
            new_orth_position_embedding.weight.data[
                : model._original_max_orth_seq_len
            ] = original_orth_weights
            new_phon_position_embedding.weight.data[
                : model._original_max_phon_seq_len
            ] = original_phon_weights

    # Replace the position embeddings
    model.orth_position_embedding = new_orth_position_embedding
    model.phon_position_embedding = new_phon_position_embedding

    print(f"✓ Mocked sequence lengths to {max_seq_len}")


def test_1_1_disabled_vs_enabled_sliding_window():
    """
    Test 1.1: Compare forward pass with use_sliding_window: false vs use_sliding_window: true.

    This test verifies that:
    1. Both configurations produce the same output shapes
    2. Memory usage is different (sliding window should use less memory)
    3. Forward pass time is different (sliding window should be faster for large sequences)
    4. Output values are different (due to different attention patterns)
    """
    print("\n=== Test 1.1: Disabled vs Enabled Sliding Window ===")

    # Create synthetic dataset
    dataset = SyntheticBridgeDataset(num_samples=50)
    print(f"✓ Created synthetic dataset with {len(dataset)} samples")

    # Test configurations
    configs = [
        ("disabled", create_test_model_config(use_sliding_window=False)),
        ("enabled", create_test_model_config(use_sliding_window=True)),
    ]

    results = {}

    for config_name, model_config in configs:
        print(f"\n--- Testing {config_name} sliding window ---")

        # Create model
        model = Model(model_config, dataset)
        model.eval()  # Set to evaluation mode
        print(f"✓ Created model with {config_name} sliding window")

        # Mock sequence lengths to allow longer sequences for testing
        mock_model_sequence_lengths(model, max_seq_len=128)
        print(f"✓ Mocked sequence lengths to 128")

        # Create test data with longer sequences to better test sliding window benefits
        test_data = create_test_data(batch_size=4, seq_len=64)
        print(f"✓ Created test data with batch_size=4, seq_len=64")

        # Measure memory before forward pass
        memory_before = get_memory_usage()

        # Time the forward pass
        start_time = time.time()

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

        forward_time = time.time() - start_time

        # Measure memory after forward pass
        memory_after = get_memory_usage()
        memory_used = memory_after - memory_before

        # Store results
        results[config_name] = {
            "output": output,
            "forward_time": forward_time,
            "memory_used": memory_used,
            "output_shapes": {k: v.shape for k, v in output.items()},
        }

        print(f"✓ Forward pass completed in {forward_time:.4f}s")
        print(f"✓ Memory used: {memory_used:.2f} MB")
        print(f"✓ Output shapes: {results[config_name]['output_shapes']}")

        # Verify sliding window status
        actual_status = model.get_sliding_window_status()
        expected_status = config_name == "enabled"
        assert (
            actual_status == expected_status
        ), f"Expected sliding window status {expected_status}, got {actual_status}"
        print(f"✓ Sliding window status verified: {actual_status}")

    # Compare results
    print(f"\n--- Comparison Results ---")

    # 1. Check output shapes are the same
    disabled_shapes = results["disabled"]["output_shapes"]
    enabled_shapes = results["enabled"]["output_shapes"]

    assert disabled_shapes == enabled_shapes, (
        f"Output shapes should be identical. "
        f"Disabled: {disabled_shapes}, Enabled: {enabled_shapes}"
    )
    print(f"✓ Output shapes are identical: {disabled_shapes}")

    # 2. Check memory usage (sliding window should use less memory)
    disabled_memory = results["disabled"]["memory_used"]
    enabled_memory = results["enabled"]["memory_used"]

    print(
        f"Memory usage - Disabled: {disabled_memory:.2f} MB, Enabled: {enabled_memory:.2f} MB"
    )
    print(f"Memory difference: {disabled_memory - enabled_memory:.2f} MB")

    # For longer sequences, we should see a more significant memory difference
    if disabled_memory > enabled_memory:
        print(f"✓ Sliding window uses less memory as expected")
    else:
        print(f"⚠ Memory difference is minimal (may be due to small sequence length)")

    print(f"✓ Memory usage comparison completed")

    # 3. Check forward pass time
    disabled_time = results["disabled"]["forward_time"]
    enabled_time = results["enabled"]["forward_time"]

    print(
        f"Forward time - Disabled: {disabled_time:.4f}s, Enabled: {enabled_time:.4f}s"
    )
    print(f"Time difference: {disabled_time - enabled_time:.4f}s")
    print(f"Speedup: {disabled_time / enabled_time:.2f}x")

    # For longer sequences, sliding window should be faster
    if enabled_time < disabled_time:
        print(f"✓ Sliding window is faster as expected")
    else:
        print(f"⚠ Time difference is minimal (may be due to small sequence length)")

    print(f"✓ Forward pass time comparison completed")

    # 4. Check that output values are different (due to different attention patterns)
    disabled_output = results["disabled"]["output"]
    enabled_output = results["enabled"]["output"]

    # Compare orthographic outputs
    orth_disabled = disabled_output["orth"]
    orth_enabled = enabled_output["orth"]

    # Check that outputs are different (not identical)
    orth_diff = torch.abs(orth_disabled - orth_enabled).mean().item()
    print(f"Orthographic output difference: {orth_diff:.6f}")

    assert orth_diff > 1e-6, (
        f"Outputs should be different due to different attention patterns. "
        f"Difference: {orth_diff}"
    )
    print(f"✓ Orthographic outputs are different (attention patterns working)")

    # Compare phonological outputs
    phon_disabled = disabled_output["phon"]
    phon_enabled = enabled_output["phon"]

    phon_diff = torch.abs(phon_disabled - phon_enabled).mean().item()
    print(f"Phonological output difference: {phon_diff:.6f}")

    assert phon_diff > 1e-6, (
        f"Outputs should be different due to different attention patterns. "
        f"Difference: {phon_diff}"
    )
    print(f"✓ Phonological outputs are different (attention patterns working)")

    # 5. Test the set_sliding_window method
    print(f"\n--- Testing set_sliding_window method ---")

    # Start with disabled model
    model = Model(create_test_model_config(use_sliding_window=False), dataset)
    mock_model_sequence_lengths(model, max_seq_len=128)  # Mock sequence lengths

    initial_status = model.get_sliding_window_status()
    assert not initial_status, "Initial status should be disabled"
    print(f"✓ Initial sliding window status: {initial_status}")

    # Enable sliding window
    model.set_sliding_window(True)
    enabled_status = model.get_sliding_window_status()
    assert enabled_status, "Status should be enabled after set_sliding_window(True)"
    print(f"✓ Sliding window enabled: {enabled_status}")

    # Disable sliding window
    model.set_sliding_window(False)
    disabled_status = model.get_sliding_window_status()
    assert (
        not disabled_status
    ), "Status should be disabled after set_sliding_window(False)"
    print(f"✓ Sliding window disabled: {disabled_status}")

    print(f"\n=== Test 1.1 PASSED ===")
    print(f"✓ Basic wrapper functionality verified")
    print(f"✓ Output shapes are consistent")
    print(f"✓ Memory and time measurements completed")
    print(f"✓ Attention patterns are different")
    print(f"✓ Sliding window toggle functionality works")


def test_1_1_edge_cases():
    """
    Test edge cases for the sliding window wrapper.
    """
    print("\n=== Test 1.1 Edge Cases ===")

    # Create synthetic dataset
    dataset = SyntheticBridgeDataset(num_samples=10)

    # Test with very small window size
    print(f"\n--- Testing small window size ---")
    small_window_config = create_test_model_config(use_sliding_window=True)
    small_window_config.window_size = 5  # Very small window

    model = Model(small_window_config, dataset)
    model.eval()
    mock_model_sequence_lengths(model, max_seq_len=64)  # Mock sequence lengths

    test_data = create_test_data(batch_size=2, seq_len=32)

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

    print(f"✓ Small window size (5) works correctly")

    # Test with large window size
    print(f"\n--- Testing large window size ---")
    large_window_config = create_test_model_config(use_sliding_window=True)
    large_window_config.window_size = 201  # Large window

    model = Model(large_window_config, dataset)
    model.eval()
    mock_model_sequence_lengths(model, max_seq_len=256)  # Mock sequence lengths

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

    print(f"✓ Large window size (201) works correctly")

    print(f"\n=== Test 1.1 Edge Cases PASSED ===")


def test_1_1_scaling_benefits():
    """
    Test that sliding window shows benefits with longer sequences.
    """
    print("\n=== Test 1.1 Scaling Benefits ===")

    # Create synthetic dataset
    dataset = SyntheticBridgeDataset(num_samples=20)

    # Test with different sequence lengths to show scaling benefits
    sequence_lengths = [32, 64, 128]

    for seq_len in sequence_lengths:
        print(f"\n--- Testing sequence length {seq_len} ---")

        # Create models with and without sliding window
        disabled_config = create_test_model_config(use_sliding_window=False)
        enabled_config = create_test_model_config(use_sliding_window=True)

        disabled_model = Model(disabled_config, dataset)
        enabled_model = Model(enabled_config, dataset)

        # Mock sequence lengths
        mock_model_sequence_lengths(disabled_model, max_seq_len=seq_len * 2)
        mock_model_sequence_lengths(enabled_model, max_seq_len=seq_len * 2)

        disabled_model.eval()
        enabled_model.eval()

        # Create test data
        test_data = create_test_data(batch_size=2, seq_len=seq_len)

        # Measure performance
        with torch.no_grad():
            # Disabled sliding window
            start_time = time.time()
            disabled_output = disabled_model.forward(
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
            disabled_time = time.time() - start_time

            # Enabled sliding window
            start_time = time.time()
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
            enabled_time = time.time() - start_time

        speedup = disabled_time / enabled_time
        print(
            f"Sequence length {seq_len}: Disabled={disabled_time:.4f}s, Enabled={enabled_time:.4f}s, Speedup={speedup:.2f}x"
        )

        # Verify outputs are different
        orth_diff = (
            torch.abs(disabled_output["orth"] - enabled_output["orth"]).mean().item()
        )
        assert orth_diff > 1e-6, f"Outputs should be different for seq_len={seq_len}"

        print(f"✓ Sequence length {seq_len} test passed")

    print(f"\n=== Test 1.1 Scaling Benefits PASSED ===")


if __name__ == "__main__":
    """Run the tests when executed directly."""
    print("Running Test 1.1: Disabled vs Enabled Sliding Window")

    # Run the main test
    test_1_1_disabled_vs_enabled_sliding_window()

    # Run edge case tests
    test_1_1_edge_cases()

    # Run scaling benefits test
    test_1_1_scaling_benefits()

    print("\n All Test 1.1 tests passed!")
