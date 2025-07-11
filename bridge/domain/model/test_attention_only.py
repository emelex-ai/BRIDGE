#!/usr/bin/env python3
"""
Test attention modules in isolation to debug memory usage.
"""

import torch
import torch.nn as nn

from bridge.domain.model.benchmarks.sdpa_full_attention_model import SDPAFullAttention
from bridge.domain.model.benchmarks.sdpa_sliding_window_model import (
    SDPASlidingWindowAttention,
)


def test_attention_only():
    """Test attention modules in isolation."""
    print("=" * 60)
    print("Testing Attention Modules in Isolation")
    print("=" * 60)

    # Test parameters
    seq_len = 4096
    d_model = 512
    nhead = 1
    window_size = 32
    batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Test configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {d_model}")
    print(f"  Number of heads: {nhead}")
    print(f"  Window size: {window_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print()

    # Create input tensor
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Test 1: SDPA Full Attention
    print("Test 1: SDPA Full Attention")
    try:
        # Clear memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create model
        full_attention = SDPAFullAttention(
            d_model, nhead, seq_len=seq_len, device=device
        )
        full_attention = full_attention.to(device)

        print(f"  Model created on {device}")
        print(f"  Initial memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

        # Forward pass
        with torch.no_grad():
            output = full_attention(x)

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak memory: {peak_memory:.1f}MB")
        print(f"  Output shape: {output.shape}")
        print()

    except Exception as e:
        print(f"  ❌ Failed: {str(e)}")
        import traceback

        traceback.print_exc()
        print()

    # Test 2: SDPA Sliding Window Attention
    print("Test 2: SDPA Sliding Window Attention")
    try:
        # Clear memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create model
        sliding_attention = SDPASlidingWindowAttention(
            d_model, nhead, window_size, seq_len=seq_len, device=device
        )
        sliding_attention = sliding_attention.to(device)

        print(f"  Model created on {device}")
        print(f"  Initial memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

        # Forward pass
        with torch.no_grad():
            output = sliding_attention(x)

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak memory: {peak_memory:.1f}MB")
        print(f"  Output shape: {output.shape}")
        print()

    except Exception as e:
        print(f"  ❌ Failed: {str(e)}")
        import traceback

        traceback.print_exc()
        print()

    # Test 3: Compare with benchmark results
    print("Test 3: Comparison with Benchmark Results")
    print("  Expected from compare_SDPA_classical_attention_only.py:")
    print("    - SDPA Full: ~96MB")
    print("    - SDPA Sliding: ~120MB")
    print("  If memory is constant, there's an issue with mask creation.")
    print()


if __name__ == "__main__":
    test_attention_only()
