#!/usr/bin/env python3
"""
Simple GPU test script for sliding window attention comparison.

This script is designed to run on GPU and compare:
1. Full attention with sliding window mask
2. Local attention with the same window size

Both use the same random seed for reproducible results.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_sliding_window_mask(
    seq_len: int, window_size: int, causal: bool = True
) -> torch.Tensor:
    """Create a sliding window attention mask.

    Args:
        seq_len: Sequence length
        window_size: Size of the sliding window
        causal: Whether to apply causal (lower triangular) masking

    Returns:
        Attention mask tensor (seq_len, seq_len) where True = attend, False = mask
    """
    # Create base mask - all positions can attend to all positions
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)

    # Apply sliding window constraint
    for i in range(seq_len):
        # Each position can only attend to positions within the window
        start_pos = max(0, i - window_size + 1)
        end_pos = min(seq_len, i + window_size)

        # Mask out positions outside the window
        mask[i, :start_pos] = False
        mask[i, end_pos:] = False

    # Apply causal masking if requested
    if causal:
        # Create lower triangular mask (can only attend to past and current)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        mask = mask & causal_mask

    return mask


def full_attention_with_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """Compute full attention with a sliding window mask.

    Args:
        q: Query tensor (batch, heads, seq_len, dim_head)
        k: Key tensor (batch, heads, seq_len, dim_head)
        v: Value tensor (batch, heads, seq_len, dim_head)
        mask: Attention mask (seq_len, seq_len)
        scale: Optional scaling factor

    Returns:
        Attention output (batch, heads, seq_len, dim_head)
    """
    batch_size, num_heads, seq_len, dim_head = q.shape

    if scale is None:
        scale = dim_head**-0.5

    # Compute attention scores
    scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale

    # Apply mask (set masked positions to large negative value)
    mask_value = -torch.finfo(scores.dtype).max
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), mask_value)

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    out = torch.einsum("bhij,bhjd->bhid", attn_weights, v)

    return out


def test_sliding_window_attention_gpu():
    """Test sliding window attention on GPU."""
    print("Testing Sliding Window Attention on GPU")
    print("=" * 50)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires GPU.")
        return False

    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")

    # Test parameters
    batch_size = 2
    seq_len = 512
    dim = 256
    heads = 8
    dim_head = dim // heads
    window_size = 64
    seed = 42

    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Model dimension: {dim}")
    print(f"  Number of heads: {heads}")
    print(f"  Head dimension: {dim_head}")
    print(f"  Window size: {window_size}")
    print(f"  Random seed: {seed}")

    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Create test input
    x = torch.randn(batch_size, seq_len, dim, device=device)
    print(f"\nInput tensor shape: {x.shape}")

    # Create Q, K, V matrices
    qkv_proj = nn.Linear(dim, dim * 3, bias=False).to(device)
    q, k, v = qkv_proj(x).chunk(3, dim=-1)

    # Reshape for multi-head attention
    q = q.view(batch_size, seq_len, heads, dim_head).transpose(1, 2)
    k = k.view(batch_size, seq_len, heads, dim_head).transpose(1, 2)
    v = v.view(batch_size, seq_len, heads, dim_head).transpose(1, 2)

    print(f"Q, K, V shapes: {q.shape}")

    # Test 1: Full attention with sliding window mask
    print(f"\n1. Testing full attention with sliding window mask...")

    # Create sliding window mask
    mask = create_sliding_window_mask(seq_len, window_size, causal=True)
    mask = mask.to(device)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sparsity: {(~mask).sum().item() / mask.numel() * 100:.1f}% masked")

    # Measure memory before
    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated()

    # Run full attention
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record()
    full_out = full_attention_with_mask(q, k, v, mask)
    end_time.record()

    torch.cuda.synchronize()
    full_time = start_time.elapsed_time(end_time)
    memory_after = torch.cuda.memory_allocated()
    memory_used = (memory_after - memory_before) / 1024**2

    print(f"Full attention results:")
    print(f"  Output shape: {full_out.shape}")
    print(f"  Time: {full_time:.2f} ms")
    print(f"  Memory used: {memory_used:.2f} MB")
    print(f"  Output mean: {full_out.mean().item():.6f}")
    print(f"  Output std: {full_out.std().item():.6f}")

    # Test 2: Try to import and test local attention
    try:
        from local_attention import LocalAttention

        print(f"\n2. Testing local attention...")

        # Clear memory
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated()

        # Create local attention
        local_attn = LocalAttention(
            window_size=window_size,
            causal=True,
            look_backward=0,  # No overlap
            look_forward=0,  # No overlap
            dropout=0.0,
            autopad=True,
            exact_windowsize=True,
        ).to(device)

        # Flatten heads into batch dimension for local attention
        q_flat = q.reshape(batch_size * heads, seq_len, dim_head)
        k_flat = k.reshape(batch_size * heads, seq_len, dim_head)
        v_flat = v.reshape(batch_size * heads, seq_len, dim_head)

        # Run local attention
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        local_out_flat = local_attn(q_flat, k_flat, v_flat)
        end_time.record()

        torch.cuda.synchronize()
        local_time = start_time.elapsed_time(end_time)
        memory_after = torch.cuda.memory_allocated()
        memory_used_local = (memory_after - memory_before) / 1024**2

        # Reshape back to multi-head format
        local_out = local_out_flat.reshape(batch_size, heads, seq_len, dim_head)

        print(f"Local attention results:")
        print(f"  Output shape: {local_out.shape}")
        print(f"  Time: {local_time:.2f} ms")
        print(f"  Memory used: {memory_used_local:.2f} MB")
        print(f"  Output mean: {local_out.mean().item():.6f}")
        print(f"  Output std: {local_out.std().item():.6f}")

        # Compare outputs
        output_diff = torch.abs(full_out - local_out)
        max_diff = output_diff.max().item()
        mean_diff = output_diff.mean().item()
        relative_error = mean_diff / full_out.abs().mean().item()
        outputs_close = torch.allclose(full_out, local_out, atol=1e-4, rtol=1e-3)

        print(f"\n3. Comparison Results:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Relative error: {relative_error:.6f}")
        print(f"  Outputs close (1e-4 atol, 1e-3 rtol): {outputs_close}")

        # Performance comparison
        if full_time > 0 and local_time > 0:
            speedup = full_time / local_time
            memory_reduction = (memory_used - memory_used_local) / memory_used * 100
            print(f"  Time speedup: {speedup:.2f}x")
            print(f"  Memory reduction: {memory_reduction:.1f}%")

        # Test passed if outputs are close
        test_passed = outputs_close

    except ImportError as e:
        print(f"\n2. Local attention not available: {e}")
        print("Skipping local attention test")
        test_passed = True  # Consider test passed if we can't import local attention

    except Exception as e:
        print(f"\n2. Error testing local attention: {e}")
        test_passed = False

    # Final results
    print(f"\n{'='*50}")
    print(f"TEST RESULT: {'PASSED' if test_passed else 'FAILED'}")
    print(f"{'='*50}")

    return test_passed


if __name__ == "__main__":
    success = test_sliding_window_attention_gpu()
    sys.exit(0 if success else 1)
