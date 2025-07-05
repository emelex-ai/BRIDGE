#!/usr/bin/env python3
"""Test exact sliding window attention comparison.

This test compares three different approaches at the same level of abstraction:
1. Raw attention with sliding window mask (true sliding window)
2. LocalAttention module (chunked approach)
3. Manual chunked attention (to understand LocalAttention's strategy)

The goal is to understand why LocalAttention gives different results and whether
we need to implement our own true sliding window attention.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_sliding_window_mask(
    seq_len: int, window_size: int, causal: bool = True, device: str = "cpu"
) -> torch.Tensor:
    """Create a sliding window attention mask.

    Args:
        seq_len: Sequence length
        window_size: Window size for sliding attention
        causal: Whether to apply causal masking
        device: Device to create mask on

    Returns:
        Boolean mask where True means attend, False means mask
    """
    # Create position indices
    i = torch.arange(seq_len, device=device)[:, None]  # (seq_len, 1)
    j = torch.arange(seq_len, device=device)[None, :]  # (1, seq_len)

    # Create sliding window mask
    # For each position i, we can attend to positions [i-window_size+1, i]
    mask = (j >= i - window_size + 1) & (j <= i)

    # Apply causal constraint if needed
    if causal:
        mask = mask & (j <= i)

    return mask


def raw_attention_with_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Raw attention computation with mask.

    This is the baseline "true" sliding window attention.
    """
    if scale is None:
        scale = q.size(-1) ** -0.5

    # Compute attention scores: Q @ K^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply mask
    mask_value = -torch.finfo(scores.dtype).max
    scores = scores.masked_fill(~mask, mask_value)

    # Softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Apply to values
    out = torch.matmul(attn_weights, v)

    return out


def chunked_attention_manual(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    causal: bool = True,
) -> torch.Tensor:
    """Manual implementation of chunked attention to understand LocalAttention.

    This implements the chunked strategy shown in the rightmost panel of your image.
    """
    batch_size, seq_len, dim = q.shape

    # For simplicity, assume seq_len is divisible by window_size
    # In practice, LocalAttention handles padding
    chunk_size = window_size
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    outputs = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, seq_len)

        # Extract current chunk
        q_chunk = q[:, start_idx:end_idx]  # (batch, chunk_len, dim)

        # For causal attention, we can only attend to current and previous chunks
        if causal:
            k_context = k[:, :end_idx]  # (batch, context_len, dim)
            v_context = v[:, :end_idx]  # (batch, context_len, dim)
        else:
            # For non-causal, we could attend to more context
            k_context = k
            v_context = v

        # Compute attention for this chunk
        chunk_out = raw_attention_with_mask(
            q_chunk,
            k_context,
            v_context,
            mask=torch.ones(
                q_chunk.size(1), k_context.size(1), dtype=torch.bool, device=q.device
            ),
        )

        outputs.append(chunk_out)

    # Concatenate all chunks
    return torch.cat(outputs, dim=1)


def test_attention_comparison():
    """Compare different attention implementations."""
    print("Testing Attention Implementation Comparison")
    print("=" * 60)

    # Test parameters
    batch_size = 1
    seq_len = 16  # Small for debugging
    dim = 8
    window_size = 4
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    # Create test data
    q = torch.randn(batch_size, seq_len, dim, device=device)
    k = torch.randn(batch_size, seq_len, dim, device=device)
    v = torch.randn(batch_size, seq_len, dim, device=device)

    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"Window size: {window_size}")

    # 1. True sliding window attention (baseline)
    print("\n1. True Sliding Window Attention (Raw)")
    print("-" * 40)

    mask = create_sliding_window_mask(
        seq_len, window_size, causal=True, device=device.type
    )
    print(f"Mask shape: {mask.shape}")
    print(f"Mask sparsity: {(~mask).sum().item() / mask.numel() * 100:.1f}% masked")

    # Show mask pattern for debugging
    print("Mask pattern (1=attend, 0=mask):")
    print(mask.int().cpu().numpy())

    true_out = raw_attention_with_mask(q, k, v, mask)
    print(f"Output shape: {true_out.shape}")
    print(f"Output mean: {true_out.mean().item():.6f}")
    print(f"Output std: {true_out.std().item():.6f}")

    # 2. LocalAttention (chunked approach)
    print("\n2. LocalAttention (Chunked)")
    print("-" * 40)

    try:
        from local_attention import LocalAttention

        # Reset seed
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        local_attn = LocalAttention(
            window_size=window_size,
            causal=True,
            look_backward=0,  # No overlap
            look_forward=0,  # No overlap
            dropout=0.0,
            autopad=True,
            exact_windowsize=True,
            use_rotary_pos_emb=False,
            use_xpos=False,
        ).to(device)

        # LocalAttention expects (batch*heads, seq_len, dim_head)
        # We'll treat our input as single-head
        q_flat = q.reshape(batch_size, seq_len, dim)
        k_flat = k.reshape(batch_size, seq_len, dim)
        v_flat = v.reshape(batch_size, seq_len, dim)

        local_out = local_attn(q_flat, k_flat, v_flat)

        print(f"Output shape: {local_out.shape}")
        print(f"Output mean: {local_out.mean().item():.6f}")
        print(f"Output std: {local_out.std().item():.6f}")

        # Compare with true sliding window
        diff = torch.abs(true_out - local_out)
        print(f"Difference from true sliding window:")
        print(f"  Max diff: {diff.max().item():.6f}")
        print(f"  Mean diff: {diff.mean().item():.6f}")
        print(
            f"  Relative error: {diff.mean().item() / true_out.abs().mean().item():.6f}"
        )

        # Check if close
        close = torch.allclose(true_out, local_out, atol=1e-4, rtol=1e-3)
        print(f"  Close (1e-4, 1e-3): {close}")

    except ImportError:
        print("LocalAttention not available")
        local_out = None
    except Exception as e:
        print(f"Error with LocalAttention: {e}")
        local_out = None

    # 3. Manual chunked attention (to understand chunking)
    print("\n3. Manual Chunked Attention")
    print("-" * 40)

    try:
        chunked_out = chunked_attention_manual(q, k, v, window_size, causal=True)

        print(f"Output shape: {chunked_out.shape}")
        print(f"Output mean: {chunked_out.mean().item():.6f}")
        print(f"Output std: {chunked_out.std().item():.6f}")

        # Compare with true sliding window
        diff = torch.abs(true_out - chunked_out)
        print(f"Difference from true sliding window:")
        print(f"  Max diff: {diff.max().item():.6f}")
        print(f"  Mean diff: {diff.mean().item():.6f}")
        print(
            f"  Relative error: {diff.mean().item() / true_out.abs().mean().item():.6f}"
        )

        # Check if close
        close = torch.allclose(true_out, chunked_out, atol=1e-4, rtol=1e-3)
        print(f"  Close (1e-4, 1e-3): {close}")

    except Exception as e:
        print(f"Error with manual chunked attention: {e}")
        chunked_out = None

    # 4. Analysis and conclusions
    print("\n4. Analysis")
    print("-" * 40)

    print("Key insights:")
    print(
        "• True sliding window: Each position attends to exactly window_size previous positions"
    )
    print(
        "• LocalAttention (chunked): Processes sequence in chunks, different computation order"
    )
    print("• Manual chunked: Simplified version of chunked processing")
    print()

    if local_out is not None:
        if torch.allclose(true_out, local_out, atol=1e-2, rtol=1e-2):
            print("✓ LocalAttention produces similar results to true sliding window")
            print("  → Can use LocalAttention for performance benefits")
        else:
            print(
                "✗ LocalAttention produces different results from true sliding window"
            )
            print("  → Need to implement true sliding window attention")

    print()
    print("Recommendation:")
    if local_out is not None and torch.allclose(
        true_out, local_out, atol=1e-2, rtol=1e-2
    ):
        print("• Use LocalAttention for memory/speed benefits")
        print("• Accept small numerical differences due to chunking")
    else:
        print("• Implement true sliding window attention")
        print("• Use raw attention with sliding window mask")

    return {
        "true_sliding_window": true_out,
        "local_attention": local_out,
        "chunked_attention": chunked_out,
    }


if __name__ == "__main__":
    results = test_attention_comparison()
