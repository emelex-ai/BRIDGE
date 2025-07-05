"""
Test script to compare full attention with sliding window mask vs local attention.

This script verifies that the local attention module works correctly by comparing:
1. Full attention with a sliding window mask applied
2. Local attention with the same window size

Both implementations should produce similar results when using the same random inputs.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import local attention modules
try:
    from local_attention import LocalAttention
    from true_sliding_window_attention import (
        TrueSlidingWindowAttention,
        TrueSlidingWindowMHA,
    )

    LOCAL_ATTENTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Local attention modules not available: {e}")
    LOCAL_ATTENTION_AVAILABLE = False


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
    scale: Optional[float] = None,
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

    return out, attn_weights


class FullAttentionMHA(nn.Module):
    """Multi-head attention with full attention and sliding window mask."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        causal: bool = True,
        prenorm: bool = False,
    ):
        """Initialize full attention MHA.

        Args:
            dim: Model dimension
            window_size: Size of sliding window for masking
            dim_head: Dimension per head
            heads: Number of attention heads
            dropout: Dropout probability
            causal: Whether to use causal masking
            prenorm: Whether to apply prenorm
        """
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.dim_head = dim_head
        self.heads = heads
        self.causal = causal
        self.scale = dim_head**-0.5

        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if prenorm else None
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dim)

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape

        if self.norm is not None:
            x = self.norm(x)

        # Project to Q, K, V
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)

        # Create sliding window mask
        mask = create_sliding_window_mask(seq_len, self.window_size, self.causal)
        mask = mask.to(x.device)

        # Apply attention with mask
        out, attn_weights = full_attention_with_mask(q, k, v, mask, self.scale)

        # Reshape output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Apply output projection
        out = self.to_out(out)
        out = self.dropout(out)

        return out, attn_weights


def compare_attention_implementations(
    batch_size: int = 2,
    seq_len: int = 512,
    dim: int = 256,
    heads: int = 8,
    window_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> dict:
    """Compare full attention with mask vs local attention.

    Args:
        batch_size: Batch size for testing
        seq_len: Sequence length
        dim: Model dimension
        heads: Number of attention heads
        window_size: Size of sliding window
        device: Device to run on
        seed: Random seed for reproducibility

    Returns:
        Dictionary with comparison results
    """
    print(f"Comparing attention implementations on {device}")
    print(
        f"Config: batch_size={batch_size}, seq_len={seq_len}, dim={dim}, heads={heads}, window_size={window_size}"
    )

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create test input
    x = torch.randn(batch_size, seq_len, dim, device=device)

    results = {}

    # Test 1: Full attention with sliding window mask
    print("\n1. Testing full attention with sliding window mask...")
    full_attn = FullAttentionMHA(
        dim=dim, window_size=window_size, heads=heads, causal=True, prenorm=True
    ).to(device)

    torch.manual_seed(seed)  # Reset seed
    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

        if device == "cuda":
            start_time.record()

        full_out, full_attn_weights = full_attn(x)

        if device == "cuda":
            end_time.record()
            torch.cuda.synchronize()
            full_time = start_time.elapsed_time(end_time)
        else:
            full_time = 0

    results["full_attention"] = {
        "output": full_out,
        "attention_weights": full_attn_weights,
        "time_ms": full_time,
        "memory_mb": torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0,
    }

    print(f"Full attention - Output shape: {full_out.shape}, Time: {full_time:.2f}ms")

    # Test 2: Local attention (if available)
    if LOCAL_ATTENTION_AVAILABLE:
        print("\n2. Testing local attention...")
        torch.cuda.empty_cache() if device == "cuda" else None

        local_attn = TrueSlidingWindowMHA(
            dim=dim, window_size=window_size, heads=heads, causal=True, prenorm=True
        ).to(device)

        torch.manual_seed(seed)  # Reset seed
        with torch.no_grad():
            start_time = (
                torch.cuda.Event(enable_timing=True) if device == "cuda" else None
            )
            end_time = (
                torch.cuda.Event(enable_timing=True) if device == "cuda" else None
            )

            if device == "cuda":
                start_time.record()

            local_out = local_attn(x)

            if device == "cuda":
                end_time.record()
                torch.cuda.synchronize()
                local_time = start_time.elapsed_time(end_time)
            else:
                local_time = 0

        results["local_attention"] = {
            "output": local_out,
            "time_ms": local_time,
            "memory_mb": torch.cuda.memory_allocated() / 1024**2
            if device == "cuda"
            else 0,
        }

        print(
            f"Local attention - Output shape: {local_out.shape}, Time: {local_time:.2f}ms"
        )

        # Compare outputs
        output_diff = torch.abs(full_out - local_out)
        max_diff = output_diff.max().item()
        mean_diff = output_diff.mean().item()

        results["comparison"] = {
            "max_difference": max_diff,
            "mean_difference": mean_diff,
            "relative_error": mean_diff / full_out.abs().mean().item(),
            "outputs_close": torch.allclose(full_out, local_out, atol=1e-4, rtol=1e-3),
        }

        print(f"\nComparison Results:")
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Relative error: {results['comparison']['relative_error']:.6f}")
        print(f"Outputs close: {results['comparison']['outputs_close']}")

        # Speed comparison
        if full_time > 0 and local_time > 0:
            speedup = full_time / local_time
            print(f"Speedup: {speedup:.2f}x")
            results["comparison"]["speedup"] = speedup

    else:
        print("\n2. Local attention not available - skipping comparison")

    return results


def visualize_attention_patterns(
    attention_weights: torch.Tensor,
    window_size: int,
    title: str = "Attention Pattern",
    save_path: Optional[str] = None,
):
    """Visualize attention patterns.

    Args:
        attention_weights: Attention weights (batch, heads, seq_len, seq_len)
        window_size: Window size for reference
        title: Plot title
        save_path: Optional path to save the plot
    """
    # Take first batch, first head for visualization
    attn = attention_weights[0, 0].cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn, cmap="Blues", cbar=True)
    plt.title(f"{title} (Window Size: {window_size})")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def run_comprehensive_test():
    """Run comprehensive test comparing attention implementations."""
    print("=" * 60)
    print("SLIDING WINDOW ATTENTION COMPARISON TEST")
    print("=" * 60)

    # Test configurations
    configs = [
        {"seq_len": 256, "window_size": 32, "dim": 128, "heads": 4},
        {"seq_len": 512, "window_size": 64, "dim": 256, "heads": 8},
        {"seq_len": 1024, "window_size": 128, "dim": 512, "heads": 8},
    ]

    all_results = []

    for i, config in enumerate(configs):
        print(f"\n{'='*40}")
        print(f"Test Configuration {i+1}")
        print(f"{'='*40}")

        results = compare_attention_implementations(**config)
        all_results.append(results)

        # Visualize attention pattern for first config
        if i == 0 and "full_attention" in results:
            visualize_attention_patterns(
                results["full_attention"]["attention_weights"],
                config["window_size"],
                f"Full Attention with Sliding Window Mask (Config {i+1})",
            )

    return all_results


if __name__ == "__main__":
    # Run the comprehensive test
    results = run_comprehensive_test()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for i, result in enumerate(results):
        print(f"\nConfiguration {i+1}:")
        if "comparison" in result:
            comp = result["comparison"]
            print(f"  Outputs match: {comp['outputs_close']}")
            print(f"  Max difference: {comp['max_difference']:.6f}")
            print(f"  Relative error: {comp['relative_error']:.6f}")
            if "speedup" in comp:
                print(f"  Local attention speedup: {comp['speedup']:.2f}x")
        else:
            print("  Local attention not available for comparison")
