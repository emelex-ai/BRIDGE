import gc
import time

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def benchmark_full_attention(seq_len, d_model, nhead, num_layers=1, batch_size=4):
    """Benchmark standard TransformerEncoder with fixed batch size."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
    )
    model = TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

    # Use fixed batch size
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(x)

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        with torch.no_grad():
            _ = model(x)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "batch_size": batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
        "attention_type": "full",
    }


def benchmark_flex_attention(
    seq_len, d_model, nhead, window_size, num_layers=1, batch_size=4
):
    """Benchmark FlexAttention implementation with fixed batch size."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Import FlexAttention implementation
        from bridge.domain.model.transformer_flex_attention import (
            FlexAttentionEncoderLayer,
        )

        # Create model
        encoder_layer = FlexAttentionEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            window_size=window_size,
            causal=False,  # Non-causal for fair comparison with full attention
        )

        # Create encoder with multiple layers if needed
        class FlexAttentionEncoder(nn.Module):
            def __init__(self, layer, num_layers):
                super().__init__()
                self.layers = nn.ModuleList([layer for _ in range(num_layers)])

            def forward(self, x, mask=None):
                for layer in self.layers:
                    x = layer(x, src_mask=mask)
                return x

        model = FlexAttentionEncoder(encoder_layer, num_layers).to(device)

        # Use fixed batch size
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)

        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(10):
            with torch.no_grad():
                _ = model(x)

        torch.cuda.synchronize()
        end_time = time.time()

        avg_time_ms = (end_time - start_time) / 10 * 1000
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        return {
            "batch_size": batch_size,
            "time_ms": avg_time_ms,
            "memory_mb": memory_mb,
            "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
            "window_size": window_size,
            "attention_type": "flex",
        }

    except ImportError as e:
        print(f"FlexAttention not available: {e}")
        return None
    except Exception as e:
        print(f"FlexAttention benchmark failed: {e}")
        return None


def compare_attention_implementations():
    """Compare FlexAttention vs Full Attention with fixed batch size."""
    # Test configurations
    configs = [
        {"seq_len": 512, "d_model": 512, "nhead": 8},
        {"seq_len": 1024, "d_model": 512, "nhead": 8},
        {"seq_len": 2048, "d_model": 512, "nhead": 8},
        {"seq_len": 4096, "d_model": 512, "nhead": 8},
        {"seq_len": 8192, "d_model": 512, "nhead": 8},
    ]

    window_sizes = [64, 128, 256]
    batch_size = 4  # Fixed batch size for fair comparison

    for config in configs:
        seq_len = config["seq_len"]
        d_model = config["d_model"]
        nhead = config["nhead"]

        print(f"\n{'='*60}")
        print(f"Testing seq_len={seq_len}, d_model={d_model}, nhead={nhead}")
        print(f"Fixed batch_size={batch_size}")
        print(f"{'='*60}")

        # Test full attention
        try:
            torch.cuda.empty_cache()  # Clear cache before each test
            full_result = benchmark_full_attention(
                seq_len, d_model, nhead, batch_size=batch_size
            )
            print(f"Full Attention: {full_result}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Full Attention: OOM at batch_size={batch_size}")
                continue
            else:
                raise e

        # Test FlexAttention for different window sizes
        for window_size in window_sizes:
            try:
                torch.cuda.empty_cache()  # Clear cache before each test
                flex_result = benchmark_flex_attention(
                    seq_len, d_model, nhead, window_size, batch_size=batch_size
                )

                if flex_result:
                    print(f"FlexAttention (w={window_size}): {flex_result}")

                    # Calculate relative performance
                    memory_ratio = flex_result["memory_mb"] / full_result["memory_mb"]
                    speed_ratio = flex_result["time_ms"] / full_result["time_ms"]
                    speedup = 1.0 / speed_ratio
                    memory_reduction = (1.0 - memory_ratio) * 100

                    print(
                        f"  Memory ratio: {memory_ratio:.2f}x ({memory_reduction:+.1f}%)"
                    )
                    print(f"  Speed ratio: {speed_ratio:.2f}x ({speedup:.2f}x speedup)")
                    print(
                        f"  Tokens/sec ratio: {flex_result['tokens_per_sec']/full_result['tokens_per_sec']:.2f}x"
                    )
                else:
                    print(f"FlexAttention (w={window_size}): Not available")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        f"FlexAttention (w={window_size}): OOM at batch_size={batch_size}"
                    )
                else:
                    print(f"FlexAttention (w={window_size}): Error - {e}")

        print()  # Add spacing between configurations


if __name__ == "__main__":
    print("🔍 Comparing Full Attention vs FlexAttention")
    print("=" * 60)
    print("This benchmark compares:")
    print("  • Full Attention: Standard PyTorch TransformerEncoderLayer (O(L²))")
    print("  • FlexAttention: PyTorch nightly sliding window attention (O(L×W))")
    print("=" * 60)

    compare_attention_implementations()

    print("\n🎯 Summary:")
    print("  • Memory ratio < 1.0 means FlexAttention uses less memory")
    print("  • Speed ratio < 1.0 means FlexAttention is faster")
    print("  • Window size determines attention span (larger = more context)")
    print("=" * 60)
