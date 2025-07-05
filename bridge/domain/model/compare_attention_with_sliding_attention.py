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
        "attention_type": "full_standard",
    }


def benchmark_flex_full_attention(seq_len, d_model, nhead, num_layers=1, batch_size=4):
    """Benchmark FlexAttention configured as full attention (no window limit)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Import FlexAttention implementation
        from bridge.domain.model.transformer_flex_attention import (
            FlexAttentionEncoderLayer,
        )

        # Create model with very large window size (effectively full attention)
        encoder_layer = FlexAttentionEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            window_size=seq_len
            * 2,  # Much larger than sequence length = full attention
            causal=False,  # Non-causal for true full attention
        )

        # Create encoder with multiple layers if needed
        class FlexFullAttentionEncoder(nn.Module):
            def __init__(self, layer, num_layers):
                super().__init__()
                self.layers = nn.ModuleList([layer for _ in range(num_layers)])

            def forward(self, x, mask=None):
                for layer in self.layers:
                    x = layer(x, src_mask=mask)
                return x

        model = FlexFullAttentionEncoder(encoder_layer, num_layers).to(device)

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
            "window_size": seq_len * 2,
            "attention_type": "flex_full",
        }

    except ImportError as e:
        print(f"FlexAttention not available: {e}")
        return None
    except Exception as e:
        print(f"FlexAttention (full) benchmark failed: {e}")
        return None


def benchmark_flex_sliding_window(
    seq_len, d_model, nhead, window_size, num_layers=1, batch_size=4
):
    """Benchmark FlexAttention with sliding window."""
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
            causal=False,  # Non-causal for fair comparison
        )

        # Create encoder with multiple layers if needed
        class FlexSlidingWindowEncoder(nn.Module):
            def __init__(self, layer, num_layers):
                super().__init__()
                self.layers = nn.ModuleList([layer for _ in range(num_layers)])

            def forward(self, x, mask=None):
                for layer in self.layers:
                    x = layer(x, src_mask=mask)
                return x

        model = FlexSlidingWindowEncoder(encoder_layer, num_layers).to(device)

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
            "attention_type": "flex_sliding",
        }

    except ImportError as e:
        print(f"FlexAttention not available: {e}")
        return None
    except Exception as e:
        print(f"FlexAttention (sliding) benchmark failed: {e}")
        return None


def compare_attention_implementations():
    """Compare Full Attention vs FlexAttention (full) vs FlexAttention (sliding)."""
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

        print(f"\n{'='*70}")
        print(f"Testing seq_len={seq_len}, d_model={d_model}, nhead={nhead}")
        print(f"Fixed batch_size={batch_size}")
        print(f"{'='*70}")

        results = {}

        # Test 1: Standard Full Attention
        print("\n1️⃣ Standard Full Attention (PyTorch)")
        print("-" * 40)
        try:
            torch.cuda.empty_cache()
            full_result = benchmark_full_attention(
                seq_len, d_model, nhead, batch_size=batch_size
            )
            results["full_standard"] = full_result
            print(f"✓ {full_result}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ OOM at batch_size={batch_size}")
                continue
            else:
                raise e

        # Test 2: FlexAttention configured as Full Attention
        print("\n2️⃣ FlexAttention (Full Attention Mode)")
        print("-" * 40)
        try:
            torch.cuda.empty_cache()
            flex_full_result = benchmark_flex_full_attention(
                seq_len, d_model, nhead, batch_size=batch_size
            )

            if flex_full_result:
                results["flex_full"] = flex_full_result
                print(f"✓ {flex_full_result}")

                # Compare with standard full attention
                if "full_standard" in results:
                    standard = results["full_standard"]
                    memory_ratio = flex_full_result["memory_mb"] / standard["memory_mb"]
                    speed_ratio = flex_full_result["time_ms"] / standard["time_ms"]
                    speedup = 1.0 / speed_ratio

                    print(f"  📊 vs Standard Full:")
                    print(f"     Memory ratio: {memory_ratio:.2f}x")
                    print(
                        f"     Speed ratio: {speed_ratio:.2f}x ({speedup:.2f}x speedup)"
                    )
            else:
                print("❌ FlexAttention not available")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ OOM at batch_size={batch_size}")
            else:
                print(f"❌ Error: {e}")

        # Test 3: FlexAttention with Sliding Window
        print("\n3️⃣ FlexAttention (Sliding Window)")
        print("-" * 40)
        for window_size in window_sizes:
            try:
                torch.cuda.empty_cache()
                flex_sliding_result = benchmark_flex_sliding_window(
                    seq_len, d_model, nhead, window_size, batch_size=batch_size
                )

                if flex_sliding_result:
                    print(f"✓ Window={window_size}: {flex_sliding_result}")

                    # Compare with both full attention methods
                    if "full_standard" in results:
                        standard = results["full_standard"]
                        memory_ratio = (
                            flex_sliding_result["memory_mb"] / standard["memory_mb"]
                        )
                        speed_ratio = (
                            flex_sliding_result["time_ms"] / standard["time_ms"]
                        )
                        speedup = 1.0 / speed_ratio
                        memory_reduction = (1.0 - memory_ratio) * 100

                        print(f"  📊 vs Standard Full:")
                        print(
                            f"     Memory: {memory_ratio:.2f}x ({memory_reduction:+.1f}%)"
                        )
                        print(
                            f"     Speed: {speed_ratio:.2f}x ({speedup:.2f}x speedup)"
                        )

                    if "flex_full" in results:
                        flex_full = results["flex_full"]
                        memory_ratio = (
                            flex_sliding_result["memory_mb"] / flex_full["memory_mb"]
                        )
                        speed_ratio = (
                            flex_sliding_result["time_ms"] / flex_full["time_ms"]
                        )
                        speedup = 1.0 / speed_ratio
                        memory_reduction = (1.0 - memory_ratio) * 100

                        print(f"  📊 vs FlexAttention Full:")
                        print(
                            f"     Memory: {memory_ratio:.2f}x ({memory_reduction:+.1f}%)"
                        )
                        print(
                            f"     Speed: {speed_ratio:.2f}x ({speedup:.2f}x speedup)"
                        )
                else:
                    print(f"❌ Window={window_size}: Not available")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ Window={window_size}: OOM")
                else:
                    print(f"❌ Window={window_size}: Error - {e}")

        print()  # Add spacing between configurations


if __name__ == "__main__":
    print("🔍 Three-Way Attention Comparison")
    print("=" * 70)
    print("This benchmark compares:")
    print("  1️⃣ Standard Full Attention: PyTorch TransformerEncoderLayer (O(L²))")
    print("  2️⃣ FlexAttention Full Mode: FlexAttention without window limit (O(L²))")
    print("  3️⃣ FlexAttention Sliding: FlexAttention with sliding window (O(L×W))")
    print("=" * 70)
    print("📊 Key Insights:")
    print("  • Compare 1️⃣ vs 2️⃣ to see FlexAttention overhead")
    print("  • Compare 2️⃣ vs 3️⃣ to see sliding window benefits")
    print("  • Compare 1️⃣ vs 3️⃣ to see overall FlexAttention sliding window performance")
    print("=" * 70)

    compare_attention_implementations()

    print("\n🎯 Interpretation Guide:")
    print("  • Memory/Speed ratio < 1.0 = improvement")
    print("  • Memory/Speed ratio > 1.0 = overhead")
    print("  • Window size determines attention span vs efficiency trade-off")
    print("=" * 70)
