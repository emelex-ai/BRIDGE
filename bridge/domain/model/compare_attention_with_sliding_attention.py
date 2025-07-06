import gc
import time

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def benchmark_full_attention(seq_len, d_model, nhead, num_layers=1, batch_size=4):
    """Benchmark standard TransformerEncoder with fixed batch size.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of encoder layers
        batch_size: Batch size for testing

    Returns:
        Dictionary with benchmark results

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking full attention on {device}")

    # Create model
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
    )
    model = TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

    # Use fixed batch size
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Warmup
    print("  Warming up...")
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
    }


def benchmark_sliding_window(
    seq_len, d_model, nhead, window_size, num_layers=1, batch_size=4
):
    """Benchmark sliding window implementation with fixed batch size.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        nhead: Number of attention heads
        window_size: Sliding window size
        num_layers: Number of encoder layers
        batch_size: Batch size for testing

    Returns:
        Dictionary with benchmark results or None if failed

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking sliding window attention on {device}")

    try:
        # Import sliding window implementation
        from bridge.domain.model.true_sliding_window_attention import (
            TrueSlidingWindowEncoderLayer,
        )

        print("  ✅ Successfully imported TrueSlidingWindowEncoderLayer")
    except ImportError as e:
        print(f"  ❌ Failed to import TrueSlidingWindowEncoderLayer: {e}")
        return None

    try:
        # Create model
        encoder_layer = TrueSlidingWindowEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            window_size=window_size,
        )

        # Create encoder with multiple layers if needed
        class SlidingWindowEncoder(nn.Module):
            def __init__(self, layer, num_layers):
                super().__init__()
                self.layers = nn.ModuleList([layer for _ in range(num_layers)])

            def forward(self, x, mask=None):
                for layer in self.layers:
                    x = layer(x, src_mask=mask)
                return x

        model = SlidingWindowEncoder(encoder_layer, num_layers).to(device)

        # Use fixed batch size
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        # Warmup
        print("  Warming up...")
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
        }
    except Exception as e:
        print(f"  ❌ Sliding window benchmark failed: {e}")
        return None


def benchmark_flex_attention(
    seq_len, d_model, nhead, window_size, num_layers=1, batch_size=4
):
    """Benchmark FlexAttention implementation with fixed batch size.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        nhead: Number of attention heads
        window_size: Sliding window size
        num_layers: Number of encoder layers
        batch_size: Batch size for testing

    Returns:
        Dictionary with benchmark results or None if failed

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking FlexAttention on {device}")

    try:
        # Import FlexAttention implementation
        from bridge.domain.model.transformer_flex_attention import (
            FlexAttentionEncoderLayer,
        )

        print("  ✅ Successfully imported FlexAttentionEncoderLayer")
    except ImportError as e:
        print(f"  ❌ Failed to import FlexAttentionEncoderLayer: {e}")
        return None

    try:
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
        print("  Warming up...")
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

    except Exception as e:
        print(f"  ❌ FlexAttention benchmark failed: {e}")
        return None


def compare_attention_implementations():
    """Compare sliding window vs full attention with fixed batch size."""
    print("🚀 Starting Attention Implementation Comparison")
    print("=" * 80)

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
            print("\n🔬 Testing Full Attention...")
            full_result = benchmark_full_attention(
                seq_len, d_model, nhead, batch_size=batch_size
            )
            print(f"✅ Full Attention: {full_result}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Full Attention: OOM at batch_size={batch_size}")
                continue
            else:
                print(f"❌ Full Attention error: {e}")
                continue

        # Test sliding window for different window sizes
        for window_size in window_sizes:
            # Test TrueSlidingWindow
            try:
                print(f"\n🔬 Testing TrueSlidingWindow (window={window_size})...")
                sw_result = benchmark_sliding_window(
                    seq_len, d_model, nhead, window_size, batch_size=batch_size
                )
                if sw_result:
                    print(f"✅ TrueSlidingWindow (w={window_size}): {sw_result}")

                    # Calculate relative performance
                    memory_ratio = sw_result["memory_mb"] / full_result["memory_mb"]
                    speed_ratio = sw_result["time_ms"] / full_result["time_ms"]
                    print(f"  📊 Memory ratio: {memory_ratio:.2f}x")
                    print(f"  📊 Speed ratio: {speed_ratio:.2f}x")
                else:
                    print(f"❌ TrueSlidingWindow (w={window_size}): Failed")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        f"❌ TrueSlidingWindow (w={window_size}): OOM at batch_size={batch_size}"
                    )
                else:
                    print(f"❌ TrueSlidingWindow (w={window_size}): Error - {e}")

            # Test FlexAttention
            try:
                print(f"\n🔬 Testing FlexAttention (window={window_size})...")
                flex_result = benchmark_flex_attention(
                    seq_len, d_model, nhead, window_size, batch_size=batch_size
                )
                if flex_result:
                    print(f"✅ FlexAttention (w={window_size}): {flex_result}")

                    # Calculate relative performance
                    memory_ratio = flex_result["memory_mb"] / full_result["memory_mb"]
                    speed_ratio = flex_result["time_ms"] / full_result["time_ms"]
                    print(f"  📊 Memory ratio: {memory_ratio:.2f}x")
                    print(f"  📊 Speed ratio: {speed_ratio:.2f}x")
                else:
                    print(f"❌ FlexAttention (w={window_size}): Failed")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        f"❌ FlexAttention (w={window_size}): OOM at batch_size={batch_size}"
                    )
                else:
                    print(f"❌ FlexAttention (w={window_size}): Error - {e}")

        # Clean up memory
        torch.cuda.empty_cache()

    print("\n🏁 Comparison complete!")


if __name__ == "__main__":
    """Main execution with error handling.
    
    """
    try:
        compare_attention_implementations()
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback

        traceback.print_exc()
