import gc
import time

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def compare_attention_implementations():
    """Compare sliding window vs full attention performance and memory usage.

    This function tests the real-world benefits of sliding window attention
    by comparing memory usage and throughput at different configurations.
    """
    # Test configurations
    configs = [
        {"seq_len": 512, "d_model": 512, "nhead": 8},
        {"seq_len": 1024, "d_model": 512, "nhead": 8},
        {"seq_len": 2048, "d_model": 512, "nhead": 8},
        {"seq_len": 4096, "d_model": 512, "nhead": 8},
    ]

    window_sizes = [64, 128, 256]

    results = []

    for config in configs:
        seq_len = config["seq_len"]
        d_model = config["d_model"]
        nhead = config["nhead"]

        print(f"\n{'='*60}")
        print(f"Testing seq_len={seq_len}, d_model={d_model}, nhead={nhead}")
        print(f"{'='*60}")

        # Test full attention
        try:
            full_result = benchmark_full_attention(seq_len, d_model, nhead)
            print(f"Full Attention: {full_result}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                full_result = {"error": "OOM", "max_batch_size": 0}
                print(f"Full Attention: OOM")
            else:
                raise e

        # Test sliding window for different window sizes
        for window_size in window_sizes:
            try:
                sw_result = benchmark_sliding_window(
                    seq_len, d_model, nhead, window_size
                )
                print(f"Sliding Window (w={window_size}): {sw_result}")

                # Calculate relative performance
                if full_result.get("error") != "OOM":
                    memory_ratio = sw_result["memory_mb"] / full_result["memory_mb"]
                    speed_ratio = sw_result["time_ms"] / full_result["time_ms"]
                    print(f"  Memory ratio: {memory_ratio:.2f}x")
                    print(f"  Speed ratio: {speed_ratio:.2f}x")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Sliding Window (w={window_size}): OOM")
                else:
                    raise e


def benchmark_full_attention(seq_len, d_model, nhead, num_layers=1):
    """Benchmark standard TransformerEncoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
    )
    model = TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

    # Find maximum batch size that fits in memory
    max_batch_size = find_max_batch_size(model, seq_len, d_model, device)

    if max_batch_size == 0:
        return {"error": "OOM", "max_batch_size": 0}

    # Benchmark with max batch size
    batch_size = max_batch_size
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
        "max_batch_size": max_batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }


def benchmark_sliding_window(seq_len, d_model, nhead, window_size, num_layers=1):
    """Benchmark your sliding window implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import your sliding window implementation
    from bridge.domain.model.true_sliding_window_attention import (
        TrueSlidingWindowEncoderLayer,
    )

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

    # Find maximum batch size
    max_batch_size = find_max_batch_size(model, seq_len, d_model, device)

    if max_batch_size == 0:
        return {"error": "OOM", "max_batch_size": 0}

    # Benchmark with max batch size
    batch_size = max_batch_size
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
        "max_batch_size": max_batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
        "window_size": window_size,
    }


def find_max_batch_size(model, seq_len, d_model, device, max_batch_size=512):
    """Binary search to find maximum batch size that fits in memory."""
    low, high = 1, max_batch_size
    max_working_batch = 0

    while low <= high:
        mid = (low + high) // 2

        try:
            torch.cuda.empty_cache()
            gc.collect()

            x = torch.randn(mid, seq_len, d_model, device=device)
            with torch.no_grad():
                _ = model(x)

            max_working_batch = mid
            low = mid + 1

        except RuntimeError as e:
            if "out of memory" in str(e):
                high = mid - 1
            else:
                raise e
        finally:
            if "x" in locals():
                del x
            torch.cuda.empty_cache()
            gc.collect()

    return max_working_batch


if __name__ == "__main__":
    compare_attention_implementations()
