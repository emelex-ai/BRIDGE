import time
from torch.nn import functional as F
import torch
from torch import nn

from bridge.domain.model.benchmarks.fast_sliding_window import (
    FastSlidingWindowModel
)


def benchmark_fast_sliding_window(
    seq_len, d_model=1024, nhead=1, window_size=128, batch_size=4
):
    """Benchmark fast sliding window attention (simple and efficient).

    This should be much faster than the chunked version.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(
        #f"Benchmarking Fast Sliding Window (TRAINING mode, O(n), window={window_size}) on {device}"
    #)

    # Create model
    model = FastSlidingWindowModel(d_model, nhead, window_size).to(device)
    model.train()

    # Create input data
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    #print(f"  Using simple mask-based approach (single SDPA call)...")

    # Warmup
    #print("  Warming up in training mode...")
    for _ in range(3):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    return {
        "attention_type": "Fast_Sliding_Window",
        "mode": "TRAINING",
        "complexity": "O(n)",
        "window_size": window_size,
        "batch_size": batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "seq_len": seq_len,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }
