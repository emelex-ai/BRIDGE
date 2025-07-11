import time
from torch.nn import functional as F
import torch
from torch import nn

from bridge.domain.model.benchmarks.sdpa_sliding_window_model import (
    SDPASlidingWindowAttention
)

def benchmark_sdpa_sliding_window(
    seq_len, d_model=1024, nhead=1, window_size=128, batch_size=4
):
    """Benchmark SDPA sliding window attention with efficient mask computation.

    This version computes masks on-the-fly to avoid memory issues with
    very large sequence lengths.

    Args:
        seq_len: Sequence length
        d_model: Model dimension (default 1024)
        nhead: Number of attention heads (default 1)
        window_size: Sliding window size
        batch_size: Batch size for testing

    Returns:
        Dictionary with benchmark results

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(
        #f"Benchmarking SDPA Sliding Window (TRAINING mode, O(n), window={window_size}) on {device}"
    #)

    # Create model with efficient sliding window
    model = SDPASlidingWindowAttention(d_model, nhead, window_size).to(device)
    model.train()  # Training mode

    # Create input data
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    #print(f"  Using efficient on-the-fly mask computation...")

    # Warmup
    #print("  Warming up in training mode...")
    for _ in range(3):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    # Clear memory before benchmark
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark with backward pass
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
        "attention_type": "SDPA_Sliding_Window_Efficient",
        "mode": "TRAINING",
        "complexity": "O(n)",
        "window_size": window_size,
        "batch_size": batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "seq_len": seq_len,
        "d_model": d_model,
        "nhead": nhead,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }
