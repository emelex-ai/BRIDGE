#!/usr/bin/env python3
"""
Test a single sliding window configuration and append to CSV.
Usage: python test_single_sliding_window.py <seq_len> <d_model> <nhead> <batch_size> <window_size> <output_csv>
"""

import csv
import gc
import os
import sys
import time

import torch
import torch.nn as nn


def test_sliding_window(seq_len, d_model, nhead, batch_size, window_size, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clear GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

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

    class SlidingWindowEncoder(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            return self.layer(x)

    model = SlidingWindowEncoder(encoder_layer).to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Reset memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(x)

    # Reset after warmup
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        with torch.no_grad():
            output = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    # Get measurements
    if device.type == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_mb = 0

    avg_time_ms = (end_time - start_time) / 10 * 1000
    tokens_per_sec = (batch_size * seq_len * 10) / (end_time - start_time)

    # Clean up
    del model, x, output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Append to CSV
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="") as csvfile:
        fieldnames = [
            "seq_len",
            "d_model",
            "nhead",
            "batch_size",
            "window_size",
            "memory_mb",
            "time_ms",
            "tokens_per_sec",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "seq_len": seq_len,
                "d_model": d_model,
                "nhead": nhead,
                "batch_size": batch_size,
                "window_size": window_size,
                "memory_mb": memory_mb,
                "time_ms": avg_time_ms,
                "tokens_per_sec": tokens_per_sec,
            }
        )

    print(
        f"SLIDING_WINDOW: seq_len={seq_len}, window={window_size}, memory={memory_mb:.1f}MB, time={avg_time_ms:.3f}ms"
    )


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(
            "Usage: python test_single_sliding_window.py <seq_len> <d_model> <nhead> <batch_size> <window_size> <output_csv>"
        )
        sys.exit(1)

    seq_len = int(sys.argv[1])
    d_model = int(sys.argv[2])
    nhead = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    window_size = int(sys.argv[5])
    output_csv = sys.argv[6]

    test_sliding_window(seq_len, d_model, nhead, batch_size, window_size, output_csv)
