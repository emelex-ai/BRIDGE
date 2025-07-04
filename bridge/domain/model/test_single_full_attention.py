#!/usr/bin/env python3
"""
Test a single full attention configuration and append to CSV.
Usage: python test_single_full_attention.py <seq_len> <d_model> <nhead> <batch_size> <output_csv>
"""

import csv
import gc
import os
import sys
import time

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def test_full_attention(seq_len, d_model, nhead, batch_size, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clear GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Create model
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
    )
    model = TransformerEncoder(encoder_layer, num_layers=1).to(device)

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
                "memory_mb": memory_mb,
                "time_ms": avg_time_ms,
                "tokens_per_sec": tokens_per_sec,
            }
        )

    print(
        f"FULL_ATTENTION: seq_len={seq_len}, memory={memory_mb:.1f}MB, time={avg_time_ms:.3f}ms"
    )


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(
            "Usage: python test_single_full_attention.py <seq_len> <d_model> <nhead> <batch_size> <output_csv>"
        )
        sys.exit(1)

    seq_len = int(sys.argv[1])
    d_model = int(sys.argv[2])
    nhead = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    output_csv = sys.argv[5]

    test_full_attention(seq_len, d_model, nhead, batch_size, output_csv)
