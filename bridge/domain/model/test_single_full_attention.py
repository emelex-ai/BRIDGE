#!/usr/bin/env python3
"""
Test a single full attention configuration and append to CSV.
Enhanced to support both test and train modes with loss computation.
Usage: python test_single_full_attention.py <seq_len> <d_model> <nhead> <batch_size> <output_csv> [mode]
"""

import csv
import gc
import os
import sys
import time

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def test_full_attention(seq_len, d_model, nhead, batch_size, output_csv, mode="test"):
    """Test full attention for a single configuration and append to CSV.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        nhead: Number of attention heads
        batch_size: Batch size
        output_csv: CSV file to append results to
        mode: Either "test" or "train" mode

    """
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

    # Create input and target for training mode
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Setup training components if in train mode
    if mode == "train":
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        criterion = nn.MSELoss()
        # Create target tensor (same shape as output)
        target = torch.randn(batch_size, seq_len, d_model, device=device)

        # Check optimizer state precision after first step
        print(f"\n=== OPTIMIZER STATE PRECISION CHECK ===")
        print(f"Model parameters dtype: {next(model.parameters()).dtype}")
        print(f"Input tensor dtype: {x.dtype}")
        print(f"Target tensor dtype: {target.dtype}")

        # Do one forward/backward pass to initialize optimizer states
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Now check optimizer state precision
        print(f"Optimizer states after first step:")
        for group in optimizer.state_dict()["state"].values():
            for k, v in group.items():
                print(f"  {k}: {v.dtype}, {v.device}")
        print(f"=========================================\n")
    else:
        model.eval()

    # Reset memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Warmup
    for _ in range(3):
        if mode == "train":
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                _ = model(x)

    # Reset after warmup
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    total_loss = 0.0
    for i in range(10):
        if mode == "train":
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        else:
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
    avg_loss = total_loss / 10 if mode == "train" else None

    # Clean up
    del model, x, output
    if mode == "train":
        del optimizer, criterion, target, loss
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
            "mode",
            "memory_mb",
            "time_ms",
            "tokens_per_sec",
        ]

        if mode == "train":
            fieldnames.append("avg_loss")

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row_data = {
            "seq_len": seq_len,
            "d_model": d_model,
            "nhead": nhead,
            "batch_size": batch_size,
            "mode": mode,
            "memory_mb": memory_mb,
            "time_ms": avg_time_ms,
            "tokens_per_sec": tokens_per_sec,
        }

        if mode == "train":
            row_data["avg_loss"] = avg_loss

        writer.writerow(row_data)

    if mode == "train":
        print(
            f"FULL_ATTENTION ({mode.upper()}): seq_len={seq_len}, "
            f"memory={memory_mb:.1f}MB, time={avg_time_ms:.3f}ms, "
            f"avg_loss={avg_loss:.6f}"
        )
    else:
        print(
            f"FULL_ATTENTION ({mode.upper()}): seq_len={seq_len}, "
            f"memory={memory_mb:.1f}MB, time={avg_time_ms:.3f}ms"
        )


if __name__ == "__main__":
    if len(sys.argv) < 6 or len(sys.argv) > 7:
        print(
            "Usage: python test_single_full_attention.py <seq_len> <d_model> <nhead> <batch_size> <output_csv> [mode]"
        )
        print("mode: 'test' (default) or 'train'")
        sys.exit(1)

    seq_len = int(sys.argv[1])
    d_model = int(sys.argv[2])
    nhead = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    output_csv = sys.argv[5]
    mode = sys.argv[6] if len(sys.argv) == 7 else "test"

    if mode not in ["test", "train"]:
        print("Error: mode must be 'test' or 'train'")
        sys.exit(1)

    print(f"Inside test_single_attention: seq_len={seq_len}, mode={mode}")

    test_full_attention(seq_len, d_model, nhead, batch_size, output_csv, mode)
