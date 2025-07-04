import csv
import os
import sys
import time

import torch
from local_attention.local_attention import LocalAttention


def benchmark_attention(seq_len, d_model, nhead, batch_size, window_size, output_csv):
    """
    Benchmarks a single sliding-window attention layer.

    Args:
        seq_len (int): The sequence length.
        d_model (int): The model dimension.
        nhead (int): The number of attention heads.
        batch_size (int): The batch size.
        window_size (int): The sliding window size.
        output_csv (str): The path to the output CSV file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print(
            "Warning: CUDA not available, running on CPU. Performance results will not be representative."
        )

    assert d_model % nhead == 0, "d_model must be divisible by nhead"
    dim_head = d_model // nhead

    model = LocalAttention(window_size=window_size, causal=True, autopad=True).to(
        device
    )

    q = torch.rand(batch_size * nhead, seq_len, dim_head, device=device)
    k = torch.rand(batch_size * nhead, seq_len, dim_head, device=device)
    v = torch.rand(batch_size * nhead, seq_len, dim_head, device=device)

    # Warm-up
    for _ in range(5):
        _ = model(q, k, v)
    torch.cuda.synchronize()

    # Measurement
    start_mem = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    _ = model(q, k, v)
    torch.cuda.synchronize()
    end_time = time.time()

    peak_mem = torch.cuda.max_memory_allocated(device)
    memory_used_mb = (peak_mem - start_mem) / (1024 * 1024)

    time_ms = (end_time - start_time) * 1000
    tokens_per_sec = (batch_size * seq_len) / (time_ms / 1000)

    print(
        f"SeqLen: {seq_len}, Window: {window_size}, Memory: {memory_used_mb:.2f}MB, Time: {time_ms:.2f}ms, Tokens/Sec: {tokens_per_sec:,.0f}"
    )

    file_exists = os.path.isfile(output_csv)

    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "seq_len",
                    "d_model",
                    "nhead",
                    "batch_size",
                    "window_size",
                    "memory_mb",
                    "time_ms",
                    "tokens_per_sec",
                ]
            )
        writer.writerow(
            [
                seq_len,
                d_model,
                nhead,
                batch_size,
                window_size,
                memory_used_mb,
                time_ms,
                tokens_per_sec,
            ]
        )


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(
            "Usage: python test_single_sliding_attention.py <seq_len> <d_model> <nhead> <batch_size> <window_size> <output_csv>"
        )
        sys.exit(1)

    seq_len = int(sys.argv[1])
    d_model = int(sys.argv[2])
    nhead = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    window_size = int(sys.argv[5])
    output_csv = sys.argv[6]

    benchmark_attention(
        seq_len, d_model, nhead, batch_size, window_size, output_csv
    )
