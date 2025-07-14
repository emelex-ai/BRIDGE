#!/usr/bin/env python3
"""
Test attention modules in isolation and full layers to debug memory usage.
"""

import datetime
import gc
import time

import torch
import torch.nn as nn

# Use the existing benchmark functions
from bridge.domain.model.benchmarks import (
    benchmark_sdpa_full_attention,
    benchmark_sdpa_sliding_window,
)


def test_attention_and_full_layer():
    """Test both attention-only and full layer implementations."""
    print("=" * 60)
    print("Testing Attention-Only vs Full Layer")
    print("=" * 60)

    # Test parameters - match your encoder_sdpa.py test
    seq_lens = [4096]
    d_models = [512]
    nheads = [1]
    batch_sizes = [1]
    window_sizes = [32]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running tests on: {device}")
    print()

    # Store all results
    all_results = []

    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            for d_model in d_models:
                for nhead in nheads:
                    # Skip invalid combinations
                    if d_model % nhead != 0:
                        continue

                    for window_size in window_sizes:
                        # Skip invalid combinations
                        if window_size > seq_len:
                            continue

                        print(
                            f"Testing: seq_len={seq_len}, d_model={d_model}, nhead={nhead}, window_size={window_size}, batch_size={batch_size}"
                        )

                        # Test 1: SDPA Full Attention (Attention Only)
                        try:
                            print("  Running SDPA Full Attention (Attention Only)...")
                            # Clean up memory before test
                            if device == "cuda":
                                torch.cuda.empty_cache()
                                gc.collect()

                            full_result = benchmark_sdpa_full_attention(
                                seq_len,
                                d_model=d_model,
                                nhead=nhead,
                                batch_size=batch_size,
                            )
                            full_result["test_type"] = "attention_only"
                            full_result["seq_len"] = seq_len
                            full_result["d_model"] = d_model
                            full_result["nhead"] = nhead
                            full_result["window_size"] = None
                            full_result["batch_size"] = batch_size
                            all_results.append(full_result)
                            print(
                                f"    ✓ Memory: {full_result['memory_mb']:.1f}MB, Time: {full_result['time_ms']:.2f}ms"
                            )
                        except Exception as e:
                            print(f"    ❌ SDPA Full failed: {str(e)}")

                        # Test 2: SDPA Sliding Window Attention (Attention Only)
                        try:
                            print(
                                "  Running SDPA Sliding Window Attention (Attention Only)..."
                            )
                            # Clean up memory before test
                            if device == "cuda":
                                torch.cuda.empty_cache()
                                gc.collect()

                            sliding_result = benchmark_sdpa_sliding_window(
                                seq_len,
                                d_model=d_model,
                                nhead=nhead,
                                window_size=window_size,
                                batch_size=batch_size,
                            )
                            sliding_result["test_type"] = "attention_only"
                            sliding_result["seq_len"] = seq_len
                            sliding_result["d_model"] = d_model
                            sliding_result["nhead"] = nhead
                            sliding_result["window_size"] = window_size
                            sliding_result["batch_size"] = batch_size
                            all_results.append(sliding_result)
                            print(
                                f"    ✓ Memory: {sliding_result['memory_mb']:.1f}MB, Time: {sliding_result['time_ms']:.2f}ms"
                            )
                        except Exception as e:
                            print(f"    ❌ SDPA Sliding failed: {str(e)}")

                        # Test 3: SDPA Full Attention (Full Layer)
                        try:
                            print("  Running SDPA Full Attention (Full Layer)...")
                            # Clean up memory before test
                            if device == "cuda":
                                torch.cuda.empty_cache()
                                gc.collect()

                            full_layer_result = (
                                benchmark_sdpa_full_attention_full_layer(
                                    seq_len,
                                    d_model=d_model,
                                    nhead=nhead,
                                    batch_size=batch_size,
                                )
                            )
                            full_layer_result["test_type"] = "full_layer"
                            full_layer_result["seq_len"] = seq_len
                            full_layer_result["d_model"] = d_model
                            full_layer_result["nhead"] = nhead
                            full_layer_result["window_size"] = None
                            full_layer_result["batch_size"] = batch_size
                            all_results.append(full_layer_result)
                            print(
                                f"    ✓ Memory: {full_layer_result['memory_mb']:.1f}MB, Time: {full_layer_result['time_ms']:.2f}ms"
                            )
                        except Exception as e:
                            print(f"    ❌ SDPA Full Layer failed: {str(e)}")

                        # Test 4: SDPA Sliding Window Attention (Full Layer)
                        try:
                            print(
                                "  Running SDPA Sliding Window Attention (Full Layer)..."
                            )
                            # Clean up memory before test
                            if device == "cuda":
                                torch.cuda.empty_cache()
                                gc.collect()

                            sliding_layer_result = (
                                benchmark_sdpa_sliding_window_full_layer(
                                    seq_len,
                                    d_model=d_model,
                                    nhead=nhead,
                                    window_size=window_size,
                                    batch_size=batch_size,
                                )
                            )
                            sliding_layer_result["test_type"] = "full_layer"
                            sliding_layer_result["seq_len"] = seq_len
                            sliding_layer_result["d_model"] = d_model
                            sliding_layer_result["nhead"] = nhead
                            sliding_layer_result["window_size"] = window_size
                            sliding_layer_result["batch_size"] = batch_size
                            all_results.append(sliding_layer_result)
                            print(
                                f"    ✓ Memory: {sliding_layer_result['memory_mb']:.1f}MB, Time: {sliding_layer_result['time_ms']:.2f}ms"
                            )
                        except Exception as e:
                            print(f"    ❌ SDPA Sliding Layer failed: {str(e)}")

                        print()

    # Save results to CSV
    try:
        import pandas as pd

        df = pd.DataFrame(all_results)

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attention_vs_full_layer_results_{timestamp}.csv"

        df.to_csv(filename, index=False)

        print("=" * 60)
        print("RESULTS SAVED")
        print("=" * 60)
        print(f"Results saved to: {filename}")
        print(f"Total tests: {len(all_results)}")

        # Show summary
        if len(df) > 0:
            print("\nSummary:")
            print(
                df[
                    [
                        "attention_type",
                        "test_type",
                        "memory_mb",
                        "time_ms",
                        "window_size",
                    ]
                ].to_string(index=False)
            )

        print("=" * 60)

    except ImportError:
        print("❌ pandas not available, saving results as JSON instead")
        import json

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attention_vs_full_layer_results_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"Results saved to: {filename}")


def benchmark_sdpa_full_attention_full_layer(
    seq_len, d_model=1024, nhead=1, num_layers=1, batch_size=4
):
    """Benchmark SDPA full attention with full layer (including feedforward, norms)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import the full layer model
    from bridge.domain.model.benchmarks.sdpa_full_attention_model import SDPAFullLayer

    # Create full layer model
    model = SDPAFullLayer(d_model=d_model, nhead=nhead).to(device)
    model.train()

    # Create input with gradient tracking for training mode
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    # Warmup
    for _ in range(3):
        output = model(x)
        loss = output.sum()
        loss.backward()
        x.grad = None  # Clear gradients

    # Clear memory before benchmark
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        output = model(x)
        loss = output.sum()
        loss.backward()
        x.grad = None  # Clear gradients

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    return {
        "attention_type": "SDPA_Full_Attention",
        "mode": "TRAINING",
        "complexity": "O(n²)",
        "batch_size": batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "seq_len": seq_len,
        "d_model": d_model,
        "nhead": nhead,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }


def benchmark_sdpa_sliding_window_full_layer(
    seq_len, d_model=1024, nhead=1, window_size=128, batch_size=4
):
    """Benchmark SDPA sliding window attention with full layer (including feedforward, norms)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import the full layer model
    from bridge.domain.model.benchmarks.sdpa_sliding_window_model import (
        SDPASlidingWindowLayer,
    )

    # Create full layer model
    model = SDPASlidingWindowLayer(
        d_model=d_model,
        nhead=nhead,
        window_size=window_size,
        seq_len=seq_len,
        device=device,
    ).to(device)
    model.train()

    # Create input with gradient tracking for training mode
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    # Warmup
    for _ in range(3):
        output = model(x)
        loss = output.sum()
        loss.backward()
        x.grad = None  # Clear gradients

    # Clear memory before benchmark
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        output = model(x)
        loss = output.sum()
        loss.backward()
        x.grad = None  # Clear gradients

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


if __name__ == "__main__":
    test_attention_and_full_layer()
