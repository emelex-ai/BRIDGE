#!/usr/bin/env python3
"""
Test attention modules in isolation to debug memory usage.
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


def test_attention_only():
    """Test attention modules in isolation and save results to CSV."""
    print("=" * 60)
    print("Testing Attention Modules in Isolation")
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

                        # Clean up memory
                        if device == "cuda":
                            torch.cuda.empty_cache()
                            gc.collect()

                        # Test 1: SDPA Full Attention
                        try:
                            print("  Running SDPA Full Attention...")
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

                        # Test 2: SDPA Sliding Window Attention
                        try:
                            print("  Running SDPA Sliding Window Attention...")
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

                        print()

    # Save results to CSV
    try:
        import pandas as pd

        df = pd.DataFrame(all_results)

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attention_only_results_{timestamp}.csv"

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
                df[["attention_type", "memory_mb", "time_ms", "window_size"]].to_string(
                    index=False
                )
            )

        print("=" * 60)

    except ImportError:
        print("❌ pandas not available, saving results as JSON instead")
        import json

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attention_only_results_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"Results saved to: {filename}")


if __name__ == "__main__":
    test_attention_only()
