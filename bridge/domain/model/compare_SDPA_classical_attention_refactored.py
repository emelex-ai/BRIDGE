import gc
import time

import torch

# Will only work if the benchmarks are in the same directory as this file
#   python -m bridge.domain.model.compare_SDPA_classical_attention_refactored
# I can also run as
#  `python bridge.domain.model.compare_SDPA_classical_attention_refactored.py``
from bridge.domain.model.benchmarks import (
    # benchmark_chunked_vectorized_sliding_window,
    benchmark_classical_full_attention,
    benchmark_classical_windowed_full_attention,
    benchmark_fast_sliding_window,
    benchmark_sdpa_full_attention,
    benchmark_sdpa_sliding_window,
    benchmark_true_vectorized_sliding_window,
    benchmark_true_vectorized_sliding_window_outer_loop,
)


def safe_divide(numerator: float, denominator: float) -> float:
    """Safely divide two numbers, returning NaN if denominator is zero."""
    return numerator / denominator if denominator != 0 else -1.0


def run_benchmark_tests(
    *,
    seq_len: int = 128,
    d_model: int = 128,
    nhead: int = 1,
    batch_size: int = 1,
    window_size: int = 32,
    run_full_attention: bool = False,
    run_sliding_window: bool = True,
) -> dict[str, dict]:
    """Run the benchmark tests.

    Args:
        seq_len: The length of the sequence.
        d_model: The dimension of the model.
        nhead: The number of attention heads.
        batch_size: The batch size.
        window_size: The size of the window.
        run_full_attention: Whether to run the full attention benchmark.
        run_sliding_window: Whether to run the sliding window benchmark.

    Returns:
        A dictionary containing the results of the benchmark tests.

    """
    results: dict[str, dict] = {}
    if run_full_attention:
        results["classical"] = benchmark_classical_full_attention(
            seq_len, d_model=d_model, nhead=nhead, batch_size=batch_size
        )

        results["sdpa_full"] = benchmark_sdpa_full_attention(
            seq_len, d_model=d_model, nhead=nhead, batch_size=batch_size
        )

    if run_sliding_window:
        results["classical_windowed"] = benchmark_classical_windowed_full_attention(
            seq_len,
            d_model=d_model,
            nhead=nhead,
            window_size=window_size,
            batch_size=batch_size,
        )

        results["sdpa_sliding"] = benchmark_sdpa_sliding_window(
            seq_len,
            d_model=d_model,
            nhead=nhead,
            window_size=window_size,
            batch_size=batch_size,
        )

        results["fast_sliding"] = benchmark_fast_sliding_window(
            seq_len,
            d_model=d_model,
            nhead=nhead,
            window_size=window_size,
            batch_size=batch_size,
        )

        results["true_vectorized_outer_loop"] = (
            benchmark_true_vectorized_sliding_window_outer_loop(
                seq_len,
                d_model=d_model,
                nhead=nhead,
                window_size=window_size,
                batch_size=batch_size,
            )
        )

        results["true_vectorized"] = benchmark_true_vectorized_sliding_window(
            seq_len,
            d_model=d_model,
            nhead=nhead,
            window_size=window_size,
            batch_size=batch_size,
        )

    return results


def print_results(result1: dict, result2: dict) -> None:
    """Print the results of two benchmarks."""
    memory_ratio = safe_divide(
        result1["memory_mb"],
        result2["memory_mb"],
    )
    time1 = result1["time_ms"]
    time2 = result2["time_ms"]
    time_ratio = time1 / time2
    label_memory = (
        f"{result1['attention_type']} memory / {result2['attention_type']} memory"
    )
    label_time = f"{result1['attention_type']} time / {result2['attention_type']} time"
    label_time_reverse = (
        f"{result2['attention_type']} time / {result1['attention_type']} time"
    )
    print()
    print(f"    {label_memory}: {memory_ratio:.2f}x")
    print(f"    {label_time}: {time_ratio:.2f}x")
    print(f"    {label_time_reverse}: {1. / time_ratio:.2f}x")


def run_test1(results_full: dict[str, dict]) -> None:
    """Run test 1.

    Args:
        results_full: The results of the full attention benchmark.
    """
    print("\n🔬 Test 1: Classical Full Attention (TRAINING mode, O(n²))...")
    classical_result = results_full["classical"]
    print(f"✅ {classical_result}")


def run_test2(results_full: dict[str, dict]) -> None:
    """Run test 2.

    Args:
        results_full: The results of the full attention benchmark.
    """
    print("\n🔬 Test 2: SDPA Full Attention (TRAINING mode, O(n²))...")
    sdpa_full_result = results_full["sdpa_full"]
    print(f"✅ {sdpa_full_result}")


def run_test1a(
    results_full: dict[str, dict],
    results_win: dict[str, dict],
    window_size: int,
) -> None:
    """Run test 1a.

    Args:
        results_win: The results of the windowed full attention benchmark.

    """
    print(
        f"\n🔬 Test 1a: Classical Windowed Full Attention (TRAINING mode, O(n²), window={window_size})..."
    )
    classical_windowed_result = results_win["classical_windowed"]
    classical_result = results_full["classical"]
    print_results(classical_result, classical_windowed_result)


def run_test3(
    results_full: dict[str, dict],
    results_win: dict[str, dict],
    window_size: int,
) -> None:
    """Run test 3.

    Args:
        results_win: The results of the sliding window benchmark.
    """
    print(
        f"\n🔬 Test 3: SDPA Sliding Window (TRAINING mode, O(n), window={window_size})..."
    )
    sdpa_full_result = results_full["sdpa_full"]
    sdpa_sliding_result = results_win["sdpa_sliding"]
    classical_result = results_full["classical"]
    classical_windowed_result = results_win["classical_windowed"]
    print(f"✅ {sdpa_full_result}")
    print(f"✅ {sdpa_sliding_result}")
    print(f"✅ {classical_result}")
    print(f"✅ {classical_windowed_result}")
    print_results(classical_result, classical_windowed_result)
    print_results(sdpa_full_result, sdpa_sliding_result)
    print_results(sdpa_full_result, classical_result)
    print_results(sdpa_sliding_result, classical_windowed_result)


def run_test4(
    results_full: dict[str, dict],
    results_win: dict[str, dict],
    window_size: int,
) -> None:
    """Run test 4.

    Args:
        results_win: The results of the sliding window benchmark.

    """
    print(
        f"\n🔬 Test 4: Fast Sliding Window (TRAINING mode, O(n), window={window_size})..."
    )
    fast_sliding_result = results_win["fast_sliding"]
    classical_windowed_result = results_win["classical_windowed"]
    classical_result = results_full["classical"]
    sdpa_full_result = results_full["sdpa_full"]
    print(f"✅ {sdpa_full_result}")
    print(f"✅ {fast_sliding_result}")
    print(f"✅ {classical_result}")
    print(f"✅ {classical_windowed_result}")
    print_results(sdpa_full_result, fast_sliding_result)
    print_results(fast_sliding_result, classical_windowed_result)


def run_test5(
    results_full: dict[str, dict],
    results_win: dict[str, dict],
    window_size: int,
) -> None:
    """Run test 6.

    Args:
        results_win: The results of the sliding window benchmark.
    """
    print(
        f"\n🔬 Test 5: True Vectorized Sliding Window (TRAINING mode, O(n×w), window={window_size})..."
    )
    true_vectorized_result = results_win["true_vectorized"]
    classical_result = results_full["classical"]
    sdpa_full_result = results_full["sdpa_full"]
    fast_sliding_result = results_win["fast_sliding"]
    print(f"✅ {true_vectorized_result}")
    print(f"✅ {classical_result}")
    print(f"✅ {sdpa_full_result}")
    print(f"✅ {fast_sliding_result}")
    print_results(classical_result, true_vectorized_result)
    print_results(sdpa_full_result, true_vectorized_result)
    print_results(fast_sliding_result, true_vectorized_result)


def run_test6(
    results_full: dict[str, dict],
    results_win: dict[str, dict],
    window_size: int,
) -> None:
    """Run test 6.

    Args:
        results_win: The results of the sliding window benchmark.
    """
    print(
        f"\n🔬 Test 6: True Vectorized Sliding Window Outer Loop (TRAINING mode, O(n×w), window={window_size})..."
    )
    true_vectorized_result = results_win["true_vectorized_outer_loop"]
    classical_result = results_full["classical"]
    sdpa_full_result = results_full["sdpa_full"]
    fast_sliding_result = results_win["fast_sliding"]
    print(f"✅ {true_vectorized_result}")
    print(f"✅ {classical_result}")
    print(f"✅ {sdpa_full_result}")
    print(f"✅ {fast_sliding_result}")
    print_results(classical_result, true_vectorized_result)
    print_results(sdpa_full_result, true_vectorized_result)
    print_results(fast_sliding_result, true_vectorized_result)


def compare_training_mode_attention() -> list[dict[str, dict]]:
    """Compare attention implementations in TRAINING mode only.

    Tests conducted (ALL IN TRAINING MODE with PRECOMPUTED MASKS):
    1. Classical Full Attention (O(n²)) - TransformerEncoderLayer.train()
    2. SDPA Full Attention (O(n²)) - scaled_dot_product_attention with no mask
    3. SDPA Sliding Window (O(n)) - scaled_dot_product_attention with precomputed
        sliding window mask

    """
    print("🚀 TRAINING MODE Attention Comparison (PRECOMPUTED MASKS)")
    print("=" * 80)
    print("ALL TESTS RUN IN TRAINING MODE (with backward pass)")
    print("Tests:")
    print("1. Classical Full Attention (O(n²)) - TransformerEncoderLayer.train()")
    print("2. SDPA Full Attention (O(n²)) - F.scaled_dot_product_attention (no mask)")
    print(
        "3. SDPA Sliding Window (O(n)) - F.scaled_dot_product_attention (precomputed mask)"
    )
    print("=" * 80)

    # Test configurations - single head, d_model=1024 as requested

    seq_lens = [1024, 2048, 4096]
    window_sizes = [32, 128, 512]  # Reasonable window sizes
    d_model = 512  # or as set earlier
    nhead = 1
    batch_size = 1

    all_results = []

    for seq_len in seq_lens:
        # Clean up memory
        torch.cuda.empty_cache()

        results_full = run_benchmark_tests(
            seq_len=seq_len,
            d_model=d_model,
            run_full_attention=True,
            run_sliding_window=False,
        )
        all_results.append(results_full)
        print(f"\n{'='*60}")
        print(f"Testing seq_len={seq_len}, d_model={d_model}, nhead={nhead}")
        print(f"Fixed batch_size={batch_size} - TRAINING MODE")
        print(f"{'='*60}")

        # Test 1: Classical Full Attention (O(n²))
        run_test1(results_full)

        # Test 2: SDPA Full Attention (O(n²))
        run_test2(results_full)

        for window_size in window_sizes:
            # Clean up memory
            torch.cuda.empty_cache()

            results_win = run_benchmark_tests(
                seq_len=seq_len,
                d_model=d_model,
                nhead=nhead,
                batch_size=batch_size,
                window_size=window_size,
                run_full_attention=False,
                run_sliding_window=True,
            )
            all_results.append(results_win)

            # Test 1a: Classical Windowed Full Attention (O(n²))
            run_test1a(results_full, results_win, window_size)

            # Test 3: SDPA Sliding Window (O(n))
            run_test3(results_full, results_win, window_size)

            # Test 4: Fast Sliding Window (O(n))
            run_test4(results_full, results_win, window_size)

            # Test 5: True Vectorized Sliding Window (O(n×w))
            # chunk_size = 32  # Default chunk size
            run_test5(results_full, results_win, window_size)

            # Test 6: True Vectorized Sliding Window Outer Loop (O(n×w))
            run_test6(results_full, results_win, window_size)

        # Clean up memory
        torch.cuda.empty_cache()

    print("\n🏁 TRAINING MODE Comparison complete!")

    return all_results


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    """Main execution with error handling."""
    try:
        all_results = compare_training_mode_attention()
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback

        traceback.print_exc()

    # save all results to a pandas dataframe file
    import pandas as pd

    # Process all_results to create the desired format
    records = []

    # Loop over each experiment result in all_results
    for experiment_results in all_results:
        # Each experiment_results is a dict with keys like 'classical', 'sdpa_full', etc.
        for attention_type, result_dict in experiment_results.items():
            if result_dict is not None:  # Skip None results
                # Add the attention_type to the result dict for clarity
                result_dict["attention_type"] = attention_type
                records.append(result_dict)

    # Create DataFrame from the records
    pivot_df = pd.DataFrame(records)

    # Reorder columns to put attention_type first
    cols = list(pivot_df.columns)
    if "attention_type" in cols:
        cols.insert(0, cols.pop(cols.index("attention_type")))
    pivot_df = pivot_df[cols]

    # Save the pivoted table
    pivot_df.to_csv("pivoted_attention_results.csv", index=False)
    print("✅ Saved pivoted results to pivoted_attention_results.csv")
    print(pivot_df.head())

    # Also save the original format for reference
    df = pd.DataFrame(all_results)
    df.to_csv("training_mode_attention_results.csv", index=False)
    print("✅ Saved original results to training_mode_attention_results.csv")
