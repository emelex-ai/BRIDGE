import gc
import time

import torch

# Will only work if the benchmarks are in the same directory as this file
#   python -m bridge.domain.model.compare_SDPA_classical_attention_refactored
from .benchmarks import (
    benchmark_chunked_vectorized_sliding_window,
    benchmark_classical_full_attention,
    benchmark_classical_windowed_full_attention,
    benchmark_fast_sliding_window,
    benchmark_sdpa_full_attention,
    benchmark_sdpa_sliding_window,
    benchmark_true_vectorized_sliding_window,
    benchmark_true_vectorized_sliding_window_outer_loop,
)


def run_benchmark_tests(
    *,
    seq_len=128,
    d_model=128,
    nhead=1,
    batch_size=1,
    window_size=32,
    run_full_attention=False,
    run_sliding_window=True,
) -> dict[str, dict]:
    results = {}

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


def compare_training_mode_attention():
    """Compare attention implementations in TRAINING mode only.

    Tests conducted (ALL IN TRAINING MODE with PRECOMPUTED MASKS):
    1. Classical Full Attention (O(n²)) - TransformerEncoderLayer.train()
    2. SDPA Full Attention (O(n²)) - scaled_dot_product_attention with no mask
    3. SDPA Sliding Window (O(n)) - scaled_dot_product_attention with precomputed sliding window mask

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

    for seq_len in seq_lens:
        results_full = run_benchmark_tests(
            seq_len=seq_len,
            d_model=d_model,
            run_full_attention=True,
            run_sliding_window=False,
        )
        print(f"\n{'='*60}")
        print(f"Testing seq_len={seq_len}, d_model={d_model}, nhead={nhead}")
        print(f"Fixed batch_size={batch_size} - TRAINING MODE")
        print(f"{'='*60}")

        # Test 1: Classical Full Attention (O(n²))
        try:
            print("\n🔬 Test 1: Classical Full Attention (TRAINING mode, O(n²))...")
            classical_result = results_full["classical"]
            print(f"✅ {classical_result}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Classical Full Attention: OOM at seq_len={seq_len}")
                continue
            else:
                print(f"❌ Classical Full Attention error: {e}")
                continue

        # Test 2: SDPA Full Attention (O(n²))
        try:
            print("\n🔬 Test 2: SDPA Full Attention (TRAINING mode, O(n²))...")
            sdpa_full_result = results_full["sdpa_full"]
            print(f"✅ {sdpa_full_result}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ SDPA Full Attention: OOM at seq_len={seq_len}")
                sdpa_full_result = None
            else:
                print(f"❌ SDPA Full Attention error: {e}")
                sdpa_full_result = None

        for window_size in window_sizes:
            results_win = run_benchmark_tests(
                seq_len=seq_len,
                d_model=d_model,
                nhead=nhead,
                batch_size=batch_size,
                window_size=window_size,
                run_full_attention=False,
                run_sliding_window=True,
            )
            # Test 1a: Classical Windowed Full Attention (O(n²))
            try:
                print(
                    f"\n🔬 Test 1a: Classical Windowed Full Attention (TRAINING mode, O(n²), window={window_size})..."
                )
                classical_windowed_result = results_win["classical_windowed"]
                print(f"✅ {classical_windowed_result}")
                # Comparison with Classical Full Attention
                memory_ratio = (
                    classical_windowed_result["memory_mb"]
                    / classical_result["memory_mb"]
                )
                time_ratio = (
                    classical_windowed_result["time_ms"] / classical_result["time_ms"]
                )
                speedup = (
                    classical_result["time_ms"] / classical_windowed_result["time_ms"]
                )
                print(f"  📊 Classical Windowed vs Classical Full:")
                print(f"    Windowed memory / Full memory: {memory_ratio:.2f}x")
                print(f"    Windowed time / Full time: {time_ratio:.2f}x")
                print(f"    Full time / Windowed time: {speedup:.2f}x")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        f"❌ Classical Windowed Full Attention: OOM at seq_len={seq_len}, window={window_size}"
                    )
                    continue
                else:
                    print(f"❌ Classical Windowed Full Attention error: {e}")
                    continue

            # Test 3: SDPA Sliding Window (O(n))
            print(
                f"\n🔬 Test 3: SDPA Sliding Window (TRAINING mode, O(n), window={window_size})..."
            )
            sdpa_sliding_result = results_win["sdpa_sliding"]
            if sdpa_sliding_result:
                print(f"✅ {sdpa_sliding_result}")
                print(f"  📊 SDPA Sliding vs Classical Full:")
                print(
                    f"    SDPA Sliding memory / Classical memory: {sdpa_sliding_result['memory_mb'] / classical_result['memory_mb']:.2f}x"
                )
                print(
                    f"    SDPA Sliding time / Classical time: {sdpa_sliding_result['time_ms'] / classical_result['time_ms']:.2f}x"
                )
                print(
                    f"    Classical time / SDPA Sliding time: {classical_result['time_ms'] / sdpa_sliding_result['time_ms']:.2f}x"
                )
                print(f"  📊 SDPA Sliding vs SDPA Full:")
                print(
                    f"    SDPA Sliding memory / SDPA Full memory: {sdpa_sliding_result['memory_mb'] / sdpa_full_result['memory_mb']:.2f}x"
                )
                print(
                    f"    SDPA Sliding time / SDPA Full time: {sdpa_sliding_result['time_ms'] / sdpa_full_result['time_ms']:.2f}x"
                )
                print(
                    f"    SDPA Full time / SDPA Sliding time: {sdpa_full_result['time_ms'] / sdpa_sliding_result['time_ms']:.2f}x"
                )
                # New: Compare SDPA Sliding to Classical Windowed Full Attention
                print(f"  📊 SDPA Sliding vs Classical Windowed Full Attention:")
                print(
                    f"    SDPA Sliding memory / Classical Windowed memory: {sdpa_sliding_result['memory_mb'] / classical_windowed_result['memory_mb']:.2f}x"
                )
                print(
                    f"    SDPA Sliding time / Classical Windowed time: {sdpa_sliding_result['time_ms'] / classical_windowed_result['time_ms']:.2f}x"
                )
                print(
                    f"    Classical Windowed time / SDPA Sliding time: {classical_windowed_result['time_ms'] / sdpa_sliding_result['time_ms']:.2f}x"
                )
            else:
                print(f"❌ SDPA Sliding Window (w={window_size}): Failed")

            # Test 4: Fast Sliding Window (O(n))
            print(
                f"\n🔬 Test 4: Fast Sliding Window (TRAINING mode, O(n), window={window_size})..."
            )
            fast_sliding_result = results_win["fast_sliding"]
            if fast_sliding_result:
                print(f"✅ {fast_sliding_result}")
                print(f"  📊 Fast Sliding vs Classical Full:")
                print(
                    f"    Fast Sliding memory / Classical memory: {fast_sliding_result['memory_mb'] / classical_result['memory_mb']:.2f}x"
                )
                print(
                    f"    Fast Sliding time / Classical time: {fast_sliding_result['time_ms'] / classical_result['time_ms']:.2f}x"
                )
                print(
                    f"    Classical time / Fast Sliding time: {classical_result['time_ms'] / fast_sliding_result['time_ms']:.2f}x"
                )
                if sdpa_full_result:
                    print(f"  📊 Fast Sliding vs SDPA Full:")
                    print(
                        f"    Fast Sliding memory / SDPA Full memory: {fast_sliding_result['memory_mb'] / sdpa_full_result['memory_mb']:.2f}x"
                    )
                    print(
                        f"    Fast Sliding time / SDPA Full time: {fast_sliding_result['time_ms'] / sdpa_full_result['time_ms']:.2f}x"
                    )
                    print(
                        f"    SDPA Full time / Fast Sliding time: {sdpa_full_result['time_ms'] / fast_sliding_result['time_ms']:.2f}x"
                    )
            else:
                print(f"❌ Fast Sliding Window (w={window_size}): Failed")

            # Test 5: True Vectorized Sliding Window (O(n×w))
            # chunk_size = 32  # Default chunk size
            print(
                f"\n🔬 Test 5: True Vectorized Sliding Window (TRAINING mode, O(n×w), window={window_size})..."
            )
            true_vectorized_result = results_win["true_vectorized"]
            if true_vectorized_result:
                print(f"✅ {true_vectorized_result}")
                print(f"  📊 True Vectorized vs Classical Full:")
                print(
                    f"    True Vectorized memory / Classical memory: {true_vectorized_result['memory_mb'] / classical_result['memory_mb']:.2f}x"
                )
                print(
                    f"    True Vectorized time / Classical time: {true_vectorized_result['time_ms'] / classical_result['time_ms']:.2f}x"
                )
                print(
                    f"    Classical time / True Vectorized time: {classical_result['time_ms'] / true_vectorized_result['time_ms']:.2f}x"
                )
                if sdpa_full_result:
                    print(f"  📊 True Vectorized vs SDPA Full:")
                    print(
                        f"    True Vectorized memory / SDPA Full memory: {true_vectorized_result['memory_mb'] / sdpa_full_result['memory_mb']:.2f}x"
                    )
                    print(
                        f"    True Vectorized time / SDPA Full time: {true_vectorized_result['time_ms'] / sdpa_full_result['time_ms']:.2f}x"
                    )
                    print(
                        f"    SDPA Full time / True Vectorized time: {sdpa_full_result['time_ms'] / true_vectorized_result['time_ms']:.2f}x"
                    )
                if fast_sliding_result:
                    print(f"  📊 True Vectorized vs Fast Sliding:")
                    print(
                        f"    True Vectorized memory / Fast Sliding memory: {true_vectorized_result['memory_mb'] / fast_sliding_result['memory_mb']:.2f}x"
                    )
                    print(
                        f"    True Vectorized time / Fast Sliding time: {true_vectorized_result['time_ms'] / fast_sliding_result['time_ms']:.2f}x"
                    )
                    print(
                        f"    Fast Sliding time / True Vectorized time: {fast_sliding_result['time_ms'] / true_vectorized_result['time_ms']:.2f}x"
                    )
            else:
                print(f"❌ True Vectorized Sliding Window (w={window_size}): Failed")

            # Test 6: True Vectorized Sliding Window Outer Loop (O(n×w))
            # run_test_6(seq_len, d_model=d_model, nhead=nhead, window_size=window_size, batch_size=batch_size,)
            print(
                f"\n🔬 Test 6: True Vectorized Sliding Window Outer Loop (TRAINING mode, O(n×w), window={window_size})..."
            )
            true_vectorized_result = results_win["true_vectorized_outer_loop"]
            if true_vectorized_result:
                print(f"✅ {true_vectorized_result}")
                print(f"  📊 True Vectorized vs Classical Full:")
                print(
                    f"    True Vectorized memory / Classical memory: {true_vectorized_result['memory_mb'] / classical_result['memory_mb']:.2f}x"
                )
                print(
                    f"    True Vectorized time / Classical time: {true_vectorized_result['time_ms'] / classical_result['time_ms']:.2f}x"
                )
                print(
                    f"    Classical time / True Vectorized time: {classical_result['time_ms'] / true_vectorized_result['time_ms']:.2f}x"
                )
                if sdpa_full_result:
                    print(f"  📊 True Vectorized vs SDPA Full:")
                    print(
                        f"    True Vectorized memory / SDPA Full memory: {true_vectorized_result['memory_mb'] / sdpa_full_result['memory_mb']:.2f}x"
                    )
                    print(
                        f"    True Vectorized time / SDPA Full time: {true_vectorized_result['time_ms'] / sdpa_full_result['time_ms']:.2f}x"
                    )
                    print(
                        f"    SDPA Full time / True Vectorized time: {sdpa_full_result['time_ms'] / true_vectorized_result['time_ms']:.2f}x"
                    )
                if fast_sliding_result:
                    print(f"  📊 True Vectorized vs Fast Sliding:")
                    print(
                        f"    True Vectorized memory / Fast Sliding memory: {true_vectorized_result['memory_mb'] / fast_sliding_result['memory_mb']:.2f}x"
                    )
                    print(
                        f"    True Vectorized time / Fast Sliding time: {true_vectorized_result['time_ms'] / fast_sliding_result['time_ms']:.2f}x"
                    )
                    print(
                        f"    Fast Sliding time / True Vectorized time: {fast_sliding_result['time_ms'] / true_vectorized_result['time_ms']:.2f}x"
                    )
            else:
                print(f"❌ True Vectorized Sliding Window (w={window_size}): Failed")

        # Clean up memory
        torch.cuda.empty_cache()

    print("\n🏁 TRAINING MODE Comparison complete!")


if __name__ == "__main__":
    """Main execution with error handling."""
    try:
        compare_training_mode_attention()
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback

        traceback.print_exc()
