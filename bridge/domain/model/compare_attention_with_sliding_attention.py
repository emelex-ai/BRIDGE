import gc
import time

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def measure_attention_only_timing(model, x, num_iterations=50, warmup_iterations=20):
    """Measure only attention computation time, excluding model overhead."""
    device = x.device

    # Extended warmup for kernel compilation
    print(f"  Warming up for {warmup_iterations} iterations...")
    for i in range(warmup_iterations):
        with torch.no_grad():
            _ = model(x)
        if i == 0:
            print(f"    First iteration complete (kernel compilation)")

    print(f"  Running {num_iterations} timed iterations...")

    # Clear cache and measure baseline memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()

    # Benchmark with more iterations for accuracy
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(x)

    torch.cuda.synchronize()
    end_time = time.time()

    # Measure peak working memory (excluding baseline)
    peak_memory = torch.cuda.max_memory_allocated()
    working_memory = (peak_memory - baseline_memory) / 1024 / 1024  # MB

    avg_time_ms = (end_time - start_time) / num_iterations * 1000
    tokens_per_sec = (x.shape[0] * x.shape[1] * num_iterations) / (
        end_time - start_time
    )

    return {
        "time_ms": avg_time_ms,
        "working_memory_mb": working_memory,
        "peak_memory_mb": peak_memory / 1024 / 1024,
        "tokens_per_sec": tokens_per_sec,
    }


def benchmark_full_attention(seq_len, d_model, nhead, num_layers=1, batch_size=4):
    """Benchmark standard TransformerEncoder with proper timing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  📊 Standard Full Attention")

    # Create model ONCE and reuse
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
    )
    model = TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

    x = torch.randn(batch_size, seq_len, d_model, device=device)

    results = measure_attention_only_timing(model, x)
    results.update(
        {
            "batch_size": batch_size,
            "attention_type": "full_standard",
        }
    )

    return results


def benchmark_flex_full_attention(seq_len, d_model, nhead, num_layers=1, batch_size=4):
    """Benchmark FlexAttention configured as full attention with proper timing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  📊 FlexAttention (Full Mode)")

    try:
        from bridge.domain.model.transformer_flex_attention import (
            FlexAttentionEncoderLayer,
        )

        # Create model ONCE with large window (effectively full attention)
        encoder_layer = FlexAttentionEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            window_size=seq_len * 2,  # Much larger than sequence length
            causal=False,
        )

        class FlexFullAttentionEncoder(nn.Module):
            def __init__(self, layer, num_layers):
                super().__init__()
                self.layers = nn.ModuleList([layer for _ in range(num_layers)])

            def forward(self, x, mask=None):
                for layer in self.layers:
                    x = layer(x, src_mask=mask)
                return x

        model = FlexFullAttentionEncoder(encoder_layer, num_layers).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        results = measure_attention_only_timing(model, x)
        results.update(
            {
                "batch_size": batch_size,
                "window_size": seq_len * 2,
                "attention_type": "flex_full",
            }
        )

        return results

    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


def benchmark_flex_sliding_window(
    seq_len, d_model, nhead, window_size, num_layers=1, batch_size=4
):
    """Benchmark FlexAttention with sliding window and proper timing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  📊 FlexAttention (Window={window_size})")

    try:
        from bridge.domain.model.transformer_flex_attention import (
            FlexAttentionEncoderLayer,
        )

        # Create model ONCE and reuse
        encoder_layer = FlexAttentionEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            window_size=window_size,
            causal=False,
        )

        class FlexSlidingWindowEncoder(nn.Module):
            def __init__(self, layer, num_layers):
                super().__init__()
                self.layers = nn.ModuleList([layer for _ in range(num_layers)])

            def forward(self, x, mask=None):
                for layer in self.layers:
                    x = layer(x, src_mask=mask)
                return x

        model = FlexSlidingWindowEncoder(encoder_layer, num_layers).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        results = measure_attention_only_timing(model, x)
        results.update(
            {
                "batch_size": batch_size,
                "window_size": window_size,
                "attention_type": "flex_sliding",
            }
        )

        return results

    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


def analyze_theoretical_vs_actual(seq_len, window_size, full_time, sliding_time):
    """Analyze theoretical vs actual performance gains."""
    theoretical_ratio = (seq_len * seq_len) / (seq_len * window_size)
    actual_ratio = full_time / sliding_time
    efficiency = (actual_ratio / theoretical_ratio) * 100

    print(f"    🧮 Theoretical Analysis:")
    print(f"       O(L²) vs O(L×W): {seq_len}² vs {seq_len}×{window_size}")
    print(f"       Expected speedup: {theoretical_ratio:.1f}x")
    print(f"       Actual speedup: {actual_ratio:.1f}x")
    print(f"       Efficiency: {efficiency:.1f}% of theoretical")

    return {
        "theoretical_speedup": theoretical_ratio,
        "actual_speedup": actual_ratio,
        "efficiency_percent": efficiency,
    }


def compare_flex_attention_only():
    """Focus on FlexAttention Full vs Sliding Window comparison."""

    # Optimal parameters to maximize theoretical differences
    seq_len = 16384  # Large sequence
    window_size = 32  # Small window
    d_model = 1024  # Large embedding (gains should be independent)
    nhead = 1  # Single head to eliminate multi-head complexity
    batch_size = 2  # Smaller batch to fit in memory
    num_layers = 1  # Single layer to isolate attention performance

    print(f"\n{'='*80}")
    print(f"🔍 FlexAttention Focus Test")
    print(
        f"   seq_len={seq_len}, window={window_size}, d_model={d_model}, nhead={nhead}"
    )
    print(f"   batch_size={batch_size}, num_layers={num_layers}")
    print(f"   Theoretical speedup: {(seq_len*seq_len)/(seq_len*window_size):.0f}x")
    print(f"{'='*80}")

    results = {}

    # Skip Case 1: Standard Full Attention (keep for reference but don't run)
    print(f"\n1️⃣ Standard Full Attention (SKIPPED)")
    print(f"    ⏭️  Skipping to focus on FlexAttention comparison")

    # Case 2: FlexAttention Full Mode (baseline for FlexAttention)
    print(f"\n2️⃣ FlexAttention (Full Mode) - Baseline")
    try:
        torch.cuda.empty_cache()
        flex_full_result = benchmark_flex_full_attention(
            seq_len, d_model, nhead, num_layers, batch_size
        )

        if flex_full_result:
            results["flex_full"] = flex_full_result
            print(f"    ✓ Time: {flex_full_result['time_ms']:.2f}ms")
            print(
                f"    ✓ Working Memory: {flex_full_result['working_memory_mb']:.2f}MB"
            )
            print(f"    ✓ Tokens/sec: {flex_full_result['tokens_per_sec']:.0f}")
        else:
            print(f"    ❌ FlexAttention full mode failed")
            return

    except Exception as e:
        print(f"    ❌ Error: {e}")
        return

    # Case 3: FlexAttention Sliding Window (the test case)
    print(f"\n3️⃣ FlexAttention (Sliding Window={window_size}) - Test Case")
    try:
        torch.cuda.empty_cache()
        flex_sliding_result = benchmark_flex_sliding_window(
            seq_len, d_model, nhead, window_size, num_layers, batch_size
        )

        if flex_sliding_result:
            print(f"    ✓ Time: {flex_sliding_result['time_ms']:.2f}ms")
            print(
                f"    ✓ Working Memory: {flex_sliding_result['working_memory_mb']:.2f}MB"
            )
            print(f"    ✓ Tokens/sec: {flex_sliding_result['tokens_per_sec']:.0f}")

            # Compare FlexAttention Full vs Sliding
            if "flex_full" in results:
                flex_full = results["flex_full"]
                time_ratio = flex_sliding_result["time_ms"] / flex_full["time_ms"]
                memory_ratio = (
                    flex_sliding_result["working_memory_mb"]
                    / flex_full["working_memory_mb"]
                )
                speedup = 1.0 / time_ratio
                memory_reduction = (1.0 - memory_ratio) * 100

                print(f"\n    📊 FlexAttention Sliding vs Full Comparison:")
                print(f"       Speedup: {speedup:.1f}x faster")
                print(
                    f"       Memory: {memory_ratio:.2f}x ({memory_reduction:+.1f}% reduction)"
                )

                # Theoretical analysis
                analysis = analyze_theoretical_vs_actual(
                    seq_len,
                    window_size,
                    flex_full["time_ms"],
                    flex_sliding_result["time_ms"],
                )

                # Interpretation
                if analysis["efficiency_percent"] > 50:
                    print(f"    ✅ FlexAttention achieving good efficiency!")
                elif analysis["efficiency_percent"] > 20:
                    print(
                        f"    ⚠️  FlexAttention showing some benefits but with overhead"
                    )
                else:
                    print(f"    ❌ FlexAttention efficiency is very low")

                # Memory analysis
                if memory_ratio < 0.5:
                    print(f"    ✅ Significant memory reduction achieved!")
                elif memory_ratio < 0.8:
                    print(f"    ⚠️  Moderate memory reduction")
                else:
                    print(f"    ❌ Minimal memory benefits")
        else:
            print(f"    ❌ FlexAttention sliding window failed")

    except Exception as e:
        print(f"    ❌ Error: {e}")

    # Summary
    print(f"\n{'='*80}")
    print(f"🎯 FlexAttention Analysis Summary")
    print(f"{'='*80}")

    if "flex_full" in results and flex_sliding_result:
        flex_full = results["flex_full"]
        speedup = flex_full["time_ms"] / flex_sliding_result["time_ms"]
        theoretical = (seq_len * seq_len) / (seq_len * window_size)
        efficiency = (speedup / theoretical) * 100

        print(f"Sequence Length: {seq_len}")
        print(f"Window Size: {window_size}")
        print(f"Theoretical Maximum Speedup: {theoretical:.0f}x")
        print(f"Actual FlexAttention Speedup: {speedup:.1f}x")
        print(f"Implementation Efficiency: {efficiency:.1f}%")

        if efficiency > 50:
            print(f"\n✅ CONCLUSION: FlexAttention is working well!")
            print(f"   Recommendation: Use FlexAttention for sliding window attention")
        elif efficiency > 20:
            print(f"\n⚠️  CONCLUSION: FlexAttention has significant overhead")
            print(f"   Recommendation: Consider alternatives or wait for optimizations")
        else:
            print(f"\n❌ CONCLUSION: FlexAttention is not performing as expected")
            print(f"   Recommendation: Use alternative implementations")
    else:
        print(f"❌ Could not complete comparison - check FlexAttention availability")


def compare_attention_implementations():
    """Legacy function - redirects to focused comparison."""
    print("🔄 Redirecting to focused FlexAttention comparison...")
    compare_flex_attention_only()


if __name__ == "__main__":
    print("🔍 FlexAttention Focused Benchmark")
    print("=" * 80)
    print("Focus: FlexAttention Full vs Sliding Window")
    print("Goal: Measure pure FlexAttention sliding window benefits")
    print("Parameters optimized for maximum theoretical difference:")
    print("  • seq_len=16384 (large sequence)")
    print("  • window=32 (small window)")
    print("  • nhead=1 (single head)")
    print("  • d_model=1024 (large embedding)")
    print("  • Expected theoretical speedup: 512x")
    print("=" * 80)

    compare_flex_attention_only()

    print("\n🎯 Key Insights:")
    print("  • Efficiency % shows FlexAttention implementation quality")
    print("  • >50% efficiency = good implementation")
    print("  • <20% efficiency = significant overhead issues")
    print("  • Working memory should scale with O(L×W) not O(L²)")
    print("=" * 80)
