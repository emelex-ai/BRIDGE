import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Import FlexAttention if available
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    FLEX_AVAILABLE = True
    print("✅ FlexAttention is available")
except ImportError:
    FLEX_AVAILABLE = False
    print("❌ FlexAttention not available")

# Import SDPA backend selection if available
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    SDPA_BACKENDS_AVAILABLE = True
    print("✅ SDPA backend selection is available")
except ImportError:
    SDPA_BACKENDS_AVAILABLE = False
    print("❌ SDPA backend selection not available")


def measure_attention_only_timing(model, x, num_iterations=100, warmup_iterations=50):
    """Measure only attention computation time with proper warmup.

    Args:
        model: The model to benchmark
        x: Input tensor
        num_iterations: Number of timed iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Average time per iteration in milliseconds

    """
    device = x.device

    # Extended warmup for kernel compilation and optimization
    print(f"  Warming up for {warmup_iterations} iterations...")
    for i in range(warmup_iterations):
        try:
            with torch.no_grad():
                _ = model(x)
        except Exception as e:
            print(f"    ❌ Error during warmup iteration {i}: {e}")
            raise
        if i == 0:
            print(f"    First iteration complete (kernel compilation)")
        if i == 10:
            print(f"    10 iterations complete (kernel optimization)")

    # Clear cache and synchronize
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print(f"  Running {num_iterations} timed iterations...")
    start_time = time.time()

    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(x)

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = (end_time - start_time) * 1000  # Convert to ms
    avg_time = total_time / num_iterations

    return avg_time


def measure_memory_usage(model, x):
    """Measure peak memory usage during forward pass.

    Args:
        model: The model to measure
        x: Input tensor

    Returns:
        Peak memory usage in MB

    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = model(x)

    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # Convert to MB
    return peak_memory


def create_flex_attention_model(
    seq_len, d_model, nhead, window_size, device, use_causal=True
):
    """Create FlexAttention model with pre-computed block mask.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        nhead: Number of heads (unused for now)
        window_size: Sliding window size
        device: Device to use
        use_causal: Whether to use causal masking

    Returns:
        Tuple of (model, mask_creation_time)

    """
    if not FLEX_AVAILABLE:
        return None, None

    # Define mask functions
    def sliding_window_mask(b, h, q_idx, kv_idx):
        window_condition = q_idx - kv_idx <= window_size
        if use_causal:
            causal_condition = q_idx >= kv_idx
            return window_condition & causal_condition
        return window_condition

    # Pre-compute block mask (this is the expensive part)
    print("  Creating BlockMask (expensive operation)...")
    start_time = time.time()
    try:
        block_mask = create_block_mask(
            sliding_window_mask,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )
        mask_time = (time.time() - start_time) * 1000
        print(f"  BlockMask creation took {mask_time:.2f}ms")
    except Exception as e:
        print(f"  ❌ BlockMask creation failed: {e}")
        return None, None

    # Create model that reuses the block mask
    class FlexAttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block_mask = block_mask
            # FIX: Try compilation with fallback to eager mode
            self.use_compiled = False
            self.flex_attention = flex_attention

            # Try to compile with safer settings
            try:
                # Disable dynamic shapes and use default mode
                self.compiled_flex = torch.compile(
                    flex_attention,
                    dynamic=False,  # Disable dynamic shapes
                    mode="default",  # Use default instead of max-autotune
                )
                # Test compilation with small tensor
                test_tensor = torch.randn(
                    1, 1, 64, d_model, device=device, dtype=torch.bfloat16
                )
                with torch.no_grad():
                    _ = self.compiled_flex(
                        test_tensor, test_tensor, test_tensor, block_mask=block_mask
                    )
                self.use_compiled = True
                print("  ✅ FlexAttention compilation successful")
            except Exception as e:
                print(f"  ⚠️  FlexAttention compilation failed: {e}")
                print("  📝 Falling back to eager mode")
                self.use_compiled = False

        def forward(self, x):
            B, S, D = x.shape
            # Simple projection to Q, K, V
            q = k = v = x.view(B, 1, S, D)  # Single head for simplicity

            if self.use_compiled:
                return self.compiled_flex(q, k, v, block_mask=self.block_mask)
            else:
                return self.flex_attention(q, k, v, block_mask=self.block_mask)

    return FlexAttentionModel(), mask_time


def create_standard_attention_model(
    seq_len, d_model, nhead, window_size, device, use_causal=True
):
    """Create standard PyTorch attention model.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        nhead: Number of heads (unused for now)
        window_size: Sliding window size
        device: Device to use
        use_causal: Whether to use causal masking

    Returns:
        Standard attention model

    """

    class StandardAttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Use compiled SDPA for fair comparison
            try:
                self.compiled_sdpa = torch.compile(
                    torch.nn.functional.scaled_dot_product_attention,
                    dynamic=True,
                    mode="max-autotune",
                )
                self.use_compiled = True
            except:
                self.use_compiled = False

        def forward(self, x):
            B, S, D = x.shape
            q = k = v = x.view(B, 1, S, D)  # Single head

            # Create attention mask
            mask = torch.ones(S, S, device=device, dtype=torch.bool)

            # Apply sliding window
            for i in range(S):
                for j in range(S):
                    if abs(i - j) > window_size:
                        mask[i, j] = False
                    if use_causal and i < j:
                        mask[i, j] = False

            # Convert to attention mask format with proper dtype
            attn_mask = torch.where(mask, 0.0, float("-inf")).to(x.dtype)

            if self.use_compiled:
                return self.compiled_sdpa(q, k, v, attn_mask=attn_mask)
            else:
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask
                )

    return StandardAttentionModel()


def create_sdpa_causal_model(seq_len, d_model, nhead, device):
    """Create SDPA model with full causal mask.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        nhead: Number of heads
        device: Device to use

    Returns:
        SDPA causal model

    """

    class SDPACausalModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Use compiled SDPA for optimal performance
            try:
                self.compiled_sdpa = torch.compile(
                    torch.nn.functional.scaled_dot_product_attention,
                    dynamic=True,
                    mode="max-autotune",
                )
                self.use_compiled = True
            except:
                self.use_compiled = False

        def forward(self, x):
            B, S, D = x.shape
            q = k = v = x.view(B, nhead, S, D // nhead)  # Multi-head

            # Use built-in causal masking for maximum efficiency
            if self.use_compiled:
                return self.compiled_sdpa(q, k, v, is_causal=True)
            else:
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )

    return SDPACausalModel()


def create_sdpa_sliding_window_model(seq_len, d_model, nhead, window_size, device):
    """Create SDPA model with sliding window + causal mask.

    Args:
        seq_len: Sequence length
        d_model: Model dimension
        nhead: Number of heads
        window_size: Sliding window size
        device: Device to use

    Returns:
        SDPA sliding window model

    """

    class SDPASlidingWindowModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.window_size = window_size
            try:
                self.compiled_sdpa = torch.compile(
                    torch.nn.functional.scaled_dot_product_attention,
                    dynamic=True,
                    mode="max-autotune",
                )
                self.use_compiled = True
            except:
                self.use_compiled = False

            # Pre-compute sliding window + causal mask
            self.register_buffer(
                "attn_mask", self._create_sliding_window_mask(seq_len, device)
            )

        def _create_sliding_window_mask(self, seq_len, device):
            """Create efficient sliding window + causal mask.

            Args:
                seq_len: Sequence length
                device: Device to use

            Returns:
                Attention mask tensor

            """
            # Start with causal mask (lower triangular)
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
            )

            # Apply sliding window constraint
            for i in range(seq_len):
                # Zero out positions beyond window
                start_pos = max(0, i - self.window_size)
                if start_pos > 0:
                    mask[i, :start_pos] = False

            # Convert to attention mask format (0 for attend, -inf for ignore)
            # FIX: Use float32 initially, will be converted to match input dtype
            return torch.where(mask, 0.0, float("-inf")).to(torch.float32)

        def forward(self, x):
            B, S, D = x.shape
            q = k = v = x.view(B, nhead, S, D // nhead)  # Multi-head

            # Use pre-computed mask for efficiency
            # FIX: Ensure mask dtype matches input tensors
            mask = self.attn_mask[:S, :S]
            if mask.dtype != x.dtype:
                mask = mask.to(x.dtype)

            if self.use_compiled:
                return self.compiled_sdpa(q, k, v, attn_mask=mask)
            else:
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask
                )

    return SDPASlidingWindowModel()


def benchmark_sdpa_comparison():
    """Compare SDPA with full causal vs sliding window masks."""
    print("🔬 SDPA Causal vs Sliding Window Benchmark")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test parameters - focus on realistic scenarios
    test_configs = [
        {"seq_len": 1024, "window_size": 32, "d_model": 1024, "nhead": 16},
        {"seq_len": 2048, "window_size": 64, "d_model": 1024, "nhead": 16},
        {"seq_len": 4096, "window_size": 128, "d_model": 1024, "nhead": 16},
        {"seq_len": 8192, "window_size": 256, "d_model": 1024, "nhead": 16},
        {"seq_len": 16384, "window_size": 512, "d_model": 1024, "nhead": 16},
    ]

    results = []

    for config in test_configs:
        seq_len = config["seq_len"]
        window_size = config["window_size"]
        d_model = config["d_model"]
        nhead = config["nhead"]

        print(f"\n📊 Testing seq_len={seq_len}, window={window_size}, nhead={nhead}")
        print("-" * 50)

        # Create input data
        x = torch.randn(1, seq_len, d_model, device=device, dtype=torch.bfloat16)

        try:
            # Test Full Causal SDPA
            print("🔧 Creating SDPA Causal model...")
            causal_model = create_sdpa_causal_model(seq_len, d_model, nhead, device)

            print("⏱️  Benchmarking SDPA Causal...")
            causal_time = measure_attention_only_timing(causal_model, x)
            causal_memory = measure_memory_usage(causal_model, x)
            print(f"   SDPA Causal: {causal_time:.2f}ms, {causal_memory:.1f}MB")

            # Test Sliding Window SDPA
            print("🔧 Creating SDPA Sliding Window model...")
            sliding_model = create_sdpa_sliding_window_model(
                seq_len, d_model, nhead, window_size, device
            )

            print("⏱️  Benchmarking SDPA Sliding Window...")
            sliding_time = measure_attention_only_timing(sliding_model, x)
            sliding_memory = measure_memory_usage(sliding_model, x)
            print(
                f"   SDPA Sliding Window: {sliding_time:.2f}ms, {sliding_memory:.1f}MB"
            )

            # Calculate improvements
            speedup = causal_time / sliding_time
            memory_reduction = (causal_memory - sliding_memory) / causal_memory * 100

            print(f"\n📈 Results:")
            print(f"   Sliding Window Speedup: {speedup:.2f}x")
            print(f"   Memory Reduction: {memory_reduction:.1f}%")

            # Theoretical analysis
            causal_ops = seq_len * seq_len / 2  # Triangular matrix
            window_ops = seq_len * min(
                window_size, seq_len
            )  # Linear in sequence length
            theoretical_speedup = causal_ops / window_ops
            efficiency = (speedup / theoretical_speedup) * 100

            print(f"   Theoretical Speedup: {theoretical_speedup:.2f}x")
            print(f"   Implementation Efficiency: {efficiency:.1f}%")

            results.append(
                {
                    "seq_len": seq_len,
                    "window_size": window_size,
                    "nhead": nhead,
                    "causal_time": causal_time,
                    "sliding_time": sliding_time,
                    "speedup": speedup,
                    "causal_memory": causal_memory,
                    "sliding_memory": sliding_memory,
                    "memory_reduction": memory_reduction,
                    "theoretical_speedup": theoretical_speedup,
                    "efficiency": efficiency,
                }
            )

        except Exception as e:
            print(f"❌ Error in benchmark: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Cleanup
        del x, causal_model, sliding_model
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("📊 SDPA BENCHMARK SUMMARY")
    print("=" * 60)

    if results:
        print(
            f"{'Seq Len':<8} {'Window':<8} {'Heads':<6} {'Speedup':<10} {'Mem Reduce':<12} {'Efficiency':<12}"
        )
        print("-" * 70)
        for r in results:
            print(
                f"{r['seq_len']:<8} {r['window_size']:<8} {r['nhead']:<6} {r['speedup']:<10.2f} "
                f"{r['memory_reduction']:<12.1f}% {r['efficiency']:<12.1f}%"
            )

        # Find best performance
        best_speedup = max(results, key=lambda x: x["speedup"])
        best_memory = max(results, key=lambda x: x["memory_reduction"])

        print(f"\n🏆 Best Results:")
        print(
            f"   Best Speedup: {best_speedup['speedup']:.2f}x at seq_len={best_speedup['seq_len']}"
        )
        print(
            f"   Best Memory Reduction: {best_memory['memory_reduction']:.1f}% at seq_len={best_memory['seq_len']}"
        )

        # Analysis
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        avg_efficiency = sum(r["efficiency"] for r in results) / len(results)

        print(f"\n🔍 Analysis:")
        print(f"   Average Speedup: {avg_speedup:.2f}x")
        print(f"   Average Efficiency: {avg_efficiency:.1f}%")

        if avg_speedup > 1.0:
            print(f"   ✅ Sliding window shows consistent benefits")
        else:
            print(f"   ⚠️  Sliding window overhead dominates")

        if avg_efficiency > 50:
            print(f"   ✅ Good implementation efficiency")
        else:
            print(f"   ⚠️  Implementation could be more efficient")
    else:
        print("❌ No successful benchmark results")


def benchmark_attention_comparison():
    """Improved benchmark focusing on FlexAttention vs Standard PyTorch."""
    print("🔬 FlexAttention vs Standard Attention Benchmark")
    print("=" * 60)

    if not FLEX_AVAILABLE:
        print("❌ FlexAttention not available, skipping benchmark")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test parameters - focusing on larger scales where benefits should emerge
    test_configs = [
        {"seq_len": 4096, "window_size": 128, "d_model": 1024, "nhead": 1},
        {"seq_len": 8192, "window_size": 256, "d_model": 1024, "nhead": 1},
        {"seq_len": 16384, "window_size": 512, "d_model": 1024, "nhead": 1},
        {
            "seq_len": 32768,
            "window_size": 1024,
            "d_model": 1024,
            "nhead": 1,
        },  # Large scale
    ]

    results = []

    for config in test_configs:
        seq_len = config["seq_len"]
        window_size = config["window_size"]
        d_model = config["d_model"]
        nhead = config["nhead"]

        print(f"\n📊 Testing seq_len={seq_len}, window={window_size}")
        print("-" * 40)

        # Create input data
        x = torch.randn(1, seq_len, d_model, device=device, dtype=torch.bfloat16)

        try:
            # Test FlexAttention
            print("🔧 Creating FlexAttention model...")
            flex_model, mask_creation_time = create_flex_attention_model(
                seq_len, d_model, nhead, window_size, device
            )

            if flex_model is not None:
                print("⏱️  Benchmarking FlexAttention...")
                flex_time = measure_attention_only_timing(flex_model, x)
                flex_memory = measure_memory_usage(flex_model, x)
                print(f"   FlexAttention: {flex_time:.2f}ms, {flex_memory:.1f}MB")
                print(
                    f"   (BlockMask creation: {mask_creation_time:.2f}ms - one-time cost)"
                )
            else:
                flex_time, flex_memory = float("inf"), float("inf")
                mask_creation_time = 0

            # Test Standard Attention
            print("🔧 Creating Standard Attention model...")
            std_model = create_standard_attention_model(
                seq_len, d_model, nhead, window_size, device
            )

            print("⏱️  Benchmarking Standard Attention...")
            std_time = measure_attention_only_timing(std_model, x)
            std_memory = measure_memory_usage(std_model, x)
            print(f"   Standard Attention: {std_time:.2f}ms, {std_memory:.1f}MB")

            # Calculate speedup
            if flex_time != float("inf"):
                speedup = std_time / flex_time
                memory_ratio = std_memory / flex_memory if flex_memory > 0 else 1.0

                print(f"\n📈 Results:")
                print(f"   FlexAttention Speedup: {speedup:.2f}x")
                print(f"   Memory Efficiency: {memory_ratio:.2f}x")

                # Theoretical analysis
                total_ops_full = seq_len * seq_len
                effective_ops_windowed = seq_len * min(
                    window_size * 2, seq_len
                )  # Approximate
                theoretical_speedup = total_ops_full / effective_ops_windowed
                efficiency = (speedup / theoretical_speedup) * 100

                print(f"   Theoretical Speedup: {theoretical_speedup:.2f}x")
                print(f"   Implementation Efficiency: {efficiency:.1f}%")

                results.append(
                    {
                        "seq_len": seq_len,
                        "window_size": window_size,
                        "flex_time": flex_time,
                        "std_time": std_time,
                        "speedup": speedup,
                        "flex_memory": flex_memory,
                        "std_memory": std_memory,
                        "memory_ratio": memory_ratio,
                        "theoretical_speedup": theoretical_speedup,
                        "efficiency": efficiency,
                        "mask_creation_time": mask_creation_time,
                    }
                )

        except Exception as e:
            print(f"❌ Error in benchmark: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Cleanup
        del x
        if "flex_model" in locals():
            del flex_model
        if "std_model" in locals():
            del std_model
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)

    if results:
        print(
            f"{'Seq Len':<8} {'Window':<8} {'Speedup':<10} {'Memory':<10} {'Efficiency':<12}"
        )
        print("-" * 60)
        for r in results:
            print(
                f"{r['seq_len']:<8} {r['window_size']:<8} {r['speedup']:<10.2f} "
                f"{r['memory_ratio']:<10.2f} {r['efficiency']:<12.1f}%"
            )

        # Find best performance
        best_result = max(results, key=lambda x: x["speedup"])
        print(f"\n🏆 Best FlexAttention performance:")
        print(f"   Sequence Length: {best_result['seq_len']}")
        print(f"   Speedup: {best_result['speedup']:.2f}x")
        print(f"   Efficiency: {best_result['efficiency']:.1f}%")

        # Analysis
        print(f"\n🔍 Analysis:")
        if best_result["speedup"] > 1.0:
            print(f"   ✅ FlexAttention shows benefits at scale")
        else:
            print(f"   ⚠️  FlexAttention overhead dominates at these scales")
            print(f"   💡 Try larger sequence lengths (64k+) or more complex patterns")
    else:
        print("❌ No successful benchmark results")


if __name__ == "__main__":
    """Main execution - run both benchmarks with comprehensive error handling.
    
    """
    print("🚀 Starting Attention Benchmarks")
    print("=" * 80)

    try:
        # Run SDPA benchmark
        benchmark_sdpa_comparison()
        print("\n" + "=" * 80 + "\n")

        # Run FlexAttention benchmark
        benchmark_attention_comparison()

    except Exception as e:
        print(f"❌ Critical error in benchmarking: {e}")
        import traceback

        traceback.print_exc()

    print("\n🏁 Benchmarking complete!")
