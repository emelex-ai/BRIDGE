import datetime
import gc
from typing import Any

import torch
import torch.nn as nn

from bridge.domain.model.benchmarks.sdpa_full_attention_model import (
    SDPAFullLayer,
)
from bridge.domain.model.benchmarks.sdpa_sliding_window_model import (
    SDPASlidingWindowLayer,
)


def check_cuda_memory():
    """Check CUDA memory usage and print statistics.

    Returns:
        None

    """
    if torch.cuda.is_available():
        # Current memory allocated by PyTorch
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB

        # Peak memory allocated
        peak = torch.cuda.max_memory_allocated() / 1024**3  # GB

        # Total GPU memory
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

        print(f"Allocated: {allocated:.2f}GB")
        print(f"Peak: {peak:.2f}GB")
        print(f"Total: {total:.2f}GB")
        print(f"Free: {total - allocated:.2f}GB")


class EncoderSDPA(nn.Module):
    """SDPA-based encoder with support for full and sliding window attention.

    This encoder can use different attention mechanisms based on the device
    and attention_type parameter. On CUDA, it supports both full attention
    and sliding window attention using SDPA. On CPU, it falls back to
    standard TransformerEncoderLayer.

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        device: str = "cpu",
        window_size: int = 512,
        causal: bool = True,  # LLM is the default
        # look_backward: int = 1,
        # look_forward: int | None = None,
        attention_type: str = "sdpa_sliding_window",  # Fixed default
    ) -> None:
        """Initialize EncoderSDPA with local attention on CUDA, standard attention on CPU.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            device: Device to use ("cpu" or "cuda")
            window_size: Local attention window size (only used for CUDA)
            causal: Whether to use causal attention (only used for CUDA)
            # look_backward: Number of windows to look backward (only used for CUDA)
            # look_forward: Number of windows to look forward (only used for CUDA)
            attention_type: Type of attention to use:
                ("sdpa_full", "sdpa_sliding_window")

        """
        super(EncoderSDPA, self).__init__()

        base_kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "batch_first": True,
            "dim_feedforward": 4 * d_model,
            "device": device,
        }

        # Use local attention on CUDA, standard attention on CPU
        if device == "cuda":
            local_kwargs = {
                **base_kwargs,
                "window_size": window_size,
                "causal": causal,
            }

            # Choose attention implementation based on attention_type
            if attention_type == "sdpa_full":
                encoder_layer = SDPAFullLayer(**local_kwargs)
                print(f"Using SDPAFullLayer with window_size={window_size}")
            elif attention_type == "sdpa_sliding_window":
                encoder_layer = SDPASlidingWindowLayer(**local_kwargs)
                print(f"Using SDPASlidingWindowLayer with window_size={window_size}")
            else:
                encoder_layer = nn.TransformerEncoderLayer(**base_kwargs)
                print("Using standard TransformerEncoderLayer")
        else:
            # CPU fallback - use standard TransformerEncoderLayer
            encoder_layer = nn.TransformerEncoderLayer(**base_kwargs)
            print("Using standard TransformerEncoderLayer on CPU")

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the encoder.

        Args:
            src: Input tensor
            src_mask: Attention mask
            src_key_padding_mask: Padding mask

        Returns:
            Encoded output tensor

        """
        output = self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        return output


# ----------------------------------------------------------------------
if __name__ == "__main__":
    """Execute EncoderSDPA testing and benchmarking suite.
    
    This module provides comprehensive testing for the EncoderSDPA class,
    including unit tests and performance benchmarks.
    
    Execution Flow:
        1. Run basic functionality tests (test_encoder_sdpa)
        2. If successful, run performance benchmarks (test_autoregressive_scaling)
        3. Save results to timestamped files
        
    Output Files:
        - autoregressive_scaling_results_YYYYMMDD_HHMMSS.csv
        - autoregressive_scaling_results_partial_*.csv (periodic saves)
        
    """

    def test_encoder_sdpa() -> bool:
        """Test basic EncoderSDPA functionality across devices."""
        print("=" * 60)
        print("Testing EncoderSDPA")
        print("=" * 60)

        # Test parameters
        batch_size, seq_len, d_model = 4, 256, 512  # Larger seq_len for local attention
        nhead = 8
        num_layers = 3
        window_size = 64

        print(f"Test configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Model dimension: {d_model}")
        print(f"  Number of heads: {nhead}")
        print(f"  Number of layers: {num_layers}")
        print(f"  Window size: {window_size}")
        print()

        success = True

        # Test 1: CPU device
        print("Test 1: EncoderSDPA with CPU device")
        try:
            cpu_encoder = EncoderSDPA(  # Fixed typo
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                device="cpu",
                window_size=window_size,
                causal=True,
            )

            # Create test input on CPU
            x_cpu = torch.randn(batch_size, seq_len, d_model)

            # Forward pass
            output_cpu = cpu_encoder(x_cpu)

            # Verify output
            assert (
                output_cpu.shape == x_cpu.shape
            ), f"CPU output shape {output_cpu.shape} != input shape {x_cpu.shape}"
            assert (
                output_cpu.device.type == "cpu"
            ), f"CPU output should be on CPU, got {output_cpu.device}"

            print(f"✓ CPU input shape: {x_cpu.shape}, device: {x_cpu.device}")
            print(
                f"✓ CPU output shape: {output_cpu.shape}, device: {output_cpu.device}"
            )

            # Test gradient flow on CPU
            x_cpu_grad = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
            output_cpu_grad = cpu_encoder(x_cpu_grad)
            loss_cpu = output_cpu_grad.sum()
            loss_cpu.backward()

            assert x_cpu_grad.grad is not None, "CPU gradients not computed"
            print("✓ CPU gradient flow successful")
            print()

        except Exception as e:
            print(f"❌ CPU test failed: {str(e)}")
            import traceback

            traceback.print_exc()
            success = False
            print()

        # Test 2: CUDA device (if available)
        print("Test 2: EncoderSDPA with CUDA device")
        if torch.cuda.is_available():
            try:
                cuda_encoder = EncoderSDPA(
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    device="cuda",
                    window_size=window_size,
                )
                cuda_encoder = cuda_encoder.to("cuda")

                # Create test input on CUDA
                x_cuda = torch.randn(batch_size, seq_len, d_model, device="cuda")

                # Forward pass
                output_cuda = cuda_encoder(x_cuda)

                # Verify output
                assert (
                    output_cuda.shape == x_cuda.shape
                ), f"CUDA output shape {output_cuda.shape} != input shape {x_cuda.shape}"
                assert (
                    output_cuda.device.type == "cuda"
                ), f"CUDA output should be on CUDA, got {output_cuda.device}"

                print(f"✓ CUDA input shape: {x_cuda.shape}, device: {x_cuda.device}")
                print(
                    f"✓ CUDA output shape: {output_cuda.shape}, device: {output_cuda.device}"
                )

                # Test gradient flow on CUDA
                x_cuda_grad = torch.randn(
                    batch_size, seq_len, d_model, device="cuda", requires_grad=True
                )
                output_cuda_grad = cuda_encoder(x_cuda_grad)
                loss_cuda = output_cuda_grad.sum()
                loss_cuda.backward()

                assert x_cuda_grad.grad is not None, "CUDA gradients not computed"
                print("✓ CUDA gradient flow successful")

                # Memory usage check
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
                _ = cuda_encoder(x_cuda)
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = (peak_memory - initial_memory) / 1024**2  # MB
                print(f"✓ CUDA memory used: {memory_used:.2f} MB")
                torch.cuda.empty_cache()
                print()

            except Exception as e:
                print(f"❌ CUDA test failed: {str(e)}")
                import traceback

                traceback.print_exc()
                success = False
                print()
        else:
            print("⚠️  CUDA not available, skipping CUDA tests")
            print("   (This is expected when running on CPU-only machines)")
            print()

        # Test 3: Attention masks
        print("Test 3: Attention masks")
        try:
            # Test with key padding mask
            key_padding_mask = torch.zeros(batch_size, seq_len).bool()
            key_padding_mask[:, -10:] = True  # Mask last 10 tokens

            masked_output_cpu = cpu_encoder(
                x_cpu, src_key_padding_mask=key_padding_mask
            )
            assert masked_output_cpu.shape == x_cpu.shape
            print("✓ CPU attention mask test successful")

            # Test with CUDA encoder if available
            if torch.cuda.is_available():
                key_padding_mask_cuda = key_padding_mask.to("cuda")

                masked_output_cuda = cuda_encoder(
                    x_cuda,
                    src_key_padding_mask=key_padding_mask_cuda,
                )
                assert masked_output_cuda.shape == x_cuda.shape
                print("✓ CUDA attention mask test successful")
            else:
                print("⚠️  CUDA device: unavailable, skipping CUDA tests")

            print()

        except Exception as e:
            print(f"❌ Attention mask test failed: {str(e)}")
            import traceback

            traceback.print_exc()
            success = False
            print()

        # Summary
        print("=" * 60)
        if success:
            print("All EncoderSDPA tests PASSED! ✓")
            print()
            print("Summary:")
            print("✓ CPU device: Uses standard TransformerEncoderLayer")
            if torch.cuda.is_available():
                print("✓ CUDA device: Uses LocalAttentionEncoderLayer")
            else:
                print("⚠️  CUDA device: Not tested (CUDA unavailable)")
            print("✓ Shape consistency maintained")
            print("✓ Gradient flow working")
            print("✓ Attention masks supported")
        else:
            print("Some EncoderSDPA tests FAILED! ❌")
        print("=" * 60)

        return success

    # Run the test
    success = test_encoder_sdpa()

    if not success:
        print("Tests failed!")
        exit(1)
    else:
        print("All tests passed successfully!")

    def test_autoregressive_scaling() -> bool:
        """Benchmark EncoderSDPA performance across parameter space."""
        print("=" * 80)
        print("Testing Autoregressive Local Attention Scaling")
        print("=" * 80)

        # Test parameters - full grid search
        batch_size = 2
        # seq_lens = [128, 512, 1024, 2048, 4096]  # 6 sequence lengths
        seq_lens = [1024, 4096]  # 6 sequence lengths
        d_models = [128, 512]  # 4 model dimensions
        nheads = [1, 8]
        batch_sizes = [1, 8]
        window_sizes = [
            # 16,
            32,
            # 64,
            # 128,
            # 256,
            # 512,
            # 1024,
            # 2048,
            # 4096,
            # 8192,
        ]  # 10 window sizes
        num_steps = 3  # Number of timing runs to average

        total_tests = len(seq_lens) * len(d_models) * len(window_sizes) * len(nheads)
        print(
            f"Total tests to run: {total_tests} ({len(seq_lens)} seq_lens × {len(d_models)} d_models × {len(window_sizes)} window_sizes × {len(nheads)} nheads)"
        )
        print(f"Test parameters:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence lengths: {seq_lens}")
        print(f"  Model dimensions: {d_models}")
        print(f"  Number of heads: {nheads}")
        print(f"  Window sizes: {window_sizes}")
        print(f"  Timing steps per test: {num_steps}")
        print()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running tests on: {device}")
        print()

        # Store all results
        all_results = []
        test_count = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(
                f"Starting with {torch.cuda.memory_allocated()/1024**3:.1f}GB GPU memory in use"
            )

        for seq_len in seq_lens:
            for d_model in d_models:
                for nhead in nheads:
                    # MEMORY CLEANUP AT START OF EACH d_model/nhead COMBINATION
                    if device == "cuda":
                        torch.cuda.empty_cache()
                        gc.collect()

                    # Skip invalid combinations where nhead doesn't divide d_model evenly
                    if d_model % nhead != 0:
                        print(
                            f"Skipping invalid combination: d_model={d_model}, nhead={nhead} (not divisible)"
                        )
                        continue

                    # Skip invalid combinations where window_size > seq_len
                    valid_window_sizes = [w for w in window_sizes if w <= seq_len]

                    for window_size in valid_window_sizes:
                        test_count += 1
                        print(
                            f"Test {test_count}/{total_tests}: seq_len={seq_len}, d_model={d_model}, nhead={nhead}, window_size={window_size}"
                        )
                        check_cuda_memory()

                        try:
                            # Create encoder
                            encoder = EncoderSDPA(
                                d_model=d_model,
                                nhead=nhead,  # Now variable
                                num_layers=2,  # Fixed for simplicity
                                device=device,
                                window_size=window_size,
                                causal=True,
                                # look_backward=1,
                                # look_forward=0,
                            )

                            if device == "cuda":
                                encoder = encoder.to("cuda")

                            # Create input tensor
                            x = torch.randn(batch_size, seq_len, d_model, device=device)

                            # Warmup run
                            with torch.no_grad():
                                _ = encoder(x)

                            if device == "cuda":
                                torch.cuda.synchronize()

                            # Time multiple runs
                            import time

                            times = []

                            for step in range(num_steps):
                                if device == "cuda":
                                    torch.cuda.synchronize()

                                start_time = time.time()
                                with torch.no_grad():
                                    output = encoder(x)
                                if device == "cuda":
                                    torch.cuda.synchronize()

                                end_time = time.time()
                                times.append(end_time - start_time)

                            # Calculate timing statistics
                            avg_time = sum(times) / len(times)
                            min_time = min(times)
                            max_time = max(times)
                            std_time = (
                                sum((t - avg_time) ** 2 for t in times) / len(times)
                            ) ** 0.5

                            # Memory measurement
                            if device == "cuda":
                                torch.cuda.empty_cache()
                                initial_memory = torch.cuda.memory_allocated()
                                with torch.no_grad():
                                    _ = encoder(x)
                                peak_memory = torch.cuda.max_memory_allocated()
                                memory_used_mb = (
                                    peak_memory - initial_memory
                                ) / 1024**2
                                torch.cuda.empty_cache()
                            else:
                                memory_used_mb = 0.0

                            # Verify output shape
                            assert (
                                output.shape == x.shape
                            ), f"Shape mismatch: {output.shape} vs {x.shape}"

                            # Calculate derived metrics
                            total_tokens = batch_size * seq_len
                            tokens_per_sec = total_tokens / avg_time

                            # Store result
                            result = {
                                "seq_len": seq_len,
                                "d_model": d_model,
                                "nhead": nhead,
                                "window_size": window_size,
                                "batch_size": batch_size,
                                "device": device,
                                "avg_time_s": avg_time,
                                "min_time_s": min_time,
                                "max_time_s": max_time,
                                "std_time_s": std_time,
                                "memory_mb": memory_used_mb,
                                "tokens_per_sec": tokens_per_sec,
                                "total_tokens": total_tokens,
                                "success": True,
                                "error": None,
                            }

                            all_results.append(result.copy())
                            print("nb results: ", len(all_results))
                            print(
                                f"  ✓ {avg_time:.4f}s, {tokens_per_sec:.0f} tokens/sec, {memory_used_mb:.1f}MB"
                            )

                        except Exception as e:
                            print(f"  ❌ Failed: {str(e)}")

                            # Store failed result
                            result = {
                                "seq_len": seq_len,
                                "d_model": d_model,
                                "nhead": nhead,
                                "window_size": window_size,
                                "batch_size": batch_size,
                                "device": device,
                                "avg_time_s": float("inf"),
                                "min_time_s": float("inf"),
                                "max_time_s": float("inf"),
                                "std_time_s": float("inf"),
                                "memory_mb": 0.0,
                                "tokens_per_sec": 0.0,
                                "total_tokens": batch_size * seq_len,
                                "success": False,
                                "error": str(e),
                            }

                            all_results.append(result.copy())

                            # Print traceback for debugging
                            import traceback

                            traceback.print_exc()

                        finally:
                            # AGGRESSIVE MEMORY CLEANUP AFTER EACH TEST
                            # Delete all local variables
                            locals_to_delete = [
                                "encoder",
                                "x",
                                "output",
                                "times",
                                "result",
                            ]
                            for var_name in locals_to_delete:
                                if var_name in locals():
                                    del locals()[var_name]

                            # Force garbage collection
                            gc.collect()

                            # Clear CUDA cache if using GPU
                            if device == "cuda":
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()

                            # Print memory status every 10 tests
                            if test_count % 10 == 0 and device == "cuda":
                                current_memory = torch.cuda.memory_allocated() / 1024**2
                                max_memory = torch.cuda.max_memory_allocated() / 1024**2
                                print(
                                    f"    Memory: {current_memory:.1f}MB current, {max_memory:.1f}MB peak"
                                )
                                torch.cuda.reset_peak_memory_stats()

                        # PERIODIC SAVE EVERY 50 TESTS
                        if test_count % 50 == 0 and len(all_results) > 0:
                            try:
                                import pandas as pd

                                df_temp = pd.DataFrame(all_results)
                                timestamp = datetime.datetime.now().strftime(
                                    "%Y%m%d_%H%M%S"
                                )
                                temp_filename = f"autoregressive_scaling_results_partial_{timestamp}.csv"
                                df_temp.to_csv(temp_filename, index=False)
                                print(f"    Saved partial results to {temp_filename}")
                            except Exception as save_e:
                                print(
                                    f"    Warning: Could not save partial results: {save_e}"
                                )

        # Save results to CSV using pandas
        try:
            import pandas as pd

            df = pd.DataFrame(all_results)

            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"autoregressive_scaling_results_{timestamp}.csv"

            # Round numeric columns to 5 significant digits before saving
            numeric_columns = [
                "avg_time_s",
                "min_time_s",
                "max_time_s",
                "std_time_s",
                "memory_mb",
                "tokens_per_sec",
            ]

            # Much simpler approach using pandas built-in methods
            for col in numeric_columns:
                if col in df.columns:
                    if col == "tokens_per_sec":
                        # Convert to integer, handling inf values
                        df[col] = df[col].replace([float("inf")], 0).astype(int)
                    else:
                        # Round to 5 significant digits, handling inf values
                        df[col] = df[col].replace([float("inf")], 0).round(5)

            df.to_csv(filename, index=False)

            print(f"\n{'='*80}")
            print("RESULTS SAVED")
            print(f"{'='*80}")
            print(f"Results saved to: {filename}")
            print(f"Total tests: {len(all_results)}")
            print(f"Successful tests: {sum(1 for r in all_results if r['success'])}")
            print(f"Failed tests: {sum(1 for r in all_results if not r['success'])}")

            # Show basic summary
            successful_df = df[df["success"] == True]
            if len(successful_df) > 0:
                print("\nBasic statistics (successful tests only):")
                print(
                    f"  Time range: {successful_df['avg_time_s'].min():.4f}s - {successful_df['avg_time_s'].max():.4f}s"
                )
                print(
                    f"  Tokens/sec range: {successful_df['tokens_per_sec'].min():.0f} - {successful_df['tokens_per_sec'].max():.0f}"
                )
                print(
                    f"  Memory range: {successful_df['memory_mb'].min():.1f}MB - {successful_df['memory_mb'].max():.1f}MB"
                )

            print(f"\nDataFrame columns: {list(df.columns)}")
            print(f"DataFrame shape: {df.shape}")
            print(f"{'='*80}")

            return len([r for r in all_results if r["success"]]) == len(all_results)

        except ImportError:
            print("❌ pandas not available, saving results as JSON instead")
            import json

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"autoregressive_scaling_results_{timestamp}.json"

            # Round numeric values in the results before saving to JSON
            rounded_results = []
            for result in all_results:
                rounded_result = result.copy()
                numeric_keys = [
                    "avg_time_s",
                    "min_time_s",
                    "max_time_s",
                    "std_time_s",
                    "memory_mb",
                    "tokens_per_sec",
                ]
                for key in numeric_keys:
                    if key in rounded_result and isinstance(
                        rounded_result[key], (int, float)
                    ):
                        if rounded_result[key] != float("inf"):
                            if key == "tokens_per_sec":
                                rounded_result[key] = int(rounded_result[key])
                            else:
                                rounded_result[key] = float(
                                    f"{rounded_result[key]:.5g}"
                                )
                rounded_results.append(rounded_result)

            with open(filename, "w") as f:
                json.dump(rounded_results, f, indent=2)

            print(f"Results saved to: {filename}")
            return len([r for r in all_results if r["success"]]) == len(all_results)

    # Run the scaling test
    print("Running autoregressive scaling tests...")
    success = test_autoregressive_scaling()

    if success:
        print("\nAll tests completed successfully!")
    else:
        print("\nSome tests failed - check the CSV/JSON file for details")
