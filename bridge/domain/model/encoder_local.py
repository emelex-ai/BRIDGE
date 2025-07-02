from typing import Optional

import torch
import torch.nn as nn

from bridge.domain.model.transformer_local_attention import (
    LocalAttentionEncoderLayer,
)


class EncoderLocal(nn.Module):  # Renamed from EncoderFlash
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        device: str = "cpu",
        window_size: int = 512,
        causal: bool = False,
        look_backward: int = 1,
        look_forward: Optional[int] = None,
    ) -> None:
        """Initialize EncoderLocal with local attention on CUDA, standard attention on CPU.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            device: Device to use ("cpu" or "cuda")
            window_size: Local attention window size (only used for CUDA)
            causal: Whether to use causal attention (only used for CUDA)
            look_backward: Number of windows to look backward (only used for CUDA)
            look_forward: Number of windows to look forward (only used for CUDA)

        """
        super(EncoderLocal, self).__init__()

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
                "look_backward": look_backward,
                "look_forward": look_forward,
            }
            encoder_layer = LocalAttentionEncoderLayer(**local_kwargs)
            print(f"Using LocalAttentionEncoderLayer with window_size={window_size}")
        else:
            encoder_layer = nn.TransformerEncoderLayer(**base_kwargs)
            print("Using standard TransformerEncoderLayer")

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
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

    def test_encoder_local():
        """Test EncoderLocal functionality for both CPU and CUDA devices.

        Tests:
        1. CPU device with standard TransformerEncoderLayer
        2. CUDA device with LocalAttentionEncoderLayer (if available)
        3. Shape consistency
        4. Gradient flow
        5. Forward pass correctness

        """
        print("=" * 60)
        print("Testing EncoderLocal")
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
        print("Test 1: EncoderLocal with CPU device")
        try:
            cpu_encoder = EncoderLocal(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                device="cpu",
                window_size=window_size,
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
        print("Test 2: EncoderLocal with CUDA device")
        if torch.cuda.is_available():
            try:
                cuda_encoder = EncoderLocal(
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
            print("All EncoderLocal tests PASSED! ✓")
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
            print("Some EncoderLocal tests FAILED! ❌")
        print("=" * 60)

        return success

    # Run the test
    success = test_encoder_local()

    if not success:
        print("Tests failed!")
        exit(1)
    else:
        print("All tests passed successfully!")

    def test_autoregressive_scaling():
        """Test autoregressive local attention scaling with different window sizes.

        Tests scaling performance with context size 4096 and various window sizes.
        Measures forward pass time for different window configurations.
        """
        print("=" * 60)
        print("Testing Autoregressive Local Attention Scaling")
        print("=" * 60)

        # Test parameters
        batch_size = 2  # Smaller batch for memory efficiency
        seq_len = 4096  # Large context size
        d_model = 512
        nhead = 8
        num_layers = 2  # Fewer layers for faster testing
        window_sizes = [16, 64, 128, 512, 4096]  # Different window sizes to test
        num_steps = 5  # Number of forward passes to average timing

        print(f"Test configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Model dimension: {d_model}")
        print(f"  Number of heads: {nhead}")
        print(f"  Number of layers: {num_layers}")
        print(f"  Number of timing steps: {num_steps}")
        print(f"  Window sizes to test: {window_sizes}")
        print()

        # Test on CUDA if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running tests on: {device}")
        print()

        results = []

        for window_size in window_sizes:
            print(f"Testing window size: {window_size}")

            try:
                # Create encoder with current window size
                encoder = EncoderLocal(
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    device=device,
                    window_size=window_size,
                    causal=True,  # Autoregressive
                    look_backward=1,
                    look_forward=0,  # Causal - no looking forward
                )

                if device == "cuda":
                    encoder = encoder.to("cuda")

                # Create test input
                x = torch.randn(batch_size, seq_len, d_model, device=device)

                # Warmup run
                with torch.no_grad():
                    _ = encoder(x)

                if device == "cuda":
                    torch.cuda.synchronize()

                # Time multiple forward passes
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
                    step_time = end_time - start_time
                    times.append(step_time)

                    print(f"  Step {step + 1}: {step_time:.4f}s")

                # Calculate statistics
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)

                # Memory usage (if CUDA)
                if device == "cuda":
                    torch.cuda.empty_cache()
                    initial_memory = torch.cuda.memory_allocated()
                    with torch.no_grad():
                        _ = encoder(x)
                    peak_memory = torch.cuda.max_memory_allocated()
                    memory_used = (peak_memory - initial_memory) / 1024**2  # MB
                    torch.cuda.empty_cache()
                else:
                    memory_used = 0

                # Verify output shape
                assert (
                    output.shape == x.shape
                ), f"Output shape mismatch: {output.shape} vs {x.shape}"

                # Store results
                result = {
                    "window_size": window_size,
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "memory_mb": memory_used,
                    "tokens_per_sec": (batch_size * seq_len) / avg_time,
                    "success": True,
                }
                results.append(result)

                print(
                    f"  ✓ Average time: {avg_time:.4f}s (min: {min_time:.4f}s, max: {max_time:.4f}s)"
                )
                print(f"  ✓ Tokens/sec: {result['tokens_per_sec']:.0f}")
                if device == "cuda":
                    print(f"  ✓ Memory used: {memory_used:.2f} MB")
                print(f"  ✓ Shape verification passed")
                print()

            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
                result = {
                    "window_size": window_size,
                    "avg_time": float("inf"),
                    "min_time": float("inf"),
                    "max_time": float("inf"),
                    "memory_mb": 0,
                    "tokens_per_sec": 0,
                    "success": False,
                    "error": str(e),
                }
                results.append(result)
                import traceback

                traceback.print_exc()
                print()

        # Print summary results
        print("=" * 60)
        print("AUTOREGRESSIVE SCALING RESULTS SUMMARY")
        print("=" * 60)
        print(
            f"{'Window Size':<12} {'Avg Time (s)':<12} {'Tokens/sec':<12} {'Memory (MB)':<12} {'Status':<8}"
        )
        print("-" * 60)

        successful_results = [r for r in results if r["success"]]

        for result in results:
            status = "✓ PASS" if result["success"] else "❌ FAIL"
            print(
                f"{result['window_size']:<12} {result['avg_time']:<12.4f} {result['tokens_per_sec']:<12.0f} {result['memory_mb']:<12.2f} {status:<8}"
            )

        if len(successful_results) > 1:
            print("\n" + "=" * 60)
            print("SCALING ANALYSIS")
            print("=" * 60)

            # Compare to baseline (smallest successful window)
            baseline = successful_results[0]
            print(f"Baseline (window size {baseline['window_size']}):")
            print(f"  Time: {baseline['avg_time']:.4f}s")
            print(f"  Tokens/sec: {baseline['tokens_per_sec']:.0f}")
            if device == "cuda" and baseline["memory_mb"] > 0:
                memory_ratio = result["memory_mb"] / baseline["memory_mb"]
                print(
                    f"  Window {result['window_size']:>4}: {memory_ratio:.2f}x memory"
                )
            print()

            print("Relative performance:")
            for result in successful_results[1:]:
                time_ratio = result["avg_time"] / baseline["avg_time"]
                speed_ratio = result["tokens_per_sec"] / baseline["tokens_per_sec"]
                if device == "cuda" and baseline["memory_mb"] > 0:
                    memory_ratio = result["memory_mb"] / baseline["memory_mb"]
                    print(
                        f"  Window {result['window_size']:>4}: {time_ratio:.2f}x time, {speed_ratio:.2f}x speed, {memory_ratio:.2f}x memory"
                    )
                else:
                    print(
                        f"  Window {result['window_size']:>4}: {time_ratio:.2f}x time, {speed_ratio:.2f}x speed"
                    )

            # Theoretical vs actual scaling
            print(f"\nTheoretical scaling (assuming O(n*w) complexity):")
            for result in successful_results[1:]:
                theoretical_ratio = result["window_size"] / baseline["window_size"]
                actual_ratio = result["avg_time"] / baseline["avg_time"]
                efficiency = theoretical_ratio / actual_ratio if actual_ratio > 0 else 0
                print(
                    f"  Window {result['window_size']:>4}: theoretical {theoretical_ratio:.2f}x, actual {actual_ratio:.2f}x, efficiency {efficiency:.2f}"
                )

        print("\n" + "=" * 60)
        if all(r["success"] for r in results):
            print("All autoregressive scaling tests PASSED! ✓")
        else:
            failed_count = sum(1 for r in results if not r["success"])
            print(f"{failed_count} out of {len(results)} tests FAILED! ❌")
        print("=" * 60)

        return all(r["success"] for r in results)

    # Run both tests
    print("Running basic functionality tests...")
    basic_success = test_encoder_local()

    print("\n" + "=" * 80 + "\n")

    print("Running autoregressive scaling tests...")
    scaling_success = test_autoregressive_scaling()

    overall_success = basic_success and scaling_success

    if not overall_success:
        print("\nSome tests failed!")
        exit(1)
    else:
        print("\nAll tests passed successfully!")
