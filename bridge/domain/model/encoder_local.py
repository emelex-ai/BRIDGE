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
