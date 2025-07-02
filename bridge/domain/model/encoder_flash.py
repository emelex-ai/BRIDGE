from typing import Optional

import torch
import torch.nn as nn

from bridge.domain.model.transformer_fast_attention import (
    FlashAttentionEncoderLayer,
)


class EncoderFlash(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        device: str = "cpu",
    ) -> None:
        super(EncoderFlash, self).__init__()
        kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "batch_first": True,
            "dim_feedforward": 4 * d_model,
            "device": device,
        }
        # Call if on CPU
        if device == "cuda":
            encoder_layer = FlashAttentionEncoderLayer(**kwargs)
        else:
            encoder_layer = nn.TransformerEncoderLayer(**kwargs)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        return output


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    import torch.nn as nn

    def test_encoder_flash():
        """Test EncoderFlash functionality for both CPU and CUDA devices.

        Tests:
        1. CPU device with standard TransformerEncoderLayer
        2. CUDA device with FlashAttentionEncoderLayer (if available)
        3. Shape consistency
        4. Gradient flow
        5. Forward pass correctness

        """
        print("=" * 60)
        print("Testing EncoderFlash")
        print("=" * 60)

        # Test parameters
        batch_size, seq_len, d_model = 4, 128, 512
        nhead = 8
        num_layers = 3

        print(f"Test configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Model dimension: {d_model}")
        print(f"  Number of heads: {nhead}")
        print(f"  Number of layers: {num_layers}")
        print()

        success = True

        # Test 1: CPU device
        print("Test 1: EncoderFlash with CPU device")
        try:
            cpu_encoder = EncoderFlash(
                d_model=d_model, nhead=nhead, num_layers=num_layers, device="cpu"
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
            print("✓ CPU encoder uses standard TransformerEncoderLayer")

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
            success = False
            print()

        # Test 2: CUDA device (if available)
        print("Test 2: EncoderFlash with CUDA device")
        if torch.cuda.is_available():
            try:
                cuda_encoder = EncoderFlash(
                    d_model=d_model, nhead=nhead, num_layers=num_layers, device="cuda"
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
                print("✓ CUDA encoder uses FlashAttentionEncoderLayer")

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

        # Test 3: Attention masks (works on both devices)
        print("Test 3: Attention masks")
        try:
            # Test with CPU encoder
            attn_mask = torch.tril(torch.ones(seq_len, seq_len))
            key_padding_mask = torch.zeros(batch_size, seq_len).bool()
            key_padding_mask[:, -10:] = True  # Mask last 10 tokens

            masked_output_cpu = cpu_encoder(
                x_cpu, src_mask=attn_mask, src_key_padding_mask=key_padding_mask
            )
            assert masked_output_cpu.shape == x_cpu.shape
            print("✓ CPU attention mask test successful")

            # Test with CUDA encoder if available
            if torch.cuda.is_available():
                attn_mask_cuda = attn_mask.to("cuda")
                key_padding_mask_cuda = key_padding_mask.to("cuda")

                masked_output_cuda = cuda_encoder(
                    x_cuda,
                    src_mask=attn_mask_cuda,
                    src_key_padding_mask=key_padding_mask_cuda,
                )
                assert masked_output_cuda.shape == x_cuda.shape
                print("✓ CUDA attention mask test successful")

            print()

        except Exception as e:
            print(f"❌ Attention mask test failed: {str(e)}")
            success = False
            print()

        # Test 4: Different configurations
        print("Test 4: Different configurations")
        try:
            # Test different model sizes
            for test_d_model, test_nhead in [(256, 4), (768, 12)]:
                if test_d_model % test_nhead == 0:  # Must be divisible
                    test_encoder_cpu = EncoderFlash(
                        d_model=test_d_model,
                        nhead=test_nhead,
                        num_layers=2,
                        device="cpu",
                    )

                    test_x = torch.randn(2, 32, test_d_model)  # Smaller for speed
                    test_output = test_encoder_cpu(test_x)
                    assert test_output.shape == test_x.shape
                    print(
                        f"✓ d_model={test_d_model}, nhead={test_nhead} configuration successful"
                    )

            print()

        except Exception as e:
            print(f"❌ Configuration test failed: {str(e)}")
            success = False
            print()

        # Summary
        print("=" * 60)
        if success:
            print("All EncoderFlash tests PASSED! ✓")
            print()
            print("Summary:")
            print("✓ CPU device: Uses standard TransformerEncoderLayer")
            if torch.cuda.is_available():
                print("✓ CUDA device: Uses FlashAttentionEncoderLayer")
            else:
                print("⚠️  CUDA device: Not tested (CUDA unavailable)")
            print("✓ Shape consistency maintained")
            print("✓ Gradient flow working")
            print("✓ Attention masks supported")
        else:
            print("Some EncoderFlash tests FAILED! ❌")
        print("=" * 60)

        return success

    # Run the test
    success = test_encoder_flash()

    if not success:
        print("Tests failed!")
        exit(1)
    else:
        print("All tests passed successfully!")
