import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer

# Add the local-attention module to the path
# sys.path.append(os.path.join(os.path.dirname(__file__), "local-attention"))

try:
    from einops import rearrange
    from local_attention import LocalAttention

    LOCAL_ATTENTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Local attention not available: {e}")
    print("Falling back to standard attention")
    LOCAL_ATTENTION_AVAILABLE = False


def rearrange_fallback(tensor, pattern, **kwargs):
    """Fallback implementation for basic rearrange operations if einops is not available.

    Args:
        tensor: Input tensor
        pattern: Rearrangement pattern
        **kwargs: Additional arguments

    Returns:
        Rearranged tensor

    """
    if pattern == "b n (h d) -> (b h) n d":
        b, n, hd = tensor.shape
        h = kwargs.get("h", hd // tensor.shape[-1])
        d = hd // h
        return tensor.view(b, n, h, d).transpose(0, 2).contiguous().view(b * h, n, d)
    elif pattern == "(b h) n d -> b n h d":
        bh, n, d = tensor.shape
        b = kwargs.get("b", bh)
        h = bh // b
        return tensor.view(b, h, n, d).transpose(1, 2)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented in fallback")


if not LOCAL_ATTENTION_AVAILABLE:
    rearrange = rearrange_fallback


class LocalAttentionEncoderLayer(TransformerEncoderLayer):
    """TransformerEncoderLayer that uses Local Attention instead of standard attention.

    This subclass replaces the standard multi-head attention with Local Attention
    for improved memory efficiency with limited attention window.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        window_size: int = 512,
        causal: bool = False,
        look_backward: int = 1,
        look_forward: Optional[int] = None,
    ):
        """Initialize LocalAttentionEncoderLayer.

        Args:
            d_model: The number of expected features in the input.
            nhead: The number of heads in the multiheadattention models.
            dim_feedforward: The dimension of the feedforward network model.
            dropout: The dropout probability.
            activation: The activation function of the intermediate layer.
            layer_norm_eps: The eps value in layer normalization components.
            batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).
            norm_first: If True, layer norm is done prior to attention and feedforward operations.
            bias: If set to False, Linear and LayerNorm layers will not learn an additive bias.
            device: Device for tensors.
            dtype: Data type for tensors.
            window_size: Size of the local attention window.
            causal: Whether to use causal attention.
            look_backward: Number of windows to look backward.
            look_forward: Number of windows to look forward.

        """
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            device,
            dtype,
        )
        print(f"Enter LocalAttentionEncoderLayer")

        # Store parameters for Local Attention
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.window_size = window_size
        self.causal = causal

        # Replace the self-attention with Local Attention compatible projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(
            d_model, d_model, bias=bias, device=device, dtype=dtype
        )

        # Initialize Local Attention if available
        if LOCAL_ATTENTION_AVAILABLE:
            self.local_attn = LocalAttention(
                window_size=window_size,
                causal=causal,
                look_backward=look_backward,
                look_forward=look_forward,
                dropout=dropout,
                dim=self.head_dim,
                autopad=True,
                exact_windowsize=False,
                use_rotary_pos_emb=True,
            )
        else:
            # Fallback to standard attention
            self.local_attn = None
            print("Warning: Using standard attention as fallback")

    def _local_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass using Local Attention.

        Args:
            query: Query tensor of shape (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)
            key: Key tensor of shape (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)
            value: Value tensor of shape (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)
            key_padding_mask: Padding mask for keys
            attn_mask: Attention mask

        Returns:
            Output tensor with same shape as query

        """
        # If local attention is not available, use the parent class's self_attn
        if not LOCAL_ATTENTION_AVAILABLE or self.local_attn is None:
            # Fallback to standard multi-head attention
            if not self.batch_first:
                # Standard transformer expects (seq_len, batch_size, d_model)
                attn_output, _ = self.self_attn(
                    query,
                    key,
                    value,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
            else:
                # Convert to seq_first for standard attention
                query = query.transpose(0, 1)
                key = key.transpose(0, 1)
                value = value.transpose(0, 1)

                attn_output, _ = self.self_attn(
                    query,
                    key,
                    value,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )

                # Convert back to batch_first
                attn_output = attn_output.transpose(0, 1)

            return attn_output

        # Handle batch_first vs seq_first
        if not self.batch_first:
            # Convert from (seq_len, batch_size, d_model) to (batch_size, seq_len, d_model)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.nhead, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.nhead, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.nhead, self.head_dim)

        # Rearrange for local attention: (batch, seq, heads, head_dim) -> (batch * heads, seq, head_dim)
        q = rearrange(q, "b n h d -> (b h) n d", b=batch_size, h=self.nhead)
        k = rearrange(k, "b n h d -> (b h) n d", b=batch_size, h=self.nhead)
        v = rearrange(v, "b n h d -> (b h) n d", b=batch_size, h=self.nhead)

        # Handle input mask for local attention
        input_mask = None
        if key_padding_mask is not None:
            # key_padding_mask is (batch_size, seq_len) - True for positions to ignore
            # Convert to input_mask for local attention - True for positions to attend to
            input_mask = ~key_padding_mask

        # Apply local attention
        output = self.local_attn(q, k, v, input_mask=input_mask)

        # Rearrange back: (batch * heads, seq, head_dim) -> (batch, seq, heads, head_dim) -> (batch, seq, d_model)
        output = rearrange(output, "(b h) n d -> b n h d", b=batch_size, h=self.nhead)
        output = output.reshape(batch_size, seq_len, self.d_model)

        # Apply output projection
        output = self.out_proj(output)

        # Convert back to original format if needed
        if not self.batch_first:
            output = output.transpose(0, 1)

        return output

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the LocalAttentionEncoderLayer.

        Args:
            src: Input tensor
            src_mask: Attention mask
            src_key_padding_mask: Padding mask
            is_causal: Whether to use causal attention

        Returns:
            Output tensor

        """
        # Pre-norm or post-norm
        if self.norm_first:
            # Pre-norm: LayerNorm -> Attention -> Residual
            src_norm = self.norm1(src)
            attn_output = self._local_attention_forward(
                src_norm,
                src_norm,
                src_norm,
                key_padding_mask=src_key_padding_mask,
                attn_mask=src_mask,
            )
            src = src + self.dropout1(attn_output)

            # Pre-norm: LayerNorm -> FFN -> Residual
            src_norm = self.norm2(src)
            ffn_output = self.linear2(
                self.dropout(self.activation(self.linear1(src_norm)))
            )
            src = src + self.dropout2(ffn_output)
        else:
            # Post-norm: Attention -> Residual -> LayerNorm
            attn_output = self._local_attention_forward(
                src, src, src, key_padding_mask=src_key_padding_mask, attn_mask=src_mask
            )
            src = self.norm1(src + self.dropout1(attn_output))

            # Post-norm: FFN -> Residual -> LayerNorm
            ffn_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(ffn_output))

        return src


class LocalAttentionEncoder(nn.Module):
    """Transformer Encoder using Local Attention layers.

    This encoder uses LocalAttentionEncoderLayer instead of standard
    TransformerEncoderLayer for better performance with long sequences.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        batch_first: bool = True,
        window_size: int = 512,
        causal: bool = False,
        look_backward: int = 1,
        look_forward: Optional[int] = None,
    ):
        """Initialize LocalAttentionEncoder.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: FFN dimension
            dropout: Dropout probability
            batch_first: Whether to use batch_first format
            window_size: Size of the local attention window
            causal: Whether to use causal attention
            look_backward: Number of windows to look backward
            look_forward: Number of windows to look forward

        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                LocalAttentionEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=batch_first,
                    window_size=window_size,
                    causal=causal,
                    look_backward=look_backward,
                    look_forward=look_forward,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        src: torch.Tensor,
        mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass through all encoder layers.

        Args:
            src: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask
            src_key_padding_mask: Padding mask

        Returns:
            Encoded output tensor

        """
        output = src

        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        return output


# ----------------------------------------------------------------------
if __name__ == "__main__":

    def test_local_attention_encoder():
        """Test LocalAttentionEncoder functionality for both CPU and CUDA devices.

        Tests:
        1. CPU device with LocalAttentionEncoderLayer
        2. CUDA device with LocalAttentionEncoderLayer (if available)
        3. Shape consistency
        4. Gradient flow
        5. Forward pass correctness

        """
        print("=" * 60)
        print("Testing LocalAttentionEncoder")
        print("=" * 60)

        # Test parameters
        batch_size, seq_len, d_model = (
            4,
            256,
            512,
        )  # Increased seq_len for local attention
        nhead = 8
        num_layers = 3
        window_size = 64  # Local attention window size

        print(f"Test configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Model dimension: {d_model}")
        print(f"  Number of heads: {nhead}")
        print(f"  Number of layers: {num_layers}")
        print(f"  Window size: {window_size}")
        print(f"  Local attention available: {LOCAL_ATTENTION_AVAILABLE}")
        print()

        success = True

        # Test 1: CPU device
        print("Test 1: LocalAttentionEncoder with CPU device")
        try:
            cpu_encoder = LocalAttentionEncoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                batch_first=True,
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
            if LOCAL_ATTENTION_AVAILABLE:
                print("✓ CPU encoder uses LocalAttentionEncoderLayer")
            else:
                print("✓ CPU encoder uses standard attention (fallback)")

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
        print("Test 2: LocalAttentionEncoder with CUDA device")
        if torch.cuda.is_available():
            try:
                cuda_encoder = LocalAttentionEncoder(
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    batch_first=True,
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
                if LOCAL_ATTENTION_AVAILABLE:
                    print("✓ CUDA encoder uses LocalAttentionEncoderLayer")
                else:
                    print("✓ CUDA encoder uses standard attention (fallback)")

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

        # Test 3: Attention masks and different window sizes
        print("Test 3: Attention masks and different window sizes")
        try:
            # Test with different window sizes
            for test_window_size in [32, 128]:
                test_encoder_cpu = LocalAttentionEncoder(
                    d_model=256,  # Smaller for speed
                    nhead=4,
                    num_layers=2,
                    batch_first=True,
                    window_size=test_window_size,
                )

                test_x = torch.randn(2, 256, 256)  # seq_len divisible by window_size
                test_output = test_encoder_cpu(test_x)
                assert test_output.shape == test_x.shape
                print(f"✓ Window size {test_window_size} test successful")

            # Test with key padding mask
            key_padding_mask = torch.zeros(batch_size, seq_len).bool()
            key_padding_mask[:, -10:] = True  # Mask last 10 tokens

            masked_output_cpu = cpu_encoder(
                x_cpu, src_key_padding_mask=key_padding_mask
            )
            assert masked_output_cpu.shape == x_cpu.shape
            print("✓ CPU attention mask test successful")

            print()

        except Exception as e:
            print(f"❌ Window size and mask test failed: {str(e)}")
            import traceback

            traceback.print_exc()
            success = False
            print()

        # Test 4: Causal attention (only if local attention is available)
        if LOCAL_ATTENTION_AVAILABLE:
            print("Test 4: Causal attention")
            try:
                causal_encoder = LocalAttentionEncoder(
                    d_model=256,
                    nhead=4,
                    num_layers=2,
                    batch_first=True,
                    window_size=64,
                    causal=True,
                )

                test_x = torch.randn(2, 128, 256)
                causal_output = causal_encoder(test_x)
                assert causal_output.shape == test_x.shape
                print("✓ Causal attention test successful")
                print()

            except Exception as e:
                print(f"❌ Causal attention test failed: {str(e)}")
                import traceback

                traceback.print_exc()
                success = False
                print()
        else:
            print("Test 4: Causal attention")
            print("⚠️  Skipping causal attention test (local attention not available)")
            print()

        # Summary
        print("=" * 60)
        if success:
            print("All LocalAttentionEncoder tests PASSED! ✓")
            print()
            print("Summary:")
            if LOCAL_ATTENTION_AVAILABLE:
                print("✓ CPU device: Uses LocalAttentionEncoderLayer")
                if torch.cuda.is_available():
                    print("✓ CUDA device: Uses LocalAttentionEncoderLayer")
                else:
                    print("⚠️  CUDA device: Not tested (CUDA unavailable)")
                print("✓ Local attention windows supported")
                print("✓ Causal attention working")
            else:
                print("⚠️  CPU device: Uses standard attention (fallback)")
                if torch.cuda.is_available():
                    print("⚠️  CUDA device: Uses standard attention (fallback)")
                else:
                    print("⚠️  CUDA device: Not tested (CUDA unavailable)")
                print("⚠️  Local attention not available - using fallback")
            print("✓ Shape consistency maintained")
            print("✓ Gradient flow working")
        else:
            print("Some LocalAttentionEncoder tests FAILED! ❌")
        print("=" * 60)

        return success

    # Run the test
    success = test_local_attention_encoder()

    if not success:
        print("Tests failed!")
        exit(1)
    else:
        print("All tests passed successfully!")
