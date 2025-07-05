#!/usr/bin/env python3
"""Exact Sliding Window Transformer Encoder Layer for BRIDGE architecture.

This module provides a TransformerEncoderLayer that uses exact sliding window
attention without chunking limitations and without requiring PyTorch nightly.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

from bridge.domain.model.exact_sliding_window_attention import (
    ExactSlidingWindowMHA,
)


class ExactSlidingWindowEncoderLayer(TransformerEncoderLayer):
    """TransformerEncoderLayer that uses exact sliding window attention.

    This subclass replaces the standard multi-head attention with
    ExactSlidingWindowMHA for true sliding window attention without
    chunking limitations.
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
        causal: bool = True,
        **kwargs,
    ):
        """Initialize ExactSlidingWindowEncoderLayer.

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
            window_size: Size of the sliding attention window.
            causal: Whether to use causal attention.
            **kwargs: Additional arguments for compatibility.
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

        # Store parameters for exact sliding window attention
        self.window_size = window_size
        self.causal = causal

        # Replace the self-attention with exact sliding window attention
        self.self_attn = ExactSlidingWindowMHA(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            window_size=window_size,
            causal=causal,
            device=device,
            dtype=dtype,
        )

        print(
            f"Initialized ExactSlidingWindowEncoderLayer with window_size={window_size}, causal={causal}"
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the ExactSlidingWindowEncoderLayer.

        Args:
            src: Input tensor
            src_mask: Attention mask (ignored - sliding window handles masking)
            src_key_padding_mask: Padding mask
            is_causal: Whether to use causal attention (ignored - set in __init__)

        Returns:
            Output tensor
        """
        # Pre-norm or post-norm
        if self.norm_first:
            # Pre-norm: LayerNorm -> Attention -> Residual
            src_norm = self.norm1(src)
            attn_output = self.self_attn(
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
            attn_output = self.self_attn(
                src, src, src, key_padding_mask=src_key_padding_mask, attn_mask=src_mask
            )
            src = self.norm1(src + self.dropout1(attn_output))

            # Post-norm: FFN -> Residual -> LayerNorm
            ffn_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(ffn_output))

        return src


def test_exact_sliding_window_encoder():
    """Test the ExactSlidingWindowEncoderLayer implementation."""
    print("Testing ExactSlidingWindowEncoderLayer")
    print("=" * 50)

    # Test parameters
    batch_size = 2
    seq_len = 128
    d_model = 256
    nhead = 8
    window_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create test data
    src = torch.randn(batch_size, seq_len, d_model, device=device)

    # Create encoder layer
    encoder_layer = ExactSlidingWindowEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        window_size=window_size,
        causal=True,
        batch_first=True,
    ).to(device)

    # Forward pass
    output = encoder_layer(src)

    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")

    # Test memory usage
    if device.type == "cuda":
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    print("ExactSlidingWindowEncoderLayer test passed!")

    return output


if __name__ == "__main__":
    test_exact_sliding_window_encoder()
