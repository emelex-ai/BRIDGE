#!/usr/bin/env python3
"""FlexAttention-based Transformer Encoder Layer for BRIDGE architecture.

This module provides a TransformerEncoderLayer that uses PyTorch's FlexAttention
for efficient sliding window attention without chunking limitations.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

# Check if FlexAttention is available (requires PyTorch nightly)
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    print(
        "Warning: FlexAttention not available. Install PyTorch nightly for FlexAttention support."
    )


class FlexAttentionEncoderLayer(TransformerEncoderLayer):
    """TransformerEncoderLayer that uses FlexAttention for sliding window attention.

    This subclass replaces the standard multi-head attention with FlexAttention
    for efficient sliding window attention without chunking limitations.
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
        """Initialize FlexAttentionEncoderLayer.

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

        print(
            f"Initializing FlexAttentionEncoderLayer with window_size={window_size}, causal={causal}"
        )

        # Store parameters for FlexAttention
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.window_size = window_size
        self.causal = causal

        # Replace the self-attention with FlexAttention compatible projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(
            d_model, d_model, bias=bias, device=device, dtype=dtype
        )

        # Initialize FlexAttention mask and functions
        if FLEX_ATTENTION_AVAILABLE:
            self._setup_flex_attention()
        else:
            print(
                "Warning: FlexAttention not available, falling back to standard attention"
            )

    def _setup_flex_attention(self):
        """Set up FlexAttention mask and score modification functions."""

        # Define sliding window + causal mask function
        def sliding_window_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx <= self.window_size
            return causal_mask & window_mask if self.causal else window_mask

        # Store the mask function
        self.mask_mod = sliding_window_causal_mask

        # Pre-compute block mask for efficiency (will be recomputed if sequence length changes)
        self.block_mask = None
        self.cached_seq_len = None

    def _get_block_mask(self, seq_len: int, device: torch.device):
        """Get or create block mask for the given sequence length."""
        if not FLEX_ATTENTION_AVAILABLE:
            return None

        # Reuse cached block mask if sequence length matches
        if self.block_mask is not None and self.cached_seq_len == seq_len:
            return self.block_mask

        # Create new block mask
        self.block_mask = create_block_mask(
            self.mask_mod,
            B=None,  # Broadcast over batch
            H=None,  # Broadcast over heads
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )
        self.cached_seq_len = seq_len

        return self.block_mask

    def _manual_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Manual attention computation as fallback when FlexAttention is not available.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            key_padding_mask: Padding mask for keys
            attn_mask: Attention mask

        Returns:
            Output tensor with same shape as query
        """
        # Handle batch_first vs seq_first
        if not self.batch_first:
            # Convert from (seq_len, batch_size, d_model) to (batch_size, seq_len, d_model)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, seq_len, _ = query.shape
        device = query.device

        # Project to Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.nhead, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.nhead, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.nhead, self.head_dim)

        # Transpose for attention computation: (batch, head, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply sliding window + causal mask
        if self.causal:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Apply sliding window mask
        if self.window_size > 0:
            # Create sliding window mask
            positions = torch.arange(seq_len, device=device)
            window_mask = (positions[:, None] - positions[None, :]) > self.window_size
            scores = scores.masked_fill(window_mask, float("-inf"))

        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Transpose back and reshape
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        # Apply output projection
        output = self.out_proj(output)

        # Convert back to original format if needed
        if not self.batch_first:
            output = output.transpose(0, 1)

        return output

    def _flex_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass using FlexAttention.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            key_padding_mask: Padding mask for keys
            attn_mask: Attention mask

        Returns:
            Output tensor with same shape as query
        """
        # Fallback to standard attention if FlexAttention not available
        if not FLEX_ATTENTION_AVAILABLE:
            # Use manual attention computation as fallback
            return self._manual_attention_forward(
                query, key, value, key_padding_mask, attn_mask
            )

        # Handle batch_first vs seq_first
        if not self.batch_first:
            # Convert from (seq_len, batch_size, d_model) to (batch_size, seq_len, d_model)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, seq_len, _ = query.shape
        device = query.device

        # Project to Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.nhead, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.nhead, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.nhead, self.head_dim)

        # Get block mask for this sequence length
        block_mask = self._get_block_mask(seq_len, device)

        # Apply FlexAttention
        output = flex_attention(q, k, v, block_mask=block_mask)

        # Reshape and project output
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        # Convert back to original format if needed
        if not self.batch_first:
            output = output.transpose(0, 1)

        return output

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the FlexAttentionEncoderLayer.

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
            attn_output = self._flex_attention_forward(
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
            attn_output = self._flex_attention_forward(
                src, src, src, key_padding_mask=src_key_padding_mask, attn_mask=src_mask
            )
            src = self.norm1(src + self.dropout1(attn_output))

            # Post-norm: FFN -> Residual -> LayerNorm
            ffn_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(ffn_output))

        return src


class FlexAttentionEncoder(nn.Module):
    """Transformer Encoder using FlexAttention layers.

    This encoder uses FlexAttentionEncoderLayer instead of standard
    TransformerEncoderLayer for efficient sliding window attention.
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
        causal: bool = True,
    ):
        """Initialize FlexAttentionEncoder.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: FFN dimension
            dropout: Dropout probability
            batch_first: Whether to use batch_first format
            window_size: Size of the sliding attention window
            causal: Whether to use causal attention
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                FlexAttentionEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=batch_first,
                    window_size=window_size,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
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


def test_flex_attention_encoder():
    """Test the FlexAttention encoder implementation."""
    print("Testing FlexAttention Encoder")
    print("=" * 50)

    # Test parameters
    batch_size = 2
    seq_len = 128
    d_model = 256
    nhead = 8
    num_layers = 4
    window_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create test data
    src = torch.randn(batch_size, seq_len, d_model, device=device)

    # Create encoder
    encoder = FlexAttentionEncoder(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        window_size=window_size,
        causal=True,
        batch_first=True,
    ).to(device)

    # Forward pass
    output = encoder(src)

    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")

    # Test memory usage
    if device.type == "cuda":
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    print("FlexAttention encoder test passed!")

    return output


if __name__ == "__main__":
    test_flex_attention_encoder()
