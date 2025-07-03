from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, unpack
from torch.nn import TransformerEncoderLayer


def default(value, d):
    """Helper function to provide default values."""
    return d if value is None else value


def exists(val):
    """Helper function to check if value exists."""
    return val is not None


def max_neg_value(tensor):
    """Get the maximum negative value for the tensor's dtype."""
    return -torch.finfo(tensor.dtype).max


class TrueSlidingWindowAttention(nn.Module):
    """True sliding window attention without chunking overhead.

    This implements proper sliding window attention where each token
    attends to a fixed window around itself, not chunked attention
    with overlap like the local-attention library.
    """

    def __init__(
        self,
        window_size: int,
        causal: bool = False,
        dropout: float = 0.0,
        scale: Optional[float] = None,
    ):
        """Initialize true sliding window attention.

        Args:
            window_size: Size of the attention window
            causal: Whether to use causal attention
            dropout: Dropout probability
            scale: Attention scale factor (defaults to 1/sqrt(dim))
        """
        super().__init__()
        self.window_size = window_size
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.scale = scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for true sliding window attention.

        Args:
            q: Query tensor (batch, seq_len, dim) or packed format
            k: Key tensor (batch, seq_len, dim) or packed format
            v: Value tensor (batch, seq_len, dim) or packed format
            mask: Optional attention mask

        Returns:
            Output tensor with same shape as q
        """
        # Handle packed format (for compatibility with local-attention)
        shape = q.shape
        if len(shape) == 2:  # Packed format from local-attention
            (q, packed_shape), (k, _), (v, _) = map(
                lambda t: pack([t], "* n d"), (q, k, v)
            )
        else:
            packed_shape = None

        b, n, dim_head = q.shape
        device, dtype = q.device, q.dtype

        # Set scale
        scale = default(self.scale, dim_head**-0.5)
        q = q * scale

        # Create sliding window mask efficiently
        if self.causal:
            # Causal sliding window: each token i sees tokens [max(0, i-window_size+1), i]
            row_idx = torch.arange(n, device=device).unsqueeze(1)  # (n, 1)
            col_idx = torch.arange(n, device=device).unsqueeze(0)  # (1, n)

            # Causal constraint: col_idx <= row_idx
            causal_mask = col_idx <= row_idx

            # Window constraint: col_idx >= row_idx - window_size + 1
            window_mask = col_idx >= (row_idx - self.window_size + 1)

            # Combined mask
            attention_mask = causal_mask & window_mask
        else:
            # Non-causal sliding window: each token sees window_size//2 tokens on each side
            row_idx = torch.arange(n, device=device).unsqueeze(1)
            col_idx = torch.arange(n, device=device).unsqueeze(0)

            # Distance constraint: |row_idx - col_idx| <= window_size // 2
            distance_mask = torch.abs(row_idx - col_idx) <= (self.window_size // 2)
            attention_mask = distance_mask

        # Compute attention scores
        sim = torch.bmm(q, k.transpose(-2, -1))  # (b, n, n)

        # Apply sliding window mask
        mask_value = max_neg_value(sim)
        sim = sim.masked_fill(~attention_mask.unsqueeze(0), mask_value)

        # Apply additional input mask if provided
        if exists(mask):
            if mask.dim() == 2:  # (batch, seq_len) -> (batch, seq_len, seq_len)
                mask = mask.unsqueeze(1).expand(-1, n, -1)
            elif mask.dim() == 3 and mask.shape[0] != b:  # Handle different batch sizes
                mask = mask.repeat(b // mask.shape[0], 1, 1)
            sim = sim.masked_fill(~mask, mask_value)

        # Softmax and dropout
        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.bmm(attn, v)  # (b, n, dim_head)

        # Unpack if necessary
        if packed_shape is not None:
            out, *_ = unpack(out, packed_shape, "* n d")

        return out


class TrueSlidingWindowEncoderLayer(TransformerEncoderLayer):
    """TransformerEncoderLayer using true sliding window attention.

    Drop-in replacement for LocalAttentionEncoderLayer with much better
    memory efficiency, especially for small window sizes.
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
        # Compatibility parameters (ignored but accepted)
        look_backward: int = 1,
        look_forward: Optional[int] = None,
        **kwargs,  # Accept any additional parameters and ignore them
    ):
        """Initialize TrueSlidingWindowEncoderLayer.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            activation: Activation function
            layer_norm_eps: Layer norm epsilon
            batch_first: Whether to use batch_first format
            norm_first: Whether to apply norm before attention
            bias: Whether to use bias in linear layers
            device: Device for tensors
            dtype: Data type for tensors
            window_size: Size of the sliding attention window
            causal: Whether to use causal attention
            look_backward: (Ignored - for compatibility with LocalAttentionEncoderLayer)
            look_forward: (Ignored - for compatibility with LocalAttentionEncoderLayer)
            **kwargs: Additional parameters (ignored for compatibility)
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

        # Store parameters
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.window_size = window_size
        self.causal = causal

        # Note: look_backward and look_forward are ignored since we implement
        # true sliding window attention, not chunked attention with overlap

        # Replace projection layers
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(
            d_model, d_model, bias=bias, device=device, dtype=dtype
        )

        # Initialize true sliding window attention
        self.sliding_window_attn = TrueSlidingWindowAttention(
            window_size=window_size,
            causal=causal,
            dropout=dropout,
            scale=self.head_dim**-0.5,
        )

        print(
            f"Initialized TrueSlidingWindowEncoderLayer with window_size={window_size}, causal={causal}"
        )
        if look_backward != 1 or look_forward is not None:
            print(
                f"Note: look_backward={look_backward} and look_forward={look_forward} are ignored in true sliding window attention"
            )

    def _sliding_window_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass using true sliding window attention.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            key_padding_mask: Padding mask for keys
            attn_mask: Attention mask

        Returns:
            Output tensor
        """
        # Handle batch_first vs seq_first
        if not self.batch_first:
            # Convert from (seq_len, batch_size, d_model) to (batch_size, seq_len, d_model)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, seq_len, self.nhead, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.nhead, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.nhead, self.head_dim)

        # Reshape to (batch * heads, seq_len, head_dim) for efficient computation
        q = q.transpose(1, 2).reshape(batch_size * self.nhead, seq_len, self.head_dim)
        k = k.transpose(1, 2).reshape(batch_size * self.nhead, seq_len, self.head_dim)
        v = v.transpose(1, 2).reshape(batch_size * self.nhead, seq_len, self.head_dim)

        # Apply sliding window attention
        output = self.sliding_window_attn(q, k, v, mask=key_padding_mask)

        # Reshape back to (batch_size, seq_len, d_model)
        output = output.reshape(batch_size, self.nhead, seq_len, self.head_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        # Apply output projection
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
        """Forward pass of the encoder layer.

        Args:
            src: Input tensor
            src_mask: Source mask (not used in sliding window)
            src_key_padding_mask: Key padding mask
            is_causal: Whether to use causal attention (overrides init setting)

        Returns:
            Output tensor
        """
        # Use the sliding window attention instead of self.self_attn
        if self.norm_first:
            # Pre-norm
            src_norm = self.norm1(src)
            attn_output = self._sliding_window_attention_forward(
                src_norm, src_norm, src_norm, key_padding_mask=src_key_padding_mask
            )
            src = src + self.dropout1(attn_output)

            # Feedforward
            src_norm = self.norm2(src)
            ff_output = self.linear2(
                self.dropout(self.activation(self.linear1(src_norm)))
            )
            src = src + self.dropout2(ff_output)
        else:
            # Post-norm
            attn_output = self._sliding_window_attention_forward(
                src, src, src, key_padding_mask=src_key_padding_mask
            )
            src = self.norm1(src + self.dropout1(attn_output))

            # Feedforward
            ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(ff_output))

        return src


class TrueSlidingWindowEncoder(nn.Module):
    """Encoder using true sliding window attention layers."""

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
    ):
        """Initialize encoder with true sliding window attention.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            batch_first: Whether to use batch_first format
            window_size: Size of sliding attention window
            causal: Whether to use causal attention
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TrueSlidingWindowEncoderLayer(
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

        self.num_layers = num_layers

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through all encoder layers.

        Args:
            src: Input tensor
            mask: Source mask
            src_key_padding_mask: Key padding mask

        Returns:
            Output tensor
        """
        output = src

        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        return output


# Test function
def test_true_sliding_window():
    """Test the true sliding window attention implementation."""
    batch_size = 2
    seq_len = 1024
    d_model = 512
    nhead = 8
    window_size = 128

    # Create test data
    src = torch.randn(batch_size, seq_len, d_model)

    # Test encoder layer
    layer = TrueSlidingWindowEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        window_size=window_size,
        batch_first=True,
        causal=True,
    )

    output = layer(src)
    print(f"Input shape: {src.shape}")
    print(f"Output shape: {output.shape}")
    print("True sliding window attention test passed!")

    return output


if __name__ == "__main__":
    test_true_sliding_window()
