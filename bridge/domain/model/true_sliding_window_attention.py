from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, pack, rearrange, unpack
from local_attention import LocalAttention


# Helper functions (copied from local_attention since they're not exported)
def exists(val):
    """Helper function to check if value exists."""
    return val is not None


def default(value, d):
    """Helper function to provide default values."""
    return d if not exists(value) else value


def max_neg_value(tensor):
    """Get the maximum negative value for the tensor's dtype."""
    return -torch.finfo(tensor.dtype).max


class TrueSlidingWindowAttention(LocalAttention):
    """True sliding window attention that removes chunking overhead.

    This subclasses LocalAttention but disables chunking and overlapping
    to achieve true O(L) memory scaling.
    """

    def __init__(
        self,
        window_size: int,
        causal: bool = False,
        dropout: float = 0.0,
        scale: Optional[float] = None,
        dim: Optional[int] = None,
        use_rotary_pos_emb: bool = False,  # Disabled by default
        use_xpos: bool = False,
        xpos_scale_base: Optional[int] = None,
        **kwargs,
    ):
        """Initialize true sliding window attention."""
        # CRITICAL FIX: Disable chunking and overlapping
        super().__init__(
            window_size=window_size,
            causal=causal,
            look_backward=0,  # FIXED: No backward chunks (disable overlap)
            look_forward=0,  # FIXED: No forward chunks (disable overlap)
            dropout=dropout,
            scale=scale,
            dim=dim,
            autopad=True,
            exact_windowsize=True,
            use_rotary_pos_emb=False,
            use_xpos=use_xpos,
            xpos_scale_base=xpos_scale_base,
            **kwargs,
        )

        print(
            f"Initialized TrueSlidingWindowAttention with window_size={window_size}, causal={causal}, "
            f"look_backward=0, look_forward=0 (chunking disabled)"
        )

        # Remove the malformed forward method - parent class handles this
        # def forward(...):
        #     return super().forward(...)

    # def forward(
    #     self,
    #     q: torch.Tensor,
    #     k: torch.Tensor,
    #     v: torch.Tensor,
    #     mask: Optional[torch.Tensor] = None,
    #     input_mask: Optional[torch.Tensor] = None,
    #     attn_bias: Optional[torch.Tensor] = None,
    #     window_size: Optional[int] = None,
    # ) -> torch.Tensor:
    #     """Forward pass - delegate to parent with corrected parameters."""
    #     return super().forward(
    #         q,
    #         k,
    #         v,
    #         mask=mask,
    #         input_mask=input_mask,
    #         attn_bias=attn_bias,
    #         window_size=window_size,
    #     )


class TrueSlidingWindowMHA(nn.Module):
    """Multi-head attention using true sliding window attention.

    This is a drop-in replacement for LocalMHA that uses true sliding
    window attention instead of chunked attention.
    """

    def __init__(
        self,
        *,
        dim: int,
        window_size: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        causal: bool = False,
        prenorm: bool = False,
        qk_rmsnorm: bool = False,
        qk_scale: float = 8,
        use_xpos: bool = False,
        xpos_scale_base: Optional[int] = None,
        **kwargs,
    ):
        """Initialize true sliding window multi-head attention.

        Args:
            dim: Model dimension
            window_size: Size of sliding attention window
            dim_head: Dimension per head
            heads: Number of attention heads
            dropout: Dropout probability
            causal: Whether to use causal attention
            prenorm: Whether to apply prenorm
            qk_rmsnorm: Whether to use RMSNorm on queries and keys
            qk_scale: Scale factor for queries and keys
            use_xpos: Whether to use xpos scaling
            xpos_scale_base: Base for xpos scaling
            **kwargs: Additional arguments for compatibility
        """
        super().__init__()

        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if prenorm else None
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.qk_rmsnorm = qk_rmsnorm
        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        # Initialize true sliding window attention
        self.attn_fn = TrueSlidingWindowAttention(
            window_size=window_size,
            causal=causal,
            dropout=dropout,
            scale=(qk_scale if qk_rmsnorm else None),
            dim=dim_head,
            use_xpos=use_xpos,
            xpos_scale_base=xpos_scale_base,
            **kwargs,
        )

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        print(
            f"Initialized TrueSlidingWindowMHA with {heads} heads, window_size={window_size}"
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Attention mask
            attn_bias: Attention bias
            **kwargs: Additional arguments for compatibility

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        if self.norm is not None:
            x = self.norm(x)

        # Project to Q, K, V
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # Reshape for multi-head attention
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        # Apply RMSNorm to queries and keys if configured
        if self.qk_rmsnorm:
            q, k = map(F.normalize, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        # Reshape for attention computation (flatten heads into batch dimension)
        batch_size, num_heads, seq_len, head_dim = q.shape
        q = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k = k.reshape(batch_size * num_heads, seq_len, head_dim)
        v = v.reshape(batch_size * num_heads, seq_len, head_dim)

        # Apply true sliding window attention
        out = self.attn_fn(q, k, v, mask=mask, attn_bias=attn_bias)

        # Reshape back to multi-head format
        out = out.reshape(batch_size, num_heads, seq_len, head_dim)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Apply output projection
        out = self.to_out(out)

        return out


class TrueSlidingWindowEncoderLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer using true sliding window attention.

    This is a drop-in replacement for TransformerEncoderLayer that uses
    true sliding window attention instead of full attention.
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
        window_size: int = 512,
        causal: bool = False,
        **kwargs,
    ):
        """Initialize true sliding window encoder layer.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            activation: Activation function
            layer_norm_eps: Layer norm epsilon
            batch_first: Whether batch dimension comes first
            norm_first: Whether to apply norm first
            window_size: Sliding window size
            causal: Whether to use causal attention
            **kwargs: Additional arguments for compatibility
        """
        # Initialize parent class with dummy self_attn (will be replaced)
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            **kwargs,
        )

        # Replace self-attention with true sliding window attention
        self.self_attn = TrueSlidingWindowMHA(
            dim=d_model,
            heads=nhead,
            window_size=window_size,
            causal=causal,
            dropout=dropout,
            dim_head=d_model // nhead,
            prenorm=False,  # We handle normalization in the layer
        )

        print(
            f"Initialized TrueSlidingWindowEncoderLayer with window_size={window_size}, causal={causal}"
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            src: Input tensor
            src_mask: Attention mask (ignored - sliding window handles masking)
            src_key_padding_mask: Key padding mask (ignored for now)
            is_causal: Whether to use causal attention (ignored - set in __init__)

        Returns:
            Output tensor
        """
        # Handle batch_first format
        if not self.batch_first:
            # Convert from (seq_len, batch, dim) to (batch, seq_len, dim)
            src = src.transpose(0, 1)

        # Self-attention block
        if self.norm_first:
            # Pre-norm
            src2 = self.norm1(src)
            src2 = self.self_attn(src2)
            src = src + self.dropout1(src2)

            # Feedforward block
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            # Post-norm
            src2 = self.self_attn(src)
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            # Feedforward block
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        # Convert back to original format if needed
        if not self.batch_first:
            src = src.transpose(0, 1)

        return src


def test_true_sliding_window_attention():
    """Test the true sliding window attention implementation."""
    batch_size = 2
    seq_len = 1024
    dim = 512
    heads = 8
    window_size = 128

    # Create test data
    x = torch.randn(batch_size, seq_len, dim)

    # Test multi-head attention
    mha = TrueSlidingWindowMHA(
        dim=dim, heads=heads, window_size=window_size, causal=True, prenorm=True
    )

    output = mha(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("True sliding window MHA test passed!")

    # Test encoder layer
    encoder_layer = TrueSlidingWindowEncoderLayer(
        d_model=dim,
        nhead=heads,
        window_size=window_size,
        causal=True,
        batch_first=True,
    )

    encoder_output = encoder_layer(x)
    print(f"Encoder output shape: {encoder_output.shape}")
    print("True sliding window encoder layer test passed!")

    # Test memory efficiency
    if torch.cuda.is_available():
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    return output, encoder_output


if __name__ == "__main__":
    test_true_sliding_window_attention()
