import torch
import torch.nn as nn
import vllm_flash_attn
from torch.nn import TransformerEncoderLayer


class FlashAttentionEncoderLayer(TransformerEncoderLayer):
    """TransformerEncoderLayer that uses Flash Attention instead of standard attention.

    This subclass replaces the standard multi-head attention with Flash Attention
    for improved memory efficiency and speed.
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
    ):
        """Initialize FlashAttentionEncoderLayer.

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
        print(f"Enter FlashAttentionEncoderLayer initialized with {kwargs}")

        # Store parameters for Flash Attention
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout
        self.batch_first = batch_first

        # Replace the self-attention with Flash Attention compatible projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(
            d_model, d_model, bias=bias, device=device, dtype=dtype
        )

    def _flash_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass using Flash Attention.

        Args:
            query: Query tensor of shape (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)
            key: Key tensor of shape (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)
            value: Value tensor of shape (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)
            key_padding_mask: Padding mask for keys
            attn_mask: Attention mask

        Returns:
            Output tensor with same shape as query
        """
        # Store original dtype for conversion back
        original_dtype = query.dtype

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

        # Convert to half precision for Flash Attention
        if q.dtype == torch.float32:
            q = q.half()  # Convert to fp16
            k = k.half()
            v = v.half()

        # Handle causal mask (if attn_mask is provided)
        # GE: Why perform this check?
        # What about for windowed transform?
        causal = False
        if attn_mask is not None:
            # Check if it's a causal mask (lower triangular)
            if attn_mask.shape[-2:] == (seq_len, seq_len):
                causal = torch.allclose(
                    attn_mask, torch.tril(torch.ones_like(attn_mask))
                )

        # Use Flash Attention
        if self.training:
            dropout_p = self.dropout_p
        else:
            dropout_p = 0.0

        # Call vllm_flash_attn function
        output = vllm_flash_attn.flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            causal=causal,
            softmax_scale=None,  # Use default 1/sqrt(head_dim)
        )

        # Convert back to original dtype
        if original_dtype == torch.float32:
            output = output.float()

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
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the FlashAttentionEncoderLayer.

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
            attn_output = self._flash_attention_forward(
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
            attn_output = self._flash_attention_forward(
                src, src, src, key_padding_mask=src_key_padding_mask, attn_mask=src_mask
            )
            src = self.norm1(src + self.dropout1(attn_output))

            # Post-norm: FFN -> Residual -> LayerNorm
            ffn_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(ffn_output))

        return src


# Example usage in your model
class FlashAttentionEncoder(nn.Module):
    """Transformer Encoder using Flash Attention layers.

    This encoder uses FlashAttentionEncoderLayer instead of standard
    TransformerEncoderLayer for better performance.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        """Initialize FlashAttentionEncoder.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: FFN dimension
            dropout: Dropout probability
            batch_first: Whether to use batch_first format
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                FlashAttentionEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=batch_first,
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
    # Test the Flash Attention encoder
    batch_size, seq_len, d_model = 4, 128, 512
    nhead = 8
    num_layers = 6

    # Add these lines before creating the encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("ERROR: FlashAttention requires CUDA")
        exit(1)

    # Create encoder
    encoder = FlashAttentionEncoder(
        d_model=d_model, nhead=nhead, num_layers=num_layers, batch_first=True
    )

    # Move everything to GPU
    encoder = encoder.to(device)

    # Test input
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Forward pass
    output = encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Flash Attention Encoder test passed!")

    # Test individual layer
    layer = FlashAttentionEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
    layer = layer.to(device)  # Move layer to GPU
    layer_output = layer(x)
    print(f"Layer output shape: {layer_output.shape}")
    print("Flash Attention Layer test passed!")
