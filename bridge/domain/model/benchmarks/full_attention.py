
    class SDPAFullAttentionLayer(nn.Module):
        """Custom layer using SDPA for full attention with precomputed mask."""

        def __init__(self, d_model, nhead):
            super().__init__()
            self.d_model = d_model
            self.nhead = nhead
            self.head_dim = d_model // nhead

            # Linear projections
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

            # Layer norm and feedforward (to match TransformerEncoderLayer)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.feedforward = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model),
            )

            # Precomputed mask (None for full attention)
            # This is useful for future assignment or for consistent model structure,
            # even if the buffer is not currently used.
            self.register_buffer("attn_mask", None)

        def forward(self, x):
            # Self-attention with residual connection
            residual = x
            x = self.norm1(x)

            # Project to Q, K, V
            B, S, D = x.shape
            q = self.q_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)

            # SDPA full attention (no masking = full O(n²) attention)
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,  # Explicit None for full attention
            )

            # Reshape and project
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
            attn_output = self.out_proj(attn_output)

            # Add residual
            x = residual + attn_output

            # Feedforward with residual connection
            residual = x
            x = self.norm2(x)
            x = residual + self.feedforward(x)

            return x

    class SDPAFullAttentionModel(nn.Module):
        def __init__(self, d_model, nhead, num_layers):
            super().__init__()
            self.layers = nn.ModuleList(
                [SDPAFullAttentionLayer(d_model, nhead) for _ in range(num_layers)]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
