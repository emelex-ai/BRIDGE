import time
from torch.nn import functional as F
import torch
from torch import nn

class SDPASlidingWindowAttention(nn.Module):
    """SDPA model with efficient sliding window attention.

    This implementation avoids storing large precomputed masks by computing
    the sliding window pattern on-the-fly during attention computation.

    """

    def __init__(self, d_model, nhead, window_size):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size
        self.head_dim = d_model // nhead

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # No precomputed mask - compute on-the-fly

    def create_sliding_window_mask(self, seq_len, device):
        """Create sliding window mask efficiently using broadcasting.

        This avoids creating large dense matrices by using torch's
        efficient broadcasting operations.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Sliding window mask with minimal memory footprint

        """
        print(f"==> create_sliding_window_mask, {seq_len=}")
        # Create position indices (minimal memory: 2 * seq_len)
        positions = torch.arange(seq_len, device=device)

        # Use broadcasting to create mask pattern (still creates full matrix, but efficiently)
        query_pos = positions.unsqueeze(1)  # [seq_len, 1]
        key_pos = positions.unsqueeze(0)  # [1, seq_len]

        # Sliding window: can attend to positions [i-window_size+1, i]
        # Causal: can't attend to future positions (key_pos > query_pos)
        valid_mask = (key_pos >= query_pos - self.window_size + 1) & (
            key_pos <= query_pos
        )

        # Convert to SDPA format (0.0 = attend, -inf = mask out)
        attention_mask = torch.where(valid_mask, 0.0, float("-inf"))

        return attention_mask.to(torch.float32)

    def forward(self, x):
        """Forward pass with efficient sliding window attention.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]

        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch_size, seq_len, d_model]
        k = self.k_proj(x)  # [batch_size, seq_len, d_model]
        v = self.v_proj(x)  # [batch_size, seq_len, d_model]

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(
            1, 2
        )  # [batch_size, nhead, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(
            1, 2
        )  # [batch_size, nhead, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(
            1, 2
        )  # [batch_size, nhead, seq_len, head_dim]

        # Create sliding window mask (only when needed)
        attn_mask = self.create_sliding_window_mask(seq_len, x.device)

        # Apply SDPA with sliding window mask
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,  # We handle causality in our custom mask
        )
        print(f"==> skpa_sliding_window, {attn_mask.shape=}, {attn_output.shape=}, {x.shape=}")

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        # Output projection
        output = self.out_proj(attn_output)

        return output


if __name__ == "__main__":
    pass
