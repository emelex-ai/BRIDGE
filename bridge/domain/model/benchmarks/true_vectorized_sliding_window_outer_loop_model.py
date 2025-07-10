import torch
from torch import nn
from torch.nn import functional as F


class TrueVectorizedSlidingWindowOuterLoopAttention(nn.Module):
    def __init__(self, d_model, nhead, window_size, seq_len):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size
        self.head_dim = d_model // nhead
        self.seq_len = seq_len
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # Precompute indices
        positions = torch.arange(seq_len)
        window_offsets = torch.arange(-window_size + 1, 1)
        key_indices = positions[:, None] + window_offsets[None, :]
        valid_mask = (key_indices >= 0) & (key_indices < seq_len)
        key_indices = torch.clamp(key_indices, 0, seq_len - 1)
        self.register_buffer("key_indices", key_indices)
        self.register_buffer("valid_mask", valid_mask)

    def vectorized_sliding_window_attention_outer_loop(self, q, k, v):
        """Compute sliding window attention using outer loops for batch and head dimensions.

        This implementation uses precomputed sliding window indices and processes
        each batch and head separately, with inner loops for head dimensions.
        This approach trades memory efficiency for computational efficiency.

        Args:
            q: Query tensor of shape [batch_size, nhead, seq_len, head_dim]
            k: Key tensor of shape [batch_size, nhead, seq_len, head_dim]
            v: Value tensor of shape [batch_size, nhead, seq_len, head_dim]

        Returns:
            Attention output tensor of shape [batch_size, nhead, seq_len, head_dim]

        """
        batch_size, nhead, seq_len, head_dim = q.shape
        output = torch.zeros_like(q)
        for b in range(batch_size):
            for h in range(nhead):
                # [seq_len, window_size, head_dim]
                k_windows = k[b, h, self.key_indices, :]
                v_windows = v[b, h, self.key_indices, :]
                for d in range(head_dim):
                    k_slice = k_windows[:, :, d]
                    v_slice = v_windows[:, :, d]
                    q_slice = q[b, h, :, d].unsqueeze(-1)  # [seq_len, 1]
                    scores = (q_slice * k_slice) / (head_dim**0.5)
                    scores = torch.where(
                        self.valid_mask, scores, torch.full_like(scores, float("-inf"))
                    )
                    attn_weights = F.softmax(scores, dim=-1)
                    output[b, h, :, d] = (attn_weights * v_slice).sum(dim=-1)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with outer loop vectorized sliding window attention.

        This method processes the input through linear projections, reshapes for
        multi-head attention, applies sliding window attention using outer loops
        for batch and head dimensions, and projects the output.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]

        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V: [batch_size, seq_len, d_model]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: [batch_size, nhead, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Apply vectorized sliding window attention with outer loops
        attn_output = self.vectorized_sliding_window_attention_outer_loop(q, k, v)

        # Reshape back: [batch_size, seq_len, d_model]
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        # Output projection: [batch_size, seq_len, d_model]
        output = self.out_proj(attn_output)

        return output

if __name__ == "__main__":
    pass
