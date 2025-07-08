import torch
from torch import nn
from torch.nn import functional as F


class TrueVectorizedSlidingWindowModel(nn.Module):
    """Fully vectorized sliding window attention - no loops, no conditionals.

    This implementation uses advanced tensor indexing and broadcasting to create
    all sliding windows simultaneously, achieving true O(n×w) complexity with
    pure vectorized operations.
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

    def create_sliding_window_indices(self, seq_len, device):
        """Create indices for vectorized sliding window gathering.

        Returns all sliding window indices in one vectorized operation.

        Args:
            seq_len: Sequence length
            device: Device for tensors

        Returns:
            key_indices: [seq_len, window_size] - indices for gathering keys/values
            valid_mask: [seq_len, window_size] - mask for valid positions
        """
        # Create base indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device)  # [seq_len]

        # Create window offsets: [-window_size+1, -window_size+2, ..., 0]
        window_offsets = torch.arange(
            -self.window_size + 1, 1, device=device
        )  # [window_size]

        # Broadcast to create all sliding window indices simultaneously
        # positions[:, None] + window_offsets[None, :] creates [seq_len, window_size]
        key_indices = (
            positions[:, None] + window_offsets[None, :]
        )  # [seq_len, window_size]

        # Create validity mask (handles boundary conditions)
        valid_mask = (key_indices >= 0) & (
            key_indices < seq_len
        )  # [seq_len, window_size]

        # Clamp indices to valid range for safe gathering
        key_indices = torch.clamp(key_indices, 0, seq_len - 1)  # [seq_len, window_size]

        return key_indices, valid_mask

    def vectorized_sliding_window_attention(self, q, k, v):
        """Compute sliding window attention using pure vectorized operations.

        No loops, no conditionals - everything is done with tensor operations.

        Args:
            q, k, v: [batch_size, nhead, seq_len, head_dim]

        Returns:
            output: [batch_size, nhead, seq_len, head_dim]
        """
        batch_size, nhead, seq_len, head_dim = q.shape

        # Create sliding window indices (vectorized)
        key_indices, valid_mask = self.create_sliding_window_indices(seq_len, q.device)

        # Gather all sliding windows simultaneously using advanced indexing
        # key_indices: [seq_len, window_size]
        # k: [batch_size, nhead, seq_len, head_dim]
        # Result: [batch_size, nhead, seq_len, window_size, head_dim]

        # Expand k and v for gathering: [batch_size, nhead, seq_len, head_dim]
        # Use advanced indexing to gather sliding windows
        k_windows = k[
            :, :, key_indices, :
        ]  # [batch_size, nhead, seq_len, window_size, head_dim]
        v_windows = v[
            :, :, key_indices, :
        ]  # [batch_size, nhead, seq_len, window_size, head_dim]

        # Compute attention scores for all windows simultaneously
        # q: [batch_size, nhead, seq_len, head_dim] -> [batch_size, nhead, seq_len, 1, head_dim]
        q_expanded = q.unsqueeze(-2)  # [batch_size, nhead, seq_len, 1, head_dim]

        # Compute scores: q @ k^T for all windows
        # q_expanded: [batch_size, nhead, seq_len, 1, head_dim]
        # k_windows: [batch_size, nhead, seq_len, window_size, head_dim]
        scores = torch.matmul(q_expanded, k_windows.transpose(-2, -1)) / (head_dim**0.5)
        scores = scores.squeeze(-2)  # [batch_size, nhead, seq_len, window_size]

        # Apply validity mask (vectorized - no loops!)
        # valid_mask: [seq_len, window_size] -> [1, 1, seq_len, window_size]
        mask_expanded = valid_mask[
            None, None, :, :
        ]  # Broadcast for batch and head dims
        scores = torch.where(
            mask_expanded, scores, torch.full_like(scores, float("-inf"))
        )

        # Apply softmax
        attn_weights = F.softmax(
            scores, dim=-1
        )  # [batch_size, nhead, seq_len, window_size]

        # Apply attention weights to values (vectorized)
        # attn_weights: [batch_size, nhead, seq_len, window_size] -> [batch_size, nhead, seq_len, 1, window_size]
        # v_windows: [batch_size, nhead, seq_len, window_size, head_dim]
        # Result: [batch_size, nhead, seq_len, 1, head_dim] -> [batch_size, nhead, seq_len, head_dim]
        output = torch.matmul(attn_weights.unsqueeze(-2), v_windows).squeeze(-2)

        return output

    def forward(self, x):
        """Forward pass with fully vectorized sliding window attention."""
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Apply vectorized sliding window attention
        attn_output = self.vectorized_sliding_window_attention(q, k, v)

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        # Output projection
        output = self.out_proj(attn_output)

        return output
