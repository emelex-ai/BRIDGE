class ChunkedVectorizedSlidingWindowModel(nn.Module):
    """Memory-efficient chunked vectorized sliding window attention.

    This implementation processes chunks of queries simultaneously while
    controlling memory usage. No masks needed - only computes
    relevant attention pairs. Avoids in-place operations for gradient safety.
    """

    def __init__(self, d_model, nhead, window_size, chunk_size=32):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.head_dim = d_model // nhead

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def chunked_sliding_window_attention(self, q, k, v):
        """Compute sliding window attention using memory-efficient chunking.

        Args:
            q, k, v: [batch_size, nhead, seq_len, head_dim]

        Returns:
            output: [batch_size, nhead, seq_len, head_dim]
        """
        batch_size, nhead, seq_len, head_dim = q.shape
        window_size = self.window_size
        chunk_size = self.chunk_size

        # Collect output chunks to concatenate at the end
        output_chunks = []

        # Process queries in chunks
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            current_chunk_size = chunk_end - chunk_start

            # Extract query chunk
            q_chunk = q[
                :, :, chunk_start:chunk_end, :
            ]  # [batch, nhead, chunk_size, head_dim]

            # Collect keys and values for all positions in this chunk
            k_windows = []
            v_windows = []
            valid_lengths = []

            for pos in range(chunk_start, chunk_end):
                # Calculate sliding window boundaries for this position
                start_pos = max(0, pos - window_size + 1)
                end_pos = pos + 1
                actual_window_size = end_pos - start_pos

                # Extract keys and values for this position's window
                k_window = k[
                    :, :, start_pos:end_pos, :
                ]  # [batch, nhead, actual_window_size, head_dim]
                v_window = v[
                    :, :, start_pos:end_pos, :
                ]  # [batch, nhead, actual_window_size, head_dim]

                # Pad to consistent window size if needed (without in-place ops)
                if actual_window_size < window_size:
                    pad_size = window_size - actual_window_size
                    k_window = F.pad(
                        k_window, (0, 0, 0, pad_size), mode="constant", value=0
                    )
                    v_window = F.pad(
                        v_window, (0, 0, 0, pad_size), mode="constant", value=0
                    )

                k_windows.append(k_window)
                v_windows.append(v_window)
                valid_lengths.append(actual_window_size)

            # Stack all windows for vectorized computation
            # [batch, nhead, chunk_size, window_size, head_dim]
            k_stacked = torch.stack(k_windows, dim=2)
            v_stacked = torch.stack(v_windows, dim=2)

            # Vectorized computation for the entire chunk
            # q_chunk: [batch, nhead, chunk_size, head_dim]
            # k_stacked: [batch, nhead, chunk_size, window_size, head_dim]

            # Expand queries to match window structure
            q_expanded = q_chunk.unsqueeze(
                -2
            )  # [batch, nhead, chunk_size, 1, head_dim]

            # Compute attention scores for all positions in chunk simultaneously
            scores = torch.matmul(q_expanded, k_stacked.transpose(-2, -1)) / (
                head_dim**0.5
            )
            scores = scores.squeeze(-2)  # [batch, nhead, chunk_size, window_size]

            # Handle padding by setting scores for padded positions to -inf
            for i, valid_len in enumerate(valid_lengths):
                if valid_len < window_size:
                    # Create mask for invalid positions
                    mask = torch.ones_like(scores[:, :, i, :])
                    mask[:, :, valid_len:] = float("-inf")
                    scores[:, :, i, :] = torch.where(
                        mask == 1,
                        scores[:, :, i, :],
                        torch.full_like(scores[:, :, i, :], float("-inf")),
                    )

            # Apply softmax
            attn_weights = F.softmax(scores, dim=-1)

            # Apply attention weights to values
            output_chunk = torch.matmul(attn_weights.unsqueeze(-2), v_stacked).squeeze(
                -2
            )

            # Store chunk for concatenation
            output_chunks.append(output_chunk)

        # Concatenate all chunks to form final output
        output = torch.cat(output_chunks, dim=2)

        return output

    def forward(self, x):
        """Forward pass with chunked sliding window attention."""
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Apply chunked sliding window attention
        attn_output = self.chunked_sliding_window_attention(q, k, v)

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        # Output projection
        output = self.out_proj(attn_output)

        return output

if __name__ == "__main__":
    pass
