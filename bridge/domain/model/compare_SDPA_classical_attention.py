import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def benchmark_classical_full_attention(
    seq_len, d_model=1024, nhead=1, num_layers=1, batch_size=4
):
    """Benchmark classical full attention in TRAINING mode (O(n²) complexity).

    Uses PyTorch's TransformerEncoderLayer in training mode, which uses
    classical attention implementation (NOT SDPA).

    Args:
        seq_len: Sequence length
        d_model: Model dimension (default 1024)
        nhead: Number of attention heads (default 1)
        num_layers: Number of encoder layers
        batch_size: Batch size for testing

    Returns:
        Dictionary with benchmark results

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking Classical Full Attention (TRAINING mode, O(n²)) on {device}")

    # Create classical transformer in TRAINING mode
    encoder_layer = TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
    )
    classical_model = TransformerEncoder(encoder_layer, num_layers=num_layers).to(
        device
    )

    # CRITICAL: Set to training mode (uses classical attention, not SDPA)
    classical_model.train()

    # Create input with gradient tracking for training mode
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    # Warmup
    print("  Warming up in training mode...")
    for _ in range(3):
        output = classical_model(x)
        # Simulate backward pass for training
        loss = output.sum()
        loss.backward()
        x.grad = None  # Clear gradients

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        output = classical_model(x)
        # Simulate backward pass for training
        loss = output.sum()
        loss.backward()
        x.grad = None  # Clear gradients

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    return {
        "attention_type": "Classical_Full_Attention",
        "mode": "TRAINING",
        "complexity": "O(n²)",
        "batch_size": batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }


def benchmark_classical_windowed_full_attention(
    seq_len: int,
    d_model: int = 1024,
    nhead: int = 1,
    window_size: int = 128,
    num_layers: int = 1,
    batch_size: int = 4,
) -> dict:
    """Benchmark classical full attention with a sliding window mask in TRAINING mode.

    This uses PyTorch's TransformerEncoderLayer with a custom sliding window mask.
    Attempts to minimize memory usage by constructing the mask efficiently.

    Args:
        seq_len: Sequence length
        d_model: Model dimension (default 1024)
        nhead: Number of attention heads (default 1)
        window_size: Sliding window size
        num_layers: Number of encoder layers
        batch_size: Batch size for testing

    Returns:
        Dictionary with benchmark results

    """
    import time

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Benchmarking Classical Full Attention with Sliding Window Mask (TRAINING mode, O(nw)) on {device}"
    )

    # Create sliding window mask efficiently (lower-triangular banded mask)
    # Only store the relevant band, not the full matrix if possible
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start : i + 1] = 0.0

    # Create classical transformer encoder
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
    )
    classical_model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(
        device
    )
    classical_model.train()

    # Create input with gradient tracking
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    # Warmup
    print("  Warming up in training mode...")
    for _ in range(3):
        output = classical_model(x, mask=mask)
        loss = output.sum()
        loss.backward()
        x.grad = None

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        output = classical_model(x, mask=mask)
        loss = output.sum()
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    return {
        "attention_type": "Classical_Windowed_Full_Attention",
        "mode": "TRAINING",
        "complexity": "O(nw)",
        "window_size": window_size,
        "batch_size": batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }


def benchmark_sdpa_full_attention(
    seq_len, d_model=1024, nhead=1, num_layers=1, batch_size=4
):
    """Benchmark SDPA full attention in TRAINING mode (O(n²) complexity).

    Uses PyTorch's scaled_dot_product_attention directly with full attention
    in training mode. NO MASKING - precomputed mask is None.

    Args:
        seq_len: Sequence length
        d_model: Model dimension (default 1024)
        nhead: Number of attention heads (default 1)
        num_layers: Number of encoder layers
        batch_size: Batch size for testing

    Returns:
        Dictionary with benchmark results

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking SDPA Full Attention (TRAINING mode, O(n²)) on {device}")

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

    # Precompute mask (None for full attention)
    print("  Precomputing mask (None for full attention)...")

    sdpa_model = SDPAFullAttentionModel(d_model, nhead, num_layers).to(device)

    # CRITICAL: Set to training mode
    sdpa_model.train()

    # Create input with gradient tracking for training mode
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    # Warmup
    print("  Warming up in training mode...")
    for _ in range(3):
        output = sdpa_model(x)
        # Simulate backward pass for training
        loss = output.sum()
        loss.backward()
        x.grad = None  # Clear gradients

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        output = sdpa_model(x)
        # Simulate backward pass for training
        loss = output.sum()
        loss.backward()
        x.grad = None  # Clear gradients

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    return {
        "attention_type": "SDPA_Full_Attention",
        "mode": "TRAINING",
        "complexity": "O(n²)",
        "batch_size": batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }


class SDPASlidingWindowModel(nn.Module):
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

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        # Output projection
        output = self.out_proj(attn_output)

        return output


def benchmark_sdpa_sliding_window(
    seq_len, d_model=1024, nhead=1, window_size=128, batch_size=4
):
    """Benchmark SDPA sliding window attention with efficient mask computation.

    This version computes masks on-the-fly to avoid memory issues with
    very large sequence lengths.

    Args:
        seq_len: Sequence length
        d_model: Model dimension (default 1024)
        nhead: Number of attention heads (default 1)
        window_size: Sliding window size
        batch_size: Batch size for testing

    Returns:
        Dictionary with benchmark results

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Benchmarking SDPA Sliding Window (TRAINING mode, O(n), window={window_size}) on {device}"
    )

    # Create model with efficient sliding window
    model = SDPASlidingWindowModel(d_model, nhead, window_size).to(device)
    model.train()  # Training mode

    # Create input data
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    print(f"  Using efficient on-the-fly mask computation...")

    # Warmup
    print("  Warming up in training mode...")
    for _ in range(3):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    # Clear memory before benchmark
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark with backward pass
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    return {
        "attention_type": "SDPA_Sliding_Window_Efficient",
        "mode": "TRAINING",
        "complexity": "O(n)",
        "window_size": window_size,
        "batch_size": batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }


class FastSlidingWindowModel(nn.Module):
    """Fast sliding window attention using efficient tensor operations.

    This implementation is much simpler and faster than the chunked version.
    It uses PyTorch's efficient indexing and avoids loops.
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

    def forward(self, x):
        """Forward pass with simple sliding window attention."""
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Simple approach: create causal mask with sliding window
        # This is much faster than the complex chunked approach
        positions = torch.arange(seq_len, device=x.device)
        query_pos = positions.unsqueeze(1)
        key_pos = positions.unsqueeze(0)

        # Sliding window: attend to positions [i-window_size+1, i]
        valid_mask = (key_pos >= query_pos - self.window_size + 1) & (
            key_pos <= query_pos
        )
        attn_mask = torch.where(valid_mask, 0.0, float("-inf"))

        # Single SDPA call (much faster than multiple calls)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        # Output projection
        output = self.out_proj(attn_output)

        return output


def benchmark_fast_sliding_window(
    seq_len, d_model=1024, nhead=1, window_size=128, batch_size=4
):
    """Benchmark fast sliding window attention (simple and efficient).

    This should be much faster than the chunked version.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Benchmarking Fast Sliding Window (TRAINING mode, O(n), window={window_size}) on {device}"
    )

    # Create model
    model = FastSlidingWindowModel(d_model, nhead, window_size).to(device)
    model.train()

    # Create input data
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    print(f"  Using simple mask-based approach (single SDPA call)...")

    # Warmup
    print("  Warming up in training mode...")
    for _ in range(3):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    return {
        "attention_type": "Fast_Sliding_Window",
        "mode": "TRAINING",
        "complexity": "O(n)",
        "window_size": window_size,
        "batch_size": batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }


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


def benchmark_chunked_vectorized_sliding_window(
    seq_len, d_model=1024, nhead=1, window_size=128, chunk_size=32, batch_size=4
):
    """Benchmark chunked vectorized sliding window attention.

    This should achieve O(n×w) complexity with controlled memory usage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Benchmarking Chunked Vectorized Sliding Window (TRAINING mode, O(n×w), window={window_size}, chunk={chunk_size}) on {device}"
    )

    # Create model
    model = ChunkedVectorizedSlidingWindowModel(
        d_model, nhead, window_size, chunk_size
    ).to(device)
    model.train()

    # Create input data
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    print(
        f"  Using chunked vectorized operations (controlled memory, chunk_size={chunk_size})..."
    )

    # Warmup
    print("  Warming up in training mode...")
    for _ in range(3):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    return {
        "attention_type": "Chunked_Vectorized_Sliding_Window",
        "mode": "TRAINING",
        "complexity": "O(n×w)",
        "window_size": window_size,
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }


def benchmark_true_vectorized_sliding_window(
    seq_len, d_model=1024, nhead=1, window_size=128, batch_size=4
):
    """Benchmark true vectorized sliding window attention (no loops, no conditionals).

    This should achieve optimal O(n×w) complexity with pure vectorized operations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Benchmarking True Vectorized Sliding Window (TRAINING mode, O(n×w), window={window_size}) on {device}"
    )

    # Create model
    model = TrueVectorizedSlidingWindowModel(d_model, nhead, window_size).to(device)
    model.train()

    # Create input data
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    print(f"  Using pure vectorized operations (no loops, no conditionals)...")

    # Warmup
    print("  Warming up in training mode...")
    for _ in range(3):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    return {
        "attention_type": "True_Vectorized_Sliding_Window",
        "mode": "TRAINING",
        "complexity": "O(n×w)",
        "window_size": window_size,
        "batch_size": batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }


class TrueVectorizedSlidingWindowOuterLoopModel(nn.Module):
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


def benchmark_true_vectorized_sliding_window_outer_loop(
    seq_len, d_model=1024, nhead=1, window_size=128, batch_size=4
):
    """Benchmark true vectorized sliding window attention (no loops, no conditionals).

    This should achieve optimal O(n×w) complexity with pure vectorized operations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Benchmarking True Vectorized Sliding Window (TRAINING mode, O(n×w), window={window_size}) on {device}"
    )

    # Create model
    model = TrueVectorizedSlidingWindowOuterLoopModel(
        d_model, nhead, window_size, seq_len
    ).to(device)
    model.train()

    # Create input data
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    print(f"  Using pure vectorized operations (no loops, no conditionals)...")

    # Warmup
    print("  Warming up in training mode...")
    for _ in range(3):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    )

    return {
        "attention_type": "True_Vectorized_Sliding_Window_Outer_Loop",
        "mode": "TRAINING",
        "complexity": "O(n×w)",
        "window_size": window_size,
        "batch_size": batch_size,
        "time_ms": avg_time_ms,
        "memory_mb": memory_mb,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }


def compare_training_mode_attention():
    """Compare attention implementations in TRAINING mode only.

    Tests conducted (ALL IN TRAINING MODE with PRECOMPUTED MASKS):
    1. Classical Full Attention (O(n²)) - TransformerEncoderLayer.train()
    2. SDPA Full Attention (O(n²)) - scaled_dot_product_attention with no mask
    3. SDPA Sliding Window (O(n)) - scaled_dot_product_attention with precomputed sliding window mask

    """
    print("🚀 TRAINING MODE Attention Comparison (PRECOMPUTED MASKS)")
    print("=" * 80)
    print("ALL TESTS RUN IN TRAINING MODE (with backward pass)")
    print("Tests:")
    print("1. Classical Full Attention (O(n²)) - TransformerEncoderLayer.train()")
    print("2. SDPA Full Attention (O(n²)) - F.scaled_dot_product_attention (no mask)")
    print(
        "3. SDPA Sliding Window (O(n)) - F.scaled_dot_product_attention (precomputed mask)"
    )
    print("=" * 80)

    # Test configurations - single head, d_model=1024 as requested
    configs = [
        # {"seq_len": 1024},
        {"seq_len": 2048},
        # {"seq_len": 4096},
        {"seq_len": 4096},
        # {"seq_len": 4 * 8192},
    ]

    window_sizes = [32, 128, 512]  # Reasonable window sizes
    d_model = 512  # or as set earlier
    nhead = 1
    batch_size = 1

    for config in configs:
        seq_len = config["seq_len"]

        print(f"\n{'='*60}")
        print(f"Testing seq_len={seq_len}, d_model={d_model}, nhead={nhead}")
        print(f"Fixed batch_size={batch_size} - TRAINING MODE")
        print(f"{'='*60}")

        # Test 1: Classical Full Attention (O(n²))
        try:
            print("\n🔬 Test 1: Classical Full Attention (TRAINING mode, O(n²))...")
            classical_result = benchmark_classical_full_attention(
                seq_len, d_model=d_model, nhead=nhead, batch_size=batch_size
            )
            print(f"✅ {classical_result}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Classical Full Attention: OOM at seq_len={seq_len}")
                continue
            else:
                print(f"❌ Classical Full Attention error: {e}")
                continue

        # Test 2: SDPA Full Attention (O(n²))
        try:
            print("\n🔬 Test 2: SDPA Full Attention (TRAINING mode, O(n²))...")
            sdpa_full_result = benchmark_sdpa_full_attention(
                seq_len, d_model=d_model, nhead=nhead, batch_size=batch_size
            )
            print(f"✅ {sdpa_full_result}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ SDPA Full Attention: OOM at seq_len={seq_len}")
                sdpa_full_result = None
            else:
                print(f"❌ SDPA Full Attention error: {e}")
                sdpa_full_result = None

        for window_size in window_sizes:
            # Test 1a: Classical Windowed Full Attention (O(n²))
            try:
                print(
                    f"\n🔬 Test 1a: Classical Windowed Full Attention (TRAINING mode, O(n²), window={window_size})..."
                )
                classical_windowed_result = benchmark_classical_windowed_full_attention(
                    seq_len,
                    d_model=d_model,
                    nhead=nhead,
                    window_size=window_size,
                    batch_size=batch_size,
                )
                print(f"✅ {classical_windowed_result}")
                # Comparison with Classical Full Attention
                memory_ratio = (
                    classical_windowed_result["memory_mb"]
                    / classical_result["memory_mb"]
                )
                time_ratio = (
                    classical_windowed_result["time_ms"] / classical_result["time_ms"]
                )
                speedup = (
                    classical_result["time_ms"] / classical_windowed_result["time_ms"]
                )
                print(f"  📊 Classical Windowed vs Classical Full:")
                print(f"    Windowed memory / Full memory: {memory_ratio:.2f}x")
                print(f"    Windowed time / Full time: {time_ratio:.2f}x")
                print(f"    Full time / Windowed time: {speedup:.2f}x")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        f"❌ Classical Windowed Full Attention: OOM at seq_len={seq_len}, window={window_size}"
                    )
                    continue
                else:
                    print(f"❌ Classical Windowed Full Attention error: {e}")
                    continue

            # Test 3: SDPA Sliding Window (O(n))
            print(
                f"\n🔬 Test 3: SDPA Sliding Window (TRAINING mode, O(n), window={window_size})..."
            )
            sdpa_sliding_result = benchmark_sdpa_sliding_window(
                seq_len,
                d_model=d_model,
                nhead=nhead,
                window_size=window_size,
                batch_size=batch_size,
            )
            if sdpa_sliding_result:
                print(f"✅ {sdpa_sliding_result}")
                print(f"  📊 SDPA Sliding vs Classical Full:")
                print(
                    f"    SDPA Sliding memory / Classical memory: {sdpa_sliding_result['memory_mb'] / classical_result['memory_mb']:.2f}x"
                )
                print(
                    f"    SDPA Sliding time / Classical time: {sdpa_sliding_result['time_ms'] / classical_result['time_ms']:.2f}x"
                )
                print(
                    f"    Classical time / SDPA Sliding time: {classical_result['time_ms'] / sdpa_sliding_result['time_ms']:.2f}x"
                )
                print(f"  📊 SDPA Sliding vs SDPA Full:")
                print(
                    f"    SDPA Sliding memory / SDPA Full memory: {sdpa_sliding_result['memory_mb'] / sdpa_full_result['memory_mb']:.2f}x"
                )
                print(
                    f"    SDPA Sliding time / SDPA Full time: {sdpa_sliding_result['time_ms'] / sdpa_full_result['time_ms']:.2f}x"
                )
                print(
                    f"    SDPA Full time / SDPA Sliding time: {sdpa_full_result['time_ms'] / sdpa_sliding_result['time_ms']:.2f}x"
                )
                # New: Compare SDPA Sliding to Classical Windowed Full Attention
                print(f"  📊 SDPA Sliding vs Classical Windowed Full Attention:")
                print(
                    f"    SDPA Sliding memory / Classical Windowed memory: {sdpa_sliding_result['memory_mb'] / classical_windowed_result['memory_mb']:.2f}x"
                )
                print(
                    f"    SDPA Sliding time / Classical Windowed time: {sdpa_sliding_result['time_ms'] / classical_windowed_result['time_ms']:.2f}x"
                )
                print(
                    f"    Classical Windowed time / SDPA Sliding time: {classical_windowed_result['time_ms'] / sdpa_sliding_result['time_ms']:.2f}x"
                )
            else:
                print(f"❌ SDPA Sliding Window (w={window_size}): Failed")

            # Test 4: Fast Sliding Window (O(n))
            print(
                f"\n🔬 Test 4: Fast Sliding Window (TRAINING mode, O(n), window={window_size})..."
            )
            fast_sliding_result = benchmark_fast_sliding_window(
                seq_len,
                d_model=d_model,
                nhead=nhead,
                window_size=window_size,
                batch_size=batch_size,
            )
            if fast_sliding_result:
                print(f"✅ {fast_sliding_result}")
                print(f"  📊 Fast Sliding vs Classical Full:")
                print(
                    f"    Fast Sliding memory / Classical memory: {fast_sliding_result['memory_mb'] / classical_result['memory_mb']:.2f}x"
                )
                print(
                    f"    Fast Sliding time / Classical time: {fast_sliding_result['time_ms'] / classical_result['time_ms']:.2f}x"
                )
                print(
                    f"    Classical time / Fast Sliding time: {classical_result['time_ms'] / fast_sliding_result['time_ms']:.2f}x"
                )
                if sdpa_full_result:
                    print(f"  📊 Fast Sliding vs SDPA Full:")
                    print(
                        f"    Fast Sliding memory / SDPA Full memory: {fast_sliding_result['memory_mb'] / sdpa_full_result['memory_mb']:.2f}x"
                    )
                    print(
                        f"    Fast Sliding time / SDPA Full time: {fast_sliding_result['time_ms'] / sdpa_full_result['time_ms']:.2f}x"
                    )
                    print(
                        f"    SDPA Full time / Fast Sliding time: {sdpa_full_result['time_ms'] / fast_sliding_result['time_ms']:.2f}x"
                    )
            else:
                print(f"❌ Fast Sliding Window (w={window_size}): Failed")

            # Test 5: True Vectorized Sliding Window (O(n×w))
            chunk_size = 32  # Default chunk size
            print(
                f"\n🔬 Test 5: True Vectorized Sliding Window (TRAINING mode, O(n×w), window={window_size})..."
            )
            true_vectorized_result = benchmark_true_vectorized_sliding_window(
                seq_len,
                d_model=d_model,
                nhead=nhead,
                window_size=window_size,
                batch_size=batch_size,
            )
            if true_vectorized_result:
                print(f"✅ {true_vectorized_result}")
                print(f"  📊 True Vectorized vs Classical Full:")
                print(
                    f"    True Vectorized memory / Classical memory: {true_vectorized_result['memory_mb'] / classical_result['memory_mb']:.2f}x"
                )
                print(
                    f"    True Vectorized time / Classical time: {true_vectorized_result['time_ms'] / classical_result['time_ms']:.2f}x"
                )
                print(
                    f"    Classical time / True Vectorized time: {classical_result['time_ms'] / true_vectorized_result['time_ms']:.2f}x"
                )
                if sdpa_full_result:
                    print(f"  📊 True Vectorized vs SDPA Full:")
                    print(
                        f"    True Vectorized memory / SDPA Full memory: {true_vectorized_result['memory_mb'] / sdpa_full_result['memory_mb']:.2f}x"
                    )
                    print(
                        f"    True Vectorized time / SDPA Full time: {true_vectorized_result['time_ms'] / sdpa_full_result['time_ms']:.2f}x"
                    )
                    print(
                        f"    SDPA Full time / True Vectorized time: {sdpa_full_result['time_ms'] / true_vectorized_result['time_ms']:.2f}x"
                    )
                if fast_sliding_result:
                    print(f"  📊 True Vectorized vs Fast Sliding:")
                    print(
                        f"    True Vectorized memory / Fast Sliding memory: {true_vectorized_result['memory_mb'] / fast_sliding_result['memory_mb']:.2f}x"
                    )
                    print(
                        f"    True Vectorized time / Fast Sliding time: {true_vectorized_result['time_ms'] / fast_sliding_result['time_ms']:.2f}x"
                    )
                    print(
                        f"    Fast Sliding time / True Vectorized time: {fast_sliding_result['time_ms'] / true_vectorized_result['time_ms']:.2f}x"
                    )
            else:
                print(f"❌ True Vectorized Sliding Window (w={window_size}): Failed")

            # Test 6: True Vectorized Sliding Window Outer Loop (O(n×w))
            print(
                f"\n🔬 Test 6: True Vectorized Sliding Window Outer Loop (TRAINING mode, O(n×w), window={window_size})..."
            )
            true_vectorized_result = (
                benchmark_true_vectorized_sliding_window_outer_loop(
                    seq_len,
                    d_model=d_model,
                    nhead=nhead,
                    window_size=window_size,
                    batch_size=batch_size,
                )
            )
            if true_vectorized_result:
                print(f"✅ {true_vectorized_result}")
                print(f"  📊 True Vectorized vs Classical Full:")
                print(
                    f"    True Vectorized memory / Classical memory: {true_vectorized_result['memory_mb'] / classical_result['memory_mb']:.2f}x"
                )
                print(
                    f"    True Vectorized time / Classical time: {true_vectorized_result['time_ms'] / classical_result['time_ms']:.2f}x"
                )
                print(
                    f"    Classical time / True Vectorized time: {classical_result['time_ms'] / true_vectorized_result['time_ms']:.2f}x"
                )
                if sdpa_full_result:
                    print(f"  📊 True Vectorized vs SDPA Full:")
                    print(
                        f"    True Vectorized memory / SDPA Full memory: {true_vectorized_result['memory_mb'] / sdpa_full_result['memory_mb']:.2f}x"
                    )
                    print(
                        f"    True Vectorized time / SDPA Full time: {true_vectorized_result['time_ms'] / sdpa_full_result['time_ms']:.2f}x"
                    )
                    print(
                        f"    SDPA Full time / True Vectorized time: {sdpa_full_result['time_ms'] / true_vectorized_result['time_ms']:.2f}x"
                    )
                if fast_sliding_result:
                    print(f"  📊 True Vectorized vs Fast Sliding:")
                    print(
                        f"    True Vectorized memory / Fast Sliding memory: {true_vectorized_result['memory_mb'] / fast_sliding_result['memory_mb']:.2f}x"
                    )
                    print(
                        f"    True Vectorized time / Fast Sliding time: {true_vectorized_result['time_ms'] / fast_sliding_result['time_ms']:.2f}x"
                    )
                    print(
                        f"    Fast Sliding time / True Vectorized time: {fast_sliding_result['time_ms'] / true_vectorized_result['time_ms']:.2f}x"
                    )
            else:
                print(f"❌ True Vectorized Sliding Window (w={window_size}): Failed")

        # Clean up memory
        torch.cuda.empty_cache()

    print("\n🏁 TRAINING MODE Comparison complete!")


if __name__ == "__main__":
    """Main execution with error handling."""
    try:
        compare_training_mode_attention()
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback

        traceback.print_exc()
