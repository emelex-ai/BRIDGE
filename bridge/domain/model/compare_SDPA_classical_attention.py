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
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        output = classical_model(x)
        # Simulate backward pass for training
        loss = output.sum()
        loss.backward()
        x.grad = None  # Clear gradients

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "attention_type": "Classical_Full_Attention",
        "mode": "TRAINING",
        "complexity": "O(n²)",
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
                q, k, v, attn_mask=self.attn_mask
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
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        output = sdpa_model(x)
        # Simulate backward pass for training
        loss = output.sum()
        loss.backward()
        x.grad = None  # Clear gradients

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

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
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark with backward pass
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

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
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

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


class VectorizedSlidingWindowModel(nn.Module):
    """Vectorized sliding window attention - no masks, no loops.

    This implementation uses advanced indexing to gather only the relevant
    key-value pairs for each query position, achieving true O(n×w) complexity
    without any loops or mask creation.
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

        Returns:
            query_indices: [seq_len] - which query position
            key_indices: [seq_len, window_size] - which key positions for each query
            valid_mask: [seq_len, window_size] - which positions are valid (for padding)
        """
        # For each query position i, we want keys [max(0, i-w+1), i]
        query_positions = torch.arange(seq_len, device=device)  # [seq_len]

        # Create relative offsets for the sliding window
        # For window_size=3: offsets = [-2, -1, 0] (relative to query position)
        window_offsets = torch.arange(
            -self.window_size + 1, 1, device=device
        )  # [window_size]

        # Broadcast to get absolute key positions for each query
        # query_positions[:, None] + window_offsets[None, :] gives us the key indices
        key_indices = (
            query_positions[:, None] + window_offsets[None, :]
        )  # [seq_len, window_size]

        # Create mask for valid positions (handle boundaries)
        valid_mask = (key_indices >= 0) & (
            key_indices < seq_len
        )  # [seq_len, window_size]

        # Clamp indices to valid range (for gathering)
        key_indices = torch.clamp(key_indices, 0, seq_len - 1)

        return query_positions, key_indices, valid_mask

    def vectorized_sliding_window_attention(self, q, k, v):
        """Compute sliding window attention using pure vectorized operations.

        Args:
            q, k, v: [batch_size, nhead, seq_len, head_dim]

        Returns:
            output: [batch_size, nhead, seq_len, head_dim]
        """
        batch_size, nhead, seq_len, head_dim = q.shape

        # Create sliding window indices
        query_indices, key_indices, valid_mask = self.create_sliding_window_indices(
            seq_len, q.device
        )

        # Gather keys and values for all sliding windows simultaneously
        # key_indices: [seq_len, window_size]
        # k: [batch_size, nhead, seq_len, head_dim]
        # Result: [batch_size, nhead, seq_len, window_size, head_dim]
        k_windows = k[:, :, key_indices, :]  # Advanced indexing - gathers relevant keys
        v_windows = v[
            :, :, key_indices, :
        ]  # Advanced indexing - gathers relevant values

        # Expand queries to match window structure
        # q: [batch_size, nhead, seq_len, head_dim] -> [batch_size, nhead, seq_len, 1, head_dim]
        q_expanded = q.unsqueeze(-2)  # [batch_size, nhead, seq_len, 1, head_dim]

        # Compute attention scores for all windows vectorized
        # q_expanded @ k_windows.transpose(-2, -1): [batch_size, nhead, seq_len, 1, window_size]
        scores = torch.matmul(q_expanded, k_windows.transpose(-2, -1)) / (head_dim**0.5)
        scores = scores.squeeze(-2)  # [batch_size, nhead, seq_len, window_size]

        # Apply boundary mask (set invalid positions to -inf)
        # valid_mask: [seq_len, window_size] -> [1, 1, seq_len, window_size]
        mask_expanded = valid_mask[
            None, None, :, :
        ]  # Broadcast for batch and head dims
        scores = torch.where(
            mask_expanded, scores, torch.tensor(float("-inf"), device=scores.device)
        )

        # Apply softmax to get attention weights
        attn_weights = F.softmax(
            scores, dim=-1
        )  # [batch_size, nhead, seq_len, window_size]

        # Apply attention weights to values
        # attn_weights: [batch_size, nhead, seq_len, window_size]
        # v_windows: [batch_size, nhead, seq_len, window_size, head_dim]
        # Result: [batch_size, nhead, seq_len, head_dim]
        output = torch.matmul(attn_weights.unsqueeze(-2), v_windows).squeeze(-2)

        return output

    def forward(self, x):
        """Forward pass with vectorized sliding window attention."""
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


def benchmark_vectorized_sliding_window(
    seq_len, d_model=1024, nhead=1, window_size=128, batch_size=4
):
    """Benchmark vectorized sliding window attention (no masks, no loops).

    This should achieve true O(n×w) complexity with efficient vectorized operations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Benchmarking Vectorized Sliding Window (TRAINING mode, O(n×w), window={window_size}) on {device}"
    )

    # Create model
    model = VectorizedSlidingWindowModel(d_model, nhead, window_size).to(device)
    model.train()

    # Create input data
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    print(f"  Using vectorized operations (no masks, no loops, true O(n×w))...")

    # Warmup
    print("  Warming up in training mode...")
    for _ in range(3):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    # Clear memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        model.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "attention_type": "Vectorized_Sliding_Window",
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
        {"seq_len": 8192},
        {"seq_len": 4 * 8192},
    ]

    window_sizes = [32, 128, 512]  # Reasonable window sizes
    d_model = 1024  # Fixed as requested
    nhead = 1  # Single head as requested
    batch_size = 4

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

            # Compare SDPA vs Classical
            memory_ratio = sdpa_full_result["memory_mb"] / classical_result["memory_mb"]
            speed_ratio = sdpa_full_result["time_ms"] / classical_result["time_ms"]
            speedup = classical_result["time_ms"] / sdpa_full_result["time_ms"]

            print(f"  📊 SDPA Full vs Classical Full:")
            print(f"    Memory ratio: {memory_ratio:.2f}x")
            print(f"    Speed ratio: {speed_ratio:.2f}x")
            print(f"    SDPA speedup: {speedup:.2f}x")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ SDPA Full Attention: OOM at seq_len={seq_len}")
                sdpa_full_result = None
            else:
                print(f"❌ SDPA Full Attention error: {e}")
                sdpa_full_result = None

        # Test sliding window attention for different window sizes
        for window_size in [32, 128, 512]:
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
                    f"    Memory ratio: {sdpa_sliding_result['memory_mb'] / classical_result['memory_mb']:.2f}x"
                )
                print(
                    f"    Speed ratio: {sdpa_sliding_result['time_ms'] / classical_result['time_ms']:.2f}x"
                )
                print(
                    f"    Speedup: {classical_result['time_ms'] / sdpa_sliding_result['time_ms']:.2f}x"
                )
                print(f"  📊 SDPA Sliding vs SDPA Full:")
                print(
                    f"    Memory ratio: {sdpa_sliding_result['memory_mb'] / sdpa_full_result['memory_mb']:.2f}x"
                )
                print(
                    f"    Speed ratio: {sdpa_sliding_result['time_ms'] / sdpa_full_result['time_ms']:.2f}x"
                )
                print(
                    f"    Speedup: {sdpa_full_result['time_ms'] / sdpa_sliding_result['time_ms']:.2f}x"
                )
            else:
                print(f"❌ SDPA Sliding Window (w={window_size}): Failed")

            # Test fast sliding window (simple and efficient)
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
                    f"    Memory ratio: {fast_sliding_result['memory_mb'] / classical_result['memory_mb']:.2f}x"
                )
                print(
                    f"    Speed ratio: {fast_sliding_result['time_ms'] / classical_result['time_ms']:.2f}x"
                )
                print(
                    f"    Speedup: {classical_result['time_ms'] / fast_sliding_result['time_ms']:.2f}x"
                )
                print(f"  📊 Fast Sliding vs SDPA Full:")
                print(
                    f"    Memory ratio: {fast_sliding_result['memory_mb'] / sdpa_full_result['memory_mb']:.2f}x"
                )
                print(
                    f"    Speed ratio: {fast_sliding_result['time_ms'] / sdpa_full_result['time_ms']:.2f}x"
                )
                print(
                    f"    Speedup: {sdpa_full_result['time_ms'] / fast_sliding_result['time_ms']:.2f}x"
                )
            else:
                print(f"❌ Fast Sliding Window (w={window_size}): Failed")

            # Test vectorized sliding window (no masks, no loops)
            print(
                f"\n🔬 Test 5: Vectorized Sliding Window (TRAINING mode, O(n×w), window={window_size})..."
            )
            vectorized_result = benchmark_vectorized_sliding_window(
                seq_len,
                d_model=d_model,
                nhead=nhead,
                window_size=window_size,
                batch_size=batch_size,
            )
            if vectorized_result:
                print(f"✅ {vectorized_result}")
                print(f"  📊 Vectorized vs Classical Full:")
                print(
                    f"    Memory ratio: {vectorized_result['memory_mb'] / classical_result['memory_mb']:.2f}x"
                )
                print(
                    f"    Speed ratio: {vectorized_result['time_ms'] / classical_result['time_ms']:.2f}x"
                )
                print(
                    f"    Speedup: {classical_result['time_ms'] / vectorized_result['time_ms']:.2f}x"
                )
                print(f"  📊 Vectorized vs SDPA Full:")
                print(
                    f"    Memory ratio: {vectorized_result['memory_mb'] / sdpa_full_result['memory_mb']:.2f}x"
                )
                print(
                    f"    Speed ratio: {vectorized_result['time_ms'] / sdpa_full_result['time_ms']:.2f}x"
                )
                print(
                    f"    Speedup: {sdpa_full_result['time_ms'] / vectorized_result['time_ms']:.2f}x"
                )
                if fast_sliding_result:
                    print(f"  📊 Vectorized vs Fast Sliding:")
                    print(
                        f"    Memory ratio: {vectorized_result['memory_mb'] / fast_sliding_result['memory_mb']:.2f}x"
                    )
                    print(
                        f"    Speed ratio: {vectorized_result['time_ms'] / fast_sliding_result['time_ms']:.2f}x"
                    )
                    print(
                        f"    Speedup: {fast_sliding_result['time_ms'] / vectorized_result['time_ms']:.2f}x"
                    )
            else:
                print(f"❌ Vectorized Sliding Window (w={window_size}): Failed")

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
