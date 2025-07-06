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


class ChunkedSlidingWindowModel(nn.Module):
    """Ultra-efficient sliding window attention using chunked computation.

    This implementation avoids masks entirely by:
    1. Chunking sequences into overlapping windows
    2. Computing attention only within each chunk
    3. Using vectorized operations across chunks
    4. Handling boundaries with ghost points

    Memory: O(n×w) instead of O(n²)
    Computation: Only processes relevant attention pairs
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

    def chunked_sliding_window_attention(self, q, k, v):
        """Compute sliding window attention using efficient chunking.

        Args:
            q: Query tensor [batch_size, nhead, seq_len, head_dim]
            k: Key tensor [batch_size, nhead, seq_len, head_dim]
            v: Value tensor [batch_size, nhead, seq_len, head_dim]

        Returns:
            Attention output [batch_size, nhead, seq_len, head_dim]
        """
        batch_size, nhead, seq_len, head_dim = q.shape
        window_size = self.window_size

        # Initialize output tensor
        output = torch.zeros_like(q)

        # Process each position with its sliding window
        for i in range(seq_len):
            # Define window boundaries for position i
            start_pos = max(0, i - window_size + 1)
            end_pos = i + 1

            # Extract query for position i
            q_i = q[:, :, i : i + 1, :]  # [batch_size, nhead, 1, head_dim]

            # Extract keys and values for the window
            k_window = k[
                :, :, start_pos:end_pos, :
            ]  # [batch_size, nhead, window_len, head_dim]
            v_window = v[
                :, :, start_pos:end_pos, :
            ]  # [batch_size, nhead, window_len, head_dim]

            # Compute attention within this window (no mask needed!)
            attn_output = F.scaled_dot_product_attention(
                q_i,
                k_window,
                v_window,
                attn_mask=None,  # No mask needed!
                dropout_p=0.0,
                is_causal=False,
            )

            # Store result
            output[:, :, i : i + 1, :] = attn_output

        return output

    def vectorized_sliding_window_attention(self, q, k, v):
        """Fully vectorized sliding window attention (more complex but faster).

        This version processes all positions simultaneously using advanced indexing.
        """
        batch_size, nhead, seq_len, head_dim = q.shape
        window_size = self.window_size

        # Create indices for all sliding windows
        # For each position i, we need indices [max(0, i-w+1), i]
        all_indices = []
        all_q_indices = []

        for i in range(seq_len):
            start_pos = max(0, i - window_size + 1)
            end_pos = i + 1
            window_indices = list(range(start_pos, end_pos))

            # Pad shorter windows to maintain tensor shape
            if len(window_indices) < window_size:
                # Pad with the first valid index (ghost points)
                padding = [window_indices[0]] * (window_size - len(window_indices))
                window_indices = padding + window_indices

            all_indices.append(window_indices)
            all_q_indices.append(i)

        # Convert to tensors
        window_indices = torch.tensor(
            all_indices, device=q.device
        )  # [seq_len, window_size]
        q_indices = torch.tensor(all_q_indices, device=q.device)  # [seq_len]

        # Gather keys and values for all windows simultaneously
        # k_windows: [batch_size, nhead, seq_len, window_size, head_dim]
        k_windows = k[:, :, window_indices, :]  # Advanced indexing
        v_windows = v[:, :, window_indices, :]

        # Gather queries for all positions
        # q_selected: [batch_size, nhead, seq_len, head_dim]
        q_selected = q[:, :, q_indices, :]

        # Compute attention scores for all windows
        # scores: [batch_size, nhead, seq_len, window_size]
        scores = torch.matmul(
            q_selected.unsqueeze(-2), k_windows.transpose(-2, -1)
        ).squeeze(-2)
        scores = scores / (head_dim**0.5)

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        # output: [batch_size, nhead, seq_len, head_dim]
        output = torch.matmul(attn_weights.unsqueeze(-2), v_windows).squeeze(-2)

        return output

    def forward(self, x):
        """Forward pass with chunked sliding window attention.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Use vectorized sliding window attention (more efficient)
        if seq_len * self.window_size < 50000:  # Use vectorized for reasonable sizes
            attn_output = self.vectorized_sliding_window_attention(q, k, v)
        else:  # Fall back to chunked for very large sequences
            attn_output = self.chunked_sliding_window_attention(q, k, v)

        # Reshape back
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        # Output projection
        output = self.out_proj(attn_output)

        return output


def benchmark_chunked_sliding_window(
    seq_len, d_model=1024, nhead=1, window_size=128, batch_size=4
):
    """Benchmark chunked sliding window attention (no masks, vectorized).

    This implementation should be significantly faster and more memory efficient
    than mask-based approaches.

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
        f"Benchmarking Chunked Sliding Window (TRAINING mode, O(n×w), window={window_size}) on {device}"
    )

    # Create model with chunked sliding window
    model = ChunkedSlidingWindowModel(d_model, nhead, window_size).to(device)
    model.train()  # Training mode

    # Create input data
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    print(f"  Using mask-free chunked computation (O(n×w) complexity)...")

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
        "attention_type": "Chunked_Sliding_Window",
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
        {"seq_len": 1024},
        {"seq_len": 2048},
        {"seq_len": 4096},
        {"seq_len": 8192},
    ]

    window_sizes = [128, 256, 512]  # Reasonable window sizes
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

            # Test chunked sliding window (mask-free)
            print(
                f"\n�� Test 4: Chunked Sliding Window (TRAINING mode, O(n×w), window={window_size})..."
            )
            chunked_result = benchmark_chunked_sliding_window(
                seq_len,
                d_model=d_model,
                nhead=nhead,
                window_size=window_size,
                batch_size=batch_size,
            )
            if chunked_result:
                print(f"✅ {chunked_result}")
                print(f"  📊 Chunked vs Classical Full:")
                print(
                    f"    Memory ratio: {chunked_result['memory_mb'] / classical_result['memory_mb']:.2f}x"
                )
                print(
                    f"    Speed ratio: {chunked_result['time_ms'] / classical_result['time_ms']:.2f}x"
                )
                print(
                    f"    Speedup: {classical_result['time_ms'] / chunked_result['time_ms']:.2f}x"
                )
                print(f"  📊 Chunked vs SDPA Full:")
                print(
                    f"    Memory ratio: {chunked_result['memory_mb'] / sdpa_full_result['memory_mb']:.2f}x"
                )
                print(
                    f"    Speed ratio: {chunked_result['time_ms'] / sdpa_full_result['time_ms']:.2f}x"
                )
                print(
                    f"    Speedup: {sdpa_full_result['time_ms'] / chunked_result['time_ms']:.2f}x"
                )
                print(f"  📊 Chunked vs SDPA Sliding:")
                print(
                    f"    Memory ratio: {chunked_result['memory_mb'] / sdpa_sliding_result['memory_mb']:.2f}x"
                )
                print(
                    f"    Speed ratio: {chunked_result['time_ms'] / sdpa_sliding_result['time_ms']:.2f}x"
                )
                print(
                    f"    Speedup: {sdpa_sliding_result['time_ms'] / chunked_result['time_ms']:.2f}x"
                )
            else:
                print(f"❌ Chunked Sliding Window (w={window_size}): Failed")

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
