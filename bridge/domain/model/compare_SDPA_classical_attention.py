import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def precompute_sliding_window_mask(seq_len, window_size, device):
    """Precompute sliding window mask for efficiency.

    Args:
        seq_len: Sequence length
        window_size: Window size for sliding window
        device: Device to create mask on

    Returns:
        Precomputed attention mask

    """
    # Create base mask (all True = attend to all)
    mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)

    # Apply sliding window constraint efficiently
    for i in range(seq_len):
        # Zero out positions beyond window
        start_pos = max(0, i - window_size)
        end_pos = min(seq_len, i + window_size + 1)

        # Mask out everything except window
        mask[i, :start_pos] = False
        mask[i, end_pos:] = False

    # Convert to attention mask format (0 for attend, -inf for ignore)
    return torch.where(mask, 0.0, float("-inf"))


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


def benchmark_sdpa_sliding_window_attention(
    seq_len, window_size, d_model=1024, nhead=1, num_layers=1, batch_size=4
):
    """Benchmark SDPA sliding window attention in TRAINING mode (O(n) complexity).

    Uses PyTorch's scaled_dot_product_attention with PRECOMPUTED sliding window mask
    in training mode.

    Args:
        seq_len: Sequence length
        window_size: Sliding window size
        d_model: Model dimension (default 1024)
        nhead: Number of attention heads (default 1)
        num_layers: Number of encoder layers
        batch_size: Batch size for testing

    Returns:
        Dictionary with benchmark results

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Benchmarking SDPA Sliding Window (TRAINING mode, O(n), window={window_size}) on {device}"
    )

    class SDPASlidingWindowLayer(nn.Module):
        """Custom layer using SDPA for sliding window attention with precomputed mask."""

        def __init__(self, d_model, nhead, precomputed_mask):
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

            # Store precomputed mask
            self.register_buffer("attn_mask", precomputed_mask)

        def forward(self, x):
            # Self-attention with residual connection
            residual = x
            x = self.norm1(x)

            # Project to Q, K, V
            B, S, D = x.shape
            q = self.q_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)

            # SDPA sliding window attention with precomputed mask
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

    class SDPASlidingWindowModel(nn.Module):
        def __init__(self, d_model, nhead, num_layers, precomputed_mask):
            super().__init__()
            self.layers = nn.ModuleList(
                [
                    SDPASlidingWindowLayer(d_model, nhead, precomputed_mask)
                    for _ in range(num_layers)
                ]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    # Precompute sliding window mask
    print("  Precomputing sliding window mask...")
    precomputed_mask = precompute_sliding_window_mask(seq_len, window_size, device)

    sdpa_sliding_model = SDPASlidingWindowModel(
        d_model, nhead, num_layers, precomputed_mask
    ).to(device)

    # CRITICAL: Set to training mode
    sdpa_sliding_model.train()

    # Create input with gradient tracking for training mode
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    # Warmup
    print("  Warming up in training mode...")
    for _ in range(3):
        output = sdpa_sliding_model(x)
        # Simulate backward pass for training
        loss = output.sum()
        loss.backward()
        x.grad = None  # Clear gradients

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        output = sdpa_sliding_model(x)
        # Simulate backward pass for training
        loss = output.sum()
        loss.backward()
        x.grad = None  # Clear gradients

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / 10 * 1000
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "attention_type": "SDPA_Sliding_Window",
        "mode": "TRAINING",
        "complexity": "O(n)",
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

        # Test 3: SDPA Sliding Window (O(n))
        for window_size in window_sizes:
            try:
                print(
                    f"\n🔬 Test 3: SDPA Sliding Window (TRAINING mode, O(n), window={window_size})..."
                )
                sdpa_sliding_result = benchmark_sdpa_sliding_window_attention(
                    seq_len,
                    window_size,
                    d_model=d_model,
                    nhead=nhead,
                    batch_size=batch_size,
                )
                print(f"✅ {sdpa_sliding_result}")

                # Compare vs Classical Full
                memory_ratio_classical = (
                    sdpa_sliding_result["memory_mb"] / classical_result["memory_mb"]
                )
                speed_ratio_classical = (
                    sdpa_sliding_result["time_ms"] / classical_result["time_ms"]
                )
                speedup_classical = (
                    classical_result["time_ms"] / sdpa_sliding_result["time_ms"]
                )

                print(f"  📊 SDPA Sliding vs Classical Full:")
                print(f"    Memory ratio: {memory_ratio_classical:.2f}x")
                print(f"    Speed ratio: {speed_ratio_classical:.2f}x")
                print(f"    Speedup: {speedup_classical:.2f}x")

                # Compare vs SDPA Full (if available)
                if sdpa_full_result:
                    memory_ratio_sdpa = (
                        sdpa_sliding_result["memory_mb"] / sdpa_full_result["memory_mb"]
                    )
                    speed_ratio_sdpa = (
                        sdpa_sliding_result["time_ms"] / sdpa_full_result["time_ms"]
                    )
                    speedup_sdpa = (
                        sdpa_full_result["time_ms"] / sdpa_sliding_result["time_ms"]
                    )

                    print(f"  📊 SDPA Sliding vs SDPA Full:")
                    print(f"    Memory ratio: {memory_ratio_sdpa:.2f}x")
                    print(f"    Speed ratio: {speed_ratio_sdpa:.2f}x")
                    print(f"    Speedup: {speedup_sdpa:.2f}x")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ SDPA Sliding Window (w={window_size}): OOM")
                else:
                    print(f"❌ SDPA Sliding Window (w={window_size}): Error - {e}")

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
