import time
import torch
from torch import nn
from torch.nn import functional as F

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
