import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.activation import MultiheadAttention


def benchmark_classical_full_attention(
    seq_len: int,
    d_model: int = 1024,
    nhead: int = 1,
    # num_layers: int = 1,
    batch_size: int = 4,
):
    """Benchmark classical full attention in TRAINING mode (O(n²) complexity).

    Uses PyTorch's TransformerEncoderLayer in training mode, which uses
    classical attention implementation (NOT SDPA).

    Args:
        seq_len: Sequence length
        d_model: Model dimension (default 1024)
        nhead: Number of attention heads (default 1)
        # num_layers: Number of encoder layers
        batch_size: Batch size for testing

    Returns:
        Dictionary with benchmark results

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Used in torch.nn.TransformerEncoderLayer
    classical_model = MultiheadAttention(d_model, nhead, batch_first=True)

    # FIX: Move model to the same device as input tensors
    classical_model = classical_model.to(device)

    # CRITICAL: Set to training mode (uses classical attention, not SDPA)
    classical_model.train()

    # Create input with gradient tracking for training mode
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    # Warmup
    # print("  Warming up in training mode...")
    for _ in range(3):
        output = classical_model(x, x, x)
        # output[0].shape=torch.Size([1, 1024, 512]), output[1].shape=torch.Size([1024, 1, 1])
        # Simulate backward pass for training
        loss = (output[0] ** 2).sum()  # [batch, seq_len, d_model]
        loss.backward()
        x.grad = None  # Clear gradients

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(10):
        output = classical_model(x, x, x)
        # Simulate backward pass for training
        loss = (output[0] ** 2).sum()  # [batch, seq_len, d_model]
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
        "seq_len": seq_len,
        "d_model": d_model,
        "nhead": nhead,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }
