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


