import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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
    #print(
        #f"Benchmarking Classical Full Attention with Sliding Window Mask (TRAINING mode, O(nw)) on {device}"
    #)

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
    #print("  Warming up in training mode...")
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
        "seq_len": seq_len,
        "tokens_per_sec": (batch_size * seq_len * 10) / (end_time - start_time),
    }
