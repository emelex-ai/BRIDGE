def benchmark_chunked_vectorized_sliding_window(
    seq_len, d_model=1024, nhead=1, window_size=128, chunk_size=32, batch_size=4
):
    """Benchmark chunked vectorized sliding window attention.

    This should achieve O(n×w) complexity with controlled memory usage.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(
        #f"Benchmarking Chunked Vectorized Sliding Window (TRAINING mode, O(n×w), window={window_size}, chunk={chunk_size}) on {device}"
    #)

    # Create model
    model = ChunkedVectorizedSlidingWindowModel(
        d_model, nhead, window_size, chunk_size
    ).to(device)
    model.train()

    # Create input data
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    #print(
        #f"  Using chunked vectorized operations (controlled memory, chunk_size={chunk_size})..."
    #)

    # Warmup
    # print("  Warming up in training mode...")
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
