"""Memory utils for tracking GPU memory usage."""

import functools
import gc
import time
from collections.abc import Callable

import torch


def track_gpu_memory(
    bound_model: Callable | torch.nn.Module,
) -> float:
    """Track standard GPU memory with cache clearing and peak reset.

    Returns:
        float: Memory used in MB
    """
    if torch.cuda.is_available():
        # Clear cache and reset peak memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Run your model/operation
        # Bind the function
        # output = model(input_tensor)
        output = bound_model()

        # Get peak memory usage
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def track_differential_memory(
    bound_model: Callable | torch.nn.Module,
) -> float:
    """Track memory usage by measuring difference from initial state."""
    if torch.cuda.is_available():
        # Clear cache first
        torch.cuda.empty_cache()

        # Get initial memory in bytes (int)
        initial_memory = torch.cuda.memory_allocated()

        # Run operation
        output = bound_model()

        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated()

        # Calculate memory used
        return (peak_memory - initial_memory) / 1024**2
    return 0.0


def check_cuda_memory():
    """Check comprehensive CUDA memory usage and print statistics."""
    if torch.cuda.is_available():
        # Current memory allocated by PyTorch
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB

        # Peak memory allocated
        peak = torch.cuda.max_memory_allocated() / 1024**3  # GB

        # Total GPU memory
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

        print(f"Allocated: {allocated:.2f}GB")
        print(f"Peak: {peak:.2f}GB")
        print(f"Total: {total:.2f}GB")
        print(f"Free: {total - allocated:.2f}GB")


def track_memory_with_cleanup(device: str = "cuda"):
    """Track memory with aggressive cleanup after each test."""
    # Before test
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Run test
    output = model(input_tensor)

    # After test - aggressive cleanup
    locals_to_delete = ["model", "input_tensor", "output"]
    for var_name in locals_to_delete:
        if var_name in locals():
            del locals()[var_name]

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Get memory stats
    current_memory = torch.cuda.memory_allocated() / 1024**2
    max_memory = torch.cuda.max_memory_allocated() / 1024**2

    return current_memory, max_memory


def benchmark_memory_usage(
    bound_model: Callable | torch.nn.Module,
    num_iterations: int = 10,
    device: str = "cuda",
    warmup_iterations: int = 3,
    training_mode: bool = False,
    msg: str = "Benchmark",
) -> tuple[float, float]:
    """Benchmark memory usage with proper cleanup.

    Args:
        bound_model: The model to benchmark. Should be a callable with no arguments
                    (use functools.partial to bind arguments). Can be either a
                    torch.nn.Module or any callable.
        num_iterations: The number of iterations to run for benchmarking.
        device: The device to run the model on ("cuda" or "cpu").
        warmup_iterations: The number of warmup iterations before benchmarking.
        training_mode: Whether to run in training mode (affects memory usage).
        msg: Descriptive message to print with the results.

    Returns:
        tuple[float, float]: (memory_mb, avg_time_sec) - Memory used in MB and average time per iteration in seconds

    Example:
        # For a model that takes input_tensor
        bound_model = functools.partial(model, input_tensor)
        memory_mb, avg_time_sec = benchmark_memory_usage(bound_model, msg="SDPA Full Attention")

        # For a model with multiple arguments
        bound_model = functools.partial(model, x, mask=mask, training=True)
        memory_mb, avg_time_sec = benchmark_memory_usage(bound_model, training_mode=True, msg="Sliding Window")
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        device = "cpu"

    if device == "cuda":
        # Clear cache and reset peak memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Set model to appropriate mode
        if hasattr(bound_model, "train") and hasattr(bound_model, "eval"):
            if training_mode:
                bound_model.train()
            else:
                bound_model.eval()

        # Warmup
        for _ in range(warmup_iterations):
            if training_mode:
                _ = bound_model()
            else:
                with torch.no_grad():
                    _ = bound_model()

        # Reset after warmup
        torch.cuda.reset_peak_memory_stats()

        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(num_iterations):
            if training_mode:
                _ = bound_model()
            else:
                with torch.no_grad():
                    _ = bound_model()

        torch.cuda.synchronize()
        end_time = time.time()

        # Get peak memory
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Cleanup
        torch.cuda.empty_cache()

        # Calculate timing
        total_time_sec = end_time - start_time
        avg_time_sec = total_time_sec / num_iterations

        # Print results with descriptive message
        print(f"{msg}: {memory_mb:.1f}MB, {avg_time_sec*1000:.2f}ms avg")

        return memory_mb, avg_time_sec

    return 0.0, 0.0


def create_bound_model(model: torch.nn.Module, *args, **kwargs) -> Callable:
    """Create a bound model for benchmarking.

    Args:
        model: The model to bind
        *args: Positional arguments for the model
        **kwargs: Keyword arguments for the model

    Returns:
        Callable: A function that can be called with no arguments

    Example:
        bound_model = create_bound_model(my_model, input_tensor, mask=mask)
        memory = benchmark_memory_usage(bound_model)
    """
    return functools.partial(model, *args, **kwargs)
