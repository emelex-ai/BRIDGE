# bridge/utils/metric_extensions.py

import functools
import os
import time

import torch

try:
    import psutil
except ImportError:
    psutil = None

from bridge.domain.model.memory_utils import (
    check_cuda_memory,
    track_differential_memory,
    track_gpu_memory,
)


def get_gpu_memory_metrics() -> dict:
    """Get current and peak GPU memory usage in MB.

    Returns:
        A dictionary with current and peak GPU memory usage in MB.

    """
    import torch

    if not torch.cuda.is_available():
        return {}
    current = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return {
        "gpu_memory_current_mb": current,
        "gpu_memory_peak_mb": peak,
    }


def measure_performance(memory_enabled: bool = True, timing_enabled: bool = False):
    """Decorator to measure time and (optionally) memory usage of a function.

    Args:
        timing_enabled: Whether to enable timing measurements.
        memory_enabled: Whether to enable CPU and GPU memory
            measurements (requires psutil for the CPU).

    Returns:
        Decorated function with timing and memory metrics attached as an attribute.

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            metrics = {}
            gpu_available = torch.cuda.is_available()
            if memory_enabled or timing_enabled:
                # CPU memory
                if memory_enabled and psutil is not None:
                    process = psutil.Process(os.getpid())
                    start_cpu_mem = process.memory_info().rss / 1024**2
                # GPU memory
                if memory_enabled and gpu_available:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    start_gpu_mem = torch.cuda.memory_allocated() / 1024**2
                start_time = time.time()
                result = func(self, *args, **kwargs)
                end_time = time.time()
                # CPU memory
                if memory_enabled and psutil is not None:
                    end_cpu_mem = process.memory_info().rss / 1024**2
                    metrics["cpu_memory_usage_mb"] = end_cpu_mem - start_cpu_mem
                    metrics["cpu_memory_rss_mb"] = end_cpu_mem
                # GPU memory (use utility)
                if memory_enabled and gpu_available:
                    gpu_metrics = get_gpu_memory_metrics()
                    metrics.update(gpu_metrics)
                    metrics["gpu_memory_usage_mb"] = (
                        gpu_metrics["gpu_memory_current_mb"] - start_gpu_mem
                    )
                if timing_enabled:
                    metrics["step_time_sec"] = end_time - start_time
            else:
                result = func(self, *args, **kwargs)
            wrapper.last_metrics = metrics
            wrapper.metrics.append(metrics.copy())
            return result

        wrapper.last_metrics = {}
        wrapper.metrics = []
        return wrapper

    return decorator


def save_metrics_to_csv(filename: str, metrics: dict):
    """Append memory and timing metrics to a CSV file, rounding floats to 5 significant digits.

    Only keys containing 'memory' or 'time' are saved.
    """
    import csv
    import os

    def round_floats(d):
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in d.items()}

    # Only keep keys related to memory or timing
    filtered_metrics = {
        k: v for k, v in metrics.items() if "memory" in k or "time" in k
    }
    filtered_metrics = round_floats(filtered_metrics)

    if not filtered_metrics:
        print("No memory or timing metrics to save.")
        return

    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=filtered_metrics.keys())
        if not file_exists or os.stat(filename).st_size == 0:
            writer.writeheader()
        writer.writerow(filtered_metrics)


class MyClass:
    @measure_performance(memory_enabled=True, timing_enabled=True)
    def compute(self, n: int) -> float:
        x = [i**2 for i in range(n)]
        return sum(x) / n


if __name__ == "__main__":
    obj = MyClass()
    result = obj.compute(1000000)  # This populates obj.compute.last_metrics
    print("Result:", result)
    print("Metrics:", obj.compute.last_metrics)
    save_metrics_to_csv("metrics.csv", obj.compute.last_metrics)
