import torch
import time
import numpy as np
from contextlib import contextmanager
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import gc
import os

# Import your modules
from bridge.domain.dataset import BridgeDataset
from bridge.domain.model import Model
from bridge.domain.datamodels import ModelConfig, DatasetConfig, TrainingConfig
from bridge.application.training import TrainingPipeline
from bridge.infra.metrics import metrics_logger_factory
from bridge.domain.datamodels import MetricsConfig


class PerformanceProfiler:
    """Comprehensive profiler for the BRIDGE model to identify bottlenecks."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.memory_usage = defaultdict(list)

    @contextmanager
    def profile(self, name, track_memory=True):
        """Context manager to profile code blocks with timing and optional memory tracking."""
        # Force garbage collection before measurement
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Track initial state
        start_time = time.perf_counter()
        if track_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()

        try:
            yield
        finally:
            # Measure final state
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            elapsed = end_time - start_time
            self.timings[name].append(elapsed)

            if track_memory and torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated()
                memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
                self.memory_usage[name].append(memory_used)

    def get_statistics(self):
        """Calculate statistics for all profiled operations."""
        stats = {}

        for name, times in self.timings.items():
            if len(times) > 0:
                stats[name] = {
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "total": np.sum(times),
                    "count": len(times),
                    "median": np.median(times),
                }

                # Add memory stats if available
                if name in self.memory_usage and len(self.memory_usage[name]) > 0:
                    stats[name]["memory_mb"] = np.mean(self.memory_usage[name])

        return stats

    def print_report(self, top_n=20):
        """Print a formatted report of profiling results."""
        stats = self.get_statistics()

        # Sort by total time
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True)

        print("\n" + "=" * 80)
        print("PERFORMANCE PROFILING REPORT")
        print("=" * 80)

        print(
            f"\n{'Operation':<40} {'Total(s)':<10} {'Mean(ms)':<10} {'Std(ms)':<10} {'Count':<8} {'Memory(MB)':<10}"
        )
        print("-" * 80)

        for name, stat in sorted_stats[:top_n]:
            memory_str = (
                f"{stat.get('memory_mb', 0):.1f}" if "memory_mb" in stat else "N/A"
            )
            print(
                f"{name:<40} {stat['total']:<10.3f} {stat['mean']*1000:<10.2f} {stat['std']*1000:<10.2f} {stat['count']:<8} {memory_str:<10}"
            )

    def plot_results(self, output_dir="profiling_results"):
        """Generate visualization plots for profiling results."""
        os.makedirs(output_dir, exist_ok=True)
        stats = self.get_statistics()

        # Create timing breakdown chart
        plt.figure(figsize=(12, 8))
        top_operations = sorted(
            stats.items(), key=lambda x: x[1]["total"], reverse=True
        )[:15]

        names = [op[0] for op in top_operations]
        totals = [op[1]["total"] for op in top_operations]

        plt.barh(names, totals)
        plt.xlabel("Total Time (seconds)")
        plt.title("Top 15 Time-Consuming Operations")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/timing_breakdown.png")
        plt.close()

        # Create memory usage chart if available
        memory_ops = [
            (name, stat["memory_mb"])
            for name, stat in stats.items()
            if "memory_mb" in stat and stat["memory_mb"] > 0
        ]

        if memory_ops:
            plt.figure(figsize=(12, 8))
            memory_ops_sorted = sorted(memory_ops, key=lambda x: x[1], reverse=True)[
                :15
            ]
            names = [op[0] for op in memory_ops_sorted]
            memory = [op[1] for op in memory_ops_sorted]

            plt.barh(names, memory)
            plt.xlabel("Average Memory Usage (MB)")
            plt.title("Top 15 Memory-Consuming Operations")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/memory_usage.png")
            plt.close()


def profile_model_components(profiler, model, dataset, config):
    """Profile individual model components and operations."""

    device = model.device
    print("Profiling model components...")

    # Test different batch sizes
    batch_sizes = [1, 8, 32, 64]

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")

        # Profile dataset loading
        with profiler.profile(f"dataset_getitem_batch_{batch_size}"):
            if batch_size == 1:
                encoding = dataset[0]
            else:
                encoding = dataset[slice(0, min(batch_size, len(dataset)))]

        if encoding is None:
            continue

        # Profile orthographic embedding
        if hasattr(encoding, "orthographic") and encoding.orthographic is not None:
            with profiler.profile(f"embed_orth_tokens_batch_{batch_size}"):
                orth_embeddings = model.embed_orth_tokens(
                    encoding.orthographic.enc_input_ids
                )

        # Profile phonological embedding - this is likely a major bottleneck
        if hasattr(encoding, "phonological") and encoding.phonological is not None:
            with profiler.profile(f"embed_phon_tokens_batch_{batch_size}"):
                phon_embeddings = model.embed_phon_tokens(
                    encoding.phonological.enc_input_ids
                )

        # Profile different pathways
        pathways = ["o2p", "p2o", "p2p", "op2op"]

        for pathway in pathways:
            if pathway == "o2p" and encoding.orthographic is not None:
                with profiler.profile(f"forward_{pathway}_batch_{batch_size}"):
                    output = model(
                        task="o2p",
                        orth_enc_input=encoding.orthographic.enc_input_ids,
                        orth_enc_pad_mask=encoding.orthographic.enc_pad_mask,
                        phon_dec_input=encoding.phonological.dec_input_ids,
                        phon_dec_pad_mask=encoding.phonological.dec_pad_mask,
                    )

            # Profile generation
            if pathway == "o2p":
                with profiler.profile(f"generate_{pathway}_batch_{batch_size}"):
                    gen_output = model.generate(
                        encoding, pathway="o2p", deterministic=True
                    )

                # Profile individual generation components
                if batch_size == 1:  # Detailed profiling for single batch
                    # Profile phonology decoder loop
                    with profiler.profile("phonology_decoder_loop_single"):
                        mask = model.generate_triangular_mask(model.max_phon_seq_len)
                        test_tokens = [
                            [
                                torch.tensor(
                                    [model.dataset.tokenizer.phon_bos_id],
                                    dtype=torch.long,
                                    device=device,
                                )
                            ]
                        ]
                        test_embeddings = model.embed_phon_tokens(test_tokens)
                        model.phonology_decoder_loop(
                            mask,
                            test_embeddings,
                            test_tokens,
                            gen_output.global_encoding,
                            deterministic=True,
                        )

                    # Profile phono_sample
                    test_probs = torch.rand(1, 2, 35, device=device)
                    test_probs = torch.softmax(test_probs, dim=1)
                    with profiler.profile("phono_sample"):
                        for _ in range(10):  # Multiple samples for better statistics
                            model.phono_sample(test_probs, deterministic=True)


def profile_training_pipeline(
    profiler, model, dataset, training_config, metrics_config
):
    """Profile the training pipeline operations."""

    print("\nProfiling training pipeline...")

    # Create a small training pipeline for testing
    pipeline = TrainingPipeline(
        model=model,
        training_config=training_config,
        dataset=dataset,
        metrics_logger=metrics_logger_factory(metrics_config),
    )

    # Profile different batch operations
    test_slices = [slice(0, 16), slice(0, 32), slice(0, 64)]

    for test_slice in test_slices:
        batch_size = test_slice.stop - test_slice.start

        # Profile single training step
        with profiler.profile(f"training_step_batch_{batch_size}"):
            metrics = pipeline.single_step(dataset, test_slice, calculate_metrics=True)

        # Profile metrics calculation separately
        batch = dataset[test_slice]
        logits = pipeline.forward(batch.orthographic, batch.phonological)

        with profiler.profile(f"compute_metrics_batch_{batch_size}"):
            metrics = pipeline.compute_metrics(
                logits, batch.orthographic, batch.phonological
            )

        # Profile individual metric calculations
        if "phon" in logits:
            from bridge.application.training.phon_metrics import (
                calculate_euclidean_distance,
                calculate_cosine_distance,
                calculate_closest_phoneme_cdist,
                calculate_phon_feature_accuracy,
            )

            phon_pred = torch.argmax(logits["phon"], dim=1)
            phon_true = batch.phonological.targets

            with profiler.profile(f"euclidean_distance_batch_{batch_size}"):
                calculate_euclidean_distance(phon_true, phon_pred)

            with profiler.profile(f"cosine_distance_batch_{batch_size}"):
                calculate_cosine_distance(phon_true, phon_pred)

            with profiler.profile(f"closest_phoneme_batch_{batch_size}"):
                calculate_closest_phoneme_cdist(
                    phon_true, phon_pred, pipeline.phon_reps, norm=2
                )


def profile_tokenizers(profiler, dataset, test_words):
    """Profile tokenizer performance."""

    print("\nProfiling tokenizers...")

    # Profile character tokenizer
    char_tokenizer = dataset.tokenizer.char_tokenizer

    with profiler.profile("char_tokenizer_single"):
        for word in test_words[:10]:
            char_tokenizer.encode(word)

    with profiler.profile("char_tokenizer_batch"):
        char_tokenizer.encode(test_words)

    # Profile phoneme tokenizer
    phoneme_tokenizer = dataset.tokenizer.phoneme_tokenizer

    with profiler.profile("phoneme_tokenizer_single"):
        for word in test_words[:10]:
            phoneme_tokenizer.encode(word)

    with profiler.profile("phoneme_tokenizer_batch"):
        phoneme_tokenizer.encode(test_words)

    # Profile full bridge tokenizer
    with profiler.profile("bridge_tokenizer_both"):
        dataset.tokenizer.encode(test_words, modality_filter="both")

    with profiler.profile("bridge_tokenizer_orth_only"):
        dataset.tokenizer.encode(test_words, modality_filter="orthography")


def main(count=1):
    """Main profiling function."""

    # Initialize profiler
    profiler = PerformanceProfiler()

    # Load configurations
    print("Loading configurations...")
    model_config = ModelConfig(
        d_model=128,
        nhead=16,
        num_phon_enc_layers=4,
        num_orth_enc_layers=2,
        num_mixing_enc_layers=2,
        num_phon_dec_layers=2,
        num_orth_dec_layers=1,
    )

    dataset_config = DatasetConfig(
        dataset_filepath="data.csv",  # Use a test dataset
        tokenizer_cache_size=10000,
    )

    training_config = TrainingConfig(
        num_epochs=1, batch_size_train=32, learning_rate=0.001, training_pathway="o2p"
    )

    metrics_config = MetricsConfig(
        batch_metrics=True,
        training_metrics=True,
        validation_metrics=True,
        modes=["stdout"],
        filename=None,
    )

    # Initialize dataset and model
    print("Initializing dataset and model...")
    dataset = BridgeDataset(dataset_config)
    model = Model(model_config, dataset)
    model.eval()  # Set to evaluation mode

    # Get test words
    test_words = dataset.words[:100] if len(dataset.words) > 100 else dataset.words

    # Run profiling
    print("\nStarting profiling...")

    for _ in range(count):
        # Profile tokenizers
        profile_tokenizers(profiler, dataset, test_words)

        # Profile model components
        profile_model_components(profiler, model, dataset, model_config)

        # Profile training pipeline
        profile_training_pipeline(
            profiler, model, dataset, training_config, metrics_config
        )

    # Generate report
    print("\nGenerating profiling report...")
    profiler.print_report(top_n=30)
    profiler.plot_results()

    # Save detailed results
    stats = profiler.get_statistics()
    df = pd.DataFrame.from_dict(stats, orient="index")
    df.to_csv("profiling_results/detailed_stats.csv")

    print("\nProfiling complete! Results saved to 'profiling_results/' directory.")

    # Identify top bottlenecks
    print("\n" + "=" * 80)
    print("TOP BOTTLENECKS IDENTIFIED:")
    print("=" * 80)

    sorted_by_total = sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True)[
        :5
    ]
    for i, (name, stat) in enumerate(sorted_by_total, 1):
        print(f"\n{i}. {name}")
        print(f"   Total time: {stat['total']:.3f}s")
        print(f"   Average time: {stat['mean']*1000:.2f}ms")
        print(f"   Called {stat['count']} times")


if __name__ == "__main__":
    main(count=5)
