#!/usr/bin/env python3
"""Test FlexAttention integration with BRIDGE architecture.

This script demonstrates how to use FlexAttention as a drop-in replacement
for other attention mechanisms in the BRIDGE encoder.
"""

from typing import Optional

import torch
import torch.nn as nn

# Import BRIDGE components
from bridge.domain.model.encoder_local import EncoderLocal


def test_flex_attention_integration():
    """Test FlexAttention integration with BRIDGE encoder."""
    print("Testing FlexAttention Integration with BRIDGE")
    print("=" * 60)

    # Test parameters
    batch_size = 2
    seq_len = 256
    d_model = 512
    nhead = 8
    num_layers = 4
    window_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create test input
    src = torch.randn(batch_size, seq_len, d_model, device=device)

    print(f"Input shape: {src.shape}")
    print(f"Window size: {window_size}")
    print()

    # Test different attention mechanisms
    attention_types = [
        "true_sliding_window",  # Your existing implementation
        "local",  # LocalAttention (chunked)
        "exact",  # ExactSlidingWindowAttention (stable PyTorch)
        "flex",  # FlexAttention (requires PyTorch nightly)
    ]

    results = {}

    for attention_type in attention_types:
        print(f"Testing {attention_type} attention...")
        print("-" * 40)

        try:
            # Create encoder with specific attention type
            encoder = EncoderLocal(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                device=device.type,
                window_size=window_size,
                causal=True,
                attention_type=attention_type,
            ).to(device)

            # Measure memory before
            if device.type == "cuda":
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()

            # Forward pass with timing
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = (
                torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            )
            end_time = (
                torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            )

            if device.type == "cuda":
                start_time.record()

            output = encoder(src)

            if device.type == "cuda":
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
            else:
                elapsed_time = 0  # Timing not available on CPU

            # Measure memory after
            if device.type == "cuda":
                memory_after = torch.cuda.memory_allocated()
                memory_used = (memory_after - memory_before) / 1024**2  # MB
            else:
                memory_used = 0

            # Store results
            results[attention_type] = {
                "output": output,
                "time": elapsed_time,
                "memory": memory_used,
                "success": True,
            }

            print(f"✓ Output shape: {output.shape}")
            print(f"✓ Output mean: {output.mean().item():.6f}")
            print(f"✓ Output std: {output.std().item():.6f}")
            if device.type == "cuda":
                print(f"✓ Time: {elapsed_time:.2f} ms")
                print(f"✓ Memory: {memory_used:.2f} MB")
            print("✓ Success!")

        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            results[attention_type] = {
                "output": None,
                "time": 0,
                "memory": 0,
                "success": False,
                "error": str(e),
            }

        print()

    # Compare results
    print("Comparison Summary")
    print("=" * 60)

    successful_results = {k: v for k, v in results.items() if v["success"]}

    if len(successful_results) > 1:
        # Compare outputs between different attention mechanisms
        print("Output Comparisons:")
        print("-" * 20)

        # Use the first successful result as baseline
        baseline_name = list(successful_results.keys())[0]
        baseline_output = successful_results[baseline_name]["output"]

        for name, result in successful_results.items():
            if name == baseline_name:
                print(f"{name}: (baseline)")
                continue

            output = result["output"]
            diff = torch.abs(baseline_output - output)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            relative_error = mean_diff / baseline_output.abs().mean().item()

            print(f"{name}:")
            print(f"  Max diff from {baseline_name}: {max_diff:.6f}")
            print(f"  Mean diff from {baseline_name}: {mean_diff:.6f}")
            print(f"  Relative error: {relative_error:.6f}")

            # Check if outputs are close
            close = torch.allclose(baseline_output, output, atol=1e-4, rtol=1e-3)
            print(f"  Close to {baseline_name}: {close}")

        print()

    # Performance comparison
    if device.type == "cuda" and len(successful_results) > 1:
        print("Performance Comparison:")
        print("-" * 25)

        # Find fastest as baseline
        fastest_time = min(r["time"] for r in successful_results.values())
        fastest_name = [
            k for k, v in successful_results.items() if v["time"] == fastest_time
        ][0]

        for name, result in successful_results.items():
            time_ratio = result["time"] / fastest_time if fastest_time > 0 else 1
            print(f"{name}:")
            print(f"  Time: {result['time']:.2f} ms ({time_ratio:.2f}x)")
            print(f"  Memory: {result['memory']:.2f} MB")

        print()

    # Recommendations
    print("Recommendations:")
    print("-" * 15)

    if "flex" in successful_results:
        print(
            "✓ FlexAttention is working and can be used for true sliding window attention"
        )
        print("✓ FlexAttention avoids the chunking limitations of LocalAttention")
        print("✓ Consider using FlexAttention for exact sliding window behavior")
    elif "flex" in results and not results["flex"]["success"]:
        print("⚠️  FlexAttention failed - likely need PyTorch nightly")
        print(
            "   Install with: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121"
        )
        print("✓ Fall back to TrueSlidingWindowEncoderLayer for exact sliding window")

    if "true_sliding_window" in successful_results:
        print(
            "✓ TrueSlidingWindowEncoderLayer works as backup for exact sliding window"
        )

    if "local" in successful_results:
        print("✓ LocalAttention works but uses chunking (different numerical results)")

    return results


def test_bridge_model_integration():
    """Test FlexAttention with a simplified BRIDGE-like model."""
    print("\nTesting BRIDGE Model Integration")
    print("=" * 60)

    # Simplified BRIDGE-like model using FlexAttention
    class SimplifiedBRIDGE(nn.Module):
        def __init__(self, vocab_size, d_model, nhead, num_layers, window_size):
            super().__init__()

            # Embeddings
            self.embedding = nn.Embedding(vocab_size, d_model)

            # Orthography encoder with FlexAttention
            self.orth_encoder = EncoderLocal(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                device="cuda" if torch.cuda.is_available() else "cpu",
                window_size=window_size,
                causal=False,  # Bidirectional for orthography
                attention_type="flex",
            )

            # Phonology encoder with FlexAttention
            self.phon_encoder = EncoderLocal(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                device="cuda" if torch.cuda.is_available() else "cpu",
                window_size=window_size,
                causal=True,  # Causal for phonology (autoregressive)
                attention_type="flex",
            )

            # Output projection
            self.output_proj = nn.Linear(d_model, vocab_size)

        def forward(self, orth_input, phon_input):
            # Embed inputs
            orth_emb = self.embedding(orth_input)
            phon_emb = self.embedding(phon_input)

            # Encode with FlexAttention
            orth_encoded = self.orth_encoder(orth_emb)
            phon_encoded = self.phon_encoder(phon_emb)

            # Simple output (just phonology for demo)
            output = self.output_proj(phon_encoded)

            return output

    # Test parameters
    vocab_size = 1000
    d_model = 256
    nhead = 8
    num_layers = 2
    window_size = 32
    batch_size = 2
    seq_len = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Create model
        model = SimplifiedBRIDGE(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            window_size=window_size,
        ).to(device)

        # Create test inputs
        orth_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        phon_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Forward pass
        output = model(orth_input, phon_input)

        print(f"✓ Model created successfully")
        print(f"✓ Orthography input shape: {orth_input.shape}")
        print(f"✓ Phonology input shape: {phon_input.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ FlexAttention integration successful!")

        return True

    except Exception as e:
        print(f"❌ BRIDGE integration failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test FlexAttention integration
    results = test_flex_attention_integration()

    # Test BRIDGE model integration
    bridge_success = test_bridge_model_integration()

    print("\nFinal Summary:")
    print("=" * 60)

    successful_attention_types = [k for k, v in results.items() if v["success"]]
    print(f"Working attention types: {successful_attention_types}")

    if bridge_success:
        print("✓ FlexAttention successfully integrated with BRIDGE architecture")
    else:
        print("❌ FlexAttention integration with BRIDGE failed")

    if "flex" in successful_attention_types:
        print("\n🎉 FlexAttention is ready for use in BRIDGE!")
        print("   You can now use attention_type='flex' in your EncoderLocal")
        print("   This gives you true sliding window attention without chunking")
    else:
        print(
            "\n⚠️  FlexAttention not available - use TrueSlidingWindowEncoderLayer instead"
        )
