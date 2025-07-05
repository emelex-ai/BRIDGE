# Sliding Window Attention Test Suite

This test suite verifies that the local attention module works correctly by comparing full attention with sliding window mask versus local attention with the same window size.

## Overview

The test compares two attention implementations:
1. **Full Attention with Sliding Window Mask**: Standard attention computation with a manually applied sliding window mask
2. **Local Attention**: Efficient sliding window attention using the `local_attention` library

Both implementations should produce similar results when using the same random inputs, confirming that the local attention module is working correctly.

## Files

### Core Test Files
- `bridge/domain/model/test_local_attention_gpu.py` - Main GPU test script
- `bridge/domain/model/test_sliding_window_comparison.py` - Comprehensive comparison with visualization
- `bridge/domain/model/true_sliding_window_attention.py` - True sliding window attention implementation

### Launcher Scripts
- `run_sliding_window_test.py` - Simple runner for the GPU test
- `launch_sliding_window_test.py` - Python launcher that submits jobs to SLURM
- `script.slurm` - SLURM batch script for GPU execution

## Usage

### Option 1: Direct Local Testing (if you have GPU access)
```bash
# Run the test directly on a machine with GPU
python run_sliding_window_test.py
```

### Option 2: SLURM Submission (recommended for cluster environments)
```bash
# Submit the test to SLURM and monitor progress
python launch_sliding_window_test.py
```

### Option 3: Manual SLURM Submission
```bash
# Submit manually to SLURM
sbatch script.slurm run_sliding_window_test.py
```

## Test Configuration

The test uses the following default parameters:
- **Batch size**: 2
- **Sequence length**: 512
- **Model dimension**: 256
- **Number of heads**: 8
- **Head dimension**: 32 (256 / 8)
- **Window size**: 64
- **Random seed**: 42

## Expected Output

The test will output:
1. **GPU Information**: Device name, CUDA version
2. **Test Configuration**: All parameters used
3. **Full Attention Results**: Output shape, timing, memory usage, statistics
4. **Local Attention Results**: Output shape, timing, memory usage, statistics
5. **Comparison Results**: 
   - Maximum difference between outputs
   - Mean difference between outputs
   - Relative error
   - Whether outputs are close (within tolerance)
   - Speed comparison (speedup factor)
   - Memory usage comparison

### Success Criteria

The test passes if:
- Both implementations run without errors
- Output tensors have the same shape
- Outputs are numerically close (within `1e-4` absolute tolerance and `1e-3` relative tolerance)

## Test Details

### Sliding Window Mask Creation

The test creates a sliding window mask where each position can only attend to positions within a window of size `W`. For causal attention, this is combined with a lower triangular mask.

```python
def create_sliding_window_mask(seq_len: int, window_size: int, causal: bool = True):
    # Each position i can attend to positions [max(0, i-W+1), min(seq_len, i+W)]
    # If causal=True, also apply lower triangular constraint
```

### Full Attention Implementation

Standard attention computation with manual masking:
```python
scores = torch.einsum('bhid,bhjd->bhij', q, k) * scale
scores = scores.masked_fill(~mask, -float('inf'))
attn_weights = F.softmax(scores, dim=-1)
output = torch.einsum('bhij,bhjd->bhid', attn_weights, v)
```

### Local Attention Implementation

Uses the `local_attention` library with:
- `window_size`: Size of the sliding window
- `causal=True`: Causal masking enabled
- `look_backward=0`, `look_forward=0`: No chunking overlap (true sliding window)
- `autopad=True`: Automatic padding
- `exact_windowsize=True`: Exact window size enforcement

## Performance Expectations

Local attention should provide:
- **Memory efficiency**: O(L×W) instead of O(L²) memory complexity
- **Speed improvement**: Especially for longer sequences
- **Numerical accuracy**: Results should match full attention within tolerance

## Troubleshooting

### Common Issues

1. **CUDA not available**: Ensure you're running on a GPU-enabled machine
2. **Import errors**: Make sure `local_attention` is installed in your environment
3. **Memory errors**: Reduce sequence length or batch size if you encounter OOM errors
4. **Numerical differences**: Small differences are expected due to different computation orders

### Debug Mode

To run with more verbose output, you can modify the test parameters directly in the script or add debug prints.

## Integration with BRIDGE

This test suite is designed to validate the local attention module before integrating it into the full BRIDGE architecture. Once validated, you can:

1. Replace standard attention layers with sliding window attention
2. Implement sliding cross-attention for encoder-decoder architectures
3. Conduct memory studies across multiple layers
4. Scale to larger models and datasets

## Next Steps

After successful validation:
1. Integrate sliding window attention into BRIDGE encoder/decoder
2. Implement sliding cross-attention mechanisms
3. Conduct multi-layer memory analysis
4. Test with real datasets (e.g., Wiki-103)
5. Implement multi-GPU parallelization strategies

## Dependencies

- PyTorch with CUDA support
- `local_attention` library
- `einops` for tensor operations
- Optional: `matplotlib` and `seaborn` for visualization (in comprehensive test)

## Example Output

```
Testing Sliding Window Attention on GPU
==================================================
Using device: cuda
GPU: NVIDIA A100-SXM4-40GB
CUDA Version: 12.1

Test Configuration:
  Batch size: 2
  Sequence length: 512
  Model dimension: 256
  Number of heads: 8
  Head dimension: 32
  Window size: 64
  Random seed: 42

1. Testing full attention with sliding window mask...
Mask shape: torch.Size([512, 512])
Mask sparsity: 87.5% masked
Full attention results:
  Output shape: torch.Size([2, 8, 512, 32])
  Time: 15.23 ms
  Memory used: 45.67 MB
  Output mean: 0.001234
  Output std: 0.987654

2. Testing local attention...
Local attention results:
  Output shape: torch.Size([2, 8, 512, 32])
  Time: 8.91 ms
  Memory used: 12.34 MB
  Output mean: 0.001235
  Output std: 0.987653

3. Comparison Results:
  Max difference: 0.000012
  Mean difference: 0.000003
  Relative error: 0.000004
  Outputs close (1e-4 atol, 1e-3 rtol): True
  Time speedup: 1.71x
  Memory reduction: 73.0%

==================================================
TEST RESULT: PASSED
==================================================
``` 