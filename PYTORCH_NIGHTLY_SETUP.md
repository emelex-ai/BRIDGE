# PyTorch Nightly Setup for FlexAttention

## Why PyTorch Nightly?

The BRIDGE architecture uses FlexAttention for true sliding window attention, which requires PyTorch nightly builds. FlexAttention provides:

- ✅ **True sliding window semantics** (no chunking artifacts)
- ✅ **O(L×W) efficiency** (memory and compute)
- ✅ **Exact numerical results** (perfect for experiments)

## Installation

### Prerequisites
Make sure your UV environment is activated:
```bash
source .venv/bin/activate  # or however you activate your environment
```

### Install PyTorch Nightly
```bash
# For CPU (development/testing)
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# For CUDA (if you have GPU support)
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from torch.nn.attention.flex_attention import flex_attention; print('✅ FlexAttention available!')"
```

## Usage in BRIDGE

### FlexAttention Encoder
```python
from bridge.domain.model.encoder_local import EncoderLocal

encoder = EncoderLocal(
    d_model=512,
    nhead=8,
    num_layers=4,
    window_size=64,
    causal=True,
    attention_type="flex"  # Use FlexAttention
)
```

### Available Attention Types
- `"flex"` - FlexAttention (requires nightly, exact sliding window)
- `"local"` - LocalAttention (stable PyTorch, chunked approximation)
- `"true_sliding_window"` - Custom implementation (inefficient O(L²))

## Fallback Behavior

If PyTorch nightly is not available, the code will automatically fall back to manual attention computation, so the codebase remains compatible with stable PyTorch for basic functionality.

## Important Notes

- **PyTorch nightly can be unstable** - use at your own risk
- **For production**, consider waiting for FlexAttention in stable PyTorch (likely 2.6+)
- **For experiments requiring exact sliding window**, nightly is currently the only option 