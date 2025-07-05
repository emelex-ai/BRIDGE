#!/bin/bash

set -e

echo "Setting up Python environment for Mac (including PyTorch nightly)..."

PROJECT_ROOT="$PWD"
SHARED_VENV="$PROJECT_ROOT/.venv"

# Clear any existing virtual environment variables
unset VIRTUAL_ENV
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found. Run this script from the root directory."
    exit 1
fi

echo "Project root: $PROJECT_ROOT"
echo "Virtual environment: $SHARED_VENV"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  source ~/.bashrc  # or restart your terminal"
    exit 1
fi

# Create/update environment
echo "Installing base dependencies with uv..."
uv sync --no-config --extra wandb

# Activate the environment
echo "Activating virtual environment..."
source "$SHARED_VENV/bin/activate"

# Install PyTorch nightly for FlexAttention support (CPU version for Mac)
echo "Installing PyTorch nightly for FlexAttention support (CPU)..."
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Verify installations
echo ""
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check FlexAttention availability
if python -c "from torch.nn.attention.flex_attention import flex_attention" 2>/dev/null; then
    echo "✅ FlexAttention: Available"
else
    echo "⚠️  FlexAttention: Not available (PyTorch nightly required)"
fi

# Check other key dependencies
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"

if python -c "import wandb" 2>/dev/null; then
    echo "✅ WandB: Available"
else
    echo "⚠️  WandB: Not available (install with --extra wandb)"
fi

echo ""
echo "✅ Mac environment setup complete!"
echo "📁 Virtual environment: $SHARED_VENV"
echo ""
echo "Notes:"
echo "  - PyTorch nightly installed for FlexAttention support (use attention_type='flex')"
echo "  - BRIDGE architecture supports exact sliding window attention via FlexAttention"
echo "  - CPU-only PyTorch (CUDA not available on Mac)"
echo ""
echo "Usage:"
echo "  source .venv/bin/activate"
echo "  python your_script.py"
echo ""
echo "To test FlexAttention:"
echo "  python -c \"from bridge.domain.model.transformer_flex_attention import FlexAttentionEncoderLayer; print('FlexAttention ready!')\"" 