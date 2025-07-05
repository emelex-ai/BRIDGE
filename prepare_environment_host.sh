#!/bin/bash

set -e

echo "Setting up shared Python environment (including CUDA packages)..."

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
echo "Shared virtual environment: $SHARED_VENV"

# Load uv
module load python-uv

# Load CUDA module
echo "Loading CUDA module for build headers..."
module load cuda/12.1

# Set up persistent cache
SCRATCH_CACHE="/tmp/scratch/gerlebacher/.cache"
export UV_CACHE_DIR="$SCRATCH_CACHE/uv"
export PIP_CACHE_DIR="$SCRATCH_CACHE/pip"
export HF_HOME="$SCRATCH_CACHE/huggingface"
export XDG_CACHE_HOME="$SCRATCH_CACHE"

echo "Using cache directory: $SCRATCH_CACHE"

# Create/update shared environment
echo "Installing base dependencies with uv..."
uv sync --no-config --extra cuda --extra wandb

# Activate the shared environment
echo "Activating shared virtual environment..."
source "$SHARED_VENV/bin/activate"

# Install PyTorch nightly for FlexAttention support
echo "Installing PyTorch nightly for FlexAttention support..."
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# Install CUDA packages (only if not already installed)
if ! python -c "import apex" 2>/dev/null; then
    echo "Installing Apex..."
    uv pip install pybind11
    uv pip install git+https://github.com/NVIDIA/apex.git --no-build-isolation
else
    echo "✅ Apex already installed"
fi

if ! python -c "import transformer_engine" 2>/dev/null; then
    echo "Installing TransformerEngine..."
    uv pip install transformer-engine
else
    echo "✅ TransformerEngine already installed"
fi

# Verify installations (skip TransformerEngine test on frontend)
echo ""
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import apex; print('Apex: OK')"

# Check FlexAttention availability
if python -c "from torch.nn.attention.flex_attention import flex_attention" 2>/dev/null; then
    echo "✅ FlexAttention: Available"
else
    echo "⚠️  FlexAttention: Not available (PyTorch nightly required)"
fi

# TransformerEngine can't be tested on frontend due to missing CUDA runtime
echo "TransformerEngine: Installed (runtime test skipped - requires GPU node)"

echo ""
echo "✅ Shared environment setup complete!"
echo "📁 Shared virtual environment: $SHARED_VENV"
echo "💾 Cache location: $SCRATCH_CACHE"
echo ""
echo "Notes:"
echo "  - TransformerEngine will work on GPU compute nodes but cannot be tested on frontend"
echo "  - PyTorch nightly installed for FlexAttention support (use attention_type='flex')"
echo "  - BRIDGE architecture supports exact sliding window attention via FlexAttention"
echo ""
echo "Usage from any project folder:"
echo "  source ../../../.venv/bin/activate  # Adjust path as needed"
echo "  python your_script.py"
echo ""
echo "Or use the project-specific script.slurm files" 
