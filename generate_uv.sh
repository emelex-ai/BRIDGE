#!/bin/bash

# setup_uv_project.sh
# Script to create a new uv project equivalent to the Poetry BRIDGE project

set -e  # Exit on any error

echo "🚀 Setting up BRIDGE project with uv..."

# Create project directory (optional - run this in existing directory)
# mkdir -p bridge && cd bridge

# Create virtual environment with specific Python version
echo "🐍 Creating virtual environment..."
uv venv --python 3.12

# Create complete pyproject.toml file
echo "📝 Creating pyproject.toml..."
cat > pyproject.toml << 'EOF'
[project]
name = "bridge"
version = "0.1.0"
description = "A computational model for naming printed words that bridges orthographic and phonological representations"
authors = [
    {name = "Nathan Crock", email = "nathan@emelex.com"}
]
readme = "README.md"
requires-python = ">=3.12,<3.13"
keywords = ["nlp", "psycholinguistics", "phonology", "orthography", "deep-learning"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "nltk>=3.8.1,<4.0.0",
    "numpy>=1.26.0,<2.0.0",
    "pandas>=2.1.2,<3.0.0",
    "protobuf>=5.28.2,<6.0.0",
    "pydantic>=2.9.2,<3.0.0",
    "pyyaml>=6.0.2,<7.0.0",
    "torch>=2.4.1,<3.0.0",
    "tqdm>=4.66.5,<5.0.0",
]

[project.optional-dependencies]
wandb = ["wandb>=0.18.3,<1.0.0"]
gcp = ["google-cloud-storage>=3.1.0,<4.0.0"]
all = ["wandb>=0.18.3,<1.0.0", "google-cloud-storage>=3.1.0,<4.0.0"]
dev = [
    "deptry>=0.23.0,<1.0.0",
    "pytest>=8.3.5,<9.0.0",
    "pytest-mock>=3.14.0,<4.0.0",
]

[project.urls]
Homepage = "https://github.com/emelex-ai/BRIDGE"
Repository = "https://github.com/emelex-ai/BRIDGE"
Documentation = "https://github.com/emelex-ai/BRIDGE"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["bridge"]

[tool.hatch.build.targets.wheel.sources]
"bridge/data" = "bridge/data"

[tool.pytest.ini_options]
pythonpath = "."
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v -s"
EOF

# Install dependencies using uv sync
echo "📚 Installing all dependencies..."

# Create project structure
echo "📁 Creating project structure..."
mkdir -p bridge/data
mkdir -p tests

# Create basic Python package files
touch bridge/__init__.py
touch tests/__init__.py

# Create a basic README
cat > README_uv.md << 'EOF'
# BRIDGE

A computational model for naming printed words that bridges orthographic and phonological representations.

## Installation

```bash
# Install with uv
uv sync

# Install with optional dependencies
uv sync --extra wandb --extra gcp --extra all

# Install with development dependencies
uv sync --extra dev
```

## Usage

```bash
# Run with uv
uv run python your_script.py

# Run tests
uv run pytest
```
EOF

# Sync dependencies
echo "🔄 Syncing dependencies..."
uv sync

# Display activation instructions
echo ""
echo "✅ Project setup complete!"
echo ""
echo "📋 To activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "🔧 Common uv commands:"
echo "   uv sync                    # Install dependencies"
echo "   uv sync --extra dev        # Install with dev dependencies"
echo "   uv sync --extra all        # Install with all optional dependencies"
echo "   uv run python script.py    # Run Python scripts"
echo "   uv run pytest             # Run tests"
echo "   uv add package_name        # Add new dependency"
echo "   uv remove package_name     # Remove dependency"
echo ""
echo "🎉 Your BRIDGE project is ready!"
