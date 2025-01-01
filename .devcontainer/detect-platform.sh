#!/bin/bash

# Detect the host operating system and architecture
OS=$(uname -s)
ARCH=$(uname -m)

# Initialize variables
DOCKERFILE="Dockerfile.cpu.dev"  # Default fallback
PLATFORM_ARGS='[]'  # Initialize empty, will be filled with JSON array string

# Detect Linux with NVIDIA GPU
if [ "$OS" = "Linux" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        DOCKERFILE="Dockerfile.gpu.dev"
        # For GPU, we create a JSON array string
        PLATFORM_ARGS='["--gpus", "all"]'
    fi
fi

# since it already contains the proper JSON string format
echo "{\"dockerfile\":\"$DOCKERFILE\",\"platformArgs\":$PLATFORM_ARGS}"