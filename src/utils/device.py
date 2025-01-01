"""
Unified device management for PyTorch computations across different platforms.
This module automatically detects and configures the best available compute device,
whether it's CUDA on Linux or MPS on Apple Silicon.
"""

import torch
import platform
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages compute device selection and tensor operations across platforms."""

    def __init__(self):
        self._device = self._detect_best_device()
        logger.info(f"Using device: {self._device}")

    def _detect_best_device(self):
        """Detects the best available compute device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif (
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def device(self):
        """Returns the current compute device."""
        return self._device

    def to_device(self, tensor_or_module):
        """Moves a tensor or module to the current device."""
        return tensor_or_module.to(self._device)

    def create_tensor(self, *args, **kwargs):
        """Creates a tensor on the current device."""
        return torch.tensor(*args, **kwargs, device=self._device)

    @property
    def is_gpu_available(self):
        """Checks if any GPU (CUDA or MPS) is available."""
        return self._device.type in ("cuda", "mps")

    def synchronize(self):
        """Synchronizes the current device if necessary."""
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        elif self._device.type == "mps":
            torch.mps.synchronize()


# Global device manager instance
device_manager = DeviceManager()
