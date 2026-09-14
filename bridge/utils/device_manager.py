import logging
import os
import platform

import torch
import yaml

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages compute device selection and tensor operations across platforms."""

    def __init__(self, device=None):
        """
        Initializes the device manager.

        If no device is provided, defaults to CPU.
        If a device is provided, it will check for its availability:
          - If 'cuda' is requested and available, use it; otherwise, fall back to CPU.
          - If 'mps' is requested and available (Apple Silicon), use it; otherwise, fall back to CPU.
        """
        self._device = self._resolve(device)
        logger.info(f"Using device: {self._device}")

    @staticmethod
    def _resolve(device) -> torch.device:
        """Turn a device request into the device that will actually be used.

        Shared by ``__init__`` and :meth:`set_device` so the two cannot drift apart, which
        is the failure a setter written as `self.__init__(...)` invites.
        """
        if device is None:
            return torch.device("cpu")

        requested_device = torch.device(device)
        if requested_device.type == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            # Resolve the index now. `torch.device("cuda") != torch.device("cuda:0")` while
            # every tensor torch allocates carries an index, so an unresolved device makes
            # every `tensor.device != self.device` check reject valid input. Resolving here
            # fixes all of them at once, and makes `device_manager.device` print what is
            # actually being used. It sits behind the availability check because
            # `torch.cuda.current_device()` raises with no CUDA runtime.
            if requested_device.index is not None:
                return requested_device
            return torch.device("cuda", torch.cuda.current_device())

        if requested_device.type == "mps":
            if (
                platform.system() == "Darwin"
                and platform.machine() == "arm64"
                and torch.backends.mps.is_available()
            ):
                return requested_device
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")

        return requested_device

    def set_device(self, device: str | torch.device | None) -> torch.device:
        """Repoint the manager, and return the device it settled on.

        Every BRIDGE object snapshots ``device_manager.device`` in its own ``__init__``, so
        this only affects objects constructed afterwards. Calling it too late is a silent
        no-op that leaves everything on CPU, and the symptom is a training run an order of
        magnitude slower than expected with nothing in the log. Call it before building
        anything.
        """
        self._device = self._resolve(device)
        logger.info(f"Using device: {self._device}")
        return self._device

    @property
    def device(self) -> torch.device:
        """Returns the current compute device."""
        return self._device

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

    def to_device(self, tensor_or_module):
        """Moves a tensor or module to the current device."""
        return tensor_or_module.to(self._device)

    def create_tensor(self, *args, **kwargs):
        """Creates a tensor on the current device."""
        kwargs["device"] = self._device
        return torch.tensor(*args, **kwargs)


def load_config(config_path):
    """
    Loads a YAML configuration file and returns its content.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path) as file:
        return yaml.safe_load(file)


# Example usage:
# config_path = "app/config/training_config.yaml"
# config = load_config(config_path)

# device_key = config.get("device", None)

# The process-wide device, selectable without editing code. There is no other supported
# way to reach a GPU: before this, the only working override was assigning the private
# attribute before constructing any BRIDGE object.
device_manager = DeviceManager(device=os.environ.get("BRIDGE_DEVICE", "cpu"))
