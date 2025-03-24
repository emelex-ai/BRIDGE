import yaml
import torch
import platform
import logging

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
        if device is None:
            self._device = torch.device("cpu")
        else:
            requested_device = torch.device(device)
            if requested_device.type == "cuda":
                if torch.cuda.is_available():
                    self._device = requested_device
                else:
                    logger.warning("CUDA requested but not available. Falling back to CPU.")
                    self._device = torch.device("cpu")
            elif requested_device.type == "mps":
                if (
                    platform.system() == "Darwin"
                    and platform.machine() == "arm64"
                    and torch.backends.mps.is_available()
                ):
                    self._device = requested_device
                else:
                    logger.warning("MPS requested but not available. Falling back to CPU.")
                    self._device = torch.device("cpu")
            else:
                self._device = requested_device
        logger.info(f"Using device: {self._device}")

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
        return torch.tensor(*args, **kwargs, device=self._device)


def load_config(config_path):
    """
    Loads a YAML configuration file and returns its content.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# Example usage:
config_path = "app/config/training_config.yaml"
config = load_config(config_path)

device_key = config.get("device", None)

device_manager = DeviceManager(device=device_key)
