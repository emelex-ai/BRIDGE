import logging
import os
import platform

import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """The process's choice of device.

    ``None`` is CPU. A request for ``cuda`` or ``mps`` that the machine cannot serve falls
    back to CPU with a warning, not an error.
    """

    def __init__(self, device=None):
        self._device = self._resolve(device)
        logger.info(f"Using device: {self._device}")

    @staticmethod
    def _resolve(device) -> torch.device:
        if device is None:
            return torch.device("cpu")

        requested_device = torch.device(device)
        if requested_device.type == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return torch.device("cpu")
            # Resolve the index now. `torch.device("cuda") != torch.device("cuda:0")` while
            # every tensor torch allocates carries an index, so an unresolved device makes
            # every `tensor.device != self.device` check reject valid input. Behind the
            # availability check because `torch.cuda.current_device()` raises with no CUDA
            # runtime.
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

        The tokenizers and ``Model`` read ``device_manager.device`` once, in their own
        ``__init__``, so this affects only what is built afterwards. Called later it is a
        silent no-op for everything that already exists. Call it before building anything.
        """
        self._device = self._resolve(device)
        logger.info(f"Using device: {self._device}")
        return self._device

    @property
    def device(self) -> torch.device:
        return self._device


# Read once, at import. After that, `set_device` is the way to change it.
device_manager = DeviceManager(device=os.environ.get("BRIDGE_DEVICE", "cpu"))
