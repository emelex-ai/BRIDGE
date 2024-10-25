from typing import Union, List, Dict
import logging
import torch


logger = logging.getLogger(__name__)


class CUDADict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to(self, device: torch.device) -> Dict:
        logger.info(f"Moving tensors to device: {device}")
        output = {}

        # Iterate through dictionary items
        for key, batches in self.items():
            logger.debug(f"Processing key '{key}' with type: {type(batches)}")
            if isinstance(batches, list):
                # Handle list of lists or list of tensors
                try:
                    output[key] = self._move_to_device(batches, device)
                except Exception as e:
                    logger.error(f"Error processing key '{key}': {e}")
                    raise
            elif isinstance(batches, torch.Tensor):
                # Handle single tensor
                logger.debug(f"Moving tensor for key '{key}' to {device}")
                output[key] = batches.to(device)
            else:
                logger.error(f"Invalid type for key '{key}'. Expected list or torch.Tensor but got {type(batches)}.")
                raise TypeError(f"Value for key '{key}' must be a list or a torch.Tensor.")

        return output

    def _move_to_device(self, data: Union[List, torch.Tensor], device: torch.device) -> Union[List, torch.Tensor]:
        """Helper function to recursively move data to the specified device."""
        if isinstance(data, list):
            logger.debug(f"Recursively moving list of data to {device}")
            return [self._move_to_device(item, device) for item in data]
        elif isinstance(data, torch.Tensor):
            logger.debug(f"Moving tensor to {device}")
            return data.to(device)
        else:
            logger.error(f"Unsupported data type {type(data)} encountered. Expected list or torch.Tensor.")
            raise TypeError(f"Unsupported data type {type(data)}. Expected list or torch.Tensor.")
