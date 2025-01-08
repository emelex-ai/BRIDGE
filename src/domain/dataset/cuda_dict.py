from typing import Union, List
import logging
import torch


logger = logging.getLogger(__name__)


class CUDADict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to(self, device: torch.device):
        output = {}
        for key in self.keys():
            batches = self[key]
            if isinstance(batches, list):
                try:
                    output[key] = [[val.to(device) for val in batch] for batch in batches]
                except:
                    print(f"batches = {batches}")
                    raise
            elif isinstance(batches, torch.Tensor):
                output[key] = batches.to(device)
            else:
                raise TypeError("Must be list or torch tensor")

        return output
