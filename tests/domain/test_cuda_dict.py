import pytest
import torch
from traindata import Traindata
from src.domain.dataset import CUDADict


def test_cuda_dict_to_tensor():
    # Setup
    data = {"tensor_key": torch.tensor([1, 2, 3]), "list_key": [torch.tensor([4, 5]), torch.tensor([6, 7])]}
    cuda_dict = CUDADict(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Action
    moved_dict = cuda_dict.to(device)

    # Assert
    for key in data:
        if isinstance(data[key], torch.Tensor):
            assert moved_dict[key].device.type == device.type
        elif isinstance(data[key], list):
            for tensor in data[key]:
                assert tensor.to(device).device.type == device.type
