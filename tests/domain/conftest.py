import pytest
from unittest.mock import Mock
import torch


@pytest.fixture
def sample_wordlist():
    return ["hello", "world", "test"]


@pytest.fixture
def sample_phonology_data():
    return {
        "hello": {
            "enc_input_ids": [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
            "dec_input_ids": [torch.tensor([1, 2, 3])],
            "targets": torch.tensor([[0, 1, 0], [1, 0, 1]]),
        },
        "world": {
            "enc_input_ids": [torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])],
            "dec_input_ids": [torch.tensor([7, 8, 9])],
            "targets": torch.tensor([[1, 0, 1], [0, 1, 0]]),
        },
        "test": {
            "enc_input_ids": [torch.tensor([13, 14, 15]), torch.tensor([16, 17, 18])],
            "dec_input_ids": [torch.tensor([13, 14, 15])],
            "targets": torch.tensor([[1, 1, 0], [0, 0, 1]]),
        },
    }


@pytest.fixture
def mock_traindata(monkeypatch, sample_phonology_data):
    mock = Mock()
    mock.traindata = {
        5: {
            "wordlist": ["hello", "world"],
            "phonSOS": [torch.ones(3), torch.ones(3)],
            "phonEOS": [torch.zeros(3), torch.zeros(3)],
        },
        4: {"wordlist": ["test"], "phonSOS": [torch.ones(3)], "phonEOS": [torch.zeros(3)]},
    }
    monkeypatch.setattr("traindata.Traindata", Mock(return_value=mock))
    return mock
