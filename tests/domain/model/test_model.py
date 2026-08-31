import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pickle
from unittest.mock import Mock

import pytest
import torch

from bridge.domain.datamodels import ModelConfig, VocabSpec
from bridge.domain.model import Model
from tests.vocab import PHONEME_TABLE, TEST_VOCAB


class MockBridgeDataset:
    """Mock implementation of BridgeDataset for testing Model."""

    def __init__(self, **kwargs):
        self.device = torch.device(kwargs.get("device", "cpu"))
        self.orthographic_vocabulary_size = kwargs.get("orthographic_vocabulary_size", 49)
        # Not a free parameter: Model derives phoneme embeddings from phonreps.csv and
        # rejects a vocab size that disagrees with it.
        self.phonological_vocabulary_size = kwargs.get(
            "phonological_vocabulary_size", PHONEME_TABLE.vocab_size
        )

        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.get_vocabulary_sizes.return_value = {
            "orthographic": self.orthographic_vocabulary_size,
            "phonological": self.phonological_vocabulary_size,
        }


def _vocab_spec_for(mock_dataset: MockBridgeDataset) -> VocabSpec:
    """Build a VocabSpec matching a MockBridgeDataset's vocab sizes.

    Derived from TEST_VOCAB rather than hand-written, so these models exercise the same
    phoneme-table fingerprint as every other test rather than the missing-fingerprint
    branch. Only the vocabulary sizes differ.
    """
    return TEST_VOCAB.model_copy(
        update={
            "orth_vocab_size": mock_dataset.orthographic_vocabulary_size,
            "phon_vocab_size": mock_dataset.phonological_vocabulary_size,
        }
    )


@pytest.fixture
def mock_dataset():
    """Fixture for mock BridgeDataset."""
    return MockBridgeDataset(
        orthographic_vocabulary_size=49,
        phonological_vocabulary_size=PHONEME_TABLE.vocab_size,
    )


@pytest.fixture
def model_config(mock_dataset):
    """Fixture for ModelConfig with vocab spec populated from the mock dataset."""
    return ModelConfig(
        num_phon_enc_layers=1,
        num_orth_enc_layers=1,
        num_mixing_enc_layers=1,
        num_phon_dec_layers=1,
        num_orth_dec_layers=1,
        d_model=64,
        nhead=2,
        d_embedding=1,
        seed=42,
        vocab=_vocab_spec_for(mock_dataset),
    )


@pytest.fixture
def model(model_config):
    """Fixture for initializing the Model."""
    return Model(model_config)


def test_embed_orth_tokens(model: Model):
    model.eval()
    with open("tests/domain/model/data/embed_orth_tokens_test_data.pkl", "rb") as f:
        data = pickle.load(f)

    input = data["input"]
    expected_output = data["output"]
    output = model.embed_orth_tokens(input)
    assert torch.allclose(output, expected_output, atol=1e-5), (
        "Output does not match expected values."
    )


def test_generate_triangular_mask(model: Model):
    """A causal mask: strictly-upper-triangular True."""
    mask = model.generate_triangular_mask(12)
    assert torch.equal(mask, torch.triu(torch.ones(12, 12, dtype=torch.bool), 1))


def test_model_initialization_with_dataset(mock_dataset, model_config):
    """Test the model picks up vocab sizes from ModelConfig.vocab."""
    model = Model(model_config)

    # Vocab sizes flow from config.vocab (which was built from mock_dataset's sizes)
    assert model.orthographic_vocabulary_size == mock_dataset.orthographic_vocabulary_size
    assert model.phonological_vocabulary_size == mock_dataset.phonological_vocabulary_size

    # Verify hardcoded sequence lengths
    assert model.max_orth_seq_len == 30
    assert model.max_phon_seq_len == 30

    # Verify embedding dimensions
    assert model.orthography_embedding.num_embeddings == mock_dataset.orthographic_vocabulary_size
    assert model.phonology_embedding.num_embeddings == mock_dataset.phonological_vocabulary_size
    assert model.orth_position_embedding.num_embeddings == 30
    assert model.phon_position_embedding.num_embeddings == 30


def test_gpu_availability():
    """Test GPU availability and basic tensor operations."""
    from bridge.utils import device_manager

    # Create test tensor
    x = device_manager.create_tensor([[1.0, 2.0], [3.0, 4.0]])
    y = device_manager.create_tensor([[5.0, 6.0], [7.0, 8.0]])

    # Perform computation
    z = torch.matmul(x, y)

    # Ensure computation was done on the right device
    assert z.device.type == device_manager.device.type
    device_manager.synchronize()  # Ensure computation is complete

    # Test basic operations
    result = z.cpu().numpy()  # Move back to CPU for comparison
    assert result.shape == (2, 2)
