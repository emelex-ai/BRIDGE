"""
Test suite for the BridgeDataset class, focusing on proper integration
with BridgeEncoding dataclass and comprehensive functionality testing.
"""

import sys
import os
import pytest
import torch
import pickle
from unittest.mock import Mock, patch

from src.domain.datamodels import DatasetConfig
from src.domain.dataset import BridgeDataset, BridgeTokenizer
from src.domain.datamodels import BridgeEncoding


@pytest.fixture
def mock_dataset_file(tmp_path):
    """Create a temporary dataset file with known test data."""
    test_data = {
        "cat": {
            "count": 1,
            "phoneme": ([1, 2, 3], [4, 5, 6]),
            "phoneme_shape": (3, 2),
            "orthography": [0, 1, 2, 3],
        },
    }

    file_path = tmp_path / "test_dataset.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(test_data, f)

    return str(file_path)


@pytest.fixture
def dataset_config(mock_dataset_file):
    """Create a DatasetConfig with test parameters."""
    return DatasetConfig(
        dataset_filepath=mock_dataset_file,
        dimension_phon_repr=31,
        orthographic_vocabulary_size=49,
        phonological_vocabulary_size=34,
        max_orth_seq_len=100,
        max_phon_seq_len=100,
        device="cpu",  # Explicitly set device for testing
    )


def create_test_encoding(word: str, device: torch.device) -> BridgeEncoding:
    """Helper function to create test BridgeEncoding instances."""
    if word == "cat":
        orth_enc_ids = torch.tensor([[0, 18, 16, 35, 1]], device=device)
        orth_enc_mask = torch.tensor(
            [[False, False, False, False, False]], device=device
        )
        orth_dec_ids = torch.tensor([[0, 18, 16, 35]], device=device)
        orth_dec_mask = torch.tensor([[False, False, False, False]], device=device)
        phon_enc_ids = [
            [
                torch.tensor([31], device=device),
                torch.tensor([4, 6], device=device),
                torch.tensor([14, 15, 17, 22, 29], device=device),
                torch.tensor([2, 6], device=device),
                torch.tensor([32], device=device),
            ]
        ]
        phon_enc_mask = torch.tensor(
            [[False, False, False, False, False]], device=device
        )
        phon_dec_ids = [
            [
                torch.tensor([31], device=device),
                torch.tensor([4, 6], device=device),
                torch.tensor([14, 15, 17, 22, 29], device=device),
                torch.tensor([2, 6], device=device),
            ]
        ]
        phon_dec_mask = torch.tensor([[False, False, False, False]], device=device)
        phon_targets = torch.zeros([1, 4, 36])
        for row, idx in enumerate(phon_enc_ids[0][1:]):
            phon_targets[0, row, idx] = 1

        return BridgeEncoding(
            orth_enc_ids=orth_enc_ids,
            orth_enc_mask=orth_enc_mask,
            orth_dec_ids=orth_dec_ids,
            orth_dec_mask=orth_dec_mask,
            phon_enc_ids=phon_enc_ids,
            phon_enc_mask=phon_enc_mask,
            phon_dec_ids=phon_dec_ids,
            phon_dec_mask=phon_dec_mask,
            phon_targets=phon_targets,
            device=device,
        )
    return None


@pytest.fixture
def mock_bridge_tokenizer():
    """Create a mock BridgeTokenizer with controlled behavior."""
    mock_tokenizer = Mock(spec=BridgeTokenizer)

    def mock_encode(word):
        return create_test_encoding(word, torch.device("cpu"))

    mock_tokenizer.encode.side_effect = mock_encode
    return mock_tokenizer


@pytest.fixture
def bridge_dataset(dataset_config, mock_bridge_tokenizer):
    """Create a BridgeDataset instance with mocked components."""
    with patch(
        "src.domain.dataset.BridgeTokenizer", return_value=mock_bridge_tokenizer
    ) as mock:
        dataset = BridgeDataset(dataset_config)
        dataset.mock_tokenizer = mock_bridge_tokenizer
        return dataset


def test_dataset_initialization(bridge_dataset, mock_dataset_file):
    """Test dataset initialization with proper configuration."""
    assert isinstance(bridge_dataset, BridgeDataset)
    assert len(bridge_dataset.words) == 1
    assert "cat" in bridge_dataset.words
    assert bridge_dataset.device == torch.device("cpu")
    assert len(bridge_dataset.encoding_cache) == 0


def test_dataset_length(bridge_dataset):
    """Test the dataset length calculation."""
    assert len(bridge_dataset) == 1


def test_get_item_by_index(bridge_dataset):
    """Test accessing items by numerical index."""
    item = bridge_dataset[0]
    assert isinstance(item, dict)
    assert set(item.keys()) == {"orthographic", "phonological"}

    # Verify tensor properties
    orth = item["orthographic"]
    assert torch.is_tensor(orth["enc_input_ids"])
    assert orth["enc_input_ids"].shape == (1, 5)
    assert orth["enc_input_ids"].device == bridge_dataset.device

    phon = item["phonological"]
    assert isinstance(phon["enc_input_ids"], list)
    assert all(torch.is_tensor(t) for t in phon["enc_input_ids"][0])


def test_get_item_by_word(bridge_dataset):
    """Test accessing items by word string."""
    item = bridge_dataset["cat"]
    assert isinstance(item, dict)
    assert torch.equal(
        item["orthographic"]["enc_input_ids"],
        torch.tensor([[0, 18, 16, 35, 1]], device=bridge_dataset.device),
    )


def test_get_item_by_slice(bridge_dataset):
    """Test accessing multiple items using slice notation."""
    items = bridge_dataset[0:2]
    assert isinstance(items, dict)
    assert items["orthographic"]["enc_input_ids"].shape[0] == 1
    assert items["phonological"]["enc_pad_mask"].shape[0] == 1


def test_invalid_index_access(bridge_dataset):
    """Test error handling for invalid indices."""
    with pytest.raises(IndexError):
        _ = bridge_dataset[100]

    with pytest.raises(KeyError):
        _ = bridge_dataset["nonexistent"]


def test_encoding_cache(bridge_dataset):
    """Test encoding cache functionality."""
    # Access same item twice - should use cache second time
    _ = bridge_dataset[0]
    _ = bridge_dataset[0]
    assert bridge_dataset.mock_tokenizer.encode.call_count == 1


def test_device_movement(dataset_config):
    """Test moving dataset between devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    with patch("src.domain.dataset.BridgeTokenizer") as mock_tokenizer:
        dataset = BridgeDataset(dataset_config)
        assert dataset.device.type == "cpu"

        # Move to CUDA
        dataset.to(torch.device("cuda"))
        assert dataset.device.type == "cuda"
        mock_tokenizer.return_value.to.assert_called_once_with(torch.device("cuda"))


def test_batch_consistency(bridge_dataset):
    """Test consistency of batch processing."""
    single = bridge_dataset[0]
    print("single = ", single)
    batch = bridge_dataset[0:1]
    print("bastch = ", batch)

    # Verify batch is properly formatted version of single
    assert torch.equal(
        single["orthographic"]["enc_input_ids"],
        batch["orthographic"]["enc_input_ids"][0],
    )
    assert torch.equal(
        single["phonological"]["enc_pad_mask"], batch["phonological"]["enc_pad_mask"][0]
    )


def test_shuffle_functionality(bridge_dataset):
    """Test dataset shuffling maintains data consistency."""
    original_words = bridge_dataset.words.copy()
    bridge_dataset.shuffle(1)

    assert len(bridge_dataset.words) == len(original_words)
    assert set(bridge_dataset.words) == set(original_words)
    assert bridge_dataset.words[1:] == original_words[1:]
    assert len(bridge_dataset.encoding_cache) == 0  # Cache should be cleared


def test_memory_management(dataset_config, mock_bridge_tokenizer):
    """Test memory management with cache size limits."""
    dataset = BridgeDataset(dataset_config, cache_size=1)

    # Access items to fill cache
    _ = dataset[0]
    _ = dataset[1]

    assert len(dataset.encoding_cache) == 1  # Only most recent should be cached


def test_data_validation(tmp_path, dataset_config):
    """Test data validation during loading."""
    # Create invalid dataset file
    invalid_data = {"invalid": 123}  # Not proper format
    file_path = tmp_path / "invalid.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(invalid_data, f)

    dataset_config.dataset_filepath = str(file_path)
    with pytest.raises(ValueError, match="Dataset file must contain a dictionary"):
        _ = BridgeDataset(dataset_config)


def test_error_handling_invalid_encodings(dataset_config, mock_bridge_tokenizer):
    """Test handling of invalid encodings from tokenizer."""
    mock_bridge_tokenizer.encode.return_value = None

    with patch(
        "src.domain.dataset.BridgeTokenizer", return_value=mock_bridge_tokenizer
    ):
        dataset = BridgeDataset(dataset_config)
        with pytest.raises(RuntimeError, match="Failed to encode word"):
            _ = dataset[0]


def test_cache_path_handling(tmp_path, dataset_config):
    """Test cache directory handling."""
    cache_path = tmp_path / "test_cache"
    dataset = BridgeDataset(dataset_config, cache_path=str(cache_path))
    assert os.path.exists(cache_path)

    # Test invalid cache path
    with pytest.raises(OSError):
        _ = BridgeDataset(dataset_config, cache_path="/invalid/path/that/doesnt/exist")


def test_integration_with_training_pipeline(bridge_dataset):
    """Test compatibility with training pipeline requirements."""
    batch = bridge_dataset[0:2]

    # Verify batch format meets training pipeline requirements
    assert isinstance(batch, dict)
    assert set(batch.keys()) == {"orthographic", "phonological"}
    assert all(
        key in batch["orthographic"]
        for key in ["enc_input_ids", "enc_pad_mask", "dec_input_ids", "dec_pad_mask"]
    )
    assert all(
        key in batch["phonological"]
        for key in [
            "enc_input_ids",
            "enc_pad_mask",
            "dec_input_ids",
            "dec_pad_mask",
            "targets",
        ]
    )
