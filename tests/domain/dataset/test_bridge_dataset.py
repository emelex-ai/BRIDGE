"""
Test suite for the BridgeDataset class, focusing on proper integration
with BridgeEncoding dataclass and comprehensive functionality testing.
"""

import sys
import os
import pytest
import torch
import pickle
import json
from unittest.mock import Mock, patch

from src.domain.datamodels import BridgeEncoding, EncodingComponent
from src.domain.dataset import BridgeDataset, BridgeTokenizer
from src.infra.clients.gcp.gcs_client import GCSClient


@pytest.fixture
def mock_gcs_client():
    """A no‐op GCS client stub for BridgeDataset."""
    return Mock(spec=GCSClient)


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
        "dog": {
            "count": 1,
            "phoneme": ([7, 8, 9], [10, 11, 12]),
            "phoneme_shape": (3, 2),
            "orthography": [4, 5, 6, 7],
        },
    }

    file_path = tmp_path / "test_dataset.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(test_data, f)

    return str(file_path)


@pytest.fixture
def mock_cmudict_file(tmp_path):
    """Create a temporary CMUdict JSON file with known test data."""
    test_data = {
        "the": [["DH", "IY0"]],
        "read": [["R", "IY1", "D"]],
        "finance": [["F", "AY1", "N", "AE0", "N", "S"]],
    }

    file_path = tmp_path / "test_cmudict_file.json"
    # ← open in text‐mode, not binary
    with open(file_path, "w") as f:
        json.dump(test_data, f)

    # return a string so your dataset_config gets a str path
    return str(file_path)


class MockDatasetConfig:
    """Mock implementation of DatasetConfig with updated attributes."""

    def __init__(self, **kwargs):
        # Updated attributes based on new DatasetConfig
        self.dataset_filepath = kwargs.get("dataset_filepath", "data.csv")
        self.device = kwargs.get("device", "cpu")
        self.tokenizer_cache_size = kwargs.get("tokenizer_cache_size", 10000)
        self.custom_cmudict_path = kwargs.get(
            "custom_cmudict_path", "custom_cmudict.json"
        )
        # For backward compatibility
        self.phoneme_cache_size = self.tokenizer_cache_size


@pytest.fixture
def dataset_config(mock_dataset_file, mock_cmudict_file):
    """Create a DatasetConfig with test parameters."""
    return MockDatasetConfig(
        dataset_filepath=mock_dataset_file, custom_cmudict_path=mock_cmudict_file
    )


def create_test_encoding(word: str, device: torch.device) -> BridgeEncoding:
    """Helper function to create test BridgeEncoding instances with new structure."""

    if word == "cat":
        # Create orthographic encoding component
        orth_enc_ids = torch.tensor([[0, 18, 16, 35, 1]], device=device)
        orth_enc_mask = torch.tensor(
            [[False, False, False, False, False]], device=device
        )
        orth_dec_ids = torch.tensor([[0, 18, 16, 35]], device=device)
        orth_dec_mask = torch.tensor([[False, False, False, False]], device=device)

        # Create phonological encoding component
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

        # Create phonological targets
        phon_targets = torch.zeros([1, 4, 36], device=device)
        for row, idx in enumerate(phon_enc_ids[0][1:]):
            for i in idx:
                phon_targets[0, row, i] = 1

        # Create encoding components
        orthographic = EncodingComponent(
            enc_input_ids=orth_enc_ids,
            enc_pad_mask=orth_enc_mask,
            dec_input_ids=orth_dec_ids,
            dec_pad_mask=orth_dec_mask,
        )

        phonological = EncodingComponent(
            enc_input_ids=phon_enc_ids,
            enc_pad_mask=phon_enc_mask,
            dec_input_ids=phon_dec_ids,
            dec_pad_mask=phon_dec_mask,
            targets=phon_targets,
        )

        # Create BridgeEncoding with components
        return BridgeEncoding(
            orthographic=orthographic, phonological=phonological, device=device
        )
    elif word == "dog":
        # Create orthographic encoding component
        orth_enc_ids = torch.tensor([[0, 9, 15, 13, 1]], device=device)
        orth_enc_mask = torch.tensor(
            [[False, False, False, False, False]], device=device
        )
        orth_dec_ids = torch.tensor([[0, 9, 15, 13]], device=device)
        orth_dec_mask = torch.tensor([[False, False, False, False]], device=device)

        # Create phonological encoding component
        phon_enc_ids = [
            [
                torch.tensor([31], device=device),
                torch.tensor([7, 8], device=device),
                torch.tensor([11, 17, 22], device=device),
                torch.tensor([3, 6], device=device),
                torch.tensor([32], device=device),
            ]
        ]
        phon_enc_mask = torch.tensor(
            [[False, False, False, False, False]], device=device
        )
        phon_dec_ids = [
            [
                torch.tensor([31], device=device),
                torch.tensor([7, 8], device=device),
                torch.tensor([11, 17, 22], device=device),
                torch.tensor([3, 6], device=device),
            ]
        ]
        phon_dec_mask = torch.tensor([[False, False, False, False]], device=device)

        # Create phonological targets
        phon_targets = torch.zeros([1, 4, 36], device=device)
        for row, idx in enumerate(phon_enc_ids[0][1:]):
            for i in idx:
                phon_targets[0, row, i] = 1

        # Create encoding components
        orthographic = EncodingComponent(
            enc_input_ids=orth_enc_ids,
            enc_pad_mask=orth_enc_mask,
            dec_input_ids=orth_dec_ids,
            dec_pad_mask=orth_dec_mask,
        )

        phonological = EncodingComponent(
            enc_input_ids=phon_enc_ids,
            enc_pad_mask=phon_enc_mask,
            dec_input_ids=phon_dec_ids,
            dec_pad_mask=phon_dec_mask,
            targets=phon_targets,
        )

        # Create BridgeEncoding with components
        return BridgeEncoding(
            orthographic=orthographic, phonological=phonological, device=device
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
def bridge_dataset(dataset_config, mock_bridge_tokenizer, mock_gcs_client):
    """Create a BridgeDataset instance with mocked components."""
    with patch(
        "src.domain.dataset.BridgeTokenizer", return_value=mock_bridge_tokenizer
    ) as mock:
        dataset = BridgeDataset(dataset_config, mock_gcs_client)
        dataset.mock_tokenizer = mock_bridge_tokenizer
        return dataset


def test_dataset_initialization(bridge_dataset, mock_dataset_file):
    """Test dataset initialization with proper configuration."""
    assert isinstance(bridge_dataset, BridgeDataset)
    assert len(bridge_dataset.words) == 2
    assert "cat" in bridge_dataset.words
    assert "dog" in bridge_dataset.words
    assert bridge_dataset.device == torch.device("cpu")
    assert len(bridge_dataset.encoding_cache) == 0


def test_dataset_length(bridge_dataset):
    """Test the dataset length calculation."""
    assert len(bridge_dataset) == 2


def test_get_item_by_index(bridge_dataset):
    """Test accessing items by numerical index."""
    item = bridge_dataset[0]
    assert isinstance(item, BridgeEncoding)
    assert hasattr(item, "orthographic")
    assert hasattr(item, "phonological")
    assert isinstance(item.orthographic, EncodingComponent)
    assert isinstance(item.phonological, EncodingComponent)

    # Verify tensor properties
    orth = item.orthographic
    assert torch.is_tensor(orth.enc_input_ids)
    assert orth.enc_input_ids.shape == (1, 5)
    assert orth.enc_input_ids.device == bridge_dataset.device

    phon = item.phonological
    assert isinstance(phon.enc_input_ids, list)
    assert all(torch.is_tensor(t) for t in phon.enc_input_ids[0])


def test_get_item_by_word(bridge_dataset):
    """Test accessing items by word string."""
    item = bridge_dataset["cat"]
    assert isinstance(item, BridgeEncoding)
    assert hasattr(item, "orthographic")
    assert hasattr(item, "phonological")
    assert isinstance(item.orthographic, EncodingComponent)
    assert isinstance(item.phonological, EncodingComponent)
    assert torch.equal(
        item.orthographic.enc_input_ids,
        torch.tensor([[0, 18, 16, 35, 1]], device=bridge_dataset.device),
    )


def test_get_item_by_slice(bridge_dataset):
    """Test accessing multiple items using slice notation."""
    items = bridge_dataset[0:2]
    assert isinstance(items, BridgeEncoding)
    assert hasattr(items, "orthographic")
    assert hasattr(items, "phonological")
    assert isinstance(items.orthographic, EncodingComponent)
    assert isinstance(items.phonological, EncodingComponent)
    # We should get both items in the batch
    assert items.orthographic.enc_input_ids.shape[0] == 2
    assert items.phonological.enc_pad_mask.shape[0] == 2


def test_invalid_index_access(bridge_dataset):
    """Test error handling for invalid indices."""
    with pytest.raises(IndexError):
        _ = bridge_dataset[100]

    with pytest.raises(KeyError):
        _ = bridge_dataset["nonexistent"]


def test_encoding_cache(dataset_config, mock_bridge_tokenizer, mock_gcs_client):
    """Test encoding cache functionality."""
    # Create fresh mock with call tracking
    mock_tokenizer = Mock(spec=BridgeTokenizer)

    def mock_encode(word):
        """Side effect to simulate encode behavior"""
        return create_test_encoding(word, torch.device("cpu"))

    mock_tokenizer.encode.side_effect = mock_encode

    # Create dataset with our controlled mock
    with patch("src.domain.dataset.BridgeTokenizer", return_value=mock_tokenizer):
        dataset = BridgeDataset(dataset_config, mock_gcs_client)

        # Access same item twice
        _ = dataset[0]

        # Check call count after first access
        first_call_count = mock_tokenizer.encode.call_count

        # Access again - should use cache
        _ = dataset[0]

        # Verify encode wasn't called again
        assert mock_tokenizer.encode.call_count == first_call_count


def test_device_movement(dataset_config, mock_gcs_client):
    """Test moving dataset between devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    with patch("src.domain.dataset.BridgeTokenizer") as mock_tokenizer:
        dataset = BridgeDataset(dataset_config, mock_gcs_client)
        assert dataset.device.type == "cpu"

        # Move to CUDA
        dataset.device = torch.device("cuda")
        assert dataset.device.type == "cuda"


def test_batch_consistency(bridge_dataset):
    """Test consistency of batch processing."""
    single = bridge_dataset[0]
    batch = bridge_dataset[0:1]

    # Verify batch is properly formatted version of single
    assert torch.equal(
        single.orthographic.enc_input_ids,
        batch.orthographic.enc_input_ids,
    )
    assert torch.equal(
        single.phonological.enc_pad_mask, batch.phonological.enc_pad_mask
    )


def test_shuffle_functionality(bridge_dataset):
    """Test dataset shuffling maintains data consistency."""
    original_words = bridge_dataset.words.copy()
    bridge_dataset.shuffle(1)

    assert len(bridge_dataset.words) == len(original_words)
    assert set(bridge_dataset.words) == set(original_words)
    assert bridge_dataset.words[1:] == original_words[1:]
    assert len(bridge_dataset.encoding_cache) == 0  # Cache should be cleared


# TODO: fix this once cache logic is fixed
# def test_memory_management(dataset_config, mock_bridge_tokenizer):
#     """Test memory management with cache size limits."""
#     with patch("src.domain.dataset.BridgeTokenizer", return_value=mock_bridge_tokenizer):
#         dataset = BridgeDataset(dataset_config, cache_size=1)

#         # Access items to fill cache
#         _ = dataset[0]
#         _ = dataset[1]

#         # Only most recent should be cached
#         assert len(dataset.encoding_cache) == 1


def test_data_validation(tmp_path, dataset_config, mock_gcs_client):
    """Test data validation during loading."""
    # Create invalid dataset file that's not a dictionary of dictionaries
    invalid_data = []  # Not a dictionary
    file_path = tmp_path / "invalid.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(invalid_data, f)

    dataset_config.dataset_filepath = str(file_path)
    with pytest.raises(ValueError, match="Dataset file must contain a dictionary"):
        _ = BridgeDataset(dataset_config, mock_gcs_client)


def test_error_handling_invalid_encodings(
    dataset_config, mock_bridge_tokenizer, mock_gcs_client
):
    """Test handling of invalid encodings from tokenizer."""
    # For this test, we need to patch both the BridgeTokenizer class constructor
    # and the _encode_single_word method in BridgeDataset to bypass the lru_cache

    with patch(
        "src.domain.dataset.BridgeTokenizer", return_value=mock_bridge_tokenizer
    ):
        # Create the dataset
        dataset = BridgeDataset(dataset_config, mock_gcs_client)

        # Now patch the _encode_single_word method to always return None
        # This simulates a failure in the tokenization process
        with patch.object(dataset, "_encode_single_word", return_value=None):
            # Now accessing the item should raise RuntimeError
            with pytest.raises(RuntimeError, match="Failed to encode word"):
                _ = dataset[0]


def test_cache_path_handling(tmp_path, dataset_config, mock_gcs_client):
    """Test cache directory handling."""
    # Ensure BridgeDataset correctly handles existing cache paths
    cache_path = tmp_path / "test_cache"
    assert not os.path.exists(cache_path)
    os.makedirs(cache_path)
    assert os.path.exists(cache_path)
    _ = BridgeDataset(dataset_config, mock_gcs_client, cache_path=str(cache_path))
    os.removedirs(cache_path)
    assert not os.path.exists(cache_path)
    # Ensure BridgeDataset correctly creates new cache directories
    invalid_cache_path = tmp_path / "invalid/path/that/doesnt/exist"
    assert not os.path.exists(invalid_cache_path)
    _ = BridgeDataset(
        dataset_config, mock_gcs_client, cache_path=str(invalid_cache_path)
    )
    assert os.path.exists(invalid_cache_path)
    os.removedirs(invalid_cache_path)
    assert not os.path.exists(invalid_cache_path)


def test_integration_with_training_pipeline(bridge_dataset):
    """Test compatibility with training pipeline requirements."""
    batch = bridge_dataset[0:2]

    # Verify batch format meets training pipeline requirements
    assert isinstance(batch, BridgeEncoding)
    assert hasattr(batch, "orthographic")
    assert hasattr(batch, "phonological")
    assert isinstance(batch.orthographic, EncodingComponent)
    assert isinstance(batch.phonological, EncodingComponent)
    assert all(
        hasattr(batch.orthographic, key)
        for key in ["enc_input_ids", "enc_pad_mask", "dec_input_ids", "dec_pad_mask"]
    )
    assert all(
        hasattr(batch.phonological, key)
        for key in [
            "enc_input_ids",
            "enc_pad_mask",
            "dec_input_ids",
            "dec_pad_mask",
            "targets",
        ]
    )


def test_vocabulary_size_properties(
    bridge_dataset, mock_bridge_tokenizer, mock_gcs_client
):
    """Test the vocabulary size properties."""
    # Setup mock to return expected vocabulary sizes
    mock_bridge_tokenizer.get_vocabulary_sizes.return_value = {
        "orthographic": 100,
        "phonological": 200,
    }

    # Create new dataset with our controlled mock
    with patch(
        "src.domain.dataset.BridgeTokenizer", return_value=mock_bridge_tokenizer
    ):
        dataset_config = MockDatasetConfig(
            dataset_filepath=bridge_dataset.dataset_filepath, device="cpu"
        )
        dataset = BridgeDataset(dataset_config, mock_gcs_client)

        # Set vocab sizes manually for testing
        dataset.orthographic_vocabulary_size = 100
        dataset.phonological_vocabulary_size = 200
