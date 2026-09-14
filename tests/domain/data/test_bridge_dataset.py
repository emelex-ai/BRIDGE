"""
Test suite for the BridgeDataset class, focusing on proper integration
with BridgeEncoding dataclass and comprehensive functionality testing.
"""

import json
import pickle
from unittest.mock import Mock, patch

import pytest
import torch

from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import BridgeEncoding, EncodingComponent
from bridge.domain.tokenizer import BridgeTokenizer
from bridge.infra.clients.gcp.gcs_client import GCSClient


@pytest.fixture
def mock_gcs_client():
    """A no‐op GCS client stub for BridgeDataset."""
    return Mock(spec=GCSClient)


@pytest.fixture
def mock_dataset_file(tmp_path):
    """Create a temporary dataset file with known test data.

    Uses the columnar pkl format expected by ``BridgeDataset._load_raw_dataframe``:
    ``{"word_raw": [...], "language": [...]}``.
    """
    test_data = {
        "word_raw": ["cat", "dog"],
        "language": ["EN", "EN"],
    }

    file_path = tmp_path / "test_dataset.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(test_data, f)

    return str(file_path)


@pytest.fixture
def mock_cmudict_file(tmp_path):
    """Create a temporary CMUdict JSON file with known test data.

    Uses the nested-by-language format expected by ``PhonemeTokenizer``:
    ``{word: {lang_code: [[phonemes], ...]}}``.
    """
    test_data = {
        "the": {"en": [["DH", "IY0"]]},
        "read": {"en": [["R", "IY1", "D"]]},
        "finance": {"en": [["F", "AY1", "N", "AE0", "N", "S"]]},
    }

    file_path = tmp_path / "test_cmudict_file.json"
    with open(file_path, "w") as f:
        json.dump(test_data, f)

    return str(file_path)


class MockDatasetConfig:
    """Mock implementation of DatasetConfig with updated attributes."""

    def __init__(self, **kwargs):
        # Updated attributes based on new DatasetConfig
        self.dataset_filepath = kwargs.get("dataset_filepath", "data.csv")
        self.device = kwargs.get("device", "cpu")
        self.custom_cmudict_path = kwargs.get("custom_cmudict_path", None)


@pytest.fixture
def dataset_config(mock_dataset_file, mock_cmudict_file):
    """Create a DatasetConfig with test parameters."""
    return MockDatasetConfig(
        dataset_filepath=mock_dataset_file, custom_cmudict_path=mock_cmudict_file
    )


_TOKENIZER = BridgeTokenizer()


def create_test_encoding(word: str, device: torch.device) -> BridgeEncoding:
    """A real encoding for a lexicon word, moved to ``device``.

    Hand-building this used to mean restating the tokenizer's output shape here, which
    then had to be migrated in lockstep with it. Encoding for real costs one lexicon
    parse for the module and cannot drift.
    """
    encoding = _TOKENIZER.encode(word)
    assert encoding is not None, f"{word!r} is missing from the pronunciation lexicon"
    return encoding.to(device)


@pytest.fixture
def mock_bridge_tokenizer(mock_cmudict_file):
    """A real tokenizer for the config's lexicon, wrapped so calls can be counted.

    ``wraps`` keeps real behaviour, since encoding is the thing under test in most of these
    cases, while still recording ``encode.call_count``. A hand-written double has to
    restate the tokenizer's batch contract, which is how this layer silently stopped
    matching it.
    """
    real = BridgeTokenizer(custom_cmudict_path=mock_cmudict_file)
    tokenizer = Mock(spec=BridgeTokenizer, wraps=real)
    # `spec` exposes class attributes only; these are set in __init__.
    tokenizer.custom_cmudict_path = real.custom_cmudict_path
    return tokenizer


@pytest.fixture
def bridge_dataset(dataset_config, mock_bridge_tokenizer, mock_gcs_client):
    """Create a BridgeDataset instance with mocked components."""
    dataset = BridgeDataset(dataset_config, mock_gcs_client, tokenizer=mock_bridge_tokenizer)
    dataset.mock_tokenizer = mock_bridge_tokenizer
    return dataset


def test_dataset_initialization(bridge_dataset, mock_dataset_file):
    """Test dataset initialization with proper configuration."""
    assert isinstance(bridge_dataset, BridgeDataset)
    assert len(bridge_dataset.words) == 2
    assert "cat" in bridge_dataset.words
    assert "dog" in bridge_dataset.words
    assert bridge_dataset.device == torch.device("cpu")


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
    # Shape is (batch=1, seq_len=6): [LANG, BOS, c, a, t, EOS]
    assert orth.enc_input_ids.shape == (1, 6)
    assert orth.enc_input_ids.device == bridge_dataset.device

    phon = item.phonological
    # Phoneme row ids, the same (batch, sequence) shape as the orthographic side.
    assert torch.is_tensor(phon.enc_input_ids)
    assert phon.enc_input_ids.shape[0] == 1
    assert phon.enc_input_ids.device == bridge_dataset.device


def test_get_item_by_word(bridge_dataset):
    """Test accessing items by word string."""
    item = bridge_dataset["cat"]
    assert isinstance(item, BridgeEncoding)
    assert hasattr(item, "orthographic")
    assert hasattr(item, "phonological")
    assert isinstance(item.orthographic, EncodingComponent)
    assert isinstance(item.phonological, EncodingComponent)
    # [LANG=7 ("EN"), BOS=0, c=21, a=19, t=38, EOS=1]. The fixture tags every word
    # as "EN", and EN sits at index 7 in the vocab (after 6 special tokens + "--").
    assert torch.equal(
        item.orthographic.enc_input_ids,
        torch.tensor([[7, 0, 21, 19, 38, 1]], device=bridge_dataset.device),
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
    """A repeated lookup must not re-encode."""
    dataset = BridgeDataset(dataset_config, mock_gcs_client, tokenizer=mock_bridge_tokenizer)

    _ = dataset[0]
    assert mock_bridge_tokenizer.encode.call_count > 0, "nothing was encoded; the spy is inert"
    first_call_count = mock_bridge_tokenizer.encode.call_count

    _ = dataset[0]
    assert mock_bridge_tokenizer.encode.call_count == first_call_count


def test_device_movement(dataset_config, mock_bridge_tokenizer, mock_gcs_client):
    """Test moving dataset between devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dataset = BridgeDataset(dataset_config, mock_gcs_client, tokenizer=mock_bridge_tokenizer)
    assert dataset.device.type == "cpu"

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
    assert torch.equal(single.phonological.enc_pad_mask, batch.phonological.enc_pad_mask)


def test_shuffle_functionality(bridge_dataset):
    """Test dataset shuffling maintains data consistency."""
    original_words = bridge_dataset.words.copy()
    bridge_dataset.shuffle(1)

    assert len(bridge_dataset.words) == len(original_words)
    assert set(bridge_dataset.words) == set(original_words)
    assert bridge_dataset.words[1:] == original_words[1:]


def test_shuffle_preserves_language_alignment(bridge_dataset):
    """Each word's language tag must follow the word through a shuffle."""
    original_pairs = list(zip(bridge_dataset.words, bridge_dataset.languages, strict=True))
    bridge_dataset.shuffle(len(bridge_dataset.words))
    shuffled_pairs = list(zip(bridge_dataset.words, bridge_dataset.languages, strict=True))
    # The set of (word, language) pairs must be identical pre- and post-shuffle.
    assert set(original_pairs) == set(shuffled_pairs)


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


def test_error_handling_invalid_encodings(dataset_config, mock_bridge_tokenizer, mock_gcs_client):
    """Test handling of invalid encodings from tokenizer."""
    dataset = BridgeDataset(dataset_config, mock_gcs_client, tokenizer=mock_bridge_tokenizer)

    # Patch the per-word encoder rather than the tokenizer, to bypass the lru_cache.
    with patch.object(dataset, "_encode_single_word", return_value=None):
        with pytest.raises(RuntimeError, match="Failed to encode word"):
            _ = dataset[0]


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


def test_vocabulary_size_properties(dataset_config, mock_bridge_tokenizer, mock_gcs_client):
    """Vocabulary sizes are taken from the tokenizer, not derived independently."""
    mock_bridge_tokenizer.get_vocabulary_sizes.return_value = {
        "orthographic": 100,
        "phonological": 200,
    }
    dataset = BridgeDataset(dataset_config, mock_gcs_client, tokenizer=mock_bridge_tokenizer)

    assert dataset.orthographic_vocabulary_size == 100
    assert dataset.phonological_vocabulary_size == 200
