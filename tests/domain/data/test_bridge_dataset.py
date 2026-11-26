import json
import pickle
from unittest.mock import Mock

import pytest
import torch

from bridge.domain.data import BridgeDataset, CSVReader
from bridge.domain.datamodels import BridgeBatch, BridgeInput, BridgeTarget, Target, Tokens
from bridge.domain.tokenizer import BridgeTokenizer


@pytest.fixture
def mock_gcs_client():
    """Specced against the ``CSVReader`` protocol the dataset declares rather than against a
    concrete cloud client: BRIDGE ships none, so the only contract is ``read_csv``."""
    return Mock(spec=CSVReader)


@pytest.fixture
def mock_dataset_file(tmp_path):
    """The columnar pkl format ``BridgeDataset._load_raw_dataframe`` reads:
    ``{"word_raw": [...], "language": [...], "target_raw": [...]}``. The target is the
    input here, which is repetition."""
    test_data = {
        "word_raw": ["cat", "dog"],
        "language": ["EN", "EN"],
        "target_raw": ["cat", "dog"],
    }

    file_path = tmp_path / "test_dataset.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(test_data, f)

    return str(file_path)


@pytest.fixture
def mock_cmudict_file(tmp_path):
    """The nested-by-language format ``PhonemeTokenizer`` reads:
    ``{word: {lang_code: [[phonemes], ...]}}``."""
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
    def __init__(self, **kwargs):
        self.dataset_filepath = kwargs.get("dataset_filepath", "data.csv")
        self.device = kwargs.get("device", "cpu")
        self.custom_cmudict_path = kwargs.get("custom_cmudict_path", None)


@pytest.fixture
def dataset_config(mock_dataset_file, mock_cmudict_file):
    return MockDatasetConfig(
        dataset_filepath=mock_dataset_file, custom_cmudict_path=mock_cmudict_file
    )


@pytest.fixture
def mock_bridge_tokenizer(mock_cmudict_file):
    """A real tokenizer for the config's lexicon, wrapped so calls can be counted.

    ``wraps`` keeps real behaviour, since encoding is the thing under test in most of these
    cases, while still recording each encoder's ``call_count``. A hand-written double has
    to restate the tokenizer's batch contract, and can drift from it unnoticed.
    """
    real = BridgeTokenizer(custom_cmudict_path=mock_cmudict_file)
    tokenizer = Mock(spec=BridgeTokenizer, wraps=real)
    # `spec` exposes class attributes only; these are set in __init__.
    tokenizer.custom_cmudict_path = real.custom_cmudict_path
    return tokenizer


@pytest.fixture
def bridge_dataset(dataset_config, mock_bridge_tokenizer, mock_gcs_client):
    dataset = BridgeDataset(dataset_config, mock_gcs_client, tokenizer=mock_bridge_tokenizer)
    dataset.mock_tokenizer = mock_bridge_tokenizer
    return dataset


def inputs_of(dataset) -> list[str]:
    return [example.input for example in dataset.examples]


def test_dataset_initialization(bridge_dataset, mock_dataset_file):
    assert isinstance(bridge_dataset, BridgeDataset)
    assert len(bridge_dataset.examples) == 2
    assert inputs_of(bridge_dataset) == ["cat", "dog"]
    assert [example.target for example in bridge_dataset.examples] == ["cat", "dog"]


def test_dataset_length(bridge_dataset):
    assert len(bridge_dataset) == 2


def test_get_item_by_index(bridge_dataset):
    """A row is a batch of one: an input and a target, each in both modalities."""
    item = bridge_dataset[0]
    assert isinstance(item, BridgeBatch)
    assert isinstance(item.input, BridgeInput)
    assert isinstance(item.target, BridgeTarget)
    assert isinstance(item.input.orth, Tokens)
    assert isinstance(item.input.phon, Tokens)
    assert isinstance(item.target.orth, Target)
    assert isinstance(item.target.phon, Target)

    # Input: (batch=1, seq_len=6), [LANG, BOS, c, a, t, EOS].
    assert item.input.orth.ids.shape == (1, 6)
    # Target: one position narrower, [LANG, BOS, c, a, t], with a label for each.
    assert item.target.orth.ids.shape == (1, 5)
    assert item.target.orth.labels.shape == (1, 5)
    # Phoneme row ids, the same (batch, sequence) shape as the orthographic side.
    assert item.input.phon.ids.shape[0] == 1
    assert item.input.device == item.target.device == torch.device("cpu")


def test_get_item_has_the_expected_ids(bridge_dataset):
    item = bridge_dataset[inputs_of(bridge_dataset).index("cat")]
    # [LANG=7 ("EN"), BOS=0, c=21, a=19, t=38, EOS=1]. The fixture tags every word
    # as "EN", and EN sits at index 7 in the vocab (after 6 special tokens + "--").
    assert item.input.orth.ids.tolist() == [[7, 0, 21, 19, 38, 1]]
    assert item.target.orth.ids.tolist() == [[7, 0, 21, 19, 38]]
    assert item.target.orth.labels.tolist() == [[0, 21, 19, 38, 1]]


def test_get_item_by_slice(bridge_dataset):
    items = bridge_dataset[0:2]
    assert isinstance(items, BridgeBatch)
    assert len(items) == 2
    assert items.input.orth.ids.shape[0] == 2
    assert items.input.phon.pad_mask.shape[0] == 2
    assert items.target.orth.ids.shape[0] == 2
    assert items.target.phon.labels.shape[0] == 2


def test_invalid_index_access(bridge_dataset):
    with pytest.raises(IndexError):
        _ = bridge_dataset[100]

    with pytest.raises(TypeError):
        _ = bridge_dataset["cat"]


def test_every_read_encodes_each_side_once(dataset_config, mock_bridge_tokenizer, mock_gcs_client):
    """Oracle: call counts on the real tokenizer. Loading encodes nothing, and a read
    encodes the input once and the target once, every time: nothing is kept between
    reads, so a row cannot come back stale."""
    dataset = BridgeDataset(dataset_config, mock_gcs_client, tokenizer=mock_bridge_tokenizer)
    assert mock_bridge_tokenizer.encode_input.call_count == 0
    assert mock_bridge_tokenizer.encode_target.call_count == 0

    first = dataset[0]
    assert mock_bridge_tokenizer.encode_input.call_count == 1
    assert mock_bridge_tokenizer.encode_target.call_count == 1

    second = dataset[0]
    assert mock_bridge_tokenizer.encode_input.call_count == 2
    assert mock_bridge_tokenizer.encode_target.call_count == 2
    assert first is not second
    assert torch.equal(first.input.orth.ids, second.input.orth.ids)


def test_batch_consistency(bridge_dataset):
    """A row read by index equals the same row read as a slice of one, tensor for tensor."""
    single = bridge_dataset[0]
    batch = bridge_dataset[0:1]

    for a, b in (
        (single.input.orth, batch.input.orth),
        (single.input.phon, batch.input.phon),
        (single.target.orth, batch.target.orth),
        (single.target.phon, batch.target.phon),
    ):
        assert torch.equal(a.ids, b.ids)
        assert torch.equal(a.pad_mask, b.pad_mask)
    assert torch.equal(single.target.orth.labels, batch.target.orth.labels)
    assert torch.equal(single.target.phon.labels, batch.target.phon.labels)


def test_shuffle_functionality(bridge_dataset):
    original = list(bridge_dataset.examples)
    bridge_dataset.shuffle(1)

    assert len(bridge_dataset.examples) == len(original)
    assert set(bridge_dataset.examples) == set(original)
    assert bridge_dataset.examples[1:] == original[1:]


def test_data_validation(tmp_path, dataset_config, mock_gcs_client):
    invalid_data = []
    file_path = tmp_path / "invalid.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(invalid_data, f)

    dataset_config.dataset_filepath = str(file_path)
    with pytest.raises(ValueError, match="Dataset file must contain a dictionary"):
        _ = BridgeDataset(dataset_config, mock_gcs_client)


def test_pkl_columns_must_have_equal_length(tmp_path, dataset_config, mock_gcs_client):
    """Oracle: the rule. A short column would pair inputs with the wrong targets."""
    file_path = tmp_path / "ragged.pkl"
    with open(file_path, "wb") as f:
        pickle.dump({"word_raw": ["cat", "dog"], "target_raw": ["cat"]}, f)

    dataset_config.dataset_filepath = str(file_path)
    with pytest.raises(ValueError, match="must have equal length, got .*'target_raw': 1"):
        _ = BridgeDataset(dataset_config, mock_gcs_client)


def test_a_side_the_tokenizer_cannot_pronounce_raises_naming_it(
    dataset_config, mock_bridge_tokenizer, mock_gcs_client
):
    """Oracle: the rule. Either side without a pronunciation stops the read, and the
    message says which row and which side. The tokenizer is stubbed, so each side fails
    alone."""
    dataset = BridgeDataset(dataset_config, mock_gcs_client, tokenizer=mock_bridge_tokenizer)
    real = mock_bridge_tokenizer.encode_input(["cat"], ["EN"])

    mock_bridge_tokenizer.encode_target.side_effect = lambda *args: None
    with pytest.raises(RuntimeError, match="No pronunciation for a word of row 0's target 'cat'$"):
        _ = dataset[0]

    mock_bridge_tokenizer.encode_target.side_effect = None
    mock_bridge_tokenizer.encode_input.side_effect = lambda *args: BridgeInput(orth=real.orth)
    with pytest.raises(RuntimeError, match="No pronunciation for a word of row 0's input 'cat'$"):
        _ = dataset[0]


def test_integration_with_training_pipeline(bridge_dataset):
    """A batch carries every tensor a training step reads, under the names it reads."""
    batch = bridge_dataset[0:2]

    assert isinstance(batch, BridgeBatch)
    for part in (batch.input.orth, batch.input.phon):
        assert all(torch.is_tensor(getattr(part, key)) for key in ("ids", "pad_mask"))
    for part in (batch.target.orth, batch.target.phon):
        assert all(torch.is_tensor(getattr(part, key)) for key in ("ids", "pad_mask", "labels"))


def test_vocabulary_size_properties(dataset_config, mock_bridge_tokenizer, mock_gcs_client):
    """Vocabulary sizes are taken from the tokenizer, not derived independently."""
    mock_bridge_tokenizer.get_vocabulary_sizes.return_value = {
        "orthographic": 100,
        "phonological": 200,
    }
    dataset = BridgeDataset(dataset_config, mock_gcs_client, tokenizer=mock_bridge_tokenizer)

    assert dataset.orthographic_vocabulary_size == 100
    assert dataset.phonological_vocabulary_size == 200
