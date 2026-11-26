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
    def __init__(self, **kwargs):
        self.device = torch.device(kwargs.get("device", "cpu"))
        self.orthographic_vocabulary_size = kwargs.get("orthographic_vocabulary_size", 49)
        # Not a free parameter: Model derives phoneme embeddings from phonreps.csv and
        # rejects a vocab size that disagrees with it.
        self.phonological_vocabulary_size = kwargs.get(
            "phonological_vocabulary_size", PHONEME_TABLE.vocab_size
        )

        self.tokenizer = Mock()
        self.tokenizer.get_vocabulary_sizes.return_value = {
            "orthographic": self.orthographic_vocabulary_size,
            "phonological": self.phonological_vocabulary_size,
        }


def _vocab_spec_for(mock_dataset: MockBridgeDataset) -> VocabSpec:
    """Derived from TEST_VOCAB rather than hand-written, so these models exercise the same
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
    return MockBridgeDataset(
        orthographic_vocabulary_size=49,
        phonological_vocabulary_size=PHONEME_TABLE.vocab_size,
    )


@pytest.fixture
def model_config(mock_dataset):
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
    return Model(model_config)


def test_embed_orth_tokens(model: Model):
    """A token embeds to its row of the embedding table, wherever it sits.

    Two oracles. Analytic: the output is the table's row for each id, bitwise. And the
    recording in ``embed_orth_tokens_test_data.pkl``, which holds each token's vector plus
    a vector that depends only on the token's position. The model's embedding carries no
    position, so it differs from the recording by that vector alone: the difference is
    the same on all 64 rows, and it is not zero.
    """
    model.eval()
    with open("tests/domain/model/data/embed_orth_tokens_test_data.pkl", "rb") as f:
        data = pickle.load(f)

    input = data["input"]
    output = model.embed_orth_tokens(input)
    assert torch.equal(output, model.orthography_embedding.weight[input])

    by_position = data["output"] - output
    assert torch.allclose(by_position, by_position[:1].expand_as(by_position), atol=1e-5), (
        "The recording differs from the token embedding by more than a per-position vector."
    )
    assert by_position.abs().max() > 0.1, "control: the recording does carry a position vector"


def test_model_initialization_with_dataset(mock_dataset, model_config):
    model = Model(model_config)

    assert model.orthographic_vocabulary_size == mock_dataset.orthographic_vocabulary_size
    assert model.phonological_vocabulary_size == mock_dataset.phonological_vocabulary_size

    assert model.orthography_embedding.num_embeddings == mock_dataset.orthographic_vocabulary_size
    assert model.phonology_embedding.num_embeddings == mock_dataset.phonological_vocabulary_size

    # One embedding table per modality and no other: nothing is indexed by position, so
    # nothing bounds the length of a sequence.
    tables = {
        name for name, module in model.named_modules() if isinstance(module, torch.nn.Embedding)
    }
    assert tables == {"orthography_embedding", "phonology_embedding"}
