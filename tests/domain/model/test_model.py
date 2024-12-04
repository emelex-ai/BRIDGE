import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.domain.datamodels import DatasetConfig, ModelConfig
from src.domain.model import Model
import pytest
import pickle
import torch

@pytest.fixture
def dataset_config():
    """Fixture for mock DatasetConfig."""
    return DatasetConfig(
        dataset_filepath="data.csv",
        orthographic_vocabulary_size=49,
        phonological_vocabulary_size=34,
        max_orth_seq_len=100,
        max_phon_seq_len=100,
    )


@pytest.fixture
def model_config():
    """Fixture for mock ModelConfig."""
    return ModelConfig(
        num_phon_enc_layers=1,
        num_orth_enc_layers=1,
        num_mixing_enc_layers=1,
        num_phon_dec_layers=1,
        num_orth_dec_layers=1,
        d_model=64,
        nhead=2,
        d_embedding=1,
        seed=42
    )


@pytest.fixture
def model(dataset_config, model_config):
    """Fixture for initializing the Model."""
    device = torch.device("cpu")
    return Model(model_config, dataset_config, device)


def test_embed_orth_tokens(model: Model):
    with open("tests/domain/model/data/embed_orth_tokens_input.pkl", "rb") as f:
        input_tokens = pickle.load(f)

    with open("tests/domain/model/data/embed_orth_tokens_output.pkl", "rb") as f:
        expected_output = pickle.load(f)

    output = model.embed_orth_tokens(input_tokens)
    assert torch.allclose(output, expected_output, atol=1e-5), "Output does not match expected values."


def test_embed_phon_tokens(model: Model):
    with open("tests/domain/model/data/embed_phon_tokens_input.pkl", "rb") as f:
        input_tokens = pickle.load(f)

    with open("tests/domain/model/data/embed_phon_tokens_output.pkl", "rb") as f:
        expected_output = pickle.load(f)
        
    output = model.embed_phon_tokens(input_tokens)
    assert torch.allclose(output, expected_output, atol=1e-5), "Output does not match expected values."


def test_generate_triangular_mask(model: Model):
    with open("tests/domain/model/data/generate_triangular_mask_input.pkl", "rb") as f:
        dec_input = pickle.load(f)

    with open("tests/domain/model/data/generate_triangular_mask_output.pkl", "rb") as f:
        expected_output = pickle.load(f)

    mask = model.generate_triangular_mask(dec_input.shape[1])
    assert torch.allclose(mask, expected_output, atol=1e-5), "Output does not match expected values."


def test_embed_o2p(model: Model):
    with open("tests/domain/model/data/embed_o2p_input_orth_enc_input.pkl", "rb") as f:
        orth_enc_input = pickle.load(f)

    with open("tests/domain/model/data/embed_o2p_input_orth_enc_pad_mask.pkl", "rb") as f:
        orth_enc_pad_mask = pickle.load(f)

    with open("tests/domain/model/data/embed_o2p_output.pkl", "rb") as f:
        expected_output = pickle.load(f)

    output = model.embed_o2p(orth_enc_input, orth_enc_pad_mask)
    assert torch.allclose(output, expected_output, atol=1e-5), "Output does not match expected values."
