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
        seed=42,
    )


@pytest.fixture
def model(dataset_config, model_config):
    """Fixture for initializing the Model."""
    device = torch.device("cpu")
    return Model(model_config, dataset_config, device)


def test_embed_orth_tokens(model: Model):
    model.eval()
    with open("tests/domain/model/data/embed_orth_tokens_test_data.pkl", "rb") as f:
        data = pickle.load(f)

    input = data["input"]
    expected_output = data["output"]
    output = model.embed_orth_tokens(input)
    assert torch.allclose(output, expected_output, atol=1e-5), "Output does not match expected values."


def test_embed_phon_tokens(model: Model):
    model.eval()
    with open("tests/domain/model/data/embed_phon_tokens_test_data.pkl", "rb") as f:
        data = pickle.load(f)

    input = data["input"]
    expected_output = data["output"]

    output = model.embed_phon_tokens(input)
    assert torch.allclose(output, expected_output, atol=1e-5), "Output does not match expected values."


def test_generate_triangular_mask(model: Model):
    model.eval()
    with open("tests/domain/model/data/generate_triangular_mask_test_data.pkl", "rb") as f:
        data = pickle.load(f)

    input = data["input"]
    expected_output = data["output"]

    mask = model.generate_triangular_mask(input.shape[1])
    assert torch.allclose(mask, expected_output, atol=1e-5), "Output does not match expected values."


def test_embed_o(model: Model):
    model.eval()
    with open("tests/domain/model/data/embed_o_test_data.pkl", "rb") as f:
        data = pickle.load(f)

    orth_enc_input = data["orth_enc_input"]
    orth_enc_pad_mask = data["orth_enc_pad_mask"]
    expected_output = data["output"]

    output = model.embed_o(orth_enc_input, orth_enc_pad_mask)
    assert torch.allclose(output, expected_output, atol=1e-5), "Output does not match expected values."


def test_embed_p(model: Model):
    model.eval()
    with open("tests/domain/model/data/embed_p_test_data.pkl", "rb") as f:
        data = pickle.load(f)

    phon_enc_input = data["phon_enc_input"]
    phon_enc_pad_mask = data["phon_enc_pad_mask"]
    expected_output = data["output"]

    output = model.embed_p(phon_enc_input, phon_enc_pad_mask)
    assert torch.allclose(output, expected_output, atol=1e-5), "Output does not match expected values."


def test_embed_op(model: Model):
    model.eval()

    with open("tests/domain/model/data/embed_op_test_data.pkl", "rb") as f:
        data = pickle.load(f)

    orth_enc_input = data["orth_enc_input"]
    orth_enc_pad_mask = data["orth_enc_pad_mask"]
    phon_enc_input = data["phon_enc_input"]
    phon_enc_pad_mask = data["phon_enc_pad_mask"]
    expected_output = data["output"]

    output = model.embed_op(orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask)
    assert torch.allclose(output, expected_output, atol=1e-5), "Output does not match expected values."


def test_forward_op2op(model: Model):
    model.eval()
    with open("tests/domain/model/data/forward_op2op_test_data.pkl", "rb") as file:
        data = pickle.load(file)

    # Extract inputs
    orth_enc_input = data["orth_enc_input"]
    orth_enc_pad_mask = data["orth_enc_pad_mask"]
    orth_dec_input = data["orth_dec_input"]
    orth_dec_pad_mask = data["orth_dec_pad_mask"]
    phon_enc_input = data["phon_enc_input"]
    phon_enc_pad_mask = data["phon_enc_pad_mask"]
    phon_dec_input = data["phon_dec_input"]
    phon_dec_pad_mask = data["phon_dec_pad_mask"]

    expected_orth_token_logits_output = data["orth_token_logits"]
    expected_phon_token_logits_output = data["phon_token_logits"]

    output = model.forward_op2op(
        orth_enc_input,
        orth_enc_pad_mask,
        phon_enc_input,
        phon_enc_pad_mask,
        orth_dec_input,
        orth_dec_pad_mask,
        phon_dec_input,
        phon_dec_pad_mask,
    )
    assert torch.allclose(
        output["orth"], expected_orth_token_logits_output, atol=1e-5
    ), "Output does not match expected values."

    assert torch.allclose(
        output["phon"], expected_phon_token_logits_output, atol=1e-5
    ), "Output does not match expected values."


def test_forward_o2p(model: Model):
    model.eval()
    with open("tests/domain/model/data/forward_o2p_test_data.pkl", "rb") as file:
        data = pickle.load(file)

    # Extract inputs
    orth_enc_input = data["orth_enc_input"]
    orth_enc_pad_mask = data["orth_enc_pad_mask"]
    phon_dec_input = data["phon_dec_input"]
    phon_dec_pad_mask = data["phon_dec_pad_mask"]

    expected_phon_token_logits_output = data["phon_token_logits"]

    output = model.forward_o2p(
        orth_enc_input,
        orth_enc_pad_mask,
        phon_dec_input,
        phon_dec_pad_mask,
    )

    assert torch.allclose(
        output["phon"], expected_phon_token_logits_output, atol=1e-5
    ), "Output does not match expected values."


def test_forward_p2o(model: Model):
    model.eval()
    with open("tests/domain/model/data/forward_p2o_test_data.pkl", "rb") as file:
        data = pickle.load(file)


    orth_dec_input = data["orth_dec_input"]
    orth_dec_pad_mask = data["orth_dec_pad_mask"]
    phon_enc_input = data["phon_enc_input"]
    phon_enc_pad_mask = data["phon_enc_pad_mask"]

    expected_orth_token_logits_output = data["orth_token_logits"]

    output = model.forward_p2o(
        phon_enc_input,
        phon_enc_pad_mask,
        orth_dec_input,
        orth_dec_pad_mask,
    )

    assert torch.allclose(
        output["orth"], expected_orth_token_logits_output, atol=1e-5
    ), "Output does not match expected values."
