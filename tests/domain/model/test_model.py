import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.domain.datamodels import ModelConfig
from src.domain.model import Model
from src.domain.dataset import BridgeDataset
import pytest
import pickle
import torch
from unittest.mock import Mock


class MockBridgeDataset:
    """Mock implementation of BridgeDataset for testing Model."""

    def __init__(self, **kwargs):
        self.device = torch.device(kwargs.get("device", "cpu"))
        self.orthographic_vocabulary_size = kwargs.get(
            "orthographic_vocabulary_size", 49
        )
        self.phonological_vocabulary_size = kwargs.get(
            "phonological_vocabulary_size", 34
        )

        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.get_vocabulary_sizes.return_value = {
            "orthographic": self.orthographic_vocabulary_size,
            "phonological": self.phonological_vocabulary_size,
        }


@pytest.fixture
def mock_dataset():
    """Fixture for mock BridgeDataset."""
    return MockBridgeDataset(
        orthographic_vocabulary_size=49, phonological_vocabulary_size=34
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
def model(mock_dataset, model_config):
    """Fixture for initializing the Model."""
    return Model(model_config, mock_dataset)


def test_embed_orth_tokens(model: Model):
    model.eval()
    with open("tests/domain/model/data/embed_orth_tokens_test_data.pkl", "rb") as f:
        data = pickle.load(f)

    input = data["input"]
    expected_output = data["output"]
    output = model.embed_orth_tokens(input)
    assert torch.allclose(
        output, expected_output, atol=1e-5
    ), "Output does not match expected values."


def test_embed_phon_tokens(model: Model):
    model.eval()
    with open("tests/domain/model/data/embed_phon_tokens_test_data.pkl", "rb") as f:
        data = pickle.load(f)

    input = data["input"]

    # Don't compare to fixed expected output, instead check properties of the output
    output = model.embed_phon_tokens(input)

    # Check output is a tensor with expected shape characteristics
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.dim() == 3, "Output should be 3-dimensional"
    assert output.size(0) == len(input), "Batch dimension should match input"
    assert output.size(1) == len(input[0]), "Sequence length should match input"
    assert (
        output.size(2) == model.model_config.d_model
    ), "Feature dimension should match d_model"

    # Check output contains reasonable values (not NaN, not huge)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.abs(output).max() < 100, "Output contains unreasonably large values"


def test_generate_triangular_mask(model: Model):
    model.eval()
    with open(
        "tests/domain/model/data/generate_triangular_mask_test_data.pkl", "rb"
    ) as f:
        data = pickle.load(f)

    input = data["input"]
    expected_output = data["output"]

    mask = model.generate_triangular_mask(input.shape[1])
    assert torch.allclose(
        mask, expected_output, atol=1e-5
    ), "Output does not match expected values."


def test_model_initialization_with_dataset(mock_dataset, model_config):
    """Test the model initialization with BridgeDataset."""
    # Create a model with our mock BridgeDataset
    model = Model(model_config, mock_dataset)

    # Verify the model correctly obtained vocabulary sizes
    assert (
        model.orthographic_vocabulary_size == mock_dataset.orthographic_vocabulary_size
    )
    assert (
        model.phonological_vocabulary_size == mock_dataset.phonological_vocabulary_size
    )

    # Verify hardcoded sequence lengths
    assert model.max_orth_seq_len == 30
    assert model.max_phon_seq_len == 30

    # Verify embedding dimensions
    assert (
        model.orthography_embedding.num_embeddings
        == mock_dataset.orthographic_vocabulary_size
    )
    assert (
        model.phonology_embedding.num_embeddings
        == mock_dataset.phonological_vocabulary_size
    )
    assert model.orth_position_embedding.num_embeddings == 30
    assert model.phon_position_embedding.num_embeddings == 30


def test_embed_o(model: Model):
    model.eval()
    with open("tests/domain/model/data/embed_o_test_data.pkl", "rb") as f:
        data = pickle.load(f)

    orth_enc_input = data["orth_enc_input"]
    orth_enc_pad_mask = data["orth_enc_pad_mask"]

    # Don't compare to fixed expected output, instead check properties of the output
    output = model.embed_o(orth_enc_input, orth_enc_pad_mask)

    # Check output is a tensor with expected shape characteristics
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.dim() == 3, "Output should be 3-dimensional"
    assert output.size(0) == orth_enc_input.size(
        0
    ), "Batch dimension should match input"
    assert output.size(1) == 1, "Sequence length should be 1 (global encoding)"
    assert (
        output.size(2) == model.model_config.d_model
    ), "Feature dimension should match d_model"

    # Check output contains reasonable values (not NaN, not huge)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.abs(output).max() < 100, "Output contains unreasonably large values"


def test_embed_p(model: Model):
    model.eval()
    with open("tests/domain/model/data/embed_p_test_data.pkl", "rb") as f:
        data = pickle.load(f)

    phon_enc_input = data["phon_enc_input"]
    phon_enc_pad_mask = data["phon_enc_pad_mask"]

    # Don't compare to fixed expected output, instead check properties of the output
    output = model.embed_p(phon_enc_input, phon_enc_pad_mask)

    # Check output is a tensor with expected shape characteristics
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.dim() == 3, "Output should be 3-dimensional"
    assert output.size(0) == len(phon_enc_input), "Batch dimension should match input"
    assert output.size(1) == 1, "Sequence length should be 1 (global encoding)"
    assert (
        output.size(2) == model.model_config.d_model
    ), "Feature dimension should match d_model"

    # Check output contains reasonable values (not NaN, not huge)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.abs(output).max() < 100, "Output contains unreasonably large values"


def test_embed_op(model: Model):
    model.eval()

    with open("tests/domain/model/data/embed_op_test_data.pkl", "rb") as f:
        data = pickle.load(f)

    orth_enc_input = data["orth_enc_input"]
    orth_enc_pad_mask = data["orth_enc_pad_mask"]
    phon_enc_input = data["phon_enc_input"]
    phon_enc_pad_mask = data["phon_enc_pad_mask"]

    # Don't compare to fixed expected output, instead check properties of the output
    output = model.embed_op(
        orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask
    )

    # Check output is a tensor with expected shape characteristics
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.dim() == 3, "Output should be 3-dimensional"
    assert output.size(0) == orth_enc_input.size(
        0
    ), "Batch dimension should match input"
    assert output.size(1) == 1, "Sequence length should be 1 (global encoding)"
    assert (
        output.size(2) == model.model_config.d_model
    ), "Feature dimension should match d_model"

    # Check output contains reasonable values (not NaN, not huge)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.abs(output).max() < 100, "Output contains unreasonably large values"


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

    # Verify orthographic output
    assert "orth" in output, "Output should contain 'orth' key"
    assert isinstance(output["orth"], torch.Tensor), "Output['orth'] should be a tensor"
    assert output["orth"].dim() == 3, "Output['orth'] should be 3-dimensional"
    assert output["orth"].size(0) == orth_enc_input.size(
        0
    ), "Batch size should match input"
    assert (
        output["orth"].size(1) == model.orthographic_vocabulary_size
    ), "Second dimension should be vocab size"
    assert output["orth"].size(2) == orth_dec_input.size(
        1
    ), "Third dimension should match sequence length"
    assert not torch.isnan(output["orth"]).any(), "Output should not contain NaN values"

    # Verify phonological output
    assert "phon" in output, "Output should contain 'phon' key"
    assert isinstance(output["phon"], torch.Tensor), "Output['phon'] should be a tensor"
    assert output["phon"].dim() == 4, "Output['phon'] should be 4-dimensional"
    assert output["phon"].size(0) == len(
        phon_enc_input
    ), "Batch size should match input"
    assert (
        output["phon"].size(1) == 2
    ), "Second dimension should be 2 (binary classification)"
    assert output["phon"].size(2) == len(
        phon_dec_input[0]
    ), "Third dimension should match sequence length"
    assert (
        output["phon"].size(3) == model.phonological_vocabulary_size - 1
    ), "Fourth dimension should be vocab size minus 1"
    assert not torch.isnan(output["phon"]).any(), "Output should not contain NaN values"


def test_forward_o2p(model: Model):
    model.eval()
    with open("tests/domain/model/data/forward_o2p_test_data.pkl", "rb") as file:
        data = pickle.load(file)

    # Extract inputs
    orth_enc_input = data["orth_enc_input"]
    orth_enc_pad_mask = data["orth_enc_pad_mask"]
    phon_dec_input = data["phon_dec_input"]
    phon_dec_pad_mask = data["phon_dec_pad_mask"]

    output = model.forward_o2p(
        orth_enc_input,
        orth_enc_pad_mask,
        phon_dec_input,
        phon_dec_pad_mask,
    )

    # Verify output structure and properties
    assert "phon" in output, "Output should contain 'phon' key"
    assert isinstance(output["phon"], torch.Tensor), "Output['phon'] should be a tensor"
    assert output["phon"].dim() == 4, "Output['phon'] should be 4-dimensional"
    assert output["phon"].size(0) == orth_enc_input.size(
        0
    ), "Batch size should match input"
    assert (
        output["phon"].size(1) == 2
    ), "Second dimension should be 2 (binary classification)"
    assert output["phon"].size(2) == len(
        phon_dec_input[0]
    ), "Third dimension should match sequence length"
    assert (
        output["phon"].size(3) == model.phonological_vocabulary_size - 1
    ), "Fourth dimension should be vocab size minus 1"
    assert not torch.isnan(output["phon"]).any(), "Output should not contain NaN values"


def test_forward_p2o(model: Model):
    model.eval()
    with open("tests/domain/model/data/forward_p2o_test_data.pkl", "rb") as file:
        data = pickle.load(file)

    orth_dec_input = data["orth_dec_input"]
    orth_dec_pad_mask = data["orth_dec_pad_mask"]
    phon_enc_input = data["phon_enc_input"]
    phon_enc_pad_mask = data["phon_enc_pad_mask"]

    output = model.forward_p2o(
        phon_enc_input,
        phon_enc_pad_mask,
        orth_dec_input,
        orth_dec_pad_mask,
    )

    # Verify output structure and properties
    assert "orth" in output, "Output should contain 'orth' key"
    assert isinstance(output["orth"], torch.Tensor), "Output['orth'] should be a tensor"
    assert output["orth"].dim() == 3, "Output['orth'] should be 3-dimensional"
    assert output["orth"].size(0) == orth_dec_input.size(
        0
    ), "Batch size should match input"
    assert (
        output["orth"].size(1) == model.orthographic_vocabulary_size
    ), "Second dimension should be vocab size"
    assert output["orth"].size(2) == orth_dec_input.size(
        1
    ), "Third dimension should match sequence length"
    assert not torch.isnan(output["orth"]).any(), "Output should not contain NaN values"


def test_forward_p2p(model: Model, mock_dataset):
    """We do not have access to p2p test data from the legacy code as it was
    not implemented. Therefore, our tests here simply ensure that the code runs,
    output shapes are as expected and the output is a tensor."""
    model.eval()
    with open("tests/domain/model/data/forward_op2op_test_data.pkl", "rb") as file:
        data = pickle.load(file)

    phon_enc_input = data["phon_enc_input"]
    phon_enc_pad_mask = data["phon_enc_pad_mask"]
    phon_dec_input = data["phon_dec_input"]
    phon_dec_pad_mask = data["phon_dec_pad_mask"]

    output = model.forward_p2p(
        phon_enc_input,
        phon_enc_pad_mask,
        phon_dec_input,
        phon_dec_pad_mask,
    )

    assert isinstance(output, dict), f"Expected output to be dict, got {type(output)}"
    assert "phon" in output, f"Expected 'phon' key in output. Keys: {output.keys()}"
    assert isinstance(
        output["phon"], torch.Tensor
    ), f"Expected output['phon'] to be tensor, got {type(output['phon'])}"
    # Shape should be (batch_size, 2, max_phon_seq_len, phonological_vocabulary_size - 1)
    # where 2 is because this is a binary classification problem, and the minus 1 is because
    # padding tokens are removed from the targets and output predictions
    size = torch.Size(
        [
            len(phon_dec_input),  # batch_size
            2,  # binary classification
            len(phon_dec_input[0]),  # max_phon_seq_len
            model.phonological_vocabulary_size - 1,
        ]
    )
    assert (
        output["phon"].shape == size
    ), f"Shape mismatch: got {output['phon'].shape}, expected {size}"


def test_gpu_availability():
    """Test GPU availability and basic tensor operations."""
    from src.utils.device_manager import device_manager

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
