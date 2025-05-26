import pytest
import torch
from unittest.mock import patch, MagicMock

from bridge.domain.datamodels import BridgeEncoding, GenerationOutput
from bridge.domain.datamodels.encodings import EncodingComponent
from bridge.domain.dataset.bridge_tokenizer import BridgeTokenizer


@pytest.fixture
def bridge_tokenizer():
    """Create a BridgeTokenizer instance for testing."""
    return BridgeTokenizer()


class TestNonwordGenerate:
    @patch("bridge.domain.model.model.Model")
    def test_o2p_nonword_generation(self, MockModel, bridge_tokenizer):
        """Test generating phonological representation from a nonword."""
        # Set up mock model
        mock_model = MockModel.return_value
        mock_model.generate.return_value = MagicMock(
            spec=GenerationOutput,
            phon_tokens=[[torch.tensor([1, 2, 3])]],
            phon_probs=[[torch.tensor([0.1, 0.2, 0.3])]],
        )

        # Create an encoding for a nonword using orthography-only
        nonword = "blathe"
        encoding = bridge_tokenizer.encode(nonword, modality_filter="orthography")

        assert encoding is not None
        assert encoding.orthographic is not None

        # Generate phonological representation using o2p pathway
        output = mock_model.generate(encoding, pathway="o2p", deterministic=True)

        # Verify model.generate was called correctly
        mock_model.generate.assert_called_once_with(
            encoding, pathway="o2p", deterministic=True
        )

        # Verify the output has the expected format
        assert output.phon_tokens is not None
        assert output.phon_probs is not None

    @patch("bridge.domain.model.model.Model")
    def test_multiple_nonwords_batch(self, MockModel, bridge_tokenizer):
        """Test generating phonological representations for multiple nonwords."""
        # Set up mock model
        mock_model = MockModel.return_value
        mock_model.generate.return_value = MagicMock(
            spec=GenerationOutput,
            phon_tokens=[
                [torch.tensor([1, 2, 3])],
                [torch.tensor([4, 5, 6])],
                [torch.tensor([7, 8, 9])],
            ],
            phon_probs=[
                [torch.tensor([0.1, 0.2, 0.3])],
                [torch.tensor([0.4, 0.5, 0.6])],
                [torch.tensor([0.7, 0.8, 0.9])],
            ],
        )

        # Create encoding for a batch of nonwords
        nonwords = ["blathe", "phleem", "stoaz"]
        encoding = bridge_tokenizer.encode(nonwords, modality_filter="orthography")

        assert encoding is not None
        assert encoding.orthographic.enc_input_ids.shape[0] == 3  # Batch size

        # Generate phonological representations using o2p pathway
        output = mock_model.generate(encoding, pathway="o2p", deterministic=True)

        # Verify model.generate was called correctly
        mock_model.generate.assert_called_once_with(
            encoding, pathway="o2p", deterministic=True
        )

        # Verify the output has the expected format
        assert output.phon_tokens is not None
        assert len(output.phon_tokens) == 3  # Batch size

    @patch("bridge.domain.model.model.Model")
    def test_mixed_words_nonwords_batch(self, MockModel, bridge_tokenizer):
        """Test generating phonological representations for a mix of words and nonwords."""
        # Set up mock model
        mock_model = MockModel.return_value
        mock_model.generate.return_value = MagicMock(
            spec=GenerationOutput,
            phon_tokens=[
                [torch.tensor([1, 2, 3])],
                [torch.tensor([4, 5, 6])],
                [torch.tensor([7, 8, 9])],
            ],
            phon_probs=[
                [torch.tensor([0.1, 0.2, 0.3])],
                [torch.tensor([0.4, 0.5, 0.6])],
                [torch.tensor([0.7, 0.8, 0.9])],
            ],
        )

        # Create encoding for a batch of words and nonwords
        mixed_words = ["cat", "blathe", "dog"]
        encoding = bridge_tokenizer.encode(mixed_words, modality_filter="orthography")

        assert encoding is not None
        assert encoding.orthographic.enc_input_ids.shape[0] == 3  # Batch size

        # Generate phonological representations using o2p pathway
        output = mock_model.generate(encoding, pathway="o2p", deterministic=True)

        # Verify model.generate was called correctly
        mock_model.generate.assert_called_once_with(
            encoding, pathway="o2p", deterministic=True
        )

        # Verify the output has the expected format
        assert output.phon_tokens is not None
        assert len(output.phon_tokens) == 3  # Batch size

    def test_bridge_tokenizer_integration(self, bridge_tokenizer):
        """Test that BridgeTokenizer correctly handles nonwords with modality filtering."""
        # Test with a known word
        known_word = "cat"
        encoding = bridge_tokenizer.encode(known_word, modality_filter="both")
        assert encoding is not None
        assert encoding.orthographic is not None
        assert encoding.phonological is not None

        # Test with a nonword using orthography-only mode
        nonword = "blathe"
        encoding = bridge_tokenizer.encode(nonword, modality_filter="orthography")
        assert encoding is not None
        assert encoding.orthographic is not None
        assert encoding.phonological is not None  # Should have placeholder values

        # Test with a nonword using both modalities - should fail
        encoding = bridge_tokenizer.encode(nonword, modality_filter="both")
        assert encoding is None
