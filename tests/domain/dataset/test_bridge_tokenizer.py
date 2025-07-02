import pytest
import torch
import logging
from unittest.mock import patch

from bridge.domain.dataset.bridge_tokenizer import BridgeTokenizer
from bridge.domain.datamodels import BridgeEncoding


@pytest.fixture
def bridge_tokenizer():
    """Create a BridgeTokenizer instance for testing."""
    return BridgeTokenizer()


class TestBridgeTokenizer:
    def test_encode_known_word(self, bridge_tokenizer):
        """Test encoding a word that is known to the tokenizer."""
        # "cat" is a common word that should be in CMUDict
        encoding = bridge_tokenizer.encode("cat")
        assert encoding is not None
        assert isinstance(encoding, BridgeEncoding)
        assert hasattr(encoding, "orth_enc_ids")
        assert hasattr(encoding, "phon_enc_ids")

    def test_encode_with_modality_filter_both(self, bridge_tokenizer):
        """Test encoding a word with modality_filter='both'."""
        encoding = bridge_tokenizer.encode("cat", modality_filter="both")
        assert encoding is not None
        assert isinstance(encoding, BridgeEncoding)
        assert hasattr(encoding, "orth_enc_ids")
        assert hasattr(encoding, "phon_enc_ids")
        
    def test_encode_nan(self, bridge_tokenizer):
        """Test encoding a word that is NaN."""
        encoding = bridge_tokenizer.encode("nan", modality_filter="both")
        assert encoding is not None
        assert isinstance(encoding, BridgeEncoding)
        assert hasattr(encoding, "orth_enc_ids")
        assert hasattr(encoding, "phon_enc_ids")


    def test_encode_with_modality_filter_orthography(self, bridge_tokenizer):
        """Test encoding a word with modality_filter='orthography'."""
        encoding = bridge_tokenizer.encode("cat", modality_filter="orthography")
        assert encoding is not None
        assert isinstance(encoding, BridgeEncoding)
        assert hasattr(encoding, "orth_enc_ids")
        # Phonological fields should have placeholders but not be None
        assert hasattr(encoding, "phon_enc_ids")

    def test_encode_with_modality_filter_phonology(self, bridge_tokenizer):
        """Test encoding a word with modality_filter='phonology'."""
        encoding = bridge_tokenizer.encode("cat", modality_filter="phonology")
        assert encoding is not None
        assert isinstance(encoding, BridgeEncoding)
        # Orthographic fields should have placeholders but not be None
        assert hasattr(encoding, "orth_enc_ids")
        assert hasattr(encoding, "phon_enc_ids")

    def test_encode_unknown_word_both_modalities(self, bridge_tokenizer):
        """Test encoding a nonword with modality_filter='both'."""
        # "blathe" is a nonword that shouldn't be in CMUDict
        encoding = bridge_tokenizer.encode("blathe", modality_filter="both")
        # Should return None because phonological encoding fails
        assert encoding is None

    def test_encode_unknown_word_orthography_modality(self, bridge_tokenizer):
        """Test encoding a nonword with modality_filter='orthography'."""
        # "blathe" is a nonword that shouldn't be in CMUDict
        encoding = bridge_tokenizer.encode("blathe", modality_filter="orthography")
        # Should return an encoding with only orthographic data
        assert encoding is not None
        assert isinstance(encoding, BridgeEncoding)
        assert hasattr(encoding, "orth_enc_ids")
        # Ensure placeholder phonological data is present
        assert hasattr(encoding, "phon_enc_ids")

    def test_encode_unknown_word_phonology_modality(self, bridge_tokenizer):
        """Test encoding a nonword with modality_filter='phonology'."""
        # "blathe" is a nonword that shouldn't be in CMUDict
        # This should still fail since we can't create phonological encoding
        encoding = bridge_tokenizer.encode("blathe", modality_filter="phonology")
        assert encoding is None

    def test_encode_batch_with_mixed_words(self, bridge_tokenizer):
        """Test encoding a batch with both known and unknown words."""
        words = ["cat", "blathe", "dog"]
        # Should return None by default because "blathe" is unknown
        encoding = bridge_tokenizer.encode(words)
        assert encoding is None

        # With orthography modality, should return encodings for all words
        encoding = bridge_tokenizer.encode(words, modality_filter="orthography")
        assert encoding is not None
        assert isinstance(encoding, BridgeEncoding)
        assert encoding.orth_enc_ids.shape[0] == 3  # Batch size

    def test_generate_compatible_encoding(self, bridge_tokenizer):
        """Test creating encodings compatible with the generate method for nonwords."""
        # Create encoding for use with o2p pathway (orthography -> phonology)
        nonword = "blathe"
        encoding = bridge_tokenizer.encode(nonword, modality_filter="orthography")

        assert encoding is not None
        assert isinstance(encoding, BridgeEncoding)

        # This encoding should be usable for o2p generation
        assert encoding.orth_enc_ids is not None
        assert encoding.orth_enc_mask is not None

        # These placeholder values should not interfere with o2p generation
        assert encoding.phon_enc_ids is not None

    def test_invalid_modality_filter(self, bridge_tokenizer):
        """Test that an invalid modality_filter raises an error."""
        with pytest.raises(ValueError):
            bridge_tokenizer.encode("cat", modality_filter="invalid")
