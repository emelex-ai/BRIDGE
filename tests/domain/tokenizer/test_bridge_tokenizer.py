import pytest

from bridge.domain.datamodels import BridgeEncoding
from bridge.domain.tokenizer.bridge_tokenizer import BridgeTokenizer


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


class TestMultilingual:
    """Tests for the multilingual / code-switching encoding paths."""

    def test_spanish_word_encodes_with_spanish_lexicon(self, bridge_tokenizer):
        """A Spanish word should encode against the Spanish phoneme lexicon."""
        encoding = bridge_tokenizer.encode(["hola"], language_map={"hola": "ES"})
        assert encoding is not None
        # The orth language token at position 0 should be "ES" (vocab index 8:
        # 6 special tokens + "--" + "EN" + "ES").
        es_idx = bridge_tokenizer.char_tokenizer.char_2_idx["ES"]
        assert encoding.orthographic.enc_input_ids[0, 0].item() == es_idx

    def test_code_switching_in_one_batch(self, bridge_tokenizer):
        """Two words in one batch encoded against two different languages."""
        encoding = bridge_tokenizer.encode(
            ["hola", "world"], language_map={"hola": "ES", "world": "EN"}
        )
        assert encoding is not None
        es_idx = bridge_tokenizer.char_tokenizer.char_2_idx["ES"]
        en_idx = bridge_tokenizer.char_tokenizer.char_2_idx["EN"]
        assert encoding.orthographic.enc_input_ids[0, 0].item() == es_idx
        assert encoding.orthographic.enc_input_ids[1, 0].item() == en_idx

    def test_default_no_language_uses_placeholder(self, bridge_tokenizer):
        """Encoding without a language_map prepends the '--' placeholder token."""
        encoding = bridge_tokenizer.encode(["cat"])
        assert encoding is not None
        placeholder_idx = bridge_tokenizer.char_tokenizer.char_2_idx["--"]
        assert encoding.orthographic.enc_input_ids[0, 0].item() == placeholder_idx

    def test_unknown_language_returns_none(self, bridge_tokenizer):
        """An unsupported language code should fail orthographic encoding, returning None.

        The character tokenizer rejects unknown language codes outright; BridgeTokenizer
        catches that and surfaces it as a failed encoding (None) rather than propagating
        the ValueError. The phoneme tokenizer is more permissive and falls back to English.
        """
        result = bridge_tokenizer.encode(["bonjour"], language_map={"bonjour": "FR"})
        assert result is None

    def test_unknown_language_raises_at_char_tokenizer(self, bridge_tokenizer):
        """Direct call to the char tokenizer DOES raise ValueError on unknown language."""
        with pytest.raises(ValueError):
            bridge_tokenizer.char_tokenizer.encode(["bonjour"], language_map={"bonjour": "FR"})
