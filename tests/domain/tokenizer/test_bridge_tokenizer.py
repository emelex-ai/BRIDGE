"""Pins what ``BridgeTokenizer`` returns for each side of an example.

``encode_input`` gives what the encoders read and ``encode_target`` gives what the decoders
are fed and scored against. Oracles: the child tokenizers called directly, hand-written
ids, and the lexicon.
"""

import pytest
import torch

from bridge.domain.datamodels import BridgeInput, BridgeTarget, Target, Tokens
from bridge.domain.tokenizer.bridge_tokenizer import BridgeTokenizer


@pytest.fixture
def bridge_tokenizer():
    return BridgeTokenizer()


class TestBridgeTokenizer:
    def test_encode_input_gives_tokens_in_both_modalities(self, bridge_tokenizer):
        """ "cat" is in the lexicon, so the input carries orthography and phonology."""
        inputs = bridge_tokenizer.encode_input(["cat"])
        assert isinstance(inputs, BridgeInput)
        assert isinstance(inputs.orth, Tokens)
        assert isinstance(inputs.phon, Tokens)

    def test_encode_target_gives_a_target_in_both_modalities(self, bridge_tokenizer):
        target = bridge_tokenizer.encode_target(["cat"])
        assert isinstance(target, BridgeTarget)
        assert isinstance(target.orth, Target)
        assert isinstance(target.phon, Target)

    def test_each_side_is_its_child_tokenizers_output(self, bridge_tokenizer):
        """Oracle: the child tokenizers called directly with the same rows and languages.

        The wrapper adds nothing of its own, and above all it does not cross the roles:
        the input holds the children's input encodings and the target their target ones.
        """
        texts, languages = ["cat", "no me", "the gato"], ["EN", "ES", "EN ES"]
        chars, phonemes = bridge_tokenizer.char_tokenizer, bridge_tokenizer.phoneme_tokenizer

        inputs = bridge_tokenizer.encode_input(texts, languages)
        for got, want in (
            (inputs.orth, chars.encode_input(texts, languages)),
            (inputs.phon, phonemes.encode_input(texts, languages)),
        ):
            assert torch.equal(got.ids, want.ids)
            assert torch.equal(got.pad_mask, want.pad_mask)

        target = bridge_tokenizer.encode_target(texts, languages)
        for got, want in (
            (target.orth, chars.encode_target(texts, languages)),
            (target.phon, phonemes.encode_target(texts, languages)),
        ):
            assert torch.equal(got.ids, want.ids)
            assert torch.equal(got.pad_mask, want.pad_mask)
            assert torch.equal(got.labels, want.labels)

    def test_the_word_nan_is_a_word(self, bridge_tokenizer):
        """ "nan" is in the lexicon and must not be read as a missing value."""
        inputs = bridge_tokenizer.encode_input(["nan"])
        assert inputs.phon is not None
        assert bridge_tokenizer.encode_target(["nan"]) is not None


class TestMultilingual:
    def test_spanish_word_encodes_with_spanish_lexicon(self, bridge_tokenizer):
        inputs = bridge_tokenizer.encode_input(["hola"], ["ES"])
        es_idx = bridge_tokenizer.char_tokenizer.char_2_idx["ES"]
        assert inputs.orth.ids[0, 0].item() == es_idx
        # "hola" is AO1 L AA0 in the Spanish lexicon and HH OW1 L AH0 in the English one,
        # so the phoneme rows say which lexicon was read. 86 and 87 are [BOS] and [EOS].
        assert inputs.phon.ids.tolist() == [[86, 22, 67, 0, 87]]
        # The target reads the same lexicon: the decoder is fed [BOS] and the phonemes.
        target = bridge_tokenizer.encode_target(["hola"], ["ES"])
        assert target.orth.ids[0, 0].item() == es_idx
        assert target.phon.ids.tolist() == [[86, 22, 67, 0]]

    def test_code_switching_in_one_batch(self, bridge_tokenizer):
        inputs = bridge_tokenizer.encode_input(["hola", "world"], ["ES", "EN"])
        es_idx = bridge_tokenizer.char_tokenizer.char_2_idx["ES"]
        en_idx = bridge_tokenizer.char_tokenizer.char_2_idx["EN"]
        assert inputs.orth.ids[0, 0].item() == es_idx
        assert inputs.orth.ids[1, 0].item() == en_idx

    def test_default_no_language_uses_placeholder(self, bridge_tokenizer):
        placeholder_idx = bridge_tokenizer.char_tokenizer.char_2_idx["--"]
        assert bridge_tokenizer.encode_input(["cat"]).orth.ids[0, 0].item() == placeholder_idx
        assert bridge_tokenizer.encode_target(["cat"]).orth.ids[0, 0].item() == placeholder_idx
