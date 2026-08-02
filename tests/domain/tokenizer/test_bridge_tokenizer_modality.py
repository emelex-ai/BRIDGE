"""Pins ``BridgeTokenizer.encode``'s modality routing and placeholder components.

``encode`` was three ~90-line branches, each wrapping every statement in its own
``try/except ... return None``. It is now one path. Two properties that the branch
structure enforced implicitly, and which nothing else asserts, are pinned here:

* **short-circuit** — a filter that excludes a modality must not invoke that
  tokenizer at all. This is what lets ``"orthography"`` encode a nonword that has no
  entry in the pronunciation lexicon.
* **placeholder shape** — the excluded modality is filled with a stub so
  ``BridgeEncoding`` (which requires both components) still validates. The stub
  builder's signature changed from ``(batch_size, seq_len)`` to ``(batch_size)``.
"""

import pytest
import torch

from bridge.domain.datamodels import BridgeEncoding
from bridge.domain.tokenizer import BridgeTokenizer

# A nonword: orthographically fine, absent from the pronunciation lexicon.
NONWORD = "blathe"
WORD = "cat"


@pytest.fixture(scope="module")
def tok():
    return BridgeTokenizer()


class Spy:
    """Wraps a bound method, recording whether it was called."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


# --- modality short-circuit ------------------------------------------------


def test_orthography_filter_never_calls_the_phoneme_tokenizer(tok, monkeypatch):
    spy = Spy(tok.phoneme_tokenizer.encode)
    monkeypatch.setattr(tok.phoneme_tokenizer, "encode", spy)
    assert tok.encode(WORD, modality_filter="orthography") is not None
    assert spy.calls == 0


def test_phonology_filter_never_calls_the_character_tokenizer(tok, monkeypatch):
    spy = Spy(tok.char_tokenizer.encode)
    monkeypatch.setattr(tok.char_tokenizer, "encode", spy)
    assert tok.encode(WORD, modality_filter="phonology") is not None
    assert spy.calls == 0


def test_both_filter_calls_each_tokenizer_once(tok, monkeypatch):
    char_spy = Spy(tok.char_tokenizer.encode)
    phon_spy = Spy(tok.phoneme_tokenizer.encode)
    monkeypatch.setattr(tok.char_tokenizer, "encode", char_spy)
    monkeypatch.setattr(tok.phoneme_tokenizer, "encode", phon_spy)
    assert tok.encode(WORD, modality_filter="both") is not None
    assert (char_spy.calls, phon_spy.calls) == (1, 1)


# --- nonwords --------------------------------------------------------------


def test_nonword_encodes_under_the_orthography_filter(tok):
    """The whole point of the filter: o2p generation from a pronounceable nonword."""
    enc = tok.encode(NONWORD, modality_filter="orthography")
    assert isinstance(enc, BridgeEncoding)
    assert enc.orthographic.enc_input_ids.shape[0] == 1


def test_nonword_returns_none_under_both(tok):
    assert tok.encode(NONWORD, modality_filter="both") is None


def test_nonword_returns_none_under_phonology(tok):
    assert tok.encode(NONWORD, modality_filter="phonology") is None


def test_a_batch_containing_a_nonword_fails_under_both(tok):
    assert tok.encode([WORD, NONWORD], modality_filter="both") is None


def test_a_batch_containing_a_nonword_succeeds_under_orthography(tok):
    enc = tok.encode([WORD, NONWORD], modality_filter="orthography")
    assert enc is not None
    assert len(enc) == 2


# --- placeholder components ------------------------------------------------


def test_placeholder_phonological_shape_and_values(tok):
    enc = tok.encode([WORD, NONWORD], modality_filter="orthography")
    phon = enc.phonological

    assert len(phon.enc_input_ids) == 2
    assert all(len(step) == 1 for step in phon.enc_input_ids)
    assert all(
        torch.equal(t, torch.tensor([tok.phon_pad_id])) for step in phon.enc_input_ids for t in step
    )
    assert phon.enc_pad_mask.shape == (2, 1)
    assert phon.dec_pad_mask.shape == (2, 1)
    assert phon.enc_pad_mask.dtype == torch.bool
    assert bool(phon.enc_pad_mask.all()), "placeholder positions are entirely padding"


def test_placeholder_phonological_targets_span_the_full_vocabulary(tok):
    """Note the last dimension is ``get_vocabulary_size()``, not ``size - 1`` as the
    model's phonological logits use. Changing it would break BridgeEncoding validation."""
    enc = tok.encode([WORD, NONWORD], modality_filter="orthography")
    targets = enc.phonological.targets
    assert targets is not None
    assert targets.shape == (2, 1, tok.phoneme_tokenizer.get_vocabulary_size())
    assert targets.dtype == torch.long
    assert bool((targets == 0).all())


def test_placeholder_orthographic_shape_and_values(tok):
    enc = tok.encode(WORD, modality_filter="phonology")
    orth = enc.orthographic

    assert orth.enc_input_ids.shape == (1, 1)
    assert orth.dec_input_ids.shape == (1, 1)
    assert orth.enc_input_ids.dtype == torch.long
    assert bool((orth.enc_input_ids == 0).all())
    assert orth.enc_pad_mask.shape == (1, 1)
    assert bool(orth.enc_pad_mask.all())
    assert orth.targets is None


def test_placeholder_batch_size_follows_the_input(tok):
    enc = tok.encode([WORD, "dog", "hat"], modality_filter="phonology")
    assert enc.orthographic.enc_input_ids.shape == (3, 1)
    assert len(enc.phonological.enc_input_ids) == 3


def test_placeholder_encoder_and_decoder_masks_are_distinct_objects(tok):
    """They are separate fields; sharing one tensor would couple two independent masks."""
    enc = tok.encode(WORD, modality_filter="phonology")
    assert enc.orthographic.enc_pad_mask is not enc.orthographic.dec_pad_mask


# --- contract --------------------------------------------------------------


def test_both_components_are_always_populated(tok):
    """``BridgeEncoding`` requires both; every filter must therefore fill both."""
    for mf in ("both", "orthography", "phonology"):
        enc = tok.encode(WORD, modality_filter=mf)
        assert enc is not None, mf
        assert enc.orthographic is not None, mf
        assert enc.phonological is not None, mf


def test_invalid_modality_filter_raises(tok):
    with pytest.raises(ValueError, match="Invalid modality_filter"):
        tok.encode(WORD, modality_filter="nope")


def test_invalid_language_code_returns_none_rather_than_raising(tok):
    """The character tokenizer raises on an unsupported language; ``encode`` converts
    every failure into ``None`` so callers have a single failure mode."""
    assert tok.encode(WORD, language_map={WORD: "FR"}) is None


def test_string_and_single_element_list_agree(tok):
    single = tok.encode(WORD)
    listed = tok.encode([WORD])
    assert torch.equal(single.orthographic.enc_input_ids, listed.orthographic.enc_input_ids)


def test_vocabulary_sizes_come_from_the_child_tokenizers(tok):
    sizes = tok.get_vocabulary_sizes()
    assert sizes["orthographic"] == tok.char_tokenizer.get_vocabulary_size()
    assert sizes["phonological"] == tok.phoneme_tokenizer.get_vocabulary_size()


def test_special_ids_are_read_from_the_child_tokenizers(tok):
    assert tok.orth_bos_id == tok.char_tokenizer.char_2_idx["[BOS]"]
    assert tok.orth_eos_id == tok.char_tokenizer.char_2_idx["[EOS]"]
    assert tok.orth_pad_id == tok.char_tokenizer.char_2_idx["[PAD]"]
    assert tok.orth_spc_id == tok.char_tokenizer.char_2_idx[" "]
    assert tok.phon_bos_id == tok.phoneme_tokenizer.special_token_dims["[BOS]"]
    assert tok.phon_eos_id == tok.phoneme_tokenizer.special_token_dims["[EOS]"]
    assert tok.phon_pad_id == tok.phoneme_tokenizer.special_token_dims["[PAD]"]
    assert tok.phon_spc_id == tok.phoneme_tokenizer.special_token_dims["[SPC]"]


# --- phoneme reverse lookup (the float-keyed special-token map) ------------


def test_every_special_token_dim_maps_back_to_its_token(tok):
    pt = tok.phoneme_tokenizer
    for token, dim in pt.special_token_dims.items():
        assert pt.phoneme_vectors_to_word([torch.tensor(float(dim))]) == [token]


def test_unrecognised_dim_maps_to_unk(tok):
    pt = tok.phoneme_tokenizer
    assert pt.phoneme_vectors_to_word([torch.tensor(float(pt.vocabulary_size + 50))]) == ["[UNK]"]


def test_non_integral_dim_maps_to_unk(tok):
    """The lookup is a float-keyed dict; a fractional value must miss, exactly as the
    old linear ``==`` scan did."""
    pt = tok.phoneme_tokenizer
    assert pt.phoneme_vectors_to_word([torch.tensor(pt.base_dim + 0.5)]) == ["[UNK]"]
