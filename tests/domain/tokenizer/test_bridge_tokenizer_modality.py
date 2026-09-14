"""Pins ``BridgeTokenizer.encode``'s modality routing and placeholder components.

``encode`` was three ~90-line branches, each wrapping every statement in its own
``try/except ... return None``. It is now one path. Two properties that the branch
structure enforced implicitly, and which nothing else asserts, are pinned here:

* **short-circuit**: a filter that excludes a modality must not invoke that
  tokenizer at all. This is what lets ``"orthography"`` encode a nonword that has no
  entry in the pronunciation lexicon.
* **placeholder shape**: the excluded modality is filled with a stub so
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

    # Phoneme *row* ids: one all-[PAD] position per batch item.
    pad_row = tok.phoneme_tokenizer.phoneme_table.row_index["[PAD]"]
    assert phon.enc_input_ids.shape == (2, 1)
    assert torch.equal(phon.enc_input_ids, torch.full((2, 1), pad_row))
    assert torch.equal(phon.dec_input_ids, torch.full((2, 1), pad_row))
    assert phon.enc_pad_mask.shape == (2, 1)
    assert phon.dec_pad_mask.shape == (2, 1)
    assert phon.enc_pad_mask.dtype == torch.bool
    assert bool(phon.enc_pad_mask.all()), "placeholder positions are entirely padding"


def test_placeholder_phonological_targets_match_the_real_ones(tok):
    """A placeholder must be usable anywhere a real phonological component is.

    That means the same width as ``encode`` produces, one narrower than the feature
    vocabulary since [PAD] is not a predicted class, and filled with the loss
    ignore_index so the position contributes nothing rather than being scored as
    "every feature off".
    """
    targets = tok.encode([WORD, NONWORD], modality_filter="orthography").phonological.targets
    real = tok.encode([WORD, "dog"]).phonological.targets
    assert targets is not None
    assert targets.shape == (2, 1, tok.phoneme_tokenizer.get_vocabulary_size() - 1)
    assert targets.shape[-1] == real.shape[-1]
    assert targets.dtype == torch.long
    assert bool((targets == tok.phon_pad_id).all())


def test_placeholder_orthographic_is_a_valid_minimal_prefix(tok):
    """Two positions holding ``[--, BOS]``, not a column of zeros.

    Zero is ``[BOS]``, so the old placeholder read as a sequence that had already started,
    and it was too narrow to supply the ``[LANG, BOS]`` prefix ``Model.generate`` seeds the
    orthographic decoder with. ``p2o`` runs on exactly this component, so a placeholder that
    could not supply the prefix would put a special case in the model for one pathway.
    ``--`` is the unspecified-language token: a phonology-only encoding does not say what
    language to spell in, and saying so is better than defaulting to one. See issue #228.
    """
    enc = tok.encode(WORD, modality_filter="phonology")
    orth = enc.orthographic
    char_2_idx = tok.char_tokenizer.char_2_idx

    assert orth.enc_input_ids.shape == (1, 2)
    assert orth.dec_input_ids.shape == (1, 2)
    assert orth.enc_input_ids.dtype == torch.long
    assert orth.dec_input_ids[0].tolist() == [char_2_idx["--"], char_2_idx["[BOS]"]]
    assert torch.equal(orth.enc_input_ids, orth.dec_input_ids)
    assert orth.enc_pad_mask.shape == (1, 2)
    assert bool(orth.enc_pad_mask.all()), "the placeholder is entirely padding"
    assert orth.targets is None


def test_the_placeholder_prefix_matches_what_a_real_encoding_puts_there(tok):
    """The placeholder and a real encoding must agree on what the first two positions are.

    This is the property that lets generation seed from ``dec_input_ids[:, :2]`` with no
    branch for the placeholder. Comparing against the real tokenizer output rather than
    against hardcoded ids means the two cannot drift apart.
    """
    placeholder = tok.encode(WORD, modality_filter="phonology").orthographic
    real = tok.encode(WORD).orthographic

    assert placeholder.dec_input_ids.shape[1] == 2
    assert real.dec_input_ids[:, 1].tolist() == placeholder.dec_input_ids[:, 1].tolist(), (
        "both must carry [BOS] at position 1"
    )
    # Position 0 is a language token in both; the real one reflects the caller's map, the
    # placeholder is unspecified.
    language_ids = {tok.char_tokenizer.char_2_idx[t] for t in tok.char_tokenizer.language_tokens}
    assert int(real.dec_input_ids[0, 0]) in language_ids
    assert int(placeholder.dec_input_ids[0, 0]) in language_ids


def test_placeholder_batch_size_follows_the_input(tok):
    enc = tok.encode([WORD, "dog", "hat"], modality_filter="phonology")
    assert enc.orthographic.enc_input_ids.shape == (3, 2)
    # Real phonology here, so only the batch dimension is fixed; the width follows the words.
    assert enc.phonological.enc_input_ids.shape[0] == 3


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


def test_placeholder_targets_do_not_alias_the_shared_target_table():
    """The placeholder must own its storage.

    ``padded_targets`` builds every placeholder position from the one [PAD] row of the
    tokenizer's target table. Returning an expanded view of that row would give the
    caller zero-stride aliases, so a single write would rewrite [PAD] for every later
    ``encode``. ``TrainingPipeline`` now shares one tokenizer between the train and test
    datasets, so that write would poison both splits.
    """
    tokenizer = BridgeTokenizer()
    phoneme_tokenizer = tokenizer.phoneme_tokenizer
    pad_row = phoneme_tokenizer.phoneme_table.row_index["[PAD]"]
    before = phoneme_tokenizer._target_table[pad_row].clone()

    encoding = tokenizer.encode([WORD], modality_filter="orthography")
    assert encoding is not None
    targets = encoding.phonological.phon_targets
    assert torch.equal(targets[0, 0], before)

    targets[0, 0, 0] = 999
    assert torch.equal(phoneme_tokenizer._target_table[pad_row], before)
