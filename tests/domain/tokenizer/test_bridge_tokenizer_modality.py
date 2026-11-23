"""Pins a missing modality, the start of orthographic generation, and the tokenizer's ids.

Three properties nothing else asserts:

* **a missing modality is ``None``**: a nonword has no phonology, and the input says so by
  carrying ``None`` where the phonemes would be. No tensor stands in for it.
* **a data condition is not an argument error**: a word with no pronunciation gives
  ``None``; a language code that does not exist, or rows in the wrong form, raises.
* **``orth_start`` is what training feeds the decoder first**: ``[LANG, BOS]``, the first
  two ids of any target whose first word is in that language.

Oracles: hand-written ids, the lexicon, and the target encoding itself for the start.
"""

import pytest
import torch

from bridge.domain.datamodels import BridgeInput
from bridge.domain.tokenizer import BridgeTokenizer

# A nonword: orthographically fine, absent from the pronunciation lexicon.
NONWORD = "blathe"
WORD = "cat"


@pytest.fixture(scope="module")
def tok():
    return BridgeTokenizer()


# --- nonwords --------------------------------------------------------------


def test_the_nonword_is_in_no_lexicon(tok):
    """The premise of this module, checked against the lexicon itself."""
    assert NONWORD not in tok.phoneme_tokenizer.pronunciation_dict
    assert "en" in tok.phoneme_tokenizer.pronunciation_dict[WORD]


def test_a_nonword_input_carries_orthography_and_no_phonology(tok):
    """The reason an input may lack a modality: o2p generation from a pronounceable
    nonword."""
    inputs = tok.encode_input([NONWORD])
    assert isinstance(inputs, BridgeInput)
    assert inputs.phon is None
    assert len(inputs) == 1


def test_the_orthography_of_a_nonword_is_what_a_word_would_get(tok):
    """Oracle: the character tokenizer. Missing phonology changes nothing else."""
    inputs = tok.encode_input([WORD, NONWORD], ["EN", "ES"])
    want = tok.char_tokenizer.encode_input([WORD, NONWORD], ["EN", "ES"])
    assert inputs.phon is None
    assert torch.equal(inputs.orth.ids, want.ids)
    assert torch.equal(inputs.orth.pad_mask, want.pad_mask)


def test_a_nonword_gives_no_target(tok):
    assert tok.encode_target([NONWORD]) is None
    assert tok.encode_target([WORD, NONWORD]) is None


def test_the_phoneme_tokenizer_returns_none_for_a_nonword(tok):
    assert tok.phoneme_tokenizer.encode_input([WORD, NONWORD]) is None
    assert tok.phoneme_tokenizer.encode_target([WORD, NONWORD]) is None


def test_a_word_carries_both_modalities(tok):
    """The control for the tests above: ``None`` is not what every input carries."""
    inputs = tok.encode_input([WORD])
    assert inputs.orth is not None
    assert inputs.phon is not None


# --- a data condition is not an argument error -------------------------------


def test_an_unsupported_language_code_raises(tok):
    """A language BRIDGE does not have is the caller's mistake, not a data condition.

    A typo in a language code and a word absent from the pronunciation lexicon must not
    be the same return value, or a caller cannot tell them apart.
    """
    with pytest.raises(ValueError, match="Invalid languages"):
        tok.encode_input([WORD], ["FR"])
    with pytest.raises(ValueError, match="Invalid languages"):
        tok.encode_target([WORD], ["FR"])


def test_a_word_absent_from_the_lexicon_is_not_an_error(tok):
    """The control for the test above, and the reason its failure means what it claims.

    Same call shape, a supported language, a word with no pronunciation. If this raised
    too then ``None`` would mean nothing; if the test above returned ``None`` the two
    conditions would be conflated.
    """
    assert tok.encode_input(["zzqxwv"], ["EN"]).phon is None
    assert tok.encode_target(["zzqxwv"], ["EN"]) is None


def test_an_argument_error_is_raised_for_a_nonword_too(tok):
    """The argument is checked before the lexicon is read, so a nonword does not hide a
    bad language code behind a ``None``."""
    with pytest.raises(ValueError, match="Invalid languages"):
        tok.encode_input(["zzqxwv"], ["FR"])
    with pytest.raises(ValueError, match="Invalid languages"):
        tok.encode_target(["zzqxwv"], ["FR"])


@pytest.mark.parametrize("encoder", ["encode_input", "encode_target"])
def test_rows_are_a_list_of_strings(tok, encoder):
    """One form only: a bare string is not a one-row list."""
    with pytest.raises(TypeError, match="The rows to encode are a list of strings"):
        getattr(tok, encoder)(WORD)


@pytest.mark.parametrize("encoder", ["encode_input", "encode_target"])
def test_languages_are_one_entry_per_row(tok, encoder):
    """One form only: a list or tuple with an entry for each row. A lone code is not
    spread over the rows, and a wrong count raises."""
    encode = getattr(tok, encoder)
    with pytest.raises(ValueError, match="one entry per row, got 'EN'"):
        encode([WORD, "dog"], "EN")
    with pytest.raises(ValueError, match="1 language entries for 2 rows"):
        encode([WORD, "dog"], ["EN"])
    assert encode([WORD, "dog"], ("EN", None)) is not None


# --- the start of orthographic generation ------------------------------------


def test_orth_start_is_lang_then_bos_for_each_row(tok):
    """Oracle: hand-written ids. 6, 7 and 8 are ``--``, ``EN`` and ``ES``; 0 is ``[BOS]``."""
    start = tok.orth_start(["ES", "EN", None, "es"])
    assert start.tolist() == [[8, 0], [7, 0], [6, 0], [8, 0]]
    assert start.dtype == torch.long


def test_orth_start_is_what_training_feeds_the_decoder_first(tok):
    """Oracle: the target encoding. Whatever follows, a target whose first word is in a
    language opens with that language's start, so generation continues from a state
    training produced. The third row changes language after its first word."""
    texts, languages = ["cat", "no me", "the gato"], ["EN", "ES", "EN ES"]
    target = tok.encode_target(texts, languages)
    assert torch.equal(tok.orth_start(["EN", "ES", "EN"]), target.orth.ids[:, :2])


def test_orth_start_takes_one_code_per_row(tok):
    """It names the first word's language. Later words are the decoder's to write."""
    with pytest.raises(ValueError, match="2 language codes"):
        tok.orth_start(["EN ES"])
    with pytest.raises(ValueError, match="Invalid languages"):
        tok.orth_start(["FR"])
    with pytest.raises(ValueError, match="one entry per row"):
        tok.orth_start("EN")


# --- contract --------------------------------------------------------------


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
    """The lookup is a float-keyed dict; a fractional value must miss, exactly as a
    linear ``==`` scan would."""
    pt = tok.phoneme_tokenizer
    assert pt.phoneme_vectors_to_word([torch.tensor(pt.base_dim + 0.5)]) == ["[UNK]"]


# --- phonological labels ---------------------------------------------------


def test_phonological_labels_are_the_fed_phonemes_one_position_on(tok):
    """Oracle: the phoneme table. The decoder is fed ``[BOS], K, AE1, T`` and scored
    against the features of ``K, AE1, T, [EOS]``, without the ``[PAD]`` column."""
    table = tok.phoneme_tokenizer.phoneme_table
    target = tok.encode_target([WORD], ["EN"]).phon
    fed = [table.row_index["[BOS]"], *(table.row_of(p) for p in ("K", "AE1", "T"))]
    scored = [*fed[1:], table.row_index["[EOS]"]]

    assert target.ids.tolist() == [fed]
    assert torch.equal(target.labels[0], table.multihot[scored][:, :-1].long())


def test_padded_label_positions_carry_the_loss_ignore_index(tok):
    """Oracle: the rule. A padded position is ``phon_pad_id`` in every feature, so the
    loss skips it; it is never "every feature off", which would be scored."""
    target = tok.encode_target(["a", "elephant"], ["EN", "EN"]).phon
    assert bool(target.pad_mask[0].any()), "the short row must be padded"
    assert bool((target.labels[0][target.pad_mask[0]] == tok.phon_pad_id).all())
    assert bool((target.labels[1] != tok.phon_pad_id).all())
    assert target.labels.shape[-1] == tok.phoneme_tokenizer.get_vocabulary_size() - 1


def test_labels_do_not_alias_the_shared_target_table():
    """A target owns its labels.

    They are gathered from one table the tokenizer keeps for its lifetime. A view into
    it would let a single write rewrite a phoneme's label for every later target, and one
    tokenizer is shared between a run's datasets.
    """
    tokenizer = BridgeTokenizer()
    phoneme_tokenizer = tokenizer.phoneme_tokenizer
    before = phoneme_tokenizer._target_table.clone()

    target = tokenizer.encode_target(["a", "elephant"], ["EN", "EN"])
    assert target is not None
    target.phon.labels[:] = 999
    assert torch.equal(phoneme_tokenizer._target_table, before)
