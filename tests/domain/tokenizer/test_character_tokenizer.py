"""Pins ``CharacterTokenizer`` encode/decode output.

``encode`` was rewritten from an element-by-element tensor fill into building Python
rows and one ``torch.tensor(rows)`` call, and its return type changed from a dict
subclass to ``EncodingComponent``. Every orthographic tensor the model ever sees comes
out of here, so the layout is pinned exactly.

Sequence layout (see ``CharacterTokenizer.encode``):
    encoder input: ``[LANG, BOS, *chars, EOS, PAD...]``  -> width 3 + max_len
    decoder input: ``[LANG, BOS, *chars, PAD...]``       -> width 2 + max_len
Downstream code slices ``[:, 2:]`` to skip LANG and BOS, so the two leading tokens are
load-bearing.
"""

import string

import pytest
import torch

from bridge.domain.datamodels import EncodingComponent
from bridge.domain.tokenizer import CharacterTokenizer


@pytest.fixture(scope="module")
def tok():
    return CharacterTokenizer()


def ids(tok, chars):
    return [tok.char_2_idx[c] for c in chars]


# --- structure -------------------------------------------------------------


def test_encode_returns_an_encoding_component(tok):
    assert isinstance(tok.encode("cat"), EncodingComponent)


def test_encoder_row_layout(tok):
    out = tok.encode("cat")
    expected = ids(tok, ["--", "[BOS]", "c", "a", "t", "[EOS]"])
    assert out.enc_input_ids.tolist() == [expected]


def test_decoder_row_layout_has_no_eos(tok):
    out = tok.encode("cat")
    expected = ids(tok, ["--", "[BOS]", "c", "a", "t"])
    assert out.dec_input_ids.tolist() == [expected]


def test_widths_are_three_and_two_plus_max_length(tok):
    out = tok.encode(["a", "abcde"])
    assert out.enc_input_ids.shape == (2, 3 + 5)
    assert out.dec_input_ids.shape == (2, 2 + 5)


def test_dtypes_are_long_and_bool(tok):
    out = tok.encode("cat")
    assert out.enc_input_ids.dtype == torch.long
    assert out.dec_input_ids.dtype == torch.long
    assert out.enc_pad_mask.dtype == torch.bool
    assert out.dec_pad_mask.dtype == torch.bool


def test_string_and_single_element_list_agree(tok):
    a, b = tok.encode("cat"), tok.encode(["cat"])
    assert torch.equal(a.enc_input_ids, b.enc_input_ids)
    assert torch.equal(a.dec_input_ids, b.dec_input_ids)


# --- padding ---------------------------------------------------------------


def test_shorter_rows_are_right_padded(tok):
    """The rewrite builds each row in Python; a row shorter than the tensor width
    would silently become trailing zeros (which is [BOS], not [PAD])."""
    out = tok.encode(["ab", "abcde"])
    pad = tok.char_2_idx["[PAD]"]

    short = out.enc_input_ids[0].tolist()
    assert short == ids(tok, ["--", "[BOS]", "a", "b", "[EOS]"]) + [pad] * 3
    assert out.enc_input_ids[1].tolist() == ids(
        tok, ["--", "[BOS]", "a", "b", "c", "d", "e", "[EOS]"]
    )


def test_pad_mask_marks_exactly_the_pad_positions(tok):
    out = tok.encode(["ab", "abcde"])
    pad = tok.char_2_idx["[PAD]"]
    assert torch.equal(out.enc_pad_mask, out.enc_input_ids == pad)
    assert torch.equal(out.dec_pad_mask, out.dec_input_ids == pad)


def test_no_row_is_left_unfilled(tok):
    """Every position is a real vocabulary id — never an accidental zero."""
    out = tok.encode(["a", "bb", "ccc"])
    assert (out.enc_input_ids >= 0).all()
    assert (out.enc_input_ids < len(tok.vocab)).all()


def test_empty_string_encodes_to_lang_bos_eos(tok):
    out = tok.encode([""])
    assert out.enc_input_ids.tolist() == [ids(tok, ["--", "[BOS]", "[EOS]"])]
    assert out.dec_input_ids.tolist() == [ids(tok, ["--", "[BOS]"])]


def test_empty_string_beside_a_longer_one_is_padded(tok):
    out = tok.encode(["", "abc"])
    pad = tok.char_2_idx["[PAD]"]
    assert out.enc_input_ids[0].tolist() == ids(tok, ["--", "[BOS]", "[EOS]"]) + [pad] * 3


# --- language tokens -------------------------------------------------------


def test_language_token_occupies_position_zero(tok):
    out = tok.encode(["cat", "gato"], language_map={"cat": "EN", "gato": "ES"})
    assert out.enc_input_ids[0, 0].item() == tok.char_2_idx["EN"]
    assert out.enc_input_ids[1, 0].item() == tok.char_2_idx["ES"]


def test_language_lookup_is_case_insensitive_on_the_word(tok):
    out = tok.encode(["CAT"], language_map={"cat": "ES"})
    assert out.enc_input_ids[0, 0].item() == tok.char_2_idx["ES"]


def test_words_absent_from_the_map_get_the_placeholder(tok):
    out = tok.encode(["cat", "dog"], language_map={"cat": "ES"})
    assert out.enc_input_ids[0, 0].item() == tok.char_2_idx["ES"]
    assert out.enc_input_ids[1, 0].item() == tok.char_2_idx["--"]


def test_unsupported_language_raises(tok):
    with pytest.raises(ValueError, match="Invalid languages"):
        tok.encode(["bonjour"], language_map={"bonjour": "FR"})


def test_language_value_case_is_accepted_either_way(tok):
    """Validation upper-cases before checking membership."""
    tok.encode(["cat"], language_map={"cat": "es"})


# --- unknown characters ----------------------------------------------------


def test_characters_outside_the_vocabulary_become_unk(tok):
    out = tok.encode(["caté"])
    unk = tok.char_2_idx["[UNK]"]
    assert out.enc_input_ids[0].tolist()[5] == unk


def test_every_printable_character_round_trips(tok):
    printable = "".join(c for c in string.printable if c not in "\x0b\x0c")
    out = tok.encode([printable])
    assert tok.decode(out.enc_input_ids.tolist()) == [printable]


# --- input validation ------------------------------------------------------


@pytest.mark.parametrize("bad", [123, ["ok", 1], [None], {"a": 1}])
def test_non_string_input_raises_type_error(tok, bad):
    with pytest.raises(TypeError, match="Input must be a string or a list of strings"):
        tok.encode(bad)


# --- decode ----------------------------------------------------------------


def test_decode_strips_special_and_language_tokens(tok):
    out = tok.encode(["cat"], language_map={"cat": "ES"})
    assert tok.decode(out.enc_input_ids.tolist()) == ["cat"]


def test_decode_strips_padding(tok):
    out = tok.encode(["ab", "abcde"])
    assert tok.decode(out.enc_input_ids.tolist()) == ["ab", "abcde"]


def test_decode_handles_multiple_rows(tok):
    out = tok.encode(["cat", "dog"])
    assert tok.decode(out.enc_input_ids.tolist()) == ["cat", "dog"]


def test_decode_of_empty_row_is_empty_string(tok):
    assert tok.decode([[]]) == [""]


def test_decode_rejects_out_of_range_indices(tok):
    with pytest.raises(KeyError):
        tok.decode([[10**6]])


def test_decode_preserves_spaces(tok):
    """A space is an ordinary printable character, not a special token."""
    out = tok.encode(["a b"])
    assert tok.decode(out.enc_input_ids.tolist()) == ["a b"]


# --- vocabulary ------------------------------------------------------------


def test_vocabulary_size_matches_the_vocab_list(tok):
    assert tok.get_vocabulary_size() == len(tok.vocab)
    assert len(tok.char_2_idx) == len(tok.vocab)


def test_vocabulary_order_is_specials_then_languages_then_printable(tok):
    """Index order is baked into checkpoints and into VocabSpec's id fields."""
    assert tok.vocab[: len(tok.special_tokens)] == tok.special_tokens
    start = len(tok.special_tokens)
    assert tok.vocab[start : start + len(tok.language_tokens)] == tok.language_tokens
    assert tok.char_2_idx["[BOS]"] == 0
    assert tok.char_2_idx["[EOS]"] == 1


def test_idx_2_char_is_the_inverse_of_char_2_idx(tok):
    assert all(tok.idx_2_char[i] == c for c, i in tok.char_2_idx.items())
