"""Pins ``CharacterTokenizer`` output: what the encoder reads and what the decoder is taught.

Every orthographic tensor the model ever sees comes out of here, so the layout is pinned
exactly.

Sequence layout:
    input ids:     ``[LANG, BOS, *chars, EOS, PAD...]``  -> width 3 + max_len
    target ids:    ``[LANG, BOS, *chars, PAD...]``       -> width 2 + max_len
    target labels: ``[BOS, *chars, EOS, PAD...]``        -> width 2 + max_len
The label at a position is the token one position on, so ``[BOS]`` predicts the first
character (``docs/decisions/0004``). ``LANG`` is the first word's language. A row whose
words change language holds one more language token per change, counted in ``max_len``;
that layout is pinned in ``test_per_word_languages.py``.

Oracle: rows written out by hand from the layout above.
"""

import string

import pytest
import torch

from bridge.domain.datamodels import Target, Tokens
from bridge.domain.tokenizer import CharacterTokenizer


@pytest.fixture(scope="module")
def tok():
    return CharacterTokenizer()


def ids(tok, chars):
    return [tok.char_2_idx[c] for c in chars]


# --- structure -------------------------------------------------------------


def test_each_encoder_returns_its_own_type(tok):
    assert isinstance(tok.encode_input(["cat"]), Tokens)
    assert isinstance(tok.encode_target(["cat"]), Target)


def test_input_row_layout(tok):
    out = tok.encode_input(["cat"])
    expected = ids(tok, ["--", "[BOS]", "c", "a", "t", "[EOS]"])
    assert out.ids.tolist() == [expected]


def test_target_row_layout_has_no_eos_in_the_ids(tok):
    out = tok.encode_target(["cat"])
    expected = ids(tok, ["--", "[BOS]", "c", "a", "t"])
    assert out.ids.tolist() == [expected]


def test_target_labels_are_the_ids_one_position_on_ending_in_eos(tok):
    """Oracle: hand-written. ``[BOS]`` is paired with the first character and the last
    character with ``[EOS]``; the opening language token is never a label."""
    out = tok.encode_target(["cat"])
    assert out.labels.tolist() == [ids(tok, ["[BOS]", "c", "a", "t", "[EOS]"])]
    pairs = list(zip(out.ids[0].tolist(), out.labels[0].tolist(), strict=True))
    assert pairs == list(
        zip(
            ids(tok, ["--", "[BOS]", "c", "a", "t"]),
            ids(tok, ["[BOS]", "c", "a", "t", "[EOS]"]),
            strict=True,
        )
    )


def test_labels_of_padded_rows_are_padded_after_eos(tok):
    """Oracle: hand-written. A short row's labels end ``[EOS], [PAD]...``, so the loss,
    which ignores ``[PAD]``, scores every real position and no padding."""
    out = tok.encode_target(["ab", "abcde"])
    pad = tok.char_2_idx["[PAD]"]
    assert out.ids[0].tolist() == ids(tok, ["--", "[BOS]", "a", "b"]) + [pad] * 3
    assert out.labels[0].tolist() == ids(tok, ["[BOS]", "a", "b", "[EOS]"]) + [pad] * 3
    assert out.labels[1].tolist() == ids(tok, ["[BOS]", "a", "b", "c", "d", "e", "[EOS]"])


def test_widths_are_three_and_two_plus_max_length(tok):
    assert tok.encode_input(["a", "abcde"]).ids.shape == (2, 3 + 5)
    target = tok.encode_target(["a", "abcde"])
    assert target.ids.shape == (2, 2 + 5)
    assert target.labels.shape == (2, 2 + 5)


def test_dtypes_are_long_and_bool(tok):
    source, target = tok.encode_input(["cat"]), tok.encode_target(["cat"])
    assert source.ids.dtype == torch.long
    assert target.ids.dtype == torch.long
    assert target.labels.dtype == torch.long
    assert source.pad_mask.dtype == torch.bool
    assert target.pad_mask.dtype == torch.bool


def test_input_and_target_of_one_string_share_everything_but_the_eos(tok):
    """Oracle: the layout. The target ids are the input ids without ``[EOS]``, and the
    labels are the input ids without the opening language token."""
    source, target = tok.encode_input(["cat", "a b"]), tok.encode_target(["cat", "a b"])
    assert torch.equal(target.labels, source.ids[:, 1:])
    eos = tok.char_2_idx["[EOS]"]
    for source_row, target_row in zip(source.ids.tolist(), target.ids.tolist(), strict=True):
        assert [i for i in source_row if i != eos][: len(target_row)] == target_row


# --- padding ---------------------------------------------------------------


def test_shorter_rows_are_right_padded(tok):
    """Each row is built in Python; a row shorter than the tensor width must not
    become trailing zeros (which is [BOS], not [PAD])."""
    out = tok.encode_input(["ab", "abcde"])
    pad = tok.char_2_idx["[PAD]"]

    short = out.ids[0].tolist()
    assert short == ids(tok, ["--", "[BOS]", "a", "b", "[EOS]"]) + [pad] * 3
    assert out.ids[1].tolist() == ids(tok, ["--", "[BOS]", "a", "b", "c", "d", "e", "[EOS]"])


def test_pad_mask_marks_exactly_the_pad_positions(tok):
    pad = tok.char_2_idx["[PAD]"]
    source, target = tok.encode_input(["ab", "abcde"]), tok.encode_target(["ab", "abcde"])
    assert torch.equal(source.pad_mask, source.ids == pad)
    assert torch.equal(target.pad_mask, target.ids == pad)
    assert bool(source.pad_mask.any()) and bool(target.pad_mask.any())


def test_no_row_is_left_unfilled(tok):
    """Every position is a real vocabulary id, never an accidental zero."""
    out = tok.encode_input(["a", "bb", "ccc"])
    assert (out.ids >= 0).all()
    assert (out.ids < len(tok.vocab)).all()


def test_empty_string_encodes_to_lang_bos_eos(tok):
    assert tok.encode_input([""]).ids.tolist() == [ids(tok, ["--", "[BOS]", "[EOS]"])]
    target = tok.encode_target([""])
    assert target.ids.tolist() == [ids(tok, ["--", "[BOS]"])]
    assert target.labels.tolist() == [ids(tok, ["[BOS]", "[EOS]"])]


def test_empty_string_beside_a_longer_one_is_padded(tok):
    out = tok.encode_input(["", "abc"])
    pad = tok.char_2_idx["[PAD]"]
    assert out.ids[0].tolist() == ids(tok, ["--", "[BOS]", "[EOS]"]) + [pad] * 3


# --- language tokens -------------------------------------------------------


def test_language_token_occupies_position_zero(tok):
    for out in (
        tok.encode_input(["cat", "gato"], ["EN", "ES"]),
        tok.encode_target(["cat", "gato"], ["EN", "ES"]),
    ):
        assert out.ids[0, 0].item() == tok.char_2_idx["EN"]
        assert out.ids[1, 0].item() == tok.char_2_idx["ES"]


def test_a_row_given_no_language_gets_the_placeholder(tok):
    out = tok.encode_input(["cat", "dog"], ["ES", None])
    assert out.ids[0, 0].item() == tok.char_2_idx["ES"]
    assert out.ids[1, 0].item() == tok.char_2_idx["--"]


def test_language_code_case_is_accepted_either_way(tok):
    """A lower-case code is the same language, and writes that language's token."""
    out = tok.encode_input(["cat"], ["es"])
    assert out.ids[0, 0].item() == tok.char_2_idx["ES"]


# --- unknown characters ----------------------------------------------------


def test_characters_outside_the_vocabulary_become_unk(tok):
    out = tok.encode_input(["caté"])
    unk = tok.char_2_idx["[UNK]"]
    assert out.ids[0].tolist()[5] == unk


def test_every_printable_character_round_trips(tok):
    printable = "".join(c for c in string.printable if c not in "\x0b\x0c")
    out = tok.encode_input([printable])
    assert tok.decode(out.ids.tolist()) == [printable]


# --- input validation ------------------------------------------------------


@pytest.mark.parametrize("bad", [123, "cat", ("cat",), ["ok", 1], [None], {"a": 1}])
def test_rows_are_a_list_of_strings(tok, bad):
    """One form only. A bare string is not a one-row list: it raises like any other."""
    with pytest.raises(TypeError, match="The rows to encode are a list of strings"):
        tok.encode_input(bad)
    with pytest.raises(TypeError, match="The rows to encode are a list of strings"):
        tok.encode_target(bad)


# --- decode ----------------------------------------------------------------


def test_decode_strips_special_and_language_tokens(tok):
    out = tok.encode_input(["cat"], ["ES"])
    assert tok.decode(out.ids.tolist()) == ["cat"]


def test_decode_strips_padding(tok):
    out = tok.encode_input(["ab", "abcde"])
    assert tok.decode(out.ids.tolist()) == ["ab", "abcde"]


def test_decode_handles_multiple_rows(tok):
    out = tok.encode_input(["cat", "dog"])
    assert tok.decode(out.ids.tolist()) == ["cat", "dog"]


def test_decode_of_empty_row_is_empty_string(tok):
    assert tok.decode([[]]) == [""]


def test_decode_rejects_out_of_range_indices(tok):
    with pytest.raises(KeyError):
        tok.decode([[10**6]])


def test_decode_preserves_spaces(tok):
    """A space is an ordinary printable character, not a special token."""
    out = tok.encode_input(["a b"])
    assert tok.decode(out.ids.tolist()) == ["a b"]


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
