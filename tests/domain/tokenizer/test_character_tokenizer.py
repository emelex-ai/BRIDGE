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
import unicodedata

import pytest
import torch

from bridge.domain.datamodels import Target, Tokens
from bridge.domain.tokenizer import CharacterTokenizer


@pytest.fixture(scope="module")
def tok():
    return CharacterTokenizer()


def ids(tok, chars):
    return [tok.char_2_idx[c] for c in chars]


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
    out = tok.encode_input(["cat"], ["es"])
    assert out.ids[0, 0].item() == tok.char_2_idx["ES"]


def test_characters_outside_the_vocabulary_become_unk(tok):
    out = tok.encode_input(["catç"])
    unk = tok.char_2_idx["[UNK]"]
    assert out.ids[0].tolist()[5] == unk


def test_every_printable_character_round_trips(tok):
    printable = "".join(c for c in string.printable if c not in "\x0b\x0c")
    out = tok.encode_input([printable])
    assert tok.decode(out.ids.tolist()) == [printable]


@pytest.mark.parametrize("bad", [123, "cat", ("cat",), ["ok", 1], [None], {"a": 1}])
def test_rows_are_a_list_of_strings(tok, bad):
    """One form only. A bare string is not a one-row list: it raises like any other."""
    with pytest.raises(TypeError, match="The rows to encode are a list of strings"):
        tok.encode_input(bad)
    with pytest.raises(TypeError, match="The rows to encode are a list of strings"):
        tok.encode_target(bad)


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


# Written out by hand, code point and id, so the test cannot agree with the constant by
# reading it. An id here is a row of every checkpoint's orthographic embedding.
SPANISH_IDS = [
    ("á", 0x00E1, 109),
    ("é", 0x00E9, 110),
    ("í", 0x00ED, 111),
    ("ó", 0x00F3, 112),
    ("ú", 0x00FA, 113),
    ("ü", 0x00FC, 114),
    ("ñ", 0x00F1, 115),
    ("Á", 0x00C1, 116),
    ("É", 0x00C9, 117),
    ("Í", 0x00CD, 118),
    ("Ó", 0x00D3, 119),
    ("Ú", 0x00DA, 120),
    ("Ü", 0x00DC, 121),
    ("Ñ", 0x00D1, 122),
    ("¿", 0x00BF, 123),
    ("¡", 0x00A1, 124),
]


def test_the_spanish_characters_follow_everything_else_and_no_earlier_id_moves(tok):
    """Oracle: hand-written. The 109 entries that came first are the special tokens, the
    language tokens and ``string.printable``, at the ids they have always had."""
    before = [*tok.special_tokens, *tok.language_tokens, *string.printable]
    assert len(before) == 109
    assert tok.vocab[:109] == before
    assert [(ch, tok.char_2_idx[ch]) for ch in tok.vocab[109:]] == [
        (chr(code_point), idx) for _, code_point, idx in SPANISH_IDS
    ]
    assert tok.get_vocabulary_size() == 125


@pytest.mark.parametrize(("char", "code_point", "idx"), SPANISH_IDS)
def test_a_spanish_character_is_one_composed_code_point_with_its_own_id(tok, char, code_point, idx):
    """A literal typed as a letter plus a combining mark would look the same in this
    file and encode as two tokens, the second ``[UNK]``."""
    assert [ord(c) for c in char] == [code_point]
    assert unicodedata.normalize("NFC", char) == char
    assert tok.encode_input([char]).ids.tolist() == [
        [tok.char_2_idx["--"], tok.char_2_idx["[BOS]"], idx, tok.char_2_idx["[EOS]"]]
    ]
    assert tok.decode([[idx]]) == [char]


def test_case_is_kept(tok):
    """Nothing folds case: each case of a letter has its own id and decodes as written."""
    for lower, upper in [("que", "QUE"), ("\u00f1u", "\u00d1U"), ("\u00e9", "\u00c9")]:
        ids_lower = tok.encode_input([lower]).ids.tolist()
        ids_upper = tok.encode_input([upper]).ids.tolist()
        assert ids_lower != ids_upper
        assert tok.decode(ids_lower) == [lower]
        assert tok.decode(ids_upper) == [upper]


@pytest.mark.parametrize(
    ("composed", "decomposed"),
    [
        ("qu\u00e9", "que\u0301"),
        ("a\u00f1o", "an\u0303o"),
        ("ping\u00fcino", "pingu\u0308ino"),
        ("\u00c1RBOL", "A\u0301RBOL"),
        ("el ni\u00f1o", "el nin\u0303o"),
    ],
    ids=["que", "ano", "pinguino", "ARBOL", "el nino"],
)
def test_a_letter_and_its_combining_mark_encode_as_the_composed_letter(tok, composed, decomposed):
    """Oracle: hand-written pairs. Both sides are spelt in escapes, so neither depends on
    how this file was saved. The two spell one word, so they are one row of ids, in an
    input and in a target, and the row decodes to the composed spelling."""
    assert len(decomposed) == len(composed) + 1
    assert tok.encode_input([decomposed]).ids.tolist() == tok.encode_input([composed]).ids.tolist()
    target, expected = tok.encode_target([decomposed]), tok.encode_target([composed])
    assert target.ids.tolist() == expected.ids.tolist()
    assert target.labels.tolist() == expected.labels.tolist()
    assert tok.char_2_idx["[UNK]"] not in target.labels.tolist()[0]
    assert tok.decode(tok.encode_input([decomposed]).ids.tolist()) == [composed]


def test_composition_is_all_that_is_done_to_a_row(tok):
    """Oracle: hand-written. ``c`` with a combining cedilla composes to a letter the
    vocabulary lacks. ``q`` with a combining acute accent composes to nothing, so the mark
    stays a character of its own. The ligature ``fi`` (U+FB01) is one character with no
    canonical decomposition: it stays one ``[UNK]``, where the compatibility forms NFKC
    and NFKD would spell it ``f``, ``i``."""
    unk = tok.char_2_idx["[UNK]"]
    a, n, q = (tok.char_2_idx[ch] for ch in "anq")
    assert tok.encode_input(["c\u0327a"]).ids.tolist()[0][2:-1] == [unk, a]
    assert tok.encode_input(["q\u0301a"]).ids.tolist()[0][2:-1] == [q, unk, a]
    assert tok.encode_input(["\ufb01n"]).ids.tolist()[0][2:-1] == [unk, n]


def test_idx_2_char_is_the_inverse_of_char_2_idx(tok):
    assert all(tok.idx_2_char[i] == c for c, i in tok.char_2_idx.items())
