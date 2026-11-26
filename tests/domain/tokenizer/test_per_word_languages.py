"""Every word of a row carries its own language, and both tokenizers honour it by position.

Orthography opens a row with its first word's language token and writes a language token
again only where the language changes, directly before that word's first character.
Phonology has no language token; it looks each word up in its own language's lexicon.

Each test names its oracle. Three kinds are used:

* **hand-written**: token names and phoneme rows typed out here, read off the design and
  the lexicon files, sharing no code with the tokenizers;
* **independent reference**: ``expected_layout`` walks a row character by character and
  ``expected_phonemes`` indexes the lexicon dict directly, where the tokenizers use the
  shared ``\\S+`` word function;
* **invariant**: a row alone equals the same row in any batch, and decoding then encoding
  returns the ids it started from.

A row is encoded twice, as an input and as a target. The two share one layout, so most
tests check both: the input ids, the target's decoder ids, which are the same row without
its last token, and the target's labels, which are the same row without its first.

The words are homographs on purpose. "no", "me", "la", "con", "sin" and "son" are in both
lexicons with different pronunciations, so the tag, never the spelling, decides the
phonemes, and a tokenizer that looks a language up by string cannot pass.
"""

import itertools
import sys

import pytest
import torch

from bridge.domain.datamodels import Target
from bridge.domain.tokenizer import BridgeTokenizer
from bridge.domain.tokenizer.languages import per_row_languages, per_word_languages, split_words

HOMOGRAPHS = ["no", "me", "la", "con", "sin", "son"]

# Phoneme rows read from the lexicon files and phonreps.csv: "no" is N OW1 in English and
# N AO1 in Spanish, "me" is M IY1 and M EH1. 86, 87 and 89 are [BOS], [EOS] and [SPC].
NO_ME_SPANISH = [86, 69, 22, 89, 68, 27, 87]
NO_ME_ENGLISH = [86, 69, 33, 89, 68, 32, 87]


@pytest.fixture(scope="module")
def tok():
    return BridgeTokenizer()


def names(tok, ids):
    return [tok.char_tokenizer.idx_2_char[int(i)] for i in ids]


def expected_layout(text: str, languages: list[str]) -> list[str]:
    """The input row as token names, built by walking the characters one at a time."""
    row = [languages[0], "[BOS]"]
    word, inside_a_word = -1, False
    for ch in text:
        if ch.isspace():
            inside_a_word = False
        elif not inside_a_word:
            inside_a_word = True
            word += 1
            if word > 0 and languages[word] != languages[word - 1]:
                row.append(languages[word])
        row.append(ch)
    return [*row, "[EOS]"]


def expected_phonemes(tok, text: str, languages: list[str]) -> list[int]:
    """The input rows of a phrase, each word read straight out of its language's lexicon."""
    lexicon = tok.phoneme_tokenizer.pronunciation_dict
    table = tok.phoneme_tokenizer.phoneme_table
    rows = [table.row_index["[BOS]"]]
    for i, (word, language) in enumerate(zip(text.split(), languages, strict=True)):
        if i:
            rows.append(table.row_index["[SPC]"])
        rows.extend(table.row_of(p) for p in lexicon[word][language.lower()][0])
    return [*rows, table.row_index["[EOS]"]]


def mixed_rows() -> list[tuple[str, list[str]]]:
    """Every EN/ES pattern of 2 to 4 words, constant patterns included: 28 rows."""
    words = itertools.cycle(HOMOGRAPHS)
    rows = []
    for n in (2, 3, 4):
        for pattern in itertools.product(["EN", "ES"], repeat=n):
            rows.append((" ".join(next(words) for _ in range(n)), list(pattern)))
    return rows


MIXED_ROWS = mixed_rows()


# --- layout ----------------------------------------------------------------


def test_a_mixed_row_is_laid_out_token_for_token(tok):
    """Oracle: hand-written. The design's own example, typed out as token names.

    The switch sits after the space and directly before ``g``. Decoding cannot see which
    side of the space it is on, so only a literal layout pins it.
    """
    texts, languages = ["the word for cat is gato"], ["EN EN EN EN EN ES"]
    source = tok.encode_input(texts, languages).orth
    target = tok.encode_target(texts, languages).orth
    expected = ["EN", "[BOS]"] + list("the word for cat is ") + ["ES"] + list("gato") + ["[EOS]"]
    assert names(tok, source.ids[0]) == expected
    assert names(tok, target.ids[0]) == expected[:-1]
    # The switch is a label like any letter: the space before it is scored against it.
    assert names(tok, target.labels[0]) == expected[1:]
    assert not source.pad_mask.any()
    assert not target.pad_mask.any()


@pytest.mark.parametrize(
    ("text", "languages", "expected"),
    [
        # A switch and a switch back each get a token.
        ("no me no", "ES EN ES", ["ES", "[BOS]", *"no ", "EN", *"me ", "ES", *"no", "[EOS]"]),
        # Two words in a row in one language share the token that opened them.
        ("no me no", "ES ES EN", ["ES", "[BOS]", *"no me ", "EN", *"no", "[EOS]"]),
        ("no me no", "EN ES ES", ["EN", "[BOS]", *"no ", "ES", *"me no", "[EOS]"]),
        # Whitespace is kept as written, and the token follows all of it.
        ("no  me", "ES EN", ["ES", "[BOS]", *"no  ", "EN", *"me", "[EOS]"]),
        ("no\tme ", "ES EN", ["ES", "[BOS]", *"no\t", "EN", *"me ", "[EOS]"]),
        # Leading whitespace does not move the opening token or add one.
        (" no me", "EN ES", ["EN", "[BOS]", *" no ", "ES", *"me", "[EOS]"]),
        # A no-break space is whitespace too, though the character vocabulary lacks it.
        ("no\u00a0me", "ES EN", ["ES", "[BOS]", *"no", "[UNK]", "EN", *"me", "[EOS]"]),
        # One language, however it is spelled, is one token.
        ("no me no", "ES", ["ES", "[BOS]", *"no me no", "[EOS]"]),
        ("no me no", "ES ES ES", ["ES", "[BOS]", *"no me no", "[EOS]"]),
        ("no me no", None, ["--", "[BOS]", *"no me no", "[EOS]"]),
    ],
)
def test_a_language_token_appears_only_where_the_language_changes(tok, text, languages, expected):
    """Oracle: hand-written layouts, one per rule of where a token goes."""
    source = tok.char_tokenizer.encode_input([text], [languages])
    target = tok.char_tokenizer.encode_target([text], [languages])
    assert names(tok, source.ids[0]) == expected
    assert names(tok, target.ids[0]) == expected[:-1]
    assert names(tok, target.labels[0]) == expected[1:]


@pytest.mark.parametrize(("text", "languages"), MIXED_ROWS)
def test_every_pattern_matches_the_reference_layout_and_lexicon(tok, text, languages):
    """Oracle: independent reference, for both modalities, over every pattern.

    A tokenizer that gave the whole row its first word's language passes only the
    constant patterns, 6 of the 28.
    """
    inputs = tok.encode_input([text], [languages])
    target = tok.encode_target([text], [languages])
    layout, phonemes = expected_layout(text, languages), expected_phonemes(tok, text, languages)
    assert names(tok, inputs.orth.ids[0]) == layout
    assert inputs.phon.ids[0].tolist() == phonemes
    # The decoder is fed the same tokens without the final one, and is scored against
    # the same tokens without the first.
    assert names(tok, target.orth.ids[0]) == layout[:-1]
    assert names(tok, target.orth.labels[0]) == layout[1:]
    assert target.phon.ids[0].tolist() == phonemes[:-1]
    table = tok.phoneme_tokenizer.phoneme_table
    assert torch.equal(target.phon.labels[0], table.multihot[phonemes[1:]][:, :-1].long())


# --- phonology -------------------------------------------------------------


def test_a_spanish_phrase_gets_spanish_phonemes(tok):
    """Oracle: hand-written rows. The regression for a phrase tagged ES read as English.

    The English rows are asserted too, as the control: the two differ at positions 2 and
    5, so the Spanish assertion cannot pass on a tokenizer that ignores the tag.
    """
    assert tok.encode_input(["no me"], ["ES"]).phon.ids.tolist() == [NO_ME_SPANISH]
    assert tok.encode_input(["no me"], ["EN"]).phon.ids.tolist() == [NO_ME_ENGLISH]
    # A target is read by the same rule: the decoder is fed the row without its [EOS].
    assert tok.encode_target(["no me"], ["ES"]).phon.ids.tolist() == [NO_ME_SPANISH[:-1]]
    assert tok.encode_target(["no me"], ["EN"]).phon.ids.tolist() == [NO_ME_ENGLISH[:-1]]


def test_capitals_are_kept_in_orthography_and_ignored_by_the_lexicon(tok):
    """Oracle: hand-written. "No Me" tagged ES is spelled as written and pronounced as "no me"."""
    out = tok.encode_input(["No Me"], ["ES"])
    assert names(tok, out.orth.ids[0]) == ["ES", "[BOS]", *"No Me", "[EOS]"]
    assert out.phon.ids.tolist() == [NO_ME_SPANISH]


@pytest.mark.parametrize("text", ["no  me", "no\tme ", " no me", "no\u00a0me"])
def test_phonology_reads_the_same_words_whatever_separates_them(tok, text):
    """Oracle: hand-written rows. Spanish "no", ``[SPC]``, English "me", for every spacing.

    The spacings are the ones the layout test above pins for orthography, so the two
    tokenizers are held to the same word boundaries from both sides.
    """
    out = tok.encode_input([text], ["ES EN"]).phon
    assert out.ids.tolist() == [[86, 69, 22, 89, 68, 32, 87]]


def test_each_word_is_read_in_its_own_language_by_position(tok):
    """Oracle: hand-written rows. One spelling, two languages, in one row.

    "no no" tagged ``EN ES`` holds the English "no" then the Spanish one. A lookup keyed by
    the string can give both words only one language.
    """
    out = tok.encode_input(["no no"], ["EN ES"]).phon
    assert out.ids.tolist() == [[86, 69, 33, 89, 69, 22, 87]]


def test_no_language_gives_the_unspecified_token_and_english_phonemes(tok):
    """Oracle: hand-written. A direct call with no language is ``--`` over English."""
    for languages in (None, [None], ["--"]):
        out = tok.encode_input(["no me"], languages)
        assert names(tok, out.orth.ids[0]) == ["--", "[BOS]", *"no me", "[EOS]"]
        assert out.phon.ids.tolist() == [NO_ME_ENGLISH]
        target = tok.encode_target(["no me"], languages)
        assert names(tok, target.orth.ids[0]) == ["--", "[BOS]", *"no me"]
        assert target.phon.ids.tolist() == [NO_ME_ENGLISH[:-1]]


def test_a_word_missing_from_its_language_is_not_given_english_phonemes(tok):
    """Oracle: the lexicon. "the" has an English entry and no Spanish one.

    Tagged ES it is a word with no pronunciation, like any other, so the row has no
    phonology and is no target. Reading the English entry instead would put English
    phonemes under an ES tag. The EN-tagged calls are the control: the same strings do
    encode.
    """
    entry = tok.phoneme_tokenizer.pronunciation_dict["the"]
    assert "en" in entry and "es" not in entry, "the premise of this test"

    for texts, languages in (
        (["the"], ["ES"]),
        (["the gato"], ["ES ES"]),
        (["no", "the"], ["ES"] * 2),
    ):
        assert tok.encode_input(texts, languages).phon is None
        assert tok.encode_target(texts, languages) is None

    for texts, languages in ((["the"], ["EN"]), (["the gato"], ["EN ES"])):
        assert tok.encode_input(texts, languages).phon is not None
        assert tok.encode_target(texts, languages) is not None
    # Orthography needs no lexicon, so the tag alone is fine there.
    spelled = tok.encode_input(["the"], ["ES"]).orth
    assert names(tok, spelled.ids[0]) == ["ES", "[BOS]", *"the", "[EOS]"]


def test_a_word_missing_from_english_is_not_given_spanish_phonemes(tok):
    """Oracle: the lexicon. "gato" has a Spanish entry and no English one.

    The same rule from the other side. Tagged EN, or left unspecified, which reads the
    English lexicon, it has no pronunciation. The ES-tagged call is the control.
    """
    entry = tok.phoneme_tokenizer.pronunciation_dict["gato"]
    assert "es" in entry and "en" not in entry, "the premise of this test"

    for texts, languages in ((["gato"], ["EN"]), (["gato"], None), (["the gato"], ["EN EN"])):
        assert tok.encode_input(texts, languages).phon is None
        assert tok.encode_target(texts, languages) is None

    assert tok.encode_input(["gato"], ["ES"]).phon is not None
    assert tok.encode_target(["gato"], ["ES"]) is not None


def test_a_target_opens_with_its_first_words_language(tok):
    """Oracle: hand-written. What the decoder is fed first is ``[first word's LANG, BOS]``,
    and ``orth_start`` gives exactly those two ids for the same languages."""
    target = tok.encode_target(["no me", "no me", "no me"], ["ES EN", "EN ES", None]).orth
    assert [names(tok, row[:2]) for row in target.ids] == [
        ["ES", "[BOS]"],
        ["EN", "[BOS]"],
        ["--", "[BOS]"],
    ]
    assert torch.equal(tok.orth_start(["ES", "EN", None]), target.ids[:, :2])


# --- a row does not depend on its neighbours --------------------------------


def unpadded(inputs, target, row: int) -> dict[str, torch.Tensor]:
    """One row of an input and a target with its padding cut off: the ids of all four
    parts and the labels of the two targets."""
    out = {}
    for name, part in (
        ("input.orth", inputs.orth),
        ("input.phon", inputs.phon),
        ("target.orth", target.orth),
        ("target.phon", target.phon),
    ):
        keep = ~part.pad_mask[row]
        # Padding is a suffix: everything kept comes before everything dropped.
        assert bool(keep[: int(keep.sum())].all())
        out[f"{name}.ids"] = part.ids[row][keep]
        if isinstance(part, Target):
            out[f"{name}.labels"] = part.labels[row][: int(keep.sum())]
    return out


def test_a_row_encodes_the_same_alone_and_in_any_batch(tok):
    """Oracle: invariant. A row's ids are a function of the row, not of its batch.

    The rows include one spelling under two languages ("no" as EN and as ES) and a phrase
    beside a single word that shares a spelling with it in the other language. A batch-wide
    string lookup changes those rows depending on what they sit next to.
    """
    rows = [
        ("no", "EN"),
        ("no", "ES"),
        ("no me", "EN"),
        ("no me", "ES"),
        ("no me", "ES EN"),
        ("me no la", "EN ES ES"),
        ("con", None),
        *[(text, " ".join(langs)) for text, langs in MIXED_ROWS[4:12]],
    ]
    alone = [
        unpadded(tok.encode_input([text], [languages]), tok.encode_target([text], [languages]), 0)
        for text, languages in rows
    ]

    orders = [list(range(len(rows))), list(reversed(range(len(rows))))]
    generator = torch.Generator().manual_seed(0)
    orders += [torch.randperm(len(rows), generator=generator).tolist() for _ in range(4)]
    for order in orders:
        for size in (2, 5, len(rows)):
            for start in range(0, len(rows), size):
                picked = order[start : start + size]
                texts, languages = [rows[i][0] for i in picked], [rows[i][1] for i in picked]
                inputs = tok.encode_input(texts, languages)
                target = tok.encode_target(texts, languages)
                for position, i in enumerate(picked):
                    got = unpadded(inputs, target, position)
                    assert got.keys() == alone[i].keys() and len(got) == 6
                    for name, want in alone[i].items():
                        assert torch.equal(got[name], want), (
                            f"{rows[i]} {name} changed inside {picked}"
                        )


# --- decoding ---------------------------------------------------------------


def test_decode_with_languages_returns_the_text_and_each_words_language(tok):
    """Oracle: hand-written. ``--`` comes back as ``None``, one entry per word."""
    texts = ["the word is gato", "no me", "no  me no", "long"]
    languages = ["EN EN EN ES", "ES", "ES EN ES", None]
    source = tok.char_tokenizer.encode_input(texts, languages)
    target = tok.char_tokenizer.encode_target(texts, languages)
    expected = [
        ("the word is gato", ["EN", "EN", "EN", "ES"]),
        ("no me", ["ES", "ES"]),
        ("no  me no", ["ES", "EN", "ES"]),
        ("long", [None]),
    ]
    assert tok.decode_with_languages(source.ids.tolist()) == expected
    assert tok.decode_with_languages(target.ids.tolist()) == expected
    # The text half is what `decode` returns.
    assert tok.decode(source.ids.tolist())["orthographic"] == [t for t, _ in expected]


def test_a_word_takes_the_language_in_force_at_its_first_character(tok):
    """Oracle: hand-written, on ids no tokenizer call writes but a decoder can emit.

    A language token inside a word does not relabel that word; it applies from the next
    word on. A token on the wrong side of the space reads the same as one on the right
    side, which is why the layout tests above, not decoding, pin where it goes.
    """
    ids = tok.char_tokenizer.char_2_idx
    rows = [
        ["EN", "[BOS]", "g", "a", "ES", "t", "o", " ", "n", "o", "[EOS]"],
        ["EN", "[BOS]", "n", "o", "ES", " ", "m", "e", "[EOS]", "[PAD]"],
        ["[BOS]", "n", "o", " ", "ES", "m", "e"],
        ["ES", "[BOS]", "n", "o", " ", "--", "m", "e", "[EOS]"],
    ]
    assert tok.decode_with_languages([[ids[t] for t in row] for row in rows]) == [
        ("gato no", ["EN", "ES"]),
        ("no me", ["EN", "ES"]),
        ("no me", [None, "ES"]),
        ("no me", ["ES", None]),
    ]


def test_an_unknown_character_is_dropped_and_its_word_keeps_its_language(tok):
    """Oracle: hand-written. ``é`` encodes as ``[UNK]``, which decoding leaves out.

    The word it sat in is still a word, so the row keeps one language per word.
    """
    out = tok.char_tokenizer.encode_input(["qué pasa"], ["ES EN"])
    assert names(tok, out.ids[0]) == [
        "ES",
        "[BOS]",
        *"qu",
        "[UNK]",
        " ",
        "EN",
        *"pasa",
        "[EOS]",
    ]
    assert tok.decode_with_languages(out.ids.tolist()) == [("qu pasa", ["ES", "EN"])]


def test_an_unspecified_row_decodes_to_none_per_word_and_encodes_back(tok):
    """Oracle: round trip. ``None`` per word is how ``--`` decodes, and it encodes as ``--``.

    ``None`` beside a real code is ``--`` beside a real code, and raises the same way.
    """
    first = tok.encode_input(["no me"])
    ((text, languages),) = tok.decode_with_languages(first.orth.ids.tolist())
    assert (text, languages) == ("no me", [None, None])

    again = tok.encode_input([text], [languages])
    assert torch.equal(again.orth.ids, first.orth.ids)
    assert torch.equal(again.phon.ids, first.phon.ids)
    with pytest.raises(ValueError, match="whole row is unspecified"):
        tok.encode_input(["no me"], [[None, "ES"]])


@pytest.mark.parametrize(("text", "languages"), MIXED_ROWS)
def test_decoding_then_encoding_returns_the_same_ids(tok, text, languages):
    """Oracle: round trip. Encoding what ``decode_with_languages`` returns is the identity.

    Both halves are checked: the decoded text and languages equal what was encoded, and
    encoding them again reproduces every id, as an input and as a target. A decode that
    ignored switches returns one language for the row and fails the first half on every
    mixed pattern.
    """
    chars = tok.char_tokenizer
    first = chars.encode_input([text], [languages])
    ((decoded_text, decoded_languages),) = tok.decode_with_languages(first.ids.tolist())
    assert (decoded_text, decoded_languages) == (text, languages)

    assert torch.equal(chars.encode_input([decoded_text], [decoded_languages]).ids, first.ids)
    target = chars.encode_target([text], [languages])
    assert tok.decode_with_languages(target.ids.tolist()) == [(text, languages)]
    again = chars.encode_target([decoded_text], [decoded_languages])
    assert torch.equal(again.ids, target.ids)
    assert torch.equal(again.labels, target.labels)


def test_padding_does_not_change_what_a_row_decodes_to(tok):
    """Oracle: round trip, on rows padded to the width of the longest in one batch."""
    texts = [text for text, _ in MIXED_ROWS]
    languages = [langs for _, langs in MIXED_ROWS]
    source = tok.char_tokenizer.encode_input(texts, languages)
    target = tok.char_tokenizer.encode_target(texts, languages)
    assert bool(source.pad_mask.any()), "the batch must contain padding"
    assert bool(target.pad_mask.any()), "the batch must contain padding"
    expected = [(text, langs) for text, langs in MIXED_ROWS]
    assert tok.decode_with_languages(source.ids.tolist()) == expected
    assert tok.decode_with_languages(target.ids.tolist()) == expected


# --- which arguments are accepted ---------------------------------------------


@pytest.mark.parametrize("encoder", ["encode_input", "encode_target"])
def test_a_code_count_must_be_one_or_the_number_of_words(tok, encoder):
    """Oracle: the rule. Two codes for three words raises, naming the row and the counts."""
    encode = getattr(tok, encoder)
    with pytest.raises(ValueError, match=r"2 language codes for 3 words in 'no me no'"):
        encode(["no me no"], ["ES EN"])
    with pytest.raises(ValueError, match=r"4 language codes for 3 words in 'no me no'"):
        encode(["no me no"], [["ES", "EN", "ES", "EN"]])
    with pytest.raises(ValueError, match=r"2 language codes for 1 words in 'no'"):
        encode(["no me", "no"], ["ES EN", "ES EN"])
    with pytest.raises(ValueError, match=r"3 language entries for 2 rows"):
        encode(["no me", "no"], ["ES", "EN", "ES"])


def test_an_unsupported_code_raises_the_same_error_everywhere(tok):
    """Oracle: the rule, plus agreement. One message, whichever tokenizer is asked."""
    chars, phonemes = tok.char_tokenizer, tok.phoneme_tokenizer
    calls = {
        "input": lambda: tok.encode_input(["no me"], ["FR"]),
        "target": lambda: tok.encode_target(["no me"], ["FR"]),
        "one word": lambda: tok.encode_input(["no me"], ["ES FR"]),
        # Wrong in two ways, a code and the count: the code is reported.
        "wrong count too": lambda: tok.encode_target(["no me no"], ["FR EN"]),
        "batch": lambda: tok.encode_input(["no", "me"], ["ES", "FR"]),
        "start": lambda: tok.orth_start(["ES", "FR"]),
        "characters, input": lambda: chars.encode_input(["no me"], ["FR"]),
        "characters, target": lambda: chars.encode_target(["no me"], ["FR"]),
        "phonemes, input": lambda: phonemes.encode_input(["no me"], ["FR"]),
        "phonemes, target": lambda: phonemes.encode_target(["no me"], ["FR"]),
    }
    messages = {}
    for name, call in calls.items():
        with pytest.raises(ValueError, match="Invalid languages") as raised:
            call()
        messages[name] = str(raised.value)
    assert set(messages.values()) == {
        "Invalid languages: ['FR']. Supported languages are: ['EN', 'ES']"
    }, messages


@pytest.mark.parametrize("languages", ["-- EN", "ES --", ["--", "ES"], ["EN", "--"], ["EN", None]])
def test_unspecified_beside_a_real_code_raises(tok, languages):
    """Oracle: the rule. ``--`` covers a whole row or none of it."""
    with pytest.raises(ValueError, match="whole row is unspecified"):
        tok.encode_input(["no me"], [languages])
    with pytest.raises(ValueError, match="whole row is unspecified"):
        tok.encode_target(["no me"], [languages])


@pytest.mark.parametrize("languages", ["", "   ", [], ()])
def test_a_blank_language_raises(tok, languages):
    """Oracle: the rule. ``None`` means unspecified; an empty value is a mistake."""
    with pytest.raises(ValueError, match="No language code given for 'no me'"):
        tok.encode_input(["no me"], [languages])
    with pytest.raises(ValueError, match="No language code given for 'no me'"):
        tok.encode_target(["no me"], [languages])


@pytest.mark.parametrize(
    ("text", "languages"),
    [
        # A set has no order, so which word gets which code would follow the hash seed.
        ("no me", {"ES", "EN"}),
        # A dict keyed by word, with keys that happen to be codes.
        ("en es", {"en": "ES", "es": "EN"}),
        ("no me", {"no": "ES", "me": "EN"}),
        ("no me", iter(["ES", "EN"])),
        ("no me", 5),
    ],
)
def test_a_rows_languages_are_a_string_a_list_or_a_tuple(tok, text, languages):
    """Oracle: the rule. Codes are matched to words by position, so the container must
    have one, and the row is named."""
    with pytest.raises(
        ValueError, match=f"a string, a list or a tuple of codes, got .* for '{text}'"
    ):
        tok.encode_input([text], [languages])
    with pytest.raises(
        ValueError, match=f"a string, a list or a tuple of codes, got .* for '{text}'"
    ):
        tok.encode_target([text], [languages])


@pytest.mark.parametrize(
    "languages",
    [{"ES", "EN"}, {"no": "ES", "me": "EN"}, 5, iter(["ES", "EN"]), "ES", "ES EN"],
)
def test_the_languages_of_a_call_are_a_list_or_a_tuple(tok, languages):
    """Oracle: the rule, one level up: the entries are matched to rows by position.

    A string is refused too. Read as one code for every row it would be a second way to
    write the same call, and ``"ES EN"`` beside two rows would have two readings.
    """
    with pytest.raises(ValueError, match="Languages are a list or a tuple with one entry per"):
        tok.encode_input(["no", "me"], languages)
    with pytest.raises(ValueError, match="Languages are a list or a tuple with one entry per"):
        tok.encode_target(["no", "me"], languages)


@pytest.mark.parametrize("encoder", ["encode_input", "encode_target"])
def test_a_language_is_not_given_by_a_word_keyed_map(tok, encoder):
    """Oracle: the signature. Neither encoder has a ``language_map`` keyword.

    A keyword that was accepted and ignored would encode the row as unspecified, with
    English phonemes, and say nothing.
    """
    with pytest.raises(TypeError, match="language_map"):
        getattr(tok, encoder)(["hola"], language_map={"hola": "ES"})


def test_codes_are_normalised_to_upper_case(tok):
    """Oracle: differential against the upper-case spelling of the same call."""
    lower = tok.encode_input(["no me"], ["es en"])
    upper = tok.encode_input(["no me"], ["ES EN"])
    assert torch.equal(lower.orth.ids, upper.orth.ids)
    assert torch.equal(lower.phon.ids, upper.phon.ids)
    assert names(tok, lower.orth.ids[0])[0] == "ES"


def test_languages_hold_one_entry_per_row(tok):
    """Oracle: hand-written. An entry is its row's languages, whichever form it takes:
    one code for the row, a string of codes, a list of codes, or ``None``."""
    languages = ["ES", "ES EN", ["EN", "ES"], None]
    rows = tok.char_tokenizer.encode_input(["no me", "la con", "sin son", "no la"], languages)
    assert tok.decode_with_languages(rows.ids.tolist()) == [
        ("no me", ["ES", "ES"]),
        ("la con", ["ES", "EN"]),
        ("sin son", ["EN", "ES"]),
        ("no la", [None, None]),
    ]


# --- the shared pieces ---------------------------------------------------------


def test_per_word_languages_returns_one_code_per_word():
    """Oracle: hand-written. The stored form: a tuple, one code per word."""
    assert per_word_languages("no me no", "ES") == ("ES", "ES", "ES")
    assert per_word_languages("no me no", "es EN es") == ("ES", "EN", "ES")
    assert per_word_languages("no me no", ("ES", "EN", "ES")) == ("ES", "EN", "ES")
    assert per_word_languages("no me no") == ("--", "--", "--")
    assert per_word_languages("no me no", "--") == ("--", "--", "--")
    assert per_word_languages("no me no", [None, None, None]) == ("--", "--", "--")
    # A row with no words still opens with a language token, so it carries one code.
    assert per_word_languages("", "ES") == ("ES",)
    assert per_word_languages("  ") == ("--",)
    with pytest.raises(ValueError, match="2 language codes for 0 words"):
        per_word_languages(" ", "ES EN")


def test_resolved_languages_are_accepted_back_unchanged():
    """Oracle: round trip. What the resolver returns is itself a valid argument.

    The dataset stores what the resolver returns and hands it to the tokenizers, which
    resolve it again, so resolving must be idempotent.
    """
    texts = ["no me no", "la", "", "con sin"]
    resolved = per_row_languages(texts, ["ES EN ES", "EN", "ES", None])
    assert resolved == [("ES", "EN", "ES"), ("EN",), ("ES",), ("--", "--")]
    assert per_row_languages(texts, resolved) == resolved
    # No languages at all leaves every row unspecified.
    assert per_row_languages(texts) == [("--",) * 3, ("--",), ("--",), ("--",) * 2]


def test_the_word_function_agrees_with_str_split():
    """Oracle: reference implementation, ``str.split``, on every whitespace character.

    Covers each code point Python counts as whitespace plus every code point below
    U+3100, each placed between two letters. The dataset's word counts and the tokenizers'
    word boundaries cannot drift apart while this holds.
    """
    whitespace = [chr(c) for c in range(sys.maxunicode + 1) if chr(c).isspace()]
    assert len(whitespace) >= 25, "the sweep must reach the Unicode spaces"
    candidates = set(whitespace) | {chr(c) for c in range(0x3100)}
    for ch in candidates:
        text = f" a{ch}b{ch}{ch}c "
        assert [m.group() for m in split_words(text)] == text.split(), hex(ord(ch))
    text = "  the\tword \n for  "
    assert [(m.start(), m.group()) for m in split_words(text)] == [
        (2, "the"),
        (6, "word"),
        (13, "for"),
    ]
