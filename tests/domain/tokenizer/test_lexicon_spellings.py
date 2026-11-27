"""Every spelling in every shipped lexicon is one the tokenizers can hold (issue #237).

A word the character tokenizer cannot spell is trained and scored as a different word:
a character outside the vocabulary encodes as ``[UNK]`` and decoding leaves it out.

Oracle: the round trip. ``decode(encode(w)) == w`` for every key of every lexicon file,
read from the files as shipped and not through the phoneme tokenizer, which lower-cases
its keys. Ids and strings are compared exactly.

The decomposed spellings are made here with ``unicodedata``, from the lexicon's own words,
so no accented literal in this file decides the result.
"""

import json
import unicodedata
from pathlib import Path

import pytest

import bridge
from bridge.domain.tokenizer import BridgeTokenizer, CharacterTokenizer, PhonemeTokenizer

LEXICON_DIR = Path(bridge.__file__).parent / "core" / "pronunciation_lexicons"
LEXICONS = sorted(LEXICON_DIR.glob("*.json"))


def spellings(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return list(json.load(f))


def language_of(path: Path) -> str:
    return path.stem.upper()


@pytest.fixture(scope="module")
def tok():
    return CharacterTokenizer()


@pytest.fixture(scope="module")
def bridge_tok():
    return BridgeTokenizer()


@pytest.fixture(scope="module")
def accented() -> list[tuple[str, str, str]]:
    """``(language, word, the same word decomposed)`` for every lexicon word that has a
    decomposed form."""
    found = [
        (language_of(path), word, unicodedata.normalize("NFD", word))
        for path in LEXICONS
        for word in spellings(path)
    ]
    return [(language, word, apart) for language, word, apart in found if apart != word]


def test_both_shipped_lexicons_are_read():
    """The tests below pass on no words at all, so the words are counted first."""
    assert {path.stem for path in LEXICONS} >= {"en", "es"}
    assert all(len(spellings(path)) > 1000 for path in LEXICONS)


@pytest.fixture(scope="module", params=LEXICONS, ids=lambda path: path.name)
def lexicon(request, tok):
    """``(file name, every word of one lexicon, its input ids)``. Encoded once for the
    module: the English lexicon is 123472 rows and takes about a second."""
    path = request.param
    words = spellings(path)
    return path.name, words, tok.encode_input(words, [language_of(path)] * len(words)).ids


def test_every_lexicon_word_survives_encode_then_decode(tok, lexicon):
    name, words, ids = lexicon
    decoded = tok.decode(ids.tolist())
    lost = [(w, back) for w, back in zip(words, decoded, strict=True) if back != w]
    assert not lost, (
        f"{len(lost)} of {len(words)} words in {name} do not survive encode then decode, "
        f"the first {lost[:5]}"
    )


def test_no_lexicon_word_holds_an_unknown_character(tok, lexicon):
    """The round trip above would also pass if decoding learnt to write ``[UNK]`` back
    as some character. This reads the ids."""
    name, words, ids = lexicon
    rows_with_unk = (ids == tok.char_2_idx["[UNK]"]).any(dim=1)
    assert int(rows_with_unk.sum()) == 0, (
        name,
        [w for w, bad in zip(words, rows_with_unk.tolist(), strict=True) if bad][:5],
    )


def test_a_decomposed_lexicon_word_encodes_as_the_word(tok, accented):
    """An accented letter typed as its base letter and a combining mark is the same
    letter. Every lexicon word that can be written that way gives one row of ids
    either way, as an input and as a target, and decodes to the spelling the lexicon
    holds. These are the words that hold a character outside ASCII, so this is also
    where a target's ids and labels are read back."""
    assert len(accented) > 1000
    languages = [language for language, _, _ in accented]
    words = [word for _, word, _ in accented]
    apart = [decomposed for _, _, decomposed in accented]

    assert tok.encode_input(apart, languages).ids.tolist() == (
        tok.encode_input(words, languages).ids.tolist()
    )
    target, expected = tok.encode_target(apart, languages), tok.encode_target(words, languages)
    assert target.ids.tolist() == expected.ids.tolist()
    assert target.labels.tolist() == expected.labels.tolist()
    assert tok.decode(tok.encode_input(apart, languages).ids.tolist()) == words
    assert tok.decode(target.ids.tolist()) == words
    assert tok.decode(target.labels.tolist()) == words


def test_a_decomposed_lexicon_word_keeps_its_pronunciation(bridge_tok, accented):
    """The phoneme side looks a word up by its spelling, so it has to read the same
    spelling the character side does. A decomposed word that missed its lexicon entry
    would come back with orthography and no phonology."""
    languages = [language for language, _, _ in accented]
    words = [word for _, word, _ in accented]
    apart = [decomposed for _, _, decomposed in accented]

    composed = bridge_tok.encode_input(words, languages)
    decomposed = bridge_tok.encode_input(apart, languages)
    assert composed.phon is not None
    assert decomposed.phon is not None, "a decomposed spelling missed its lexicon entry"
    assert decomposed.phon.ids.tolist() == composed.phon.ids.tolist()
    assert decomposed.orth.ids.tolist() == composed.orth.ids.tolist()

    target = bridge_tok.encode_target(apart, languages)
    expected = bridge_tok.encode_target(words, languages)
    assert target is not None and expected is not None
    assert target.phon.labels.tolist() == expected.phon.labels.tolist()
    assert target.orth.labels.tolist() == expected.orth.labels.tolist()


def test_an_upper_case_accented_word_keeps_its_letters_and_its_pronunciation(bridge_tok, accented):
    """The two sides treat case differently, and both must hold for an accented capital.
    Orthography keeps it, so the word decodes as written. Phonology looks the word up in
    lower case, so it is pronounced as its lexicon entry, composed or decomposed."""
    spanish = [(word, apart) for language, word, apart in accented if language == "ES"]
    assert len(spanish) > 1000
    words = [word for word, _ in spanish]
    upper = [word.upper() for word in words]
    upper_apart = [apart.upper() for _, apart in spanish]
    languages = ["ES"] * len(words)
    assert all(u != w for u, w in zip(upper, words, strict=True))

    lower = bridge_tok.encode_input(words, languages)
    for rows in (upper, upper_apart):
        encoded = bridge_tok.encode_input(rows, languages)
        assert bridge_tok.char_tokenizer.decode(encoded.orth.ids.tolist()) == upper
        assert encoded.phon is not None
        assert encoded.phon.ids.tolist() == lower.phon.ids.tolist()


def test_a_decomposed_word_keeps_its_place_in_a_row_of_several_languages(bridge_tok, accented):
    """A combining mark is not a word and not a space, so composing a row moves no word
    boundary: each word keeps its language, and the language token sits where it did."""
    spanish = [(word, apart) for language, word, apart in accented if language == "ES"]
    rows = [f"the {word} is {word}" for word, _ in spanish[:200]]
    rows_apart = [f"the {apart} is {apart}" for _, apart in spanish[:200]]
    languages = ["EN ES EN ES"] * len(rows)

    composed = bridge_tok.encode_input(rows, languages)
    decomposed = bridge_tok.encode_input(rows_apart, languages)
    assert decomposed.orth.ids.tolist() == composed.orth.ids.tolist()
    assert decomposed.phon is not None and composed.phon is not None
    assert decomposed.phon.ids.tolist() == composed.phon.ids.tolist()
    assert bridge_tok.decode_with_languages(decomposed.orth.ids.tolist()) == [
        (row, ["EN", "ES", "EN", "ES"]) for row in rows
    ]


@pytest.mark.parametrize("filed_as", ["NFC", "NFD"])
def test_a_custom_lexicon_entry_is_found_however_its_key_was_typed(tmp_path, filed_as):
    """A caller's own lexicon may hold a key in either form, and in either case. The
    entry is filed under the composed lower-case spelling, which is what a lookup uses.

    The word is in no shipped lexicon, so without the custom file it has no phonology:
    that is the control showing the entry found is the custom one."""
    word = unicodedata.normalize("NFC", "zzu" + chr(0x0308) + "en" + chr(0x0303) + "o")
    apart = unicodedata.normalize("NFD", word)
    assert len(word) == 6 and len(apart) == 8
    assert PhonemeTokenizer().encode_input([word], ["ES"]) is None

    path = tmp_path / "custom.json"
    key = unicodedata.normalize(filed_as, word).upper()
    path.write_text(json.dumps({key: {"es": [["K", "EH1"]]}}), encoding="utf-8")
    custom = PhonemeTokenizer(custom_cmudict_path=str(path))

    expected = custom.encode_input(["que"], ["ES"])
    assert expected is not None
    for spelling in (word, apart, word.upper(), apart.upper()):
        found = custom.encode_input([spelling], ["ES"])
        assert found is not None, [f"U+{ord(c):04X}" for c in spelling]
        assert found.ids.tolist() == expected.ids.tolist()
