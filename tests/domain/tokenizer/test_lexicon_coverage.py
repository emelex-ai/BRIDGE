"""Every word of every shipped pronunciation lexicon encodes to phonemes the table has.

A symbol with no row in ``bridge/core/phonreps.csv`` encodes as the ``[UNK]`` row and
raises nothing, so a lexicon and a table that disagree train on unknowns in silence. Issue
#238: seven Spanish symbols had no row, and 2639 of the 7095 Spanish words held ``[UNK]``.
"""

import json
from pathlib import Path

import pytest

from bridge.domain.tokenizer.languages import SUPPORTED_LANGUAGES, split_words
from bridge.domain.tokenizer.phoneme_tokenizer import PhonemeTokenizer
from bridge.utils import get_project_root

LEXICON_DIR = Path(get_project_root()) / "bridge/core/pronunciation_lexicons"
LANGUAGES = sorted(path.stem for path in LEXICON_DIR.glob("*.json"))


def test_a_lexicon_is_shipped_for_each_supported_language():
    assert [language.upper() for language in LANGUAGES] == sorted(SUPPORTED_LANGUAGES)


@pytest.mark.parametrize("language", LANGUAGES)
def test_no_word_of_a_shipped_lexicon_encodes_to_unk(language):
    """Oracle: invariant, over the whole lexicon, through the call a dataset row takes.

    The tokenizer reads a word's first pronunciation and no other, so that is what this
    covers. A key that is not one word is left out: no row can look it up. The English
    lexicon holds one such key, the empty string.
    """
    with open(LEXICON_DIR / f"{language}.json", encoding="utf-8") as handle:
        lexicon = json.load(handle)
    words = [word for word in lexicon if len(split_words(word)) == 1]
    assert len(words) >= len(lexicon) - 1

    tokenizer = PhonemeTokenizer()
    table = tokenizer.phoneme_table
    encoded = tokenizer.encode_input(words, [language.upper()] * len(words))
    assert encoded is not None, f"a {language} word has no pronunciation in its own lexicon"

    holds_unk = (encoded.ids == table.row_index["[UNK]"]).any(dim=1).tolist()
    unknown = [word for word, flagged in zip(words, holds_unk, strict=True) if flagged]
    symbols = sorted({p for word in unknown for p in lexicon[word][0] if p not in table.row_index})
    assert not unknown, (
        f"{len(unknown)} of {len(words)} {language} words encode to [UNK], such as "
        f"{unknown[:5]}; symbols with no row in phonreps.csv: {symbols}"
    )


def test_a_symbol_with_no_row_does_encode_to_unk(tmp_path):
    """The control for the test above: it can only pass honestly if an unknown symbol
    reaches the ids as ``[UNK]``. A word is planted whose pronunciation holds a symbol no
    table has."""
    custom = tmp_path / "planted.json"
    custom.write_text(json.dumps({"zzplanted": {"es": [["K", "NOT_A_PHONEME", "AA0"]]}}))
    tokenizer = PhonemeTokenizer(custom_cmudict_path=str(custom))
    row = tokenizer.phoneme_table.row_index

    encoded = tokenizer.encode_input(["zzplanted"], ["ES"])

    assert encoded is not None
    assert encoded.ids.tolist() == [
        [row["[BOS]"], row["K"], row["[UNK]"], row["AA0"], row["[EOS]"]]
    ]
