"""Pins the phoneme -> feature table contract (issue #221).

The table is what lets the tokenizer and the model agree on phoneme identity without
holding references to each other, so its row order and its inverse mapping back to
feature sets are load-bearing.
"""

import json
import os
import re

import pandas as pd
import torch

from bridge.core.phonreps import SPECIAL_TOKENS, load_phoneme_table, row_normalize
from bridge.domain.tokenizer.phoneme_tokenizer import PhonemeTokenizer
from bridge.utils import get_project_root


def phonreps_csv() -> pd.DataFrame:
    """The CSV read independently of the loader under test."""
    path = os.path.join(get_project_root(), "bridge/core/phonreps.csv")
    return pd.read_csv(path).set_index("phone")


def test_row_order_is_the_first_86_csv_rows_then_special_tokens_then_the_rest():
    """The special tokens are written out as literals, with the ids other tests hold.

    Deriving them from the loader's own constant would agree with any placement. A phoneme
    appended to the csv lands after row 90 and moves no id below it.
    """
    csv = phonreps_csv()
    table = load_phoneme_table()

    for index, phoneme in enumerate(csv.index[:86]):
        assert table.row_index[phoneme] == index

    assert [table.row_index[token] for token in SPECIAL_TOKENS] == [86, 87, 88, 89, 90]

    for index, phoneme in enumerate(csv.index[86:]):
        assert table.row_index[phoneme] == 91 + index

    assert table.num_rows == len(csv.index) + len(SPECIAL_TOKENS)
    assert list(table.row_index.values()) == list(range(table.num_rows))


def test_phoneme_rows_reproduce_the_feature_table():
    csv = phonreps_csv()
    table = load_phoneme_table()

    for position, phoneme in enumerate(csv.index):
        index = table.row_index[phoneme]
        expected = torch.tensor(csv.values[position], dtype=torch.float)
        assert torch.equal(table.multihot[index, : table.base_dim], expected), phoneme
        # A real phoneme never carries a special-token feature.
        assert not table.multihot[index, table.base_dim :].any(), phoneme


def test_phonetic_features_are_the_csv_rows_in_csv_order():
    """Oracle: the csv, read independently. ``phonetic_features[i]`` belongs to
    ``phonemes[i]``, and past the special tokens that is not table row ``i``."""
    csv = phonreps_csv()
    table = load_phoneme_table()

    assert table.phonemes == list(csv.index)
    assert torch.equal(table.phonetic_features, torch.tensor(csv.values, dtype=torch.float))
    assert table.row_index[table.phonemes[86]] == 91


def test_no_two_rows_share_a_feature_vector():
    """Two phonemes with one row would be one phoneme to the model, and to every
    closest-phoneme metric. The control is a copy of the table with one row planted on
    another, which the same count must report."""
    table = load_phoneme_table()

    def distinct(matrix: torch.Tensor) -> int:
        return len({tuple(row) for row in matrix.to(torch.uint8).tolist()})

    assert distinct(table.multihot) == table.num_rows

    planted = table.multihot.clone()
    planted[table.row_index["ZH"]] = planted[table.row_index["_LL_"]]
    assert distinct(planted) == table.num_rows - 1


# What each Spanish phoneme is, read off the words of the Spanish lexicon, and the row it
# holds. `_JJ_` and `_LL_` are named against their sounds: `_JJ_` is the "ll" of "calle" and
# `_LL_` is the "y" of "yo". See test_the_two_palatal_symbols_are_told_apart_by_spelling.
SPANISH_PHONEMES = {
    "X": (91, {"velar", "fricative"}),
    "_BB_": (92, {"labial", "fricative", "voice"}),
    "_GG_": (93, {"velar", "fricative", "voice"}),
    "_JJ_": (94, {"palatal", "liquid", "voice"}),
    "_LL_": (95, {"palatal", "fricative", "glide", "voice"}),
    "_NN_": (96, {"palatal", "nasal", "voice"}),
    "_RR_": (97, {"alveolar", "liquid", "rhotic", "tap", "voice"}),
}


def test_the_spanish_phonemes_hold_the_documented_rows_and_features():
    """Oracle: the feature sets written out by hand above."""
    features = [name.lstrip("#") for name in phonreps_csv().columns]
    table = load_phoneme_table()

    for phoneme, (row, expected) in SPANISH_PHONEMES.items():
        assert table.row_index[phoneme] == row, phoneme
        active = {features[column] for column in table.features_of(row).tolist()}
        assert active == expected, phoneme


# Rows whose coding is a decision, not a transcription of a source. The reasons are in
# docs/decisions/0018.
DECIDED_ROWS = {
    "AE1": {"voice", "front", "near-open", "primary"},
    "W": {"labial", "glide", "voice"},
    "WH": {"labial", "glide"},
    "NG": {"velar", "nasal", "voice"},
    "CH": {"palatal", "stop", "fricative", "affricate"},
    "JH": {"palatal", "stop", "fricative", "affricate", "voice"},
}


def test_the_decided_rows_hold_the_documented_features():
    """Oracle: the feature sets written out by hand above."""
    features = [name.lstrip("#") for name in phonreps_csv().columns]
    table = load_phoneme_table()

    for phoneme, expected in DECIDED_ROWS.items():
        active = {
            features[column] for column in table.features_of(table.row_index[phoneme]).tolist()
        }
        assert active == expected, phoneme


# Every pair of rows that differ in one articulatory feature, with that feature. Scoring
# takes the nearest row, so one wrong feature at such a row reads as the other phoneme.
# No word of either lexicon uses `WH`, `NX` or `IX`.
ONE_FEATURE_APART = {
    ("B", "P"): "voice",
    ("CH", "JH"): "voice",
    ("D", "T"): "voice",
    ("DH", "TH"): "voice",
    ("F", "V"): "voice",
    ("G", "K"): "voice",
    ("S", "Z"): "voice",
    ("SH", "ZH"): "voice",
    ("W", "WH"): "voice",
    ("X", "_GG_"): "voice",
    ("IH", "IY"): "tense",
    ("UH", "UW"): "tense",
    ("DH", "V"): "labial",
    ("F", "TH"): "labial",
    ("AO", "OY"): "post_y",
    ("V", "_BB_"): "dental",
    ("Y", "_LL_"): "fricative",
    ("ZH", "_LL_"): "glide",
    ("DX", "NX"): "nasal",
    ("N", "NX"): "tap",
    ("IX", "UW"): "round",
}


def test_the_rows_one_feature_apart_are_the_documented_pairs():
    """A row edit that puts two phonemes side by side, or parts two, fails here by name.

    Stress is set aside: a vowel is read at primary stress with the two stress columns
    dropped, so its three rows count once. Oracle: the pairs written out by hand above.
    """
    frame = phonreps_csv()
    frame.columns = [name.lstrip("#") for name in frame.columns]
    rows = frame[~frame.index.str.endswith(("0", "2"))].drop(columns=["primary", "secondary"])
    rows.index = rows.index.str.removesuffix("1")

    found = {}
    names = list(rows.index)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            differing = rows.columns[(rows.loc[first] != rows.loc[second]).to_numpy()]
            if len(differing) == 1:
                found[(first, second)] = differing[0]

    assert found == ONE_FEATURE_APART


def test_the_two_palatal_symbols_are_told_apart_by_spelling():
    """The evidence for which row is the lateral and which the fricative.

    Every word holding ``_JJ_`` is spelt with "ll", and no word holding ``_LL_`` is: those
    are spelt with "y", or with "hi" before a vowel as in "hielo". A lexicon that swapped
    the two symbols would fail here, and the rows above would then be the wrong way round.
    """
    path = os.path.join(get_project_root(), "bridge/core/pronunciation_lexicons/es.json")
    with open(path, encoding="utf-8") as handle:
        lexicon = json.load(handle)

    with_jj = [word for word, variants in lexicon.items() if "_JJ_" in variants[0]]
    with_ll = [word for word, variants in lexicon.items() if "_LL_" in variants[0]]
    assert with_jj and with_ll
    assert all("ll" in word for word in with_jj)
    assert not any("ll" in word for word in with_ll)
    assert all(re.search(r"y|hi[aeo]", word) for word in with_ll)


def test_special_tokens_occupy_the_documented_columns():
    """The column layout, written out independently of the code that derives it.

    Three places slice against this order: the tokenizer drops the last column to build
    loss targets, the decoder head is sized ``vocab_size - 1`` to match, and
    ``phon_metrics`` slices the trailing specials. Deriving the expectation from
    ``SPECIAL_TOKENS`` would make this test agree with any reordering.
    """
    table = load_phoneme_table()
    for offset, token in enumerate(["[BOS]", "[EOS]", "[UNK]", "[SPC]", "[PAD]"]):
        column = table.base_dim + offset
        assert table.feature_of(token) == column, token
        row = table.multihot[table.row_index[token]]
        assert row.sum() == 1 and row[column] == 1, token
    assert table.feature_of("[PAD]") == table.vocab_size - 1


def test_the_tokenizer_reports_the_tables_special_token_columns():
    """The tokenizer's public ids must be the table's, since the model indexes with both."""
    table = load_phoneme_table()
    assert PhonemeTokenizer().special_token_dims == {
        token: table.feature_of(token) for token in SPECIAL_TOKENS
    }


def test_features_of_inverts_the_multi_hot_encoding():
    """For a real phoneme the feature set is its CSV row; for a special token it is the
    single dedicated column.

    Together they cover every row, which is what makes a `(B, L)` tensor of row ids
    lossless against a ragged `list[list[Tensor]]` of feature ids. See
    docs/decisions/0002.
    """
    csv = phonreps_csv()
    table = load_phoneme_table()

    for phoneme, row in table.row_index.items():
        if phoneme in SPECIAL_TOKENS:
            expected = [table.feature_of(phoneme)]
        else:
            values = torch.tensor(csv.loc[phoneme].values)
            expected = sorted(torch.nonzero(values == 1).flatten().tolist())
        assert sorted(table.features_of(row).tolist()) == expected, phoneme


def test_unknown_phonemes_fall_back_to_unk():
    table = load_phoneme_table()
    assert table.row_of("not-a-phoneme") == table.row_index["[UNK]"]
    assert table.row_of("[PAD]") == table.row_index["[PAD]"]


def test_normalized_rows_average_their_features():
    table = load_phoneme_table()
    counts = table.multihot.sum(-1)
    normalized = row_normalize(table.multihot)

    populated = counts > 0
    assert torch.allclose(normalized[populated].sum(-1), torch.ones(int(populated.sum())))

    # phonreps.csv contains one featureless phoneme ('_'); it must stay zero, not NaN.
    assert not populated.all(), "expected at least one featureless phoneme"
    assert not normalized.isnan().any()
    assert (normalized[~populated] == 0).all()


def test_fingerprint_matches_the_committed_digest():
    """Pinned to a literal, because the digest is written into saved checkpoints.

    Two calls agreeing in one process proves only that the function is deterministic.
    Changing the recipe (the row-name encoding, the truncation length, the byte cast)
    silently invalidates every fingerprint already recorded, so it must break here first.
    """
    assert load_phoneme_table().fingerprint == "7ec53c751f53eb0a"


def test_fingerprint_is_stable_and_order_sensitive():
    assert load_phoneme_table().fingerprint == load_phoneme_table().fingerprint

    table = load_phoneme_table()
    perturbed = table.multihot.clone()
    perturbed[0, 0] = 1 - perturbed[0, 0]
    from bridge.core.phonreps import PhonemeTable

    changed = PhonemeTable(multihot=perturbed, row_index=table.row_index, base_dim=table.base_dim)
    assert changed.fingerprint != table.fingerprint

    reordered_index = {k: v for k, v in reversed(list(table.row_index.items()))}
    reordered = PhonemeTable(
        multihot=table.multihot, row_index=reordered_index, base_dim=table.base_dim
    )
    assert reordered.fingerprint != table.fingerprint


def test_the_cache_key_is_normalized_across_device_spellings():
    """``"cpu"`` and ``torch.device("cpu")`` name one device, so they get one table.

    Without normalization each spelling is a distinct cache key, and the CSV is parsed
    twice: ``tests/vocab.py`` passes the default string while
    ``PhonemeTokenizer.__init__`` passes ``device_manager.device``.
    """
    assert load_phoneme_table() is load_phoneme_table(torch.device("cpu"))
    assert load_phoneme_table("cpu") is load_phoneme_table(torch.device("cpu"))
