"""Pins the phoneme -> feature table contract (issue #221).

The table is what lets the tokenizer and the model agree on phoneme identity without
holding references to each other, so its row order and its inverse mapping back to
feature sets are load-bearing.
"""

import os

import pandas as pd
import torch

from bridge.core.phonreps import SPECIAL_TOKENS, load_phoneme_table, row_normalize
from bridge.domain.tokenizer.phoneme_tokenizer import PhonemeTokenizer
from bridge.utils import get_project_root


def phonreps_csv() -> pd.DataFrame:
    """The CSV read independently of the loader under test."""
    path = os.path.join(get_project_root(), "bridge/core/phonreps.csv")
    return pd.read_csv(path).set_index("phone")


def test_row_order_is_phonreps_order_then_special_tokens():
    """The contract: CSV order first, specials appended in SPECIAL_TOKENS order."""
    csv = phonreps_csv()
    table = load_phoneme_table()

    for index, phoneme in enumerate(csv.index):
        assert table.row_index[phoneme] == index

    for offset, token in enumerate(SPECIAL_TOKENS):
        assert table.row_index[token] == len(csv.index) + offset

    assert table.num_rows == len(csv.index) + len(SPECIAL_TOKENS)


def test_phoneme_rows_reproduce_the_feature_table():
    csv = phonreps_csv()
    table = load_phoneme_table()

    for index, phoneme in enumerate(csv.index):
        expected = torch.tensor(csv.values[index], dtype=torch.float)
        assert torch.equal(table.multihot[index, : table.base_dim], expected), phoneme
        # A real phoneme never carries a special-token feature.
        assert not table.multihot[index, table.base_dim :].any(), phoneme


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
    """A row id carries exactly the feature set the ragged representation used to carry.

    For real phonemes that set is the CSV row; for special tokens it is the single
    dedicated column. Together they cover every row, which is what makes a `(B, L)`
    tensor of row ids lossless against the `list[list[Tensor]]` it replaced.
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
    assert load_phoneme_table().fingerprint == "4c4a9c388f46bc35"


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
    twice: ``tests/vocab.py`` passes the default string while ``Model.__init__`` and
    ``PhonemeTokenizer.__init__`` pass ``device_manager.device``.
    """
    assert load_phoneme_table() is load_phoneme_table(torch.device("cpu"))
    assert load_phoneme_table("cpu") is load_phoneme_table(torch.device("cpu"))
