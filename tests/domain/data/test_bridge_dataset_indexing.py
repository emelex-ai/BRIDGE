"""Pins ``BridgeDataset`` indexing, memoization and shuffling.

The ``int`` indexer shares the ``slice`` branch inside ``__getitem__``, and single-row
encodings are memoized in an instance dict. The dataset is the training loop's only
entry point, so each of those behaviours is asserted directly. Rows are indexed by
position only. What a ``language`` cell may hold, what is stored for it, how it follows
its row through a slice and a shuffle, and what happens to a bad one, is in
``test_bridge_dataset_languages.py``.
"""

import pytest
import torch

from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import BridgeEncoding, DatasetConfig

DATA = "tests/domain/model/data/data.csv"


@pytest.fixture(scope="module")
def dataset():
    return BridgeDataset(DatasetConfig(dataset_filepath=DATA))


@pytest.fixture
def fresh():
    return BridgeDataset(DatasetConfig(dataset_filepath=DATA))


# --- indexing forms --------------------------------------------------------


def test_int_index_returns_a_single_row_encoding(dataset):
    enc = dataset[0]
    assert isinstance(enc, BridgeEncoding)
    assert len(enc) == 1


def test_int_index_agrees_with_the_equivalent_slice(dataset):
    """``int`` is normalised to ``slice(i, i+1)``; both must encode identically."""
    by_int = dataset[3]
    by_slice = dataset[3:4]
    assert torch.equal(by_int.orthographic.enc_input_ids, by_slice.orthographic.enc_input_ids)
    assert torch.equal(by_int.orthographic.enc_pad_mask, by_slice.orthographic.enc_pad_mask)


def test_slice_returns_one_row_per_word(dataset):
    assert len(dataset[0:4]) == 4


def test_strided_slice_is_supported(dataset):
    assert len(dataset[0:8:2]) == 4


def test_len_matches_the_word_list(dataset):
    assert len(dataset) == len(dataset.words)


# --- index errors ----------------------------------------------------------


def test_negative_int_is_rejected_before_slice_conversion(dataset):
    """The bounds check must run before ``int`` becomes ``slice``; otherwise a
    negative index silently wraps to the end of the corpus."""
    with pytest.raises(IndexError, match=r"Index -1 out of range \[0, \d+\)"):
        dataset[-1]


def test_out_of_range_int_is_rejected(dataset):
    with pytest.raises(IndexError, match="out of range"):
        dataset[len(dataset)]


@pytest.mark.parametrize("index", [1.5, "long", ["long", "pencil"], [1, 2], None])
def test_only_an_int_or_a_slice_indexes_a_row(dataset, index):
    """A word is not an index, even one the dataset holds: "long" is row 0 here."""
    assert dataset.words[0] == "long"
    with pytest.raises(TypeError, match="Invalid index type"):
        dataset[index]


def test_a_row_is_reached_through_its_index_only(dataset):
    """A lookup taking a word and a language would have to choose between rows that share
    a spelling, so the dataset has none."""
    assert not hasattr(dataset, "get_encoding")


def test_batch_encode_failure_message(dataset, monkeypatch):
    """One shared message covers both the single-row and batch paths."""
    monkeypatch.setattr(dataset.tokenizer, "encode", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="Failed to encode word\\(s\\): "):
        dataset[0:3]


# --- memoization -----------------------------------------------------------


def test_cache_is_initialised_before_row_zero_is_validated(fresh):
    """``__init__`` encode-validates the first row, so it must create the cache dict
    *before* calling ``_process_raw_dataframe``. Reordering those two statements is an
    AttributeError waiting to happen, and only row 0 exercises it."""
    assert isinstance(fresh._encoding_cache, dict)
    assert len(fresh._encoding_cache) == 1, "the first row is encoded during __init__"


def test_single_word_encodings_are_memoized(fresh):
    calls = []
    original = fresh.tokenizer.encode

    def counting(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("text"))
        return original(*args, **kwargs)

    fresh.tokenizer.encode = counting
    index = 5  # not the row __init__ already warmed
    first = fresh[index]
    second = fresh[index]
    assert len(calls) == 1, "second lookup should hit the cache"
    assert first is second


def test_cache_is_per_instance(fresh):
    """The cache is instance state. A class-level ``@lru_cache`` keyed on ``self`` would
    pin every dataset, and its lexicon, for the process lifetime."""
    other = BridgeDataset(DatasetConfig(dataset_filepath=DATA))
    fresh[5]
    new_keys = set(fresh._encoding_cache) - set(other._encoding_cache)
    assert new_keys, "indexing fresh should have added a cache entry"
    assert not (new_keys & set(other._encoding_cache)), "caches must not be shared"


def test_cache_is_bounded(fresh):
    for i in range(min(len(fresh), fresh._ENCODING_CACHE_SIZE + 20)):
        fresh[i]
    assert len(fresh._encoding_cache) <= fresh._ENCODING_CACHE_SIZE


def test_cache_key_includes_the_language(tmp_path):
    """One spelling under two languages is two cache entries and two encodings.

    "no" is N OW1 in English and N AO1 in Spanish. A cache keyed by the word alone would
    hand the second row the first row's encoding.
    """
    path = tmp_path / "homographs.csv"
    path.write_text("word_raw,language\nno,EN\nno,ES\nno me,ES EN\nno me,ES ES\n")
    homographs = BridgeDataset(DatasetConfig(dataset_filepath=str(path)))

    english, spanish = homographs[0], homographs[1]

    assert {("no", ("EN",)), ("no", ("ES",))} <= set(homographs._encoding_cache)
    assert english.phonological.enc_input_ids.tolist() == [[86, 69, 33, 87]]
    assert spanish.phonological.enc_input_ids.tolist() == [[86, 69, 22, 87]]
    # Every word's language is in the key, not the first word's alone: these two rows
    # differ only in the second word, "me", M IY1 in English and M EH1 in Spanish.
    assert homographs[2].phonological.enc_input_ids.tolist() == [[86, 69, 22, 89, 68, 32, 87]]
    assert homographs[3].phonological.enc_input_ids.tolist() == [[86, 69, 22, 89, 68, 27, 87]]


# --- shuffle ---------------------------------------------------------------


def test_shuffle_is_a_permutation(fresh):
    before = list(fresh.words)
    fresh.shuffle(len(fresh))
    assert len(fresh.words) == len(before)
    assert sorted(fresh.words) == sorted(before)


def test_shuffle_leaves_the_tail_untouched(fresh):
    cutoff = 5
    tail_before = list(fresh.words[cutoff:])
    fresh.shuffle(cutoff)
    assert fresh.words[cutoff:] == tail_before


def test_shuffle_rejects_a_cutoff_past_the_end(fresh):
    with pytest.raises(ValueError, match="exceeds dataset size"):
        fresh.shuffle(len(fresh) + 1)


def test_indexing_still_works_after_shuffle(fresh):
    fresh.shuffle(len(fresh))
    assert len(fresh[0]) == 1
    assert len(fresh[0:3]) == 3
