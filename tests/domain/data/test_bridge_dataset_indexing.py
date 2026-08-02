"""Pins ``BridgeDataset`` indexing, language resolution, memoization and shuffling.

Three collapses happened here: the ``int`` indexer now shares the ``slice`` branch,
``_get_encoding`` / ``_get_encoding_batch`` were inlined into ``_get_encoding_unified``,
and the class-level ``@lru_cache`` became an instance dict. The dataset is the
training loop's only entry point, so each collapsed behaviour is asserted directly.
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


def test_string_index_looks_up_by_word(dataset):
    word = dataset.words[0]
    assert len(dataset[word]) == 1


def test_list_of_strings_indexes_a_batch(dataset):
    words = dataset.words[:3]
    assert len(dataset[words]) == len(words)


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


def test_unknown_word_raises_key_error(dataset):
    with pytest.raises(KeyError, match="not found in dataset"):
        dataset["zzzznotarealword"]


def test_non_string_list_entries_are_rejected(dataset):
    with pytest.raises(TypeError, match="List indices must be strings"):
        dataset[[1, 2]]


def test_unsupported_index_type_is_rejected(dataset):
    with pytest.raises(TypeError, match="Invalid index type"):
        dataset[1.5]


def test_batch_encode_failure_message(dataset, monkeypatch):
    """One shared message now covers both the single-word and batch paths.

    Before the dispatch layers were inlined, the batch path raised
    "Batch encoding failed for words: ..." from ``_get_encoding_batch``. That helper
    is gone; both paths now report through this guard.
    """
    monkeypatch.setattr(dataset.tokenizer, "encode", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="Failed to encode word\\(s\\): "):
        dataset[0:3]


# --- language resolution ---------------------------------------------------


def test_positional_index_uses_the_stored_language(dataset):
    """Positional indexers read the language off the parallel list, not the lookup map."""
    enc = dataset[0]
    lang_token = dataset.tokenizer.char_tokenizer.char_2_idx[dataset.languages[0].upper()]
    assert enc.orthographic.enc_input_ids[0, 0].item() == lang_token


def test_slice_assigns_each_word_its_own_language(dataset):
    enc = dataset[0:4]
    c2i = dataset.tokenizer.char_tokenizer.char_2_idx
    for row, lang in enumerate(dataset.languages[0:4]):
        assert enc.orthographic.enc_input_ids[row, 0].item() == c2i[lang.upper()]


def test_get_encoding_accepts_an_explicit_language_override(dataset):
    word = dataset.words[0]
    enc = dataset.get_encoding(word, language_map={word.lower(): "ES"})
    es = dataset.tokenizer.char_tokenizer.char_2_idx["ES"]
    assert enc.orthographic.enc_input_ids[0, 0].item() == es


def test_word_languages_index_covers_every_word(dataset):
    assert set(dataset._word_languages) == set(dataset.words)
    for langs in dataset._word_languages.values():
        assert langs
        assert langs <= set(dataset.languages)


def test_word_languages_matches_a_brute_force_scan(dataset):
    """The prebuilt index replaced an O(n) scan per word; it must agree with it."""
    for word in dataset.words[:25]:
        expected = {
            lang for w, lang in zip(dataset.words, dataset.languages, strict=False) if w == word
        }
        assert dataset._word_languages[word] == expected


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
    """The old ``@lru_cache`` lived on the class and keyed on ``self``, pinning every
    dataset (and its lexicon) for the process lifetime. The cache is now instance state."""
    other = BridgeDataset(DatasetConfig(dataset_filepath=DATA))
    fresh[5]
    new_keys = set(fresh._encoding_cache) - set(other._encoding_cache)
    assert new_keys, "indexing fresh should have added a cache entry"
    assert not (new_keys & set(other._encoding_cache)), "caches must not be shared"


def test_cache_is_bounded(fresh):
    for i in range(min(len(fresh), fresh._ENCODING_CACHE_SIZE + 20)):
        fresh[i]
    assert len(fresh._encoding_cache) <= fresh._ENCODING_CACHE_SIZE


def test_cache_key_includes_the_language(fresh):
    word = fresh.words[0]
    fresh.get_encoding(word, language_map={word.lower(): "EN"})
    fresh.get_encoding(word, language_map={word.lower(): "ES"})
    langs = {key[1] for key in fresh._encoding_cache}
    assert {"EN", "ES"} <= langs


# --- shuffle ---------------------------------------------------------------


def test_shuffle_is_a_permutation(fresh):
    """The two removed trailing asserts were the only statement of this invariant."""
    before = list(fresh.words)
    fresh.shuffle(len(fresh))
    assert len(fresh.words) == len(before)
    assert sorted(fresh.words) == sorted(before)


def test_shuffle_keeps_words_and_languages_aligned(fresh):
    pairs_before = sorted(zip(fresh.words, fresh.languages, strict=True))
    fresh.shuffle(len(fresh))
    assert sorted(zip(fresh.words, fresh.languages, strict=True)) == pairs_before


def test_shuffle_leaves_the_tail_untouched(fresh):
    cutoff = 5
    tail_before = list(fresh.words[cutoff:])
    fresh.shuffle(cutoff)
    assert fresh.words[cutoff:] == tail_before


def test_shuffle_rejects_a_cutoff_past_the_end(fresh):
    with pytest.raises(ValueError, match="exceeds dataset size"):
        fresh.shuffle(len(fresh) + 1)


def test_word_languages_survives_shuffle(fresh):
    """``_word_languages`` is built once in ``__init__`` on the assumption that
    ``shuffle`` only permutes. If that ever stops holding, this is the tripwire."""
    snapshot = {w: set(langs) for w, langs in fresh._word_languages.items()}
    fresh.shuffle(len(fresh))
    rebuilt: dict[str, set[str]] = {}
    for word, lang in zip(fresh.words, fresh.languages, strict=False):
        rebuilt.setdefault(word, set()).add(lang)
    assert snapshot == rebuilt


def test_indexing_still_works_after_shuffle(fresh):
    fresh.shuffle(len(fresh))
    assert len(fresh[0]) == 1
    assert len(fresh[0:3]) == 3
