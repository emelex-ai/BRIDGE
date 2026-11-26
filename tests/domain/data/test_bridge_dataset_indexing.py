"""Pins ``BridgeDataset`` indexing and shuffling.

The ``int`` indexer shares the ``slice`` branch inside ``__getitem__``. The dataset is the
training loop's only entry point, so each of those behaviours is asserted directly. Rows
are indexed by position only. What a language cell may hold, what is stored for it, how
it follows its row through a slice and a shuffle, and what happens to a bad one, is in
``test_bridge_dataset_languages.py``. What the target columns add is in
``test_bridge_dataset_pairs.py``.
"""

import pytest
import torch

from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import BridgeBatch, DatasetConfig

DATA = "tests/domain/model/data/data.csv"


@pytest.fixture(scope="module")
def dataset():
    return BridgeDataset(DatasetConfig(dataset_filepath=DATA))


@pytest.fixture
def fresh():
    return BridgeDataset(DatasetConfig(dataset_filepath=DATA))


def test_int_index_returns_a_batch_of_one(dataset):
    batch = dataset[0]
    assert isinstance(batch, BridgeBatch)
    assert len(batch) == 1


def test_int_index_agrees_with_the_equivalent_slice(dataset):
    """``int`` is normalised to ``slice(i, i+1)``; both must encode identically."""
    by_int = dataset[3]
    by_slice = dataset[3:4]
    assert torch.equal(by_int.input.orth.ids, by_slice.input.orth.ids)
    assert torch.equal(by_int.input.orth.pad_mask, by_slice.input.orth.pad_mask)
    assert torch.equal(by_int.target.phon.labels, by_slice.target.phon.labels)


def test_slice_returns_one_row_per_word(dataset):
    assert len(dataset[0:4]) == 4


def test_strided_slice_is_supported(dataset):
    assert len(dataset[0:8:2]) == 4


def test_len_matches_the_examples(dataset):
    assert len(dataset) == len(dataset.examples) == 7300


def test_negative_int_is_rejected_before_slice_conversion(dataset):
    """The bounds check must run before ``int`` becomes ``slice``; otherwise a
    negative index silently wraps to the end of the corpus."""
    with pytest.raises(IndexError, match=r"Index -1 out of range \[0, \d+\)"):
        dataset[-1]


def test_out_of_range_int_is_rejected(dataset):
    with pytest.raises(IndexError, match="out of range"):
        dataset[len(dataset)]


@pytest.mark.parametrize("empty", [slice(0, 0), slice(5, 5), slice(4, 2), slice(7300, 7400)])
def test_a_slice_that_selects_no_rows_is_refused_by_name(dataset, empty):
    """Oracle: the rule. There is no batch of zero rows, so the read says so itself and
    does not leave it to whatever the tokenizer does with an empty list."""
    with pytest.raises(IndexError, match=r"selects no rows of a dataset of 7300"):
        dataset[empty]
    assert len(dataset[7298:7400]) == 2, "a slice that runs past the end keeps its rows"


@pytest.mark.parametrize("index", [1.5, "long", ["long", "pencil"], [1, 2], None])
def test_only_an_int_or_a_slice_indexes_a_row(dataset, index):
    """A word is not an index, even one the dataset holds: "long" is row 0 here."""
    assert dataset.examples[0].input == "long"
    with pytest.raises(TypeError, match="Invalid index type"):
        dataset[index]


def test_a_row_is_reached_through_its_index_only(dataset):
    """A lookup taking a word and a language would have to choose between rows that share
    a spelling, so the dataset has none."""
    assert not hasattr(dataset, "get_encoding")


def test_nothing_is_kept_between_reads(tmp_path):
    """Oracle: hand-written rows. One spelling under two languages is two rows and two
    encodings, read fresh each time.

    "no" is N OW1 in English and N AO1 in Spanish. Anything that remembered a row by its
    spelling would hand the second row the first row's encoding.
    """
    path = tmp_path / "homographs.csv"
    path.write_text(
        "word_raw,language,target_raw\nno,EN,no\nno,ES,no\nno me,ES EN,la\nno me,ES ES,la\n"
    )
    with pytest.raises(ValueError, match="Add a 'target_language' column"):
        BridgeDataset(DatasetConfig(dataset_filepath=str(path)))

    path.write_text(
        "word_raw,language,target_raw,target_language\n"
        "no,EN,no,EN\nno,ES,no,ES\nno me,ES EN,la,ES\nno me,ES ES,la,ES\n"
    )
    homographs = BridgeDataset(DatasetConfig(dataset_filepath=str(path)))

    for _ in range(2):
        assert homographs[0].input.phon.ids.tolist() == [[86, 69, 33, 87]]
        assert homographs[1].input.phon.ids.tolist() == [[86, 69, 22, 87]]
        # Every word's language counts, not the first word's alone: these two rows differ
        # only in the second word, "me", M IY1 in English and M EH1 in Spanish.
        assert homographs[2].input.phon.ids.tolist() == [[86, 69, 22, 89, 68, 32, 87]]
        assert homographs[3].input.phon.ids.tolist() == [[86, 69, 22, 89, 68, 27, 87]]
    assert homographs[0] is not homographs[0]


def test_shuffle_is_a_permutation(fresh):
    before = list(fresh.examples)
    fresh.shuffle(len(fresh), seed=0)
    assert fresh.examples != before, "the shuffle must have moved something"
    assert len(fresh.examples) == len(before)
    assert sorted(fresh.examples) == sorted(before)


def test_shuffle_leaves_the_tail_untouched(fresh):
    cutoff = 5
    before = list(fresh.examples)
    fresh.shuffle(cutoff, seed=0)
    assert fresh.examples[cutoff:] == before[cutoff:]
    assert fresh.examples[:cutoff] != before[:cutoff], "the head must have moved"
    assert sorted(fresh.examples[:cutoff]) == sorted(before[:cutoff])


def test_a_seeded_shuffle_is_reproducible_and_touches_no_global_state(fresh):
    """Oracle: the standard library. The order is ``random.Random(seed).shuffle`` applied
    to the rows, and the global generator is left where it was."""
    import random

    expected = list(fresh.examples)
    random.Random(7).shuffle(expected)
    random.seed(123)
    state = random.getstate()

    fresh.shuffle(len(fresh), seed=7)

    assert fresh.examples == expected
    assert random.getstate() == state


def test_shuffle_rejects_a_cutoff_past_the_end(fresh):
    with pytest.raises(ValueError, match="exceeds dataset size"):
        fresh.shuffle(len(fresh) + 1)


def test_indexing_still_works_after_shuffle(fresh):
    fresh.shuffle(len(fresh))
    assert len(fresh[0]) == 1
    assert len(fresh[0:3]) == 3
