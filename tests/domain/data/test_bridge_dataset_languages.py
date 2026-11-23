"""``BridgeDataset`` carries a language for every word of every row, checked at load.

The ``language`` cell is one code for the whole row (``EN``) or one code per word
(``EN EN ES``); a pkl file may hold a list or tuple of codes. Every cell is validated when
the file loads, so a bad one raises there, naming the row, instead of at some later index.

The rows here are repetition: the target is the input, in the input's languages, so every
rule is checked on both sides of the example at once. The ``target_language`` column's own
rules are in ``test_bridge_dataset_pairs.py``.

Oracles: hand-written phoneme rows and token names for the rows whose language decides the
answer, the tokenizer called directly for whole-encoding equality (its own layout is
pinned in ``tests/domain/tokenizer/test_per_word_languages.py``), and the invariant that a row
encodes the same alone, in a slice and after a shuffle.
"""

import pickle

import pytest
import torch

from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import BridgeBatch, DatasetConfig
from tests.conftest import batch_tensors

# (word_raw, language cell). "no" and "me" are in both lexicons with different
# pronunciations, so only the tag can decide their phonemes.
ROWS = [
    ("no me", "ES"),
    ("no me", "EN"),
    ("no", "EN"),
    ("no", "ES"),
    ("the word is gato", "EN EN EN ES"),
    ("no me no", "es en es"),
    ("la con", "ES ES"),
]


def write_csv(path, rows, header="word_raw,language,target_raw,target_language"):
    """Each ``(word, cell)`` as a repetition row: the target and its language cell are the
    input's, cut to however many columns the header names."""
    width = header.count(",") + 1
    lines = [",".join((word, cell, word, cell)[:width]) for word, cell in rows]
    path.write_text(header + "\n" + "".join(line + "\n" for line in lines))
    return str(path)


def write_pkl(path, words, cells):
    columns = {"word_raw": words, "language": cells, "target_raw": words, "target_language": cells}
    with open(path, "wb") as f:
        pickle.dump(columns, f)
    return str(path)


@pytest.fixture
def dataset(tmp_path, load):
    return load(write_csv(tmp_path / "rows.csv", ROWS))


def same_encoding(a, b) -> bool:
    x, y = batch_tensors(a), batch_tensors(b)
    assert len(x) == 10
    return all(x[name].dtype == y[name].dtype and torch.equal(x[name], y[name]) for name in x)


def from_tokenizer(tokenizer, texts, languages) -> BridgeBatch:
    """The batch the tokenizer gives for repetition rows, called directly."""
    return BridgeBatch(
        input=tokenizer.encode_input(texts, languages),
        target=tokenizer.encode_target(texts, languages),
    )


def stored(dataset) -> list[tuple[str, tuple[str, ...]]]:
    """Each row's input and its languages, as stored."""
    return [(example.input, example.input_languages) for example in dataset.examples]


def lang_names(dataset, batch):
    idx_2_char = dataset.tokenizer.char_tokenizer.idx_2_char
    return [idx_2_char[int(i)] for i in batch.input.orth.ids[:, 0]]


# --- what is stored ----------------------------------------------------------


def test_languages_are_stored_as_one_code_per_word(dataset):
    """Oracle: hand-written. A tuple per row, upper case, one entry per word."""
    expected = [
        ("ES", "ES"),
        ("EN", "EN"),
        ("EN",),
        ("ES",),
        ("EN", "EN", "EN", "ES"),
        ("ES", "EN", "ES"),
        ("ES", "ES"),
    ]
    assert [example.input_languages for example in dataset.examples] == expected
    assert [example.target_languages for example in dataset.examples] == expected
    # Hashable and sortable, so rows can be put in a set or sorted.
    assert len(set(dataset.examples)) == len(ROWS)
    assert sorted(dataset.examples)


def test_a_missing_language_column_means_english(tmp_path, load):
    """Oracle: hand-written. No column at all is the one case with a default, and the
    target takes it too."""
    path = tmp_path / "w.csv"
    path.write_text("word_raw,n,target_raw\nno me,1,no me\nla,2,la\n")
    dataset = load(str(path))
    assert [example.input_languages for example in dataset.examples] == [("EN", "EN"), ("EN",)]
    assert [example.target_languages for example in dataset.examples] == [("EN", "EN"), ("EN",)]
    assert lang_names(dataset, dataset[0:2]) == ["EN", "EN"]


# --- rows encode in their own languages ---------------------------------------


def test_a_spanish_phrase_row_gets_spanish_phonemes(dataset):
    """Oracle: hand-written rows, the same ones the tokenizer test pins.

    Through the dataset this time: a row's language has to reach each of its words, which
    a lookup keyed by the whole phrase cannot do.
    """
    assert dataset[0].input.phon.ids.tolist() == [[86, 69, 22, 89, 68, 27, 87]]
    assert dataset[1].input.phon.ids.tolist() == [[86, 69, 33, 89, 68, 32, 87]]
    # The target is the same string in the same language: fed without its [EOS].
    assert dataset[0].target.phon.ids.tolist() == [[86, 69, 22, 89, 68, 27]]
    assert dataset[1].target.phon.ids.tolist() == [[86, 69, 33, 89, 68, 32]]


def test_one_spelling_under_two_languages_keeps_both_in_a_slice(dataset):
    """Oracle: hand-written rows. "no" as EN and "no" as ES, side by side."""
    batch = dataset[2:4]
    assert lang_names(dataset, batch) == ["EN", "ES"]
    assert batch.input.phon.ids.tolist() == [[86, 69, 33, 87], [86, 69, 22, 87]]
    assert batch.target.phon.ids.tolist() == [[86, 69, 33], [86, 69, 22]]


def test_every_row_equals_the_tokenizer_called_with_its_cell(dataset):
    """Oracle: the tokenizer called directly with the cell as written in the file."""
    for i, (word, cell) in enumerate(ROWS):
        assert same_encoding(dataset[i], from_tokenizer(dataset.tokenizer, [word], [cell])), word
    batch = from_tokenizer(dataset.tokenizer, [w for w, _ in ROWS], [c for _, c in ROWS])
    assert same_encoding(dataset[0 : len(ROWS)], batch)


def test_a_row_is_the_same_alone_in_a_slice_and_after_a_shuffle(dataset):
    """Oracle: invariant. A row's phonemes do not depend on its neighbours or its place.

    Phonology is compared, since that is where a lookup shared across a batch lets a
    neighbour's language leak in.
    The pairs are followed through the shuffle by value, word and languages together.
    """
    eos_row = dataset.tokenizer.phoneme_tokenizer.phoneme_table.row_index["[EOS]"]

    def phonemes(batch, row):
        ids = batch.input.phon.ids[row].tolist()
        return ids[: ids.index(eos_row) + 1]

    alone = {key: phonemes(dataset[i], 0) for i, key in enumerate(stored(dataset))}
    assert len(alone) == len(ROWS)

    for seed in (None, 0, 1, 2):
        if seed is not None:
            dataset.shuffle(len(dataset), seed=seed)
        for size in (2, 3, len(dataset)):
            for start in range(0, len(dataset), size):
                batch = dataset[start : start + size]
                for row in range(len(batch)):
                    key = stored(dataset)[start + row]
                    assert phonemes(batch, row) == alone[key], (key, seed, size)


def test_a_shuffle_moves_each_rows_languages_with_its_word(dataset):
    """Oracle: invariant. The (word, languages) pairs are the same set before and after.

    The rows hold one spelling under two languages, so a shuffle that moved the words and
    left the languages in place would pair "no me" with the wrong tuple.
    """
    before = sorted(stored(dataset))
    order_before = stored(dataset)
    for seed in (0, 1, 2):
        dataset.shuffle(len(dataset), seed=seed)
        assert sorted(stored(dataset)) == before
    assert stored(dataset) != order_before, "the shuffles must have moved something"


# --- a bad cell raises at load, naming the row --------------------------------


@pytest.mark.parametrize(
    ("cell", "message"),
    [
        ("ES EN", r"Row 2 \('no me no', language 'ES EN'\): 2 language codes for 3 words"),
        ("ES EN ES EN", r"Row 2 .*4 language codes for 3 words"),
        ("FR", r"Row 2 \('no me no', language 'FR'\): Invalid languages: \['FR'\]\. Supported"),
        ("ES FR ES", r"Row 2 .*Invalid languages: \['FR'\]"),
        ("-- EN ES", r"Row 2 .*whole row is unspecified"),
        ("", r"Row 2 \('no me no', language ''\): No language code given"),
        ("   ", r"Row 2 .*No language code given"),
        # Unspecified is for a direct tokenizer call. A file never holds it.
        ("--", r"Row 2 \('no me no', language '--'\): a dataset row names the language"),
        ("-- -- --", r"Row 2 .*a dataset row names the language"),
    ],
)
def test_a_bad_language_cell_raises_at_load_naming_the_row(tmp_path, load, cell, message):
    """Oracle: the rule. The file never loads, so no later index can hit the bad row."""
    rows = [("no", "ES"), ("me", "EN"), ("no me no", cell), ("la", "ES")]
    with pytest.raises(ValueError, match=message):
        load(write_csv(tmp_path / "bad.csv", rows))


def test_a_bad_first_row_raises_like_any_other(tmp_path, load):
    """Oracle: the rule. Row 0 is checked like every other row, and is not left out in
    place of raising."""
    with pytest.raises(ValueError, match=r"Row 0 .*Invalid languages: \['FR'\]"):
        load(write_csv(tmp_path / "bad.csv", [("no", "FR"), ("me", "EN")]))


@pytest.mark.parametrize(
    "cell", [None, float("nan"), 7, ["ES", None], [["ES"]], {"ES"}, [None], ("--",)]
)
def test_a_missing_or_malformed_pkl_cell_raises_naming_the_row(tmp_path, load, cell):
    """Oracle: the rule. A cell is a string of codes or a list or tuple of them."""
    path = write_pkl(tmp_path / "bad.pkl", ["no", "me"], ["ES", cell])
    with pytest.raises(ValueError, match=r"Row 1 \('me', language "):
        load(path)


def test_a_comma_between_codes_raises_instead_of_loading_the_codes_as_words(tmp_path, load):
    """Oracle: the rule. ``no me,ES,EN`` is three fields under a two-field header.

    pandas reads such a file without complaint when every row has the extra field: the
    first field becomes an index, the words become ``ES`` and the languages ``EN``.
    """
    path = tmp_path / "comma.csv"
    path.write_text("word_raw,language\nno me,ES,EN\nla con,ES,ES\n")
    with pytest.raises(ValueError, match="more fields than the header names"):
        load(str(path))


def test_a_client_frame_keeps_its_rows_whatever_its_index(shared_tokenizer):
    """Oracle: the frame. A ``gs://`` client may hand back a filtered or sorted frame.

    The comma check belongs to the local csv read, where pandas builds the index. A frame
    a client filtered has a gapped index and no extra field, and loads row for row.
    """
    import pandas as pd

    words = ["no", "skip", "me"]
    frame = pd.DataFrame({"word_raw": words, "language": ["ES", "EN", "EN"], "target_raw": words})

    class Client:
        def read_csv(self, bucket_name, blob_name, **kwargs):
            return frame[frame["word_raw"] != "skip"]

    dataset = BridgeDataset(
        DatasetConfig(dataset_filepath="gs://bucket/rows.csv"),
        gcs_client=Client(),
        tokenizer=shared_tokenizer,
    )
    assert stored(dataset) == [("no", ("ES",)), ("me", ("EN",))]


def test_a_word_with_no_pronunciation_in_its_language_is_found_when_the_row_is_read(
    tmp_path, load, shared_tokenizer
):
    """Oracle: the lexicon. "the" tagged ES has no pronunciation, like any unknown word.

    Language cells are checked at load and pronunciations are not, so the row loads
    wherever it sits, first included, and raises when it is read, alone or in a slice,
    naming the row. Rows that do not hold it read as usual.
    """
    assert "es" not in shared_tokenizer.phoneme_tokenizer.pronunciation_dict["the"]

    for name, rows, bad in (
        ("first", [("the", "ES"), ("no", "ES")], 0),
        ("later", [("no", "ES"), ("the", "ES")], 1),
    ):
        dataset = load(write_csv(tmp_path / f"{name}.csv", rows))
        assert [word for word, _ in stored(dataset)] == [word for word, _ in rows]
        # The file is repetition, so the row's target holds the word too.
        message = (
            f"No pronunciation for a word of row {bad}'s input 'the', row {bad}'s target 'the'$"
        )
        with pytest.raises(RuntimeError, match=message):
            dataset[bad]
        with pytest.raises(RuntimeError, match=message):
            dataset[0:2]
        assert len(dataset[1 - bad]) == 1


def test_the_load_error_is_the_tokenizers_own_message(tmp_path, load, shared_tokenizer):
    """Oracle: agreement. An unsupported code reads the same at load as on a direct call."""
    with pytest.raises(ValueError) as direct:
        shared_tokenizer.encode_input(["no"], ["FR"])
    with pytest.raises(ValueError) as at_load:
        load(write_csv(tmp_path / "bad.csv", [("me", "EN"), ("no", "FR")]))
    assert str(at_load.value) == f"Row 1 ('no', language 'FR'): {direct.value}"


# --- pkl cells ------------------------------------------------------------------


def test_a_pkl_file_may_hold_lists_or_tuples_of_codes(tmp_path, load, dataset):
    """Oracle: differential against the same rows written as CSV strings."""
    words = [word for word, _ in ROWS]
    cells = [
        ["ES"],
        ("EN", "EN"),
        "EN",
        ("ES",),
        ["EN", "EN", "EN", "ES"],
        ("es", "en", "es"),
        "ES ES",
    ]
    from_pkl = load(write_pkl(tmp_path / "rows.pkl", words, cells))

    assert from_pkl.examples == dataset.examples
    for i in range(len(ROWS)):
        assert same_encoding(from_pkl[i], dataset[i])
    assert same_encoding(from_pkl[0 : len(ROWS)], dataset[0 : len(ROWS)])
