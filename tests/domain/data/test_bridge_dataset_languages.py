"""``BridgeDataset`` carries a language for every word of every row, checked at load.

The ``language`` cell is one code for the whole row (``EN``) or one code per word
(``EN EN ES``); a pkl file may hold a list or tuple of codes. Every cell is validated when
the file loads, so a bad one raises there, naming the row, instead of at some later index.

Oracles: hand-written phoneme rows and token names for the rows whose language decides the
answer, the tokenizer called directly for whole-encoding equality (its own layout is
pinned in ``tests/domain/tokenizer/test_per_word_languages.py``), and the invariant that a row
encodes the same alone, in a slice and after a shuffle.
"""

import pickle

import pytest
import torch

from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import DatasetConfig

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

FIELDS = ("enc_input_ids", "enc_pad_mask", "dec_input_ids", "dec_pad_mask", "targets")


def write_csv(path, rows, header="word_raw,language"):
    path.write_text(header + "\n" + "".join(f"{word},{cell}\n" for word, cell in rows))
    return str(path)


def write_pkl(path, words, cells):
    with open(path, "wb") as f:
        pickle.dump({"word_raw": words, "language": cells}, f)
    return str(path)


@pytest.fixture
def load(shared_tokenizer):
    def build(path: str) -> BridgeDataset:
        return BridgeDataset(DatasetConfig(dataset_filepath=path), tokenizer=shared_tokenizer)

    return build


@pytest.fixture
def dataset(tmp_path, load):
    return load(write_csv(tmp_path / "rows.csv", ROWS))


def same_encoding(a, b) -> bool:
    for side in ("orthographic", "phonological"):
        for field in FIELDS:
            x, y = getattr(getattr(a, side), field), getattr(getattr(b, side), field)
            if (x is None) != (y is None):
                return False
            if x is not None and not (x.dtype == y.dtype and torch.equal(x, y)):
                return False
    return True


def lang_names(dataset, encoding):
    idx_2_char = dataset.tokenizer.char_tokenizer.idx_2_char
    return [idx_2_char[int(i)] for i in encoding.orthographic.enc_input_ids[:, 0]]


# --- what is stored ----------------------------------------------------------


def test_languages_are_stored_as_one_code_per_word(dataset):
    """Oracle: hand-written. A tuple per row, upper case, one entry per word."""
    assert dataset.languages == [
        ("ES", "ES"),
        ("EN", "EN"),
        ("EN",),
        ("ES",),
        ("EN", "EN", "EN", "ES"),
        ("ES", "EN", "ES"),
        ("ES", "ES"),
    ]
    # Hashable and sortable, so rows can be put in a set or sorted beside their words.
    assert len(set(zip(dataset.words, dataset.languages, strict=True))) == len(ROWS)
    assert sorted(zip(dataset.words, dataset.languages, strict=True))


def test_a_missing_language_column_means_english(tmp_path, load):
    """Oracle: hand-written. No column at all is the one case with a default."""
    dataset = load(write_csv(tmp_path / "w.csv", [("no me", ""), ("la", "")], header="word_raw,n"))
    assert dataset.languages == [("EN", "EN"), ("EN",)]
    assert lang_names(dataset, dataset[0:2]) == ["EN", "EN"]


# --- rows encode in their own languages ---------------------------------------


def test_a_spanish_phrase_row_gets_spanish_phonemes(dataset):
    """Oracle: hand-written rows, the same ones the tokenizer test pins.

    Through the dataset this time: a row's language has to reach each of its words, which
    a lookup keyed by the whole phrase cannot do.
    """
    assert dataset[0].phonological.enc_input_ids.tolist() == [[86, 69, 22, 89, 68, 27, 87]]
    assert dataset[1].phonological.enc_input_ids.tolist() == [[86, 69, 33, 89, 68, 32, 87]]


def test_one_spelling_under_two_languages_keeps_both_in_a_slice(dataset):
    """Oracle: hand-written rows. "no" as EN and "no" as ES, side by side."""
    batch = dataset[2:4]
    assert lang_names(dataset, batch) == ["EN", "ES"]
    assert batch.phonological.enc_input_ids.tolist() == [[86, 69, 33, 87], [86, 69, 22, 87]]


def test_every_row_equals_the_tokenizer_called_with_its_cell(dataset):
    """Oracle: the tokenizer called directly with the cell as written in the file."""
    for i, (word, cell) in enumerate(ROWS):
        assert same_encoding(dataset[i], dataset.tokenizer.encode(word, languages=cell)), word
    batch = dataset.tokenizer.encode([w for w, _ in ROWS], languages=[c for _, c in ROWS])
    assert same_encoding(dataset[0 : len(ROWS)], batch)


def test_a_row_is_the_same_alone_in_a_slice_and_after_a_shuffle(dataset):
    """Oracle: invariant. A row's phonemes do not depend on its neighbours or its place.

    Phonology is compared, since that is where a lookup shared across a batch lets a
    neighbour's language leak in.
    The pairs are followed through the shuffle by value, word and languages together.
    """
    eos_row = dataset.tokenizer.phoneme_tokenizer.phoneme_table.row_index["[EOS]"]

    def phonemes(encoding, row):
        ids = encoding.phonological.enc_input_ids[row].tolist()
        return ids[: ids.index(eos_row) + 1]

    alone = {
        (dataset.words[i], dataset.languages[i]): phonemes(dataset[i], 0)
        for i in range(len(dataset))
    }
    assert len(alone) == len(ROWS)

    for seed in (None, 0, 1, 2):
        if seed is not None:
            dataset.shuffle(len(dataset), seed=seed)
        for size in (2, 3, len(dataset)):
            for start in range(0, len(dataset), size):
                batch = dataset[start : start + size]
                for row in range(len(batch)):
                    key = (dataset.words[start + row], dataset.languages[start + row])
                    assert phonemes(batch, row) == alone[key], (key, seed, size)


def test_a_shuffle_moves_each_rows_languages_with_its_word(dataset):
    """Oracle: invariant. The (word, languages) pairs are the same set before and after.

    The rows hold one spelling under two languages, so a shuffle that moved the words and
    left the languages in place would pair "no me" with the wrong tuple.
    """
    before = sorted(zip(dataset.words, dataset.languages, strict=True))
    order_before = list(dataset.words)
    for seed in (0, 1, 2):
        dataset.shuffle(len(dataset), seed=seed)
        assert sorted(zip(dataset.words, dataset.languages, strict=True)) == before
    assert dataset.words != order_before, "the shuffles must have moved something"


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
    """Oracle: the rule. Row 0 is the row the load also encodes, and it must not be
    dropped in place of raising."""
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

    frame = pd.DataFrame({"word_raw": ["no", "skip", "me"], "language": ["ES", "EN", "EN"]})

    class Client:
        def read_csv(self, bucket_name, blob_name, **kwargs):
            return frame[frame["word_raw"] != "skip"]

    dataset = BridgeDataset(
        DatasetConfig(dataset_filepath="gs://bucket/rows.csv"),
        gcs_client=Client(),
        tokenizer=shared_tokenizer,
    )
    assert dataset.words == ["no", "me"]
    assert dataset.languages == [("ES",), ("EN",)]


def test_a_word_with_no_pronunciation_in_its_language_is_found_when_the_row_is_read(
    tmp_path, load, shared_tokenizer, caplog
):
    """Oracle: the lexicon. "the" tagged ES has no pronunciation, like any unknown word.

    Language cells are checked at load; pronunciations are not, except for the first row,
    which the load encodes once. So the row is left out with a warning when it comes
    first, and anywhere else it loads and raises when it is read.
    """
    assert "es" not in shared_tokenizer.phoneme_tokenizer.pronunciation_dict["the"]

    with caplog.at_level("WARNING"):
        first = load(write_csv(tmp_path / "first.csv", [("the", "ES"), ("no", "ES")]))
    assert (first.words, first.languages) == (["no"], [("ES",)])
    assert "Initial encoding validation failed for the" in caplog.text

    later = load(write_csv(tmp_path / "later.csv", [("no", "ES"), ("the", "ES")]))
    assert later.words == ["no", "the"]
    with pytest.raises(RuntimeError, match=r"Failed to encode word\(s\): the"):
        later[1]
    with pytest.raises(RuntimeError, match=r"Failed to encode word\(s\): no, the"):
        later[0:2]


def test_the_load_error_is_the_tokenizers_own_message(tmp_path, load, shared_tokenizer):
    """Oracle: agreement. An unsupported code reads the same at load as on a direct call."""
    with pytest.raises(ValueError) as direct:
        shared_tokenizer.encode("no", languages="FR")
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

    assert from_pkl.words == dataset.words
    assert from_pkl.languages == dataset.languages
    for i in range(len(ROWS)):
        assert same_encoding(from_pkl[i], dataset[i])
    assert same_encoding(from_pkl[0 : len(ROWS)], dataset[0 : len(ROWS)])
