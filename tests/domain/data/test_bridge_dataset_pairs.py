"""A dataset row is an input and a target: two strings, each with its own languages.

Pins what the ``target_raw`` and ``target_language`` columns mean, and that a row's four
fields stay together. Each test names its oracle.

The rows are built so that input and target differ in content, in language and in
length, and so that a mix-up is visible: "no" is N OW1 in English and N AO1 in Spanish,
so swapping the two sides, or giving one side the other's language, changes phoneme ids.
"""

import pickle

import pandas as pd
import pytest
import torch

from bridge.domain.data import BridgeDataset, Example
from bridge.domain.datamodels import DatasetConfig
from tests.conftest import batch_tensors

# (word_raw, language, target_raw, target_language)
PAIRS = [
    ("no", "EN", "no", "ES"),
    ("no", "ES", "no", "EN"),
    ("the word is gato", "EN EN EN ES", "la con", "ES"),
    ("la", "ES", "the word", "EN"),
    ("no me", "ES EN", "me no la", "EN ES ES"),
    ("sin son", "ES", "gato", "ES"),
    ("me", "EN", "me", "EN"),
]
HEADER = "word_raw,language,target_raw,target_language"


def write_csv(path, rows, header=HEADER):
    path.write_text(header + "\n" + "".join(",".join(row) + "\n" for row in rows))
    return str(path)


@pytest.fixture
def pairs(tmp_path, load):
    return load(write_csv(tmp_path / "pairs.csv", PAIRS))


def assert_batch_is(batch, tokenizer, examples):
    """Oracle: the tokenizer called directly, the input fields to ``encode_input`` and the
    target fields to ``encode_target``."""
    inputs = tokenizer.encode_input(
        [e.input for e in examples], [e.input_languages for e in examples]
    )
    target = tokenizer.encode_target(
        [e.target for e in examples], [e.target_languages for e in examples]
    )
    want = {
        "input.orth.ids": inputs.orth.ids,
        "input.orth.pad_mask": inputs.orth.pad_mask,
        "input.phon.ids": inputs.phon.ids,
        "input.phon.pad_mask": inputs.phon.pad_mask,
        "target.orth.ids": target.orth.ids,
        "target.orth.pad_mask": target.orth.pad_mask,
        "target.orth.labels": target.orth.labels,
        "target.phon.ids": target.phon.ids,
        "target.phon.pad_mask": target.phon.pad_mask,
        "target.phon.labels": target.phon.labels,
    }
    got = batch_tensors(batch)
    assert got.keys() == want.keys()
    for name, tensor in want.items():
        assert got[name].dtype == tensor.dtype and torch.equal(got[name], tensor), name


def test_a_row_is_stored_as_one_example_of_four_fields(pairs):
    """Oracle: hand-written. Each side keeps its own string and one code per word."""
    assert pairs.examples[:5] == [
        Example("no", ("EN",), "no", ("ES",)),
        Example("no", ("ES",), "no", ("EN",)),
        Example("the word is gato", ("EN", "EN", "EN", "ES"), "la con", ("ES", "ES")),
        Example("la", ("ES",), "the word", ("EN", "EN")),
        Example("no me", ("ES", "EN"), "me no la", ("EN", "ES", "ES")),
    ]
    assert len(pairs) == len(PAIRS)


def test_each_side_is_encoded_from_its_own_string_and_languages(pairs):
    """Oracle: hand-written phoneme rows. 86 and 87 are ``[BOS]`` and ``[EOS]``.

    Row 0 reads the English "no" and is taught the Spanish one; row 1 is the reverse.
    Sides exchanged, or one language used for both, cannot give these four rows.
    """
    assert pairs[0].input.phon.ids.tolist() == [[86, 69, 33, 87]]
    assert pairs[0].target.phon.ids.tolist() == [[86, 69, 22]]
    assert pairs[1].input.phon.ids.tolist() == [[86, 69, 22, 87]]
    assert pairs[1].target.phon.ids.tolist() == [[86, 69, 33]]

    idx_2_char = pairs.tokenizer.char_tokenizer.idx_2_char
    batch = pairs[0:2]
    assert [idx_2_char[int(i)] for i in batch.input.orth.ids[:, 0]] == ["EN", "ES"]
    assert [idx_2_char[int(i)] for i in batch.target.orth.ids[:, 0]] == ["ES", "EN"]


def test_every_row_and_slice_equals_the_tokenizer_called_side_by_side(pairs):
    for i, example in enumerate(pairs.examples):
        assert_batch_is(pairs[i], pairs.tokenizer, [example])
    for size in (2, 3, len(pairs)):
        for start in range(0, len(pairs), size):
            assert_batch_is(
                pairs[start : start + size], pairs.tokenizer, pairs.examples[start : start + size]
            )


def test_input_and_target_widths_are_independent(pairs):
    """Oracle: hand-counted. Row 2 reads 16 characters and a switch and is taught 6."""
    batch = pairs[2]
    # [EN, BOS, 12 characters and spaces, ES, 4 characters, EOS]
    assert batch.input.orth.ids.shape == (1, 2 + len("the word is gato") + 1 + 1)
    # [ES, BOS, 6 characters and a space]
    assert batch.target.orth.ids.shape == (1, 2 + len("la con"))
    assert batch.target.orth.labels.shape == (1, 2 + len("la con"))
    assert batch.input.phon.ids.shape[1] != batch.target.phon.ids.shape[1] + 1


@pytest.mark.parametrize(
    "header",
    [
        "word_raw",
        "word_raw,language",
        "word_raw,target",
        "word_raw,targets_raw",
        "word_raw,Target_raw",
    ],
)
def test_a_file_without_target_raw_raises_naming_the_fix(tmp_path, load, header):
    """Oracle: the rule. No column, or a misspelt one, does not load as repetition.

    A default would turn a typo in the header into a run that trains the model to repeat
    its input, with nothing to say so. The message names the column, the columns found,
    which is where a misspelling shows, and what to do for repetition.
    """
    width = header.count(",") + 1
    rows = [("cat", "EN")[:width], ("dog", "EN")[:width]]
    with pytest.raises(KeyError) as raised:
        load(write_csv(tmp_path / "no_target.csv", rows, header=header))
    message = str(raised.value)
    assert "no 'target_raw' column" in message
    assert str(header.split(",")) in message
    assert "for repetition" in message and "copy 'word_raw' into 'target_raw'" in message


def test_a_pkl_or_a_client_frame_without_target_raw_raises_the_same(
    tmp_path, load, shared_tokenizer
):
    """Oracle: agreement. The rule belongs to the dataset, not to one file format."""
    path = tmp_path / "no_target.pkl"
    with open(path, "wb") as f:
        pickle.dump({"word_raw": ["cat"], "language": ["EN"]}, f)
    with pytest.raises(KeyError, match="no 'target_raw' column"):
        load(str(path))

    class Client:
        def read_csv(self, bucket_name, blob_name, **kwargs):
            return pd.DataFrame({"word_raw": ["cat"], "language": ["EN"]})

    with pytest.raises(KeyError, match="no 'target_raw' column"):
        BridgeDataset(
            DatasetConfig(dataset_filepath="gs://bucket/rows.csv"),
            gcs_client=Client(),
            tokenizer=shared_tokenizer,
        )


def test_a_file_without_word_raw_raises(tmp_path, load):
    with pytest.raises(KeyError, match="no 'word_raw' column"):
        load(write_csv(tmp_path / "no_input.csv", [("cat",)], header="target_raw"))


@pytest.mark.parametrize(
    ("row", "column"),
    [
        (("no", "ES", "", "ES"), "target_raw"),
        (("no", "ES", "   ", "ES"), "target_raw"),
        (("", "ES", "no", "ES"), "word_raw"),
    ],
)
def test_a_blank_string_cell_raises_at_load_naming_the_row_and_the_column(
    tmp_path, load, row, column
):
    """Oracle: the rule. A blank target would teach the model to write nothing, in
    silence, so it is refused like a blank language cell."""
    rows = [("no", "ES", "no", "ES"), row]
    with pytest.raises(ValueError, match=rf"Row 1: the '{column}' cell is blank"):
        load(write_csv(tmp_path / "blank.csv", rows))


def test_a_row_missing_its_last_field_raises_at_load(tmp_path, load):
    """Oracle: the rule. pandas fills a short row with an empty cell, which under a
    header ending in ``target_raw`` is a blank target."""
    path = tmp_path / "short.csv"
    path.write_text("word_raw,language,target_raw\nno,ES,no\ngato,ES\n")
    with pytest.raises(ValueError, match="Row 1: the 'target_raw' cell is blank"):
        load(str(path))


@pytest.mark.parametrize("column", ["word_raw", "target_raw"])
def test_a_csv_column_of_digits_stays_strings(tmp_path, load, column):
    """Oracle: hand-written. A column pandas could read as numbers is text: the rows
    load as the strings written, and are not left out as non-str entries."""
    cells = {"word_raw": ["no", "me"], "target_raw": ["no", "me"]} | {column: ["7", "12"]}
    rows = list(zip(cells["word_raw"], cells["target_raw"], strict=True))
    dataset = load(write_csv(tmp_path / "digits.csv", rows, header="word_raw,target_raw"))
    assert [(e.input, e.target) for e in dataset.examples] == rows


def test_the_storage_client_is_asked_for_both_strings_as_written(shared_tokenizer):
    """Oracle: the keywords, written out here. A cell reading ``nan`` or ``12`` stays the
    string it is, on either side, when a client reads the file."""
    seen = {}

    class Client:
        def read_csv(self, bucket_name, blob_name, **kwargs):
            seen.update(kwargs)
            return pd.DataFrame({"word_raw": ["cat"], "target_raw": ["cat"]})

    BridgeDataset(
        DatasetConfig(dataset_filepath="gs://bucket/rows.csv"),
        gcs_client=Client(),
        tokenizer=shared_tokenizer,
    )
    assert seen["dtype"] == {"word_raw": str, "target_raw": str}
    assert seen["keep_default_na"] is False


def test_repetition_is_a_file_whose_two_columns_match(tmp_path, load):
    """Oracle: the tokenizer called with one string for both sides. Nothing marks the
    file as repetition, and nothing needs to."""
    dataset = load(
        write_csv(
            tmp_path / "rep.csv", [("cat", "cat"), ("no me", "no me")], header="word_raw,target_raw"
        )
    )
    assert dataset.examples == [
        Example("cat", ("EN",), "cat", ("EN",)),
        Example("no me", ("EN", "EN"), "no me", ("EN", "EN")),
    ]
    assert_batch_is(dataset[0:2], dataset.tokenizer, dataset.examples)


@pytest.mark.parametrize(
    ("header", "row", "expected"),
    [
        # One code for the row is the row's language, so the target has it too.
        ("word_raw,language,target_raw", ("la", "ES", "sin son"), ("ES", "ES")),
        ("word_raw,language,target_raw", ("no me", "ES", "la"), ("ES",)),
        ("word_raw,language,target_raw", ("no me", "es", "la"), ("ES",)),
        # No language column is English throughout.
        ("word_raw,target_raw", ("the", "the word"), ("EN", "EN")),
    ],
)
def test_a_missing_target_language_copies_a_single_code(tmp_path, load, header, row, expected):
    """Oracle: hand-written. One code says nothing about word positions, so it carries
    over to a target of any length."""
    dataset = load(write_csv(tmp_path / "default.csv", [row], header=header))
    assert dataset.examples[0].target_languages == expected


@pytest.mark.parametrize(
    ("word", "cell", "target"),
    [
        # The dangerous case: the counts match, so the codes would fit the target's words
        # and be wrong. "gato the" is not "the gato" in the other order's languages.
        ("the gato", "EN ES", "gato the"),
        # Even when the target is the input.
        ("the gato", "EN ES", "the gato"),
        ("the gato", "EN ES", "la"),
        # Per word is a property of the cell, not of the codes in it.
        ("no me", "ES ES", "la con"),
    ],
)
def test_a_missing_target_language_is_not_copied_from_per_word_codes(
    tmp_path, load, word, cell, target
):
    """Oracle: the rule. Per-word codes belong to the input's words. Without a
    ``target_language`` column the load raises, naming the row and the column to add."""
    rows = [("la", "ES", "la"), (word, cell, target)]
    message = (
        rf"Row 1 \('{word}', language '{cell}'\): .*'{target}'.* Add a 'target_language' column"
    )
    with pytest.raises(ValueError, match=message):
        load(write_csv(tmp_path / "per_word.csv", rows, header="word_raw,language,target_raw"))


def test_a_pkl_list_of_codes_follows_the_same_default_rule(tmp_path, load):
    """Oracle: the rule, on pkl cells. A list holding one code is one code."""

    def pkl(name, cells):
        path = tmp_path / name
        with open(path, "wb") as f:
            pickle.dump(
                {"word_raw": ["no me", "la"], "language": cells, "target_raw": ["la", "no me"]}, f
            )
        return str(path)

    dataset = load(pkl("one.pkl", [["ES"], ("EN",)]))
    assert [e.target_languages for e in dataset.examples] == [("ES",), ("EN", "EN")]
    with pytest.raises(ValueError, match=r"Row 0 .*Add a 'target_language' column"):
        load(pkl("two.pkl", [["ES", "EN"], "EN"]))


def test_a_target_language_column_is_read_without_a_language_column(tmp_path, load):
    """Oracle: hand-written. "no" is N OW1 in English (33) and N AO1 in Spanish (22), so
    the input is English by default and the target is what its own column says."""
    dataset = load(
        write_csv(
            tmp_path / "target_only.csv",
            [("no", "no", "ES")],
            header="word_raw,target_raw,target_language",
        )
    )
    assert dataset.examples == [Example("no", ("EN",), "no", ("ES",))]
    assert dataset[0].input.phon.ids.tolist() == [[86, 69, 33, 87]]
    assert dataset[0].target.phon.ids.tolist() == [[86, 69, 22]]


def test_a_target_language_column_is_read_as_written(pairs):
    """Oracle: hand-written. With the column there, nothing is copied from ``language``."""
    assert [e.target_languages for e in pairs.examples] == [
        ("ES",),
        ("EN",),
        ("ES", "ES"),
        ("EN", "EN"),
        ("EN", "ES", "ES"),
        ("ES",),
        ("EN",),
    ]


@pytest.mark.parametrize(
    ("cell", "message"),
    [
        ("ES EN", r"Row 1 \('me no la', target_language 'ES EN'\): 2 language codes for 3 words"),
        ("FR", r"Row 1 \('me no la', target_language 'FR'\): Invalid languages: \['FR'\]"),
        ("", r"Row 1 \('me no la', target_language ''\): No language code given"),
        ("--", r"Row 1 \('me no la', target_language '--'\): a dataset row names the language"),
        ("-- EN ES", r"Row 1 .*target_language.*whole row is unspecified"),
    ],
)
def test_a_bad_target_language_cell_raises_at_load_naming_the_row(tmp_path, load, cell, message):
    """Oracle: the rule. Every check a ``language`` cell gets, a ``target_language`` cell
    gets, and the message says which column it was."""
    rows = [("no", "ES", "no", "ES"), ("no me", "ES EN", "me no la", cell)]
    with pytest.raises(ValueError, match=message):
        load(write_csv(tmp_path / "bad.csv", rows))


@pytest.mark.parametrize("bad", [0, 1])
def test_a_target_with_no_pronunciation_raises_when_its_row_is_read(tmp_path, load, bad):
    """Oracle: the lexicon. "the" has no Spanish entry, so "the" tagged ES is no target.

    The file loads, wherever the row sits, the first row included. Reading the row raises
    and names the row and the target side; the other row reads.
    """
    rows = [("no", "ES", "no", "ES"), ("no", "ES", "no", "ES")]
    rows[bad] = ("no", "ES", "the", "ES")
    dataset = load(write_csv(tmp_path / "target.csv", rows))
    assert len(dataset) == 2

    message = f"No pronunciation for a word of row {bad}'s target 'the'$"
    with pytest.raises(RuntimeError, match=message):
        dataset[bad]
    with pytest.raises(RuntimeError, match=message):
        dataset[0:2]
    assert len(dataset[1 - bad]) == 1


def test_a_failed_read_names_every_row_and_side_without_a_pronunciation(tmp_path, load):
    """Oracle: hand-written. The rows are numbered by their place in the dataset, not in
    the slice, and a row that reads is not named."""
    good, bad_input = ("no", "ES", "no", "ES"), ("the", "ES", "no", "ES")
    bad_target = ("no", "ES", "the long", "ES")
    both = ("the", "ES", "the", "ES")
    dataset = load(write_csv(tmp_path / "several.csv", [good, bad_input, good, bad_target, both]))

    with pytest.raises(RuntimeError) as raised:
        dataset[1:5]
    assert str(raised.value) == (
        "No pronunciation for a word of row 1's input 'the', row 3's target 'the long', "
        "row 4's input 'the', row 4's target 'the'"
    )
    with pytest.raises(RuntimeError, match="word of row 3's target 'the long'$"):
        dataset[0:5:3]
    # Each side alone stops the read: row 1's target reads, and row 3's input does.
    with pytest.raises(RuntimeError, match="word of row 1's input 'the'$"):
        dataset[1]
    with pytest.raises(RuntimeError, match="word of row 3's target 'the long'$"):
        dataset[3]
    assert len(dataset[0:3:2]) == 2


def test_a_row_without_a_string_on_either_side_is_left_out(tmp_path, load, caplog):
    """Oracle: hand-written. A pkl cell that is not a string is not an example."""
    path = tmp_path / "gaps.pkl"
    columns = {
        "word_raw": ["no", None, "la"],
        "language": ["ES"] * 3,
        "target_raw": ["no", "me", 7],
    }
    with open(path, "wb") as f:
        pickle.dump(columns, f)
    with caplog.at_level("WARNING"):
        dataset = load(str(path))
    assert dataset.examples == [Example("no", ("ES",), "no", ("ES",))]
    assert "Skipping row 1" in caplog.text and "Skipping row 2" in caplog.text


def test_a_shuffle_keeps_each_rows_four_fields_together(pairs):
    """Oracle: invariant. The set of four-field rows is the same before and after, and
    every position still encodes as its own four fields say.

    The rows share strings across sides and languages ("no" four ways), so moving the
    inputs without their targets, or the strings without their languages, pairs a field
    with the wrong row and changes the set.
    """
    before = list(pairs.examples)
    assert len(set(before)) == len(before)
    moved = False
    for seed in (0, 1, 2):
        pairs.shuffle(len(pairs), seed=seed)
        assert sorted(pairs.examples) == sorted(before)
        moved = moved or pairs.examples != before
        for i, example in enumerate(pairs.examples):
            assert_batch_is(pairs[i], pairs.tokenizer, [example])
        assert_batch_is(pairs[0 : len(pairs)], pairs.tokenizer, pairs.examples)
    assert moved, "the shuffles must have moved something"


def test_a_shuffle_leaves_the_tail_rows_whole_and_in_place(pairs):
    before = list(pairs.examples)
    pairs.shuffle(4, seed=0)
    assert pairs.examples[4:] == before[4:]
    assert sorted(pairs.examples[:4]) == sorted(before[:4])
