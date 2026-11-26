"""
BridgeDataset: the training examples of a data file, each an input and a target, encoded
on demand into a BridgeBatch by a BridgeTokenizer.
"""

import logging
import pickle
import random
from pathlib import Path
from typing import NamedTuple, Protocol

import pandas as pd

from bridge.domain.datamodels import BridgeBatch, DatasetConfig
from bridge.domain.tokenizer.bridge_tokenizer import BridgeTokenizer
from bridge.domain.tokenizer.languages import UNSPECIFIED, RowLanguages, per_word_languages

logger = logging.getLogger(__name__)

COLUMNS = ("word_raw", "language", "target_raw", "target_language")


class CSVReader(Protocol):
    """What a ``gs://`` dataset path needs from a storage client.

    Structural, not a base class: BRIDGE ships no cloud client, so a caller reading
    from object storage passes their own object with this one method.
    """

    def read_csv(self, bucket_name: str, blob_name: str, **kwargs) -> pd.DataFrame: ...


class Example(NamedTuple):
    """One row of a dataset: what the model reads and what it is taught to write.

    Each side is a string and one language code per word of it. One value holds all four
    fields, so nothing that reorders the rows can separate them.
    """

    input: str
    input_languages: tuple[str, ...]
    target: str
    target_languages: tuple[str, ...]


class BridgeDataset:
    """
    The examples of a data file. Each one is an input and a target, and they are two
    different things: the model reads the input and is taught to write the target.
    Repetition is a file whose two columns hold the same strings.

    The source has four columns:

    ``word_raw``         the input. Required.
    ``language``         the language of each input word: one code for the whole row
                         (``EN``) or one code per word (``EN EN ES``). A pkl file may hold
                         a list or tuple of codes. Without the column, every word is
                         English.
    ``target_raw``       the target. Required.
    ``target_language``  the same for the target. Without the column, a row whose
                         ``language`` is one code gives its target that code. A row
                         whose ``language`` is per word does not: its codes belong to the
                         input's words.

    ``self.examples`` holds one :class:`Example` per row. Rows are read by position, an
    ``int`` or a ``slice``: a string does not identify a row, since one spelling can
    appear under two languages and one input beside two targets.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        gcs_client: CSVReader | None = None,
        tokenizer: BridgeTokenizer | None = None,
    ):
        """
        Args:
            gcs_client: Any object with a ``read_csv(bucket_name, blob_name, **kwargs)``
                method, for reading datasets from ``gs://`` paths. Required only if
                ``dataset_filepath`` is a GCS URI.
            tokenizer: An existing tokenizer to reuse. Constructing one parses the
                pronunciation lexicons, costing ~1 s and ~81 MB retained, so paired
                train/test datasets should share a single instance rather than
                building one each. It must have been built from the same
                ``custom_cmudict_path`` as this config, since that lexicon is baked
                into the tokenizer at construction.
        """
        self.dataset_config = dataset_config
        self.gcs_client = gcs_client

        if tokenizer is not None and (
            tokenizer.custom_cmudict_path != dataset_config.custom_cmudict_path
        ):
            raise ValueError(
                f"Injected tokenizer was built with custom_cmudict_path="
                f"{tokenizer.custom_cmudict_path!r}, but this dataset's config specifies "
                f"{dataset_config.custom_cmudict_path!r}. The custom lexicon is merged in at "
                f"tokenizer construction, so sharing one across configs that disagree would "
                f"silently encode against the wrong pronunciations."
            )
        self.tokenizer = tokenizer or BridgeTokenizer(
            custom_cmudict_path=dataset_config.custom_cmudict_path,
        )

        vocab_sizes = self.tokenizer.get_vocabulary_sizes()
        self.orthographic_vocabulary_size = vocab_sizes["orthographic"]
        self.phonological_vocabulary_size = vocab_sizes["phonological"]

        self.dataset_filepath = dataset_config.dataset_filepath
        self.examples = self._read_examples(self._load_raw_dataframe(self.dataset_filepath))

        logger.info(f"Loaded {len(self.examples)} examples.")

    def _load_raw_dataframe(self, path: str) -> pd.DataFrame:
        if path.startswith("gs://"):
            parts = path.replace("gs://", "").split("/", 1)
            bucket, blob = parts[0], parts[1]
            return self._read_gcs_csv(bucket, blob)
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(
                path,
                keep_default_na=False,
                low_memory=False,
                dtype={"word_raw": str, "target_raw": str},
            )
            # pandas reads a row's first field as an index when the rows hold one more
            # field than the header names, which is what a comma between two codes does.
            # The codes would then load as the words.
            if not isinstance(df.index, pd.RangeIndex):
                raise ValueError(
                    "The rows hold more fields than the header names. The codes of a "
                    "language cell are separated by spaces, not commas: EN EN ES"
                )
            return df
        if ext == ".pkl":
            data = pickle.load(open(path, "rb"))
            if not isinstance(data, dict):
                raise ValueError("Dataset file must contain a dictionary")
            columns = {name: list(data[name]) for name in COLUMNS if name in data}
            lengths = {name: len(column) for name, column in columns.items()}
            if len(set(lengths.values())) > 1:
                raise ValueError(f"The columns of a dataset must have equal length, got {lengths}")
            return pd.DataFrame(columns)
        raise ValueError(f"Unsupported data format: {ext}")

    def _read_gcs_csv(self, bucket: str, blob: str) -> pd.DataFrame:
        if self.gcs_client is None:
            raise ValueError("A GCS client must be passed to BridgeDataset to read gs:// paths.")
        # Unlike the local read, the result is not checked for rows longer than the header.
        return self.gcs_client.read_csv(
            bucket_name=bucket,
            blob_name=blob,
            sep=",",
            parse_dates=True,
            dtype={"word_raw": str, "target_raw": str},
            keep_default_na=False,
            low_memory=False,
        )

    def _read_examples(self, df: pd.DataFrame) -> list[Example]:
        """Validate every row and return it as an :class:`Example`.

        Both required columns are checked first, then all four cells of every row, so
        a missing column, a blank or missing cell, ``--``, an unsupported code or a
        code count that does not match the row's words raises here, naming the row,
        before anything trains on it. Pronunciations are not checked: a row holding a
        word its lexicon lacks loads, and raises when it is read.
        """
        if "word_raw" not in df.columns:
            raise KeyError(f"The dataset has no 'word_raw' column. It has {list(df.columns)}")
        if "target_raw" not in df.columns:
            raise KeyError(
                f"The dataset has no 'target_raw' column. It has {list(df.columns)}. Every "
                "row is an input and a target; for repetition, where the model is taught "
                "to write what it read, copy 'word_raw' into 'target_raw'."
            )

        cells = df["language"].tolist() if "language" in df.columns else ["EN"] * len(df)
        target_cells = df["target_language"].tolist() if "target_language" in df.columns else None

        examples: list[Example] = []
        for row, (text, cell, target) in enumerate(
            zip(df["word_raw"], cells, df["target_raw"], strict=True)
        ):
            if not isinstance(text, str) or not isinstance(target, str):
                logger.warning(f"Skipping row {row}, a non-str entry: {text!r}, {target!r}")
                continue
            for column, value in (("word_raw", text), ("target_raw", target)):
                if not value.strip():
                    raise ValueError(
                        f"Row {row}: the {column!r} cell is blank. A row is an input and "
                        f"a target, and each holds at least one word."
                    )
            languages = self._languages(row, text, "language", cell)
            if target_cells is not None:
                target_cell = target_cells[row]
            elif len(cell.split() if isinstance(cell, str) else cell) == 1:
                target_cell = cell  # one code is the row's language, whatever its words
            else:
                raise ValueError(
                    f"Row {row} ({text!r}, language {cell!r}): the language is given per "
                    f"word, and those are the input's words, so it says nothing about the "
                    f"target {target!r}. Add a 'target_language' column."
                )
            target_languages = self._languages(row, target, "target_language", target_cell)
            examples.append(Example(text, languages, target, target_languages))
        return examples

    @staticmethod
    def _languages(row: int, text: str, column: str, cell: RowLanguages) -> tuple[str, ...]:
        """One code per word of ``text`` from its language cell, or a ``ValueError`` naming
        the row."""
        try:
            languages = per_word_languages(text, cell)
            if UNSPECIFIED in languages:
                raise ValueError("a dataset row names the language of its words")
        except ValueError as e:
            raise ValueError(f"Row {row} ({text!r}, {column} {cell!r}): {e}") from e
        return languages

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int | slice) -> BridgeBatch:
        """Encode the example at ``idx``, or those of a slice, each side in its stored
        languages. A slice that selects no rows raises ``IndexError``, and a row holding
        a word with no pronunciation raises ``RuntimeError`` naming the row."""
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.examples):
                raise IndexError(f"Index {idx} out of range [0, {len(self.examples)})")
            idx = slice(idx, idx + 1)
        elif not isinstance(idx, slice):
            raise TypeError(
                f"Invalid index type: {type(idx)}. Rows are indexed by position, an int or a slice."
            )
        examples = self.examples[idx]
        if not examples:
            raise IndexError(f"{idx} selects no rows of a dataset of {len(self.examples)}")
        inputs = [example.input for example in examples]
        targets = [example.target for example in examples]

        batch_input = self.tokenizer.encode_input(inputs, [e.input_languages for e in examples])
        batch_target = self.tokenizer.encode_target(targets, [e.target_languages for e in examples])
        if batch_input.phon is None or batch_target is None:
            raise RuntimeError(self._unpronounced(range(len(self.examples))[idx]))
        return BridgeBatch(input=batch_input, target=batch_target)

    def _unpronounced(self, rows: range) -> str:
        """Name the rows and sides of a read that hold a word its lexicon lacks, found by
        encoding each side of each row alone."""
        tokenizer, found = self.tokenizer, []
        for row in rows:
            e = self.examples[row]
            if tokenizer.encode_input([e.input], [e.input_languages]).phon is None:
                found.append(f"row {row}'s input {e.input!r}")
            if tokenizer.encode_target([e.target], [e.target_languages]) is None:
                found.append(f"row {row}'s target {e.target!r}")
        return "No pronunciation for a word of " + ", ".join(found)

    def shuffle(self, cutoff: int, seed: int | None = None) -> None:
        """Shuffle the dataset up to ``cutoff`` (exclusive), leaving the tail in place.

        Whole examples move, so an input keeps its target and both keep their languages.
        The tail stays put so a validation partition remains comparable across epochs.

        ``seed`` makes the order reproducible. Without it this draws from the global
        ``random`` module, so the order follows whatever last seeded that, and
        ``Model.__init__`` does: it calls ``set_seed``, which calls ``random.seed``. A
        seed here uses its own ``random.Random`` and touches no global state.
        """
        if cutoff > len(self.examples):
            raise ValueError(f"Cutoff {cutoff} exceeds dataset size {len(self.examples)}")

        head = self.examples[:cutoff]
        (random.Random(seed) if seed is not None else random).shuffle(head)
        self.examples[:cutoff] = head
