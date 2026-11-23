"""
BridgeDataset: A dataset class for managing orthographic and phonological data using
the unified BridgeEncoding dataclass and BridgeTokenizer for processing.
"""

import logging
import pickle
import random
from pathlib import Path
from typing import Protocol

import pandas as pd

from bridge.domain.datamodels import BridgeEncoding, DatasetConfig
from bridge.domain.tokenizer.bridge_tokenizer import BridgeTokenizer
from bridge.domain.tokenizer.languages import UNSPECIFIED, per_word_languages
from bridge.utils import device_manager

logger = logging.getLogger(__name__)


class CSVReader(Protocol):
    """What a ``gs://`` dataset path needs from a storage client.

    Structural, not a base class: BRIDGE ships no cloud client, so a caller reading
    from object storage passes their own object with this one method.
    """

    def read_csv(self, bucket_name: str, blob_name: str, **kwargs) -> pd.DataFrame: ...


class BridgeDataset:
    """
    Dataset class managing orthographic and phonological representations of words.
    Uses BridgeTokenizer for unified tokenization and BridgeEncoding for data storage.

    Every row carries the language of each of its words: ``self.languages`` holds one
    tuple of codes per row, one code per word. In the source data the ``language`` column
    holds one code for the whole row (``EN``) or one code per word (``EN EN ES``), and a
    pkl file may hold a list or tuple of codes instead. When the source has no
    ``language`` column, every word is English. Rows are read by position, an ``int`` or a
    ``slice``: a string does not identify a row, since one spelling can appear under two
    languages.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        gcs_client: CSVReader | None = None,
        tokenizer: BridgeTokenizer | None = None,
    ):
        """
        Initialize the dataset with configuration and setup tokenization.

        Args:
            dataset_config: Configuration object containing dataset parameters
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
        self.device = device_manager.device

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

        # Memoized single-row encodings, keyed by (word, languages). Instance-scoped, so
        # this dict dies with the dataset. The parsed lexicon behind it does not:
        # `_load_lexicon` and `load_phoneme_table` are process-lifetime caches, deliberately,
        # since re-reading them per dataset costs ~1 s and ~81 MB each time.
        self._encoding_cache: dict[tuple[str, tuple[str, ...]], BridgeEncoding | None] = {}

        self.dataset_filepath = dataset_config.dataset_filepath
        raw_df = self._load_raw_dataframe(self.dataset_filepath)
        self.words, self.languages = self._process_raw_dataframe(raw_df)

        logger.info(f"Loaded {len(self.words)} valid words.")

    def _load_raw_dataframe(self, path: str) -> pd.DataFrame:
        """
        Dispatch method to load raw data based on path scheme or extension.
        Returns a DataFrame with at least a ``word_raw`` column, and optionally a
        ``language`` column.
        """
        if path.startswith("gs://"):
            parts = path.replace("gs://", "").split("/", 1)
            bucket, blob = parts[0], parts[1]
            return self._read_gcs_csv(bucket, blob)
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(path, keep_default_na=False, low_memory=False, dtype={"word_raw": str})
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
            if "word_raw" not in data:
                raise KeyError("Dataset dictionary must contain 'word_raw' key")
            df_data = {"word_raw": list(data["word_raw"])}
            if "language" in data:
                if len(data["word_raw"]) != len(data["language"]):
                    raise ValueError("'word_raw' and 'language' columns must have equal length")
                df_data["language"] = list(data["language"])
            return pd.DataFrame(df_data)
        raise ValueError(f"Unsupported data format: {ext}")

    def _read_gcs_csv(self, bucket: str, blob: str) -> pd.DataFrame:
        """Read a CSV blob from GCS into a DataFrame."""
        if self.gcs_client is None:
            raise ValueError("A GCS client must be passed to BridgeDataset to read gs:// paths.")
        return self.gcs_client.read_csv(
            bucket_name=bucket,
            blob_name=blob,
            sep=",",
            parse_dates=True,
            dtype={"word_raw": str},
            keep_default_na=False,
            low_memory=False,
        )

    def _process_raw_dataframe(self, df: pd.DataFrame) -> tuple[list[str], list[tuple[str, ...]]]:
        """
        Validate ``word_raw`` and ``language`` entries.

        Returns parallel lists ``(words, languages)``, where each row's languages are one
        code per word. If the input DataFrame has no ``language`` column, every word is
        tagged ``"EN"``. Every language cell is checked here, so a blank or missing cell,
        ``--``, an unsupported code or a code count that does not match the row's words
        raises ``ValueError`` naming the row, before anything trains on it.
        """
        if "word_raw" not in df.columns:
            raise KeyError("DataFrame must contain 'word_raw' column")

        if "language" in df.columns:
            cells = df["language"].tolist()
        else:
            cells = ["EN"] * len(df)

        valid_words: list[str] = []
        valid_languages: list[tuple[str, ...]] = []
        for row, (word, cell) in enumerate(zip(df["word_raw"], cells, strict=True)):
            if not isinstance(word, str):
                logger.warning(f"Skipping non-str entry: {word}")
                continue
            try:
                languages = per_word_languages(word, cell)
                if UNSPECIFIED in languages:
                    raise ValueError("a dataset row names the language of its words")
            except ValueError as e:
                raise ValueError(f"Row {row} ({word!r}, language {cell!r}): {e}") from e
            if row == 0 and self._encode_single_word(word, languages) is None:
                logger.warning(f"Initial encoding validation failed for {word} ({languages})")
                continue
            valid_words.append(word)
            valid_languages.append(languages)
        return valid_words, valid_languages

    _ENCODING_CACHE_SIZE = 128

    def _encode_single_word(self, word: str, languages: tuple[str, ...]) -> BridgeEncoding | None:
        """Encode a single row using the tokenizer, memoized on ``(word, languages)``."""
        key = (word, languages)
        if key in self._encoding_cache:
            return self._encoding_cache[key]

        try:
            encoding = self.tokenizer.encode(word, languages=languages)
            encoding = encoding.to(self.device) if encoding is not None else None
        except Exception as e:
            logger.error(f"Error encoding word '{word}': {e}")
            encoding = None

        if len(self._encoding_cache) >= self._ENCODING_CACHE_SIZE:
            self._encoding_cache.pop(next(iter(self._encoding_cache)))
        self._encoding_cache[key] = encoding
        return encoding

    def __len__(self) -> int:
        """Return the number of valid words in the dataset."""
        return len(self.words)

    def __getitem__(self, idx: int | slice) -> BridgeEncoding:
        """Encode the row at ``idx``, or the rows of a slice, each in its stored languages."""
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.words):
                raise IndexError(f"Index {idx} out of range [0, {len(self.words)})")
            idx = slice(idx, idx + 1)
        elif not isinstance(idx, slice):
            raise TypeError(
                f"Invalid index type: {type(idx)}. Rows are indexed by position, an int or a slice."
            )
        words, languages = self.words[idx], self.languages[idx]

        # Single rows go through the memoizing path; batches straight to the tokenizer.
        if len(words) == 1:
            encoding = self._encode_single_word(words[0], languages[0])
        else:
            encoding = self.tokenizer.encode(words, languages=languages)

        if encoding is None:
            raise RuntimeError(f"Failed to encode word(s): {', '.join(words)}")
        return encoding

    def shuffle(self, cutoff: int, seed: int | None = None) -> None:
        """Shuffle the dataset up to ``cutoff`` (exclusive), leaving the tail in place.

        Shuffling is performed on indices so the parallel ``words`` / ``languages`` lists
        stay aligned, and the tail stays put so a validation partition remains comparable
        across epochs.

        ``seed`` makes the order reproducible. Without it this draws from the global
        ``random`` module, which is how data-order reproducibility used to depend on
        ``ModelConfig.seed`` by accident: ``Model.__init__`` calls ``set_seed``, which
        calls ``random.seed``, so constructing a model silently determined the shuffle.
        A seed here uses its own ``random.Random`` and touches no global state.
        """
        if cutoff > len(self.words):
            raise ValueError(f"Cutoff {cutoff} exceeds dataset size {len(self.words)}")

        indices = list(range(cutoff))
        (random.Random(seed) if seed is not None else random).shuffle(indices)

        self.words = [self.words[i] for i in indices] + self.words[cutoff:]
        self.languages = [self.languages[i] for i in indices] + self.languages[cutoff:]
