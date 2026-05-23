"""
BridgeDataset: A dataset class for managing orthographic and phonological data using
the unified BridgeEncoding dataclass and BridgeTokenizer for processing.
"""

import logging
import pickle
import random
from functools import lru_cache
from pathlib import Path

import pandas as pd

from bridge.domain.datamodels import BridgeEncoding, DatasetConfig
from bridge.domain.tokenizer.bridge_tokenizer import BridgeTokenizer
from bridge.infra.clients.gcp.gcs_client import GCSClient
from bridge.utils import device_manager

logger = logging.getLogger(__name__)


class BridgeDataset:
    """
    Dataset class managing orthographic and phonological representations of words.
    Uses BridgeTokenizer for unified tokenization and BridgeEncoding for data storage.

    Each word in the dataset is paired with a language code (``self.languages``);
    when no language column is supplied in the source data, every word defaults to
    English. The language is plumbed through to the tokenizer so that
    code-switched corpora encode correctly.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        gcs_client: GCSClient | None = None,
    ):
        """
        Initialize the dataset with configuration and setup tokenization.

        Args:
            dataset_config: Configuration object containing dataset parameters
            gcs_client: Optional GCS client for reading datasets from
                ``gs://`` paths. Required only if ``dataset_filepath`` is a GCS URI.
        """
        self.dataset_config = dataset_config
        self.gcs_client = gcs_client
        self.device = device_manager.device

        self.tokenizer = BridgeTokenizer(
            phoneme_cache_size=getattr(dataset_config, "tokenizer_cache_size", 10000),
            custom_cmudict_path=dataset_config.custom_cmudict_path,
        )

        vocab_sizes = self.tokenizer.get_vocabulary_sizes()
        self.orthographic_vocabulary_size = vocab_sizes["orthographic"]
        self.phonological_vocabulary_size = vocab_sizes["phonological"]

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
            return pd.read_csv(
                path, keep_default_na=False, low_memory=False, dtype={"word_raw": str}
            )
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

    def _process_raw_dataframe(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Validate and filter ``word_raw`` entries.

        Returns parallel lists ``(words, languages)``. If the input DataFrame has no
        ``language`` column, every word is tagged ``"EN"`` by default.
        """
        if "word_raw" not in df.columns:
            raise KeyError("DataFrame must contain 'word_raw' column")

        if "language" in df.columns:
            input_languages = df["language"].tolist()
        else:
            input_languages = ["EN"] * len(df)

        valid_words: list[str] = []
        valid_languages: list[str] = []
        for idx, (word, lang) in enumerate(zip(df["word_raw"], input_languages, strict=False)):
            if not isinstance(word, str):
                logger.warning(f"Skipping non-str entry: {word}")
                continue
            if idx == 0 and self._encode_single_word(word, language=lang) is None:
                logger.warning(f"Initial encoding validation failed for {word} ({lang})")
                continue
            valid_words.append(word)
            valid_languages.append(lang)
        return valid_words, valid_languages

    @lru_cache(maxsize=128)  # noqa: B019 - dataset instance lifetime == one training run, bounded memory
    def _encode_single_word(self, word: str, language: str | None = None) -> BridgeEncoding | None:
        """
        Encode a single word using the tokenizer with caching.

        ``language`` is passed as a hashable scalar (not a dict) so this method
        can use ``@lru_cache``. It's expanded into a single-entry language_map
        before being handed to the tokenizer.
        """
        language_map = {word.lower(): language.upper()} if language else None

        try:
            encoding = self.tokenizer.encode(word, language_map=language_map)
            if encoding is None:
                return None
            return encoding.to(self.device)
        except Exception as e:
            logger.error(f"Error encoding word '{word}': {e}")
            return None

    def _get_encoding_batch(
        self, words: list[str], language_map: dict[str, str] | None = None
    ) -> BridgeEncoding:
        """Get batched encodings for a list of words."""
        batch_encoding = self.tokenizer.encode(words, language_map=language_map)
        if batch_encoding is None:
            raise RuntimeError("Batch encoding failed for words: " + ", ".join(words))
        return batch_encoding

    def _get_encoding(
        self,
        word: str | list[str],
        language_map: dict[str, str] | None = None,
    ) -> BridgeEncoding | None:
        """
        Get encoding for a single word or a list of words.

        Single-word encodings are memoized via the ``lru_cache`` on
        ``_encode_single_word``.
        """
        if isinstance(word, str):
            lang = (language_map or {}).get(word.lower())
            return self._encode_single_word(word, language=lang)
        if isinstance(word, list):
            return self._get_encoding_batch(word, language_map=language_map)
        raise TypeError("Input must be a string or a list of strings")

    def _resolve_language_map(
        self,
        words: list[str],
        provided_language_map: dict[str, str] | None = None,
        strict_conflicts: bool = True,
    ) -> dict[str, str]:
        """
        Build a ``{lowercase_word: UPPERCASE_LANGUAGE}`` map from the dataset.

        If ``provided_language_map`` is supplied, it is returned unchanged.
        Otherwise, languages are looked up from ``self.languages`` for each word.

        Args:
            strict_conflicts: When True (the default for ``__getitem__``), raise
                ``ValueError`` if a word appears in more than one language. When
                False (used by the public ``get_encoding``), fall back to the first
                language encountered.
        """
        if provided_language_map is not None:
            return provided_language_map
        if not self.languages:
            return {}

        language_map: dict[str, str] = {}
        for word in words:
            word_indices = [i for i, w in enumerate(self.words) if w == word]
            if not word_indices:
                raise KeyError(f"Word '{word}' not found in dataset")
            unique_languages = {self.languages[i] for i in word_indices}
            if len(unique_languages) > 1 and strict_conflicts:
                raise ValueError(
                    f"Word '{word}' found in multiple languages: "
                    f"{', '.join(sorted(unique_languages))}. "
                    "Use get_encoding() with an explicit language_map to disambiguate."
                )
            language_map[word.lower()] = next(iter(unique_languages)).upper()
        return language_map

    def _get_encoding_unified(
        self,
        idx: int | slice | str | list[str],
        language_map: dict[str, str] | None = None,
        strict_conflicts: bool = True,
    ) -> BridgeEncoding:
        """
        Unified path for ``__getitem__`` and ``get_encoding`` — turns any indexer
        into a (words, language_map) pair, then encodes.
        """
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.words):
                raise IndexError(f"Index {idx} out of range [0, {len(self.words)})")
            words = [self.words[idx]]
            if language_map is None and self.languages:
                resolved_language_map = {words[0].lower(): self.languages[idx].upper()}
            else:
                resolved_language_map = language_map or {}

        elif isinstance(idx, slice):
            words = self.words[idx]
            if language_map is None and self.languages:
                selected_langs = self.languages[idx]
                resolved_language_map = {
                    w.lower(): lang.upper() for w, lang in zip(words, selected_langs, strict=False)
                }
            else:
                resolved_language_map = language_map or {}

        elif isinstance(idx, str):
            if idx not in self.words:
                raise KeyError(f"Word '{idx}' not found in dataset")
            words = [idx]
            resolved_language_map = self._resolve_language_map(
                words,
                provided_language_map=language_map,
                strict_conflicts=strict_conflicts,
            )

        elif isinstance(idx, list):
            if not all(isinstance(i, str) for i in idx):
                raise TypeError("List indices must be strings")
            words = idx
            resolved_language_map = self._resolve_language_map(
                words,
                provided_language_map=language_map,
                strict_conflicts=strict_conflicts,
            )
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

        if len(words) == 1:
            encoding = self._get_encoding(words[0], resolved_language_map)
        else:
            encoding = self._get_encoding(words, resolved_language_map)

        if encoding is None:
            raise RuntimeError(f"Failed to encode word(s): {', '.join(words)}")
        return encoding

    def __len__(self) -> int:
        """Return the number of valid words in the dataset."""
        return len(self.words)

    def __getitem__(self, idx: int | slice | str | list[str]) -> BridgeEncoding:
        """
        Retrieve encoded data for specified index. Uses each word's stored language
        with strict conflict checking — raises ``ValueError`` if a word string
        appears in the dataset with more than one language. In that case use
        :meth:`get_encoding` with an explicit ``language_map`` instead.
        """
        return self._get_encoding_unified(idx, language_map=None, strict_conflicts=True)

    def get_encoding(
        self,
        idx: int | slice | str | list[str],
        language_map: dict[str, str] | None = None,
    ) -> BridgeEncoding:
        """
        Retrieve encoded data with optional explicit language override.

        Unlike :meth:`__getitem__`, this does NOT raise on cross-language
        homographs — it falls back to the first language found if no override
        is supplied.
        """
        return self._get_encoding_unified(idx, language_map=language_map, strict_conflicts=False)

    def shuffle(self, cutoff: int) -> None:
        """
        Shuffle the dataset up to ``cutoff`` (exclusive). Shuffling is performed
        on indices so the parallel ``words`` / ``languages`` lists stay aligned.
        """
        if cutoff > len(self.words):
            raise ValueError(f"Cutoff {cutoff} exceeds dataset size {len(self.words)}")

        original_words = self.words.copy()

        indices = list(range(cutoff))
        random.shuffle(indices)

        self.words = [self.words[i] for i in indices] + self.words[cutoff:]
        self.languages = [self.languages[i] for i in indices] + self.languages[cutoff:]

        assert len(self.words) == len(original_words)
        assert set(self.words) == set(original_words)
