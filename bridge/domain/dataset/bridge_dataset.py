"""
BridgeDataset: A dataset class for managing orthographic and phonological data using
the unified BridgeEncoding dataclass and BridgeTokenizer for processing.
"""

import os
import pickle
import random
import logging
from pathlib import Path
from collections import OrderedDict
from functools import lru_cache
import pandas as pd
from bridge.domain.datamodels import DatasetConfig, BridgeEncoding
from bridge.utils import device_manager
from bridge.domain.dataset import BridgeTokenizer
from bridge.infra.clients.gcp.gcs_client import GCSClient
from pathlib import PosixPath

logger = logging.getLogger(__name__)


class BridgeDataset:
    """
    Dataset class managing orthographic and phonological representations of words.
    Uses BridgeTokenizer for unified tokenization and BridgeEncoding for data storage.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        gcs_client: GCSClient | None = None,
        cache_path: str | None = "data/.cache",
        cache_size: int = 1000,
    ):
        """
        Initialize the dataset with configuration and setup tokenization.

        Args:
            dataset_config: Configuration object containing dataset parameters
            device: Optional device specification for tensor placement
            cache_path: Optional path for caching tokenized data
            cache_size: Maximum number of encodings to keep in memory
        """
        self.cache_path = cache_path
        self.dataset_config = dataset_config
        self.gcs_client = gcs_client
        # Initialize device - prioritize config device over parameter
        self.device = device_manager.device

        # Initialize tokenizer with matching device
        self.tokenizer = BridgeTokenizer(
            phoneme_cache_size=getattr(dataset_config, "tokenizer_cache_size", 10000),
            custom_cmudict_path=dataset_config.custom_cmudict_path,
        )

        # Store vocabulary sizes for model initialization
        vocab_sizes = self.tokenizer.get_vocabulary_sizes()
        self.orthographic_vocabulary_size = vocab_sizes["orthographic"]
        self.phonological_vocabulary_size = vocab_sizes["phonological"]

        # Setup LRU cache for encodings
        self.encoding_cache = OrderedDict()
        self.max_cache_size = cache_size
        self.dataset_filepath = dataset_config.dataset_filepath
        # Load raw data into a DataFrame
        raw_df = self._load_raw_dataframe(self.dataset_filepath)
        # Process DataFrame into validated word list
        self.words, self.languages = self._process_raw_dataframe(raw_df)
        logger.info(f"Loaded {len(self.words)} valid words.")

        # Create cache directory if needed
        if cache_path and not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)

    def _load_raw_dataframe(self, path: str | PosixPath) -> pd.DataFrame:
        """
        Dispatch method to load raw data based on path scheme or extension.
        Returns a DataFrame with a 'word_raw' column.
        """
        if str(path).startswith("gs://"):
            # strip gs://<bucket>/ prefix
            parts = str(path).replace("gs://", "").split("/", 1)
            bucket, blob = parts[0], parts[1]
            return self._read_gcs_csv(bucket, blob)
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path)
        if ext == ".pkl":
            data = pickle.load(open(path, "rb"))
            if not isinstance(data, dict):
                raise ValueError("Dataset file must contain a dictionary")
            if "word_raw" not in data.keys():
                raise KeyError("DataFrame must contain 'word_raw' column")

            # convert dict keys to DataFrame
            dataframe = pd.DataFrame(
                {
                    "word_raw": list(data["word_raw"]),
                }
            )
            if "language" in data.keys():
                if len(data["word_raw"]) != len(data["language"]):
                    raise ValueError(
                        "Length of 'word_raw' and 'language' columns must match"
                    )
                dataframe["language"] = list(data["language"])

            return dataframe
        raise ValueError(f"Unsupported data format: {ext}")

    def _read_gcs_csv(self, bucket: str, blob: str) -> pd.DataFrame:
        """Read a CSV blob from GCS into a DataFrame."""
        if not self.gcs_client:
            raise ValueError(
                "Must provide GCS client to BridgeDataset to read GCS files. Use the gcs_client parameter"
            )
        return self.gcs_client.read_csv(
            bucket_name=bucket,
            blob_name=blob,
            sep=",",
            parse_dates=True,
            dtype={"col1": str},
        )

    def _process_raw_dataframe(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        Validate and filter 'word_raw' entries, ensuring tokenization consistency.
        """

        languages: list[str] = []
        if "language" in df.columns:
            languages = df["language"].tolist()
        else:
            languages = ["EN"] * len(df)

        valid_words: list[str] = []
        valid_languages: list[str] = []
        for idx, (word, lang) in enumerate(zip(df["word_raw"], languages)):
            if not isinstance(word, str) and isinstance(lang, str):
                logger.warning(f"Skipping non-str entry: {word}, {lang}")
                continue
            if idx == 0:
                # test encoding of first valid word
                if self._encode_single_word(word, language=lang) is None:
                    logger.warning(
                        f"Initial encoding validation failed for {word}, {lang}"
                    )
                    continue
            valid_words.append(word)
            valid_languages.append(lang)
        return valid_words, valid_languages

    @lru_cache(maxsize=128)
    def _encode_single_word(
        self, word: str, language: str | None = None
    ) -> BridgeEncoding | None:
        """
        Encode a single word using the tokenizer with caching.

        Ensures that all encoded sequences use the configured max sequence lengths
        to maintain consistency across batches.

        Args:
            word: Word to encode
            language: Encode the word using this languages phonology

        Returns:
            BridgeEncoding object or None if encoding fails
        """
        # The lru_cache requires the function to be hashable, so we use a string
        # for the current language and expand it back to the full language map
        # when passing it to each of the encoders
        if language is None:
            language_map = None
        else:
            language_map = {word: language.upper()}

        try:
            # Use the tokenizer directly for initial encoding
            encoding = self.tokenizer.encode(word, language_map=language_map)
            if encoding is None:
                return None

            # Ensure encoding is on correct device
            encoding = encoding.to(self.device)

            return encoding
        except Exception as e:
            logger.error(f"Error encoding word '{word}': {e}")
            return None

    def _get_encoding_batch(
        self, words: list[str], language_map: dict[str, str] | None = None
    ) -> BridgeEncoding:
        """
        Get batched encodings for a list of words without caching.

        Uses the tokenizer's batch encode() method to process the list of words.
        Then, each encoding is extracted (using __getitem__) and converted to a BridgeEncoding
        via from_dict.

        Args:
            words: List of words to encode.
            language_map: Optional mapping of words to their corresponding languages.

        Returns:
            List of BridgeEncoding objects.
        """
        batch_encoding = self.tokenizer.encode(words, language_map=language_map)
        if batch_encoding is None:
            raise RuntimeError("Batch encoding failed for words: " + ", ".join(words))

        return batch_encoding

    # TODO: Work on encoding cache logic
    def _get_encoding(
        self, word: str | list[str], language_map: dict[str, str] | None = None
    ) -> BridgeEncoding | None:
        """
        Get encoding for a word, using cache when available.

        Args:
            word: Word to retrieve encoding for
            language_map: Optional mapping of words to their corresponding languages.

        Returns:
            BridgeEncoding object or None if not available
        """
        # Check in-memory cache first
        # if word in self.encoding_cache:
        #     return self.encoding_cache[word]

        # Encode word if not in cache
        if isinstance(word, str):
            if language_map is None:
                language_map = {}
            return self._encode_single_word(
                word, language=language_map.get(word.lower(), None)
            )
        elif isinstance(word, list):
            return self._get_encoding_batch(word, language_map=language_map)
        else:
            raise TypeError("Input must be a string or a list of strings")

        # if encoding is not None:
        #     # Update cache with LRU policy
        #     if len(self.encoding_cache) >= self.max_cache_size:
        #         self.encoding_cache.popitem(last=False)
        #     self.encoding_cache[word] = encoding

    def __len__(self) -> int:
        """Return the number of valid words in the dataset."""
        return len(self.words)

    def _resolve_language_map(
        self,
        words: list[str],
        provided_language_map: dict[str, str] | None = None,
        strict_conflicts: bool = True,
    ) -> dict[str, str]:
        """
        Resolve language mapping for a list of words.

        Args:
            words: List of words to resolve languages for
            provided_language_map: Optional explicit language mapping
            strict_conflicts: If True, raise errors on language conflicts

        Returns:
            Dictionary mapping words to languages

        Raises:
            ValueError: If strict_conflicts=True and word has multiple languages
            KeyError: If word not found in dataset
        """
        if provided_language_map is not None:
            return provided_language_map

        if not self.languages:
            return {}

        language_map = {}
        for word in words:
            # Find all indices where this word appears
            word_indices = [i for i, w in enumerate(self.words) if w == word]

            if not word_indices:
                raise KeyError(f"Word '{word}' not found in dataset")

            # Get all languages for this word
            word_languages = [self.languages[i] for i in word_indices]
            unique_languages = set(word_languages)

            if len(unique_languages) > 1 and strict_conflicts:
                raise ValueError(
                    f"Word '{word}' found in multiple languages: {', '.join(unique_languages)}. "
                    "Use get_encoding() with language_map parameter to specify target language."
                )
            elif len(unique_languages) >= 1:
                # Use the first/only language found
                language_map[word.lower()] = list(unique_languages)[0].upper()

        return language_map

    def _get_encoding_unified(
        self,
        idx: int | slice | str | list[str],
        language_map: dict[str, str] | None = None,
        strict_conflicts: bool = True,
    ) -> BridgeEncoding:
        """
        Unified method to get encodings for any index type.

        Args:
            idx: Index specification (int, slice, str, or list[str])
            language_map: Optional explicit language mapping
            strict_conflicts: If True, raise errors on language conflicts

        Returns:
            BridgeEncoding object

        Raises:
            IndexError: If int index out of range
            TypeError: If idx is unsupported type
            KeyError: If string not found in dataset
            ValueError: If language conflicts and strict_conflicts=True
            RuntimeError: If encoding fails
        """
        # Convert index to list of words and resolve languages
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.words):
                raise IndexError(f"Index {idx} out of range [0, {len(self.words)})")
            words = [self.words[idx]]

            # For single index, use the corresponding language directly
            if language_map is None and self.languages:
                lang = self.languages[idx]
                resolved_language_map = {words[0].lower(): lang.upper()}
            else:
                resolved_language_map = language_map or {}

        elif isinstance(idx, slice):
            words = self.words[idx]

            # For slice, use the corresponding languages directly (no conflict checking needed)
            if language_map is None and self.languages:
                selected_langs = self.languages[idx]
                resolved_language_map = dict(
                    zip(
                        [w.lower() for w in words],
                        [l.upper() for l in selected_langs],
                    )
                )
            else:
                resolved_language_map = language_map or {}

        elif isinstance(idx, str):
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

        # Get encoding
        if len(words) == 1:
            # Single word - preserve original behavior
            encoding = self._get_encoding(words[0], resolved_language_map)
        else:
            # Multiple words - batch encoding
            encoding = self._get_encoding(words, resolved_language_map)

        if encoding is None:
            word_list = ", ".join(words)
            raise RuntimeError(f"Failed to encode word(s): {word_list}")

        return encoding

    def __getitem__(self, idx: int | slice | str | list[str]) -> BridgeEncoding:
        """
        Retrieve encoded data for specified index or slice.
        Uses dataset's built-in language mapping with strict conflict checking.

        Args:
            idx: Can be int, slice, str, or list[str]

        Returns:
            BridgeEncoding object

        Raises:
            ValueError: If word found in multiple languages (directs to get_encoding)
        """
        return self._get_encoding_unified(idx, language_map=None, strict_conflicts=True)

    def get_encoding(
        self,
        idx: int | slice | str | list[str],
        language_map: dict[str, str] | None = None,
    ) -> BridgeEncoding:
        """
        Retrieve encoded data with optional explicit language mapping.
        Allows language_map override and handles conflicts gracefully.

        Args:
            idx: Can be int, slice, str, or list[str]
            language_map: Optional mapping of words to languages

        Returns:
            BridgeEncoding object
        """
        return self._get_encoding_unified(
            idx, language_map=language_map, strict_conflicts=False
        )

    def shuffle(self, cutoff: int) -> None:
        """
        Shuffle the dataset up to the specified cutoff point.
        Maintains cache consistency during shuffling.

        Args:
            cutoff: Index to shuffle up to (exclusive)
        """
        if cutoff > len(self.words):
            raise ValueError(f"Cutoff {cutoff} exceeds dataset size {len(self.words)}")

        # Store original order for validation
        original_words = self.words.copy()

        # Create indices and shuffle them to maintain word-language alignment
        indices = list(range(cutoff))
        random.shuffle(indices)

        # Apply the same shuffle to both words and languages
        shuffled_words = [self.words[i] for i in indices] + self.words[cutoff:]
        shuffled_languages = [self.languages[i] for i in indices] + self.languages[
            cutoff:
        ]

        self.words = shuffled_words
        self.languages = shuffled_languages

        # Validate no words were lost
        assert len(self.words) == len(original_words)
        assert set(self.words) == set(original_words)

        # Clear cache since order changed
        self.encoding_cache.clear()
