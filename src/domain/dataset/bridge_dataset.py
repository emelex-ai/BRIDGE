"""
BridgeDataset: A dataset class for managing orthographic and phonological data using
the unified BridgeEncoding dataclass and BridgeTokenizer for processing.
"""

import os
import pickle
import random
import logging
from typing import Union, Any
from pathlib import Path
from collections import OrderedDict
from functools import lru_cache
import pandas as pd
from src.domain.datamodels import DatasetConfig, BridgeEncoding
from src.utils.device_manager import device_manager
from src.domain.dataset import BridgeTokenizer
from src.infra.clients.gcp.gcs_client import GCSClient

logger = logging.getLogger(__name__)


class BridgeDataset:
    """
    Dataset class managing orthographic and phonological representations of words.
    Uses BridgeTokenizer for unified tokenization and BridgeEncoding for data storage.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        gcs_client: GCSClient,
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
        self.words = self._process_raw_dataframe(raw_df)
        logger.info(f"Loaded {len(self.words)} valid words.")

        # Create cache directory if needed
        if cache_path and not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)

    def _load_raw_dataframe(self, path: str) -> pd.DataFrame:
        """
        Dispatch method to load raw data based on path scheme or extension.
        Returns a DataFrame with a 'word_raw' column.
        """
        if path.startswith("gs://"):
            # strip gs://<bucket>/ prefix
            parts = path.replace("gs://", "").split("/", 1)
            bucket, blob = parts[0], parts[1]
            return self._read_gcs_csv(bucket, blob)
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path)
        if ext == ".pkl":
            data = pickle.load(open(path, "rb"))
            if not isinstance(data, dict):
                raise ValueError("Dataset file must contain a dictionary")
            # convert dict keys to DataFrame
            return pd.DataFrame({"word_raw": list(data.keys())})
        raise ValueError(f"Unsupported data format: {ext}")

    def _read_gcs_csv(self, bucket: str, blob: str) -> pd.DataFrame:
        """Read a CSV blob from GCS into a DataFrame."""
        return self.gcs_client.read_csv(
            bucket_name=bucket,
            blob_name=blob,
            sep=",",
            parse_dates=True,
            dtype={"col1": str},
        )

    def _process_raw_dataframe(self, df: pd.DataFrame) -> list[str]:
        """
        Validate and filter 'word_raw' entries, ensuring tokenization consistency.
        """
        if "word_raw" not in df.columns:
            raise KeyError("DataFrame must contain 'word_raw' column")
        valid = []
        for idx, word in enumerate(df["word_raw"]):
            if not isinstance(word, str):
                logger.warning(f"Skipping non-str entry: {word}")
                continue
            if idx == 0:
                # test encoding of first valid word
                if self._encode_single_word(word) is None:
                    logger.warning(f"Initial encoding validation failed for {word}")
                    continue
            valid.append(word)
        return valid

    @lru_cache(maxsize=128)
    def _encode_single_word(self, word: str) -> BridgeEncoding | None:
        """
        Encode a single word using the tokenizer with caching.

        Ensures that all encoded sequences use the configured max sequence lengths
        to maintain consistency across batches.

        Args:
            word: Word to encode

        Returns:
            BridgeEncoding object or None if encoding fails
        """
        try:
            # Use the tokenizer directly for initial encoding
            encoding = self.tokenizer.encode(word)
            if encoding is None:
                return None

            # Ensure encoding is on correct device
            encoding = encoding.to(self.device)

            return encoding
        except Exception as e:
            logger.error(f"Error encoding word '{word}': {e}")
            return None

    def _get_encoding_batch(self, words: list[str]) -> BridgeEncoding:
        """
        Get batched encodings for a list of words without caching.

        Uses the tokenizer's batch encode() method to process the list of words.
        Then, each encoding is extracted (using __getitem__) and converted to a BridgeEncoding
        via from_dict.

        Args:
            words: List of words to encode.

        Returns:
            List of BridgeEncoding objects.
        """
        batch_encoding = self.tokenizer.encode(words)
        if batch_encoding is None:
            raise RuntimeError("Batch encoding failed for words: " + ", ".join(words))

        return batch_encoding

    # TODO: Work on encoding cache logic
    def _get_encoding(self, word: Union[str, list[str]]) -> BridgeEncoding | None:
        """
        Get encoding for a word, using cache when available.

        Args:
            word: Word to retrieve encoding for

        Returns:
            BridgeEncoding object or None if not available
        """
        # Check in-memory cache first
        # if word in self.encoding_cache:
        #     return self.encoding_cache[word]

        # Encode word if not in cache
        if isinstance(word, str):
            return self._encode_single_word(word)
        elif isinstance(word, list):
            return self._get_encoding_batch(word)
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

    def __getitem__(
        self, idx: Union[int, slice, str, list[str]]
    ) -> dict[str, dict[str, Any]]:
        """
        Retrieve encoded data for specified index or slice.
        Maintains compatibility with training pipeline expectations.

        Args:
            idx: Can be:
                - int: Single word index
                - slice: Range of word indices
                - str: Specific word
                - list[str]: List of words:w

        Returns:
            Dictionary containing orthographic and phonological encodings
        """
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.words):
                raise IndexError(f"Index {idx} out of range [0, {len(self.words)})")

            word = self.words[idx]
            encoding = self._get_encoding(word)
            if encoding is None:
                raise RuntimeError(f"Failed to encode word: {word}")

            return encoding

        elif isinstance(idx, slice):
            selected_words = self.words[idx]
            encodings = []

            encodings = self._get_encoding(selected_words)
            if encodings is None:
                raise RuntimeError(f"Failed to encode word: {word}")

            if not encodings:
                raise ValueError("No valid encodings in slice")

            return encodings

        elif isinstance(idx, str):
            if idx not in self.words:
                raise KeyError(f"Word '{idx}' not found in dataset")

            encoding = self._get_encoding(idx)
            if encoding is None:
                raise RuntimeError(f"Failed to encode word: {idx}")

            return encoding

        elif isinstance(idx, list):
            if not all(isinstance(i, str) for i in idx):
                raise TypeError("List indices must be strings")
            encodings = self._get_encoding(idx)
            if encodings is None:
                raise RuntimeError(f"Failed to encode words: {', '.join(idx)}")

            return encodings

        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

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

        # Shuffle words up to cutoff
        shuffled_section = self.words[:cutoff]
        random.shuffle(shuffled_section)
        self.words = shuffled_section + self.words[cutoff:]

        # Validate no words were lost
        assert len(self.words) == len(original_words)
        assert set(self.words) == set(original_words)

        # Clear cache since order changed
        self.encoding_cache.clear()
