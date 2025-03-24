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
import torch
from collections import OrderedDict
from functools import lru_cache

from src.domain.datamodels import DatasetConfig, BridgeEncoding
from src.utils.device_manager import device_manager
from src.domain.dataset import BridgeTokenizer

logger = logging.getLogger(__name__)


class BridgeDataset:
    """
    Dataset class managing orthographic and phonological representations of words.
    Uses BridgeTokenizer for unified tokenization and BridgeEncoding for data storage.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
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

        # Initialize device - prioritize config device over parameter
        self.device = device_manager.device

        # Initialize tokenizer with matching device
        self.tokenizer = BridgeTokenizer(
            phoneme_cache_size=getattr(dataset_config, "tokenizer_cache_size", 10000),
        )

        # Store vocabulary sizes for model initialization
        vocab_sizes = self.tokenizer.get_vocabulary_sizes()
        self.orthographic_vocabulary_size = vocab_sizes["orthographic"]
        self.phonological_vocabulary_size = vocab_sizes["phonological"]

        # Setup LRU cache for encodings
        self.encoding_cache = OrderedDict()
        self.max_cache_size = cache_size

        # Load and process the raw data
        self.dataset_filepath = dataset_config.dataset_filepath
        self._load_and_process_data()

        # Create cache directory if needed
        if cache_path and not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)

    def _load_and_process_data(self) -> None:
        """
        Load raw data from file and process it using the BridgeTokenizer.
        Implements efficient data loading and validation.
        """
        logger.info(f"Loading data from {self.dataset_filepath}")
        try:
            with open(self.dataset_filepath, "rb") as f:
                raw_data = pickle.load(f)

            if not isinstance(raw_data, dict):
                raise ValueError("Dataset file must contain a dictionary")

            # Validate and sort words for deterministic ordering
            valid_words = []
            for word in sorted(raw_data.keys()):
                if not isinstance(word, str):
                    logger.warning(f"Skipping invalid word entry: {word}")
                    continue

                # Pre-encode first word to validate structure
                if not valid_words:
                    test_encoding = self._encode_single_word(word)
                    if test_encoding is None:
                        logger.warning("Initial encoding validation failed")
                        continue

                valid_words.append(word)

            self.words = valid_words
            logger.info(f"Successfully loaded {len(self.words)} valid words")

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

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

    def __getitem__(self, idx: Union[int, slice, str]) -> dict[str, dict[str, Any]]:
        """
        Retrieve encoded data for specified index or slice.
        Maintains compatibility with training pipeline expectations.

        Args:
            idx: Can be:
                - int: Single word index
                - slice: Range of word indices
                - str: Specific word

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

            # Convert single encoding to batch format
            return encoding.to_dict()

        elif isinstance(idx, slice):
            selected_words = self.words[idx]
            encodings = []

            encodings = self._get_encoding(selected_words)
            if encodings is None:
                raise RuntimeError(f"Failed to encode word: {word}")

            if not encodings:
                raise ValueError("No valid encodings in slice")

            # Merge encodings into batch
            return encodings.to_dict()

        elif isinstance(idx, str):
            if idx not in self.words:
                raise KeyError(f"Word '{idx}' not found in dataset")

            encoding = self._get_encoding(idx)
            if encoding is None:
                raise RuntimeError(f"Failed to encode word: {idx}")

            return encoding.to_dict()

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
