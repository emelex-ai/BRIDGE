"""
BridgeDataset: A dataset class for managing orthographic and phonological data using
the unified BridgeEncoding dataclass and BridgeTokenizer for processing.
"""

import os
import pickle
import random
import logging
from typing import Union, Optional, Dict, Any
from pathlib import Path
import torch
from collections import OrderedDict
from functools import lru_cache

from src.domain.datamodels import DatasetConfig, BridgeEncoding
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
        device: Optional[str] = None,
        cache_path: Optional[str] = "data/.cache",
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
        self.device = torch.device(
            dataset_config.device
            if hasattr(dataset_config, "device")
            else (device if device else "cpu")
        )

        # Initialize tokenizer with matching device
        self.tokenizer = BridgeTokenizer(
            device=self.device,
            phoneme_cache_size=getattr(dataset_config, "phoneme_cache_size", 10000),
        )

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
    def _encode_single_word(self, word: str) -> Optional[BridgeEncoding]:
        """
        Encode a single word using the tokenizer with caching.

        Args:
            word: Word to encode

        Returns:
            BridgeEncoding object or None if encoding fails
        """
        try:
            encoding = self.tokenizer.encode(word)
            if encoding is not None:
                # Ensure encoding is on correct device
                return encoding.to(self.device)
            return None
        except Exception as e:
            logger.error(f"Error encoding word '{word}': {e}")
            return None

    def _get_encoding(self, word: str) -> Optional[BridgeEncoding]:
        """
        Get encoding for a word, using cache when available.

        Args:
            word: Word to retrieve encoding for

        Returns:
            BridgeEncoding object or None if not available
        """
        # Check in-memory cache first
        if word in self.encoding_cache:
            return self.encoding_cache[word]

        # Encode word if not in cache
        encoding = self._encode_single_word(word)
        if encoding is not None:
            # Update cache with LRU policy
            if len(self.encoding_cache) >= self.max_cache_size:
                self.encoding_cache.popitem(last=False)
            self.encoding_cache[word] = encoding

        return encoding

    def __len__(self) -> int:
        """Return the number of valid words in the dataset."""
        return len(self.words)

    def __getitem__(self, idx: Union[int, slice, str]) -> Dict[str, Dict[str, Any]]:
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

            for word in selected_words:
                encoding = self._get_encoding(word)
                if encoding is None:
                    raise RuntimeError(f"Failed to encode word: {word}")
                encodings.append(encoding)

            if not encodings:
                raise ValueError("No valid encodings in slice")

            # Merge encodings into batch
            return self._format_batch_encodings(encodings)

        elif isinstance(idx, str):
            if idx not in self.words:
                raise KeyError(f"Word '{idx}' not found in dataset")

            encoding = self._get_encoding(idx)
            if encoding is None:
                raise RuntimeError(f"Failed to encode word: {idx}")

            return encoding.to_dict()

        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def _format_batch_encodings(
        self, encodings: list[BridgeEncoding]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Format a list of BridgeEncodings into training pipeline batch format.

        Args:
            encodings: List of BridgeEncoding objects

        Returns:
            Dictionary with batched tensor data
        """
        if not encodings:
            raise ValueError("No encodings to format")

        # Convert first encoding to dict to initialize structure
        batch = encodings[0].to_dict()

        # Stack subsequent encodings
        for encoding in encodings[1:]:
            enc_dict = encoding.to_dict()

            # Stack orthographic tensors
            for key in batch["orthographic"]:
                if isinstance(batch["orthographic"][key], torch.Tensor):
                    batch["orthographic"][key] = torch.cat(
                        [batch["orthographic"][key], enc_dict["orthographic"][key]]
                    )

            # Stack phonological tensors
            for key in batch["phonological"]:
                if key in ["enc_input_ids", "dec_input_ids"]:
                    # These are lists of lists of tensors
                    batch["phonological"][key].extend(enc_dict["phonological"][key])
                else:
                    batch["phonological"][key] = torch.cat(
                        [batch["phonological"][key], enc_dict["phonological"][key]]
                    )

        return batch

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

    def to(self, device: torch.device) -> None:
        """
        Move dataset to specified device.
        Updates both tokenizer and cached encodings.

        Args:
            device: Target device
        """
        if device == self.device:
            return

        self.device = device
        self.tokenizer = self.tokenizer.to(device)

        # Move cached encodings to new device
        new_cache = OrderedDict()
        for word, encoding in self.encoding_cache.items():
            new_cache[word] = encoding.to(device)
        self.encoding_cache = new_cache
