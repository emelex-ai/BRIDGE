import os
import pickle
import random
import torch
import pandas as pd
from typing import List, Union
from torch.utils.data import Dataset
from src.domain.datamodels import DatasetConfig
from src.domain.datamodels.encodings import (
    OrthographicEncoding,
    PhonologicalEncoding,
    BridgeEncoding,
)
from src.domain.dataset import Phonemizer, CharacterTokenizer
import logging
from typing import Literal


logger = logging.getLogger(__name__)


class BridgeDataset(Dataset):
    """
    BridgeDataset for Matt's Phonological Feature Vectors.
    Uses (31, 32, 33) to represent ('[BOS]', '[EOS]', '[PAD]').
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        device: str | None = None,
        cache_path: str | None = "data/.cache",
    ):
        """
        Initializes the dataset, precomputes encodings, and loads data onto the specified device.

        Args:
            dataset_config (DatasetConfig): Configuration object for the dataset.
            cache_path (str): Directory path to store or load cached phonology data.
        """
        self.cache_path = cache_path
        self.dataset_config = dataset_config
        self.dataset_filepath = dataset_config.dataset_filepath
        self.device = torch.device(device) if device else "cpu"

        # Load input data containing orthographic and phonologic
        # representations of words to be used during training
        input_data = self.read_orthographic_phonologic_data()
        self.words = sorted(input_data.keys())

        # Initialize phonemizer and character tokenizer
        self.phonemizer = self.read_phonology_data(input_data)
        list_of_characters = sorted(set(c for word in self.words for c in word))
        self.character_tokenizer = CharacterTokenizer(list_of_characters)

        # Finalize word data to filter and determine max sequence lengths
        self.finalize_word_data()

        # Precompute all encodings and transfer them to the specified device
        self.data = self.encode(self.words)

    def _validate_dataset_config(
        self,
        max_phon_seq_len: int,
        max_orth_seq_len: int,
        phon_vocab_size: int,
        orth_vocab_size: int,
    ) -> None:
        """
        Validate and update the DatasetConfig based on the provided values.

        This function updates the corresponding values in the DatasetConfig if they are currently set to None.
        It also checks if any of the provided values exceed the maximum values defined in the DatasetConfig.

        Args:
            max_phon_seq_len (int): The maximum phonological sequence length in the dataset.
            max_orth_seq_len (int): The maximum orthographic sequence length in the dataset.
            phon_vocab_size (int): The size of the phonological vocabulary.
            orth_vocab_size (int): The size of the orthographic vocabulary.

        Raises:
            ValueError: If any of the provided values exceed the corresponding maximum values defined in the DatasetConfig.
        """
        # Update the DatasetConfig values if they are currently set to None
        # Pydantic model already validates input data...
        if self.dataset_config.max_phon_seq_len is None:
            self.dataset_config.max_phon_seq_len = max_phon_seq_len
        if self.dataset_config.max_orth_seq_len is None:
            self.dataset_config.max_orth_seq_len = max_orth_seq_len
        if self.dataset_config.phonological_vocabulary_size is None:
            self.dataset_config.phonological_vocabulary_size = phon_vocab_size
        if self.dataset_config.orthographic_vocabulary_size is None:
            self.dataset_config.orthographic_vocabulary_size = orth_vocab_size

        # Check if any of the provided values exceed the maximum values defined in the DatasetConfig
        if max_phon_seq_len > self.dataset_config.max_phon_seq_len:
            raise ValueError(
                f"Maximum phonological sequence length in dataset ({max_phon_seq_len}) "
                f"exceeds the maximum defined in DatasetConfig ({self.dataset_config.max_phon_seq_len})."
            )
        if max_orth_seq_len > self.dataset_config.max_orth_seq_len:
            raise ValueError(
                f"Maximum orthographic sequence length in dataset ({max_orth_seq_len}) "
                f"exceeds the maximum defined in DatasetConfig ({self.dataset_config.max_orth_seq_len})."
            )
        if (
            phon_vocab_size - len(self.phonemizer.extra_token)
            > self.dataset_config.phonological_vocabulary_size
        ):
            raise ValueError(
                f"Phonological vocabulary size ({phon_vocab_size}) "
                f"exceeds the size defined in DatasetConfig ({self.dataset_config.phonological_vocabulary_size})."
            )
        if orth_vocab_size > self.dataset_config.orthographic_vocabulary_size:
            raise ValueError(
                f"Orthographic vocabulary size ({orth_vocab_size}) "
                f"exceeds the size defined in DatasetConfig ({self.dataset_config.orthographic_vocabulary_size})."
            )

    def read_orthographic_phonologic_data(self) -> dict:
        """
        Reads orthographic and phonologic from the input dataset.

        Returns:
            dict[str, WordData]: Dictionary with words as keys and their processed data as values.
                Each word's data includes count, phoneme representation, phoneme shape, and orthographic representation.
                {
                    "word": {
                        count: int,
                        phoneme: tuple[np.array, np.array],  # result of np.where
                        phoneme_shape: tuple[int, int],
                        orthography: np.array
                    },
                    "word2": {}
                    ...
                }
        """
        with open(self.dataset_filepath, "rb") as f:
            input_data = pickle.load(f)

        return input_data

    def read_phonology_data(self, input_data: dict) -> Phonemizer:
        """
        Reads or creates a phonology tokenizer from cached data, saving to cache if created.

        Args:
            words (pd.Series): Series of words to encode.

        Returns:
            Phonemizer: Initialized phonemizer object.
        """
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path, exist_ok=True)

        dataset_name = os.path.splitext(os.path.basename(self.dataset_filepath))[0]
        cache_file = f"{dataset_name}_phonology.pkl"
        cache_path = os.path.join(self.cache_path, cache_file)

        # Load phonemizer from cache if available, otherwise initialize and cache it
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                phonemizer = pickle.load(f)
            logger.info(f"Phonemizer data loaded from: {cache_path}")
        else:
            phonemizer = Phonemizer(input_data, self.dataset_config)
            with open(cache_path, "wb") as f:
                pickle.dump(phonemizer, f)
            logger.info(f"Created cache folder for phonemizer: {cache_path}")

        return phonemizer

    def finalize_word_data(self) -> None:
        """
        Processes words to filter out invalid entries and calculates maximum sequence lengths for orthography and phonology.
        """
        logger.info("Finalizing words list.")
        final_words = []
        max_phon_seq_len = 0
        max_orth_seq_len = 0

        for word in self.words:
            # Skip empty or invalid entries
            if not word:
                continue

            # Encode phonology; if valid, update sequence lengths
            phonology = self.phonemizer.encode([word])
            if phonology:  # Ensure the word is valid in the phoneme dictionary
                final_words.append(word)

                # Calculate max phonological and orthographic sequence lengths
                max_phon_seq_len = max(
                    max_phon_seq_len, len(phonology["enc_pad_mask"][0])
                )
                max_orth_seq_len = max(
                    max_orth_seq_len,
                    len(self.character_tokenizer.encode(word)["enc_input_ids"][0]),
                )

        phon_vocab_size = self.phonemizer.get_vocabulary_size()
        orth_vocab_size = self.character_tokenizer.get_vocabulary_size()

        self._validate_dataset_config(
            max_phon_seq_len, max_orth_seq_len, phon_vocab_size, orth_vocab_size
        )
        self.words = final_words

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Number of words in the dataset.
        """
        return len(self.words)

    def __getitem__(
        self, idx: Union[int, slice, str]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Retrieves precomputed data for the given index.

        Args:
            idx (Union[int, slice, str]): Index of the item to retrieve.
                - int: Single word
                - slice: Range of words
                - str: Specific word (must be in the dataset)

        Returns:
            dict[str, dict[str, torch.Tensor]]: Encoded orthographic and phonological data for the specified index.
        """
        if isinstance(idx, int):
            # Wrap single index in a slice to retrieve one item
            if idx < 0 or idx >= len(self.words):
                raise IndexError(f"Index {idx} out of range: [0,{len(self.words)}].")
            return {
                k: {sub_k: sub_v[idx : idx + 1] for sub_k, sub_v in v.items()}
                for k, v in self.data.items()
            }
        elif isinstance(idx, slice):
            # Return slice of data
            return {
                k: {sub_k: sub_v[idx] for sub_k, sub_v in v.items()}
                for k, v in self.data.items()
            }
        elif isinstance(idx, str):
            if idx not in self.words:
                raise ValueError(f'Word "{idx}" not found in the dataset.')
            word_index = self.words.index(idx)
            return self.__getitem__(word_index)
        else:
            raise TypeError("Index must be an int, slice, or string.")

    def encode(self, words: Union[str, list[str]]) -> BridgeEncoding:
        """
        Encodes a list of words into both orthographic and phonological representations.

        This method provides a unified interface to encode words for both modalities,
        properly packaging the outputs into validated Pydantic models. It ensures
        consistency between the orthographic and phonological representations,
        including matching batch sizes and device placement.

        Args:
            words: A word or list of words to encode. Words should be a valid string that
                exists in the phonological vocabularies (CMUDict).

        Returns:
            BridgeEncoding containing both orthographic and phonological encodings
            in their respective Pydantic model containers.

        Raises:
            ValueError: If any word is not found in the phonological dictionary
            TypeError: If input is not a list of strings
        """
        # Input validation
        if isinstance(words, str):
            words = [words]
        if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
            raise TypeError("Input must be a list of strings")

        # Encode using both tokenizers
        orthographic_encoding = self.character_tokenizer.encode(words)
        phonological_encoding = self.phonemizer.encode(words)

        # Validate phonological encoding succeeded
        if phonological_encoding is None:
            failed_words = [
                w for w in words if self.phonemizer.enc_inputs.get(w) is None
            ]
            raise ValueError(
                f"Phonological encoding failed for words: {failed_words}. "
                "These words were not found in the phonological dictionary."
            )

        # Package into Pydantic models
        orthographic = OrthographicEncoding(
            enc_input_ids=orthographic_encoding["enc_input_ids"],
            enc_pad_mask=orthographic_encoding["enc_pad_mask"],
            dec_input_ids=orthographic_encoding["dec_input_ids"],
            dec_pad_mask=orthographic_encoding["dec_pad_mask"],
        )

        phonological = PhonologicalEncoding(
            enc_input_ids=phonological_encoding["enc_input_ids"],
            enc_pad_mask=phonological_encoding["enc_pad_mask"],
            dec_input_ids=phonological_encoding["dec_input_ids"],
            dec_pad_mask=phonological_encoding["dec_pad_mask"],
            targets=phonological_encoding["targets"],
        )

        # Create and return combined output
        # The BridgeEncoding constructor will validate consistency
        # between orthographic and phonological components
        return BridgeEncoding(orthographic=orthographic, phonological=phonological)

    def shuffle(self, cutoff: int):
        """Split the data by the cutoff point, shuffle the elements before the cutoff point, and reassemble the data"""
        data_items = list(self.data.items())
        shuffled_data = data_items[:cutoff]
        random.shuffle(shuffled_data)
        data_items = shuffled_data + data_items[cutoff:]
        # Recreate the data dict
        self.data = {
            k: {sub_k: sub_v for sub_k, sub_v in v.items()} for k, v in data_items
        }
