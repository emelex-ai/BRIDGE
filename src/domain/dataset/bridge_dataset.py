import os
import pickle
import random
import torch
import pandas as pd
from typing import Union
from torch.utils.data import Dataset
from src.domain.datamodels import DatasetConfig
from src.domain.datamodels.encodings import (
    OrthographicEncoding,
    PhonologicalEncoding,
    BridgeEncoding,
)
from src.domain.dataset import PhonemeTokenizer, CharacterTokenizer
import logging
from typing import Literal


logger = logging.getLogger(__name__)


class BridgeDataset(Dataset):

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
            device (str, optional): Device to load the data onto. Defaults to None.
            cache_path (str, optional): Directory path to store or load cached phonology data.
        """
        self.cache_path = cache_path
        self.dataset_config = dataset_config
        self.dataset_filepath = dataset_config.dataset_filepath
        self.device = torch.device(device) if device else "cpu"

        # Load input data containing orthographic and phonologic
        # representations of words to be used during training
        input_data = self.read_orthographic_phonologic_data()
        self.words = sorted(input_data.keys())

        # Initialize phoneme and character tokenizer
        self.phoneme_tokenizer = self.read_phonology_data(input_data)
        self.character_tokenizer = CharacterTokenizer(device=self.device)

        # Finalize word data to filter and determine max sequence lengths
        self.finalize_word_data()

        # Precompute all encodings and transfer them to the specified device
        self.data = self.encode(self.words)

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

    def read_phonology_data(self, input_data: dict) -> PhonemeTokenizer:
        """
        Reads or creates a phonology tokenizer from cached data, saving to cache if created.

        Args:
            words (pd.Series): Series of words to encode.

        Returns:
            PhonemeTokenizer: Initialized phoneme tokenizer object.
        """
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path, exist_ok=True)

        dataset_name = os.path.splitext(os.path.basename(self.dataset_filepath))[0]
        cache_file = f"{dataset_name}_phonology.pkl"
        cache_path = os.path.join(self.cache_path, cache_file)

        # Load phoneme tokenizer from cache if available, otherwise initialize and cache it
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                phoneme_tokenizer = pickle.load(f)
            logger.info(f"PhonemeTokenizer data loaded from: {cache_path}")
        else:
            phoneme_tokenizer = PhonemeTokenizer(device=self.device)
            with open(cache_path, "wb") as f:
                pickle.dump(phoneme_tokenizer, f)
            logger.info(f"Created cache folder for phoneme_tokenizer: {cache_path}")

        return phoneme_tokenizer

    def finalize_word_data(self) -> None:
        """
        Processes words to filter out invalid entries and calculates maximum sequence lengths for orthography and phonology.
        """
        logger.info("Finalizing words list.")
        final_words = []

        for word in self.words:
            # Skip empty or invalid entries
            if not word:
                continue

            # Encode phonology; if valid, update sequence lengths
            phonology = self.phoneme_tokenizer.encode([word])
            if phonology:  # Ensure the word is valid in the phoneme dictionary
                final_words.append(word)

        phon_vocab_size = self.phoneme_tokenizer.get_vocabulary_size()
        orth_vocab_size = self.character_tokenizer.get_vocabulary_size()

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
        phonological_encoding = self.phoneme_tokenizer.encode(words)

        # Validate phonological encoding succeeded
        if phonological_encoding is None:
            failed_words = [
                w for w in words if self.phoneme_tokenizer.enc_inputs.get(w) is None
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
