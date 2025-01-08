import os
import pickle
import random
import torch
import pandas as pd
from typing import List, Union
from torch.utils.data import Dataset
from src.domain.datamodels import DatasetConfig
from src.domain.dataset import Phonemizer, CharacterTokenizer
import logging

logger = logging.getLogger(__name__)


class BridgeDataset(Dataset):
    """
    BridgeDataset for Matt's Phonological Feature Vectors.
    Uses (31, 32, 33) to represent ('[BOS]', '[EOS]', '[PAD]').
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        device: str,
        cache_path: str = "data/.cache",
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
        self.device = torch.device(device)

        # Load input data containing orthographic and phonologic
        # representations of words to be used during training
        input_data = self.read_orthographic_phonologic_data()
        self.words = sorted(input_data.keys())

        # Initialize phonemizer and character tokenizer
        self.phonemizer = self.read_phonology_data(input_data)
        self.dataset_config.phonological_vocabulary_size = (
            self.phonemizer.get_vocabulary_size()
        )
        list_of_characters = sorted(set(c for word in self.words for c in word))
        self.character_tokenizer = CharacterTokenizer(list_of_characters)
        self.dataset_config.orthographic_vocabulary_size = (
            self.character_tokenizer.get_vocabulary_size()
        )

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
            phonemizer = Phonemizer(input_data, self.dataset_config.dimension_phon_repr)
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

                # Update max phonological and orthographic sequence lengths
                max_phon_seq_len = max(
                    max_phon_seq_len, len(phonology["enc_pad_mask"][0])
                )
                max_orth_seq_len = max(
                    max_orth_seq_len,
                    len(self.character_tokenizer.encode(word)["enc_input_ids"][0]),
                )

        # Save updated max sequence lengths to the configuration
        self.dataset_config.max_phon_seq_len = max_phon_seq_len
        self.dataset_config.max_orth_seq_len = max_orth_seq_len
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

    def encode(
        self, content_to_encode: Union[str, List[str]]
    ) -> dict[str, dict[str, List[int]]]:
        """
        Encodes orthographic and phonological data for given content.

        Args:
            content_to_encode (Union[str, List[str]]): Content to encode; can be a single word or a list of words.

        Returns:
            dict[str, dict[str, List[int]]]: Encoded orthographic and phonological data.
        """
        logger.info(f"Encoding orthography and phonology.")
        if isinstance(content_to_encode, str):
            content_to_encode = [content_to_encode]  # Ensure it is wrapped in a list

        orth_tokenized = self.character_tokenizer.encode(content_to_encode).to(
            device=self.device
        )
        phon_tokenized = self.phonemizer.encode(content_to_encode).to(
            device=self.device
        )

        if phon_tokenized is None:
            raise ValueError(
                f"Phonology encoding failed for input: {content_to_encode}"
            )

        logger.info(f"Encoding done, data moved to device.")
        return {"orthography": orth_tokenized, "phonology": phon_tokenized}

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
