import os
import pickle
import pandas as pd
from typing import List, Union
from torch.utils.data import Dataset
from src.domain.dataset import CUDADict, CharacterTokenizer, Phonemizer
from src.domain.datamodels import DatasetConfig


class ConnTextULDataset(Dataset):
    """
    ConnTextULDataset for Matt's Phonological Feature Vectors.
    Uses (31, 32, 33) to represent ('[BOS]', '[EOS]', '[PAD]').
    """

    def __init__(self, config: DatasetConfig, cache_path: str):
        self.cache_path = cache_path
        self.dataset_filepath = config.dataset_filepath
        self.words = self.read_orthographic_data()
        self.phonology_tokenizer = self.read_phonology_data(self.words)

        list_of_characters = sorted(set(c for word in self.words for c in word))
        self.character_tokenizer = CharacterTokenizer(list_of_characters)

        self.max_orth_seq_len = 0
        self.max_phon_seq_len = 0
        self.finalize_word_data()

        self.cmudict = self.phonology_tokenizer.traindata.cmudict
        self.convert_numeric_prediction = self.phonology_tokenizer.traindata.convert_numeric_prediction

    def read_orthographic_data(self) -> pd.Series:
        """Reads and cleans orthographic data (word list) from the dataset."""
        dataset = pd.read_csv(self.dataset_filepath)
        dataset.dropna(subset=["word_raw"], inplace=True)
        return dataset["word_raw"].str.lower()

    def read_phonology_data(self, words: pd.Series) -> Phonemizer:
        """Reads or creates a phonology tokenizer from cached data."""
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path, exist_ok=True)

        dataset_name = os.path.splitext(os.path.basename(self.dataset_filepath))[0]
        cache_file = f"{dataset_name}_phonology.pkl"
        cache_path = os.path.join(self.cache_path, cache_file)

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                phonology_tokenizer = pickle.load(f)
        else:
            phonology_tokenizer = Phonemizer(words.tolist())
            with open(cache_path, "wb") as f:
                pickle.dump(phonology_tokenizer, f)

        return phonology_tokenizer

    def finalize_word_data(self):
        """Processes words and calculates the maximum orthographic and phonological sequence lengths."""
        final_words = []
        for word in self.words:
            if not word:  # Skip empty strings
                continue

            phonology = self.phonology_tokenizer.encode([word])
            if phonology:  # Ensure the word is valid in the phoneme dict
                final_words.append(word)
                self.max_phon_seq_len = max(self.max_phon_seq_len, len(phonology["enc_pad_mask"][0]))
                self.max_orth_seq_len = max(
                    self.max_orth_seq_len, len(self.character_tokenizer.encode(word)["enc_input_ids"][0])
                )

        self.words = final_words

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, idx: Union[int, slice, str]):
        """
        Fetches the encoded data for a word or list of words.
        idx can be an int (single word), slice (range of words), or str (specific word).
        """
        if isinstance(idx, int):
            string_input = self.words[idx : idx + 1]  # Single word wrapped in a list
        elif isinstance(idx, slice):
            string_input = self.words[idx]  # Slice of words
        elif isinstance(idx, str):
            if idx not in self.words:
                raise ValueError(f'Word "{idx}" not found in the dataset.')
            string_input = [idx]  # Single word wrapped in a list
        else:
            raise TypeError("Index must be an int, slice, or string.")

        return self.encode(string_input)

    def encode(self, content_to_encode: dict) -> dict:
        """Encodes orthographic and phonological data for given content."""
        if isinstance(content_to_encode, str):
            content_to_encode = [content_to_encode]  # Ensure it is wrapped in a list

        orth_tokenized = self.character_tokenizer.encode(content_to_encode)
        phon_tokenized = self.phonology_tokenizer.encode(content_to_encode)

        if phon_tokenized is None:
            raise ValueError(f"Phonology encoding failed for input: {content_to_encode}")

        return {"orthography": orth_tokenized, "phonology": phon_tokenized}
