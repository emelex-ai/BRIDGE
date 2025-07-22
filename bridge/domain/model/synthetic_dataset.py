import random
import string
from typing import List, Optional

import torch

from bridge.domain.datamodels import DatasetConfig
from bridge.domain.datamodels.encodings import BridgeEncoding
from bridge.domain.dataset.bridge_dataset import BridgeDataset
from bridge.domain.dataset.bridge_tokenizer import BridgeTokenizer
from bridge.utils import device_manager


class SyntheticBridgeDataset(BridgeDataset):
    def __init__(self, num_samples: int = 100):
        """
        Initialize the synthetic dataset with a specified number of samples.

        Args:
            num_samples: Number of synthetic samples to generate.
        """
        self.device = device_manager.device
        self.tokenizer = BridgeTokenizer()

        # Get real words from the CMU dictionary
        self.real_words = self._get_real_words_from_cmu_dict()
        self.words = self._generate_synthetic_words(num_samples)

        # Set vocabulary sizes using the tokenizer
        vocab_sizes = self.tokenizer.get_vocabulary_sizes()
        self.orthographic_vocabulary_size = vocab_sizes["orthographic"]
        self.phonological_vocabulary_size = vocab_sizes["phonological"]

    def _get_real_words_from_cmu_dict(self) -> list[str]:
        """
        Get real words from the CMU dictionary that the tokenizer uses.

        Returns:
            List of real English words that can be properly encoded.
        """
        # Access the pronunciation dictionary from the phoneme tokenizer
        pronunciation_dict = self.tokenizer.phoneme_tokenizer.pronunciation_dict

        # Convert to list and filter for reasonable word lengths
        real_words = []
        for word in pronunciation_dict.keys():
            # Filter for words that are reasonable for testing (3-10 characters)
            if 3 <= len(word) <= 10 and word.isalpha():
                real_words.append(word)

        # Sort for reproducibility
        real_words.sort()

        print(f"Loaded {len(real_words)} real words from CMU dictionary")
        return real_words

    def _generate_synthetic_words(self, num_samples: int) -> list[str]:
        """Generate a list of synthetic words using real words from CMU dictionary.

        Args:
            num_samples: Number of synthetic words to generate.

        Returns:
            A list of real words randomly selected from CMU dictionary.
        """
        if not self.real_words:
            raise RuntimeError("No real words available from CMU dictionary")

        # Randomly sample from real words
        return random.sample(self.real_words, min(num_samples, len(self.real_words)))

    def __len__(self) -> int:
        """Return the number of synthetic words in the dataset."""
        return len(self.words)

    def __getitem__(self, idx: int) -> BridgeEncoding:
        """Retrieve encoded data for a specified index.

        Args:
            idx: Index of the word to retrieve.

        Returns:
            BridgeEncoding object for the word.
        """
        if idx < 0 or idx >= len(self.words):
            raise IndexError(f"Index {idx} out of range [0, {len(self.words)})")

        word = self.words[idx]

        # Encode the real word - this should always work since it's from CMU dict
        encoding = self.tokenizer.encode(word)
        if encoding is None:
            raise RuntimeError(f"Failed to encode word from CMU dictionary: {word}")

        return encoding


class SyntheticBridgeDatasetMultiWord(BridgeDataset):
    """Synthetic dataset that creates multi-word sequences respecting max_seq_len.

    This class generates sequences of multiple words that are concatenated together,
    ensuring the total sequence length (in tokens) does not exceed max_seq_len.
    """

    def __init__(
        self,
        num_samples: int = 100,
        max_seq_len: int = 128,
        min_words_per_sequence: int = 2,
        max_words_per_sequence: Optional[int] = None,
        word_length_range: tuple[int, int] = (3, 10),
        seed: Optional[int] = None,
    ):
        """Initialize the multi-word synthetic dataset.

        Args:
            num_samples: Number of sequence samples to generate.
            max_seq_len: Maximum sequence length in tokens (including BOS/EOS).
            min_words_per_sequence: Minimum number of words per sequence.
            max_words_per_sequence: Maximum number of words per sequence.
                                    If None, will be determined dynamically.
            word_length_range: Range of word lengths (min, max) in characters.
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)

        self.device = device_manager.device
        self.tokenizer = BridgeTokenizer()
        self.max_seq_len = max_seq_len
        self.min_words_per_sequence = min_words_per_sequence
        self.max_words_per_sequence = max_words_per_sequence
        self.word_length_range = word_length_range

        # Get real words from the CMU dictionary
        self.real_words = self._get_real_words_from_cmu_dict()

        # Generate multi-word sequences
        self.sequences = self._generate_multi_word_sequences(num_samples)

        # Set vocabulary sizes using the tokenizer
        vocab_sizes = self.tokenizer.get_vocabulary_sizes()
        self.orthographic_vocabulary_size = vocab_sizes["orthographic"]
        self.phonological_vocabulary_size = vocab_sizes["phonological"]

    def _get_real_words_from_cmu_dict(self) -> list[str]:
        """Get real words from the CMU dictionary that the tokenizer uses.

        Returns:
            List of real English words that can be properly encoded.
        """
        # Access the pronunciation dictionary from the phoneme tokenizer
        pronunciation_dict = self.tokenizer.phoneme_tokenizer.pronunciation_dict

        # Convert to list and filter for reasonable word lengths
        min_len, max_len = self.word_length_range
        real_words = []
        for word in pronunciation_dict.keys():
            # Filter for words that match the specified length range
            if min_len <= len(word) <= max_len and word.isalpha():
                real_words.append(word)

        # Sort for reproducibility
        real_words.sort()

        print(
            f"Loaded {len(real_words)} real words from CMU dictionary (length {min_len}-{max_len})"
        )
        return real_words

    def _generate_multi_word_sequences(self, num_samples: int) -> list[list[str]]:
        """Generate multi-word sequences that respect max_seq_len (in characters).

        Args:
            num_samples: Number of sequences to generate.

        Returns:
            List of word sequences, where each sequence is a list of words.
        """
        if not self.real_words:
            raise RuntimeError("No real words available from CMU dictionary")

        sequences = []

        for _ in range(num_samples):
            sequence = []
            current_length = 0
            # Add words until adding another would exceed max_seq_len (including spaces)
            while True:
                word = random.choice(self.real_words)
                # +1 for the space if not the first word
                add_length = len(word) if not sequence else len(word) + 1
                if current_length + add_length > self.max_seq_len:
                    break
                sequence.append(word)
                current_length += add_length
            # Ensure at least min_words_per_sequence
            # print(f"==> {len(sequence)=}")
            if len(sequence) < self.min_words_per_sequence:
                # If not enough, forcibly add the first N words (may exceed max_seq_len)
                raise RuntimeError(
                    "Unable to generate a sequence with the required minimum number of words "
                    f"({self.min_words_per_sequence}) without exceeding max_seq_len ({self.max_seq_len}). "
                    "Consider increasing max_seq_len or reducing min_words_per_sequence."
                )
            sequences.append(sequence)

        return sequences

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(
        self, idx: int | slice | str | list[str]
    ) -> BridgeEncoding | list[BridgeEncoding]:
        """
        Retrieve encoded data for specified index, slice, string, or list of strings.

        Args:
            idx: Can be:
                - int: Single sequence index
                - slice: Range of sequence indices
                - str: A multi-word string (must be in self.words)
                - list[str]: List of multi-word strings

        Returns:
            BridgeEncoding object or list of BridgeEncoding objects.
        """
        print(f"__getitem__, arg: {idx=}")
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.sequences):
                raise IndexError(f"Index {idx} out of range [0, {len(self.words)})")
            print(f"{self.sequences[idx]=}")
            text_sequence = " ".join(self.sequences[idx])
            encoding = self.tokenizer.encode(text_sequence)
            if encoding is None:
                raise RuntimeError(f"Failed to encode sequence: {text_sequence}")
            return encoding

        elif isinstance(idx, slice):
            # Convert the slice to a list of indices
            indices = range(*idx.indices(len(self.sequences)))
            if (
                not indices
                or min(indices) < 0
                or max(indices, default=-1) >= len(self.sequences)
            ):
                raise IndexError(
                    f"Slice {idx} is out of range for dataset of length {len(self.sequences)}"
                )
            selected_sequences = [self.sequences[i] for i in indices]
            if not selected_sequences:
                raise IndexError(
                    f"Slice {idx} results in an empty selection for dataset of length {len(self.sequences)}"
                )
            encoding = self.tokenizer.encode(selected_sequences)
            if encoding is None:
                raise RuntimeError(f"Failed to encode sequences: {selected_sequences}")
            return encoding

        # elif isinstance(idx, str):
        #     if idx not in self.words:
        #         raise KeyError(f"Sequence '{idx}' not found in dataset")
        #     encoding = self.tokenizer.encode(idx)
        #     if encoding is None:
        #         raise RuntimeError(f"Failed to encode sequence: {idx}")
        #     return encoding

        # elif isinstance(idx, list):
        #     if not all(isinstance(i, str) for i in idx):
        #         raise TypeError("List indices must be strings")
        #     encoding = self.tokenizer.encode(idx)
        #     if encoding is None:
        #         raise RuntimeError(f"Failed to encode sequences: {', '.join(idx)}")
        #     return encoding

        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def get_sequence_info(self, idx: int) -> dict:
        """Get information about a specific sequence.

        Args:
            idx: Index of the sequence.

        Returns:
            Dictionary with sequence information.
        """
        if idx < 0 or idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of range [0, {len(self.sequences)})")

        word_sequence = self.sequences[idx]
        text_sequence = " ".join(word_sequence)

        return {
            "words": word_sequence,
            "text": text_sequence,
            "num_words": len(word_sequence),
            "text_length": len(text_sequence),
            "word_lengths": [len(word) for word in word_sequence],
        }


# Example usage
if __name__ == "__main__":
    # Test single word dataset
    """
    print("=== Single Word Dataset ===")
    synthetic_dataset = SyntheticBridgeDataset(num_samples=10)
    print(f"Number of samples: {len(synthetic_dataset)}")
    sample_encoding = synthetic_dataset[0]
    print(f"Sample encoding shape: {sample_encoding.orthographic.enc_input_ids.shape}")

    # Test multi-word dataset
    print("\n=== Multi-Word Dataset ===")
    multi_word_dataset = SyntheticBridgeDatasetMultiWord(
        num_samples=10,
        max_seq_len=20,
        min_words_per_sequence=2,
        max_words_per_sequence=8,
    )
    print(f"Number of sequences: {len(multi_word_dataset)}")

    # Show some sequence examples
    for i in range(min(3, len(multi_word_dataset))):
        print("================================================")
        info = multi_word_dataset.get_sequence_info(i)
        sample = multi_word_dataset[i]
        # Print the list of words
        # print(f"Sample {i} words: {info['words']}")
        # Print the full sequence as a string (words joined by space)
        # print(f"Sample {i} sequence: {' '.join(info['words'])}")
        # Print the list of phonemes
        if hasattr(sample.phonological, "enc_input_phonemes"):
            phonemes = sample.phonological.enc_input_phonemes
            print(f"Sample {i} phonemes: {phonemes}")
        else:
            print(f"Sample {i} phonemes: (phoneme information not available)")
        print()
    """
    # Add seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    multi_word_dataset = SyntheticBridgeDatasetMultiWord(
        num_samples=5, max_seq_len=20, seed=42
    )
    batch = multi_word_dataset.sequences[:]
    for i, b in enumerate(batch):
        print(f"==== seq {i} ====")
        print(f"seq: {b}")
        seq = " ".join(b)
        print(f"{len(seq)=}")

    print("multi_word_dataset[3]")
    # print(multi_word_dataset[3])
    for i in range(len(multi_word_dataset)):
        print(f"===== {i=} =================================================")
        print(multi_word_dataset[i].orthographic.enc_input_ids.shape)
        print(multi_word_dataset[i].orthographic.dec_input_ids.shape)
        print(f"{multi_word_dataset[i].orthographic.enc_input_ids=}")
        print(f"{multi_word_dataset[i].orthographic.dec_input_ids=}")
        print(f"{multi_word_dataset[i].orthographic.enc_pad_mask.shape=}")
        print(f"{multi_word_dataset[i].orthographic.dec_pad_mask.shape=}")
        if hasattr(multi_word_dataset[i].phonological, "enc_input_ids"):
            print(f"{multi_word_dataset[i].phonological.enc_input_ids=}")
 
