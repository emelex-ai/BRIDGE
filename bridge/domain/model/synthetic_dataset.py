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

    def _estimate_tokens_for_words(self, words: list[str]) -> int:
        """Estimate the number of tokens that a sequence of words will produce.

        This is an approximation since we don't want to tokenize every combination.
        We assume each word adds roughly its length + 1 (for space) tokens.

        Args:
            words: List of words to estimate token count for.

        Returns:
            Estimated number of tokens.
        """
        # BOS token
        total_tokens = 1

        for i, word in enumerate(words):
            # Word length + 1 for space (except last word)
            word_tokens = len(word)
            if i < len(words) - 1:  # Not the last word
                word_tokens += 1  # Space
            total_tokens += word_tokens

        # EOS token
        total_tokens += 1

        return total_tokens

    def _generate_multi_word_sequences(self, num_samples: int) -> list[list[str]]:
        """Generate multi-word sequences that respect max_seq_len.

        Args:
            num_samples: Number of sequences to generate.

        Returns:
            List of word sequences, where each sequence is a list of words.
        """
        if not self.real_words:
            raise RuntimeError("No real words available from CMU dictionary")

        sequences = []

        for _ in range(num_samples):
            # Determine how many words to use in this sequence
            if self.max_words_per_sequence is None:
                # Dynamically determine max words based on average word length
                avg_word_len = sum(self.word_length_range) / 2
                estimated_max_words = max(
                    self.min_words_per_sequence,
                    (self.max_seq_len - 2)
                    // (avg_word_len + 1),  # -2 for BOS/EOS, +1 for space
                )
                max_words = min(estimated_max_words, len(self.real_words))
            else:
                max_words = min(self.max_words_per_sequence, len(self.real_words))

            # Randomly choose number of words for this sequence
            num_words = random.randint(self.min_words_per_sequence, max_words)

            # Build sequence word by word, checking token count
            sequence = []
            current_tokens = 1  # BOS token

            for _ in range(num_words):
                # Try to add a word
                word = random.choice(self.real_words)

                # Estimate tokens if we add this word
                test_sequence = sequence + [word]
                estimated_tokens = self._estimate_tokens_for_words(test_sequence)

                if estimated_tokens <= self.max_seq_len:
                    sequence.append(word)
                    current_tokens = estimated_tokens
                else:
                    # This word would make the sequence too long, stop here
                    break

            # Ensure we have at least min_words_per_sequence
            if len(sequence) < self.min_words_per_sequence:
                # Take first min_words_per_sequence words from vocabulary
                sequence = self.real_words[: self.min_words_per_sequence]

            sequences.append(sequence)

        return sequences

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> BridgeEncoding:
        """Retrieve encoded data for a specified index.

        Args:
            idx: Index of the sequence to retrieve.

        Returns:
            BridgeEncoding object for the sequence.
        """
        if idx < 0 or idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of range [0, {len(self.sequences)})")

        # Join words with spaces to create a text sequence
        word_sequence = self.sequences[idx]
        text_sequence = " ".join(word_sequence)

        # Encode the real word sequence - this should always work since words are from CMU dict
        encoding = self.tokenizer.encode(text_sequence)
        if encoding is None:
            raise RuntimeError(
                f"Failed to encode sequence from CMU dictionary: {text_sequence}"
            )

        return encoding

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
        print(f"Sample {i} words: {info['words']}")
        # Print the full sequence as a string (words joined by space)
        print(f"Sample {i} sequence: {' '.join(info['words'])}")
        # Print the list of phonemes
        if hasattr(sample.phonological, "enc_input_phonemes"):
            phonemes = sample.phonological.enc_input_phonemes
            print(f"Sample {i} phonemes: {phonemes}")
        else:
            print(f"Sample {i} phonemes: (phoneme information not available)")
        print()
