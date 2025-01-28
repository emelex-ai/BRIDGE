from typing import Union
import pandas as pd
import numpy as np
import torch
import logging
from nltk.corpus import cmudict
from src.domain.dataset import CUDADict
from src.domain.datamodels import DatasetConfig

logger = logging.getLogger(__name__)


class PhonemeTokenizer:
    """PhonemeTokenizer converts words to phonetic feature vectors using CMU dict and phonetic features."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        max_cache_size: int = 10000,
    ):
        if isinstance(device, str):
            assert device in [
                "cpu",
                "cuda",
                "mps",
            ], "Device must be 'cpu', 'cuda', or 'mps'"

        # Set device - defaulting to CPU if None provided
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

        # Load phonetic representations from config
        self.phonreps = pd.read_csv("data/phonreps.csv")
        self.phonreps.set_index("phone", inplace=True)
        self.base_dim = len(self.phonreps.columns)

        # Convert phonreps to numpy for faster lookup
        self.phonreps_array = self.phonreps.values
        self.phonreps_index = {p: i for i, p in enumerate(self.phonreps.index)}

        # Load CMU dict and clean problematic entries
        self.pronunciation_dict = {
            word: pron[0]
            for word, pron in cmudict.dict().items()
            if pron  # Skip empty pronunciations
        }

        # Special tokens at end of vector space - added [SPC] token
        self.special_token_dims = {
            "[BOS]": self.base_dim,
            "[EOS]": self.base_dim + 1,
            "[PAD]": self.base_dim + 2,
            "[UNK]": self.base_dim + 3,
            "[SPC]": self.base_dim + 4,
        }
        self.vocabulary_size = self.base_dim + len(self.special_token_dims)

        # Pre-compute special token vectors
        self.special_vecs = {
            token: np.array([dim], dtype=np.int64)
            for token, dim in self.special_token_dims.items()
        }

        # Efficient cache using numpy arrays
        self.vector_cache = {}
        self.max_cache_size = max_cache_size

    def _get_word_phonemes(self, word: str) -> list | None:
        """Get phonemes for a single word, handling spaces."""
        if not word:  # Handle empty string
            return []

        lookup_word = word.lower()
        if lookup_word in self.pronunciation_dict:
            return self.pronunciation_dict[lookup_word]
        return None

    def _get_phrase_phonemes(self, phrase: str) -> list | None:
        """Convert a phrase into phonemes, handling spaces between words."""
        words = phrase.strip().split()
        if not words:  # Handle empty or whitespace-only input
            return []

        result = []
        for i, word in enumerate(words):
            phonemes = self._get_word_phonemes(word)
            if phonemes is None:
                logger.warning(f"Word not found in phrase: {word}")
                return None
            result.extend(phonemes)
            # Add space between words, but not after the last word
            if i < len(words) - 1:
                result.append("[SPC]")
        return result

    def _get_phoneme_indices(self, phoneme: str) -> np.ndarray:
        """Get active feature indices for a phoneme."""
        if phoneme in self.vector_cache:
            return self.vector_cache[phoneme]

        # Handle special tokens directly
        if phoneme in self.special_vecs:
            return self.special_vecs[phoneme]

        # Get phoneme feature vector
        if phoneme in self.phonreps_index:
            idx = self.phonreps_index[phoneme]
            active_indices = np.where(self.phonreps_array[idx] == 1)[0]

            # Cache management
            if len(self.vector_cache) >= self.max_cache_size:
                self.vector_cache.pop(next(iter(self.vector_cache)))
            self.vector_cache[phoneme] = active_indices

            return active_indices
        return self.special_vecs["[UNK]"]

    def encode(self, words: Union[str, list[str]]) -> CUDADict | None:
        """Encode words or phrases to phonetic feature indices."""
        if isinstance(words, str):
            words = [words]

        # Process each word/phrase into phonemes
        word_phonemes = []
        for phrase in words:
            phonemes = self._get_phrase_phonemes(phrase)
            if phonemes is None:
                return None
            word_phonemes.append(phonemes)

        # Rest of the method remains unchanged
        batch_size = len(words)
        max_length = max(len(p) for p in word_phonemes)
        enc_length = max_length + 2  # Add BOS, EOS
        dec_length = max_length + 1  # Add BOS

        # Pre-compute sequences for entire batch
        enc_indices = []
        dec_indices = []

        # Prepare target tensors - will be full binary vectors
        targets = torch.full(
            (batch_size, dec_length, self.vocabulary_size), 2, dtype=torch.long
        )

        for i, phoneme_seq in enumerate(word_phonemes):
            # Convert phonemes to index arrays
            phoneme_indices = [self._get_phoneme_indices(p) for p in phoneme_seq]

            # Build sequences with special tokens
            enc_seq = (
                [self.special_vecs["[BOS]"]]
                + phoneme_indices
                + [self.special_vecs["[EOS]"]]
            )
            dec_seq = [self.special_vecs["[BOS]"]] + phoneme_indices

            # Pad sequences
            while len(enc_seq) < enc_length:
                enc_seq.append(self.special_vecs["[PAD]"])
            while len(dec_seq) < dec_length:
                dec_seq.append(self.special_vecs["[PAD]"])

            enc_indices.append(enc_seq)
            dec_indices.append(dec_seq)

            # Fill target tensors with one-hot encodings
            for j, indices in enumerate(phoneme_indices + [self.special_vecs["[EOS]"]]):
                targets[i, j].fill_(0)  # Clear the 2's for this position
                targets[i, j, indices] = 1  # Set active features to 1

        # Create padding masks efficiently and move to device
        enc_pad_mask = torch.arange(enc_length, device=self.device).expand(
            batch_size, enc_length
        )
        enc_pad_mask = enc_pad_mask >= torch.tensor(
            [len(p) + 2 for p in word_phonemes], device=self.device
        ).unsqueeze(1)

        dec_pad_mask = torch.arange(dec_length, device=self.device).expand(
            batch_size, dec_length
        )
        dec_pad_mask = dec_pad_mask >= torch.tensor(
            [len(p) + 1 for p in word_phonemes], device=self.device
        ).unsqueeze(1)

        # Move targets to device
        targets = targets.to(self.device)

        return CUDADict(
            {
                "enc_input_ids": enc_indices,
                "enc_pad_mask": enc_pad_mask,
                "dec_input_ids": dec_indices,
                "dec_pad_mask": dec_pad_mask,
                "targets": targets,
            }
        )

    # decode and get_vocabulary_size methods remain unchanged
    def decode(self, indices_batch: list[list[int]]) -> torch.Tensor:
        """Convert feature indices to sparse one-hot vectors."""
        batch_size = len(indices_batch)
        values = torch.ones(sum(len(indices) for indices in indices_batch))

        # Build sparse tensor indices
        row_indices = torch.repeat_interleave(
            torch.arange(batch_size),
            torch.tensor([len(indices) for indices in indices_batch]),
        )
        col_indices = torch.cat([torch.tensor(idx) for idx in indices_batch])

        indices = torch.stack([row_indices, col_indices])
        return torch.sparse_coo_tensor(
            indices, values, (batch_size, self.vocabulary_size), device=self.device
        ).to_dense()

    def get_vocabulary_size(self) -> int:
        return self.vocabulary_size
