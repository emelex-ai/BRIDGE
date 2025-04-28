import logging
import os

from nltk.corpus import cmudict
import pandas as pd
import torch
import json

from src.domain.dataset import CUDADict
from src.utils.helper_functions import get_project_root
from src.utils.device_manager import device_manager

logger = logging.getLogger(__name__)


class PhonemeTokenizer:
    """PhonemeTokenizer converts words to phonetic feature vectors using CMU dict and phonetic features."""

    def __init__(
        self,
        max_cache_size: int = 10000,
        custom_cmudict_path: str | None = None,
    ):
        # Set device - defaulting to CPU if None provided
        self.device = device_manager.device

        # Load phonetic representations from config
        self.phonreps = pd.read_csv(
            os.path.join(get_project_root(), "data/phonreps.csv")
        )
        self.phonreps.set_index("phone", inplace=True)
        self.base_dim = len(self.phonreps.columns)

        # Convert phonreps to PyTorch tensor for faster lookup
        self.phonreps_array = torch.tensor(
            self.phonreps.values, dtype=torch.float, device=self.device
        )
        self.phonreps_index = {p: i for i, p in enumerate(self.phonreps.index)}
        # --- BEGIN Custom CMU-dict loading ---
        custom_pron = {}
        if custom_cmudict_path:
            if os.path.isfile(custom_cmudict_path):
                try:
                    with open(custom_cmudict_path, "r") as f:
                        data = json.load(f)
                    for word, prons in data.items():
                        if prons:
                            # take the first pronunciation variant
                            custom_pron[word] = prons[0]
                except Exception as e:
                    logger.warning(
                        f"Failed to load custom CMU dict at {custom_cmudict_path}: {e}"
                    )
            else:
                logger.warning(f"Custom CMU dict not found at {custom_cmudict_path}")

        # Load official CMU dict as fallback
        fallback_pron = {word: pron[0] for word, pron in cmudict.dict().items() if pron}
        # Merge: use custom entries where available, otherwise fallback
        self.pronunciation_dict = {**fallback_pron, **custom_pron}

        # Special tokens at end of vector space - added [SPC] token
        self.special_token_dims = {
            "[BOS]": self.base_dim,
            "[EOS]": self.base_dim + 1,
            "[UNK]": self.base_dim + 2,
            "[SPC]": self.base_dim + 3,
            "[PAD]": self.base_dim + 4,
        }
        self.vocabulary_size = self.base_dim + len(self.special_token_dims)

        # Pre-compute special token vectors as PyTorch tensors
        self.special_vecs = {
            token: torch.tensor([dim], dtype=torch.long, device=self.device)
            for token, dim in self.special_token_dims.items()
        }

        # Efficient cache using PyTorch tensors
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

    def _get_phoneme_indices(self, phoneme: str) -> torch.Tensor:
        """Get active feature indices for a phoneme."""
        if phoneme in self.vector_cache:
            return self.vector_cache[phoneme]

        # Handle special tokens directly
        if phoneme in self.special_vecs:
            return self.special_vecs[phoneme]

        # Get phoneme feature vector
        if phoneme in self.phonreps_index:
            idx = self.phonreps_index[phoneme]
            # Find indices where features are active (1)
            active_indices = torch.nonzero(
                self.phonreps_array[idx] == 1, as_tuple=True
            )[0].to(dtype=torch.long)

            # Cache management
            if len(self.vector_cache) >= self.max_cache_size:
                self.vector_cache.pop(next(iter(self.vector_cache)))
            self.vector_cache[phoneme] = active_indices

            return active_indices
        return self.special_vecs["[UNK]"]

    def encode(self, words: str | list[str]) -> CUDADict | None:
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

        batch_size = len(words)
        max_length = max(len(p) for p in word_phonemes)
        enc_length = max_length + 2  # Add BOS, EOS
        dec_length = max_length + 1  # Add BOS

        # Pre-compute sequences for entire batch
        enc_indices = []
        dec_indices = []

        # Prepare target tensors - will be full binary vectors
        targets = torch.full(
            (batch_size, dec_length, self.vocabulary_size - 1),
            self.special_token_dims["[PAD]"],
            dtype=torch.long,
            device=self.device,
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
                one_hot = torch.isin(
                    torch.arange(self.vocabulary_size - 1, device=self.device), indices
                ).long()
                targets[i, j] = one_hot

        # Create padding masks efficiently
        seq_lengths = torch.tensor(
            [len(p) + 2 for p in word_phonemes], device=self.device
        )
        position_indices = torch.arange(enc_length, device=self.device).expand(
            batch_size, enc_length
        )

        enc_pad_mask = position_indices >= seq_lengths.unsqueeze(1)

        # For decoder mask, use sequence length + 1 (BOS only, no EOS)
        dec_seq_lengths = seq_lengths - 1
        dec_position_indices = torch.arange(dec_length, device=self.device).expand(
            batch_size, dec_length
        )

        dec_pad_mask = dec_position_indices >= dec_seq_lengths.unsqueeze(1)

        return CUDADict(
            {
                "enc_input_ids": enc_indices,
                "enc_pad_mask": enc_pad_mask,
                "dec_input_ids": dec_indices,
                "dec_pad_mask": dec_pad_mask,
                "targets": targets,
            }
        )

    def decode(self, indices_batch: list[list[int]]) -> torch.Tensor:
        """Convert feature indices to sparse one-hot vectors."""
        batch_size = len(indices_batch)

        # Convert all inputs to tensors
        lengths = torch.tensor(
            [len(indices) for indices in indices_batch], device=self.device
        )
        values = torch.ones(lengths.sum(), device=self.device)

        # Build sparse tensor indices
        row_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=self.device), lengths
        )
        col_indices = torch.cat(
            [torch.tensor(idx, device=self.device) for idx in indices_batch]
        )

        indices = torch.stack([row_indices, col_indices])
        return torch.sparse_coo_tensor(
            indices, values, (batch_size, self.vocabulary_size), device=self.device
        ).to_dense()

    def get_vocabulary_size(self) -> int:
        return self.vocabulary_size
