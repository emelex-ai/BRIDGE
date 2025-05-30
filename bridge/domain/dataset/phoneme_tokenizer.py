import logging
import os
from pathlib import PosixPath

import pandas as pd
import torch
import json
from scipy.spatial.distance import hamming

from bridge.domain.dataset import CUDADict
from bridge.utils import get_project_root
from bridge.utils import device_manager

logger = logging.getLogger(__name__)


class PhonemeTokenizer:
    """PhonemeTokenizer converts words to phonetic feature vectors using CMU dict and phonetic features."""

    def __init__(
        self,
        max_cache_size: int = 10000,
        lang_codes: list[str] | None = None,
        custom_cmudict_path: str | PosixPath | None = None,
    ):
        # Set device - defaulting to CPU if None provided
        self.device = device_manager.device

        # Load phonetic representations from config
        self.phonreps = pd.read_csv(
            os.path.join(get_project_root(), "bridge/core/phonreps.csv")
        )
        self.phonreps.set_index("phone", inplace=True)
        self.base_dim = len(self.phonreps.columns)

        # Convert phonreps to PyTorch tensor for faster lookup
        self.phonreps_array = torch.tensor(
            self.phonreps.values, dtype=torch.float, device=self.device
        )
        self.phonreps_index = {p: i for i, p in enumerate(self.phonreps.index)}

        # Create inverse mapping from phoneme vectors to phoneme strings
        self._create_inverse_phoneme_mapping()

        # Load multilingual phonological lexicon dict as foundation
        self.pronunciation_dict = self._load_multilingual_vocab(lang_codes=lang_codes)

        # --- BEGIN Custom CMU-dict loading ---
        custom_prons = {}
        if custom_cmudict_path:
            if os.path.isfile(custom_cmudict_path):
                try:
                    with open(custom_cmudict_path, "r") as f:
                        custom_pron = json.load(f)
                except Exception as e:
                    logger.warning(
                        f"Failed to load custom CMU dict at {custom_cmudict_path}: {e}"
                    )
            else:
                logger.warning(f"Custom word dict not found at {custom_cmudict_path}")

        # Overwrite any entries in the lexicon with custom pronunciations
        for word, langs in custom_prons.items():
            for lang, pronunciation in langs.items():
                self.pronunciation_dict[word.lower()][lang.lower()] = pronunciation

        # Special tokens at end of vector space...
        # We should probably add a new special token for 'null' or something.
        # This is needed for when the generate routine produces a phoneme vector
        # with NO active features. This is not a 'valid' phoneme, but we need
        # to handle it gracefully.
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

    def _load_multilingual_vocab(
        self, directory: str | None = None, lang_codes: list[str] | None = None
    ) -> dict:
        """Load all language-specific JSON files into a unified vocabulary structure.""" """Load language-specific JSON files into a unified vocabulary structure."""
        if directory is None:
            directory = os.path.join(
                get_project_root(), "bridge/core/pronunciation_lexicons"
            )
        if lang_codes is None:
            lang_codes = ["en", "es"]  # Default supported languages

        vocab = {}
        if not os.path.exists(directory):
            logger.error(f"Pronunciation lexicon directory not found: {directory}")
            return vocab
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                # Extract ISO 639 language code from filename
                lang_code = filename.split(".")[0].lower()
                if lang_code.lower() not in lang_codes:
                    continue
                if len(lang_code) != 2:
                    logger.warning(
                        f"Invalid language code in filename: {filename}. Skipping. Must be two letter ISO 639 code."
                    )
                    continue
                file_path = os.path.join(directory, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    lang_dict = json.load(f)

                for word, pronunciations in lang_dict.items():
                    normalized_word = word.lower()

                    if normalized_word not in vocab:
                        vocab[normalized_word] = {}

                    # Store pronunciations under language code
                    vocab[normalized_word][lang_code] = pronunciations

        return vocab

    def _get_word_phonemes(self, word: str, language: str = "en") -> list | None:
        """Get phonemes for a single word and language"""
        if not word:  # Handle empty string
            return []

        lookup_word = word.lower()
        if lookup_word in self.pronunciation_dict:
            lang_code = language.lower()
            # Check if pronunciation exists for requested language
            if lang_code in self.pronunciation_dict[lookup_word]:
                # Take the first pronunciation available for the word (for now)
                return self.pronunciation_dict[lookup_word][lang_code][0]
            # Fall back to English if requested language not available
            elif "en" in self.pronunciation_dict[lookup_word]:
                logger.warning(
                    f"Word '{word}' not found in {language}, falling back to English"
                )
                # Take the first pronunciation available for the word (for now)
                return self.pronunciation_dict[lookup_word]["en"][0]
            else:
                logger.warning(f"Word '{word}' exists but not in {language} or English")
        else:
            logger.debug(f"Word '{word}' not found in pronunciation lexicon")

        return None

    def _get_phrase_phonemes(
        self, phrase: str, language_map: dict[str, str] | None = None
    ) -> list | None:
        """Convert a phrase into phonemes, handling spaces between words."""
        words = phrase.strip().split()
        if not words:  # Handle empty or whitespace-only input
            return []

        if language_map is None:
            language_map = {word: "en" for word in words}

        result = []
        for i, word in enumerate(words):
            phonemes = self._get_word_phonemes(word, language_map[word])
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

    def encode(
        self, words: str | list[str], language_map: dict[str, str] | None = None
    ) -> CUDADict | None:
        """Encode words or phrases to phonetic feature indices."""

        if isinstance(words, str):
            words = [words]

        # Process each word/phrase into phonemes
        word_phonemes = []
        for phrase in words:
            phonemes = self._get_phrase_phonemes(phrase, language_map)
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
        values = torch.ones(int(lengths.sum()), device=self.device)

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

    def _create_inverse_phoneme_mapping(self):
        """Create an inverse mapping from phoneme vectors to phoneme strings."""
        # Since vectors might not be unique, we'll use a more sophisticated approach
        # Store both the vectors and a quick lookup for exact matches
        self.phoneme_vectors_to_strings = {}

        # For each phoneme, store its vector representation
        for phoneme, idx in self.phonreps_index.items():
            # Convert the vector to a tuple for hashability
            vector = self.phonreps_array[idx]
            vector_tuple = tuple(vector.cpu().numpy().astype(int))

            # Store the mapping (handling potential collisions)
            if vector_tuple not in self.phoneme_vectors_to_strings:
                self.phoneme_vectors_to_strings[vector_tuple] = []
            self.phoneme_vectors_to_strings[vector_tuple].append(phoneme)

        # Also store all phoneme vectors as a tensor for distance calculations
        self.all_phoneme_vectors = self.phonreps_array
        self.all_phoneme_names = list(self.phonreps_index.keys())

    def phoneme_vector_to_phoneme(self, phoneme_vector, distance_fn=None, top_k=1):
        """
        Map a phoneme vector back to phoneme string(s).

        Args:
            phoneme_vector: Tensor representing a phoneme's features
            distance_fn: Optional distance function. If None, uses Hamming distance
            top_k: Number of closest phonemes to return if no exact match

        Returns:
            List of phoneme strings (exact matches or closest matches)
        """
        # Ensure vector is on the correct device
        if isinstance(phoneme_vector, torch.Tensor):
            phoneme_vector = phoneme_vector.to(self.device)
        else:
            phoneme_vector = torch.tensor(phoneme_vector, device=self.device)

        # Check if this is a special token vector
        # Special tokens are one-hot encoded at dimensions >= base_dim
        if phoneme_vector.dim() == 1 and len(phoneme_vector) > self.base_dim:
            # Check for special token by finding the active dimension
            active_dims = torch.nonzero(phoneme_vector, as_tuple=True)[0]
            if len(active_dims) == 1 and active_dims[0] >= self.base_dim:
                # This is a special token
                for token, dim in self.special_token_dims.items():
                    if active_dims[0].item() == dim:
                        return [token]
                return ["[UNK]"]

            # If not a special token, extract only the base phoneme features
            phoneme_vector = phoneme_vector[: self.base_dim]

        # First, try exact match
        vector_tuple = tuple(phoneme_vector.cpu().numpy().astype(int))
        if vector_tuple in self.phoneme_vectors_to_strings:
            return self.phoneme_vectors_to_strings[vector_tuple]

        # If no exact match, find closest phoneme(s)
        if distance_fn is None:
            # Default to Hamming distance for binary vectors
            def hamming_distance(v1, v2):
                return (v1 != v2).float().sum()

            distance_fn = hamming_distance

        # Calculate distances to all phonemes
        distances = torch.tensor(
            [
                distance_fn(phoneme_vector, self.all_phoneme_vectors[i])
                for i in range(len(self.all_phoneme_vectors))
            ],
            device=self.device,
        )

        # Get top-k closest phonemes
        _, indices = torch.topk(distances, k=min(top_k, len(distances)), largest=False)

        # Return the phoneme strings
        closest_phonemes = [self.all_phoneme_names[idx.item()] for idx in indices]

        # If only one closest match requested and distance is reasonable, return just that
        if top_k == 1:
            return closest_phonemes

        # Otherwise, return all requested matches with their distances for debugging
        return [
            (self.all_phoneme_names[idx.item()], distances[idx].item())
            for idx in indices
        ]

    def phoneme_vectors_to_word(self, phoneme_vectors, distance_fn=None):
        """
        Convert a sequence of phoneme vectors back to a sequence of phonemes.

        Args:
            phoneme_vectors: List or tensor of phoneme vectors
            distance_fn: Optional distance function for inexact matches

        Returns:
            List of phoneme strings representing the word
        """
        phonemes = []

        for vector in phoneme_vectors:
            # Handle different vector types
            if isinstance(vector, torch.Tensor):
                # Check if it's a scalar (special token index)
                if vector.dim() == 0:
                    # This might be a special token index
                    for token, dim in self.special_token_dims.items():
                        if vector.item() == dim:
                            phonemes.append(token)
                            break
                    else:
                        # Not a special token, treat as unknown
                        phonemes.append("[UNK]")
                else:
                    # Regular phoneme vector - use the updated method
                    matches = self.phoneme_vector_to_phoneme(
                        vector, distance_fn, top_k=1
                    )
                    if matches:
                        phonemes.append(matches[0])
                    else:
                        phonemes.append("[UNK]")
            else:
                # Non-tensor input
                matches = self.phoneme_vector_to_phoneme(vector, distance_fn, top_k=1)
                if matches:
                    phonemes.append(matches[0])
                else:
                    phonemes.append("[UNK]")

        return phonemes
