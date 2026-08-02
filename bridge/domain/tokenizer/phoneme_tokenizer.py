import json
import logging
import os

import torch

from bridge.core.phonreps import load_phonreps
from bridge.domain.datamodels.encodings import EncodingComponent
from bridge.utils import device_manager, get_project_root

logger = logging.getLogger(__name__)


class PhonemeTokenizer:
    """PhonemeTokenizer converts words to phonetic feature vectors using per-language
    pronunciation lexicons (shipped under ``bridge/core/pronunciation_lexicons/``) and
    the phonetic feature table at ``bridge/core/phonreps.csv``.

    The tokenizer is multilingual by default: supply ``lang_codes`` to control which
    language lexicons are loaded, and pass a per-word ``language_map`` to
    :meth:`encode` for code-switching.
    """

    def __init__(
        self,
        max_cache_size: int = 10000,
        lang_codes: list[str] | None = None,
        custom_cmudict_path: str | None = None,
    ):
        self.device = device_manager.device

        phonreps = load_phonreps(device=self.device)
        self.base_dim = phonreps.base_dim
        self.phonreps_array = phonreps.array
        self.phonreps_index = phonreps.index

        self._create_inverse_phoneme_mapping()

        # Load multilingual pronunciation lexicon. Shape: {word: {lang_code: [[phonemes], ...]}}
        self.pronunciation_dict = self._load_multilingual_vocab(lang_codes=lang_codes)

        # Optional custom CMU dict: same nested-by-language shape as the lexicons, used
        # to override / extend the bundled lexicons.
        custom_prons: dict = {}
        if custom_cmudict_path:
            if os.path.isfile(custom_cmudict_path):
                try:
                    with open(custom_cmudict_path) as f:
                        custom_prons = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load custom CMU dict at {custom_cmudict_path}: {e}")
            else:
                logger.warning(f"Custom CMU dict not found at {custom_cmudict_path}")

        for word, langs in custom_prons.items():
            self.pronunciation_dict.setdefault(word.lower(), {})
            for lang, pronunciation in langs.items():
                self.pronunciation_dict[word.lower()][lang.lower()] = pronunciation

        # Special tokens at end of vector space.
        self.special_token_dims = {
            "[BOS]": self.base_dim,
            "[EOS]": self.base_dim + 1,
            "[UNK]": self.base_dim + 2,
            "[SPC]": self.base_dim + 3,
            "[PAD]": self.base_dim + 4,
        }
        # Keyed by float so a scalar `tensor.item()` from either an int or float tensor
        # can be looked up directly; non-integral values simply miss.
        self._special_token_by_dim: dict[float, str] = {
            float(dim): token for token, dim in self.special_token_dims.items()
        }
        self.vocabulary_size = self.base_dim + len(self.special_token_dims)

        self.special_vecs = {
            token: torch.tensor([dim], dtype=torch.long, device=self.device)
            for token, dim in self.special_token_dims.items()
        }
        # Loop-invariant index vector used to one-hot encode targets in `encode`.
        self._feature_range = torch.arange(self.vocabulary_size - 1, device=self.device)

        self.vector_cache: dict[str, torch.Tensor] = {}
        self.max_cache_size = max_cache_size

    def _load_multilingual_vocab(
        self,
        directory: str | None = None,
        lang_codes: list[str] | None = None,
    ) -> dict[str, dict[str, list]]:
        """Load language-specific JSON lexicons into a unified ``{word: {lang: [variants]}}`` dict.

        Each file under ``directory`` is named ``<iso639_two_letter_code>.json`` and contains
        ``{word: [[phoneme, ...], ...]}`` (a list of variant pronunciations). Only files whose
        language code appears in ``lang_codes`` are loaded; the default is ``["en", "es"]``.
        """
        if directory is None:
            directory = os.path.join(get_project_root(), "bridge/core/pronunciation_lexicons")
        if lang_codes is None:
            lang_codes = ["en", "es"]

        vocab: dict[str, dict[str, list]] = {}
        if not os.path.isdir(directory):
            logger.error(f"Pronunciation lexicon directory not found: {directory}")
            return vocab

        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".json"):
                continue
            lang_code = filename.rsplit(".", 1)[0].lower()
            if len(lang_code) != 2:
                logger.warning(
                    f"Skipping {filename}: filename stem must be a two-letter ISO 639 code."
                )
                continue
            if lang_code not in lang_codes:
                continue

            with open(os.path.join(directory, filename), encoding="utf-8") as f:
                lang_dict = json.load(f)

            for word, pronunciations in lang_dict.items():
                normalized = word.lower()
                vocab.setdefault(normalized, {})[lang_code] = pronunciations

        return vocab

    def _get_word_phonemes(self, word: str, language: str = "en") -> list | None:
        """Get the first pronunciation variant for ``word`` in ``language``.

        Falls back to English if the word exists in the lexicon but not in the requested
        language. Returns ``None`` if the word is unknown.
        """
        if not word:
            return []

        lookup_word = word.lower()
        if lookup_word not in self.pronunciation_dict:
            logger.debug(f"Word '{word}' not found in pronunciation lexicon")
            return None

        lang_code = language.lower()
        word_entry = self.pronunciation_dict[lookup_word]
        if lang_code in word_entry:
            return word_entry[lang_code][0]
        if "en" in word_entry:
            logger.warning(f"Word '{word}' not found in {language}, falling back to English")
            return word_entry["en"][0]
        logger.warning(f"Word '{word}' exists but not in {language} or English")
        return None

    def _get_phrase_phonemes(
        self, phrase: str, language_map: dict[str, str] | None = None
    ) -> list | None:
        """Convert a phrase into phonemes, with optional per-word language tags."""
        words = phrase.strip().split()
        if not words:
            return []

        if language_map is None:
            language_map = {}

        result: list = []
        for i, word in enumerate(words):
            phonemes = self._get_word_phonemes(word, language_map.get(word, "en"))
            if phonemes is None:
                logger.warning(f"Word not found in phrase: {word}")
                return None
            result.extend(phonemes)
            if i < len(words) - 1:
                result.append("[SPC]")
        return result

    def _get_phoneme_indices(self, phoneme: str) -> torch.Tensor:
        """Get active feature indices for a phoneme."""
        if phoneme in self.vector_cache:
            return self.vector_cache[phoneme]

        if phoneme in self.special_vecs:
            return self.special_vecs[phoneme]

        if phoneme in self.phonreps_index:
            idx = self.phonreps_index[phoneme]
            active_indices = torch.nonzero(self.phonreps_array[idx] == 1, as_tuple=True)[0].to(
                dtype=torch.long
            )

            if len(self.vector_cache) >= self.max_cache_size:
                self.vector_cache.pop(next(iter(self.vector_cache)))
            self.vector_cache[phoneme] = active_indices

            return active_indices
        return self.special_vecs["[UNK]"]

    def encode(
        self,
        words: str | list[str],
        language_map: dict[str, str] | None = None,
    ) -> EncodingComponent | None:
        """Encode words or phrases to phonetic feature indices.

        ``language_map`` is an optional ``{word: language_code}`` mapping. Defaults to
        treating every word as English. Returns ``None`` if any word is missing from the
        pronunciation lexicon.
        """
        if isinstance(words, str):
            words = [words]

        word_phonemes = []
        for phrase in words:
            phonemes = self._get_phrase_phonemes(phrase, language_map)
            if phonemes is None:
                return None
            word_phonemes.append(phonemes)

        batch_size = len(words)
        max_length = max(len(p) for p in word_phonemes)
        enc_length = max_length + 2  # BOS, EOS
        dec_length = max_length + 1  # BOS

        pad_vec = self.special_vecs["[PAD]"]
        bos_vec = self.special_vecs["[BOS]"]
        eos_vec = self.special_vecs["[EOS]"]

        enc_indices = []
        dec_indices = []

        targets = torch.full(
            (batch_size, dec_length, self.vocabulary_size - 1),
            self.special_token_dims["[PAD]"],
            dtype=torch.long,
            device=self.device,
        )

        for i, phoneme_seq in enumerate(word_phonemes):
            phoneme_indices = [self._get_phoneme_indices(p) for p in phoneme_seq]

            enc_seq = [bos_vec, *phoneme_indices, eos_vec]
            dec_seq = [bos_vec, *phoneme_indices]
            enc_seq += [pad_vec] * (enc_length - len(enc_seq))
            dec_seq += [pad_vec] * (dec_length - len(dec_seq))

            enc_indices.append(enc_seq)
            dec_indices.append(dec_seq)

            for j, indices in enumerate([*phoneme_indices, eos_vec]):
                targets[i, j] = torch.isin(self._feature_range, indices).long()

        seq_lengths = torch.tensor([len(p) + 2 for p in word_phonemes], device=self.device)
        enc_positions = torch.arange(enc_length, device=self.device).expand(batch_size, enc_length)
        dec_positions = torch.arange(dec_length, device=self.device).expand(batch_size, dec_length)

        return EncodingComponent(
            enc_input_ids=enc_indices,
            enc_pad_mask=enc_positions >= seq_lengths.unsqueeze(1),
            dec_input_ids=dec_indices,
            dec_pad_mask=dec_positions >= (seq_lengths - 1).unsqueeze(1),
            targets=targets,
        )

    def decode(self, indices_batch: list[list[int]]) -> torch.Tensor:
        """Convert feature indices to sparse one-hot vectors."""
        batch_size = len(indices_batch)

        lengths = torch.tensor([len(indices) for indices in indices_batch], device=self.device)
        values = torch.ones(int(lengths.sum()), device=self.device)

        row_indices = torch.repeat_interleave(torch.arange(batch_size, device=self.device), lengths)
        col_indices = torch.cat([torch.tensor(idx, device=self.device) for idx in indices_batch])

        indices = torch.stack([row_indices, col_indices])
        return torch.sparse_coo_tensor(
            indices, values, (batch_size, self.vocabulary_size), device=self.device
        ).to_dense()

    def get_vocabulary_size(self) -> int:
        return self.vocabulary_size

    def _create_inverse_phoneme_mapping(self):
        """Create an inverse mapping from phoneme vectors to phoneme strings."""
        self.phoneme_vectors_to_strings: dict[tuple, list[str]] = {}

        for phoneme, idx in self.phonreps_index.items():
            vector_tuple = tuple(self.phonreps_array[idx].cpu().numpy().astype(int))
            self.phoneme_vectors_to_strings.setdefault(vector_tuple, []).append(phoneme)

        self.all_phoneme_names = list(self.phonreps_index.keys())

    def phoneme_vector_to_phoneme(self, phoneme_vector, distance_fn=None, top_k=1):
        """Map a phoneme vector back to phoneme string(s).

        Args:
            phoneme_vector: Tensor representing a phoneme's features
            distance_fn: Optional distance function. If None, uses Hamming distance
            top_k: Number of closest phonemes to return if no exact match

        Returns:
            List of phoneme strings (exact matches or closest matches)
        """
        if isinstance(phoneme_vector, torch.Tensor):
            phoneme_vector = phoneme_vector.to(self.device)
        else:
            phoneme_vector = torch.tensor(phoneme_vector, device=self.device)

        if phoneme_vector.dim() == 1 and len(phoneme_vector) > self.base_dim:
            active_dims = torch.nonzero(phoneme_vector, as_tuple=True)[0]
            if len(active_dims) == 1 and active_dims[0] >= self.base_dim:
                return [self._special_token_by_dim.get(float(active_dims[0].item()), "[UNK]")]

            phoneme_vector = phoneme_vector[: self.base_dim]

        vector_tuple = tuple(phoneme_vector.cpu().numpy().astype(int))
        if vector_tuple in self.phoneme_vectors_to_strings:
            return self.phoneme_vectors_to_strings[vector_tuple]

        if distance_fn is None:

            def distance_fn(v1, v2):  # Hamming distance
                return (v1 != v2).float().sum()

        distances = torch.tensor(
            [distance_fn(phoneme_vector, rep) for rep in self.phonreps_array],
            device=self.device,
        )

        _, indices = torch.topk(distances, k=min(top_k, len(distances)), largest=False)

        if top_k == 1:
            return [self.all_phoneme_names[idx.item()] for idx in indices]

        return [(self.all_phoneme_names[idx.item()], distances[idx].item()) for idx in indices]

    def phoneme_vectors_to_word(self, phoneme_vectors, distance_fn=None):
        """Convert a sequence of phoneme vectors back to a sequence of phonemes."""
        phonemes = []

        for vector in phoneme_vectors:
            if isinstance(vector, torch.Tensor) and vector.dim() == 0:
                phonemes.append(self._special_token_by_dim.get(float(vector.item()), "[UNK]"))
            else:
                matches = self.phoneme_vector_to_phoneme(vector, distance_fn, top_k=1)
                phonemes.append(matches[0] if matches else "[UNK]")

        return phonemes
