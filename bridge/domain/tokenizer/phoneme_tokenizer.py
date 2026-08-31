import functools
import json
import logging
import os

import torch

from bridge.core.phonreps import SPECIAL_TOKENS, load_phoneme_table
from bridge.domain.datamodels.encodings import EncodingComponent
from bridge.utils import device_manager, get_project_root

logger = logging.getLogger(__name__)


@functools.cache
def _load_lexicon(directory: str, lang_codes: tuple[str, ...]) -> dict[str, dict[str, list]]:
    """Parse the pronunciation lexicons once per process.

    Reading them costs ~1 s and ~80 MB, and every ``PhonemeTokenizer`` wants the same
    result. The returned dict is shared, so callers must treat it as read-only.
    """
    return PhonemeTokenizer._read_multilingual_vocab(directory, lang_codes)


class PhonemeTokenizer:
    """PhonemeTokenizer converts words to phoneme *row* ids using per-language
    pronunciation lexicons (shipped under ``bridge/core/pronunciation_lexicons/``) and
    the phonetic feature table at ``bridge/core/phonreps.csv``.

    :meth:`encode` emits ``(batch, sequence)`` row ids, the same shape the character
    tokenizer emits for orthography. Feature vectors appear in only two places: the loss
    ``targets`` that ``encode`` also returns, and the output of :meth:`decode`.

    The tokenizer is multilingual by default: supply ``lang_codes`` to control which
    language lexicons are loaded, and pass a per-word ``language_map`` to
    :meth:`encode` for code-switching.
    """

    def __init__(
        self,
        lang_codes: list[str] | None = None,
        custom_cmudict_path: str | None = None,
    ):
        self.device = device_manager.device
        # Recorded so a caller sharing this tokenizer can check it matches their config.
        self.custom_cmudict_path = custom_cmudict_path

        # Phoneme -> feature multi-hot. Row ids from this table are the phonological
        # model input; see PhonemeTable for the row-space / feature-space distinction.
        self.phoneme_table = load_phoneme_table(device=self.device)
        self.base_dim = self.phoneme_table.base_dim

        self._create_inverse_phoneme_mapping()

        # Shape: {word: {lang_code: [[phonemes], ...]}}. Shared across instances, so the
        # custom-dictionary overrides below copy rather than mutate it.
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

        if custom_prons:
            self.pronunciation_dict = dict(self.pronunciation_dict)
            for word, langs in custom_prons.items():
                merged = dict(self.pronunciation_dict.get(word.lower(), {}))
                merged.update({lang.lower(): pron for lang, pron in langs.items()})
                self.pronunciation_dict[word.lower()] = merged

        # Special tokens at end of vector space. Derived from the table so the two cannot
        # drift apart. The table's column layout is the definition.
        self.special_token_dims = {
            token: self.phoneme_table.feature_of(token) for token in SPECIAL_TOKENS
        }
        # Keyed by float so a scalar `tensor.item()` from either an int or float tensor
        # can be looked up directly; non-integral values simply miss.
        self._special_token_by_dim: dict[float, str] = {
            float(dim): token for token, dim in self.special_token_dims.items()
        }
        self.vocabulary_size = self.phoneme_table.vocab_size

        # Per-phoneme loss targets, gathered by row id in `encode`.
        #
        # Two things make this table differ from `phoneme_table.multihot`:
        #   * the [PAD] feature column is dropped (targets are one class per feature,
        #     excluding [PAD] itself), and
        #   * the [PAD] *row* is then filled with `phon_pad_id`, which is the
        #     CrossEntropyLoss `ignore_index`, so a padded position contributes no loss.
        # Slicing alone would leave the [PAD] row all-zeros, i.e. "every feature is off",
        # which would be scored rather than ignored.
        target_table = self.phoneme_table.multihot[:, :-1].long().clone()
        target_table[self.phoneme_table.row_index["[PAD]"]] = self.special_token_dims["[PAD]"]
        self._target_table = target_table

    def _load_multilingual_vocab(
        self,
        directory: str | None = None,
        lang_codes: list[str] | None = None,
    ) -> dict[str, dict[str, list]]:
        """Cached wrapper. See :func:`_load_lexicon`."""
        if directory is None:
            directory = os.path.join(get_project_root(), "bridge/core/pronunciation_lexicons")
        return _load_lexicon(directory, tuple(lang_codes) if lang_codes else ("en", "es"))

    @staticmethod
    def _read_multilingual_vocab(
        directory: str,
        lang_codes: tuple[str, ...],
    ) -> dict[str, dict[str, list]]:
        """Load language-specific JSON lexicons into a unified ``{word: {lang: [variants]}}`` dict.

        Each file under ``directory`` is named ``<iso639_two_letter_code>.json`` and contains
        ``{word: [[phoneme, ...], ...]}`` (a list of variant pronunciations). Only files whose
        language code appears in ``lang_codes`` are loaded; the default is ``["en", "es"]``.
        """
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

    def encode(
        self,
        words: str | list[str],
        language_map: dict[str, str] | None = None,
    ) -> EncodingComponent | None:
        """Encode words or phrases to phoneme row ids.

        Returns ``(B, L)`` tensors of *row-space* ids: indices into
        :class:`~bridge.core.phonreps.PhonemeTable`, i.e. "which phoneme", not "which
        feature". The model recovers the features by gathering from the same table; see
        the ``PhonemeTable`` docstring for the two id spaces.

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

        max_length = max(len(p) for p in word_phonemes)
        enc_length = max_length + 2  # BOS, EOS
        dec_length = max_length + 1  # BOS

        table = self.phoneme_table
        bos = table.row_index["[BOS]"]
        eos = table.row_index["[EOS]"]
        pad = table.row_index["[PAD]"]

        # Plain ints, so the whole batch becomes one tensor build rather than a
        # per-position device write.
        enc_rows: list[list[int]] = []
        dec_rows: list[list[int]] = []
        tgt_rows: list[list[int]] = []
        for phoneme_seq in word_phonemes:
            rows = [table.row_of(p) for p in phoneme_seq]
            n = len(rows)
            enc_rows.append([bos, *rows, eos, *[pad] * (enc_length - n - 2)])
            dec_rows.append([bos, *rows, *[pad] * (dec_length - n - 1)])
            # Targets are the decoder inputs shifted left by one: predict each phoneme,
            # then EOS. Padded positions carry the [PAD] row, whose target row is the
            # loss ignore_index.
            tgt_rows.append([*rows, eos, *[pad] * (dec_length - n - 1)])

        enc_input_ids = torch.tensor(enc_rows, dtype=torch.long, device=self.device)
        dec_input_ids = torch.tensor(dec_rows, dtype=torch.long, device=self.device)
        target_ids = torch.tensor(tgt_rows, dtype=torch.long, device=self.device)

        return EncodingComponent(
            enc_input_ids=enc_input_ids,
            enc_pad_mask=enc_input_ids == pad,
            dec_input_ids=dec_input_ids,
            dec_pad_mask=dec_input_ids == pad,
            targets=self._target_table[target_ids],
        )

    def padded_targets(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Loss targets for positions that are entirely padding.

        Every position carries the ``[PAD]`` row's target, which is the CrossEntropyLoss
        ``ignore_index``, so these positions contribute no loss, and the width matches
        what :meth:`encode` produces.
        """
        pad_row = self.phoneme_table.row_index["[PAD]"]
        # repeat, not expand: an expanded view has zero strides into `_target_table`, so a
        # caller writing to the result would rewrite the [PAD] row for every later encode.
        return self._target_table[pad_row].repeat(batch_size, seq_len, 1)

    def decode(self, indices_batch: list[list[int]]) -> torch.Tensor:
        """Convert *feature* indices to dense multi-hot vectors, one row per position.

        The input space is feature space: what ``GenerationOutput.phon_tokens`` holds.
        It is deliberately not the inverse of :meth:`encode`, which emits phoneme *row*
        ids; use ``phoneme_table.features_of`` to go from a row id to its features.
        """
        batch_size = len(indices_batch)

        lengths = torch.tensor([len(indices) for indices in indices_batch], device=self.device)
        values = torch.ones(int(lengths.sum()), device=self.device)

        row_indices = torch.repeat_interleave(torch.arange(batch_size, device=self.device), lengths)
        col_indices = torch.cat([torch.tensor(idx, device=self.device) for idx in indices_batch])

        # Unchecked, an out-of-range column silently corrupts the sparse tensor rather
        # than raising. That is the failure mode when row ids are passed here by mistake.
        if len(col_indices) and bool(
            ((col_indices < 0) | (col_indices >= self.vocabulary_size)).any()
        ):
            raise ValueError(
                f"Feature indices must lie in [0, {self.vocabulary_size}); got "
                f"[{int(col_indices.min())}, {int(col_indices.max())}]. Phoneme row ids "
                f"index the phoneme table, not the feature vocabulary; convert them with "
                f"PhonemeTable.features_of first."
            )

        indices = torch.stack([row_indices, col_indices])
        return torch.sparse_coo_tensor(
            indices, values, (batch_size, self.vocabulary_size), device=self.device
        ).to_dense()

    def get_vocabulary_size(self) -> int:
        return self.vocabulary_size

    def _create_inverse_phoneme_mapping(self):
        """Create an inverse mapping from phoneme vectors to phoneme strings."""
        self.phoneme_vectors_to_strings: dict[tuple, list[str]] = {}

        self.all_phoneme_names = self.phoneme_table.phonemes
        for row, phoneme in enumerate(self.all_phoneme_names):
            vector_tuple = tuple(
                self.phoneme_table.phonetic_features[row].cpu().numpy().astype(int)
            )
            self.phoneme_vectors_to_strings.setdefault(vector_tuple, []).append(phoneme)

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
            [distance_fn(phoneme_vector, rep) for rep in self.phoneme_table.phonetic_features],
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
