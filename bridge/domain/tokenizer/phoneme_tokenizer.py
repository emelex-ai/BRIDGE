import functools
import json
import logging
import os

import torch

from bridge.core.phonreps import SPECIAL_TOKENS, load_phoneme_table
from bridge.domain.datamodels.encodings import Target, Tokens, pad_rows
from bridge.domain.tokenizer.languages import (
    UNSPECIFIED,
    Languages,
    per_row_languages,
    split_words,
)
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

    :meth:`encode_input` and :meth:`encode_target` emit ``(batch, sequence)`` row ids, the
    same shape the character tokenizer emits for orthography. Feature vectors appear in
    only two places: the ``labels`` of a target, and the output of :meth:`decode`.

    The tokenizer is multilingual by default: supply ``lang_codes`` to control which
    language lexicons are loaded, and pass ``languages`` to either encoder to say which
    language each word is in.
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

        # Per-phoneme loss labels, gathered by row id in `encode_target`.
        #
        # Two things make this table differ from `phoneme_table.multihot`:
        #   * the [PAD] feature column is dropped (labels are one class per feature,
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

    def _get_phrase_phonemes(self, phrase: str, word_langs: tuple[str, ...]) -> list | None:
        """The phonemes of a row, each word looked up in its own language's lexicon.

        Words are joined by ``[SPC]``, and each contributes its first pronunciation
        variant. ``None`` if a word is missing from its language's lexicon, whichever other
        lexicons hold it: a word tagged with one language is never pronounced in another.
        An unspecified language reads the English lexicon.
        """
        result: list = []
        for i, word in enumerate(split_words(phrase)):
            lexicon = "en" if word_langs[i] == UNSPECIFIED else word_langs[i].lower()
            entry = self.pronunciation_dict.get(word.group().lower(), {})
            if lexicon not in entry:
                logger.debug(f"No {lexicon} pronunciation for '{word.group()}'")
                return None
            if i:
                result.append("[SPC]")
            result.extend(entry[lexicon][0])
        return result

    def encode_input(self, texts: list[str], languages: Languages = None) -> Tokens | None:
        """Encode rows as what the phonological encoder reads.

        Each row is laid out as ``[BOS, ...phonemes, EOS, PAD, ...]`` in *row-space* ids:
        indices into :class:`~bridge.core.phonreps.PhonemeTable`, i.e. "which phoneme", not
        "which feature". The model recovers the features by gathering from the same table;
        see the ``PhonemeTable`` docstring for the two id spaces.

        ``texts`` is a list of rows. ``languages`` says which language each word is in, one
        entry per row, in the forms
        :func:`~bridge.domain.tokenizer.languages.per_row_languages` accepts. Each word is
        looked up by its position in the row, so one spelling can be English in one place
        and Spanish in another. Returns ``None`` if any word is missing from its
        language's pronunciation lexicon.
        """
        rows = self._rows(texts, languages)
        if rows is None:
            return None
        eos = self.phoneme_table.row_index["[EOS]"]
        return Tokens(*self._padded([[*row, eos] for row in rows]))

    def encode_target(self, texts: list[str], languages: Languages = None) -> Target | None:
        """Encode rows as what the phonological decoder is fed and scored against.

        The ids are ``[BOS, ...phonemes, PAD, ...]`` in row space. The labels are the same
        row one position on, ``[...phonemes, EOS, PAD, ...]``, as feature vectors: one class
        per phonetic feature, in feature space. A padded position carries the ``[PAD]``
        feature id throughout, which is the loss ``ignore_index``.

        ``texts``, ``languages`` and the ``None`` return are as for :meth:`encode_input`.
        """
        rows = self._rows(texts, languages)
        if rows is None:
            return None
        ids, pad_mask = self._padded(rows)
        eos = self.phoneme_table.row_index["[EOS]"]
        label_rows, _ = self._padded([[*row[1:], eos] for row in rows])
        return Target(ids, pad_mask, self._target_table[label_rows])

    def _rows(self, texts: list[str], languages: Languages) -> list[list[int]] | None:
        """Each row as ``[BOS, ...phoneme row ids]``, unpadded, or ``None`` if a word has no
        pronunciation in its language.

        Every row's languages are resolved before any word is looked up, so an argument
        error raises whatever the lexicon holds.
        """
        table = self.phoneme_table
        rows = []
        for text, word_langs in zip(texts, per_row_languages(texts, languages), strict=True):
            phonemes = self._get_phrase_phonemes(text, word_langs)
            if phonemes is None:
                return None
            rows.append([table.row_index["[BOS]"], *(table.row_of(p) for p in phonemes)])
        return rows

    def _padded(self, rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        return pad_rows(rows, self.phoneme_table.row_index["[PAD]"], self.device)

    def decode(self, indices_batch: list[list[int]]) -> torch.Tensor:
        """Convert *feature* indices to dense multi-hot vectors, one row per position.

        The input space is feature space: what ``GenerationOutput.phon_tokens`` holds.
        It is deliberately not the inverse of :meth:`encode_input`, which emits phoneme
        *row* ids; use ``phoneme_table.features_of`` to go from a row id to its features.
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
        self.phoneme_vectors_to_strings: dict[tuple, list[str]] = {}

        self.all_phoneme_names = self.phoneme_table.phonemes
        for row, phoneme in enumerate(self.all_phoneme_names):
            vector_tuple = tuple(
                self.phoneme_table.phonetic_features[row].cpu().numpy().astype(int)
            )
            self.phoneme_vectors_to_strings.setdefault(vector_tuple, []).append(phoneme)

    def phoneme_vector_to_phoneme(self, phoneme_vector, distance_fn=None, top_k=1):
        """The phoneme names a feature vector stands for.

        Args:
            phoneme_vector: The phonetic features of one phoneme. A full-width vector,
                special-token columns included, is accepted: one whose only active entry
                is a special-token column returns that token, and any other is cut to its
                phonetic features.
            distance_fn: Called as ``distance_fn(vector, phoneme)`` when nothing matches
                exactly. Hamming distance when ``None``.
            top_k: How many of the nearest phonemes to return when nothing matches exactly.

        Returns:
            Every phoneme whose features match exactly, as names, whatever ``top_k`` is.
            Otherwise the nearest ``top_k``: names when ``top_k`` is 1, and
            ``(name, distance)`` pairs when it is more.
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
        """One phoneme name per vector, not a spelled word. A 0-dimensional tensor is read
        as the feature id of a special token."""
        phonemes = []

        for vector in phoneme_vectors:
            if isinstance(vector, torch.Tensor) and vector.dim() == 0:
                phonemes.append(self._special_token_by_dim.get(float(vector.item()), "[UNK]"))
            else:
                matches = self.phoneme_vector_to_phoneme(vector, distance_fn, top_k=1)
                phonemes.append(matches[0] if matches else "[UNK]")

        return phonemes
