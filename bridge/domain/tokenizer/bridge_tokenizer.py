"""
The BridgeTokenizer is a wrapper class that combines orthographic (character-based) and
phonological (phoneme-based) tokenization to support the various pathways in the BRIDGE model.

Objects:
- CharacterTokenizer: maps text to character ids
- PhonemeTokenizer: maps words to phoneme row ids
- BridgeTokenizer: A wrapper around both tokenizers

Both tokenizers emit the same shape: a ``(batch, sequence)`` integer tensor of ids plus a
matching boolean pad mask. Only the id space differs. Character ids index the character
vocabulary; phoneme ids index rows of the feature table in ``phonreps.csv`` (*which
phoneme*, not which phonetic feature).

The PhonemeTokenizer reads the per-language pronunciation lexicons under
``bridge/core/pronunciation_lexicons/`` together with ``phonreps.csv``. It requires words
to exist in a lexicon to work properly.

The BridgeTokenizer handles combining both encoding types into a unified format, with
special handling for nonwords (words not in CMUDict).
"""

import logging
from typing import Literal

import torch

from bridge.domain.datamodels.encodings import BridgeEncoding, EncodingComponent
from bridge.domain.tokenizer.character_tokenizer import CharacterTokenizer
from bridge.domain.tokenizer.languages import Languages, rows_and_languages
from bridge.domain.tokenizer.phoneme_tokenizer import PhonemeTokenizer
from bridge.utils import device_manager

logger = logging.getLogger(__name__)

ModalityFilter = Literal["both", "orthography", "phonology"]


class BridgeTokenizer:
    """
    A wrapper tokenizer that combines orthographic (character-based) and
    phonological (phoneme-based) tokenization.

    Supports filtering by modality to handle nonwords that only exist in one modality.
    """

    def __init__(
        self,
        custom_cmudict_path: str | None = None,
    ):
        # Initialize device
        self.device = device_manager.device
        # Initialize both tokenizers with the same device
        self.custom_cmudict_path = custom_cmudict_path
        self.char_tokenizer = CharacterTokenizer()
        self.phoneme_tokenizer = PhonemeTokenizer(custom_cmudict_path=custom_cmudict_path)
        self.phon_table_fingerprint = self.phoneme_tokenizer.phoneme_table.fingerprint
        self.phon_bos_id = self.phoneme_tokenizer.special_token_dims["[BOS]"]
        self.phon_pad_id = self.phoneme_tokenizer.special_token_dims["[PAD]"]
        # Feature space, unlike the row ids `encode` emits. See VocabSpec.phon_pad_id.
        self.phon_spc_id = self.phoneme_tokenizer.special_token_dims["[SPC]"]
        self.phon_eos_id = self.phoneme_tokenizer.special_token_dims["[EOS]"]
        self.orth_bos_id = self.char_tokenizer.char_2_idx["[BOS]"]
        self.orth_pad_id = self.char_tokenizer.char_2_idx["[PAD]"]
        self.orth_spc_id = self.char_tokenizer.char_2_idx[" "]
        self.orth_eos_id = self.char_tokenizer.char_2_idx["[EOS]"]
        self.orth_language_ids = tuple(
            self.char_tokenizer.char_2_idx[token] for token in self.char_tokenizer.language_tokens
        )

        logger.info(
            f"BridgeTokenizer initialized on device {self.device} "
            f"with vocabulary sizes - Orthographic: {self.char_tokenizer.get_vocabulary_size()}, "
            f"Phonological: {self.phoneme_tokenizer.get_vocabulary_size()}"
        )

    def encode(
        self,
        text: str | list[str],
        modality_filter: ModalityFilter = "both",
        languages: Languages = None,
    ) -> BridgeEncoding | None:
        """
        Encode text using tokenizers based on the specified modality filter.

        Args:
            text: Single string or list of strings to encode
            modality_filter: Which modality to encode:
                - "both": Encode both orthography and phonology (default)
                - "orthography": Encode only orthography, create placeholder phonology
                - "phonology": Encode only phonology, create placeholder orthography
            languages: The language of every word. For a single string: one code for the
                whole row (``"ES"``), or one code per word (``"EN EN ES"``, or a list or
                tuple of codes). For a list of strings: one string applied to every row,
                or a list or tuple with one such entry per row. So ``["ES", "EN"]`` is one
                code per word beside a single string and one code per row beside a list of
                two strings. ``None`` leaves a row unspecified: the ``"--"`` token
                orthographically and the English lexicon phonologically. Orthography marks
                each change of language inside a row; phonology looks each word up in its
                own language.

        Returns:
            BridgeEncoding containing encodings according to the modality filter,
            or None if a word has no pronunciation in its language's lexicon
        """
        if modality_filter not in ("both", "orthography", "phonology"):
            raise ValueError(
                f"Invalid modality_filter: {modality_filter}. "
                f"Must be one of ['both', 'orthography', 'phonology']"
            )

        # Resolved here so an argument error is raised whichever modality is asked for.
        # Each child tokenizer resolves the result again, which returns it unchanged.
        rows, row_languages = rows_and_languages(text, languages)

        # No blanket `except Exception: return None` here. `None` means one thing, "this
        # word is not in the pronunciation lexicon", which is an ordinary data condition a
        # caller must handle. Catching everything made it mean that *and* "the caller
        # passed a language code that does not exist" *and* "something in here is broken",
        # and a caller cannot tell those apart from a `None`. A TypeError raised inside
        # BridgeEncoding surfaced 150 lines away as
        # `RuntimeError: Failed to encode word(s): long, pencil, ...`, with the real
        # message logged and discarded. Argument errors and defects propagate; a missing
        # pronunciation returns None.
        if modality_filter in ("both", "orthography"):
            orthographic = self.char_tokenizer.encode(rows, row_languages)
        else:
            orthographic = self._create_placeholder_orthographic(
                [word_langs[0] for word_langs in row_languages]
            )

        if modality_filter in ("both", "phonology"):
            phonological = self.phoneme_tokenizer.encode(rows, row_languages)
            if phonological is None:
                logger.info(
                    "No pronunciation for %r in the lexicon (modality_filter=%r)",
                    text,
                    modality_filter,
                )
                return None
        else:
            phonological = self._create_placeholder_phonological(len(rows))

        return BridgeEncoding(orthographic=orthographic, phonological=phonological)

    def _placeholder_pad_masks(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder/decoder pad masks for a single all-padding placeholder position."""
        mask = torch.ones((batch_size, 1), dtype=torch.bool, device=self.device)
        return mask, mask.clone()

    def _create_placeholder_phonological(self, batch_size: int) -> EncodingComponent:
        """Create a minimal phonological component for orthography-only encoding."""
        # Row space, not feature space: this indexes the phoneme table, so it is
        # `[PAD]`'s row rather than `self.phon_pad_id` (its feature index).
        pad_row = self.phoneme_tokenizer.phoneme_table.row_index["[PAD]"]
        pad_ids = torch.full((batch_size, 1), pad_row, dtype=torch.long, device=self.device)
        enc_pad_mask, dec_pad_mask = self._placeholder_pad_masks(batch_size)

        return EncodingComponent(
            enc_input_ids=pad_ids,
            enc_pad_mask=enc_pad_mask,
            dec_input_ids=pad_ids.clone(),
            dec_pad_mask=dec_pad_mask,
            targets=self.phoneme_tokenizer.padded_targets(batch_size, 1),
        )

    def _create_placeholder_orthographic(self, first_languages: list[str]) -> EncodingComponent:
        """A minimal but *valid* orthographic component, for phonology-only encoding.

        Two positions per row, holding ``[LANG, BOS]``, which is the shortest prefix
        ``CharacterTokenizer.encode`` can produce. ``LANG`` is the language of the row's
        first word, ``--`` when the caller gave none. A single column of zeros would not
        do: zero is ``[BOS]``, so the placeholder would look like a sequence that had
        already started.

        The width matters because :meth:`Model.generate` seeds the orthographic decoder with
        ``dec_input_ids[:, :2]``, the same prefix training puts at positions 0 and 1. A
        placeholder that cannot supply that prefix would force a special case into the model
        for the one pathway, `p2o`, that emits orthography without consuming any. The seed
        names only the first word's language; any later change is the decoder's to emit.
        """
        bos = self.char_tokenizer.char_2_idx["[BOS]"]
        prefix = torch.tensor(
            [[self.char_tokenizer.char_2_idx[language], bos] for language in first_languages],
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.ones_like(prefix, dtype=torch.bool)

        return EncodingComponent(
            enc_input_ids=prefix,
            enc_pad_mask=mask,
            dec_input_ids=prefix.clone(),
            dec_pad_mask=mask.clone(),
        )

    def decode(
        self,
        ortho_indices: list[list[int]] | None = None,
        phono_indices: list[list[int]] | None = None,
    ) -> dict | None:
        """
        Decode both orthographic and phonological representations.

        Args:
            ortho_indices: Optional List of lists of character indices
            phono_indices: Optional list of lists of phoneme *feature* indices, as held by
                ``GenerationOutput.phon_tokens``. Not the row ids ``encode`` returns.

        Returns:
            Dictionary containing decoded strings and tensors or None if no input
        """
        if ortho_indices is None and phono_indices is None:
            return None

        encodings: dict[str, list[str] | torch.Tensor] = {}
        if ortho_indices is not None:
            encodings["orthographic"] = self.char_tokenizer.decode(ortho_indices)
        if phono_indices is not None:
            encodings["phonological"] = self.phoneme_tokenizer.decode(phono_indices)

        return encodings

    def decode_with_languages(
        self, ortho_indices: list[list[int]]
    ) -> list[tuple[str, list[str | None]]]:
        """Decode orthographic rows to ``(text, language of each word)``.

        See :meth:`CharacterTokenizer.decode_with_languages`.
        """
        return self.char_tokenizer.decode_with_languages(ortho_indices)

    def get_vocabulary_sizes(self) -> dict:
        """Return vocabulary sizes for both tokenizers."""
        return {
            "orthographic": self.char_tokenizer.get_vocabulary_size(),
            "phonological": self.phoneme_tokenizer.get_vocabulary_size(),
        }

    def phoneme_vectors_to_word(self, phoneme_vectors, distance_fn=None):
        """Convert phoneme vectors to word."""
        return self.phoneme_tokenizer.phoneme_vectors_to_word(phoneme_vectors, distance_fn)

    def phoneme_vector_to_phoneme(self, phoneme_vector, distance_fn=None, top_k=1):
        """Convert a single phoneme vector to phoneme string(s)."""
        return self.phoneme_tokenizer.phoneme_vector_to_phoneme(phoneme_vector, distance_fn, top_k)
