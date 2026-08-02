"""
The BridgeTokenizer is a wrapper class that combines orthographic (character-based) and
phonological (phoneme-based) tokenization to support the various pathways in the BRIDGE model.

Objects:
- CharacterTokenizer: Simple tokenizer for mapping text to character indices
- PhonemeTokenizer: More complex tokenizer that maps words to phonetic features
- BridgeTokenizer: A wrapper around both tokenizers

The CharacterTokenizer is lean and simple. It contains all of the logic for encoding
and decoding strings into token tensors.

The PhonemeTokenizer handles mapping words to phonetic feature vectors using CMUDict
and phonreps.csv. It requires words to exist in the CMUDict to work properly.

The BridgeTokenizer handles combining both encoding types into a unified format, with
special handling for nonwords (words not in CMUDict).
"""

import logging
from typing import Literal

import torch

from bridge.domain.datamodels.encodings import BridgeEncoding, EncodingComponent
from bridge.domain.tokenizer.character_tokenizer import CharacterTokenizer
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
        phoneme_cache_size: int = 10000,
        custom_cmudict_path: str | None = None,
    ):
        # Initialize device
        self.device = device_manager.device
        # Initialize both tokenizers with the same device
        self.char_tokenizer = CharacterTokenizer()
        self.phoneme_tokenizer = PhonemeTokenizer(
            max_cache_size=phoneme_cache_size, custom_cmudict_path=custom_cmudict_path
        )
        self.phon_bos_id = self.phoneme_tokenizer.special_token_dims["[BOS]"]
        self.phon_pad_id = self.phoneme_tokenizer.special_token_dims["[PAD]"]
        self.phon_spc_id = self.phoneme_tokenizer.special_token_dims["[SPC]"]
        self.phon_eos_id = self.phoneme_tokenizer.special_token_dims["[EOS]"]
        self.orth_bos_id = self.char_tokenizer.char_2_idx["[BOS]"]
        self.orth_pad_id = self.char_tokenizer.char_2_idx["[PAD]"]
        self.orth_spc_id = self.char_tokenizer.char_2_idx[" "]
        self.orth_eos_id = self.char_tokenizer.char_2_idx["[EOS]"]

        logger.info(
            f"BridgeTokenizer initialized on device {self.device} "
            f"with vocabulary sizes - Orthographic: {self.char_tokenizer.get_vocabulary_size()}, "
            f"Phonological: {self.phoneme_tokenizer.get_vocabulary_size()}"
        )

    def encode(
        self,
        text: str | list[str],
        modality_filter: ModalityFilter = "both",
        language_map: dict[str, str] | None = None,
    ) -> BridgeEncoding | None:
        """
        Encode text using tokenizers based on the specified modality filter.

        Args:
            text: Single string or list of strings to encode
            modality_filter: Which modality to encode:
                - "both": Encode both orthography and phonology (default)
                - "orthography": Encode only orthography, create placeholder phonology
                - "phonology": Encode only phonology, create placeholder orthography
            language_map: Optional ``{word: language_code}`` mapping for code-switching.
                Passed to both child tokenizers. The character tokenizer accepts uppercase
                codes (``"EN"``, ``"ES"``); the phoneme tokenizer accepts lowercase ones
                (``"en"``, ``"es"``). Missing words default to ``"--"`` orthographically
                and ``"en"`` phonologically.

        Returns:
            BridgeEncoding containing encodings according to the modality filter,
            or None if encoding fails based on the filter rules
        """
        if modality_filter not in ("both", "orthography", "phonology"):
            logger.error(f"Invalid modality_filter: {modality_filter}")
            raise ValueError(
                f"Invalid modality_filter: {modality_filter}. "
                f"Must be one of ['both', 'orthography', 'phonology']"
            )

        batch_size = len(text) if isinstance(text, list) else 1

        try:
            # Each modality is either encoded for real or stubbed with a placeholder;
            # `BridgeEncoding` always carries both components.
            if modality_filter in ("both", "orthography"):
                orthographic = self.char_tokenizer.encode(text, language_map=language_map)
                if orthographic is None:
                    logger.error(f"Orthographic encoding failed for text: {text}")
                    return None
            else:
                orthographic = self._create_placeholder_orthographic(batch_size)

            if modality_filter in ("both", "phonology"):
                phonological = self.phoneme_tokenizer.encode(text, language_map=language_map)
                if phonological is None:
                    logger.error(
                        f"Phonological encoding failed — word not found in the pronunciation "
                        f"lexicon. Text: {text}, modality_filter: {modality_filter}"
                    )
                    return None
            else:
                phonological = self._create_placeholder_phonological(batch_size)

            return BridgeEncoding(
                orthographic=orthographic,
                phonological=phonological,
                device=self.device,
            )
        except Exception:
            logger.exception(
                f"Encoding failed for text: {text}, modality_filter: {modality_filter}"
            )
            return None

    def _placeholder_pad_masks(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder/decoder pad masks for a single all-padding placeholder position."""
        mask = torch.ones((batch_size, 1), dtype=torch.bool, device=self.device)
        return mask, mask.clone()

    def _create_placeholder_phonological(self, batch_size: int) -> EncodingComponent:
        """Create a minimal phonological component for orthography-only encoding."""
        pad_index = torch.tensor([self.phon_pad_id], dtype=torch.long, device=self.device)
        enc_pad_mask, dec_pad_mask = self._placeholder_pad_masks(batch_size)

        return EncodingComponent(
            enc_input_ids=[[pad_index.clone()] for _ in range(batch_size)],
            enc_pad_mask=enc_pad_mask,
            dec_input_ids=[[pad_index.clone()] for _ in range(batch_size)],
            dec_pad_mask=dec_pad_mask,
            targets=torch.zeros(
                (batch_size, 1, self.phoneme_tokenizer.get_vocabulary_size()),
                dtype=torch.long,
                device=self.device,
            ),
        )

    def _create_placeholder_orthographic(self, batch_size: int) -> EncodingComponent:
        """Create a minimal orthographic component for phonology-only encoding."""
        zeros = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        enc_pad_mask, dec_pad_mask = self._placeholder_pad_masks(batch_size)

        return EncodingComponent(
            enc_input_ids=zeros,
            enc_pad_mask=enc_pad_mask,
            dec_input_ids=zeros.clone(),
            dec_pad_mask=dec_pad_mask,
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
            phono_indices: Optional list of lists of phoneme feature indices

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
