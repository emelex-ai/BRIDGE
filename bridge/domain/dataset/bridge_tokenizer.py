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

import torch
import logging
from typing import Literal, Optional, List, Union, Dict, Any
from bridge.domain.datamodels.encodings import BridgeEncoding, EncodingComponent
from bridge.domain.dataset.character_tokenizer import CharacterTokenizer
from bridge.domain.dataset.phoneme_tokenizer import PhonemeTokenizer
from bridge.utils import device_manager

logger = logging.getLogger(__name__)


class BridgeTokenizer:
    """
    A wrapper tokenizer that combines orthographic (character-based) and
    phonological (phoneme-based) tokenization.

    Supports filtering by modality to handle nonwords that only exist in one modality.
    """

    def __init__(
        self,
        phoneme_cache_size: int = 10000,
        custom_cmudict_path: Optional[str] = None,
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
        self.phon_eos_id = self.phoneme_tokenizer.special_token_dims["[EOS]"]
        self.orth_bos_id = self.char_tokenizer.char_2_idx["[BOS]"]
        self.orth_pad_id = self.char_tokenizer.char_2_idx["[PAD]"]
        self.orth_eos_id = self.char_tokenizer.char_2_idx["[EOS]"]

        logger.info(
            f"BridgeTokenizer initialized on device {self.device} "
            f"with vocabulary sizes - Orthographic: {self.char_tokenizer.get_vocabulary_size()}, "
            f"Phonological: {self.phoneme_tokenizer.get_vocabulary_size()}"
        )

    def encode(
        self,
        text: str | list[str],
        modality_filter: Literal["both", "orthography", "phonology"] = "both",
    ) -> BridgeEncoding | None:
        """
        Encode text using tokenizers based on the specified modality filter.

        Args:
            text: Single string or list of strings to encode
            modality_filter: Which modality to encode:
                - "both": Encode both orthography and phonology (default)
                - "orthography": Encode only orthography, create placeholder phonology
                - "phonology": Encode only phonology, create placeholder orthography

        Returns:
            BridgeEncoding containing encodings according to the modality filter,
            or None if encoding fails based on the filter rules
        """

        # Validate modality filter
        if modality_filter not in ["both", "orthography", "phonology"]:
            logger.error(f"Invalid modality_filter: {modality_filter}")
            raise ValueError(
                f"Invalid modality_filter: {modality_filter}. "
                f"Must be one of ['both', 'orthography', 'phonology']"
            )

        # Get orthographic encoding if needed
        ortho_encoding = None
        if modality_filter in ["both", "orthography"]:
            try:
                logger.debug("Attempting orthographic encoding...")
                ortho_encoding = self.char_tokenizer.encode(text)

                if ortho_encoding is None:
                    logger.error(
                        f"Orthographic encoding returned None for text: {text_preview}, "
                        f"modality_filter: {modality_filter}"
                    )
                else:
                    logger.debug(
                        f"Orthographic encoding successful. Keys: {list(ortho_encoding.keys()) if ortho_encoding else 'None'}"
                    )

            except Exception as e:
                logger.error(
                    f"Orthographic encoding failed with exception: {type(e).__name__}: {str(e)}, "
                    f"text: {text_preview}, modality_filter: {modality_filter}",
                    exc_info=True,
                )
                ortho_encoding = None

        # Get phonological encoding if needed
        phono_encoding = None
        if modality_filter in ["both", "phonology"]:
            try:
                logger.debug("Attempting phonological encoding...")
                phono_encoding = self.phoneme_tokenizer.encode(text)

                if phono_encoding is None:
                    logger.warning(
                        f"Phonological encoding returned None - word not found in CMU dictionary. "
                        f"Text: {text_preview}, modality_filter: {modality_filter}"
                    )
                    # If phonological encoding fails and it's required, return None
                    if modality_filter in ["both", "phonology"]:
                        logger.error(
                            f"Phonological encoding required but failed for modality_filter: {modality_filter}"
                        )
                        return None
                else:
                    logger.debug(
                        f"Phonological encoding successful. Keys: {list(phono_encoding.keys()) if phono_encoding else 'None'}"
                    )

            except Exception as e:
                logger.error(
                    f"Phonological encoding failed with exception: {type(e).__name__}: {str(e)}, "
                    f"text: {text_preview}, modality_filter: {modality_filter}",
                    exc_info=True,
                )
                phono_encoding = None

                # If phonological encoding fails and it's required, return None
                if modality_filter in ["both", "phonology"]:
                    logger.error(
                        f"Phonological encoding required but failed due to exception for modality_filter: {modality_filter}"
                    )
                    return None

        # Build the BridgeEncoding based on the modality filter
        try:
            if modality_filter == "both":
                # Classic behavior - need both encodings to succeed
                if ortho_encoding is None or phono_encoding is None:
                    logger.error(
                        f"Both encodings required but one failed. "
                        f"ortho_encoding is None: {ortho_encoding is None}, "
                        f"phono_encoding is None: {phono_encoding is None}, "
                        f"text: {text_preview}"
                    )
                    return None

                logger.debug(
                    "Creating orthographic and phonological components for 'both' mode..."
                )

                # Create orthographic component
                try:
                    orthographic = EncodingComponent(
                        enc_input_ids=ortho_encoding["enc_input_ids"],
                        enc_pad_mask=ortho_encoding["enc_pad_mask"],
                        dec_input_ids=ortho_encoding["dec_input_ids"],
                        dec_pad_mask=ortho_encoding["dec_pad_mask"],
                    )
                    logger.debug("Orthographic component created successfully")
                except Exception as e:
                    logger.error(
                        f"Failed to create orthographic component: {type(e).__name__}: {str(e)}, "
                        f"ortho_encoding keys: {list(ortho_encoding.keys()) if ortho_encoding else 'None'}",
                        exc_info=True,
                    )
                    return None

                # Create phonological component
                try:
                    phonological = EncodingComponent(
                        enc_input_ids=phono_encoding["enc_input_ids"],
                        enc_pad_mask=phono_encoding["enc_pad_mask"],
                        dec_input_ids=phono_encoding["dec_input_ids"],
                        dec_pad_mask=phono_encoding["dec_pad_mask"],
                        targets=phono_encoding["targets"],
                    )
                    logger.debug("Phonological component created successfully")
                except Exception as e:
                    logger.error(
                        f"Failed to create phonological component: {type(e).__name__}: {str(e)}, "
                        f"phono_encoding keys: {list(phono_encoding.keys()) if phono_encoding else 'None'}",
                        exc_info=True,
                    )
                    return None

                # Create final BridgeEncoding
                try:
                    bridge_encoding = BridgeEncoding(
                        orthographic=orthographic,
                        phonological=phonological,
                        device=self.device,
                    )
                    logger.debug(
                        f"BridgeEncoding created successfully for 'both' mode, device: {self.device}"
                    )
                    return bridge_encoding
                except Exception as e:
                    logger.error(
                        f"Failed to create BridgeEncoding: {type(e).__name__}: {str(e)}, "
                        f"device: {self.device}",
                        exc_info=True,
                    )
                    return None

            elif modality_filter == "orthography":
                # Orthography-only mode for o2p pathway with nonwords
                if ortho_encoding is None:
                    logger.error(
                        f"Orthographic encoding required but failed for 'orthography' mode, text: {text_preview}"
                    )
                    return None

                logger.debug("Creating components for 'orthography' mode...")

                # Create orthographic component
                try:
                    orthographic = EncodingComponent(
                        enc_input_ids=ortho_encoding["enc_input_ids"],
                        enc_pad_mask=ortho_encoding["enc_pad_mask"],
                        dec_input_ids=ortho_encoding["dec_input_ids"],
                        dec_pad_mask=ortho_encoding["dec_pad_mask"],
                    )
                    logger.debug("Orthographic component created successfully")
                except Exception as e:
                    logger.error(
                        f"Failed to create orthographic component in 'orthography' mode: {type(e).__name__}: {str(e)}",
                        exc_info=True,
                    )
                    return None

                # Create placeholder phonological component
                try:
                    batch_size = len(text) if isinstance(text, list) else 1
                    seq_len = 1  # Minimal length
                    logger.debug(
                        f"Creating placeholder phonological component with batch_size: {batch_size}, seq_len: {seq_len}"
                    )

                    phonological = self._create_placeholder_phonological(
                        batch_size, seq_len
                    )
                    logger.debug(
                        "Placeholder phonological component created successfully"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to create placeholder phonological component: {type(e).__name__}: {str(e)}, "
                        f"batch_size: {len(text) if isinstance(text, list) else 1}",
                        exc_info=True,
                    )
                    return None

                # Create final BridgeEncoding
                try:
                    bridge_encoding = BridgeEncoding(
                        orthographic=orthographic,
                        phonological=phonological,
                        device=self.device,
                    )
                    logger.debug(
                        f"BridgeEncoding created successfully for 'orthography' mode, device: {self.device}"
                    )
                    return bridge_encoding
                except Exception as e:
                    logger.error(
                        f"Failed to create BridgeEncoding in 'orthography' mode: {type(e).__name__}: {str(e)}",
                        exc_info=True,
                    )
                    return None

            elif modality_filter == "phonology":
                # Phonology-only mode (p2o pathway)
                if phono_encoding is None:
                    logger.error(
                        f"Phonological encoding required but failed for 'phonology' mode, text: {text_preview}"
                    )
                    return None

                logger.debug("Creating components for 'phonology' mode...")

                # Create placeholder orthographic component
                try:
                    batch_size = len(text) if isinstance(text, list) else 1
                    seq_len = 1  # Minimal length
                    logger.debug(
                        f"Creating placeholder orthographic component with batch_size: {batch_size}, seq_len: {seq_len}"
                    )

                    orthographic = self._create_placeholder_orthographic(
                        batch_size, seq_len
                    )
                    logger.debug(
                        "Placeholder orthographic component created successfully"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to create placeholder orthographic component: {type(e).__name__}: {str(e)}, "
                        f"batch_size: {len(text) if isinstance(text, list) else 1}",
                        exc_info=True,
                    )
                    return None

                # Create phonological component
                try:
                    phonological = EncodingComponent(
                        enc_input_ids=phono_encoding["enc_input_ids"],
                        enc_pad_mask=phono_encoding["enc_pad_mask"],
                        dec_input_ids=phono_encoding["dec_input_ids"],
                        dec_pad_mask=phono_encoding["dec_pad_mask"],
                        targets=phono_encoding["targets"],
                    )
                    logger.debug("Phonological component created successfully")
                except Exception as e:
                    logger.error(
                        f"Failed to create phonological component in 'phonology' mode: {type(e).__name__}: {str(e)}",
                        exc_info=True,
                    )
                    return None

                # Create final BridgeEncoding
                try:
                    bridge_encoding = BridgeEncoding(
                        orthographic=orthographic,
                        phonological=phonological,
                        device=self.device,
                    )
                    logger.debug(
                        f"BridgeEncoding created successfully for 'phonology' mode, device: {self.device}"
                    )
                    return bridge_encoding
                except Exception as e:
                    logger.error(
                        f"Failed to create BridgeEncoding in 'phonology' mode: {type(e).__name__}: {str(e)}",
                        exc_info=True,
                    )
                    return None

        except Exception as e:
            logger.error(
                f"Unexpected error in encode method: {type(e).__name__}: {str(e)}, "
                f"text: {text_preview}, modality_filter: {modality_filter}",
                exc_info=True,
            )
            return None

        # This should never be reached, but adding as a safety net
        logger.error(
            f"Encode method reached unexpected end without returning, modality_filter: {modality_filter}"
        )
        return None

    def _create_placeholder_phonological(
        self, batch_size: int, seq_len: int
    ) -> EncodingComponent:
        """Create a minimal phonological component for orthography-only encoding."""
        # Create placeholder phonological tensors
        placeholder_feature_indices = torch.tensor(
            [self.phon_pad_id], dtype=torch.long, device=self.device
        )

        # Each item in each batch gets a pad token feature index
        placeholder_enc_input_ids = [
            [placeholder_feature_indices.clone()] for _ in range(batch_size)
        ]
        placeholder_dec_input_ids = [
            [placeholder_feature_indices.clone()] for _ in range(batch_size)
        ]

        # Create padding masks
        placeholder_enc_pad_mask = torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=self.device
        )
        placeholder_dec_pad_mask = torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=self.device
        )

        # Create target tensor
        phon_vocab_size = self.phoneme_tokenizer.get_vocabulary_size()
        placeholder_targets = torch.zeros(
            (batch_size, seq_len, phon_vocab_size), dtype=torch.long, device=self.device
        )

        return EncodingComponent(
            enc_input_ids=placeholder_enc_input_ids,
            enc_pad_mask=placeholder_enc_pad_mask,
            dec_input_ids=placeholder_dec_input_ids,
            dec_pad_mask=placeholder_dec_pad_mask,
            targets=placeholder_targets,
        )

    def _create_placeholder_orthographic(
        self, batch_size: int, seq_len: int
    ) -> EncodingComponent:
        """Create a minimal orthographic component for phonology-only encoding."""
        # Create placeholder orthographic tensors
        placeholder_enc_input_ids = torch.zeros(
            (batch_size, seq_len), dtype=torch.long, device=self.device
        )
        placeholder_dec_input_ids = torch.zeros(
            (batch_size, seq_len), dtype=torch.long, device=self.device
        )

        # Create padding masks
        placeholder_enc_pad_mask = torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=self.device
        )
        placeholder_dec_pad_mask = torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=self.device
        )

        return EncodingComponent(
            enc_input_ids=placeholder_enc_input_ids,
            enc_pad_mask=placeholder_enc_pad_mask,
            dec_input_ids=placeholder_dec_input_ids,
            dec_pad_mask=placeholder_dec_pad_mask,
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
        # If nothing passed in return nothing
        if ortho_indices is None and phono_indices is None:
            return None

        # Decode orthographic representation
        encodings = dict()
        ortho_decoded = None
        if ortho_indices is not None:
            ortho_decoded = self.char_tokenizer.decode(ortho_indices)
            encodings["orthographic"] = ortho_decoded

        # Decode phonological representation if provided
        phono_decoded = None
        if phono_indices is not None:
            phono_decoded = self.phoneme_tokenizer.decode(phono_indices)
            encodings["phonological"] = phono_decoded

        return encodings

    def get_vocabulary_sizes(self) -> dict:
        """Return vocabulary sizes for both tokenizers."""
        return {
            "orthographic": self.char_tokenizer.get_vocabulary_size(),
            "phonological": self.phoneme_tokenizer.get_vocabulary_size(),
        }

    def phoneme_vectors_to_word(self, phoneme_vectors, distance_fn=None):
        """Convert phoneme vectors to word."""
        return self.phoneme_tokenizer.phoneme_vectors_to_word(
            phoneme_vectors, distance_fn
        )

    def phoneme_vector_to_phoneme(self, phoneme_vector, distance_fn=None, top_k=1):
        """Convert a single phoneme vector to phoneme string(s)."""
        return self.phoneme_tokenizer.phoneme_vector_to_phoneme(
            phoneme_vector, distance_fn, top_k
        )
