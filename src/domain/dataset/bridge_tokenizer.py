"""
Here is what I am thinking

Objects:

CharacterTokenizer

PhonemeTokenizer

BridgeTokenizer
- CharacterTokenizer
- PhonemeTokenizer

BridgeDataset
- BridgeTokenizer

The CharacterTokenizer is lean and simple. It contains all of the logic for encoding
and decoding strings into token tensors. The vocabulary size will be the number of unique
characters it receives. No max sequence length. That will be removed by adding a context-agnostic
postional encoding to the model (ALiBi or RoPE)

The PhonemeTokenizer is more complex. It will require that CMUDict and the phonreps.csv.
It will contain all logic for encoding and decoding phonemes into token tensors. Given a word
it will look up the phonetic representation in CMUDict and then look up the vector representation
of each phoneme in phonreps.csv. We will have a fixed number of phoneme features, so the vocabulary
size will be known independent of the input data. Now max sequence length. That will be removed by
adding a context-agnostic positional encoding to the model (ALiBi or RoPE)

The BridgeTokenizer will contain both the CharacterTokenizer and the PhonemeTokenizer. It will have
a method called `build_tokenizer` that accepts a list of words. From that list of words it will 
construct the CharacterTokenizer's char_2_idx and idx_2_char dictionaries and calculate the 
orth_vocab_size. It will also construct the PhonemeTokenizer's 

"""

import torch
import logging
from src.domain.datamodels import BridgeEncoding
from src.domain.dataset.character_tokenizer import CharacterTokenizer
from src.domain.dataset.phoneme_tokenizer import PhonemeTokenizer

logger = logging.getLogger(__name__)


class BridgeTokenizer:
    """
    A wrapper tokenizer that combines orthographic (character-based) and
    phonological (phoneme-based) tokenization.
    """

    def __init__(
        self,
        device: str | torch.device | None = None,
        phoneme_cache_size: int = 10000,
    ):
        # Validate device if provided as string
        if isinstance(device, str):
            assert device in [
                "cpu",
                "cuda",
                "mps",
            ], "Device must be 'cpu', 'cuda', or 'mps'"

        # Initialize device
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

        # Initialize both tokenizers with the same device
        self.char_tokenizer = CharacterTokenizer(device=self.device)
        self.phoneme_tokenizer = PhonemeTokenizer(
            device=self.device, max_cache_size=phoneme_cache_size
        )

        logger.info(
            f"BridgeTokenizer initialized on device {self.device} "
            f"with vocabulary sizes - Orthographic: {self.char_tokenizer.get_vocabulary_size()}, "
            f"Phonological: {self.phoneme_tokenizer.get_vocabulary_size()}"
        )

    def encode(self, text: str | list[str]) -> BridgeEncoding | None:
        """
        Encode text using both tokenizers and return combined results.

        Args:
            text: Single string or list of strings to encode

        Returns:
            BridgeEncoding containing both orthographic and phonological encodings,
            or None if phonological encoding fails
        """
        # Get orthographic encoding
        ortho_encoding = self.char_tokenizer.encode(text)

        # Get phonological encoding - may return None for unknown words
        phono_encoding = self.phoneme_tokenizer.encode(text)

        # If phonological encoding fails, return None
        if phono_encoding is None:
            logger.warning(
                "Phonological encoding failed - word not found in CMU dictionary"
            )
            return None

        # Combine both encodings into an instance of BridgeEncoding
        return BridgeEncoding(
            orth_enc_ids=ortho_encoding["enc_input_ids"],
            orth_enc_mask=ortho_encoding["enc_pad_mask"],
            orth_dec_ids=ortho_encoding["dec_input_ids"],
            orth_dec_mask=ortho_encoding["dec_pad_mask"],
            phon_enc_ids=phono_encoding["enc_input_ids"],
            phon_dec_ids=phono_encoding["dec_input_ids"],
            phon_enc_mask=phono_encoding["enc_pad_mask"],
            phon_dec_mask=phono_encoding["dec_pad_mask"],
            phon_targets=phono_encoding["targets"],
            device=self.device,
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
