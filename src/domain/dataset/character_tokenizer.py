from src.utils.device_manager import device_manager
from src.domain.dataset import CUDADict
import logging
import torch
import string

logger = logging.getLogger(__name__)


class CharacterTokenizer:
    def __init__(self):
        # Set device and default to CPU
        self.device = device_manager.device

        # Initialize vocabulary with special tokens
        self.special_tokens = ["[BOS]", "[EOS]", "[PAD]", "[UNK]", "[CLS]", "[SEP]"]

    #heterophonic homographs
        # Language tokens help the model resolve interlingual homographs. '--' is a placeholder for unspecified languages that preserves tensor shape.
        self.language_tokens = ["--", "EN", "ES"]
        self.vocab = self.special_tokens + self.language_tokens + list(string.printable)

        # Create mappings from characters to indices and vice versa
        self.char_2_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_2_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocabulary_size = len(self.vocab)

        logger.info(
            f"CharacterTokenizer initialized with vocabulary size: {self.vocabulary_size}"
        )

    def get_vocabulary_size(self) -> int:
        return self.vocabulary_size

    def encode(
        self,
        list_of_strings: str | list[str],
        language_map: dict[str, str] = None,
    ) -> CUDADict:
        """
        Encode a list of strings into a tensor representation.
        Args:
            list_of_strings (str | list[str]): A string or a list of strings to be encoded.
            language_map (dict[str, str], optional): A mapping of words to their corresponding languages.
                Defaults to None, which means no language mapping is applied. Keys are words and
                values are language codes (e.g., "EN", "ES"). If a word is not in the map, it defaults to "--".
        Returns:
            CUDADict: A dictionary containing the encoded input IDs and padding masks.
        """

        # Ensure the languages in the map are valid
        if language_map is None:
            language_map = {}
        for lang in language_map.values():
            if lang.upper() not in self.language_tokens:
                err_str = f"Invalid language: {lang}. Supported languages are: {self.language_tokens[:-1]}"
                logger.error(err_str)
                raise ValueError(err_str)

        # Ensure the input is either a string or a list of strings
        if isinstance(list_of_strings, str):
            list_of_strings = [list_of_strings]
        elif not isinstance(list_of_strings, list) or not all(
            isinstance(s, str) for s in list_of_strings
        ):
            logger.error("Input must be a string or a list of strings")
            raise TypeError("Input must be a string or a list of strings")

        max_length = max(len(s) for s in list_of_strings)

        enc_pad = (
            lambda s: [language_map.get(s.upper(), "--")]
            + ["[BOS]"]
            + list(s)
            + ["[EOS]"]
            + ["[PAD]"] * (max_length - len(s))
        )
        dec_pad = (
            lambda s: [language_map.get(s.upper(), "--")]
            + ["[BOS]"]
            + list(s)
            + ["[PAD]"] * (max_length - len(s))
        )

        # Create encoder-padded and decoder-padded string lists
        enc_strings = [enc_pad(s) for s in list_of_strings]
        dec_strings = [dec_pad(s) for s in list_of_strings]

        # Initialize tensors for encoded input
        enc_input_ids = torch.zeros(
            # +3 for [LANG], [BOS], [EOS]
            (len(enc_strings), 3 + max_length),
            dtype=torch.long,
            device=self.device,
        )
        for i, enc_str in enumerate(enc_strings):
            for j, ch in enumerate(enc_str):
                enc_input_ids[i, j] = self.char_2_idx.get(ch, self.char_2_idx["[UNK]"])

        # Initialize tensors for decoder input
        dec_input_ids = torch.zeros(
            # +2 for [LANG], [BOS]
            (len(dec_strings), 2 + max_length),
            dtype=torch.long,
            device=self.device,
        )
        for i, dec_str in enumerate(dec_strings):
            for j, ch in enumerate(dec_str):
                dec_input_ids[i, j] = self.char_2_idx.get(ch, self.char_2_idx["[UNK]"])

        PAD_TOKEN = self.char_2_idx["[PAD]"]
        enc_pad_mask = enc_input_ids == PAD_TOKEN
        dec_pad_mask = dec_input_ids == PAD_TOKEN

        return CUDADict(
            {
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": dec_input_ids,
                "enc_pad_mask": enc_pad_mask.bool(),
                "dec_pad_mask": dec_pad_mask.bool(),
            }
        )

    def decode(self, list_of_ints: list[list[int]]) -> list[str]:
        try:
            decoded_strings = [
                "".join(
                    [
                        self.idx_2_char[i]
                        for i in ints
                        if self.idx_2_char[i] not in self.special_tokens
                    ]
                )
                for ints in list_of_ints
            ]
        except KeyError as e:
            logger.error(f"Invalid index encountered during decoding: {e}")
            raise KeyError from e

        return decoded_strings
