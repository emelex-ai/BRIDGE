import logging
import string

import torch

from bridge.domain.tokenizer.cuda_dict import CUDADict
from bridge.utils import device_manager

logger = logging.getLogger(__name__)


class CharacterTokenizer:
    def __init__(self):
        self.device = device_manager.device

        self.special_tokens = ["[BOS]", "[EOS]", "[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        # Language tokens disambiguate interlingual homographs (e.g. English "read"
        # vs Spanish "leer" can share spellings or share phonemes in mixed corpora).
        # ``"--"`` is the unspecified-language placeholder that preserves tensor shape.
        self.language_tokens = ["--", "EN", "ES"]
        self.vocab = self.special_tokens + self.language_tokens + list(string.printable)

        self.char_2_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_2_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocabulary_size = len(self.vocab)

        logger.info(f"CharacterTokenizer initialized with vocabulary size: {self.vocabulary_size}")

    def get_vocabulary_size(self) -> int:
        return self.vocabulary_size

    def encode(
        self,
        list_of_strings: str | list[str],
        language_map: dict[str, str] | None = None,
    ) -> CUDADict:
        """Encode strings to orthographic feature indices, prepending a language token.

        Each sequence is laid out as ``[LANG, BOS, ...chars, EOS, PAD, ...]`` for encoder
        input and ``[LANG, BOS, ...chars, PAD, ...]`` for decoder input. Slicing past the
        BOS therefore requires ``[:, 2:]`` (skip both the language token and BOS).

        Args:
            list_of_strings: A string or list of strings to encode.
            language_map: Optional ``{word: language_code}`` map. Keys are matched
                case-insensitively against each string. Values must be in
                ``self.language_tokens`` (``"--"``, ``"EN"``, ``"ES"``). Missing words
                default to ``"--"``.

        Returns:
            CUDADict with ``enc_input_ids``, ``dec_input_ids``, ``enc_pad_mask``,
            ``dec_pad_mask``.
        """
        if language_map is None:
            language_map = {}
        valid_langs = set(self.language_tokens)
        invalid_langs = [lang for lang in language_map.values() if lang.upper() not in valid_langs]
        if invalid_langs:
            err_str = (
                f"Invalid languages: {invalid_langs}. "
                f"Supported languages are: {self.language_tokens[1:]}"
            )
            logger.error(err_str)
            raise ValueError(err_str)

        if isinstance(list_of_strings, str):
            list_of_strings = [list_of_strings]
        elif not isinstance(list_of_strings, list) or not all(
            isinstance(s, str) for s in list_of_strings
        ):
            logger.error("Input must be a string or a list of strings")
            raise TypeError("Input must be a string or a list of strings")

        max_length = max(len(s) for s in list_of_strings)

        def enc_pad(s: str) -> list[str]:
            return (
                [language_map.get(s.lower(), "--")]
                + ["[BOS]"]
                + list(s)
                + ["[EOS]"]
                + ["[PAD]"] * (max_length - len(s))
            )

        def dec_pad(s: str) -> list[str]:
            return (
                [language_map.get(s.lower(), "--")]
                + ["[BOS]"]
                + list(s)
                + ["[PAD]"] * (max_length - len(s))
            )

        enc_strings = [enc_pad(s) for s in list_of_strings]
        dec_strings = [dec_pad(s) for s in list_of_strings]

        # +3 for [LANG], [BOS], [EOS]
        enc_input_ids = torch.zeros(
            (len(enc_strings), 3 + max_length), dtype=torch.long, device=self.device
        )
        for i, enc_str in enumerate(enc_strings):
            for j, ch in enumerate(enc_str):
                enc_input_ids[i, j] = self.char_2_idx.get(ch, self.char_2_idx["[UNK]"])

        # +2 for [LANG], [BOS]
        dec_input_ids = torch.zeros(
            (len(dec_strings), 2 + max_length), dtype=torch.long, device=self.device
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
                        if (
                            self.idx_2_char[i] not in self.special_tokens
                            and self.idx_2_char[i] not in self.language_tokens
                        )
                    ]
                )
                for ints in list_of_ints
            ]
        except KeyError as e:
            logger.error(f"Invalid index encountered during decoding: {e}")
            raise KeyError from e

        return decoded_strings
