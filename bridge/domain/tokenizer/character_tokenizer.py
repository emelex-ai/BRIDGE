import logging
import string

import torch

from bridge.domain.datamodels.encodings import EncodingComponent
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
        # Non-content tokens stripped during decoding.
        self._non_content_tokens = set(self.special_tokens) | set(self.language_tokens)

        logger.info(f"CharacterTokenizer initialized with vocabulary size: {self.vocabulary_size}")

    def get_vocabulary_size(self) -> int:
        return self.vocabulary_size

    def encode(
        self,
        list_of_strings: str | list[str],
        language_map: dict[str, str] | None = None,
    ) -> EncodingComponent:
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
            An orthographic :class:`EncodingComponent`.
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
        unk = self.char_2_idx["[UNK]"]

        def to_ids(tokens: list[str]) -> list[int]:
            return [self.char_2_idx.get(t, unk) for t in tokens]

        def prefix(s: str) -> list[str]:
            return [language_map.get(s.lower(), "--"), "[BOS]", *s]

        # Rows are uniform length by construction: 3 + max_length for the encoder
        # ([LANG], [BOS], [EOS]) and 2 + max_length for the decoder ([LANG], [BOS]).
        enc_rows = [
            to_ids([*prefix(s), "[EOS]", *["[PAD]"] * (max_length - len(s))])
            for s in list_of_strings
        ]
        dec_rows = [
            to_ids([*prefix(s), *["[PAD]"] * (max_length - len(s))]) for s in list_of_strings
        ]

        enc_input_ids = torch.tensor(enc_rows, dtype=torch.long, device=self.device)
        dec_input_ids = torch.tensor(dec_rows, dtype=torch.long, device=self.device)

        pad_token = self.char_2_idx["[PAD]"]
        return EncodingComponent(
            enc_input_ids=enc_input_ids,
            enc_pad_mask=enc_input_ids == pad_token,
            dec_input_ids=dec_input_ids,
            dec_pad_mask=dec_input_ids == pad_token,
        )

    def decode(self, list_of_ints: list[list[int]]) -> list[str]:
        try:
            return [
                "".join(
                    ch
                    for ch in (self.idx_2_char[i] for i in ints)
                    if ch not in self._non_content_tokens
                )
                for ints in list_of_ints
            ]
        except KeyError as e:
            logger.error(f"Invalid index encountered during decoding: {e}")
            raise KeyError from e
