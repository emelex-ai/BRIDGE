import logging
import string

import torch

from bridge.domain.datamodels.encodings import EncodingComponent
from bridge.domain.tokenizer.languages import (
    SUPPORTED_LANGUAGES,
    UNSPECIFIED,
    Languages,
    rows_and_languages,
    split_words,
)
from bridge.utils import device_manager

logger = logging.getLogger(__name__)


class CharacterTokenizer:
    def __init__(self):
        self.device = device_manager.device

        self.special_tokens = ["[BOS]", "[EOS]", "[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        # Language tokens disambiguate interlingual homographs (e.g. English "read"
        # vs Spanish "leer" can share spellings or share phonemes in mixed corpora).
        # ``"--"`` is the unspecified-language placeholder that preserves tensor shape.
        self.language_tokens = [UNSPECIFIED, *SUPPORTED_LANGUAGES]
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
        languages: Languages = None,
    ) -> EncodingComponent:
        """Encode strings to character ids, marking the language of every word.

        Each row is laid out as ``[LANG, BOS, ...chars, EOS, PAD, ...]`` for encoder input
        and ``[LANG, BOS, ...chars, PAD, ...]`` for decoder input, where ``LANG`` is the
        first word's language. Slicing past the BOS therefore requires ``[:, 2:]`` (skip
        both the language token and BOS).

        A language token appears again only where the language changes, directly before
        the first character of the word that changes it, so after the whitespace::

            the word is gato        EN EN EN ES
            [EN] [BOS] t h e _ w o r d _ i s _ [ES] g a t o [EOS]

        Characters are kept exactly as written, whitespace included, so a row in one
        language holds a single language token.

        Args:
            list_of_strings: A string or list of strings to encode.
            languages: The language of every word, in the forms
                :func:`~bridge.domain.tokenizer.languages.rows_and_languages` accepts.
                ``None`` leaves every row unspecified, ``"--"``.

        Returns:
            An orthographic :class:`EncodingComponent`.
        """
        rows, row_languages = rows_and_languages(list_of_strings, languages)
        bodies = [
            self._body(row, word_langs) for row, word_langs in zip(rows, row_languages, strict=True)
        ]
        max_length = max(len(body) for body in bodies)
        bos, eos, pad = (self.char_2_idx[t] for t in ("[BOS]", "[EOS]", "[PAD]"))

        # Rows are uniform length by construction: 3 + max_length for the encoder
        # ([LANG], [BOS], [EOS]) and 2 + max_length for the decoder ([LANG], [BOS]).
        enc_rows, dec_rows = [], []
        for body, word_langs in zip(bodies, row_languages, strict=True):
            prefix = [self.char_2_idx[word_langs[0]], bos]
            padding = [pad] * (max_length - len(body))
            enc_rows.append([*prefix, *body, eos, *padding])
            dec_rows.append([*prefix, *body, *padding])

        enc_input_ids = torch.tensor(enc_rows, dtype=torch.long, device=self.device)
        dec_input_ids = torch.tensor(dec_rows, dtype=torch.long, device=self.device)

        return EncodingComponent(
            enc_input_ids=enc_input_ids,
            enc_pad_mask=enc_input_ids == pad,
            dec_input_ids=dec_input_ids,
            dec_pad_mask=dec_input_ids == pad,
        )

    def _body(self, row: str, word_langs: tuple[str, ...]) -> list[int]:
        """The ids between ``[BOS]`` and ``[EOS]``: the characters, plus a language token
        before each word whose language differs from the previous word's."""
        unk = self.char_2_idx["[UNK]"]
        ids = [self.char_2_idx.get(ch, unk) for ch in row]
        words = split_words(row)
        # Right to left, so an insertion never moves an offset still to be used.
        for i in range(len(words) - 1, 0, -1):
            if word_langs[i] != word_langs[i - 1]:
                ids.insert(words[i].start(), self.char_2_idx[word_langs[i]])
        return ids

    def decode(self, list_of_ints: list[list[int]]) -> list[str]:
        """The text of each row: its characters, without special or language tokens."""
        return [text for text, _ in self.decode_with_languages(list_of_ints)]

    def decode_with_languages(
        self, list_of_ints: list[list[int]]
    ) -> list[tuple[str, list[str | None]]]:
        """Decode each row to its text and the language of each of its words.

        The inverse of the layout :meth:`encode` writes. A word's language is the one in
        force at its first character, which is the last language token before it, so a
        language token found inside a word takes effect at the next word. ``"--"`` comes
        back as ``None``, which :meth:`encode` reads as ``"--"`` again. A row with no words
        returns no language, so its opening token is not in the result.
        """
        decoded = []
        for ints in list_of_ints:
            chars: list[str] = []
            in_force: list[str | None] = []
            language: str | None = None
            for token in (self.idx_2_char[i] for i in ints):
                if token in self.language_tokens:
                    language = None if token == UNSPECIFIED else token
                elif token not in self.special_tokens:
                    chars.append(token)
                    in_force.append(language)
            text = "".join(chars)
            decoded.append((text, [in_force[word.start()] for word in split_words(text)]))
        return decoded
