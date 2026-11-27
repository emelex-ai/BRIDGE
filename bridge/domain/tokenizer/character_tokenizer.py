import logging
import string

import torch

from bridge.domain.datamodels.encodings import Target, Tokens, pad_rows
from bridge.domain.tokenizer.languages import (
    SUPPORTED_LANGUAGES,
    UNSPECIFIED,
    Languages,
    read_rows,
    split_words,
)
from bridge.utils import device_manager

logger = logging.getLogger(__name__)

# The characters of Spanish that ``string.printable`` lacks, each one code point. A
# character's place in the vocabulary is its row in a checkpoint's orthographic embedding
# and output head, so this is written out, not read from a lexicon, and it only grows at
# its end: a character added anywhere else moves every id after it.
SPANISH_CHARACTERS = "áéíóúüñÁÉÍÓÚÜÑ¿¡"


class CharacterTokenizer:
    def __init__(self):
        self.device = device_manager.device

        self.special_tokens = ["[BOS]", "[EOS]", "[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        # Language tokens disambiguate interlingual homographs (e.g. English "read"
        # vs Spanish "leer" can share spellings or share phonemes in mixed corpora).
        # ``"--"`` is the unspecified-language placeholder that preserves tensor shape.
        self.language_tokens = [UNSPECIFIED, *SUPPORTED_LANGUAGES]
        self.vocab = [
            *self.special_tokens,
            *self.language_tokens,
            *string.printable,
            *SPANISH_CHARACTERS,
        ]

        self.char_2_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_2_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocabulary_size = len(self.vocab)

        logger.info(f"CharacterTokenizer initialized with vocabulary size: {self.vocabulary_size}")

    def get_vocabulary_size(self) -> int:
        return self.vocabulary_size

    def encode_input(self, texts: list[str], languages: Languages = None) -> Tokens:
        """Encode rows as what the orthographic encoder reads.

        Each row is laid out as ``[LANG, BOS, ...chars, EOS, PAD, ...]``, where ``LANG`` is
        the first word's language. A language token appears again only where the language
        changes, directly before the first character of the word that changes it, so
        after the whitespace::

            the word is gato        EN EN EN ES
            [EN] [BOS] t h e _ w o r d _ i s _ [ES] g a t o [EOS]

        Each row is first composed to Unicode NFC, so ``e`` followed by a combining acute
        accent is the one character ``é``. Past that, characters are kept as written, case
        and whitespace included, so a row in one language holds a single language token.
        A character outside the vocabulary encodes as ``[UNK]``.

        Args:
            texts: The rows to encode, a list of strings.
            languages: The language of every word, one entry per row, in the forms
                :func:`~bridge.domain.tokenizer.languages.per_row_languages` accepts.
                ``None`` leaves every row unspecified, ``"--"``.
        """
        eos = self.char_2_idx["[EOS]"]
        return Tokens(*self._padded([[*row, eos] for row in self._rows(texts, languages)]))

    def encode_target(self, texts: list[str], languages: Languages = None) -> Target:
        """Encode rows as what the orthographic decoder is fed and scored against.

        The ids are ``[LANG, BOS, ...chars, PAD, ...]`` and the labels are the same row one
        position on, ``[BOS, ...chars, EOS, PAD, ...]``. Decoder position ``i`` is scored
        against the token at ``i + 1``, so ``[BOS]`` predicts the first character and the
        last character predicts ``[EOS]``. Generation seeds the decoder with ``[LANG,
        BOS]``, and the token it samples next must be the first character, which is what
        that pairing trains. A mid-row language token sits in the ids and in the labels,
        so the decoder learns to emit the switch. See ``docs/decisions/0004``.

        ``texts`` and ``languages`` are as for :meth:`encode_input`.
        """
        rows = self._rows(texts, languages)
        ids, pad_mask = self._padded(rows)
        labels, _ = self._padded([[*row[1:], self.char_2_idx["[EOS]"]] for row in rows])
        return Target(ids, pad_mask, labels)

    def _rows(self, texts: list[str], languages: Languages) -> list[list[int]]:
        """Each row as ``[LANG, BOS, ...body]``, unpadded."""
        bos = self.char_2_idx["[BOS]"]
        return [
            [self.char_2_idx[word_langs[0]], bos, *self._body(text, word_langs)]
            for text, word_langs in read_rows(texts, languages)
        ]

    def _padded(self, rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        return pad_rows(rows, self.char_2_idx["[PAD]"], self.device)

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
        """The text of each row: its characters, without special or language tokens.

        ``[UNK]`` is a special token, so a character the vocabulary lacks is absent from
        the text and not marked in it.
        """
        return [text for text, _ in self.decode_with_languages(list_of_ints)]

    def decode_with_languages(
        self, list_of_ints: list[list[int]]
    ) -> list[tuple[str, list[str | None]]]:
        """Decode each row to its text and the language of each of its words.

        The inverse of the layout :meth:`encode_input` writes. A word's language is the one
        in force at its first character, which is the last language token before it, so a
        language token found inside a word takes effect at the next word. ``"--"`` comes
        back as ``None``, which :meth:`encode_input` reads as ``"--"`` again. A row with no
        words returns no language, so its opening token is not in the result.
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
