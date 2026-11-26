"""BridgeTokenizer: the character tokenizer and the phoneme tokenizer behind one call.

Both tokenizers emit the same shape: a ``(batch, sequence)`` integer tensor of ids plus a
matching boolean pad mask. Only the id space differs. Character ids index the character
vocabulary; phoneme ids index rows of the feature table in ``phonreps.csv`` (*which
phoneme*, not which phonetic feature).

The PhonemeTokenizer reads the per-language pronunciation lexicons under
``bridge/core/pronunciation_lexicons/`` together with ``phonreps.csv``. A word has
phonology only if a lexicon holds it.

The BridgeTokenizer encodes the two sides of a training example separately:
``encode_input`` gives what the encoders read and ``encode_target`` gives what the decoders
are fed and scored against. A nonword, which no lexicon holds, encodes as an input with
orthography alone.
"""

import logging
from collections.abc import Sequence

import torch

from bridge.domain.datamodels.encodings import BridgeInput, BridgeTarget
from bridge.domain.tokenizer.character_tokenizer import CharacterTokenizer
from bridge.domain.tokenizer.languages import Languages
from bridge.domain.tokenizer.phoneme_tokenizer import PhonemeTokenizer
from bridge.utils import device_manager

logger = logging.getLogger(__name__)


class BridgeTokenizer:
    """Encodes each side of a training example in both modalities: ``encode_input`` for
    what the encoders read, ``encode_target`` for what the decoders are fed and scored
    against."""

    def __init__(
        self,
        custom_cmudict_path: str | None = None,
    ):
        self.device = device_manager.device
        self.custom_cmudict_path = custom_cmudict_path
        self.char_tokenizer = CharacterTokenizer()
        self.phoneme_tokenizer = PhonemeTokenizer(custom_cmudict_path=custom_cmudict_path)
        self.phon_table_fingerprint = self.phoneme_tokenizer.phoneme_table.fingerprint
        # The four phon ids are in feature space, unlike the row ids the encoders emit.
        # See VocabSpec.phon_pad_id.
        self.phon_bos_id = self.phoneme_tokenizer.special_token_dims["[BOS]"]
        self.phon_pad_id = self.phoneme_tokenizer.special_token_dims["[PAD]"]
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

    def encode_input(self, texts: list[str], languages: Languages = None) -> BridgeInput:
        """Encode rows as what the model reads.

        Args:
            texts: The rows to encode, a list of strings.
            languages: The language of every word: a list or tuple with one entry per row.
                An entry is one code for the whole row (``"ES"``), one code per word
                (``"EN EN ES"``, or a list or tuple of codes), or ``None``, which leaves
                the row unspecified: the ``"--"`` token orthographically and the English
                lexicon phonologically. Omitting ``languages`` leaves every row
                unspecified. Orthography marks each change of language inside a row;
                phonology looks each word up in its own language.

        Returns:
            A :class:`BridgeInput` carrying orthography, and phonology when every word has
            a pronunciation in its language's lexicon. When one does not, ``phon`` is
            ``None``: a nonword is read through its spelling alone, and a pathway that
            reads phonology refuses the input by name.

        Raises:
            TypeError, ValueError: for rows or languages in the wrong form. An argument
                error raises and a missing pronunciation does not: "this word is not in
                the lexicon" is an ordinary data condition a caller must be able to
                handle, and "this language code does not exist" is a mistake to fix.
                Reporting both the same way leaves a caller unable to tell them apart.
        """
        phon = self.phoneme_tokenizer.encode_input(texts, languages)
        if phon is None:
            logger.info("No pronunciation for %r in the lexicon; the input has no phon", texts)
        return BridgeInput(orth=self.char_tokenizer.encode_input(texts, languages), phon=phon)

    def encode_target(self, texts: list[str], languages: Languages = None) -> BridgeTarget | None:
        """Encode rows as what the model is taught to write.

        ``texts`` and ``languages`` are as for :meth:`encode_input`, and so are the
        argument errors. A target is scored in both modalities, so one without a
        pronunciation is not a target: the return is ``None`` if a word is missing from
        its language's lexicon.
        """
        phon = self.phoneme_tokenizer.encode_target(texts, languages)
        if phon is None:
            logger.info("No pronunciation for %r in the lexicon; there is no target", texts)
            return None
        return BridgeTarget(orth=self.char_tokenizer.encode_target(texts, languages), phon=phon)

    def orth_start(self, languages: Sequence[str | None]) -> torch.Tensor:
        """The ``(batch, 2)`` ``[LANG, BOS]`` rows that start orthographic generation.

        ``languages`` holds one code per row, the language of the first word the decoder
        is to write, or ``None`` for ``"--"``. Any later change of language is the
        decoder's to emit. This is what training feeds the orthographic decoder at
        positions 0 and 1: the ids of an empty target in that language.
        """
        return self.char_tokenizer.encode_target([""] * len(languages), languages).ids

    def decode(
        self,
        ortho_indices: list[list[int]] | None = None,
        phono_indices: list[list[int]] | None = None,
    ) -> dict | None:
        """Decode either modality, or both.

        Args:
            ortho_indices: Rows of character ids.
            phono_indices: Rows of phoneme *feature* indices, as held by
                ``GenerationOutput.phon_tokens``. Not the row ids ``encode_input`` returns.

        Returns:
            A dict holding ``"orthographic"``, a list of strings, and ``"phonological"``,
            a dense multi-hot tensor, each only if its argument was given. ``None`` when
            neither was.
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
        return {
            "orthographic": self.char_tokenizer.get_vocabulary_size(),
            "phonological": self.phoneme_tokenizer.get_vocabulary_size(),
        }

    def phoneme_vectors_to_word(self, phoneme_vectors, distance_fn=None):
        """The phoneme name of each vector, not a spelled word. See
        :meth:`PhonemeTokenizer.phoneme_vectors_to_word`."""
        return self.phoneme_tokenizer.phoneme_vectors_to_word(phoneme_vectors, distance_fn)

    def phoneme_vector_to_phoneme(self, phoneme_vector, distance_fn=None, top_k=1):
        """See :meth:`PhonemeTokenizer.phoneme_vector_to_phoneme`."""
        return self.phoneme_tokenizer.phoneme_vector_to_phoneme(phoneme_vector, distance_fn, top_k)
