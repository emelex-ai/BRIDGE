"""VocabSpec: what a ``Model`` needs to know of its two vocabularies.

The sizes that shape its embedding layers and the special-token ids that drive
generation. A tokenizer defines them, but the model holds these numbers and not the
tokenizer, so a model loads, runs and generates with no tokenizer in the process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from bridge.domain.tokenizer.bridge_tokenizer import BridgeTokenizer


class VocabSpec(BaseModel):
    """The sizes, the pad, bos and eos ids and the language-token ids are required.

    Build one with :meth:`from_tokenizer`. A spec written by hand is checked against the
    phoneme table when a ``Model`` is built from it.
    """

    # An unknown key is a typo in an experiment config. Ignored, the run does something
    # other than what the file says.
    model_config = ConfigDict(extra="forbid")

    orth_vocab_size: int = Field(description="Orthographic vocabulary size.")
    phon_vocab_size: int = Field(description="Phonological vocabulary size.")
    orth_pad_id: int = Field(description="Orthographic [PAD] token index.")
    orth_bos_id: int = Field(description="Orthographic [BOS] token index.")
    orth_eos_id: int = Field(description="Orthographic [EOS] token index.")
    orth_language_ids: tuple[int, ...] = Field(
        description=(
            "Orthographic language-token indices. A label holding one of these is a "
            "change of language, not a letter, which is what the letters-only accuracy "
            "figures leave out. Empty for a vocabulary with no language tokens."
        ),
    )
    phon_pad_id: int = Field(
        description=(
            "Phonological [PAD] id, in *feature* space: a column of the phoneme feature "
            "table, not one of its rows. Row ids are what the tokenizer emits."
        )
    )
    phon_bos_id: int = Field(
        description=(
            "Phonological [BOS] id, in *feature* space: a column of the phoneme feature "
            "table, not one of its rows. Row ids are what the tokenizer emits."
        )
    )
    phon_eos_id: int = Field(
        description=(
            "Phonological [EOS] id, in *feature* space: a column of the phoneme feature "
            "table, not one of its rows. Row ids are what the tokenizer emits."
        )
    )
    phon_table_fingerprint: str | None = Field(
        default=None,
        description=(
            "Digest of the phoneme feature table these ids were derived from. Lets a "
            "checkpoint notice that phonreps.csv changed underneath it. Optional and "
            "None by default, so specs written before the field still deserialize."
        ),
    )

    @classmethod
    def from_tokenizer(cls, tokenizer: BridgeTokenizer) -> VocabSpec:
        sizes = tokenizer.get_vocabulary_sizes()
        return cls(
            orth_vocab_size=sizes["orthographic"],
            phon_vocab_size=sizes["phonological"],
            orth_pad_id=tokenizer.orth_pad_id,
            orth_bos_id=tokenizer.orth_bos_id,
            orth_eos_id=tokenizer.orth_eos_id,
            orth_language_ids=tokenizer.orth_language_ids,
            phon_pad_id=tokenizer.phon_pad_id,
            phon_bos_id=tokenizer.phon_bos_id,
            phon_eos_id=tokenizer.phon_eos_id,
            phon_table_fingerprint=tokenizer.phon_table_fingerprint,
        )
