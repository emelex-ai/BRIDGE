"""VocabSpec: the vocabulary contract between Model and Tokenizer.

A `Model` needs to know its vocabulary sizes (to size `nn.Embedding` layers)
and a handful of special-token IDs (to drive autoregressive generation). This
information is intrinsic to whichever tokenizer produced the inputs but is
*not* the tokenizer itself. The model can be loaded, run, and generate from
without ever holding a tokenizer reference if it has these numbers on its
config.

This is the same separation used by HuggingFace `PretrainedConfig` (where
`vocab_size`, `pad_token_id`, `eos_token_id`, etc. live alongside architecture
hyperparameters). BRIDGE has two vocabularies (orthographic + phonological),
so the fields are prefixed accordingly.

The companion `from_tokenizer` classmethod is the one-line bridge for
downstream researchers wiring a `BridgeTokenizer` to a `Model`:

    >>> tok = BridgeTokenizer()
    >>> config.vocab = VocabSpec.from_tokenizer(tok)
    >>> model = Model(config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from bridge.domain.tokenizer.bridge_tokenizer import BridgeTokenizer


class VocabSpec(BaseModel):
    """The full vocab contract the `Model` needs at construction and generation.

    All fields are required. Use `VocabSpec.from_tokenizer(...)` to derive an
    instance from an existing `BridgeTokenizer`, or construct one manually
    with hardcoded numbers (e.g. for testing or when loading a checkpoint
    without instantiating the tokenizer).
    """

    orth_vocab_size: int = Field(description="Orthographic vocabulary size.")
    phon_vocab_size: int = Field(description="Phonological vocabulary size.")
    orth_pad_id: int = Field(description="Orthographic [PAD] token index.")
    orth_bos_id: int = Field(description="Orthographic [BOS] token index.")
    orth_eos_id: int = Field(description="Orthographic [EOS] token index.")
    phon_pad_id: int = Field(
        description=(
            "Phonological [PAD] id, in *feature* space: a column of the phoneme feature "
            "table, not one of its rows. Row ids are what encode() emits."
        )
    )
    phon_bos_id: int = Field(
        description=(
            "Phonological [BOS] id, in *feature* space: a column of the phoneme feature "
            "table, not one of its rows. Row ids are what encode() emits."
        )
    )
    phon_eos_id: int = Field(
        description=(
            "Phonological [EOS] id, in *feature* space: a column of the phoneme feature "
            "table, not one of its rows. Row ids are what encode() emits."
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
        """Build a `VocabSpec` from a constructed `BridgeTokenizer`.

        This is the canonical bridge from tokenizer to model config. Any
        downstream code constructing a Model from a BridgeTokenizer should
        flow through this classmethod.
        """
        sizes = tokenizer.get_vocabulary_sizes()
        return cls(
            orth_vocab_size=sizes["orthographic"],
            phon_vocab_size=sizes["phonological"],
            orth_pad_id=tokenizer.orth_pad_id,
            orth_bos_id=tokenizer.orth_bos_id,
            orth_eos_id=tokenizer.orth_eos_id,
            phon_pad_id=tokenizer.phon_pad_id,
            phon_bos_id=tokenizer.phon_bos_id,
            phon_eos_id=tokenizer.phon_eos_id,
            phon_table_fingerprint=tokenizer.phon_table_fingerprint,
        )
