import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bridge.domain.datamodels.vocab_spec import VocabSpec


def _divisors(n: int) -> list[int]:
    """Every positive divisor of ``n``, ascending, found in ``sqrt(n)`` steps."""
    small = [i for i in range(1, math.isqrt(n) + 1) if n % i == 0]
    return sorted({*small, *(n // i for i in small)})


class ModelConfig(BaseModel):
    # An unknown key is a typo in an experiment config. Ignored, the run does something
    # other than what the file says.
    model_config = ConfigDict(extra="forbid")

    # Every size has a floor of one, so that a config which validates builds a model on
    # which all five pathways train and generate. Left to torch, none of these fail where
    # the researcher can read them: a stack with no layers raises a bare `IndexError` on
    # the first forward of each pathway that uses it, a negative count builds that same
    # empty stack, and `d_embedding=0` trains to a finite loss on a decoder that cannot
    # see what the encoder read. See docs/decisions/0014. `d_model` and `nhead` answer to
    # one more rule, which relates the two and is the validator below.
    num_phon_enc_layers: int = Field(default=2, ge=1)
    num_orth_enc_layers: int = Field(default=2, ge=1)
    num_mixing_enc_layers: int = Field(default=2, ge=1)
    num_phon_dec_layers: int = Field(default=2, ge=1)
    num_orth_dec_layers: int = Field(default=2, ge=1)
    d_model: int = Field(default=64, ge=1)
    nhead: int = Field(default=2, ge=1)
    # The number of rows in the global memory, which is all a decoder reads of the input.
    d_embedding: int = Field(default=2, ge=1)
    seed: int | None = Field(default=None)
    # Build via `VocabSpec.from_tokenizer(bridge_tokenizer)`.
    vocab: VocabSpec

    @model_validator(mode="after")
    def heads_fit_the_width(self) -> Self:
        """Reject a ``d_model`` that ``nhead`` does not divide into heads of at least two
        dimensions, and say what would work.

        Two rules, and they relate the same two fields. ``nhead`` must divide ``d_model``,
        which is torch's rule. And each head needs two dimensions, because position is a
        rotation of pairs of a head's dimensions (``layers.rotate``): a head of one has no
        pair, so nothing turns and the encoders read "cat" and "act" as the same word,
        with a finite loss and no error.

        A model validator, not a field validator on ``d_model``, because the rules relate
        two fields. Pydantic validates fields in declaration order and skips defaults, so
        a field validator on ``d_model`` finds ``nhead`` missing and never runs at all
        when ``d_model`` is left at its default. This runs once every field is set, and
        only if each passed its own floor, so ``nhead`` is never zero here.

        The message is for whoever typed the config. Torch reports a width the heads do
        not divide as ``AssertionError: embed_dim must be divisible by num_heads``, raised
        while the model is being built and naming neither field.
        """
        head_dim, left_over = divmod(self.d_model, self.nhead)
        if left_over == 0 and head_dim >= 2:
            return self

        if left_over:
            problem = (
                f"d_model={self.d_model} is not divisible by nhead={self.nhead}. Each "
                f"attention head gets d_model / nhead dimensions, so that must be a whole "
                f"number."
            )
        else:
            problem = (
                f"d_model={self.d_model} gives each of nhead={self.nhead} heads one "
                f"dimension. Position is a rotation of pairs of a head's dimensions, so a "
                f"head needs at least two, or the model cannot read the order of its input."
            )
        # The nearest sizes either side that pass both rules. A width of one has no head
        # count that works, so that half of the advice can be absent.
        legal = [n for n in _divisors(self.d_model) if 2 * n <= self.d_model]
        heads = [n for n in legal if n < self.nhead][-1:] + [n for n in legal if n > self.nhead][:1]
        below = self.d_model - left_over
        widths = sorted({max(width, 2 * self.nhead) for width in (below, below + self.nhead)})
        fixes = [f"keep nhead={self.nhead} and set d_model to {' or '.join(map(str, widths))}"]
        if heads:
            fixes.insert(
                0, f"keep d_model={self.d_model} and set nhead to {' or '.join(map(str, heads))}"
            )
        raise ValueError(f"{problem} Nearest fixes: {', or '.join(fixes)}.")
