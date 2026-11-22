import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bridge.domain.datamodels.vocab_spec import VocabSpec


def _divisors(n: int) -> list[int]:
    """Every positive divisor of ``n``, ascending, found in ``sqrt(n)`` steps."""
    small = [i for i in range(1, math.isqrt(n) + 1) if n % i == 0]
    return sorted({*small, *(n // i for i in small)})


class ModelConfig(BaseModel):
    # An unknown key is a typo in someone's experiment config, and silently
    # ignoring it means the run does something other than what the file says.
    # Removing a field (gcs_path, max_nb_steps) made that reachable.
    model_config = ConfigDict(extra="forbid")

    # Every size has a floor of one, so that a config which validates builds a model on
    # which all five pathways train and generate. Left to torch, none of these fail where
    # the researcher can read them: a stack with no layers raises a bare `IndexError` on
    # the first forward of each pathway that uses it, a negative count builds that same
    # empty stack, and `d_embedding=0` trains to a finite loss on a decoder that cannot
    # see what the encoder read. See docs/decisions/0014.
    num_phon_enc_layers: int = Field(default=2, ge=1)
    num_orth_enc_layers: int = Field(default=2, ge=1)
    num_mixing_enc_layers: int = Field(default=2, ge=1)
    num_phon_dec_layers: int = Field(default=2, ge=1)
    num_orth_dec_layers: int = Field(default=2, ge=1)
    d_model: int = Field(default=64, ge=1)
    nhead: int = Field(default=2, ge=1)
    d_embedding: int = Field(default=1, ge=1)
    seed: int | None = Field(default=None)
    # Build via `VocabSpec.from_tokenizer(bridge_tokenizer)`.
    vocab: VocabSpec

    @model_validator(mode="after")
    def heads_divide_width(self) -> Self:
        """Reject a ``d_model`` that ``nhead`` does not divide, and say what would work.

        A model validator, not a field validator on ``d_model``, because the rule relates
        two fields. Pydantic validates fields in declaration order and skips defaults, so
        a field validator on ``d_model`` finds ``nhead`` missing and never runs at all
        when ``d_model`` is left at its default. This runs once every field is set, and
        only if each passed its own floor, so ``nhead`` is never zero here.

        The message is for whoever typed the config. Torch reports the same mistake as
        ``AssertionError: embed_dim must be divisible by num_heads``, raised while the
        model is being built and naming neither field.
        """
        if self.d_model % self.nhead == 0:
            return self

        divisors = _divisors(self.d_model)
        # One always divides and `nhead` is at least two here, so a smaller head count
        # always exists. A larger one does not when `nhead` exceeds `d_model`.
        heads = [max(n for n in divisors if n < self.nhead)]
        heads += [n for n in divisors if n > self.nhead][:1]
        below = self.d_model - self.d_model % self.nhead
        widths = [width for width in (below, below + self.nhead) if width > 0]
        raise ValueError(
            f"d_model={self.d_model} is not divisible by nhead={self.nhead}. Each attention "
            f"head gets d_model / nhead dimensions, so that must be a whole number. Nearest "
            f"fixes: keep d_model={self.d_model} and set nhead to "
            f"{' or '.join(map(str, heads))}, or keep nhead={self.nhead} and set d_model to "
            f"{' or '.join(map(str, widths))}."
        )
