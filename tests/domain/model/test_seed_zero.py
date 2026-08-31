"""Zero is a seed like any other.

``Model.__init__`` decides whether to seed by testing the truth of
``model_config.seed``. Zero is a perfectly ordinary seed and a perfectly falsy int, so a
model asked for ``seed=0`` is built from whatever state the global RNG happened to be in.
Nothing warns. The run looks seeded, the config records a seed, and the weights are
different every time, for that one seed value only.

The oracle is the behaviour of ``set_seed`` (``bridge/utils/helper_functions.py``): it
calls ``torch.manual_seed``, and every parameter here is drawn from that one global
generator. So initialization is an exact function of the seed. Two models built from the
same seed must agree bit for bit, and ``torch.equal`` rather than a tolerance is the right
comparator, since a tolerance would pass a model that was never seeded at all as long as
the two draws happened to land close.

``seed=1`` is the control. It exercises the same code along the same path and must pass
both before and after the fix; if it ever fails, the defect is in this file rather than in
``Model``. The complementary control is that different seeds must produce *different*
weights, which is what rules out the trivial pass where every model comes out the same.

``seed=None`` is the third case and the reason the fix is ``is not None`` rather than a
truthiness test that also happens to accept zero: absent means unseeded, and unseeded
models must still differ from one another.
"""

import pytest
import torch

from bridge.domain.datamodels import ModelConfig
from bridge.domain.model import Model
from tests.vocab import TEST_VOCAB

D_MODEL = 16


def build(seed: int | None) -> Model:
    """A small model whose only varying input is the seed."""
    return Model(ModelConfig(vocab=TEST_VOCAB, d_model=D_MODEL, nhead=2, seed=seed))


def first_difference(left: Model, right: Model) -> tuple[str, float] | None:
    """Name of the first parameter that differs, with its max absolute deviation.

    ``None`` when every parameter is bitwise identical. Two models built from the same
    config always have the same parameter names in the same order, so pairing them
    positionally is safe.
    """
    for (name, left_param), (_, right_param) in zip(
        left.named_parameters(), right.named_parameters(), strict=True
    ):
        if not torch.equal(left_param, right_param):
            deviation = (left_param - right_param).abs().max().item()
            return name, deviation
    return None


@pytest.mark.parametrize("seed", [0, 1])
def test_same_seed_gives_bitwise_identical_parameters(seed):
    """The defect at seed=0, and its control at seed=1.

    Initialization draws from the generator ``set_seed`` seeds, so the same seed must
    reproduce the same weights exactly. seed=1 takes the identical path and passes today;
    seed=0 is falsy and skips the seeding call, so its two models are built from an
    uncontrolled generator.
    """
    difference = first_difference(build(seed), build(seed))
    assert difference is None, (
        f"two models built with seed={seed} are not identical: parameter "
        f"{difference[0]} differs by {difference[1]:.6e}. A seeded model must be "
        f"reproducible; zero is a seed, not an absence of one."
    )


def test_different_seeds_give_different_parameters():
    """Rules out the trivial pass where the seed is ignored and every model is the same.

    If construction were deterministic regardless of the seed, the equality test above
    would pass for the wrong reason. Two different seeds must land on different weights.
    """
    assert first_difference(build(0), build(1)) is not None, (
        "seed=0 and seed=1 produced identical parameters, so the seed is not reaching "
        "initialization and the equality tests above prove nothing."
    )


def test_absent_seed_leaves_the_model_unseeded():
    """seed=None must stay unseeded, which is what makes zero a real distinction.

    The fix is ``is not None``, not ``>= 0`` or a default. Constructing a model consumes
    randomness from the global generator, so two unseeded models drawn back to back are
    built from different states and must differ.
    """
    assert first_difference(build(None), build(None)) is not None, (
        "two models built with seed=None are identical, so something is seeding the "
        "unseeded case and 'absent' can no longer be told apart from 'zero'."
    )
