"""Pins what ``ModelConfig`` accepts: the architectures a ``Model`` can run, and no others.

The contract is that a config which validates builds a model on which all five pathways
train and generate. A size that cannot do that is rejected when the config is built, with
an error naming the field the researcher typed, rather than later and from inside torch.
See ``docs/decisions/0014``.

Three oracles, named per test:

- **reference implementation**: ``torch.nn.MultiheadAttention`` decides which
  ``(d_model, nhead)`` pairs build, and ``layers.rotate`` decides which of those give a
  head anything to turn, so the config is compared against the two over a grid rather
  than against a rule restated here. That catches a wrong rejection as well as a wrong
  acceptance.
- **invariant**: every config that validates runs every pathway, and every fix an error
  message offers is itself a config that validates.
- **analytic**: one divisibility message checked against values worked out by hand.
"""

import math
import re

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from bridge.application.training import TrainingPipeline
from bridge.domain.datamodels import ModelConfig, TrainingConfig, VocabSpec
from bridge.domain.datamodels.pathways import PATHWAYS
from bridge.domain.model import Model
from bridge.domain.model.layers import rotate
from tests.vocab import TEST_VOCAB

LAYER_COUNTS = (
    "num_phon_enc_layers",
    "num_orth_enc_layers",
    "num_mixing_enc_layers",
    "num_phon_dec_layers",
    "num_orth_dec_layers",
)
SIZES = (*LAYER_COUNTS, "d_model", "nhead", "d_embedding")


def rejection(**sizes) -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        ModelConfig(vocab=TEST_VOCAB, **sizes)
    return caught.value


def validates(**sizes) -> bool:
    try:
        ModelConfig(vocab=TEST_VOCAB, **sizes)
    except ValidationError:
        return False
    return True


def test_head_counts_are_accepted_exactly_when_torch_builds_them_and_position_turns_them():
    """Oracle: reference implementation, in two parts. Torch's attention says which pairs
    build at all. ``rotate``, the one place position enters, says which of those have a
    pair of dimensions in each head for it to turn: on a head of one dimension it returns
    its input, and a model built so reads "cat" and "act" alike.

    The grid runs through zero and into the negatives, so it covers both of torch's
    refusals: the range check and the divisibility check. It holds odd head dimensions,
    which turn all but their last coordinate and are accepted.
    """

    def torch_accepts(d_model: int, nhead: int) -> bool:
        try:
            nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        except (ValueError, AssertionError):
            return False
        return True

    def position_turns_a_head(d_model: int, nhead: int) -> bool:
        head = torch.ones(1, d_model // nhead)
        return not torch.equal(rotate(head, torch.tensor([1])), head)

    grid = [(d_model, nhead) for d_model in range(-2, 25) for nhead in range(-2, 13)]
    disagreements = [
        (d_model, nhead)
        for d_model, nhead in grid
        if validates(d_model=d_model, nhead=nhead)
        != (torch_accepts(d_model, nhead) and position_turns_a_head(d_model, nhead))
    ]
    assert disagreements == [], (
        f"(d_model, nhead) pairs the config and its two references disagree on: {disagreements}"
    )
    assert validates(d_model=3, nhead=1), "control: an odd head dimension is accepted"
    assert validates(d_model=4, nhead=2), "control: a head dimension of two is accepted"
    assert not validates(d_model=4, nhead=4), "control: a head dimension of one is not"


def test_a_head_of_one_dimension_is_rejected_by_name_with_its_reason():
    """Oracle: analytic. Four heads on a width of four get one dimension each. Two heads
    is the nearest count that leaves a pair, and eight is the nearest width that does."""
    message = str(rejection(d_model=4, nhead=4))

    assert "d_model=4 gives each of nhead=4 heads one dimension" in message
    assert "cannot read the order of its input" in message
    assert "keep d_model=4 and set nhead to 2, or keep nhead=4 and set d_model to 8." in message


def test_divisibility_is_checked_when_d_model_is_left_at_its_default():
    """Pydantic skips field validators for defaults, so a rule hung on ``d_model`` alone
    never sees the commonest config there is: one that only sets ``nhead``."""
    default = ModelConfig.model_fields["d_model"].default
    assert default % 3 != 0, "pick a head count the default width is not divisible by"

    assert f"d_model={default} is not divisible by nhead=3" in str(rejection(nhead=3))


def test_the_divisibility_error_offers_the_nearest_fixes():
    """Oracle: analytic. 10 has divisors 1, 2, 5, 10, so 2 and 5 flank 4; 8 and 12 are the
    multiples of 4 either side of 10."""
    message = str(rejection(d_model=10, nhead=4))

    assert "d_model=10 is not divisible by nhead=4" in message
    assert "keep d_model=10 and set nhead to 2 or 5" in message
    assert "keep nhead=4 and set d_model to 8 or 12" in message


@pytest.mark.parametrize(
    ("d_model", "nhead"),
    [(10, 4), (64, 3), (7, 2), (3, 4), (1, 2), (30, 7), (97, 96), (768, 5), (4, 4), (8, 8), (1, 1)],
)
def test_every_fix_the_error_offers_validates_and_none_nearer_exists(d_model, nhead):
    """Oracle: invariant, with a brute-force search as the reference for "nearest".

    The fixes offered are the nearest sizes that validate, on each side where one exists.
    The cases include a prime width, a width smaller than the head count, a head of one
    dimension, and a width of one, for which no head count validates at all.
    """
    message = str(rejection(d_model=d_model, nhead=nhead))

    def offered(field: str) -> list[int]:
        found = re.search(rf"set {field} to (\d+(?: or \d+)*)", message)
        return [int(n) for n in found.group(1).split(" or ")] if found else []

    def nearest(legal: list[int], rejected: int) -> list[int]:
        below = [n for n in legal if n < rejected][-1:]
        return below + [n for n in legal if n > rejected][:1]

    legal_heads = [n for n in range(1, d_model + 1) if validates(d_model=d_model, nhead=n)]
    legal_widths = [
        w for w in range(1, d_model + 2 * nhead + 1) if validates(d_model=w, nhead=nhead)
    ]

    assert offered("nhead") == nearest(legal_heads, nhead)
    assert offered("d_model") == nearest(legal_widths, d_model)
    assert offered("d_model"), "a wider model always validates, so a width is always offered"


@pytest.mark.parametrize("field", SIZES)
@pytest.mark.parametrize("value", [0, -1])
def test_a_size_below_one_is_rejected_at_its_own_field(field, value):
    """The error is located at the field that is wrong, and it is the only error.

    Checking the location rather than only that something raised is what separates "this
    bound exists" from "some other check happened to fire first".
    """
    errors = rejection(**{field: value}).errors()

    assert [error["loc"] for error in errors] == [(field,)]
    assert errors[0]["type"] == "greater_than_equal"


def test_a_zero_head_count_is_reported_as_out_of_range_not_as_a_division():
    """Wrong in two ways at once: ``nhead=0`` is out of range and cannot divide anything.

    The range check owns this one. The divisibility rule takes a modulus, so reaching it
    with a zero would surface as ``ZeroDivisionError`` rather than as a validation error.
    """
    errors = rejection(d_model=10, nhead=0).errors()

    assert [error["loc"] for error in errors] == [("nhead",)]


def test_the_global_memory_defaults_to_two_rows(shared_tokenizer):
    """Oracle: the decision. ``d_embedding`` is the number of rows in the global memory,
    which is all a decoder reads of the input, and it defaults to two. The model is built
    as well as the config, so the default is seen where it matters: in the shape of what
    the decoders are handed."""
    config = ModelConfig(vocab=VocabSpec.from_tokenizer(shared_tokenizer))
    assert config.d_embedding == 2

    model = Model(config).eval()
    inputs = shared_tokenizer.encode_input(["cat", "dog"])
    assert model.global_embedding.shape == (1, 2, config.d_model)
    assert model.generate(inputs, "o2p", max_new_tokens=1).global_encoding.shape == (
        2,
        2,
        config.d_model,
    )


def test_exactly_the_sizes_at_or_above_their_floor_validate():
    """The floor is one for every size but the width, whose floor is two, and a floor is
    the only thing rejected below three.

    Each size is swept through -1, 0, 1, 2, 3 with the others held at one, except a width
    of six. One head divides every width, and six gives every head count swept a head of
    two dimensions or more. That leaves the floors as the only rules in play. The width's
    floor is two because one head on a width of one is a head of one dimension. Asserting
    the accepted set, rather than only that the bad values raise, is what catches a floor
    set too high as well as one set too low.
    """
    base = dict.fromkeys(SIZES, 1) | {"d_model": 6}
    values = (-1, 0, 1, 2, 3)
    floor = dict.fromkeys(SIZES, 1) | {"d_model": 2}

    accepted = {
        (field, value) for field in SIZES for value in values if validates(**base | {field: value})
    }

    assert accepted == {
        (field, value) for field in SIZES for value in values if value >= floor[field]
    }


# An odd head count makes torch warn that it is declining an inference fast path. One is
# odd, and one is the floor under test.
@pytest.mark.filterwarnings("ignore:enable_nested_tensor is True")
@pytest.mark.parametrize("d_model", [2, 12])
def test_a_model_with_every_size_at_its_floor_runs_every_pathway(
    words_dataset, shared_tokenizer, d_model
):
    """Oracle: invariant. What the config lets through, the model can run.

    The floor is where a bound that is one too low would show, so it is the case built
    here: one layer in each stack, one head, one global row, and a width of two. The width
    of twelve repeats it with only ``d_model`` lifted, so a failure says which of the two
    it belongs to. Larger sizes are what the rest of the suite runs on.

    "Runs" is an optimizer step with a finite loss and a generation, on all five
    pathways. Generation is in because ``d_embedding=0`` survives the step: measured, it
    reaches a finite loss with logits bitwise identical for different input words.
    """
    config = ModelConfig(
        vocab=VocabSpec.from_tokenizer(shared_tokenizer),
        seed=5,
        **dict.fromkeys(SIZES, 1) | {"d_model": d_model},
    )
    model = Model(config)

    for pathway in PATHWAYS:
        pipeline = TrainingPipeline(
            model=model, training_config=TrainingConfig(training_pathway=pathway)
        )
        loss = pipeline.single_step(words_dataset, slice(0, 4), calculate_metrics=False)["loss"]
        assert math.isfinite(float(loss)), f"{pathway}: loss is not finite"

    model.eval()
    inputs = shared_tokenizer.encode_input(["cat", "dog"])
    start = shared_tokenizer.orth_start([None, None])
    with torch.no_grad():
        for pathway in PATHWAYS:
            model.generate(inputs, pathway, max_new_tokens=8, orth_start=start, deterministic=True)
