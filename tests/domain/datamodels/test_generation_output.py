"""Pins the per-field validation rules of ``GenerationOutput``.

Two near-identical validators (``validate_probability_list`` /
``validate_phonological_vectors``) were merged into ``validate_nested_tensors`` plus a
pluggable per-element check. The merge is only correct if each field keeps exactly the
rules it had. The old code selected them with ``if name == "orth_probs"`` /
``if name == "phon_vecs"`` string comparisons, which is easy to get wrong when
refactoring. The asymmetries below are the whole point of this file:

* ``orth_probs`` is a softmax over the orthographic vocabulary -> must sum to 1
* ``phon_probs`` is an independent per-feature probability -> must NOT be required to
  sum to 1, but must stay within [0, 1]
* ``phon_vecs`` is a sampled binary feature vector -> must be exactly 0 or 1
* ``phon_tokens`` holds active feature indices -> unconstrained beyond being 1-D
"""

import pytest
import torch

from bridge.domain.datamodels import GenerationOutput
from bridge.domain.datamodels.generate_models import (
    check_is_distribution,
    validate_nested_tensors,
)

BATCH = 2
GLOBAL = torch.randn(BATCH, 1, 8)


def nested(*values):
    """One step per batch item, each a 1-D tensor."""
    return [[torch.tensor(values, dtype=torch.float)] for _ in range(BATCH)]


def simplex():
    return [[torch.softmax(torch.randn(5), dim=0)] for _ in range(BATCH)]


def orth_tokens():
    return torch.zeros((BATCH, 3), dtype=torch.long)


def build(**kwargs):
    return GenerationOutput(global_encoding=GLOBAL, **kwargs)


def phon_kwargs(**over):
    kwargs = {
        "phon_probs": nested(0.5, 0.5, 0.5),
        "phon_vecs": nested(0.0, 1.0, 0.0),
        "phon_tokens": nested(3.0, 7.0),
    }
    kwargs.update(over)
    return kwargs


# --- happy paths -----------------------------------------------------------


def test_orthographic_only_output_is_valid():
    build(orth_probs=simplex(), orth_tokens=orth_tokens())


def test_phonological_only_output_is_valid():
    build(**phon_kwargs())


def test_both_modalities_is_valid():
    build(orth_probs=simplex(), orth_tokens=orth_tokens(), **phon_kwargs())


# --- the asymmetries the validator merge had to preserve -------------------


def test_orth_probs_must_sum_to_one():
    with pytest.raises(ValueError, match="probabilities must sum to 1"):
        build(orth_probs=nested(0.5, 0.5, 0.5), orth_tokens=orth_tokens())


def test_phon_probs_need_NOT_sum_to_one():
    """Per-feature independent probabilities. Requiring a simplex here would reject
    every real phonological generation."""
    build(**phon_kwargs(phon_probs=nested(0.9, 0.9, 0.9)))


def test_phon_probs_must_stay_within_unit_range():
    with pytest.raises(ValueError, match="probabilities must be between 0 and 1"):
        build(**phon_kwargs(phon_probs=nested(1.5, 0.2, 0.1)))


def test_phon_probs_reject_negative_values():
    with pytest.raises(ValueError, match="probabilities must be between 0 and 1"):
        build(**phon_kwargs(phon_probs=nested(-0.1, 0.2, 0.1)))


def test_phon_vecs_must_be_binary():
    with pytest.raises(ValueError, match="must contain only binary values"):
        build(**phon_kwargs(phon_vecs=nested(0.0, 0.5, 1.0)))


def test_phon_tokens_are_unconstrained_indices():
    """Feature indices are neither binary nor within [0, 1]."""
    build(**phon_kwargs(phon_tokens=nested(3.0, 17.0, 29.0)))


# --- shared structural rules ----------------------------------------------


@pytest.mark.parametrize("field", ["orth_probs", "phon_probs", "phon_vecs", "phon_tokens"])
def test_nested_fields_reject_non_list_of_lists(field):
    """The shape guard lives in ``validate_nested_tensors``.

    It is only reachable by calling the validator directly: pydantic coerces a bare
    ``list[Tensor]`` into ``list[list[Tensor]]`` (a 1-D tensor is iterable) before the
    model validator runs, so through ``GenerationOutput`` the rank check fires instead.
    """
    with pytest.raises(ValueError, match="must be a list of lists of tensors"):
        validate_nested_tensors([torch.tensor([0.5])], field)


@pytest.mark.parametrize("field", ["orth_probs", "phon_probs", "phon_vecs", "phon_tokens"])
def test_nested_fields_accept_none(field):
    assert validate_nested_tensors(None, field) is None


@pytest.mark.parametrize("field", ["orth_probs", "phon_probs", "phon_vecs", "phon_tokens"])
def test_nested_fields_reject_two_dimensional_tensors(field):
    kwargs = (
        {"orth_tokens": orth_tokens(), "orth_probs": simplex()}
        if field == "orth_probs"
        else phon_kwargs()
    )
    kwargs[field] = [[torch.zeros(2, 2)] for _ in range(BATCH)]
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        build(**kwargs)


def test_a_structural_error_names_the_field_and_position():
    kwargs = {
        "orth_probs": [[torch.zeros(2, 2)] for _ in range(BATCH)],
        "orth_tokens": orth_tokens(),
    }
    with pytest.raises(ValueError, match=r"orth_probs\[0\]\[0\]"):
        build(**kwargs)


def test_a_numeric_error_names_the_field():
    """Numeric checks run once over every row at once, so they name the field only.

    Per-tensor checking cost thousands of device syncs per ``generate()``; the position
    is worth less than that.
    """
    with pytest.raises(ValueError, match="orth_probs probabilities must sum to 1"):
        build(orth_probs=nested(0.1, 0.1), orth_tokens=orth_tokens())


# --- cross-field consistency ----------------------------------------------


def test_at_least_one_modality_required():
    with pytest.raises(ValueError, match="At least one modality"):
        build()


def test_orth_probs_and_tokens_must_appear_together():
    with pytest.raises(ValueError, match="must either both be present or both be None"):
        build(orth_probs=simplex(), **phon_kwargs())


def test_all_three_phonological_fields_must_appear_together():
    with pytest.raises(ValueError, match="All phonological components"):
        build(phon_probs=nested(0.5, 0.5), orth_probs=simplex(), orth_tokens=orth_tokens())


def test_orth_batch_size_must_match_global_encoding():
    with pytest.raises(ValueError, match="orth_probs batch size mismatch"):
        build(
            orth_probs=[[torch.softmax(torch.randn(5), dim=0)]],  # batch of 1 vs global's 2
            orth_tokens=orth_tokens(),
        )


def test_orth_tokens_batch_size_must_match_global_encoding():
    with pytest.raises(ValueError, match="orth_tokens batch size mismatch"):
        build(orth_probs=simplex(), orth_tokens=torch.zeros((1, 3), dtype=torch.long))


@pytest.mark.parametrize("field", ["phon_probs", "phon_vecs", "phon_tokens"])
def test_phonological_batch_mismatch_names_the_field(field):
    """The message used to interpolate the whole tensor list, which was unreadable."""
    kwargs = phon_kwargs()
    kwargs[field] = [[torch.tensor([1.0])]]  # batch of 1 vs global's 2 (binary-safe)
    with pytest.raises(ValueError, match=f"{field} batch size mismatch"):
        build(**kwargs)


# --- global_encoding and orth_tokens ---------------------------------------


def test_global_encoding_must_be_three_dimensional():
    with pytest.raises(ValueError, match="global_encoding must be 3-dimensional"):
        GenerationOutput(
            global_encoding=torch.randn(BATCH, 8),
            orth_probs=simplex(),
            orth_tokens=orth_tokens(),
        )


def test_orth_tokens_must_be_integral():
    with pytest.raises(ValueError, match="orth_tokens must have dtype torch.long or torch.int"):
        build(orth_probs=simplex(), orth_tokens=torch.zeros((BATCH, 3)))


def test_orth_tokens_reject_negative_indices():
    with pytest.raises(ValueError, match="orth_tokens cannot contain negative indices"):
        build(orth_probs=simplex(), orth_tokens=torch.full((BATCH, 3), -1, dtype=torch.long))


def test_orth_tokens_must_be_two_dimensional():
    with pytest.raises(ValueError, match="orth_tokens must be 2-dimensional"):
        build(orth_probs=simplex(), orth_tokens=torch.zeros(BATCH, dtype=torch.long))


def test_rows_of_unequal_length_are_a_validation_error():
    """Batching the numeric check must not let a shape mismatch escape as RuntimeError.

    ``element_check`` now runs once over every row stacked together rather than per
    tensor. ``torch.stack`` raises a bare ``RuntimeError`` on ragged input, which would
    bypass pydantic entirely, so the width is agreed in the structural walk instead.
    """
    with pytest.raises(ValueError, match="has length 3, expected 2"):
        validate_nested_tensors(
            [[torch.tensor([1.0, 0.0]), torch.tensor([0.5, 0.25, 0.25])]],
            "orth_probs",
            check_is_distribution,
        )


def test_rows_on_different_devices_are_a_validation_error():
    """Same reason: ``torch.stack`` would raise past the validator."""
    meta = torch.tensor([1.0, 0.0], device="meta")
    with pytest.raises(ValueError, match="is on device"):
        validate_nested_tensors(
            [[torch.tensor([1.0, 0.0]), meta]], "orth_probs", check_is_distribution
        )


def test_phon_tokens_rows_stay_ragged():
    """The width agreement must not leak onto the field that has no numeric check.

    ``phon_tokens`` holds each position's active feature indices, and phonemes differ in
    how many features they carry, so its rows are ragged by construction. Every
    phon-producing pathway emits one.
    """
    validate_nested_tensors([[torch.tensor([1, 4, 9]), torch.tensor([2, 7])]], "phon_tokens", None)
