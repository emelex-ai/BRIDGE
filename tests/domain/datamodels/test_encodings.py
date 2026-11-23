"""Pins the five types an example is made of: construction, validation and device moves.

``Tokens`` is what an encoder reads and ``Target`` is what a decoder is fed and scored
against. ``BridgeInput`` and ``BridgeTarget`` hold the two modalities of one side, and a
``BridgeBatch`` is one of each. Every tokenizer output flows through their validation, so
this file states the contract.

Oracle: the rule each test names, checked on hand-built tensors. A valid object is built
first in every case, so a rejection can only come from the one field the test changes.
"""

import dataclasses

import pytest
import torch

from bridge.domain.datamodels import BridgeBatch, BridgeInput, BridgeTarget, Target, Tokens

FEATURES = 5


def tokens(batch=2, length=3, **over) -> Tokens:
    kwargs = {
        "ids": torch.zeros((batch, length), dtype=torch.long),
        "pad_mask": torch.zeros((batch, length), dtype=torch.bool),
    }
    return Tokens(**{**kwargs, **over})


def orth_target(batch=2, length=4, **over) -> Target:
    kwargs = {
        "ids": torch.zeros((batch, length), dtype=torch.long),
        "pad_mask": torch.zeros((batch, length), dtype=torch.bool),
        "labels": torch.zeros((batch, length), dtype=torch.long),
    }
    return Target(**{**kwargs, **over})


def phon_target(batch=2, length=3, **over) -> Target:
    kwargs = {
        "ids": torch.zeros((batch, length), dtype=torch.long),
        "pad_mask": torch.zeros((batch, length), dtype=torch.bool),
        "labels": torch.zeros((batch, length, FEATURES), dtype=torch.long),
    }
    return Target(**{**kwargs, **over})


def bridge_input(**over) -> BridgeInput:
    return BridgeInput(**{"orth": tokens(), "phon": tokens(length=2), **over})


def bridge_target(**over) -> BridgeTarget:
    return BridgeTarget(**{"orth": orth_target(), "phon": phon_target(), **over})


def batch(**over) -> BridgeBatch:
    return BridgeBatch(**{"input": bridge_input(), "target": bridge_target(), **over})


# --- happy path ------------------------------------------------------------


def test_valid_objects_construct_and_report_batch_size():
    assert len(bridge_input()) == 2
    assert len(bridge_target()) == 2
    assert len(batch()) == 2


def test_device_is_read_off_the_tensors():
    cpu = tokens().ids.device
    assert bridge_input().device == cpu
    assert bridge_target().device == cpu


def test_input_and_target_lengths_are_independent():
    """Nothing ties the input's width to the target's: they are different strings."""
    wide = BridgeInput(orth=tokens(length=19), phon=tokens(length=17))
    narrow = BridgeTarget(orth=orth_target(length=4), phon=phon_target(length=3))
    assert len(BridgeBatch(input=wide, target=narrow)) == 2


# --- what one sequence must be ----------------------------------------------

SEQUENCE_INVALID = [
    ("ids_not_a_tensor", {"ids": [[1, 2, 3]] * 2}, "ids must be a torch.Tensor"),
    ("ids_wrong_rank", {"ids": torch.zeros(3, dtype=torch.long)}, "ids must be 2-dimensional"),
    ("ids_float", {"ids": torch.zeros((2, 3))}, "ids must have dtype torch.long or torch.int"),
    (
        "ids_negative",
        {"ids": torch.full((2, 3), -1, dtype=torch.long)},
        "ids cannot contain negative indices",
    ),
    ("mask_not_a_tensor", {"pad_mask": None}, "pad_mask must be a torch.Tensor"),
    ("mask_not_bool", {"pad_mask": torch.zeros((2, 3))}, "pad_mask must have dtype torch.bool"),
    (
        "mask_wrong_rank",
        {"pad_mask": torch.zeros(3, dtype=torch.bool)},
        "pad_mask must be 2-dimensional",
    ),
    (
        "mask_other_batch",
        {"pad_mask": torch.zeros((5, 3), dtype=torch.bool)},
        "ids are (2, 3) but pad_mask is (5, 3)",
    ),
    (
        "mask_wider_than_ids",
        {"pad_mask": torch.zeros((2, 7), dtype=torch.bool)},
        "ids are (2, 3) but pad_mask is (2, 7)",
    ),
]


@pytest.mark.parametrize(
    ("override", "message"), [pytest.param(o, m, id=i) for i, o, m in SEQUENCE_INVALID]
)
def test_invalid_tokens_are_rejected(override, message):
    with pytest.raises(ValueError) as exc:
        tokens(**override)
    assert "Tokens " + message in str(exc.value)


@pytest.mark.parametrize(
    ("override", "message"), [pytest.param(o, m, id=i) for i, o, m in SEQUENCE_INVALID]
)
def test_a_target_checks_its_ids_and_mask_as_tokens_do(override, message):
    with pytest.raises(ValueError) as exc:
        phon_target(**override)
    assert "Target " + message in str(exc.value)


TARGET_INVALID = [
    ("labels_not_a_tensor", {"labels": [[0, 0, 0, 0]] * 2}, "labels must be a torch.Tensor"),
    ("labels_float", {"labels": torch.zeros((2, 4))}, "labels must have dtype torch.long, got"),
    # The loss takes int64 labels only, so a narrower integer is refused here and not there.
    (
        "labels_int32",
        {"labels": torch.zeros((2, 4), dtype=torch.int32)},
        "labels must have dtype torch.long, got torch.int32",
    ),
    (
        "labels_other_batch",
        {"labels": torch.zeros((7, 4), dtype=torch.long)},
        "Target labels are (7, 4), which is not one label per decoder position; ids are (2, 4)",
    ),
    (
        "labels_one_position_short",
        {"labels": torch.zeros((2, 3), dtype=torch.long)},
        "Target labels are (2, 3), which is not one label per decoder position; ids are (2, 4)",
    ),
    (
        "labels_one_position_long",
        {"labels": torch.zeros((2, 5, FEATURES), dtype=torch.long)},
        "Target labels are (2, 5, 5), which is not one label per decoder position",
    ),
]


@pytest.mark.parametrize(
    ("override", "message"), [pytest.param(o, m, id=i) for i, o, m in TARGET_INVALID]
)
def test_labels_give_one_row_per_decoder_position(override, message):
    """A mismatch here is otherwise a shape error inside ``cross_entropy``, which names
    neither tensor."""
    with pytest.raises(ValueError) as exc:
        orth_target(**override)
    assert message in str(exc.value)


# --- the two modalities of one side ------------------------------------------


def test_the_modalities_of_an_input_agree_on_batch_size():
    with pytest.raises(ValueError, match=r"Batch size mismatch in BridgeInput: .*2.*3"):
        BridgeInput(orth=tokens(batch=2), phon=tokens(batch=3))


def test_the_modalities_of_a_target_agree_on_batch_size():
    with pytest.raises(ValueError, match=r"Batch size mismatch in BridgeTarget: .*2.*3"):
        BridgeTarget(orth=orth_target(batch=2), phon=phon_target(batch=3))


def test_a_batch_has_one_target_per_input():
    with pytest.raises(ValueError, match="Batch size mismatch in BridgeBatch: 2 inputs, 3"):
        BridgeBatch(
            input=bridge_input(),
            target=BridgeTarget(orth=orth_target(batch=3), phon=phon_target(batch=3)),
        )


def test_swapped_target_modalities_are_rejected():
    """Orthographic labels are ``(batch, seq)`` and phonological ones carry a feature
    axis, which is the one thing that tells the two targets apart."""
    with pytest.raises(ValueError, match="BridgeTarget.orth labels must be 2-dimensional"):
        BridgeTarget(orth=phon_target(), phon=orth_target())
    with pytest.raises(ValueError, match="BridgeTarget.phon labels must be 3-dimensional"):
        BridgeTarget(orth=orth_target(), phon=orth_target())


# --- a modality the input does not carry -------------------------------------


@pytest.mark.parametrize("carried", ["orth", "phon"])
def test_an_input_may_carry_one_modality(carried):
    """``None`` stands for the missing one. No tensor is made up to fill its place."""
    inputs = BridgeInput(**{carried: tokens()})
    missing = "phon" if carried == "orth" else "orth"

    assert getattr(inputs, missing) is None
    assert len(inputs) == 2
    assert inputs.device == tokens().ids.device
    assert getattr(inputs.to(torch.device("cpu")), missing) is None


def test_an_input_carries_at_least_one_modality():
    with pytest.raises(ValueError, match="at least one modality"):
        BridgeInput()


def test_a_modality_of_an_input_is_passed_by_name():
    """Oracle: the rule. Both modalities are ``Tokens``, so a positional one would land in
    ``orth`` whatever it held, and phoneme ids would be read as characters."""
    with pytest.raises(TypeError, match="positional"):
        BridgeInput(tokens())
    with pytest.raises(TypeError, match="positional"):
        BridgeInput(tokens(), tokens(length=2))


def test_read_returns_a_carried_modality_and_names_a_missing_one():
    inputs = BridgeInput(orth=tokens())
    assert inputs.read("orth", "o2p") is inputs.orth
    with pytest.raises(
        ValueError, match="pathway 'p2o' reads phon, which this input does not carry"
    ):
        inputs.read("phon", "p2o")


def test_a_target_has_both_modalities():
    with pytest.raises(TypeError, match="BridgeTarget.phon must be a Target, got NoneType"):
        BridgeTarget(orth=orth_target(), phon=None)


# --- an input is not a target ------------------------------------------------


def test_a_batch_of_two_inputs_is_rejected():
    """The swap a repetition test cannot see: both sides hold the same strings, so only
    the type says which is which."""
    with pytest.raises(TypeError, match="BridgeBatch.target must be a BridgeTarget, got"):
        BridgeBatch(input=bridge_input(), target=bridge_input())


def test_a_batch_of_two_targets_is_rejected():
    with pytest.raises(TypeError, match="BridgeBatch.input must be a BridgeInput, got"):
        BridgeBatch(input=bridge_target(), target=bridge_target())


def test_a_batch_with_its_sides_exchanged_is_rejected():
    with pytest.raises(TypeError, match="BridgeBatch.input must be a BridgeInput"):
        BridgeBatch(input=bridge_target(), target=bridge_input())


def test_tokens_are_not_a_target_and_a_target_is_not_tokens():
    with pytest.raises(TypeError, match="BridgeTarget.orth must be a Target, got Tokens"):
        BridgeTarget(orth=tokens(), phon=phon_target())
    with pytest.raises(TypeError, match="BridgeInput.orth must be a Tokens, got Target"):
        BridgeInput(orth=orth_target(), phon=tokens())
    assert not isinstance(orth_target(), Tokens)


def test_tokens_have_no_labels():
    """What stops an input reaching a loss: the loss reads ``labels``."""
    assert not hasattr(tokens(), "labels")


# --- device movement -------------------------------------------------------


def test_to_same_device_preserves_contents():
    one = batch()
    moved = one.to(torch.device("cpu"))
    assert torch.equal(moved.input.orth.ids, one.input.orth.ids)
    assert torch.equal(moved.target.phon.labels, one.target.phon.labels)
    assert moved.input.device == moved.target.device == one.input.device


class Spy(torch.Tensor):
    """A tensor that records each ``.to()`` made on it, under its label."""

    seen: list[str]
    label: str

    def to(self, *args, **kwargs):
        self.seen.append(self.label)
        return super().to(*args, **kwargs)


def test_to_reaches_every_tensor_of_a_batch():
    """Oracle: the ten tensors, named by hand. Needs no second device, so it runs where
    the CUDA tests below skip: a ``.to()`` that leaves a field behind never asks that
    field's tensor to move, whatever device it was headed for."""
    seen: list[str] = []

    def spy(tensor: torch.Tensor, label: str) -> torch.Tensor:
        out = tensor.as_subclass(Spy)
        out.seen, out.label = seen, label
        return out

    def spied(part, name: str):
        fields = {f.name: getattr(part, f.name) for f in dataclasses.fields(part)}
        return type(part)(**{f: spy(tensor, f"{name}.{f}") for f, tensor in fields.items()})

    one = batch()
    spies = BridgeBatch(
        input=BridgeInput(
            orth=spied(one.input.orth, "input.orth"), phon=spied(one.input.phon, "input.phon")
        ),
        target=BridgeTarget(
            orth=spied(one.target.orth, "target.orth"), phon=spied(one.target.phon, "target.phon")
        ),
    )
    seen.clear()
    spies.to(torch.device("cpu"))
    assert sorted(seen) == sorted(
        [
            "input.orth.ids",
            "input.orth.pad_mask",
            "input.phon.ids",
            "input.phon.pad_mask",
            "target.orth.ids",
            "target.orth.pad_mask",
            "target.orth.labels",
            "target.phon.ids",
            "target.phon.pad_mask",
            "target.phon.labels",
        ]
    )


def test_a_mask_or_labels_on_another_device_are_rejected():
    """Oracle: the rule, on the ``meta`` device, which every build of torch has. These
    two checks read placement only, so they can be shown without CUDA."""
    meta = torch.device("meta")
    with pytest.raises(ValueError, match="Device mismatch: Tokens ids on cpu, pad_mask on meta"):
        Tokens(ids=tokens().ids, pad_mask=tokens().pad_mask.to(meta))
    with pytest.raises(ValueError, match="Device mismatch: Target ids on cpu, labels on meta"):
        Target(
            ids=orth_target().ids,
            pad_mask=orth_target().pad_mask,
            labels=orth_target().labels.to(meta),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_to_other_device_moves_every_tensor():
    moved = batch().to(torch.device("cuda"))
    for part in (moved.input.orth, moved.input.phon, moved.target.orth, moved.target.phon):
        assert part.ids.device.type == "cuda"
        assert part.pad_mask.device.type == "cuda"
    assert moved.target.orth.labels.device.type == "cuda"
    assert moved.target.phon.labels.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_parts_on_mixed_devices_are_rejected():
    cuda = torch.device("cuda")
    with pytest.raises(ValueError, match="Device mismatch: Tokens ids on"):
        Tokens(ids=tokens().ids.to(cuda), pad_mask=tokens().pad_mask)
    with pytest.raises(ValueError, match="Device mismatch: Target ids on"):
        Target(
            ids=orth_target().ids,
            pad_mask=orth_target().pad_mask,
            labels=orth_target().labels.to(cuda),
        )
    with pytest.raises(ValueError, match="Device mismatch in BridgeInput"):
        BridgeInput(orth=tokens().to(cuda), phon=tokens())
    with pytest.raises(ValueError, match="Device mismatch in BridgeTarget"):
        BridgeTarget(orth=orth_target().to(cuda), phon=phon_target())
    with pytest.raises(ValueError, match="Device mismatch in BridgeBatch"):
        BridgeBatch(input=bridge_input().to(cuda), target=bridge_target())


# --- immutability ----------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "field"),
    [
        (tokens, "ids"),
        (orth_target, "labels"),
        (bridge_input, "orth"),
        (bridge_target, "phon"),
        (batch, "target"),
    ],
)
def test_every_type_is_frozen(build, field):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(build(), field, None)
