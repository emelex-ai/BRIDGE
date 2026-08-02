"""Pins ``EncodingComponent`` / ``BridgeEncoding`` construction, validation and device moves.

``BridgeEncoding.__post_init__`` is the only place the orthographic and phonological
component shapes are checked, and every tokenizer output flows through it. The
validation was reorganised (validate first, then derive ``device``) and three copies
of the component-walk in ``from_dict`` / ``to`` / ``__getitem__`` were collapsed into
``EncodingComponent.to``, so this file states the surviving contract.
"""

import dataclasses

import pytest
import torch

from bridge.domain.datamodels import BridgeEncoding, EncodingComponent


def orth_component(batch=2, enc_len=3, dec_len=2, **over):
    kwargs = {
        "enc_input_ids": torch.zeros((batch, enc_len), dtype=torch.long),
        "enc_pad_mask": torch.zeros((batch, enc_len), dtype=torch.bool),
        "dec_input_ids": torch.zeros((batch, dec_len), dtype=torch.long),
        "dec_pad_mask": torch.zeros((batch, dec_len), dtype=torch.bool),
    }
    kwargs.update(over)
    return EncodingComponent(**kwargs)


def phon_component(batch=2, steps=1, targets=None, **over):
    kwargs = {
        "enc_input_ids": [[torch.tensor([0]) for _ in range(steps)] for _ in range(batch)],
        "enc_pad_mask": torch.zeros((batch, steps), dtype=torch.bool),
        "dec_input_ids": [[torch.tensor([0]) for _ in range(steps)] for _ in range(batch)],
        "dec_pad_mask": torch.zeros((batch, steps), dtype=torch.bool),
        "targets": targets,
    }
    kwargs.update(over)
    return EncodingComponent(**kwargs)


def encoding(**over):
    kwargs = {"orthographic": orth_component(), "phonological": phon_component()}
    kwargs.update(over)
    return BridgeEncoding(**kwargs)


# --- happy path ------------------------------------------------------------


def test_valid_encoding_constructs_and_reports_batch_size():
    assert len(encoding()) == 2


def test_device_is_derived_from_the_orthographic_tensors_not_the_argument():
    """The ``device`` field is advisory: __post_init__ overwrites it from the tensors."""
    enc = BridgeEncoding(
        orthographic=orth_component(),
        phonological=phon_component(),
        device=torch.device("meta"),  # deliberately wrong
    )
    assert enc.device == torch.device("cpu")


def test_targets_are_optional_on_a_component():
    assert encoding().phonological.targets is None


def test_phon_targets_returns_the_tensor_when_present():
    targets = torch.zeros((2, 1, 5), dtype=torch.long)
    enc = encoding(phonological=phon_component(targets=targets))
    assert torch.equal(enc.phonological.phon_targets, targets)


def test_phon_targets_raises_when_absent():
    with pytest.raises(AttributeError, match="Phonological targets are not available"):
        _ = encoding().phonological.phon_targets


# --- validation surface ----------------------------------------------------

ORTH_INVALID = [
    (
        "not_a_tensor",
        {"enc_input_ids": [1, 2]},
        "Orthographic enc_input_ids must be a torch.Tensor",
    ),
    (
        "wrong_rank",
        {"enc_input_ids": torch.zeros(3, dtype=torch.long)},
        "Orthographic enc_input_ids must be 2-dimensional",
    ),
    (
        "float_dtype",
        {"enc_input_ids": torch.zeros((2, 3))},
        "Orthographic enc_input_ids must have dtype torch.long or torch.int",
    ),
    (
        "negative_index",
        {"enc_input_ids": torch.full((2, 3), -1, dtype=torch.long)},
        "Orthographic enc_input_ids cannot contain negative indices",
    ),
    (
        "mask_not_bool",
        {"enc_pad_mask": torch.zeros((2, 3))},
        "Orthographic enc_pad_mask must have dtype torch.bool",
    ),
    (
        "mask_wrong_rank",
        {"enc_pad_mask": torch.zeros(3, dtype=torch.bool)},
        "Orthographic enc_pad_mask must be 2-dimensional",
    ),
    (
        "mask_batch_mismatch",
        {"enc_pad_mask": torch.zeros((5, 3), dtype=torch.bool)},
        "Batch size mismatch: orthographic enc_pad_mask has size 5",
    ),
]


@pytest.mark.parametrize(
    ("override", "message"), [pytest.param(o, m, id=i) for i, o, m in ORTH_INVALID]
)
def test_invalid_orthographic_component_is_rejected(override, message):
    with pytest.raises(ValueError, match=None) as exc:
        encoding(orthographic=orth_component(**override))
    assert message in str(exc.value)


PHON_INVALID = [
    (
        "not_a_list",
        {"enc_input_ids": torch.zeros((2, 1))},
        "Phonological enc_input_ids must be a list of lists of tensors",
    ),
    (
        "inner_not_a_list",
        {"enc_input_ids": [torch.tensor([0]), torch.tensor([0])]},
        "Each batch in phonological enc_input_ids must be a list",
    ),
    (
        "leaf_not_a_tensor",
        {"enc_input_ids": [[0], [0]]},
        "All elements in phonological enc_input_ids must be torch.Tensor",
    ),
    (
        "mask_not_bool",
        {"enc_pad_mask": torch.zeros((2, 1))},
        "Phonological enc_pad_mask must have dtype torch.bool",
    ),
    (
        "targets_wrong_rank",
        {"targets": torch.zeros((2, 3))},
        "Phonological targets must be 3-dimensional",
    ),
    (
        "targets_batch_mismatch",
        {"targets": torch.zeros((7, 1, 5))},
        "Batch size mismatch: phonological targets has size 7",
    ),
]


@pytest.mark.parametrize(
    ("override", "message"), [pytest.param(o, m, id=i) for i, o, m in PHON_INVALID]
)
def test_invalid_phonological_component_is_rejected(override, message):
    with pytest.raises(ValueError) as exc:
        encoding(phonological=phon_component(**override))
    assert message in str(exc.value)


def test_component_batch_sizes_must_agree():
    with pytest.raises(ValueError, match="Batch size mismatch: orthographic component has 2"):
        encoding(orthographic=orth_component(batch=2), phonological=phon_component(batch=3))


def test_non_tensor_orthographic_ids_raise_validation_error_not_attribute_error():
    """Validation runs before ``device`` is read off the tensors.

    Reversing that order surfaces an AttributeError from ``.device`` instead of the
    intended ValueError.
    """
    with pytest.raises(ValueError, match="Orthographic enc_input_ids must be a torch.Tensor"):
        encoding(orthographic=orth_component(enc_input_ids="not a tensor"))


# --- device movement -------------------------------------------------------


def test_to_same_device_returns_self():
    """Documented short-circuit: no copy when already on the target device."""
    enc = encoding()
    assert enc.to(torch.device("cpu")) is enc


def test_to_same_device_preserves_contents():
    enc = encoding()
    moved = enc.to(torch.device("cpu"))
    assert torch.equal(moved.orthographic.enc_input_ids, enc.orthographic.enc_input_ids)
    assert moved.device == enc.device


def test_component_to_handles_both_component_shapes():
    """Orthographic ids are a tensor; phonological ids are list[list[Tensor]]."""
    cpu = torch.device("cpu")
    orth = orth_component().to(cpu)
    assert isinstance(orth.enc_input_ids, torch.Tensor)

    phon = phon_component(targets=torch.zeros((2, 1, 4), dtype=torch.long)).to(cpu)
    assert isinstance(phon.enc_input_ids, list)
    assert isinstance(phon.enc_input_ids[0], list)
    assert isinstance(phon.enc_input_ids[0][0], torch.Tensor)
    assert phon.targets is not None


def test_component_to_keeps_targets_none_when_absent():
    assert phon_component().to(torch.device("cpu")).targets is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_to_other_device_moves_every_tensor():
    cuda = torch.device("cuda")
    moved = encoding(
        phonological=phon_component(targets=torch.zeros((2, 1, 4), dtype=torch.long))
    ).to(cuda)
    assert moved.device.type == "cuda"
    assert moved.orthographic.enc_input_ids.device.type == "cuda"
    assert moved.phonological.enc_pad_mask.device.type == "cuda"
    assert moved.phonological.targets.device.type == "cuda"
    assert all(t.device.type == "cuda" for b in moved.phonological.enc_input_ids for t in b)


def test_components_on_mixed_devices_are_rejected():
    """Only reachable with a real second device; the check itself is cheap to state."""
    enc = encoding()
    assert enc.orthographic.enc_input_ids.device == enc.device
    assert all(t.device == enc.device for b in enc.phonological.enc_input_ids for t in b)


# --- immutability ----------------------------------------------------------


def test_encoding_is_frozen():
    enc = encoding()
    with pytest.raises(dataclasses.FrozenInstanceError):
        enc.device = torch.device("meta")


def test_component_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        orth_component().enc_input_ids = torch.zeros((1, 1), dtype=torch.long)
