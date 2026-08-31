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
    """Phonological ids are phoneme *row* ids, the same ``(batch, sequence)`` shape as
    orthographic ids since issue #221, so both modalities validate identically."""
    kwargs = {
        "enc_input_ids": torch.zeros((batch, steps), dtype=torch.long),
        "enc_pad_mask": torch.zeros((batch, steps), dtype=torch.bool),
        "dec_input_ids": torch.zeros((batch, steps), dtype=torch.long),
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


def test_device_is_read_off_the_tensors():
    assert encoding().device == orth_component().enc_input_ids.device


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
        "not_a_tensor",
        {"enc_input_ids": [[0], [0]]},
        "Phonological enc_input_ids must be a torch.Tensor",
    ),
    (
        "wrong_rank",
        {"enc_input_ids": torch.zeros(2, dtype=torch.long)},
        "Phonological enc_input_ids must be 2-dimensional",
    ),
    (
        "float_ids",
        {"enc_input_ids": torch.zeros((2, 1))},
        "Phonological enc_input_ids must have dtype torch.long or torch.int",
    ),
    (
        "negative_ids",
        {"enc_input_ids": torch.full((2, 1), -1, dtype=torch.long)},
        "Phonological enc_input_ids cannot contain negative indices",
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


def test_component_to_moves_both_modalities_identically():
    """Both modalities are plain ``(batch, sequence)`` id tensors since issue #221, so
    ``to`` no longer needs a nested walk for the phonological side."""
    cpu = torch.device("cpu")
    orth = orth_component().to(cpu)
    assert isinstance(orth.enc_input_ids, torch.Tensor)

    phon = phon_component(targets=torch.zeros((2, 1, 4), dtype=torch.long)).to(cpu)
    assert isinstance(phon.enc_input_ids, torch.Tensor)
    assert phon.enc_input_ids.dim() == 2
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


def test_components_on_mixed_devices_are_rejected():
    """Only reachable with a real second device; the check itself is cheap to state."""
    enc = encoding()
    assert enc.orthographic.enc_input_ids.device == enc.device
    assert enc.phonological.enc_input_ids.device == enc.device


# --- immutability ----------------------------------------------------------


def test_encoding_is_frozen():
    enc = encoding()
    with pytest.raises(dataclasses.FrozenInstanceError):
        enc.orthographic = orth_component()


def test_component_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        orth_component().enc_input_ids = torch.zeros((1, 1), dtype=torch.long)


def test_a_pad_mask_must_cover_exactly_its_ids():
    """The mask and the ids it masks must agree on both dimensions.

    Checkable only since both modalities became rectangular: at the ragged stage the
    phonological ids had no ``.shape`` to compare against. Previously a too-wide mask
    validated cleanly here and failed much later, inside the encoder.
    """

    def component(enc_mask_width: int) -> EncodingComponent:
        return EncodingComponent(
            enc_input_ids=torch.zeros(2, 3, dtype=torch.long),
            enc_pad_mask=torch.zeros(2, enc_mask_width, dtype=torch.bool),
            dec_input_ids=torch.zeros(2, 3, dtype=torch.long),
            dec_pad_mask=torch.zeros(2, 3, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="enc_pad_mask is"):
        BridgeEncoding(orthographic=component(7), phonological=component(3))


def test_phonological_targets_must_have_one_row_per_decoder_position():
    """A mismatch used to surface only as a shape error inside CrossEntropyLoss."""
    with pytest.raises(ValueError, match="Sequence length mismatch"):
        BridgeEncoding(
            orthographic=EncodingComponent(
                enc_input_ids=torch.zeros(2, 4, dtype=torch.long),
                enc_pad_mask=torch.zeros(2, 4, dtype=torch.bool),
                dec_input_ids=torch.zeros(2, 4, dtype=torch.long),
                dec_pad_mask=torch.zeros(2, 4, dtype=torch.bool),
            ),
            phonological=EncodingComponent(
                enc_input_ids=torch.zeros(2, 4, dtype=torch.long),
                enc_pad_mask=torch.zeros(2, 4, dtype=torch.bool),
                dec_input_ids=torch.zeros(2, 4, dtype=torch.long),
                dec_pad_mask=torch.zeros(2, 4, dtype=torch.bool),
                targets=torch.zeros(2, 99, 5, dtype=torch.long),
            ),
        )
