"""Pins the error surface of ``Model._validate_generate_input``.

The validation used to be five hand-copied per-pathway blocks applying four different
subsets of the available checks, and this file was 404 lines pinning those differences in
place: ``o2p`` alone skipped the range and device checks, ``op2op`` alone reported a
non-tensor as ``TypeError``, and only ``op2op`` bounded the orthographic sequence length.
None of that was designed; it was what a partial refactor left behind, and issue #233
reported the gaps as defects.

There is one set of checks now, applied to whatever a pathway reads, so the test is a
sweep rather than a list. Each malformation is generated for every pathway/modality pair
that ``PATHWAY_IO`` says is read, which means a new pathway is covered the moment it is
added to the table, and a check that stops applying to one modality fails here rather
than silently narrowing.
"""

import pytest
import torch

from bridge.domain.datamodels import ModelConfig
from bridge.domain.model import Model
from bridge.domain.model.model import MODALITIES, PATHWAY_IO, PATHWAYS
from tests.vocab import PHONEME_TABLE, TEST_VOCAB

VOCAB = TEST_VOCAB


@pytest.fixture(scope="module")
def model():
    return Model(ModelConfig(vocab=VOCAB, d_model=32, nhead=2, seed=1))


def ids(rows=2, cols=4, dtype=torch.long):
    """A well-formed id tensor. Zero indexes both id spaces, so it suits either modality."""
    return torch.zeros((rows, cols), dtype=dtype)


def mask(rows=2, cols=4, dtype=torch.bool):
    return torch.zeros((rows, cols), dtype=dtype)


def well_formed(pathway):
    """The keyword arguments ``pathway`` accepts: real tensors for what it reads, None else."""
    reads = PATHWAY_IO[pathway][0]
    return {
        "o": ids() if "orth" in reads else None,
        "om": mask() if "orth" in reads else None,
        "p": ids() if "phon" in reads else None,
        "pm": mask() if "phon" in reads else None,
    }


def validate(model, pathway, o=None, om=None, p=None, pm=None):
    model._validate_generate_input(pathway, o, om, p, pm)


# The id-space bound differs per modality; nothing else does.
ID_SPACE = {"orth": VOCAB.orth_vocab_size, "phon": PHONEME_TABLE.num_rows}

# (case id, what to substitute for the modality's ids/mask, expected message fragment).
# `{m}` interpolates the modality prefix the validator uses in its messages.
MALFORMED = [
    ("input_not_tensor", lambda m: {"ids": [1, 2]}, "{m}_enc_input must be a torch.Tensor"),
    ("mask_not_tensor", lambda m: {"mask": [1, 2]}, "{m}_enc_pad_mask must be a torch.Tensor"),
    (
        "input_1d",
        lambda m: {"ids": torch.zeros(4, dtype=torch.long)},
        "Expected 2D input tensor for {m}_enc_input",
    ),
    (
        "input_float",
        lambda m: {"ids": ids(dtype=torch.float)},
        "{m}_enc_input must have dtype torch.long or torch.int",
    ),
    (
        "mask_float",
        lambda m: {"mask": mask(dtype=torch.float)},
        "{m}_enc_pad_mask must have dtype torch.bool",
    ),
    (
        "shape_mismatch",
        lambda m: {"mask": mask(cols=6)},
        "Shape mismatch: {m}_enc_input is (2, 4) but {m}_enc_pad_mask is (2, 6)",
    ),
    (
        "empty_batch",
        lambda m: {"ids": ids(rows=0), "mask": mask(rows=0)},
        "{m}_enc_input has no rows",
    ),
    (
        "sequence_too_long",
        lambda m: {"ids": ids(cols=31), "mask": mask(cols=31)},
        "{m}_enc_input sequence length 31 exceeds maximum allowed length 30",
    ),
    (
        "id_out_of_range",
        lambda m: {"ids": torch.full((2, 4), ID_SPACE[m])},
        "ids must lie in [0, {space})",
    ),
    (
        "negative_id",
        lambda m: {"ids": torch.full((2, 4), -1)},
        "ids must lie in [0, {space})",
    ),
]

SWEEP = [
    pytest.param(pathway, modality, mutate, message, id=f"{pathway}_{modality}_{case}")
    for pathway in PATHWAYS
    for modality in MODALITIES
    if modality in PATHWAY_IO[pathway][0]
    for case, mutate, message in MALFORMED
]


@pytest.mark.parametrize(("pathway", "modality", "mutate", "message"), SWEEP)
def test_every_pathway_rejects_every_malformation_of_what_it_reads(
    model, pathway, modality, mutate, message
):
    """One check set, applied to whatever the pathway reads.

    Oracle: the invariant that both modalities carry the same ``(batch, sequence)`` shape,
    which ``docs/architecture.md`` calls the load-bearing structural fact. If the shapes
    are the same then the checks are the same, and a malformation rejected for one
    modality on one pathway must be rejected everywhere that modality is read.
    """
    kwargs = well_formed(pathway)
    prefix = {"orth": ("o", "om"), "phon": ("p", "pm")}[modality]
    for field, value in mutate(modality).items():
        kwargs[prefix[0] if field == "ids" else prefix[1]] = value

    expected = message.format(m=modality, space=ID_SPACE[modality])
    with pytest.raises(ValueError) as excinfo:
        validate(model, pathway, **kwargs)
    assert expected in str(excinfo.value), (
        f"{pathway}/{modality}: expected a ValueError containing\n  {expected}\n"
        f"got {type(excinfo.value).__name__}:\n  {excinfo.value}"
    )


@pytest.mark.parametrize("pathway", PATHWAYS)
@pytest.mark.parametrize("modality", MODALITIES)
def test_a_modality_a_pathway_does_not_read_must_be_absent(model, pathway, modality):
    """Passing phonology to an orthography-only pathway is a mistake, not an extra.

    ``o2p`` used to accept it silently, which is one of the three shapes issue #233
    reported: a phonology-only encoding reached pathways that read orthography and failed
    somewhere deep instead of at the boundary.
    """
    if modality in PATHWAY_IO[pathway][0]:
        pytest.skip(f"{pathway} reads {modality}")
    kwargs = well_formed(pathway)
    key, mask_key = {"orth": ("o", "om"), "phon": ("p", "pm")}[modality]
    kwargs[key], kwargs[mask_key] = ids(), mask()

    with pytest.raises(ValueError, match="to be None as they are not used"):
        validate(model, pathway, **kwargs)


@pytest.mark.parametrize("pathway", PATHWAYS)
@pytest.mark.parametrize("missing", ["ids", "mask"])
def test_a_modality_a_pathway_reads_must_be_present(model, pathway, missing):
    """Half a modality is as unusable as none of it."""
    for modality in PATHWAY_IO[pathway][0]:
        kwargs = well_formed(pathway)
        key, mask_key = {"orth": ("o", "om"), "phon": ("p", "pm")}[modality]
        kwargs[key if missing == "ids" else mask_key] = None
        with pytest.raises(ValueError, match="Received None value"):
            validate(model, pathway, **kwargs)


def test_an_unknown_pathway_is_rejected(model):
    with pytest.raises(ValueError, match="Invalid pathway: nope"):
        validate(model, "nope")


def test_op2op_rejects_a_cross_modality_batch_mismatch(model):
    """The one check that belongs to a pathway rather than to a modality."""
    with pytest.raises(ValueError, match="Batch size mismatch"):
        validate(model, "op2op", o=ids(rows=5), om=mask(rows=5), p=ids(rows=2), pm=mask(rows=2))


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_valid_inputs_pass(model, pathway):
    """Every pathway accepts a well-formed instance of exactly what it consumes.

    The control for the sweep above: without it, a validator that rejected everything
    would pass every rejection case.
    """
    validate(model, pathway, **well_formed(pathway))


def test_pathways_constant_matches_the_io_table():
    """``PATHWAYS`` drives the validity gate and ``PATHWAY_IO`` drives every check.

    A pathway in one and not the other is a hole: listed as valid but with nothing to say
    what it reads, or described but unreachable.
    """
    assert set(PATHWAYS) == set(PATHWAY_IO)
    assert set(PATHWAYS) == {"o2p", "p2o", "op2op", "p2p", "o2o"}
    for pathway, (reads, writes) in PATHWAY_IO.items():
        assert reads <= set(MODALITIES), pathway
        assert writes <= set(MODALITIES), pathway
        assert reads and writes, f"{pathway} must both read and write something"
