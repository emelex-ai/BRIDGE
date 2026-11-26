"""The error surface of ``Model._validate_generate_input``.

There is one set of checks, applied to whatever a pathway reads, so the test is a sweep
rather than a list. A check that one pathway applies and another skips is the defect issue
#233 reports. Each malformation is generated for every pathway/modality pair
that ``PATHWAY_IO`` says is read, which means a new pathway is covered the moment it is
added to the table, and a check that stops applying to one modality fails here rather
than silently narrowing.

A pathway that writes orthography also takes ``orth_start``, the ``[LANG, BOS]`` rows the
caller names the output language with. It is the caller's tensor, so it is checked like
the encoder inputs, and its sweep runs over every pathway that writes orthography.

Every pathway takes ``max_new_tokens``, the cap on what it writes. It is a positive int
and is checked on every pathway. The length of what a pathway reads is not checked: any
length runs.
"""

import numpy as np
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


def start(rows=2, cols=2, dtype=torch.long):
    """A well-formed orthographic start: zeros index the vocabulary like any other id."""
    return torch.zeros((rows, cols), dtype=dtype)


def well_formed(pathway):
    """The keyword arguments ``pathway`` accepts: real tensors for what it reads and, if
    it writes orthography, a start. None else."""
    reads, writes = PATHWAY_IO[pathway]
    return {
        "o": ids() if "orth" in reads else None,
        "om": mask() if "orth" in reads else None,
        "p": ids() if "phon" in reads else None,
        "pm": mask() if "phon" in reads else None,
        "s": start() if "orth" in writes else None,
    }


def validate(model, pathway, o=None, om=None, p=None, pm=None, s=None, cap=5):
    model._validate_generate_input(pathway, o, om, p, pm, s, cap)


# The id-space bound differs per modality; nothing else does.
ID_SPACE = {"orth": VOCAB.orth_vocab_size, "phon": PHONEME_TABLE.num_rows}

# (case id, what to substitute for the modality's ids/mask, expected message fragment).
# `{m}` interpolates the modality prefix the validator uses in its messages.
MALFORMED = [
    (
        "empty_batch",
        lambda m: {"ids": ids(rows=0), "mask": mask(rows=0)},
        "{m}_enc_input has no rows",
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
    # These two cases are what hold the validator's `tensor.device != self.device` loop
    # in place. "meta" is a real torch device that needs no hardware, so they run
    # everywhere rather than only where CUDA is present.
    (
        "ids_on_another_device",
        lambda m: {"ids": ids().to("meta")},
        "_enc_input must be on device",
    ),
    (
        "mask_on_another_device",
        lambda m: {"mask": mask().to("meta")},
        "_enc_pad_mask must be on device",
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

    Accepting it silently is one of the three shapes issue #233 reports: a phonology-only
    encoding reaches pathways that read orthography and fails somewhere deep instead of at
    the boundary.
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


WRITES_ORTH = [pathway for pathway in PATHWAYS if "orth" in PATHWAY_IO[pathway][1]]

# (case id, the start to pass, expected message fragment).
MALFORMED_START = [
    ("missing", None, "writes orthography, so it needs orth_start"),
    ("not_tensor", [[7, 0], [7, 0]], "orth_start must be a torch.Tensor"),
    ("float", start(dtype=torch.float), "orth_start must have dtype torch.long or torch.int"),
    ("one_row_for_two", start(rows=1), "orth_start must be (2, 2)"),
    ("a_row_too_many", start(rows=3), "orth_start must be (2, 2)"),
    ("lone_bos", start(cols=1), "orth_start must be (2, 2)"),
    ("three_wide", start(cols=3), "orth_start must be (2, 2)"),
    ("one_dimensional", torch.zeros(2, dtype=torch.long), "orth_start must be (2, 2)"),
    ("on_another_device", start().to("meta"), "orth_start must be on device"),
    (
        "id_out_of_range",
        torch.full((2, 2), VOCAB.orth_vocab_size),
        f"orth_start ids must lie in [0, {VOCAB.orth_vocab_size})",
    ),
    ("negative_id", torch.full((2, 2), -1), "orth_start ids must lie in [0, "),
]


@pytest.mark.parametrize("pathway", WRITES_ORTH)
@pytest.mark.parametrize(
    ("bad_start", "message"), [pytest.param(s, m, id=i) for i, s, m in MALFORMED_START]
)
def test_every_pathway_that_writes_orthography_checks_its_start(model, pathway, bad_start, message):
    """Oracle: the rule that the start is ``[LANG, BOS]`` for each row of the batch.

    Everything else about the call is well formed, so the rejection is the start's. A
    missing start names the pathway, since which pathways need one is the thing a caller
    has to learn from the message.
    """
    assert WRITES_ORTH == ["p2o", "op2op", "o2o"]
    kwargs = {**well_formed(pathway), "s": bad_start}
    with pytest.raises(ValueError) as excinfo:
        validate(model, pathway, **kwargs)
    assert message in str(excinfo.value)
    if bad_start is None:
        assert f"pathway {pathway!r}" in str(excinfo.value)


@pytest.mark.parametrize("pathway", [p for p in PATHWAYS if p not in WRITES_ORTH])
def test_a_pathway_that_writes_no_orthography_needs_no_start(model, pathway):
    """The control for the sweep above: the start is asked for by what a pathway writes."""
    assert well_formed(pathway)["s"] is None
    validate(model, pathway, **well_formed(pathway))


# (case id, the cap to pass). Each is something other than a positive int.
MALFORMED_CAP = [
    ("zero", 0),
    ("negative", -1),
    ("float", 2.5),
    ("whole_float", 3.0),
    ("string", "3"),
    ("none", None),
    ("true", True),
    ("tensor", torch.tensor(3)),
    ("numpy_int", np.int64(3)),
]


@pytest.mark.parametrize("pathway", PATHWAYS)
@pytest.mark.parametrize("bad_cap", [pytest.param(cap, id=i) for i, cap in MALFORMED_CAP])
def test_every_pathway_rejects_a_cap_that_is_not_a_positive_int(model, pathway, bad_cap):
    """Oracle: the rule that ``max_new_tokens`` is a positive int, on every pathway.

    Everything else about the call is well formed, so the rejection is the cap's. The
    message shows the value that was passed and its type, since an integer of numpy's
    prints like an int: ``got 3`` alone would read as a contradiction. ``True`` is in the
    list because Python counts it as the int 1.
    """
    with pytest.raises(ValueError) as excinfo:
        validate(model, pathway, **well_formed(pathway), cap=bad_cap)
    assert "max_new_tokens must be a positive int" in str(excinfo.value)
    assert str(excinfo.value).endswith(f"got {bad_cap!r} ({type(bad_cap).__name__})")


@pytest.mark.parametrize("pathway", PATHWAYS)
@pytest.mark.parametrize("cap", [1, 2, 30, 10_000])
def test_every_positive_int_is_a_cap(model, pathway, cap):
    """The control for the sweep above, at the boundary and far from it: one is the
    smallest cap there is, and no cap is too large."""
    validate(model, pathway, **well_formed(pathway), cap=cap)


@pytest.mark.parametrize("pathway", WRITES_ORTH)
def test_a_missing_start_is_reported_before_a_bad_cap(model, pathway):
    """A call wrong in two ways reports the start. Which check fires is part of the
    contract, and the start is the one a caller cannot guess a value for."""
    with pytest.raises(ValueError, match="needs orth_start"):
        validate(model, pathway, **{**well_formed(pathway), "s": None}, cap=0)


@pytest.mark.parametrize("pathway", PATHWAYS)
@pytest.mark.parametrize("cols", [31, 60, 500])
def test_an_input_of_any_length_is_accepted(model, pathway, cols):
    """Oracle: the design. Position is a rotation computed for whatever length arrives,
    so no length is refused at the boundary."""
    reads = PATHWAY_IO[pathway][0]
    kwargs = well_formed(pathway)
    if "orth" in reads:
        kwargs["o"], kwargs["om"] = ids(cols=cols), mask(cols=cols)
    if "phon" in reads:
        kwargs["p"], kwargs["pm"] = ids(cols=cols), mask(cols=cols)
    validate(model, pathway, **kwargs)


def test_an_unknown_pathway_is_rejected(model):
    with pytest.raises(ValueError, match="Invalid pathway: nope"):
        validate(model, "nope")


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
