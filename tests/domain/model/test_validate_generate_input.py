"""Pins the full error surface of ``Model._validate_generate_input``.

This validation used to be five hand-copied per-pathway blocks (417 lines). It is now
shared helpers plus per-pathway specifics, so a single edit can silently change the
behaviour of a pathway that no other test covers. Each case below asserts the exact
exception TYPE and message for one malformed input, per pathway.

Three properties matter and are easy to break:

* **type** — ``op2op`` reports a non-tensor as ``TypeError``; the single-modality
  pathways report it as ``ValueError``. That asymmetry is load-bearing for callers
  that catch one and not the other.
* **order** — when an input is wrong in two ways at once, which error fires is
  determined by check order. ``o2p`` checks the mask's dtype *before* the
  input/mask shape match, so ``mask_dtype_beats_shape_mismatch`` below pins that.
* **coverage** — ``o2p`` deliberately performs neither vocabulary-bound nor device
  checks, while ``o2o`` performs both. Adding a check to the shared helper would
  start rejecting input ``o2p`` used to accept.
"""

import pytest
import torch

from bridge.domain.datamodels import ModelConfig, VocabSpec
from bridge.domain.model import Model
from bridge.domain.model.model import PATHWAYS

VOCAB = VocabSpec(
    orth_vocab_size=49,
    phon_vocab_size=34,
    orth_pad_id=2,
    orth_bos_id=0,
    orth_eos_id=1,
    orth_spc_id=41,
    phon_pad_id=33,
    phon_bos_id=29,
    phon_eos_id=30,
    phon_spc_id=32,
)


@pytest.fixture(scope="module")
def model():
    return Model(ModelConfig(vocab=VOCAB, d_model=32, nhead=2, seed=1))


def orth(rows=2, cols=5, dtype=torch.long):
    return torch.zeros((rows, cols), dtype=dtype)


def orth_mask(rows=2, cols=5, dtype=torch.bool):
    return torch.zeros((rows, cols), dtype=dtype)


def phon(rows=2, steps=3):
    return [[torch.tensor([0]) for _ in range(steps)] for _ in range(rows)]


def phon_mask(rows=2, cols=3, dtype=torch.bool):
    return torch.zeros((rows, cols), dtype=dtype)


def validate(model, pathway, o=None, om=None, p=None, pm=None):
    model._validate_generate_input(pathway, o, om, p, pm)


# (id, pathway, kwargs, expected exception, expected message fragment)
CASES = [
    # ---- pathway gate -----------------------------------------------------
    ("invalid_pathway", "nope", {}, ValueError, "Invalid pathway: nope"),
    # ---- global phonological sequence-length bound (applies to every pathway)
    (
        "phon_seq_too_long",
        "p2o",
        {"p": [[torch.tensor([0])] * 31 for _ in range(2)], "pm": phon_mask(2, 31)},
        ValueError,
        "Phonological input sequence length 31 exceeds maximum allowed length 30",
    ),
    # ---- o2p --------------------------------------------------------------
    ("o2p_missing_input", "o2p", {}, ValueError, "orth_enc_input is required for o2p pathway"),
    (
        "o2p_missing_mask",
        "o2p",
        {"o": orth()},
        ValueError,
        "orth_enc_pad_mask is required for o2p pathway",
    ),
    (
        "o2p_input_not_tensor",
        "o2p",
        {"o": [1, 2], "om": orth_mask()},
        ValueError,  # NOT TypeError — only op2op upgrades this
        "orth_enc_input must be a torch.Tensor",
    ),
    (
        "o2p_input_1d",
        "o2p",
        {"o": torch.zeros(5, dtype=torch.long), "om": orth_mask()},
        ValueError,
        "Expected 2D input tensor for orth_enc_input, got shape: (5,)",
    ),
    (
        "o2p_input_float",
        "o2p",
        {"o": orth(dtype=torch.float), "om": orth_mask()},
        ValueError,
        "orth_enc_input must have dtype torch.long or torch.int",
    ),
    (
        "o2p_mask_not_tensor",
        "o2p",
        {"o": orth(), "om": [1]},
        ValueError,
        "orth_enc_pad_mask must be a torch.Tensor",
    ),
    (
        "o2p_mask_1d",
        "o2p",
        {"o": orth(), "om": torch.zeros(5, dtype=torch.bool)},
        ValueError,
        "Expected 2D input tensor for orth_enc_pad_mask",
    ),
    (
        # Ordering guard: the mask is BOTH the wrong dtype and the wrong shape.
        # The dtype check must win, as it did before the refactor.
        "o2p_mask_dtype_beats_shape_mismatch",
        "o2p",
        {"o": orth(2, 5), "om": orth_mask(2, 6, dtype=torch.float)},
        ValueError,
        "orth_enc_pad_mask must have dtype torch.bool",
    ),
    (
        "o2p_shape_mismatch",
        "o2p",
        {"o": orth(2, 5), "om": orth_mask(2, 6)},
        ValueError,
        "Input and mask shapes must match",
    ),
    # ---- p2o / p2p (identical checks, pathway name interpolated) -----------
    *[
        case
        for pw in ("p2o", "p2p")
        for case in [
            (
                f"{pw}_orth_must_be_none",
                pw,
                {"o": orth(), "p": phon(), "pm": phon_mask()},
                ValueError,
                f"{pw} pathway expects orthographic inputs (orth_enc_input, orth_enc_pad_mask) "
                "to be None as they are not used in this pathway.",
            ),
            (
                f"{pw}_missing_phon",
                pw,
                {},
                ValueError,
                f"{pw} pathway requires phonological inputs (phon_enc_input, phon_enc_pad_mask). "
                "Received None value(s).",
            ),
            (
                f"{pw}_phon_not_list",
                pw,
                {"p": "x", "pm": phon_mask()},
                TypeError,
                "phon_enc_input must be a list of lists of tensors",
            ),
            (
                f"{pw}_phon_inner_not_list",
                pw,
                {"p": [torch.tensor([0])], "pm": phon_mask()},
                TypeError,
                "Each item in phon_enc_input must be a list of tensors containing feature indices",
            ),
            (
                f"{pw}_phon_inner_not_tensor",
                pw,
                {"p": [[1]], "pm": phon_mask()},
                TypeError,
                "Feature indices in phon_enc_input must be torch.Tensor objects",
            ),
            (
                f"{pw}_mask_not_tensor",
                pw,
                {"p": phon(), "pm": [1]},
                TypeError,
                "phon_enc_pad_mask must be a torch.Tensor",
            ),
            (
                f"{pw}_mask_dtype",
                pw,
                {"p": phon(), "pm": phon_mask(dtype=torch.float)},
                TypeError,
                "phon_enc_pad_mask must be a boolean tensor",
            ),
            (
                f"{pw}_batch_mismatch",
                pw,
                {"p": phon(rows=2), "pm": phon_mask(rows=5)},
                ValueError,
                "Batch size mismatch: phon_enc_input has 2 items but phon_enc_pad_mask has 5 items",
            ),
            (
                f"{pw}_feature_out_of_vocab",
                pw,
                {"p": [[torch.tensor([VOCAB.phon_vocab_size])]], "pm": phon_mask(1, 1)},
                ValueError,
                f"Feature indices must be less than vocabulary size ({VOCAB.phon_vocab_size})",
            ),
        ]
    ],
    # ---- o2o --------------------------------------------------------------
    (
        "o2o_phon_must_be_none",
        "o2o",
        {"o": orth(), "om": orth_mask(), "p": phon(), "pm": phon_mask()},
        ValueError,
        "o2o pathway expects phonological inputs (phon_enc_input, phon_enc_pad_mask) "
        "to be None as they are not used in this pathway.",
    ),
    (
        "o2o_missing_orth",
        "o2o",
        {},
        ValueError,
        "o2o pathway requires orthographic inputs (orth_enc_input, orth_enc_pad_mask). "
        "Received None value(s).",
    ),
    (
        "o2o_input_not_tensor",
        "o2o",
        {"o": [1], "om": orth_mask()},
        ValueError,
        "orth_enc_input must be a torch.Tensor",
    ),
    (
        "o2o_token_out_of_vocab",
        "o2o",
        {"o": torch.full((2, 5), VOCAB.orth_vocab_size), "om": orth_mask()},
        ValueError,
        f"Input tokens must be less than vocabulary size ({VOCAB.orth_vocab_size})",
    ),
    # ---- op2op ------------------------------------------------------------
    (
        "op2op_missing_orth",
        "op2op",
        {"om": orth_mask(), "p": phon(), "pm": phon_mask()},
        ValueError,
        "op2op pathway requires orthographic inputs (orth_enc_input, orth_enc_pad_mask)",
    ),
    (
        "op2op_missing_phon",
        "op2op",
        {"o": orth(), "om": orth_mask(), "pm": phon_mask()},
        ValueError,
        "op2op pathway requires phonological inputs (phon_enc_input, phon_enc_pad_mask)",
    ),
    (
        # op2op is the ONLY pathway that reports a non-tensor as TypeError.
        "op2op_input_not_tensor_is_TypeError",
        "op2op",
        {"o": [1, 2, 3], "om": orth_mask(), "p": phon(), "pm": phon_mask()},
        TypeError,
        "orth_enc_input must be a torch.Tensor",
    ),
    (
        "op2op_mask_not_tensor_is_TypeError",
        "op2op",
        {"o": orth(), "om": [1], "p": phon(), "pm": phon_mask()},
        TypeError,
        "orth_enc_pad_mask must be a torch.Tensor",
    ),
    (
        "op2op_mask_dtype_is_ValueError",
        "op2op",
        {"o": orth(), "om": orth_mask(dtype=torch.float), "p": phon(), "pm": phon_mask()},
        ValueError,
        "orth_enc_pad_mask must have dtype torch.bool",
    ),
    (
        # Only op2op bounds the orthographic sequence length.
        "op2op_orth_seq_too_long",
        "op2op",
        {"o": orth(2, 31), "om": orth_mask(2, 31), "p": phon(), "pm": phon_mask()},
        ValueError,
        "Orthographic input sequence length 31 exceeds maximum allowed length 30",
    ),
    (
        "op2op_cross_modality_batch_mismatch",
        "op2op",
        {"o": orth(rows=5), "om": orth_mask(rows=5), "p": phon(rows=2), "pm": phon_mask(rows=2)},
        ValueError,
        "Batch size mismatch: orthographic input has 5 items but phonological input has 2 items",
    ),
    (
        # Before the shared helper existed, op2op raised the shorter
        # "...must be a list of tensors" here while p2o/p2p raised this longer form.
        # Deduplicating adopted the more informative wording for all three; the
        # exception type is unchanged. Pinned so the unification stays deliberate.
        "op2op_phon_inner_not_list",
        "op2op",
        {
            "o": orth(),
            "om": orth_mask(),
            "p": [torch.tensor([0]), torch.tensor([1])],
            "pm": phon_mask(),
        },
        TypeError,
        "Each item in phon_enc_input must be a list of tensors containing feature indices",
    ),
    (
        "op2op_phon_not_list",
        "op2op",
        {"o": orth(), "om": orth_mask(), "p": "x", "pm": phon_mask()},
        TypeError,
        "phon_enc_input must be a list of lists of tensors",
    ),
    (
        "op2op_phon_leaf_not_tensor",
        "op2op",
        {"o": orth(), "om": orth_mask(), "p": [[1], [1]], "pm": phon_mask()},
        TypeError,
        "Feature indices in phon_enc_input must be torch.Tensor objects",
    ),
    (
        "op2op_phon_mask_dtype",
        "op2op",
        {"o": orth(), "om": orth_mask(), "p": phon(), "pm": phon_mask(dtype=torch.float)},
        TypeError,
        "phon_enc_pad_mask must be a boolean tensor",
    ),
    (
        "op2op_phon_out_of_vocab",
        "op2op",
        {
            "o": orth(),
            "om": orth_mask(),
            "p": [[torch.tensor([VOCAB.phon_vocab_size])] for _ in range(2)],
            "pm": phon_mask(2, 1),
        },
        ValueError,
        f"Feature indices must be less than vocabulary size ({VOCAB.phon_vocab_size})",
    ),
    (
        # Distinct wording from o2o's "Input tokens ..." for the same condition.
        "op2op_orth_out_of_vocab",
        "op2op",
        {
            "o": torch.full((2, 5), VOCAB.orth_vocab_size),
            "om": orth_mask(),
            "p": phon(),
            "pm": phon_mask(),
        },
        ValueError,
        f"Orthographic tokens must be less than vocabulary size ({VOCAB.orth_vocab_size})",
    ),
]


@pytest.mark.parametrize(
    ("pathway", "kwargs", "exc", "message"),
    [pytest.param(*c[1:], id=c[0]) for c in CASES],
)
def test_validation_error_surface(model, pathway, kwargs, exc, message):
    with pytest.raises(exc) as excinfo:
        validate(model, pathway, **kwargs)
    assert message in str(excinfo.value), (
        f"expected {exc.__name__} containing:\n  {message}\ngot {type(excinfo.value).__name__}:\n  {excinfo.value}"
    )


# --- checks that must NOT be performed ------------------------------------


def test_o2p_does_not_check_vocabulary_bounds(model):
    """o2p never bounded orthographic token ids; o2o and op2op do.

    Hoisting the bound check into the shared orthographic helper would start
    rejecting input that o2p has always accepted.
    """
    validate(
        model,
        "o2p",
        o=torch.full((2, 5), 10**6, dtype=torch.long),
        om=orth_mask(),
    )


def test_o2p_does_not_require_phon_inputs_to_be_none(model):
    """Unlike o2o, o2p never rejected stray phonological arguments."""
    validate(model, "o2p", o=orth(), om=orth_mask(), p=phon(), pm=phon_mask())


def test_op2op_does_not_check_device(model):
    """o2o checks tensor device placement; op2op never did."""
    validate(model, "op2op", o=orth(), om=orth_mask(), p=phon(), pm=phon_mask())


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_valid_inputs_pass(model, pathway):
    """Every pathway accepts a well-formed instance of exactly what it consumes."""
    uses_orth = pathway in ("o2p", "o2o", "op2op")
    uses_phon = pathway in ("p2o", "p2p", "op2op")
    validate(
        model,
        pathway,
        o=orth() if uses_orth else None,
        om=orth_mask() if uses_orth else None,
        p=phon() if uses_phon else None,
        pm=phon_mask() if uses_phon else None,
    )


def test_pathways_constant_matches_literal():
    """PATHWAYS drives the validity gate; keep it in sync with the Pathway alias."""
    assert set(PATHWAYS) == {"o2p", "p2o", "op2op", "p2p", "o2o"}
