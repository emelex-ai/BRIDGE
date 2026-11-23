"""Equivalence against the pre-refactor behavioural fingerprint (issue #221).

``tests/fixtures/phon_baseline.pt`` was recorded on the commit before the phonological
refactor began. This module rebuilds the same objects and asserts every recorded value.

**The fixture is immutable.** If a value here changes, either the refactor changed behaviour
or the change was intentional and the spec says so. Regenerating the fixture to make a test
pass defeats its entire purpose.

One value has since been superseded, deliberately, and the fixture was not touched.
Issue #228 changed how the orthographic decoder is seeded at generation: from a lone ``[BOS]``
at position 0 to the ``[LANG, BOS]`` prefix training actually uses. The decoder therefore
continues from a different state and emits different tokens, so the recorded
``orth_tokens`` and ``orth_probs`` describe behaviour that was wrong. Measured at the commit
that changed it, the effect is bounded to exactly that:

===============  ==============  =============  ==============
pathway          global_encoding  orth_tokens    phonological
===============  ==============  =============  ==============
o2p              same             n/a            same
p2o              same             **moved**      n/a
p2p              same             n/a            same
op2op            same             **moved**      same
o2o              same             **moved**      n/a
===============  ==============  =============  ==============

``op2op`` is the load-bearing row: it generates both modalities from one encoding, and its
phonological output is unchanged while its orthographic output moved. Shapes are unchanged
throughout, ``(9, 30)`` and ``(9, 30, 109)``. The orthographic equality assertions below are
reduced to shape, and the new behaviour is pinned against analytic oracles in
``tests/domain/model/test_orth_generation_prefix.py`` rather than against a re-recording.

A second value is superseded the same way. The fixture records ``placeholder_phon``, the
stand-in phonology an orthography-only encoding carried: one position per row, all of it
padding. An input that has no phonology carries ``None`` there and no stand-in exists, so
there is no tensor to compare. What the stand-in was for still holds and is what the
reduced assertion checks: an input with orthography alone generates, on a pathway that
reads orthography alone, exactly what the fixture records for that pathway. Measured
against the commit before the change, generation from such an input is bitwise equal to
generation from the stand-in encoding on ``o2p`` and ``o2o``, 729 of 729 recorded values.
The behaviour of a missing modality is pinned against ``PATHWAY_IO`` in
``tests/domain/model/test_generate.py``.

**The plumbing is not.** The refactor changes the phonological representation from a ragged
``list[list[Tensor]]`` to a ``(batch, sequence)`` tensor of row ids, and the adapters in the
ADAPTERS block below absorb that; everything under ASSERTIONS compares recorded numbers and
should not need to change. The fixture names its tensors by the encoder and decoder that
read them. The encoder's are the ids and mask of the input, the decoder's are the ids and
mask of the target, and the recorded targets are the target's labels; ``FIELD_OF`` below
is that mapping. The words are encoded once as an input and once as a target, which is
repetition, the task the fixture recorded.

Runs on CPU: ``device_manager`` defaults to CPU, and CUDA reduction order differs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bridge.domain.datamodels import BridgeInput, ModelConfig, VocabSpec
from bridge.domain.model import Model
from bridge.domain.tokenizer.bridge_tokenizer import BridgeTokenizer

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "phon_baseline.pt"

FORWARD_ATOL = 1e-6
# Gradients diverge further than the forward pass under a reduction-order change: measured
# worst case is 3.8e-6 absolute, which clears the default rtol=1e-5 but not atol alone.
GRAD_ATOL, GRAD_RTOL = 1e-6, 1e-5


# --------------------------------------------------------------------------- ADAPTERS
# The only places that should need editing as the refactor lands.


def canon_features(ids, tokenizer) -> list[list[tuple[int, ...]]]:
    """Canonicalise phonological inputs to per-position sorted feature-index tuples.

    Accepts both representations: the pre-refactor ragged ``list[list[Tensor]]`` of feature
    indices, and the post-refactor ``(B, L)`` tensor of phoneme row ids.
    """
    if isinstance(ids, torch.Tensor):
        table = tokenizer.phoneme_tokenizer.phoneme_table
        return [[tuple(sorted(table.features_of(int(r)).tolist())) for r in row] for row in ids]
    return [[tuple(sorted(int(v) for v in t.reshape(-1).tolist())) for t in row] for row in ids]


# How a recorded tensor name maps onto an example: the encoder reads the input, the decoder
# is fed the target.
FIELD_OF = {
    "enc_input_ids": ("input", "ids"),
    "enc_pad_mask": ("input", "pad_mask"),
    "dec_input_ids": ("target", "ids"),
    "dec_pad_mask": ("target", "pad_mask"),
}


def recorded_field(example: dict, modality: str, field: str) -> torch.Tensor:
    """The tensor the fixture calls ``<modality>_<field>``, from an input and a target."""
    side, name = FIELD_OF[field]
    return getattr(getattr(example[side], modality), name)


def phon_targets_of(example: dict) -> torch.Tensor:
    """Phonological loss targets."""
    return example["target"].phon.labels


def run_forward(model: Model, example: dict, pathway: str) -> dict[str, torch.Tensor]:
    """Forward pass, whichever calling convention the model currently exposes."""
    parts = {
        f"{modality}_{side}": {
            f"{modality}_{side}_input": recorded_field(example, modality, f"{side}_input_ids"),
            f"{modality}_{side}_pad_mask": recorded_field(example, modality, f"{side}_pad_mask"),
        }
        for modality in ("orth", "phon")
        for side in ("enc", "dec")
    }
    needed = {
        "o2p": ("orth_enc", "phon_dec"),
        "p2o": ("phon_enc", "orth_dec"),
        "p2p": ("phon_enc", "phon_dec"),
        "op2op": ("orth_enc", "orth_dec", "phon_enc", "phon_dec"),
    }[pathway]
    kwargs = {k: v for name in needed for k, v in parts[name].items()}
    return model(task=pathway, **kwargs)


def stack_nested(nested) -> torch.Tensor:
    return torch.stack([torch.stack(row) for row in nested])


# --------------------------------------------------------------------------- FIXTURES


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"{FIXTURE_PATH.name} missing; regenerate with "
            "`uv run python tests/fixtures/capture_phon_baseline.py` on the pre-refactor commit"
        )
    return torch.load(FIXTURE_PATH, weights_only=False)


@pytest.fixture(scope="module")
def rebuilt(baseline):
    tokenizer = BridgeTokenizer()
    example = {
        "input": tokenizer.encode_input(baseline["words"]),
        "target": tokenizer.encode_target(baseline["words"]),
    }
    assert example["input"].phon is not None, "baseline words no longer encode"
    assert example["target"] is not None, "baseline words no longer encode"
    model = Model(
        ModelConfig(
            vocab=VocabSpec(
                **{k: v for k, v in baseline["vocab"].items() if k in VocabSpec.model_fields}
            ),
            d_model=baseline["d_model"],
            nhead=baseline["nhead"],
            seed=baseline["seed"],
        )
    )
    model.eval()
    return tokenizer, example, model


# --------------------------------------------------------------------------- ASSERTIONS


# Recorded fields that VocabSpec no longer carries. `orth_spc_id`/`phon_spc_id` were
# dropped deliberately: nothing ever read them.
RETIRED_VOCAB_FIELDS = {"orth_spc_id", "phon_spc_id"}


def test_vocab_spec_unchanged(baseline, rebuilt):
    tokenizer, _, _ = rebuilt
    current = VocabSpec.from_tokenizer(tokenizer).model_dump()
    for key, expected in baseline["vocab"].items():
        if key in RETIRED_VOCAB_FIELDS:
            assert key not in current, f"VocabSpec.{key} came back"
            continue
        assert current[key] == expected, f"VocabSpec.{key} changed"


@pytest.mark.parametrize(
    "field",
    ["orth_enc_input_ids", "orth_dec_input_ids", "orth_enc_pad_mask", "orth_dec_pad_mask"],
)
def test_orthographic_encoding_unchanged(baseline, rebuilt, field):
    _, example, _ = rebuilt
    actual = recorded_field(example, "orth", field.removeprefix("orth_"))
    assert torch.equal(actual, baseline["tok"][field])


@pytest.mark.parametrize("field", ["phon_enc_pad_mask", "phon_dec_pad_mask"])
def test_phonological_pad_masks_unchanged(baseline, rebuilt, field):
    _, example, _ = rebuilt
    actual = recorded_field(example, "phon", field.removeprefix("phon_"))
    assert torch.equal(actual, baseline["tok"][field])


@pytest.mark.parametrize("side", ["enc", "dec"])
def test_phonological_feature_sets_unchanged(baseline, rebuilt, side):
    """The load-bearing one: same features per position, whatever the storage."""
    tokenizer, example, _ = rebuilt
    actual = canon_features(recorded_field(example, "phon", f"{side}_input_ids"), tokenizer)
    assert actual == baseline["tok"][f"phon_{side}_features"]


def test_phon_targets_unchanged(baseline, rebuilt):
    _, example, _ = rebuilt
    assert torch.equal(phon_targets_of(example), baseline["tok"]["phon_targets"])


def test_an_input_without_phonology_generates_what_the_recorded_stand_in_did(baseline, rebuilt):
    """Superseded in part; see this module's docstring.

    The recorded stand-in is checked to be what the docstring says it was, all padding,
    so nothing it held is being dropped. Then the reason it existed is asserted against
    the recorded generation: ``o2p`` from an input carrying orthography alone.
    """
    tokenizer, example, model = rebuilt
    recorded = baseline["placeholder_phon"]
    rows = len(baseline["words"])
    pad_feature = (tokenizer.phon_pad_id,)
    for side in ("enc", "dec"):
        assert recorded[f"{side}_features"] == [[pad_feature]] * rows
        assert recorded[f"{side}_pad_mask"].shape == (rows, 1)
        assert bool(recorded[f"{side}_pad_mask"].all())

    orthography_only = BridgeInput(orth=example["input"].orth)
    assert orthography_only.phon is None

    result = model.generate(orthography_only, "o2p", deterministic=True)
    expected = baseline["generate"]["o2p"]
    assert torch.allclose(result.global_encoding, expected["global_encoding"], atol=FORWARD_ATOL)
    assert canon_features(result.phon_tokens, tokenizer) == expected["phon_tokens"]
    assert torch.equal(stack_nested(result.phon_vecs), expected["phon_vecs"])


@pytest.mark.parametrize("side", ["enc", "dec"])
def test_embed_phon_tokens_unchanged(baseline, rebuilt, side):
    _, example, model = rebuilt
    ids = recorded_field(example, "phon", f"{side}_input_ids")
    actual = model.embed_phon_tokens(ids)
    assert torch.allclose(actual, baseline["embed"][side], atol=FORWARD_ATOL)


@pytest.mark.parametrize("table", ["phonology_embedding", "phon_position_embedding"])
def test_gradients_unchanged(baseline, rebuilt, table):
    _, example, model = rebuilt
    model.zero_grad(set_to_none=True)
    model.embed_phon_tokens(example["target"].phon.ids).sum().backward()
    actual = getattr(model, table).weight.grad
    model.zero_grad(set_to_none=True)
    assert torch.allclose(actual, baseline["grads"][table], atol=GRAD_ATOL, rtol=GRAD_RTOL)


@pytest.mark.parametrize("pathway", ["o2p", "p2o", "p2p", "op2op"])
def test_forward_logits_unchanged(baseline, rebuilt, pathway):
    _, example, model = rebuilt
    with torch.no_grad():
        actual = run_forward(model, example, pathway)
    expected = baseline["logits"][pathway]
    assert set(actual) == set(expected), f"{pathway} output keys changed"
    for key, want in expected.items():
        assert torch.allclose(actual[key], want, atol=FORWARD_ATOL), f"{pathway}/{key}"


@pytest.mark.parametrize("pathway", ["o2p", "p2o", "p2p", "op2op", "o2o"])
def test_generation_unchanged(baseline, rebuilt, pathway):
    tokenizer, example, model = rebuilt
    # The words carry no language, so orthography starts unspecified, which is the prefix
    # the recorded run's encoding supplied.
    start = tokenizer.orth_start([None] * len(baseline["words"]))
    result = model.generate(example["input"], pathway, orth_start=start, deterministic=True)
    expected = baseline["generate"][pathway]

    assert torch.allclose(result.global_encoding, expected["global_encoding"], atol=FORWARD_ATOL), (
        f"{pathway}/global_encoding"
    )

    if "orth_tokens" in expected:
        assert result.orth_tokens is not None
        # Values superseded by #228; see this module's docstring. Shape is still pinned,
        # because a width change would mean something other than the seeding moved.
        assert result.orth_tokens.shape == expected["orth_tokens"].shape, (
            f"{pathway}/orth_tokens shape"
        )
        assert stack_nested(result.orth_probs).shape == expected["orth_probs"].shape, (
            f"{pathway}/orth_probs shape"
        )
    else:
        assert result.orth_tokens is None

    if "phon_tokens" in expected:
        assert result.phon_tokens is not None
        assert canon_features(result.phon_tokens, tokenizer) == expected["phon_tokens"], (
            f"{pathway}/phon_tokens"
        )
        assert torch.allclose(
            stack_nested(result.phon_probs), expected["phon_probs"], atol=FORWARD_ATOL
        ), f"{pathway}/phon_probs"
        assert torch.equal(stack_nested(result.phon_vecs), expected["phon_vecs"]), (
            f"{pathway}/phon_vecs"
        )
    else:
        assert result.phon_tokens is None
