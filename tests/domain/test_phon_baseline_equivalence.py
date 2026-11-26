"""Asserts what the golden master ``tests/fixtures/phon_baseline.pt`` still says of a model.

The fixture is a behavioural fingerprint of the phonological path, recorded on the commit
before the phonological refactor (issue #221). This module rebuilds the same objects and
asserts every recorded value that still describes them.

**The fixture is immutable.** Regenerating it to make a test pass destroys the only thing
it is for. Three groups of recorded values are superseded. Each is handled by reducing its
assertions here to what still holds, and the fixture is not touched.

**Orthographic generation.** Generation seeds the orthographic decoder with the
``[LANG, BOS]`` prefix training uses (issue #228, ``docs/decisions/0008``). The recording
seeded it with a lone ``[BOS]`` at position 0, so the recorded ``orth_tokens`` and
``orth_probs`` describe behaviour that was wrong. Measured at the commit that changed it,
the effect is bounded to exactly that:

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
throughout, ``(9, 30)`` and ``(9, 30, 109)``. The orthographic assertions below are reduced
to shape, and the seeded behaviour is pinned against analytic oracles in
``tests/domain/model/test_orth_generation_prefix.py``, not against a re-recording.

**The stand-in phonology.** The fixture records ``placeholder_phon``, the stand-in an
orthography-only encoding carried: one position per row, all of it padding. An input with
no phonology carries ``None`` there (``docs/decisions/0016``), so there is no tensor to
compare. The reduced assertion checks what the stand-in was for: an input with orthography
alone generates on a pathway that reads orthography alone, bitwise what an input carrying
both modalities generates there. Measured against the commit before that change,
generation from such an input is bitwise equal to generation from the stand-in encoding on
``o2p`` and ``o2o``, 729 of 729 recorded values. The behaviour of a missing modality is
pinned against ``PATHWAY_IO`` in ``tests/domain/model/test_generate.py``.

**Every value that passes through the model's weights.** Position is a rotation inside
self-attention, computed for any length, and the model holds no position table
(``docs/decisions/0017``). The recording was made by a model with two such tables of 30
rows. Without them a seed's draws fall differently across the parameters that remain, so
no recorded embedding, logit, encoding or generated token can be produced again. Measured
2026-11-23 against ``a3cb285``, the last commit with the tables, on CPU: with that commit's
tables zeroed and the rotation switched off, teacher-forced logits of the two models are
bitwise equal on all five pathways, 36 of 36 outputs over six models, so position is all
that moved.

The assertions on those values are reduced to what the recording still says about the
model: which outputs a pathway returns, their shapes and dtypes, which fields a generation
fills, that ids stay inside their vocabulary, and that a generated phonological sequence
opens with the recorded ``[BOS]``. Two kinds of recorded value are asserted exactly. One
is every tokenizer and vocabulary value. The other is the gradient of an embedded batch
with respect to the phonological embedding, which counts features and depends on no
weight. The recorded gradient of the position table has no parameter to be compared with;
its test checks that the recording is the count of rows per slot, and that the
phonological embedding is the only parameter an embedding reaches.

What the model computes is pinned against reference and analytic oracles in
``tests/domain/model/test_rotary_layers.py``, ``tests/domain/model/test_any_length.py`` and
``tests/domain/model/test_max_new_tokens.py``, not against a re-recording.

**Two differences of form are bridged, not asserted.** The fixture holds phonology as a
ragged ``list[list[Tensor]]`` of feature indices and the tokenizer returns a
``(batch, sequence)`` tensor of row ids; ``canon_features`` reduces both to one form. The
fixture names its tensors by the encoder and decoder that read them. The encoder's are the
ids and mask of the input, the decoder's are the ids and mask of the target, and the
recorded targets are the target's labels; ``FIELD_OF`` below is that mapping. The words
are encoded once as an input and once as a target, which is repetition, the task the
fixture recorded.

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

# Gradients diverge further than the forward pass under a reduction-order change: measured
# worst case is 3.8e-6 absolute, which clears the default rtol=1e-5 but not atol alone.
GRAD_ATOL, GRAD_RTOL = 1e-6, 1e-5


def canon_features(ids, tokenizer) -> list[list[tuple[int, ...]]]:
    """Per-position sorted feature-index tuples, from either form.

    The fixture holds a ragged ``list[list[Tensor]]`` of feature indices. The tokenizer and
    the model give a ``(B, L)`` tensor of phoneme row ids.
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
    return example["target"].phon.labels


def run_forward(model: Model, example: dict, pathway: str) -> dict[str, torch.Tensor]:
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


def recorded_cap(baseline: dict) -> int:
    """The most tokens a recorded generation wrote after its start.

    Every recorded orthographic row is 30 wide, the two tokens of ``[LANG, BOS]`` and 28
    more. A generation here is given the same 28, so it can be no wider than the recording.
    """
    widths = {
        recorded["orth_tokens"].shape[1]
        for recorded in baseline["generate"].values()
        if "orth_tokens" in recorded
    }
    assert widths == {30}, widths
    return 28


def assert_same_form(actual: torch.Tensor, recorded: torch.Tensor, name: str) -> None:
    assert actual.shape == recorded.shape, f"{name} shape"
    assert actual.dtype == recorded.dtype, f"{name} dtype"
    assert bool(torch.isfinite(actual).all()), f"{name} holds a non-finite value"


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
    # The recording holds no language ids and no count of memory rows. The ids are the
    # tokenizer's. The count is one: every recorded ``global_encoding`` has one row.
    assert {g["global_encoding"].shape[1] for g in baseline["generate"].values()} == {1}
    model = Model(
        ModelConfig(
            vocab=VocabSpec(
                **{k: v for k, v in baseline["vocab"].items() if k in VocabSpec.model_fields},
                orth_language_ids=tokenizer.orth_language_ids,
            ),
            d_model=baseline["d_model"],
            nhead=baseline["nhead"],
            d_embedding=1,
            seed=baseline["seed"],
        )
    )
    model.eval()
    return tokenizer, example, model


# Recorded fields that VocabSpec deliberately does not carry: nothing read them.
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


def test_an_input_without_phonology_generates_on_the_pathway_the_stand_in_served(baseline, rebuilt):
    """Superseded in part; see this module's docstring.

    The recorded stand-in is checked to be what the docstring says it was, all padding,
    so nothing it held is being dropped. Then the reason it existed is asserted: an input
    carrying orthography alone generates on ``o2p``. Oracle: invariant. ``o2p`` reads no
    phonology, so that input generates bitwise what the input carrying both generates,
    and it fills the fields the recording fills, in the recorded form.
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

    cap = recorded_cap(baseline)
    result = model.generate(orthography_only, "o2p", max_new_tokens=cap, deterministic=True)
    both = model.generate(example["input"], "o2p", max_new_tokens=cap, deterministic=True)
    expected = baseline["generate"]["o2p"]

    assert_same_form(result.global_encoding, expected["global_encoding"], "global_encoding")
    assert torch.equal(result.global_encoding, both.global_encoding)
    assert canon_features(result.phon_tokens, tokenizer) == canon_features(
        both.phon_tokens, tokenizer
    )
    for field in ("phon_probs", "phon_vecs"):
        ours, theirs = getattr(result, field), getattr(both, field)
        assert [len(row) for row in ours] == [len(row) for row in theirs], field
        assert all(
            torch.equal(a, b)
            for row, other in zip(ours, theirs, strict=True)
            for a, b in zip(row, other, strict=True)
        ), field
    assert result.orth_tokens is None and "orth_tokens" not in expected


@pytest.mark.parametrize("side", ["enc", "dec"])
def test_embed_phon_tokens_has_the_recorded_form(baseline, rebuilt, side):
    """Superseded in part; see this module's docstring. The recorded values held a
    position vector and another draw of the weights. The shape and dtype still hold."""
    _, example, model = rebuilt
    ids = recorded_field(example, "phon", f"{side}_input_ids")
    assert_same_form(model.embed_phon_tokens(ids), baseline["embed"][side], f"embed/{side}")


def test_gradients_unchanged(baseline, rebuilt):
    """Compared with the recording at the tolerance it has always had, ``GRAD_ATOL`` and
    ``GRAD_RTOL``. The gradient of an embedded batch's sum with respect to the
    phonological embedding counts how often each feature is active, divided by its
    phoneme's feature count. No weight enters it and no position does."""
    _, example, model = rebuilt
    model.zero_grad(set_to_none=True)
    model.embed_phon_tokens(example["target"].phon.ids).sum().backward()
    actual = model.phonology_embedding.weight.grad
    model.zero_grad(set_to_none=True)
    assert torch.allclose(
        actual, baseline["grads"]["phonology_embedding"], atol=GRAD_ATOL, rtol=GRAD_RTOL
    )


def test_the_recorded_position_gradient_has_no_parameter_left_to_reach(baseline, rebuilt):
    """Superseded; see this module's docstring.

    The recording holds the gradient of the same sum with respect to a table of 30
    position embeddings. It is checked to be what that would be, the number of rows in
    every slot the batch reaches and zero in the rest, so nothing else it held is being
    dropped. The model has no such table, and what holds in its place is asserted: the
    phonological embedding is the only parameter an embedding reaches.
    """
    _, example, model = rebuilt
    ids = example["target"].phon.ids
    rows, width = ids.shape
    recorded = baseline["grads"]["phon_position_embedding"]
    assert recorded.shape == (30, baseline["d_model"])
    assert torch.equal(recorded[:width], torch.full_like(recorded[:width], rows))
    assert torch.equal(recorded[width:], torch.zeros_like(recorded[width:]))

    model.zero_grad(set_to_none=True)
    model.embed_phon_tokens(ids).sum().backward()
    reached = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
    model.zero_grad(set_to_none=True)
    assert reached == ["phonology_embedding.weight"]


@pytest.mark.parametrize("pathway", ["o2p", "p2o", "p2p", "op2op"])
def test_forward_logits_have_the_recorded_form(baseline, rebuilt, pathway):
    """Superseded in part; see this module's docstring. The values moved with the weights.
    Which outputs the pathway returns, and the shape and dtype of each, still hold."""
    _, example, model = rebuilt
    with torch.no_grad():
        actual = run_forward(model, example, pathway)
    expected = baseline["logits"][pathway]
    assert set(actual) == set(expected), f"{pathway} output keys changed"
    for key, want in expected.items():
        assert_same_form(actual[key], want, f"{pathway}/{key}")


@pytest.mark.parametrize("pathway", ["o2p", "p2o", "p2p", "op2op", "o2o"])
def test_generation_fills_the_recorded_fields(baseline, rebuilt, pathway):
    """Superseded in part; see this module's docstring. The generated values moved with
    the weights. What still holds of each recorded field is asserted: that the pathway
    fills it and no other, its dtype, its row count and vector widths, that every id lies
    in its vocabulary, and that every sequence has the recorded length.

    The lengths are equal because the two models end their rows alike, which is a fact
    about these weights and not a rule. Given the recorded cap, no orthographic row of
    either ends before it, and every phonological row of either ends on its first step.
    Measured 2026-11-24 on CPU."""
    tokenizer, example, model = rebuilt
    words = len(baseline["words"])
    cap = recorded_cap(baseline)
    # The words carry no language, so orthography starts unspecified.
    start = tokenizer.orth_start([None] * words)
    result = model.generate(
        example["input"], pathway, max_new_tokens=cap, orth_start=start, deterministic=True
    )
    expected = baseline["generate"][pathway]
    vocab = baseline["vocab"]

    assert_same_form(result.global_encoding, expected["global_encoding"], "global_encoding")

    if "orth_tokens" in expected:
        assert result.orth_tokens is not None
        recorded_tokens, recorded_probs = expected["orth_tokens"], expected["orth_probs"]
        assert result.orth_tokens.dtype == recorded_tokens.dtype
        assert result.orth_tokens.shape == recorded_tokens.shape == (words, 2 + cap)
        assert torch.equal(result.orth_tokens[:, :2], start)
        assert int(result.orth_tokens.min()) >= 0
        assert int(result.orth_tokens.max()) < vocab["orth_vocab_size"]
        assert len(result.orth_probs) == words
        for row in result.orth_probs:
            assert len(row) == recorded_probs.shape[1]
            assert {step.shape for step in row} == {recorded_probs.shape[2:]}
            assert {step.dtype for step in row} == {recorded_probs.dtype}
    else:
        assert result.orth_tokens is None and result.orth_probs is None

    if "phon_tokens" in expected:
        assert result.phon_tokens is not None
        tokens = canon_features(result.phon_tokens, tokenizer)
        assert len(tokens) == len(result.phon_probs) == len(result.phon_vecs) == words
        for row, recorded_row in zip(tokens, expected["phon_tokens"], strict=True):
            assert row[0] == recorded_row[0] == (vocab["phon_bos_id"],)
            assert len(row) == len(recorded_row)
            assert all(0 <= feature < vocab["phon_vocab_size"] for p in row for feature in p)
        for field in ("phon_probs", "phon_vecs"):
            recorded = expected[field]
            for row, tokens_row in zip(getattr(result, field), tokens, strict=True):
                assert len(row) == len(tokens_row) - 1, f"{pathway}/{field} length"
                assert {step.shape for step in row} == {recorded.shape[2:]}, f"{pathway}/{field}"
                assert {step.dtype for step in row} == {recorded.dtype}, f"{pathway}/{field}"
        vecs = torch.stack([step for row in result.phon_vecs for step in row])
        assert bool(((vecs == 0) | (vecs == 1)).all()), f"{pathway}/phon_vecs is not binary"
    else:
        assert result.phon_tokens is None
        assert result.phon_probs is None and result.phon_vecs is None
