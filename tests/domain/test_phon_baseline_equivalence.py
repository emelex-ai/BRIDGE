"""Equivalence against the pre-refactor behavioural fingerprint (issue #221).

``tests/fixtures/phon_baseline.pt`` was recorded on the commit before the phonological
refactor began. This module rebuilds the same objects and asserts every recorded value.

**The fixture is immutable.** If a value here changes, either the refactor changed behaviour
or the change was intentional and the spec says so. Regenerating the fixture to make a test
pass defeats its entire purpose.

**The plumbing is not.** The refactor changes the phonological representation from a ragged
``list[list[Tensor]]`` to a ``(batch, sequence)`` tensor of row ids, and the adapters in the
ADAPTERS block below absorb that; everything under ASSERTIONS compares recorded numbers and
should not need to change. Two signature changes the spec sketched were not made:
``phon_targets`` stayed a property on ``EncodingComponent`` rather than moving onto ``Model``,
and ``forward`` still takes loose tensor kwargs rather than a ``BridgeEncoding``.

Runs on CPU: ``device_manager`` defaults to CPU, and CUDA reduction order differs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bridge.domain.datamodels import ModelConfig, VocabSpec
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


def phon_targets_of(model: Model, phon):
    """Phonological loss targets."""
    return phon.phon_targets


def run_forward(model: Model, encoding, pathway: str) -> dict[str, torch.Tensor]:
    """Forward pass, whichever calling convention the model currently exposes."""
    orth, phon = encoding.orthographic, encoding.phonological
    parts = {
        "orth_enc": {
            "orth_enc_input": orth.enc_input_ids,
            "orth_enc_pad_mask": orth.enc_pad_mask,
        },
        "orth_dec": {
            "orth_dec_input": orth.dec_input_ids,
            "orth_dec_pad_mask": orth.dec_pad_mask,
        },
        "phon_enc": {
            "phon_enc_input": phon.enc_input_ids,
            "phon_enc_pad_mask": phon.enc_pad_mask,
        },
        "phon_dec": {
            "phon_dec_input": phon.dec_input_ids,
            "phon_dec_pad_mask": phon.dec_pad_mask,
        },
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
    encoding = tokenizer.encode(baseline["words"])
    assert encoding is not None, "baseline words no longer encode"
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
    return tokenizer, encoding, model


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
    _, encoding, _ = rebuilt
    actual = getattr(encoding.orthographic, field.removeprefix("orth_"))
    assert torch.equal(actual, baseline["tok"][field])


@pytest.mark.parametrize("field", ["phon_enc_pad_mask", "phon_dec_pad_mask"])
def test_phonological_pad_masks_unchanged(baseline, rebuilt, field):
    _, encoding, _ = rebuilt
    actual = getattr(encoding.phonological, field.removeprefix("phon_"))
    assert torch.equal(actual, baseline["tok"][field])


@pytest.mark.parametrize("side", ["enc", "dec"])
def test_phonological_feature_sets_unchanged(baseline, rebuilt, side):
    """The load-bearing one: same features per position, whatever the storage."""
    tokenizer, encoding, _ = rebuilt
    actual = canon_features(getattr(encoding.phonological, f"{side}_input_ids"), tokenizer)
    assert actual == baseline["tok"][f"phon_{side}_features"]


def test_phon_targets_unchanged(baseline, rebuilt):
    _, encoding, model = rebuilt
    actual = phon_targets_of(model, encoding.phonological)
    assert torch.equal(actual, baseline["tok"]["phon_targets"])


def test_placeholder_phonological_unchanged(baseline, rebuilt):
    tokenizer, _, model = rebuilt
    placeholder = tokenizer.encode(baseline["words"], modality_filter="orthography")
    assert placeholder is not None
    ph = placeholder.phonological
    expected = baseline["placeholder_phon"]
    assert canon_features(ph.enc_input_ids, tokenizer) == expected["enc_features"]
    assert canon_features(ph.dec_input_ids, tokenizer) == expected["dec_features"]
    assert torch.equal(ph.enc_pad_mask, expected["enc_pad_mask"])
    assert torch.equal(ph.dec_pad_mask, expected["dec_pad_mask"])

    # Deliberate divergence from the baseline. The recorded targets are (B, 1, 36) of
    # zeros, one column wider than the (B, L, 35) every real encoding produces and filled
    # with a scored value rather than the loss ignore_index, so a placeholder could not be
    # used where a real component could. Both defects predate the refactor; the fixture
    # simply recorded them. The placeholder is now the [PAD] row of the same target table.
    targets = phon_targets_of(model, ph)
    assert targets.shape[-1] == expected["targets"].shape[-1] - 1
    real = tokenizer.encode(baseline["words"]).phonological
    assert targets.shape[-1] == real.targets.shape[-1]
    assert bool((targets == tokenizer.phon_pad_id).all())


@pytest.mark.parametrize("side", ["enc", "dec"])
def test_embed_phon_tokens_unchanged(baseline, rebuilt, side):
    _, encoding, model = rebuilt
    ids = getattr(encoding.phonological, f"{side}_input_ids")
    actual = model.embed_phon_tokens(ids)
    assert torch.allclose(actual, baseline["embed"][side], atol=FORWARD_ATOL)


@pytest.mark.parametrize("table", ["phonology_embedding", "phon_position_embedding"])
def test_gradients_unchanged(baseline, rebuilt, table):
    _, encoding, model = rebuilt
    model.zero_grad(set_to_none=True)
    model.embed_phon_tokens(encoding.phonological.dec_input_ids).sum().backward()
    actual = getattr(model, table).weight.grad
    model.zero_grad(set_to_none=True)
    assert torch.allclose(actual, baseline["grads"][table], atol=GRAD_ATOL, rtol=GRAD_RTOL)


@pytest.mark.parametrize("pathway", ["o2p", "p2o", "p2p", "op2op"])
def test_forward_logits_unchanged(baseline, rebuilt, pathway):
    _, encoding, model = rebuilt
    with torch.no_grad():
        actual = run_forward(model, encoding, pathway)
    expected = baseline["logits"][pathway]
    assert set(actual) == set(expected), f"{pathway} output keys changed"
    for key, want in expected.items():
        assert torch.allclose(actual[key], want, atol=FORWARD_ATOL), f"{pathway}/{key}"


@pytest.mark.parametrize("pathway", ["o2p", "p2o", "p2p", "op2op", "o2o"])
def test_generation_unchanged(baseline, rebuilt, pathway):
    tokenizer, encoding, model = rebuilt
    result = model.generate(encoding, pathway, deterministic=True)
    expected = baseline["generate"][pathway]

    assert torch.allclose(result.global_encoding, expected["global_encoding"], atol=FORWARD_ATOL), (
        f"{pathway}/global_encoding"
    )

    if "orth_tokens" in expected:
        assert result.orth_tokens is not None
        assert torch.equal(result.orth_tokens, expected["orth_tokens"]), f"{pathway}/orth_tokens"
        assert torch.allclose(
            stack_nested(result.orth_probs), expected["orth_probs"], atol=FORWARD_ATOL
        ), f"{pathway}/orth_probs"
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
