"""End-to-end generation against a real Model.

Every pathway, exercised through the public :meth:`Model.generate` entry point. The
assertions are structural: which fields a pathway populates, what shapes they carry, and
that a deterministic run repeats. The numeric values are pinned separately by
``tests/domain/test_phon_baseline_equivalence.py`` and the per-step mechanics by
``tests/domain/model/test_decoder_loops.py``.
"""

import pytest
import torch

from bridge.domain.datamodels import GenerationOutput, ModelConfig
from bridge.domain.model import Model
from bridge.domain.tokenizer import BridgeTokenizer
from tests.vocab import TEST_VOCAB

TOKENIZER = BridgeTokenizer()
WORDS = ["cat", "dog", "elephant"]

# Which GenerationOutput fields each pathway is responsible for filling in.
ORTH_FIELDS = ("orth_tokens", "orth_probs")
PHON_FIELDS = ("phon_tokens", "phon_probs", "phon_vecs")
PRODUCES = {
    "o2p": PHON_FIELDS,
    "p2o": ORTH_FIELDS,
    "p2p": PHON_FIELDS,
    "o2o": ORTH_FIELDS,
    "op2op": ORTH_FIELDS + PHON_FIELDS,
}


@pytest.fixture(scope="module")
def model():
    model = Model(ModelConfig(vocab=TEST_VOCAB, d_model=32, nhead=2, seed=11))
    model.eval()
    return model


@pytest.fixture(scope="module")
def encoding():
    encoding = TOKENIZER.encode(WORDS)
    assert encoding is not None
    return encoding


@pytest.mark.parametrize("pathway", sorted(PRODUCES))
def test_each_pathway_fills_exactly_its_own_fields(model, encoding, pathway):
    output = model.generate(encoding, pathway, deterministic=True)

    assert isinstance(output, GenerationOutput)
    assert output.global_encoding.shape[0] == len(WORDS)
    for field in PRODUCES[pathway]:
        assert getattr(output, field) is not None, f"{pathway} should produce {field}"
    for field in set(ORTH_FIELDS + PHON_FIELDS) - set(PRODUCES[pathway]):
        assert getattr(output, field) is None, f"{pathway} should not produce {field}"


@pytest.mark.parametrize("pathway", sorted(PRODUCES))
def test_generated_batches_keep_their_alignment(model, encoding, pathway):
    """One entry per input word, in every ragged field a pathway produces."""
    output = model.generate(encoding, pathway, deterministic=True)

    for field in PRODUCES[pathway]:
        value = getattr(output, field)
        length = value.shape[0] if isinstance(value, torch.Tensor) else len(value)
        assert length == len(WORDS), f"{pathway}/{field}"


def same(a, b) -> bool:
    """Compare nested generation output, whose inner tensors are genuinely ragged."""
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("pathway", sorted(PRODUCES))
def test_deterministic_generation_repeats(model, encoding, pathway):
    first = model.generate(encoding, pathway, deterministic=True)
    second = model.generate(encoding, pathway, deterministic=True)

    for field in PRODUCES[pathway]:
        assert same(getattr(first, field), getattr(second, field)), f"{pathway}/{field}"


def test_an_unknown_pathway_is_rejected(model, encoding):
    with pytest.raises(ValueError, match="Invalid pathway"):
        model.generate(encoding, "nonsense", deterministic=True)
