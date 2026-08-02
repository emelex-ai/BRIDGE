"""Pins the semantics of ``Model.embed_phon_tokens``.

``embed_phon_tokens`` averages the embeddings of a phoneme's *active phonetic
features* and adds a position embedding. It currently does this with a Python double
loop; issue #221 proposes replacing that with a single padded gather, which is a large
rewrite of a hot path. These tests state the contract that rewrite must preserve, and
are deliberately written against observable behaviour rather than the implementation.
"""

import pytest
import torch

from bridge.domain.datamodels import ModelConfig, VocabSpec
from bridge.domain.model import Model

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
D_MODEL = 16


@pytest.fixture
def model():
    return Model(ModelConfig(vocab=VOCAB, d_model=D_MODEL, nhead=2, seed=11))


def test_output_shape_is_batch_by_seq_by_dmodel(model):
    tokens = [[torch.tensor([1, 2]), torch.tensor([3])] for _ in range(4)]
    assert model.embed_phon_tokens(tokens).shape == (4, 2, D_MODEL)


def test_each_position_is_the_mean_of_its_feature_embeddings(model):
    """The defining property: a phoneme's embedding is the mean over active features."""
    feats = torch.tensor([1, 5, 9])
    out = model.embed_phon_tokens([[feats]])

    expected = (
        model.phonology_embedding(feats).mean(dim=0) + model.phon_position_embedding.weight[0]
    )
    assert torch.allclose(out[0, 0], expected, atol=1e-6)


def test_single_feature_phoneme_is_just_that_embedding(model):
    feats = torch.tensor([7])
    out = model.embed_phon_tokens([[feats]])
    expected = model.phonology_embedding.weight[7] + model.phon_position_embedding.weight[0]
    assert torch.allclose(out[0, 0], expected, atol=1e-6)


def test_position_embedding_is_added_per_position(model):
    """Two identical phonemes at different positions differ only by position embedding."""
    feats = torch.tensor([4, 8])
    out = model.embed_phon_tokens([[feats, feats.clone()]])
    delta = out[0, 1] - out[0, 0]
    expected = model.phon_position_embedding.weight[1] - model.phon_position_embedding.weight[0]
    assert torch.allclose(delta, expected, atol=1e-6)


def test_variable_feature_counts_within_one_batch(model):
    """Feature counts are ragged across positions AND across batch items."""
    tokens = [
        [torch.tensor([1]), torch.tensor([2, 3, 4, 5])],
        [torch.tensor([6, 7, 8]), torch.tensor([9, 10])],
    ]
    out = model.embed_phon_tokens(tokens)
    assert out.shape == (2, 2, D_MODEL)
    for b, row in enumerate(tokens):
        for i, feats in enumerate(row):
            expected = (
                model.phonology_embedding(feats).mean(dim=0)
                + model.phon_position_embedding.weight[i]
            )
            assert torch.allclose(out[b, i], expected, atol=1e-6), f"batch {b} position {i}"


def test_batch_items_are_independent(model):
    """Reordering batch items reorders rows and changes nothing else."""
    a = [torch.tensor([1, 2]), torch.tensor([3])]
    b = [torch.tensor([9]), torch.tensor([4, 5, 6])]
    both = model.embed_phon_tokens([a, b])
    swapped = model.embed_phon_tokens([b, a])
    assert torch.allclose(both[0], swapped[1], atol=1e-6)
    assert torch.allclose(both[1], swapped[0], atol=1e-6)


def test_feature_order_within_a_phoneme_does_not_matter(model):
    """A mean is order-invariant; a padded-gather rewrite must stay so."""
    out_a = model.embed_phon_tokens([[torch.tensor([2, 5, 11])]])
    out_b = model.embed_phon_tokens([[torch.tensor([11, 2, 5])]])
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_gradients_reach_both_embedding_tables(model):
    """The rewrite must keep both tables in the autograd graph."""
    model.zero_grad(set_to_none=True)
    tokens = [[torch.tensor([1, 2]), torch.tensor([3])] for _ in range(2)]
    model.embed_phon_tokens(tokens).sum().backward()

    assert model.phonology_embedding.weight.grad is not None
    assert model.phon_position_embedding.weight.grad is not None
    # Only the rows for the features actually used receive gradient.
    used = torch.tensor([1, 2, 3])
    grad = model.phonology_embedding.weight.grad
    assert (grad[used] != 0).any()
    unused = torch.tensor([20, 21, 22])
    assert torch.equal(grad[unused], torch.zeros_like(grad[unused]))


def test_gradient_magnitude_reflects_the_mean_divisor(model):
    """A 2-feature phoneme contributes 1/2 per feature, a 1-feature phoneme 1/1.

    Pins the divisor, which a masked-sum rewrite could get wrong (e.g. dividing by
    the padded width instead of the true feature count).
    """
    model.zero_grad(set_to_none=True)
    model.embed_phon_tokens([[torch.tensor([1, 2])]]).sum().backward()
    two_feat = model.phonology_embedding.weight.grad[1].clone()

    model.zero_grad(set_to_none=True)
    model.embed_phon_tokens([[torch.tensor([1])]]).sum().backward()
    one_feat = model.phonology_embedding.weight.grad[1].clone()

    assert torch.allclose(two_feat * 2, one_feat, atol=1e-6)


def test_ragged_counts_do_not_leak_across_positions(model):
    """A long feature list next to a short one must not change the short one's value.

    This is the failure mode of a padded rewrite that forgets to mask: position 0
    would pick up padding rows from position 1's width.
    """
    alone = model.embed_phon_tokens([[torch.tensor([1])]])
    beside_long = model.embed_phon_tokens([[torch.tensor([1]), torch.tensor([2, 3, 4, 5, 6])]])
    assert torch.allclose(alone[0, 0], beside_long[0, 0], atol=1e-6)


def test_padding_feature_index_is_embedded_like_any_other(model):
    """[PAD] is a real vocabulary row here — placeholder components rely on it."""
    pad = torch.tensor([VOCAB.phon_pad_id])
    out = model.embed_phon_tokens([[pad]])
    expected = (
        model.phonology_embedding.weight[VOCAB.phon_pad_id]
        + model.phon_position_embedding.weight[0]
    )
    assert torch.allclose(out[0, 0], expected, atol=1e-6)


def test_matches_a_reference_padded_gather_implementation(model):
    """Equivalence with the vectorized form proposed in issue #221.

    If this fails after the rewrite, the rewrite changed the math.
    """
    tokens = [
        [torch.tensor([1]), torch.tensor([2, 3, 4]), torch.tensor([5, 6])],
        [torch.tensor([7, 8, 9, 10]), torch.tensor([11]), torch.tensor([12, 13])],
    ]

    batch, seq = len(tokens), len(tokens[0])
    width = max(len(t) for row in tokens for t in row)
    ids = torch.zeros((batch, seq, width), dtype=torch.long)
    mask = torch.zeros((batch, seq, width), dtype=torch.bool)
    for b, row in enumerate(tokens):
        for i, feats in enumerate(row):
            ids[b, i, : len(feats)] = feats
            mask[b, i, : len(feats)] = True

    embedded = model.phonology_embedding(ids) * mask.unsqueeze(-1)
    reference = embedded.sum(2) / mask.sum(-1, keepdim=True).clamp(min=1)
    reference = reference + model.phon_position_embedding.weight[None, :seq]

    assert torch.allclose(model.embed_phon_tokens(tokens), reference, atol=1e-6)
