"""The semantics of the phonological embedding path.

A phoneme's embedding is the *mean of its active phonetic features'* embeddings and
nothing else: it carries no position, which is read inside self-attention. Two entry
points share that operation (issue #221):

* :meth:`Model.embed_phon_tokens` takes ``(B, L)`` phoneme row ids, used for training and
  teacher forcing. Gathers from a per-phoneme table precomputed once per call.
* :meth:`Model.embed_phon_vectors` takes ``(B, L, V)`` binary feature vectors, used during
  generation, where a sampled vector need not be any real phoneme.

Arbitrary feature sets are expressed through ``embed_phon_vectors`` because the row table
is fixed by ``phonreps.csv`` and cannot be made to contain a hand-picked feature
combination.
"""

import pytest
import torch

from bridge.domain.datamodels import ModelConfig
from bridge.domain.model import Model
from tests.vocab import PHONEME_TABLE as TABLE
from tests.vocab import TEST_VOCAB as VOCAB

V = TABLE.vocab_size
D_MODEL = 16


@pytest.fixture
def model():
    return Model(ModelConfig(vocab=VOCAB, d_model=D_MODEL, nhead=2, seed=11))


def multihot(*feature_sets: list[int], batch: int = 1) -> torch.Tensor:
    """(batch, len(feature_sets), V) binary vectors from explicit feature index lists."""
    out = torch.zeros((batch, len(feature_sets), V))
    for position, features in enumerate(feature_sets):
        out[:, position, features] = 1.0
    return out


def rows_with_feature_counts() -> dict[int, int]:
    """One representative table row per distinct active-feature count."""
    counts = TABLE.multihot.sum(-1)
    return {int(c): int((counts == c).nonzero()[0]) for c in counts.unique()}


def test_output_shape_is_batch_by_seq_by_dmodel(model):
    rows = torch.zeros((4, 2), dtype=torch.long)
    assert model.embed_phon_tokens(rows).shape == (4, 2, D_MODEL)
    assert model.embed_phon_vectors(multihot([1, 2], [3], batch=4)).shape == (4, 2, D_MODEL)


def test_each_position_is_the_mean_of_its_feature_embeddings(model):
    feats = [1, 5, 9]
    out = model.embed_phon_vectors(multihot(feats))

    expected = model.phonology_embedding(torch.tensor(feats)).mean(dim=0)
    assert torch.allclose(out[0, 0], expected, atol=1e-6)


def test_row_ids_embed_as_the_mean_of_that_phonemes_features(model):
    # One phoneme per batch item.
    rows = torch.arange(TABLE.num_rows).unsqueeze(1)
    out = model.embed_phon_tokens(rows)

    for row in range(TABLE.num_rows):
        features = TABLE.features_of(row)
        if len(features) == 0:
            continue  # covered by test_featureless_phoneme_embeds_to_zero
        expected = model.phonology_embedding(features).mean(dim=0)
        assert torch.allclose(out[row, 0], expected, atol=1e-6), f"row {row}"


def test_single_feature_phoneme_is_just_that_embedding(model):
    out = model.embed_phon_vectors(multihot([7]))
    assert torch.allclose(out[0, 0], model.phonology_embedding.weight[7], atol=1e-6)


def test_the_two_entry_points_agree(model):
    rows = torch.tensor([[3, 17, TABLE.row_index["[EOS]"]], [42, 8, TABLE.row_index["[PAD]"]]])
    dense = TABLE.multihot[rows]
    assert torch.allclose(model.embed_phon_tokens(rows), model.embed_phon_vectors(dense), atol=1e-6)


def test_a_phoneme_embeds_the_same_at_every_position(model):
    """Oracle: analytic. The embedding is a function of the phoneme alone, so one phoneme
    at 200 positions gives 200 equal vectors, bitwise, through both entry points. The
    control is a second phoneme, which must differ from the first."""
    rows = torch.full((1, 200), 5)
    by_row = model.embed_phon_tokens(rows)
    assert torch.equal(by_row, by_row[:, :1].expand_as(by_row))

    by_vector = model.embed_phon_vectors(multihot(*[[4, 8]] * 200))
    assert torch.equal(by_vector, by_vector[:, :1].expand_as(by_vector))

    assert not torch.equal(model.embed_phon_tokens(torch.tensor([[6]]))[0, 0], by_row[0, 0])


def test_variable_feature_counts_within_one_batch(model):
    dense = torch.zeros((2, 2, V))
    sets = [[[1], [2, 3, 4, 5]], [[6, 7, 8], [9, 10]]]
    for b, row in enumerate(sets):
        for i, features in enumerate(row):
            dense[b, i, features] = 1.0

    out = model.embed_phon_vectors(dense)
    assert out.shape == (2, 2, D_MODEL)
    for b, row in enumerate(sets):
        for i, features in enumerate(row):
            expected = model.phonology_embedding(torch.tensor(features)).mean(dim=0)
            assert torch.allclose(out[b, i], expected, atol=1e-6), f"batch {b} position {i}"


def test_batch_items_are_independent(model):
    rows = torch.tensor([[1, 2], [9, 4]])
    both = model.embed_phon_tokens(rows)
    swapped = model.embed_phon_tokens(rows.flip(0))
    assert torch.allclose(both[0], swapped[1], atol=1e-6)
    assert torch.allclose(both[1], swapped[0], atol=1e-6)


def test_ragged_counts_do_not_leak_across_positions(model):
    """A long feature list next to a short one must not change the short one's value.

    The failure mode of a padded rewrite that forgets to mask: position 0 would pick up
    padding rows from position 1's width.
    """
    alone = model.embed_phon_vectors(multihot([1]))
    beside_long = model.embed_phon_vectors(multihot([1], [2, 3, 4, 5, 6]))
    assert torch.allclose(alone[0, 0], beside_long[0, 0], atol=1e-6)


def test_feature_order_within_a_phoneme_does_not_matter(model):
    """A mean is order-invariant, and a set representation must stay so."""
    assert torch.allclose(
        model.embed_phon_vectors(multihot([2, 5, 11])),
        model.embed_phon_vectors(multihot([11, 2, 5])),
        atol=1e-6,
    )


def test_padding_row_is_embedded_like_any_other(model):
    """[PAD] is a real vocabulary row here, and every padded position relies on it."""
    pad_row = TABLE.row_index["[PAD]"]
    out = model.embed_phon_tokens(torch.tensor([[pad_row]]))
    assert torch.allclose(out[0, 0], model.phonology_embedding.weight[VOCAB.phon_pad_id], atol=1e-6)


def test_featureless_phoneme_embeds_to_zero_not_nan(model):
    """phonreps.csv contains one phoneme with no active features ('_').

    A ``mean()`` over its empty index tensor is NaN; the table path returns the zero
    vector, deliberately.
    """
    counts = TABLE.multihot.sum(-1)
    empty_rows = (counts == 0).nonzero().flatten()
    assert len(empty_rows) > 0, "expected a featureless phoneme in phonreps.csv"

    out = model.embed_phon_tokens(empty_rows[:1].unsqueeze(0))
    assert not out.isnan().any()
    assert torch.equal(out[0, 0], torch.zeros(D_MODEL))


def test_all_zero_feature_vector_embeds_to_zero_not_nan(model):
    out = model.embed_phon_vectors(torch.zeros((1, 1, V)))
    assert not out.isnan().any()
    assert torch.equal(out[0, 0], torch.zeros(D_MODEL))


def test_gradients_reach_the_embedding_table(model):
    """The embedding table is in the autograd graph, and only its used rows get gradient."""
    model.zero_grad(set_to_none=True)
    model.embed_phon_vectors(multihot([1, 2], [3], batch=2)).sum().backward()

    assert model.phonology_embedding.weight.grad is not None
    used = torch.tensor([1, 2, 3])
    grad = model.phonology_embedding.weight.grad
    assert (grad[used] != 0).any()
    unused = torch.tensor([20, 21, 22])
    assert torch.equal(grad[unused], torch.zeros_like(grad[unused]))


def test_gradients_reach_the_embedding_table_through_row_ids(model):
    """The table path routes gradient back through the feature matrix, not around it."""
    model.zero_grad(set_to_none=True)
    row = TABLE.row_index["[EOS]"]
    model.embed_phon_tokens(torch.tensor([[row]])).sum().backward()

    grad = model.phonology_embedding.weight.grad
    assert grad is not None
    assert (grad[VOCAB.phon_eos_id] != 0).any(), "[EOS]'s own feature row got no gradient"


def test_gradient_magnitude_reflects_the_mean_divisor(model):
    """A 2-feature phoneme contributes 1/2 per feature, a 1-feature phoneme 1/1.

    Pins the divisor, which a masked-sum rewrite could get wrong (e.g. dividing by the
    padded width instead of the true feature count).
    """
    model.zero_grad(set_to_none=True)
    model.embed_phon_vectors(multihot([1, 2])).sum().backward()
    two_feat = model.phonology_embedding.weight.grad[1].clone()

    model.zero_grad(set_to_none=True)
    model.embed_phon_vectors(multihot([1])).sum().backward()
    one_feat = model.phonology_embedding.weight.grad[1].clone()

    assert torch.allclose(two_feat * 2, one_feat, atol=1e-6)


def test_row_path_gradient_magnitude_reflects_the_mean_divisor(model):
    """Same divisor guarantee through the table path, across real phoneme rows."""
    by_count = rows_with_feature_counts()
    counts = sorted(c for c in by_count if c > 0)
    assert len(counts) >= 2, "need phonemes with differing feature counts"

    magnitudes = {}
    for count in counts:
        row = by_count[count]
        feature = int(TABLE.features_of(row)[0])
        model.zero_grad(set_to_none=True)
        model.embed_phon_tokens(torch.tensor([[row]])).sum().backward()
        magnitudes[count] = model.phonology_embedding.weight.grad[feature].abs().sum().item()

    # Gradient per feature scales as 1/count, so count * magnitude is constant.
    scaled = [count * magnitudes[count] for count in counts]
    assert max(scaled) - min(scaled) < 1e-5, f"divisor is not the feature count: {magnitudes}"
