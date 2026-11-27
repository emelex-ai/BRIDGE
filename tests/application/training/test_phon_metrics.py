"""Pinned values for the phonological metrics, over two committed fixtures.

The fixtures are untouched and have never been regenerated. Each pinned value rests on
three choices. Each table below gives the value the other choice produces, so a number
that moves to it names its cause.

**The pad sentinel.** ``phon_true.pt`` is 54.2% padding, and a mask of ``!= 2`` cannot
see it: 2 is the orthographic pad id and phonological targets take values in {0, 1, 35}.
Under that mask every value is computed over padded positions as well as real ones. See
docs/decisions/0005.

    metric                    over padding    real positions only     ratio
    cosine similarity              0.5262                  0.9487      1.80
    euclidean distance           112.2570                  0.2339     0.002
    closest phoneme, L2            0.3942                  0.8609      2.18
    closest phoneme, cosine        0.4411                  0.9633      2.18

**The candidate phoneme set.** ``PHON_REPS`` is ``PhonemeTable.phonetic_features``,
``(93, 31)``, which is what ``TrainingPipeline`` passes and what
``test_phon_metrics_padding.py`` uses. The same matrix without the row of ``'_'``, the
featureless phoneme, is ``(92, 31)``, a candidate set the pipeline never uses.

    metric                  92 candidates    93 candidates (production)
    closest phoneme, L1            0.8609                        0.8635
    closest phoneme, L2            0.8609                        0.8635
    closest phoneme, cosine        0.9633                        0.9633  (as similarity)

One position of 381 scores differently, which is what an all-zero candidate row in an
L1/L2 search should do. Cosine is unaffected, and the two metrics that never
touch the matrix, cosine similarity and euclidean distance, are 0.9487 and 0.2339 under
both sets, which is what shows the candidate set is all that differs.

**The cosine direction.** Cosine is expressed as a distance, `1 - similarity`, so every
variant reduces with `argmin`. `argmin` over a cosine *similarity* selects the least
similar phoneme, so a metric named "closest phoneme" scores agreement on the *farthest*
one. Over the 93 real phoneme rows that reduction picks a row's own index 0 times out of
93, where `argmax` picks it 92 times, and on a fully scrambled prediction it reads 0.5914
where the L2 control correctly reads 0.0.

    metric                  as similarity (wrong)    as distance (correct)
    closest phoneme, cosine                0.9633                   0.8609

The correct figure sits beside L1 and L2 at 0.8635 rather than 10 points above them,
which is what a metric measuring the same thing under a different norm should do. A
`closest_phoneme_cosine_accuracy` computed as a similarity is void.

The identity cases are the control throughout. Comparing a tensor against itself is 1.0,
or 0.0 for a distance, at every position any mask could select, so those are the same
under every choice above.
"""

import math

import pytest
import torch

from bridge.application.training.phon_metrics import (
    calculate_closest_phoneme,
    calculate_cosine_distance,
    calculate_euclidean_distance,
    nearest_phoneme,
)
from tests.vocab import PHONEME_TABLE

# Feature space, where the phonological labels live. Not the orthographic 2.
PHON_PAD_ID = 35

# What TrainingPipeline passes: the real phonemes' feature vectors, special-token rows and
# columns removed. Built by PhonemeTable rather than re-derived from the CSV here.
PHON_REPS = PHONEME_TABLE.phonetic_features


def load_fixture(name):
    return torch.load(f"tests/application/training/data/{name}.pt", weights_only=True)


def real_rows(*tensors):
    """The non-padded positions of each tensor, as ``(positions, features)`` float rows.

    The preparation ``calculate_phon_metrics`` does before calling any of the helpers.
    Masking is driven by the *first* tensor, which is the target in every call below, so
    the rows stay paired across both arguments.
    """
    valid = tensors[0] != PHON_PAD_ID
    width = tensors[0].shape[-1]
    return [t.float()[valid].reshape(-1, width) for t in tensors]


def base_block(*rows):
    """Just the phonetic-feature columns, which is all ``phon_reps`` has to compare against."""
    return [r[:, : PHON_REPS.shape[1]] for r in rows]


def test_the_fixture_really_is_mostly_padding():
    """The premise every pinned value below depends on.

    If this drifts, the pinned numbers stop meaning what the docstring says they mean.
    """
    phon_true = load_fixture("phon_true")
    assert sorted(int(v) for v in phon_true.unique()) == [0, 1, PHON_PAD_ID]
    assert not (phon_true == 2).any(), "2 is not a sentinel in this tensor"
    padded = (phon_true == PHON_PAD_ID).float().mean().item()
    assert math.isclose(padded, 0.5421, rel_tol=1e-3)
    # Rows are padded all or nothing, which is what lets the elementwise masks reshape.
    rows = (phon_true == PHON_PAD_ID).all(-1).float().mean().item()
    assert math.isclose(rows, padded, rel_tol=1e-9)


def test_the_candidate_set_is_the_one_the_pipeline_passes():
    """Oracle: a differential against the table the pipeline reads.

    With a matrix built here one row shorter, every closest-phoneme number below
    describes a search the production path never runs. See the module docstring.
    """
    assert PHON_REPS.shape == (93, 31)
    assert PHON_REPS.shape[0] == PHONEME_TABLE.num_rows - 5, "one row per real phoneme"
    assert "_" in PHONEME_TABLE.phonemes, "the featureless phoneme is a candidate"


def test_cosine_distance_identity():
    phon_pred = load_fixture("phon_pred")
    (rows,) = real_rows(phon_pred)
    assert math.isclose(calculate_cosine_distance(rows, rows).item(), 1.0, rel_tol=1e-2)


def test_cosine_distance():
    true_rows, pred_rows = real_rows(load_fixture("phon_true"), load_fixture("phon_pred"))
    assert math.isclose(
        calculate_cosine_distance(true_rows, pred_rows).item(), 0.9487, rel_tol=1e-3
    )


def test_euclidean_distance_identity():
    phon_pred = load_fixture("phon_pred")
    (rows,) = real_rows(phon_pred)
    assert calculate_euclidean_distance(rows, rows).item() < 0.01


def test_euclidean_distance():
    true_rows, pred_rows = real_rows(load_fixture("phon_true"), load_fixture("phon_pred"))
    assert math.isclose(
        calculate_euclidean_distance(true_rows, pred_rows).item(), 0.2339, rel_tol=1e-3
    )


@pytest.mark.parametrize(
    ("metric", "expected"), [("l1", 0.8635), ("l2", 0.8635), ("cosine", 0.8609)]
)
def test_closest_phoneme(metric, expected):
    true_base, pred_base = base_block(
        *real_rows(load_fixture("phon_true"), load_fixture("phon_pred"))
    )
    assert math.isclose(
        calculate_closest_phoneme(true_base, pred_base, PHON_REPS, metric).item(),
        expected,
        rel_tol=1e-3,
    )


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
def test_closest_phoneme_identity(metric):
    (true_base,) = base_block(*real_rows(load_fixture("phon_true")))
    assert calculate_closest_phoneme(true_base, true_base, PHON_REPS, metric).item() == 1.0


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
def test_a_phoneme_is_its_own_nearest_phoneme(metric):
    """The analytic oracle, and the one the identity tests above cannot supply.

    Comparing a tensor against itself passes for *any* reduction, including ``argmin`` over
    a similarity, because both sides pick the same wrong row. Asking instead whether each
    phoneme's own feature vector resolves to its own index is what distinguishes nearest
    from farthest, and it is what ``argmin`` over a similarity fails 93 times out of 93.

    ``cosine`` is 92 of 93 rather than 93: ``phonreps.csv`` contains one featureless
    phoneme, ``'_'``, and an all-zero vector has no direction for cosine to compare.
    """
    own = torch.arange(PHON_REPS.shape[0])
    hits = int((nearest_phoneme(PHON_REPS, PHON_REPS, metric) == own).sum())
    assert hits == (92 if metric == "cosine" else 93), f"{metric} resolved {hits}/93"


@pytest.mark.parametrize("metric", ["l1", "l2", "cosine"])
def test_a_scrambled_prediction_scores_zero(metric):
    """The control that fails loudest on a reversed comparison.

    A prediction that is every phoneme except its own must agree nowhere. ``argmin`` over
    a cosine similarity scores 0.5914 here.

    The scramble is a shift by one row, which leaves no row in place whatever the row
    count. A seeded permutation can leave one: ``randperm(93)`` at seed 0 does.
    """
    scrambled = torch.roll(PHON_REPS, 1, 0)
    assert calculate_closest_phoneme(PHON_REPS, scrambled, PHON_REPS, metric).item() == 0.0
