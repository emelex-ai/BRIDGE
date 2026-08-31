"""Pinned values for the phonological metrics, over two committed fixtures.

The numbers here moved when the pad sentinel was corrected, and the size of the move is
the point rather than an inconvenience. ``phon_true.pt`` is 54.2% padding, and the old
mask, ``!= 2``, could never see it: 2 is the orthographic pad id and phonological targets
take values in {0, 1, 35}. Every value below was therefore computed over padded positions
as well as real ones.

The fixtures are untouched. Only the expectations changed, and both readings are recorded
so the move is auditable rather than a silently regenerated baseline:

    metric                    over padding    real positions only     ratio
    cosine similarity              0.5262                  0.9487      1.80
    euclidean distance           112.2570                  0.2339     0.002
    closest phoneme, L2            0.3894                  0.8504      2.18
    closest phoneme, cosine        0.4483                  0.9738      2.17

The identity cases are the control. Comparing a tensor against itself is 1.0, or 0.0 for a
distance, at every position any mask could select, so those four must not move at all, and
they do not. That is what shows the substitution touched the sentinel and nothing else.

See docs/decisions/0005-phonological-metrics-take-the-pad-id.md.
"""

import math
import os

import pandas as pd
import torch

from bridge.application.training.phon_metrics import (
    calculate_closest_phoneme_cdist,
    calculate_closest_phoneme_cosine,
    calculate_cosine_distance,
    calculate_euclidean_distance,
)
from bridge.utils import get_project_root

# Feature space, where phon_targets lives. Not the orthographic 2.
PHON_PAD_ID = 35


def load_fixture(name):
    return torch.load(f"tests/application/training/data/{name}.pt", weights_only=True)


def load_phon_reps():
    phonreps = pd.read_csv(os.path.join(get_project_root(), "bridge/core/phonreps.csv"))
    phonreps.set_index("phone", inplace=True)
    return torch.tensor(phonreps.values, dtype=torch.float)[:-1]


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


def test_cosine_distance_identity():
    phon_pred = load_fixture("phon_pred")
    assert torch.sum(phon_pred - phon_pred) == 0
    assert math.isclose(
        calculate_cosine_distance(phon_pred, phon_pred, PHON_PAD_ID).item(), 1.0, rel_tol=1e-2
    )


def test_cosine_distance():
    phon_pred = load_fixture("phon_pred")
    phon_true = load_fixture("phon_true")
    assert math.isclose(
        calculate_cosine_distance(phon_true, phon_pred, PHON_PAD_ID).item(), 0.9487, rel_tol=1e-3
    )


def test_euclidean_distance_identity():
    phon_pred = load_fixture("phon_pred")
    assert torch.sum(phon_pred - phon_pred) == 0
    assert calculate_euclidean_distance(phon_pred, phon_pred, PHON_PAD_ID).item() < 0.01


def test_euclidean_distance():
    phon_pred = load_fixture("phon_pred")
    phon_true = load_fixture("phon_true")
    assert math.isclose(
        calculate_euclidean_distance(phon_true, phon_pred, PHON_PAD_ID).item(), 0.2339, rel_tol=1e-3
    )


def test_closest_phoneme_cdist():
    phon_reps = load_phon_reps()
    phon_pred = load_fixture("phon_pred")
    phon_true = load_fixture("phon_true")
    assert math.isclose(
        calculate_closest_phoneme_cdist(phon_true, phon_pred, phon_reps, PHON_PAD_ID).item(),
        0.8504,
        rel_tol=1e-3,
    )


def test_closest_phoneme_cdist_identity():
    phon_reps = load_phon_reps()
    phon_true = load_fixture("phon_true")
    resp = calculate_closest_phoneme_cdist(phon_true, phon_true, phon_reps, PHON_PAD_ID)
    assert resp == 1.0


def test_closest_phoneme_cosine():
    phon_reps = load_phon_reps()
    phon_pred = load_fixture("phon_pred")
    phon_true = load_fixture("phon_true")
    assert math.isclose(
        calculate_closest_phoneme_cosine(phon_true, phon_pred, phon_reps, PHON_PAD_ID).item(),
        0.9738,
        rel_tol=1e-3,
    )


def test_closest_phoneme_cosine_identity():
    phon_reps = load_phon_reps()
    phon_true = load_fixture("phon_true")
    resp = calculate_closest_phoneme_cosine(phon_true, phon_true, phon_reps, PHON_PAD_ID) == 1.0
    assert resp == 1.0
