from src.application.training.phon_metrics import (
    calculate_closest_phoneme_cdist,
    calculate_closest_phoneme_cosine,
    calculate_cosine_distance,
    calculate_euclidean_distance,
    calculate_phon_reps_distance,
)
import math
import torch
from traindata import utilities


def test_cosine_distance_identity():
    phon_pred = torch.load("tests/application/training/data/phon_pred.pt", weights_only=True)
    assert torch.sum(phon_pred - phon_pred) == 0
    assert calculate_cosine_distance(phon_pred, phon_pred).item() == 1.0


def test_cosine_distance():
    phon_pred = torch.load("tests/application/training/data/phon_pred.pt", weights_only=True)
    phon_true = torch.load("tests/application/training/data/phon_true.pt", weights_only=True)
    assert math.isclose(
        calculate_cosine_distance(phon_true, phon_pred).item(), 0.242, rel_tol=1e-2
    )


def test_euclidean_distance_identity():
    phon_pred = torch.load("tests/application/training/data/phon_pred.pt", weights_only=True)
    assert torch.sum(phon_pred - phon_pred) == 0
    assert calculate_euclidean_distance(phon_pred, phon_pred).item() < 0.01


def test_euclidean_distance():
    phon_pred = torch.load("tests/application/training/data/phon_pred.pt", weights_only=True)
    phon_true = torch.load("tests/application/training/data/phon_true.pt", weights_only=True)
    assert math.isclose(
        calculate_euclidean_distance(phon_true, phon_pred).item(), 4.048, rel_tol=1e-2
    )


def test_closest_phoneme_cdist():
    phon_reps = torch.tensor(
        utilities.phontable("data/phonreps.csv").values, dtype=torch.float
    )[:-1]
    phon_pred = torch.load("tests/application/training/data/phon_pred.pt", weights_only=True)
    phon_true = torch.load("tests/application/training/data/phon_true.pt", weights_only=True)
    assert math.isclose(
        calculate_closest_phoneme_cdist(phon_true, phon_pred, phon_reps).item(),
        0.00813,
        rel_tol=1e-5,
    )


def test_closest_phoneme_cdist_identity():
    phon_reps = torch.tensor(
        utilities.phontable("data/phonreps.csv").values, dtype=torch.float
    )[:-1]
    phon_true = torch.load("tests/application/training/data/phon_true.pt", weights_only=True)
    resp = calculate_closest_phoneme_cdist(phon_true, phon_true, phon_reps)
    assert resp == 1.0


def test_closest_phoneme_cosine():
    phon_reps = torch.tensor(
        utilities.phontable("data/phonreps.csv").values, dtype=torch.float
    )[:-1]
    phon_pred = torch.load("tests/application/training/data/phon_pred.pt", weights_only=True)
    phon_true = torch.load("tests/application/training/data/phon_true.pt", weights_only=True)
    assert math.isclose(
        calculate_closest_phoneme_cosine(phon_true, phon_pred, phon_reps).item(),
        0.186,
        rel_tol=1e-2,
    )


def test_closest_phoneme_cosine_identity():
    phon_reps = torch.tensor(
        utilities.phontable("data/phonreps.csv").values, dtype=torch.float
    )[:-1]
    phon_true = torch.load("tests/application/training/data/phon_true.pt", weights_only=True)
    resp = calculate_closest_phoneme_cosine(phon_true, phon_true, phon_reps) == 1.0
    assert resp == 1.0
