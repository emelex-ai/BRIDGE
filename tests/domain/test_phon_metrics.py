from src.application.training.phon_metrics import (
    calculate_closest_phoneme_cdist,
    calculate_closest_phoneme_cosine,
    calculate_cosine_distance,
    calculate_euclidean_distance,
    calculate_phon_reps_distance,
)
import pytest
import math
import torch
from traindata import utilities


@pytest.fixture(scope="session")
def phon_pred():
    """Loads the phon_pred object once per test session."""
    return torch.load("tests/domain/model/data/phon_pred.pt", weights_only=True)


@pytest.fixture(scope="session")
def phon_true():
    """Loads the phon_true object once per test session."""
    return torch.load("tests/domain/model/data/phon_true.pt", weights_only=True)


@pytest.fixture(scope="session")
def phon_reps():
    """
    Loads the phoneme representations from CSV once per test session.
    The '[:-1]' slice is preserved from your original code.
    """
    data = utilities.phontable("data/phonreps.csv").values
    return torch.tensor(data, dtype=torch.float)[:-1]


def test_cosine_distance_identity(phon_pred):
    assert torch.sum(phon_pred - phon_pred) == 0
    assert calculate_cosine_distance(phon_pred, phon_pred).item() == 1.0


def test_cosine_distance(phon_pred, phon_true):
    assert math.isclose(
        calculate_cosine_distance(phon_true, phon_pred).item(), 0.242, rel_tol=1e-2
    )


def test_euclidean_distance_identity(phon_pred):
    assert torch.sum(phon_pred - phon_pred) == 0
    assert calculate_euclidean_distance(phon_pred, phon_pred).item() < 0.01


def test_euclidean_distance(phon_pred, phon_true):
    assert math.isclose(
        calculate_euclidean_distance(phon_true, phon_pred).item(), 4.048, rel_tol=1e-2
    )


def test_closest_phoneme_cdist(phon_pred, phon_true, phon_reps):
    assert math.isclose(
        calculate_closest_phoneme_cdist(phon_true, phon_pred, phon_reps).item(),
        0.00813,
        rel_tol=1e-5,
    )


def test_closest_phoneme_cdist_identity(phon_true, phon_reps):
    resp = calculate_closest_phoneme_cdist(phon_true, phon_true, phon_reps)
    assert resp == 1.0


def test_closest_phoneme_cosine(phon_pred, phon_true, phon_reps):
    assert math.isclose(
        calculate_closest_phoneme_cosine(phon_true, phon_pred, phon_reps).item(),
        0.186,
        rel_tol=1e-2,
    )


def test_closest_phoneme_cosine_identity(phon_true, phon_reps):
    resp = calculate_closest_phoneme_cosine(phon_true, phon_true, phon_reps)
    assert resp == 1.0
