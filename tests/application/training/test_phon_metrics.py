from src.application.training.phon_metrics import (
    calculate_closest_phoneme_cdist,
    calculate_closest_phoneme_cosine,
    calculate_cosine_distance,
    calculate_euclidean_distance,
)
from src.utils.helper_functions import get_project_root
import math
import torch
import os
import pandas as pd


def test_cosine_distance_identity():
    phon_pred = torch.load(
        "tests/application/training/data/phon_pred.pt", weights_only=True
    )
    assert torch.sum(phon_pred - phon_pred) == 0
    assert calculate_cosine_distance(phon_pred, phon_pred).item() == 1.0


def test_cosine_distance():
    phon_pred = torch.load(
        "tests/application/training/data/phon_pred.pt", weights_only=True
    )
    phon_true = torch.load(
        "tests/application/training/data/phon_true.pt", weights_only=True
    )
    assert math.isclose(
        calculate_cosine_distance(phon_true, phon_pred).item(), 0.2453, rel_tol=1e-2
    )


def test_euclidean_distance_identity():
    phon_pred = torch.load(
        "tests/application/training/data/phon_pred.pt", weights_only=True
    )
    assert torch.sum(phon_pred - phon_pred) == 0
    assert calculate_euclidean_distance(phon_pred, phon_pred).item() < 0.01


def test_euclidean_distance():
    phon_pred = torch.load(
        "tests/application/training/data/phon_pred.pt", weights_only=True
    )
    phon_true = torch.load(
        "tests/application/training/data/phon_true.pt", weights_only=True
    )
    assert math.isclose(
        calculate_euclidean_distance(phon_true, phon_pred).item(), 98.0208, rel_tol=1e-2
    )


def test_closest_phoneme_cdist():
    phonreps = pd.read_csv(os.path.join(get_project_root(), "data/phonreps.csv"))
    phonreps.set_index("phone", inplace=True)
    phon_reps = torch.tensor(phonreps.values, dtype=torch.float)[:-1]
    phon_pred = torch.load(
        "tests/application/training/data/phon_pred.pt", weights_only=True
    )
    phon_true = torch.load(
        "tests/application/training/data/phon_true.pt", weights_only=True
    )
    assert math.isclose(
        calculate_closest_phoneme_cdist(phon_true, phon_pred, phon_reps).item(),
        0.0016025,
        rel_tol=1e-2,
    )


def test_closest_phoneme_cdist_identity():
    phonreps = pd.read_csv(os.path.join(get_project_root(), "data/phonreps.csv"))
    phonreps.set_index("phone", inplace=True)
    phon_reps = torch.tensor(phonreps.values, dtype=torch.float)[:-1]
    phon_true = torch.load(
        "tests/application/training/data/phon_true.pt", weights_only=True
    )
    resp = calculate_closest_phoneme_cdist(phon_true, phon_true, phon_reps)
    assert resp == 1.0


def test_closest_phoneme_cosine():
    phonreps = pd.read_csv(os.path.join(get_project_root(), "data/phonreps.csv"))
    phonreps.set_index("phone", inplace=True)
    phon_reps = torch.tensor(phonreps.values, dtype=torch.float)[:-1]
    phon_pred = torch.load(
        "tests/application/training/data/phon_pred.pt", weights_only=True
    )
    phon_true = torch.load(
        "tests/application/training/data/phon_true.pt", weights_only=True
    )
    print(calculate_closest_phoneme_cosine(phon_true, phon_pred, phon_reps).item())
    assert math.isclose(
        calculate_closest_phoneme_cosine(phon_true, phon_pred, phon_reps).item(),
        0.7532,
        rel_tol=1e-2,
    )


def test_closest_phoneme_cosine_identity():
    phonreps = pd.read_csv(os.path.join(get_project_root(), "data/phonreps.csv"))
    phonreps.set_index("phone", inplace=True)
    phon_reps = torch.tensor(phonreps.values, dtype=torch.float)[:-1]
    phon_true = torch.load(
        "tests/application/training/data/phon_true.pt", weights_only=True
    )
    resp = calculate_closest_phoneme_cosine(phon_true, phon_true, phon_reps) == 1.0
    assert resp == 1.0
