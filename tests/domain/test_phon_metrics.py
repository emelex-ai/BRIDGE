from src.application.training.phon_metrics import calculate_cosine_distance, calculate_euclidean_distance
import torch

def test_cosine_distance():
    phon_pred = torch.load("tests/data/phon_pred.pt")
    phon_true = torch.load("tests/data/phon_pred.pt")
    print(phon_pred - phon_true)
    assert torch.sum(phon_pred-phon_true) == 0
    assert calculate_cosine_distance(phon_true, phon_pred).item() == 1.0
    
def test_euclidean_distance():
    phon_pred = torch.load("tests/data/phon_pred.pt")
    phon_true = torch.load("tests/data/phon_pred.pt")
    assert torch.sum(phon_pred-phon_true) == 0
    assert calculate_euclidean_distance(phon_true, phon_pred).item() < 0.01