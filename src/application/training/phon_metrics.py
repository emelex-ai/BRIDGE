import torch


def calculate_phon_word_accuracy(phon_true, phoneme_wise_mask):
    word_accuracies = [
        word[target != 2].all().int()
        for word, target in zip(phoneme_wise_mask, phon_true)
    ]
    phon_word_accuracy = sum(word_accuracies) / len(word_accuracies)
    return phon_word_accuracy

 
def calculate_phoneme_wise_accuracy(phon_true, masked_phon_true, phoneme_wise_mask):
    return phoneme_wise_mask.all(dim=-1).sum() / (
        masked_phon_true.shape[0] / phon_true.shape[-1]
    )


def calculate_phon_feature_accuracy(
    phon_valid_mask, masked_phon_true, masked_phon_pred
):
    correct_features = (masked_phon_pred == masked_phon_true).sum()
    phon_feature_accuracy = correct_features.float() / phon_valid_mask.sum().float()

    return phon_feature_accuracy

def calculate_euclidean_distance(
    phon_true: torch.Tensor, phon_pred: torch.Tensor
):
    distances = torch.Tensor()
    for i in range(phon_true.shape[0]):
        true = phon_true[i].type(torch.float)
        pred = phon_pred[i].type(torch.float)
        mask = true != 2
        true_masked = true[mask].reshape((-1,true.shape[1]))
        pred_masked = pred[mask].reshape((-1,true.shape[1]))
        f = torch.nn.PairwiseDistance()
        distances = torch.cat([distances, f(true_masked, pred_masked)])
    return torch.mean(distances)

def calculate_cosine_distance(
    phon_true: torch.Tensor, phon_pred: torch.Tensor
) -> torch.Tensor:
    cosine_sims = torch.Tensor()
    for i in range(phon_true.shape[0]):
        true = phon_true[i].type(torch.float)
        pred = phon_pred[i].type(torch.float)
        mask = true != 2
        true_masked = true[mask].reshape((-1,true.shape[1]))
        pred_masked = pred[mask].reshape((-1,true.shape[1]))
        f = torch.nn.CosineSimilarity(dim=1)
        cosine_sims = torch.cat([cosine_sims, f(pred_masked, true_masked)])
    return torch.mean(cosine_sims)

def calculate_phon_metrics(
    logits: dict[str, torch.Tensor], phonology: dict[str, torch.Tensor]
) -> dict[str, float]:
    phon_pred = torch.argmax(logits["phon"], dim=1)
    phon_true = phonology["targets"]
    phon_valid_mask = phon_true != 2
    masked_phon_true = phon_true[phon_valid_mask]
    masked_phon_pred = phon_pred[phon_valid_mask]
    cosine_accuracy = calculate_cosine_distance(phon_true, phon_pred)
    euclidean_distance = calculate_euclidean_distance(phon_true, phon_pred)
    phon_feature_accuracy = calculate_phon_feature_accuracy(
        phon_valid_mask, masked_phon_true, masked_phon_pred
    )
    phoneme_wise_mask = phon_pred == phon_true
    phoneme_wise_accuracy = calculate_phoneme_wise_accuracy(
        phon_true, masked_phon_true, phoneme_wise_mask
    )
    phon_word_accuracy = calculate_phon_word_accuracy(phon_true, phoneme_wise_mask)
    return {
        "phon_cosine_similarity": cosine_accuracy.item(),
        "phon_euclidean_distance": euclidean_distance.item(),
        "phon_feature_accuracy": phon_feature_accuracy.item(),
        "phon_phoneme_wise_accuracy": phoneme_wise_accuracy.item(),
        "phon_word_accuracy": phon_word_accuracy,
    }
