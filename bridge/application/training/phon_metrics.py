import torch

from bridge.domain.datamodels import EncodingComponent


def calculate_phon_word_accuracy(phon_true, phoneme_wise_mask, phon_pad_id: int):
    word_accuracies = [
        word[target != phon_pad_id].all().int()
        for word, target in zip(phoneme_wise_mask, phon_true, strict=False)
    ]
    phon_word_accuracy = sum(word_accuracies) / len(word_accuracies)
    return phon_word_accuracy


def calculate_phoneme_wise_accuracy(phon_true, masked_phon_true, phoneme_wise_mask):
    return phoneme_wise_mask.all(dim=-1).sum() / (masked_phon_true.shape[0] / phon_true.shape[-1])


def calculate_phon_feature_accuracy(phon_valid_mask, masked_phon_true, masked_phon_pred):
    correct_features = (masked_phon_pred == masked_phon_true).sum()
    phon_feature_accuracy = correct_features.float() / phon_valid_mask.sum().float()

    return phon_feature_accuracy


def calculate_euclidean_distance(
    phon_true: torch.Tensor, phon_pred: torch.Tensor, phon_pad_id: int
) -> torch.Tensor:
    """
    Calculates euclidean distance between predicted and target phoneme
    """

    # For each phoneme metric calculation:
    # phon_true and phon_pred start as 3D tensors
    # The first dimension is the batch
    # The second is the phonemes in each word
    # And the third is the vector representing the phoneme
    true = phon_true.type(torch.float)
    pred = phon_pred.type(torch.float)
    mask = true != phon_pad_id
    # Filtering the true and pred tensors by mask results in a flattened 1D tensor.
    # After reshaping, the tensors are 2D:
    # Number of phonemes in the batch x phoneme vector
    true_masked = true[mask].reshape(-1, true.shape[-1])
    pred_masked = pred[mask].reshape(-1, true.shape[-1])
    distances = torch.nn.functional.pairwise_distance(true_masked, pred_masked)
    return torch.mean(distances)


def calculate_closest_phoneme_cdist(
    phon_true: torch.Tensor,
    phon_pred: torch.Tensor,
    phon_reps: torch.Tensor,
    phon_pad_id: int,
    norm=2,
):
    """
    Finds the p norm closest phoneme to the model's output, and compares this to the target phoneme
    """
    true = phon_true.type(torch.float)
    pred = phon_pred.type(torch.float)
    mask = true != phon_pad_id

    # The target tensor includes EOS, UNK, SPC and PAD  tokens, so we remove them (the last four indices)
    true_masked = true[mask].reshape(-1, true.shape[-1])[:, :-4]
    pred_masked = pred[mask].reshape(-1, true.shape[-1])[:, :-4]
    res = torch.cdist(pred_masked, phon_reps, norm)
    res2 = torch.cdist(true_masked, phon_reps, norm)
    return torch.mean(
        torch.eq(torch.argmin(res2, dim=1), torch.argmin(res, dim=1)).type(dtype=torch.float)
    )


def calculate_closest_phoneme_cosine(
    phon_true: torch.Tensor,
    phon_pred: torch.Tensor,
    phon_reps: torch.Tensor,
    phon_pad_id: int,
    eps=1e-8,
):
    """
    Finds the closest phoneme to the model's output using cosine distance, and compares this to the target phoneme
    """
    true = phon_true.type(torch.float)
    pred = phon_pred.type(torch.float)
    mask = true != phon_pad_id

    # The target tensor includes EOS, UNK, SPC and PAD  tokens, so we remove them (the last four indices)
    true_masked = true[mask].reshape(-1, true.shape[-1])[:, :-4]
    pred_masked = pred[mask].reshape(-1, true.shape[-1])[:, :-4]

    # Cosine similarity calculation
    phon_n = phon_reps.norm(p=2, dim=1, keepdim=True)
    pred_n = pred_masked.norm(p=2, dim=1, keepdim=True)
    true_n = true_masked.norm(p=2, dim=1, keepdim=True)
    true_norm = (true_n * phon_n.t()).clamp(min=eps)
    pred_norm = (pred_n * phon_n.t()).clamp(min=eps)
    res = torch.mm(pred_masked, phon_reps.t()) / pred_norm
    res2 = torch.mm(true_masked, phon_reps.t()) / true_norm
    # Find rate at which the closest predicted phoneme equals the target phoneme
    return torch.mean(
        torch.eq(torch.argmin(res2, dim=1), torch.argmin(res, dim=1)).type(dtype=torch.float)
    )


def calculate_cosine_distance(
    phon_true: torch.Tensor, phon_pred: torch.Tensor, phon_pad_id: int
) -> torch.Tensor:
    """
    Calculates euclidean distance between predicted and target phoneme
    """
    true = phon_true.type(torch.float)
    pred = phon_pred.type(torch.float)
    mask = true != phon_pad_id
    true_masked = true[mask].view(-1, true.size(-1))
    pred_masked = pred[mask].view(-1, pred.size(-1))
    f = torch.nn.CosineSimilarity(dim=1)
    cosine_sims = f(pred_masked, true_masked)
    return torch.mean(cosine_sims)


def calculate_phon_metrics(
    logits: dict[str, torch.Tensor],
    phonology: EncodingComponent,
    phon_reps: torch.Tensor,
    phon_pad_id: int,
) -> dict[str, float]:
    """Score the phonological predictions, ignoring padded positions.

    ``phon_pad_id`` is in *feature* space: a column of the phoneme feature table, which is
    where ``phon_targets`` lives. It is 35, not the orthographic pad id of 2. Passing the
    wrong one produces a mask that selects everything rather than an error, so the argument
    is required and has no default. See
    docs/decisions/0005-phonological-metrics-take-the-pad-id.md.

    Each mask is elementwise while the reshape that follows assumes whole rows survive it.
    That holds because a padded target row is entirely ``phon_pad_id`` and a real one holds
    only 0 and 1, so rows are selected all or nothing. A target encoding that broke the
    property would raise from the reshape rather than return a wrong number.
    """
    phon_pred = torch.argmax(logits["phon"], dim=1)
    phon_true = phonology.phon_targets

    phon_valid_mask = phon_true != phon_pad_id
    masked_phon_true = phon_true[phon_valid_mask]
    masked_phon_pred = phon_pred[phon_valid_mask]
    cosine_accuracy = calculate_cosine_distance(phon_true, phon_pred, phon_pad_id)
    euclidean_distance = calculate_euclidean_distance(phon_true, phon_pred, phon_pad_id)
    phon_feature_accuracy = calculate_phon_feature_accuracy(
        phon_valid_mask, masked_phon_true, masked_phon_pred
    )
    phoneme_wise_mask = phon_pred == phon_true
    phoneme_wise_accuracy = calculate_phoneme_wise_accuracy(
        phon_true, masked_phon_true, phoneme_wise_mask
    )
    phon_word_accuracy = calculate_phon_word_accuracy(phon_true, phoneme_wise_mask, phon_pad_id)
    closest_phoneme = calculate_closest_phoneme_cdist(
        phon_true, phon_pred, phon_reps, phon_pad_id, 1
    )
    closest_phoneme_2 = calculate_closest_phoneme_cdist(
        phon_true, phon_pred, phon_reps, phon_pad_id, 2
    )
    closest_phoneme_cosine = calculate_closest_phoneme_cosine(
        phon_true, phon_pred, phon_reps, phon_pad_id
    )
    return {
        "phon_cosine_similarity": cosine_accuracy.item(),
        "phon_euclidean_distance": euclidean_distance.item(),
        "phon_feature_accuracy": phon_feature_accuracy.item(),
        "phon_phoneme_wise_accuracy": phoneme_wise_accuracy.item(),
        "phon_word_accuracy": phon_word_accuracy,
        "closest_phoneme_l1_accuracy": closest_phoneme.item(),
        "closest_phoneme_l2_accuracy": closest_phoneme_2.item(),
        "closest_phoneme_cosine_accuracy": closest_phoneme_cosine.item(),
    }
