"""Scoring for the phonological predictions.

Every metric here reads the same thing: the real (non-padded) phoneme positions of a
batch, as float rows. :func:`calculate_phon_metrics` builds that once and passes it down.
"""

import torch


def calculate_phon_word_accuracy(
    phon_true: torch.Tensor, phoneme_wise_mask: torch.Tensor, phon_pad_id: int
) -> torch.Tensor:
    """Fraction of words whose every real phoneme position is predicted exactly right.

    A padded position cannot make a word wrong, so it is forced true before the reduction
    and not indexed out. That makes this one reduction over the batch: 0.053 ms at batch
    64, against 1.185 ms for a per-row loop that returns the same values at six corruption
    levels from accuracy 1.0 down to 0.0.
    """
    real = phon_true != phon_pad_id
    return (phoneme_wise_mask | ~real).all(dim=2).all(dim=1).float().mean()


def calculate_phoneme_wise_accuracy(
    phon_true: torch.Tensor, masked_phon_true: torch.Tensor, phoneme_wise_mask: torch.Tensor
) -> torch.Tensor:
    """Fraction of positions whose whole feature vector is right, over the real positions."""
    return phoneme_wise_mask.all(dim=-1).sum() / (masked_phon_true.shape[0] / phon_true.shape[-1])


def calculate_phon_feature_accuracy(
    phon_valid_mask: torch.Tensor, masked_phon_true: torch.Tensor, masked_phon_pred: torch.Tensor
) -> torch.Tensor:
    """Fraction of individual features predicted right, over the real positions."""
    correct_features = (masked_phon_pred == masked_phon_true).sum()
    return correct_features.float() / phon_valid_mask.sum().float()


def calculate_euclidean_distance(true_rows: torch.Tensor, pred_rows: torch.Tensor) -> torch.Tensor:
    """Mean euclidean distance between predicted and target feature vectors.

    ``true_rows`` and ``pred_rows`` are ``(positions, features)``: one row per real
    phoneme in the batch, padding already removed.
    """
    return torch.mean(torch.nn.functional.pairwise_distance(true_rows, pred_rows))


def calculate_cosine_distance(true_rows: torch.Tensor, pred_rows: torch.Tensor) -> torch.Tensor:
    """Mean cosine similarity between predicted and target feature vectors."""
    return torch.mean(torch.nn.CosineSimilarity(dim=1)(pred_rows, true_rows))


def nearest_phoneme(rows: torch.Tensor, phon_reps: torch.Tensor, metric: str) -> torch.Tensor:
    """Index of the closest real phoneme to each row of ``rows``, under ``metric``.

    ``rows`` and ``phon_reps`` both carry only the phonetic-feature columns.

    Every metric is computed as a *distance*, so the reduction is always ``argmin``.
    Cosine is ``1 - similarity`` for that reason: ``argmin`` over a similarity selects the
    least similar phoneme. Over the 93 real phoneme rows it picks a row's own index 0
    times, where ``argmax`` picks it 92 times.
    """
    if metric == "cosine":
        eps = 1e-8
        a = rows / rows.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
        b = phon_reps / phon_reps.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
        distances = 1.0 - torch.mm(a, b.t())
    else:
        distances = torch.cdist(rows, phon_reps, {"l1": 1.0, "l2": 2.0}[metric])
    return distances.argmin(dim=1)


def calculate_closest_phoneme(
    true_base: torch.Tensor,
    pred_base: torch.Tensor,
    phon_reps: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """Rate at which the prediction's nearest real phoneme is also the target's.

    Not "is the prediction correct" but "does it round to the same phoneme", which is the
    tolerant reading: a feature vector can be wrong in a bit or two and still decode to the
    intended phoneme.
    """
    return (
        torch.eq(
            nearest_phoneme(true_base, phon_reps, metric),
            nearest_phoneme(pred_base, phon_reps, metric),
        )
        .float()
        .mean()
    )


def calculate_phon_metrics(
    logits: dict[str, torch.Tensor],
    labels: torch.Tensor,
    phon_reps: torch.Tensor,
    phon_pad_id: int,
) -> dict[str, float]:
    """Score the phonological predictions, ignoring padded positions.

    ``labels`` is the phonological target's ``labels``. ``phon_pad_id`` is in *feature*
    space: a column of the phoneme feature table, which is where the labels live. It is
    35, not the orthographic pad id of 2. Passing the wrong one produces a mask that
    selects everything rather than an error, so the argument is required and has no
    default. See docs/decisions/0005-phonological-metrics-take-the-pad-id.md.

    Each mask is elementwise while the reshape that follows assumes whole rows survive it.
    That holds because a padded label row is entirely ``phon_pad_id`` and a real one holds
    only 0 and 1, so rows are selected all or nothing. A target encoding that broke the
    property would raise from the reshape rather than return a wrong number.
    """
    phon_pred = torch.argmax(logits["phon"], dim=1)

    phon_valid_mask = labels != phon_pad_id
    masked_labels = labels[phon_valid_mask]
    masked_phon_pred = phon_pred[phon_valid_mask]
    phoneme_wise_mask = phon_pred == labels

    # One row per real phoneme in the batch, cast once for every distance metric below.
    width = labels.shape[-1]
    true_rows = labels.float()[phon_valid_mask].reshape(-1, width)
    pred_rows = phon_pred.float()[phon_valid_mask].reshape(-1, width)
    # The phonetic-feature block alone, without the special-token columns. The width comes
    # off `phon_reps`, which is `PhonemeTable.phonetic_features`, and not from a literal:
    # the labels have already lost the [PAD] column, so the count of special columns
    # here is one less than `SPECIAL_TOKENS` holds.
    base = phon_reps.shape[1]
    true_base, pred_base = true_rows[:, :base], pred_rows[:, :base]

    return {
        "phon_cosine_similarity": calculate_cosine_distance(true_rows, pred_rows).item(),
        "phon_euclidean_distance": calculate_euclidean_distance(true_rows, pred_rows).item(),
        "phon_feature_accuracy": calculate_phon_feature_accuracy(
            phon_valid_mask, masked_labels, masked_phon_pred
        ).item(),
        "phon_phoneme_wise_accuracy": calculate_phoneme_wise_accuracy(
            labels, masked_labels, phoneme_wise_mask
        ).item(),
        "phon_word_accuracy": calculate_phon_word_accuracy(
            labels, phoneme_wise_mask, phon_pad_id
        ).item(),
        **{
            f"closest_phoneme_{metric}_accuracy": calculate_closest_phoneme(
                true_base, pred_base, phon_reps, metric
            ).item()
            for metric in ("l1", "l2", "cosine")
        },
    }
