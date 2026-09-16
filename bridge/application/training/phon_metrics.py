"""Scoring for the phonological predictions.

Every metric here reads the same thing: the real (non-padded) phoneme positions of a
batch, as float rows. That preamble is built once, in :func:`calculate_phon_metrics`, and
passed down. It used to be rebuilt inside each helper, four times identically and a fifth
with ``.view`` instead of ``.reshape``, which is ~1.3 ms of a 7.7 ms call at batch 64 and
is how the ``[:, :-4]`` literal below came to be written twice with a comment naming the
wrong columns.
"""

import torch

from bridge.domain.datamodels import EncodingComponent


def calculate_phon_word_accuracy(
    phon_true: torch.Tensor, phoneme_wise_mask: torch.Tensor, phon_pad_id: int
) -> torch.Tensor:
    """Fraction of words whose every real phoneme position is predicted exactly right.

    A padded position cannot make a word wrong, so it is forced true before the reduction
    rather than indexed out. That is what lets this be one reduction over the batch
    instead of a Python loop doing a boolean index and two reductions per row, which cost
    1.185 ms against 0.053 ms at batch 64 and was verified identical against the loop at
    six corruption levels spanning accuracy 1.0 down to 0.0.
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


def calculate_closest_phoneme_cdist(
    true_base: torch.Tensor,
    pred_base: torch.Tensor,
    phon_reps: torch.Tensor,
    norm: float = 2,
) -> torch.Tensor:
    """Rate at which the p-norm-closest real phoneme to the prediction is the target's.

    ``true_base`` and ``pred_base`` carry only the phonetic-feature columns, since
    ``phon_reps`` (``PhonemeTable.phonetic_features``) has no special-token columns to
    compare against.
    """
    return torch.mean(
        torch.eq(
            torch.argmin(torch.cdist(true_base, phon_reps, norm), dim=1),
            torch.argmin(torch.cdist(pred_base, phon_reps, norm), dim=1),
        ).float()
    )


def calculate_closest_phoneme_cosine(
    true_base: torch.Tensor,
    pred_base: torch.Tensor,
    phon_reps: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """The same comparison as :func:`calculate_closest_phoneme_cdist`, under cosine distance."""
    phon_n = phon_reps.norm(p=2, dim=1, keepdim=True)
    pred_norm = (pred_base.norm(p=2, dim=1, keepdim=True) * phon_n.t()).clamp(min=eps)
    true_norm = (true_base.norm(p=2, dim=1, keepdim=True) * phon_n.t()).clamp(min=eps)
    return torch.mean(
        torch.eq(
            torch.argmin(torch.mm(true_base, phon_reps.t()) / true_norm, dim=1),
            torch.argmin(torch.mm(pred_base, phon_reps.t()) / pred_norm, dim=1),
        ).float()
    )


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
    phoneme_wise_mask = phon_pred == phon_true

    # One row per real phoneme in the batch, cast once for every distance metric below.
    width = phon_true.shape[-1]
    true_rows = phon_true.float()[phon_valid_mask].reshape(-1, width)
    pred_rows = phon_pred.float()[phon_valid_mask].reshape(-1, width)
    # The phonetic-feature block alone, without the special-token columns. The width comes
    # off `phon_reps`, which is `PhonemeTable.phonetic_features`, rather than being written
    # as the literal `[:, :-4]`: that number was correct only by coincidence, its comment
    # named PAD (already dropped by the tokenizer) and omitted BOS, and a sixth special
    # token in phonreps.csv would have shifted it silently.
    base = phon_reps.shape[1]
    true_base, pred_base = true_rows[:, :base], pred_rows[:, :base]

    return {
        "phon_cosine_similarity": calculate_cosine_distance(true_rows, pred_rows).item(),
        "phon_euclidean_distance": calculate_euclidean_distance(true_rows, pred_rows).item(),
        "phon_feature_accuracy": calculate_phon_feature_accuracy(
            phon_valid_mask, masked_phon_true, masked_phon_pred
        ).item(),
        "phon_phoneme_wise_accuracy": calculate_phoneme_wise_accuracy(
            phon_true, masked_phon_true, phoneme_wise_mask
        ).item(),
        "phon_word_accuracy": calculate_phon_word_accuracy(
            phon_true, phoneme_wise_mask, phon_pad_id
        ).item(),
        "closest_phoneme_l1_accuracy": calculate_closest_phoneme_cdist(
            true_base, pred_base, phon_reps, 1
        ).item(),
        "closest_phoneme_l2_accuracy": calculate_closest_phoneme_cdist(
            true_base, pred_base, phon_reps, 2
        ).item(),
        "closest_phoneme_cosine_accuracy": calculate_closest_phoneme_cosine(
            true_base, pred_base, phon_reps
        ).item(),
    }
