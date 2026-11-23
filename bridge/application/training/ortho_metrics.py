from collections.abc import Sequence

import torch

from bridge.domain.datamodels import EncodingComponent


def calculate_orth_metrics(
    logits: dict[str, torch.Tensor],
    orthography: EncodingComponent,
    orth_pad_id: int,
    orth_language_ids: Sequence[int] | None,
) -> dict[str, float]:
    """Letter-wise and word-wise accuracy over the real (non-padded) positions, twice.

    ``letter_wise_accuracy`` and ``word_wise_accuracy`` count every scored token. A
    mid-sequence language token, the switch that marks a change of language, is one of
    them, so a missed switch costs a letter and fails its row. These are the figures for
    diagnosing the model.

    The ``_without_switches`` pair scores the same positions except those whose target is
    a language token, so it measures spelling alone. On a row with no switch the two pairs
    are equal. Telling a switch from a letter takes ``orth_language_ids``; with ``None``
    the pair is left out, since reporting it would mean guessing.
    """
    orth_pred = torch.argmax(logits["orth"], dim=1)
    # The encoder ids shifted left by one, which must stay identical to the slice
    # `TrainingPipeline.compute_loss` uses: a metric scoring different positions than the
    # loss trains reports on a model that was never optimized, which is issue #225. The
    # layout that makes the shift correct is documented once, at compute_loss, rather than
    # restated here. See docs/decisions/0004.
    orth_true = orthography.enc_input_ids[:, 1:]
    correct = orth_pred == orth_true
    scored = orth_true != orth_pad_id

    def accuracies(positions: torch.Tensor) -> tuple[float, float]:
        letter_wise = (correct & positions).sum().float() / positions.sum().float()
        # A word counts only when every one of its scored positions agrees.
        word_wise = (correct | ~positions).all(dim=1).float().mean()
        return letter_wise.item(), word_wise.item()

    letter_wise, word_wise = accuracies(scored)
    metrics = {"letter_wise_accuracy": letter_wise, "word_wise_accuracy": word_wise}
    if orth_language_ids is not None:
        # The shift drops the opening language token, so a language id still in the
        # target is a switch.
        switches = torch.isin(
            orth_true, torch.tensor(orth_language_ids, dtype=torch.long, device=orth_true.device)
        )
        letters_only, words_only = accuracies(scored & ~switches)
        metrics["letter_wise_accuracy_without_switches"] = letters_only
        metrics["word_wise_accuracy_without_switches"] = words_only
    return metrics
