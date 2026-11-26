from collections.abc import Sequence

import torch


def calculate_orth_metrics(
    logits: dict[str, torch.Tensor],
    labels: torch.Tensor,
    orth_pad_id: int,
    orth_language_ids: Sequence[int],
) -> dict[str, float]:
    """Letter-wise and word-wise accuracy over the real (non-padded) positions, twice.

    ``letter_wise_accuracy`` and ``word_wise_accuracy`` count every scored token. A
    mid-sequence language token, the switch that marks a change of language, is one of
    them, so a missed switch costs a letter and fails its row. These are the figures for
    diagnosing the model.

    The ``_without_switches`` pair scores the same positions except those whose label is
    a language token, so it measures spelling alone. On a row with no switch the two pairs
    are equal. ``orth_language_ids`` is what tells a switch from a letter: the ids of the
    language tokens.

    ``labels`` is the orthographic target's ``labels``, the tensor the loss is computed
    against. A metric scoring different positions than the loss trains reports on a model
    that was never optimized, which is issue #225, so both read the one tensor. See
    docs/decisions/0004.
    """
    orth_pred = torch.argmax(logits["orth"], dim=1)
    correct = orth_pred == labels
    scored = labels != orth_pad_id

    def accuracies(positions: torch.Tensor) -> tuple[float, float]:
        letter_wise = (correct & positions).sum().float() / positions.sum().float()
        # A word counts only when every one of its scored positions agrees.
        word_wise = (correct | ~positions).all(dim=1).float().mean()
        return letter_wise.item(), word_wise.item()

    letter_wise, word_wise = accuracies(scored)
    # The labels start after the opening language token, so a language id among them is
    # a switch.
    switches = torch.isin(
        labels, torch.tensor(orth_language_ids, dtype=torch.long, device=labels.device)
    )
    letters_only, words_only = accuracies(scored & ~switches)
    return {
        "letter_wise_accuracy": letter_wise,
        "word_wise_accuracy": word_wise,
        "letter_wise_accuracy_without_switches": letters_only,
        "word_wise_accuracy_without_switches": words_only,
    }
