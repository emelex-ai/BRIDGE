import torch

from bridge.domain.datamodels import EncodingComponent


def calculate_orth_metrics(
    logits: dict[str, torch.Tensor],
    orthography: EncodingComponent,
    orth_pad_id: int,
) -> dict[str, float]:
    """Letter-wise and word-wise accuracy over the real (non-padded) positions.

    Both were single-call three-line helpers, and one of them wrote into ``orth_pred`` in
    place, so their order at the call site was load-bearing and invisible. Inlined, the
    write happens where a reader can see that the letter-wise figure is taken first.
    """
    orth_pred = torch.argmax(logits["orth"], dim=1)
    # The encoder ids shifted left by one, which must stay identical to the slice
    # `TrainingPipeline.compute_loss` uses: a metric scoring different positions than the
    # loss trains reports on a model that was never optimized, which is issue #225. The
    # layout that makes the shift correct is documented once, at compute_loss, rather than
    # restated here. See docs/decisions/0004.
    orth_true = orthography.enc_input_ids[:, 1:]
    valid = orth_true != orth_pad_id

    letter_wise = (orth_pred[valid] == orth_true[valid]).sum().float() / valid.sum().float()

    # Padding is forced to match so it cannot make a word wrong, then a word counts only
    # when every one of its positions agrees.
    orth_pred[~valid] = orth_pad_id
    word_wise = (orth_pred == orth_true).all(dim=1).float().mean()

    return {
        "letter_wise_accuracy": letter_wise.item(),
        "word_wise_accuracy": word_wise.item(),
    }
