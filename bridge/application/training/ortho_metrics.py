import torch

from bridge.domain.datamodels import EncodingComponent


def calculate_orth_word_accuracy(
    orth_pred: torch.Tensor,
    orth_true: torch.Tensor,
    orth_valid_mask: torch.Tensor,
    orth_pad_id: int,
):
    orth_pred[~orth_valid_mask] = orth_pad_id
    word_wise_mask = orth_pred == orth_true
    orth_word_accuracy = word_wise_mask.all(dim=1).float().mean()
    return orth_word_accuracy


def calculate_letter_wise_accuracy(
    orth_valid_mask: torch.Tensor,
    masked_orth_true: torch.Tensor,
    masked_orth_pred: torch.Tensor,
):
    correct_matches = (masked_orth_pred == masked_orth_true).sum()
    letter_wise_accuracy = correct_matches.float() / orth_valid_mask.sum().float()
    return letter_wise_accuracy


def calculate_orth_metrics(
    logits: dict[str, torch.Tensor],
    orthography: EncodingComponent,
    orth_pad_id: int,
) -> dict[str, float]:
    orth_pred = torch.argmax(logits["orth"], dim=1)
    # Skip the prepended language token AND [BOS]. The character tokenizer lays
    # each sequence out as [LANG, BOS, ...chars, EOS, PAD, ...], so the model
    # predicts indices [LANG, ...chars, EOS] when paired with shifted inputs;
    # [:, 2:] aligns ground truth with predictions starting from the first char.
    orth_true = orthography.enc_input_ids[:, 2:]
    orth_valid_mask = orth_true != orth_pad_id
    masked_orth_true = orth_true[orth_valid_mask]
    masked_orth_pred = orth_pred[orth_valid_mask]
    letter_wise_accuracy = calculate_letter_wise_accuracy(
        orth_valid_mask, masked_orth_true, masked_orth_pred
    )
    orth_word_accuracy = calculate_orth_word_accuracy(
        orth_pred, orth_true, orth_valid_mask, orth_pad_id
    )
    return {
        "letter_wise_accuracy": letter_wise_accuracy.item(),
        "word_wise_accuracy": orth_word_accuracy.item(),
    }
