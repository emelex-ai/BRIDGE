import torch
from typing import TypedDict


class GenerationOutput(TypedDict):
    """Unified output format for all generation pathways.

    This consistent structure makes analysis and experimentation easier
    by providing a predictable interface regardless of pathway.
    """

    global_encoding: torch.Tensor  # (batch_size, d_embedding, d_model)
    orth_probs: list[list[torch.Tensor]] | None  # [batch_size][seq_steps](vocab_size)
    orth_tokens: torch.Tensor | None  # (batch_size, seq_len)
    phon_probs: list[list[torch.Tensor]] | None  # [batch_size][seq_steps](num_features)
    phon_vecs: list[list[torch.Tensor]] | None  # [batch_size][seq_steps](num_features)
    phon_tokens: list[list[torch.Tensor]] | None  # [batch_size][seq_steps](active_features)
