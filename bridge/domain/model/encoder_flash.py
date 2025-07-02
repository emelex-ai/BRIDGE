from typing import Optional

import torch
import torch.nn as nn

from bridge.domain.model.transformer_fast_attention import (
    FlashAttentionEncoderLayer,
)


class EncoderFlash(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        device: str = "cpu",
    ) -> None:
        super(Encoder, self).__init__()
        kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "batch_first": True,
            "dim_feedforward": 4 * d_model,
            "device": device,
        }
        # Call if on CPU
        if device == "cuda":
            encoder_layer = FlashAttentionEncoderLayer(**kwargs)
        else:
            encoder_layer = nn.TransformerEncoderLayer(**kwargs)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        return output
