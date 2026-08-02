"""Thin transformer wrappers that pin BRIDGE's shared layer conventions.

Both classes exist only to fix ``batch_first=True`` and ``dim_feedforward=4 * d_model``
in one place across the five encoder/decoder stacks that :class:`~bridge.domain.model.model.Model`
builds. They add no behavior of their own.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=4 * d_model
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )


class Decoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=4 * d_model
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
