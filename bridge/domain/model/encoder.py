from typing import Optional

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=4 * d_model,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        print("encoder: 1")
        print(f"encoder: {src.shape=}")  # 6, 17, 1024
        if src_mask is not None:
            print(f"encoder: {src_mask.shape=}")
        if src_key_padding_mask is not None:
            print(f"encoder: {src_key_padding_mask.shape=}")
        ## encoder: src.shape=torch.Size([6, 21, 1024])
        ## encoder: src_key_padding_mask.shape=torch.Size([6, 18])
        output = self.transformer_encoder(  # <<< ERROR
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        print("encoder: 2")
        return output
