from src.utils.helper_functions import set_seed
from typing import Optional
import torch.nn as nn
import torch



class Encoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int) -> None:
        super(Encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=4 * d_model
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output
