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
        tgt_is_causal: bool | None = None,
    ) -> torch.Tensor:
        """``tgt_is_causal`` is passed through rather than left for torch to work out.

        With a ``tgt_mask`` supplied and this left as ``None``, every layer call runs
        ``_detect_is_causal_mask``, which allocates a reference triangular mask and
        compares against it. The answer is always True for BRIDGE's masks, which are
        built by ``generate_triangular_mask``, so the work is always wasted, and on CUDA
        the comparison's ``bool(...)`` forces a device sync: 28 to 34 of them per
        ``generate()`` call, measured with ``torch.cuda.set_sync_debug_mode("warn")``.
        """
        return self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
        )
