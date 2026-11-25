"""The transformer blocks whose self-attention reads position as a rotation.

Position is rotary (Su et al., 2021). Inside self-attention every query and key is turned
by an angle proportional to its position before the dot product, so a score depends on how
far apart two tokens are and on nothing else about where they sit. No table is indexed by
position, so a sequence may be any length.

The turn has to happen between the query and key projections and the dot product, which
``nn.MultiheadAttention`` does not expose, so the two encoders and the two decoders are
built from the blocks here. Cross-attention is not rotated: the global memory a decoder
reads has no order. The mixer needs no rotation and is torch's own layer.

Everything but the rotation is what torch's stock layers do: post-norm, ReLU, a feedforward
width of ``4 * d_model``, dropout of 0.1, and one initial draw shared by the layers of a
stack. Parameter names and shapes are those of ``nn.TransformerEncoder`` and
``nn.TransformerDecoder``, so a stock stack's ``state_dict`` loads into :class:`Encoder` or
:class:`Decoder`. :func:`rotate` is the only place position enters, and with it replaced by
the identity these blocks compute what the stock ones compute on the same weights.
``tests/domain/model/test_rotary_layers.py`` asserts that.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

ROTARY_BASE = 10000.0
DROPOUT = 0.1


def rotate(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Turn each consecutive pair of coordinates of ``x`` by an angle set by its position.

    ``x`` is ``(..., length, head_dim)`` and ``positions`` is ``(length,)``. Pair ``i`` of
    ``n`` turns by ``position * ROTARY_BASE ** (-i / n)``: the first pair a full radian per
    token, the last almost not at all. An odd ``head_dim`` keeps its last coordinate as it
    is.
    """
    pairs = x.shape[-1] // 2
    rate = ROTARY_BASE ** (-torch.arange(pairs, device=x.device) / pairs)
    angle = positions[:, None] * rate
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., 0 : 2 * pairs : 2], x[..., 1 : 2 * pairs : 2]
    turned = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    return torch.cat((turned, x[..., 2 * pairs :]), dim=-1)


class Attention(nn.Module):
    """Multi-head attention with the parameters of ``nn.MultiheadAttention``.

    Given ``positions``, it rotates the queries and the keys, which is self-attention.
    Without them nothing is rotated, which is cross-attention. The values never are.
    ``allowed`` is True where a query may read a key, and broadcasts over the batch, the
    heads, the queries and the keys. ``None`` is everywhere.
    """

    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.nhead = nhead
        self.dropout = DROPOUT
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        allowed: torch.Tensor | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, d_model = query.shape
        (w_q, w_k, w_v), (b_q, b_k, b_v) = self.in_proj_weight.chunk(3), self.in_proj_bias.chunk(3)

        def heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch, x.shape[1], self.nhead, -1).transpose(1, 2)

        q = heads(F.linear(query, w_q, b_q))
        k = heads(F.linear(memory, w_k, b_k))
        v = heads(F.linear(memory, w_v, b_v))
        if positions is not None:
            # Self-attention: the queries and the keys have one shape, so one call turns both.
            q, k = rotate(torch.stack((q, k)), positions)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out_proj(out.transpose(1, 2).reshape(batch, length, d_model))


class EncoderLayer(nn.Module):
    """Rotary self-attention, then a feedforward block, each added back and normed."""

    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.self_attn = Attention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.dropout = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.dropout2 = nn.Dropout(DROPOUT)

    def forward(
        self, x: torch.Tensor, allowed: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, allowed, positions)))
        return self.norm2(x + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(x))))))


class DecoderLayer(nn.Module):
    """Rotary self-attention, cross-attention to the memory with no rotation, then a
    feedforward block, each added back and normed."""

    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.self_attn = Attention(d_model, nhead)
        self.multihead_attn = Attention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.dropout = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.dropout2 = nn.Dropout(DROPOUT)
        self.dropout3 = nn.Dropout(DROPOUT)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        allowed: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, allowed, positions)))
        x = self.norm2(x + self.dropout2(self.multihead_attn(x, memory, None)))
        return self.norm3(x + self.dropout3(self.linear2(self.dropout(F.relu(self.linear1(x))))))


def _stack(layer: nn.Module, num_layers: int) -> nn.ModuleList:
    """``num_layers`` copies of ``layer``, so a stack starts from one draw, as torch's do."""
    return nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))


class Encoder(nn.Module):
    """A stack of encoder layers. Token ``i`` of ``src`` sits at position ``i``."""

    def __init__(self, d_model: int, nhead: int, num_layers: int) -> None:
        super().__init__()
        self.layers = _stack(EncoderLayer(d_model, nhead), num_layers)

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """``src_key_padding_mask`` is ``(batch, length)``, True where the token is padding."""
        positions = torch.arange(src.shape[1], device=src.device)
        allowed = ~src_key_padding_mask[:, None, None, :]
        for layer in self.layers:
            src = layer(src, allowed, positions)
        return src


class Decoder(nn.Module):
    """A stack of decoder layers, causal: a token reads itself and the tokens before it.
    Token ``i`` of ``tgt`` sits at position ``i``, however much of the sequence is passed,
    so a prefix decodes as the start of the whole."""

    def __init__(self, d_model: int, nhead: int, num_layers: int) -> None:
        super().__init__()
        self.layers = _stack(DecoderLayer(d_model, nhead), num_layers)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``tgt_key_padding_mask`` is ``(batch, length)``, True where the token is padding."""
        length = tgt.shape[1]
        positions = torch.arange(length, device=tgt.device)
        allowed = torch.ones(length, length, dtype=torch.bool, device=tgt.device).tril()
        if tgt_key_padding_mask is not None:
            allowed = allowed & ~tgt_key_padding_mask[:, None, None, :]
        for layer in self.layers:
            tgt = layer(tgt, memory, allowed, positions)
        return tgt
