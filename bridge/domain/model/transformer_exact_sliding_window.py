#!/usr/bin/env python3
"""Exact Sliding Window Transformer Encoder Layer for BRIDGE architecture.

This module provides a TransformerEncoderLayer that uses exact sliding window
attention without chunking limitations and without requiring PyTorch nightly.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

from bridge.domain.model.exact_sliding_window_attention import (
    ExactSlidingWindowMHA,
)


class ExactSlidingWindowEncoderLayer(TransformerEncoderLayer):
    """TransformerEncoderLayer that uses exact sliding window attention.

    This subclass replaces the standard multi-head attention with
    ExactSlidingWindowMHA for true sliding window attention without
    chunking limitations.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        window_size: int = 512,
        causal: bool = True,
        **kwargs,
    ):
        """Initialize ExactSlidingWindowEncoderLayer.

        Args:
            d_model: The number of expected features in the input.
            nhead: The number of heads in the multiheadattention models.
            dim_feedforward: The dimension of the feedforward network model.
            dropout: The dropout probability.
            activation: The activation function of the intermediate layer.
            layer_norm_eps: The eps value in layer normalization components.
            batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).
            norm_first: If True, layer norm is done prior to attention and feedforward operations.
            bias: If set to False, Linear and LayerNorm layers will not learn an additive bias.
 