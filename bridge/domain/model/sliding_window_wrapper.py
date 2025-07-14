from typing import Optional

import torch
from torch import nn


class SlidingWindowEncoderWrapper(nn.Module):
    """Wrapper that adds sliding window attention to existing Encoder instances.

    This wrapper can be toggled on/off without modifying the underlying Encoder.
    When enabled, it creates sliding window masks and passes them to the encoder.
    When disabled, it passes through the original inputs unchanged.
    """

    def __init__(
        self,
        encoder: nn.Module,
        window_size: int = 61,  # ±30 + current position
        enabled: bool = False,
    ) -> None:
        """Initialize the sliding window wrapper.

        Args:
            encoder: The underlying Encoder instance
            window_size: Size of sliding attention window
            enabled: Whether sliding window attention is enabled
        """
        super().__init__()
        self.encoder = encoder
        self.window_size = window_size
        self.enabled = enabled

    def create_sliding_window_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create sliding window attention mask for encoders.

        For encoders, we allow bidirectional attention within the window.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Boolean mask where True means attend, False means mask out
        """
        positions = torch.arange(seq_len, device=device)
        query_pos = positions.unsqueeze(1)  # [seq_len, 1]
        key_pos = positions.unsqueeze(0)  # [1, seq_len]

        # Sliding window: attend to positions [i-window_size//2, i+window_size//2]
        # This creates a symmetric window around each position
        half_window = self.window_size // 2
        window_mask = (key_pos >= query_pos - half_window) & (
            key_pos <= query_pos + half_window
        )

        return window_mask

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional sliding window attention.

        Args:
            src: Input tensor [batch_size, seq_len, d_model]
            src_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask

        Returns:
            Encoder output tensor
        """
        if not self.enabled:
            # Pass through unchanged when sliding window is disabled
            return self.encoder(src, src_mask, src_key_padding_mask)

        # Create sliding window mask
        seq_len = src.shape[1]
        sliding_mask = self.create_sliding_window_mask(seq_len, src.device)

        # Combine with existing mask if provided
        if src_mask is not None:
            final_mask = sliding_mask & src_mask
        else:
            final_mask = sliding_mask

        # Convert to PyTorch attention mask format (True = attend, False = mask out)
        # PyTorch expects the opposite convention, so we invert
        attention_mask = ~final_mask

        return self.encoder(src, attention_mask, src_key_padding_mask)


class SlidingWindowDecoderWrapper(nn.Module):
    """Wrapper that adds sliding window attention to existing Decoder instances.

    This wrapper can be toggled on/off without modifying the underlying Decoder.
    When enabled, it creates causal sliding window masks for the decoder.
    When disabled, it passes through the original inputs unchanged.
    """

    def __init__(
        self,
        decoder: nn.Module,
        window_size: int = 61,  # ±30 + current position
        enabled: bool = False,
    ) -> None:
        """Initialize the sliding window wrapper.

        Args:
            decoder: The underlying Decoder instance
            window_size: Size of sliding attention window
            enabled: Whether sliding window attention is enabled
        """
        super().__init__()
        self.decoder = decoder
        self.window_size = window_size
        self.enabled = enabled

    def create_sliding_window_causal_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create causal sliding window attention mask for decoders.

        For decoders, we enforce causality (can't attend to future) within the window.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Boolean mask where True means attend, False means mask out
        """
        positions = torch.arange(seq_len, device=device)
        query_pos = positions.unsqueeze(1)  # [seq_len, 1]
        key_pos = positions.unsqueeze(0)  # [1, seq_len]

        # Causal constraint: can't attend to future positions
        causal_mask = key_pos <= query_pos

        # Sliding window constraint: can only attend to positions within window
        # For decoders, we look back within the window
        window_mask = (key_pos >= query_pos - self.window_size + 1) & (
            key_pos <= query_pos
        )

        # Combine both constraints
        combined_mask = causal_mask & window_mask

        return combined_mask

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional sliding window attention.

        Args:
            tgt: Target tensor [batch_size, seq_len, d_model]
            memory: Memory tensor from encoder
            tgt_mask: Optional target attention mask
            memory_mask: Optional memory attention mask
            tgt_key_padding_mask: Optional target padding mask
            memory_key_padding_mask: Optional memory padding mask

        Returns:
            Decoder output tensor
        """
        if not self.enabled:
            # Pass through unchanged when sliding window is disabled
            return self.decoder(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )

        # Create sliding window causal mask for target
        seq_len = tgt.shape[1]
        sliding_causal_mask = self.create_sliding_window_causal_mask(
            seq_len, tgt.device
        )

        # Combine with existing target mask if provided
        if tgt_mask is not None:
            final_tgt_mask = sliding_causal_mask & tgt_mask
        else:
            final_tgt_mask = sliding_causal_mask

        # Convert to PyTorch attention mask format (True = attend, False = mask out)
        # PyTorch expects the opposite convention, so we invert
        attention_mask = ~final_tgt_mask

        return self.decoder(
            tgt,
            memory,
            attention_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
        )


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    causal: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Utility function to create sliding window masks.

    Args:
        seq_len: Sequence length
        window_size: Size of sliding attention window
        causal: Whether to apply causal masking
        device: Device to create mask on

    Returns:
        Boolean mask where True means attend, False means mask out
    """
    positions = torch.arange(seq_len, device=device)
    query_pos = positions.unsqueeze(1)  # [seq_len, 1]
    key_pos = positions.unsqueeze(0)  # [1, seq_len]

    if causal:
        # Causal sliding window: attend to positions [i-window_size+1, i]
        window_mask = (key_pos >= query_pos - window_size + 1) & (key_pos <= query_pos)
    else:
        # Bidirectional sliding window: attend to positions [i-window_size//2, i+window_size//2]
        half_window = window_size // 2
        window_mask = (key_pos >= query_pos - half_window) & (
            key_pos <= query_pos + half_window
        )

    return window_mask
