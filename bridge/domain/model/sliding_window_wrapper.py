import time
from typing import Optional

import torch
from beartype import beartype
from torch import nn

from bridge.domain.model.utils_debug import check_nan


@beartype
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
        is_causal: bool = False,
        max_seq_len: int = 4096,
        ensure_contiguous: bool = False,  # New parameter
    ) -> None:
        """Initialize the sliding window wrapper.

        Args:
            encoder: The underlying Encoder instance
            window_size: Size of sliding attention window
            enabled: Whether sliding window attention is enabled
            is_causal: Whether to apply causal masking (for encoder)
            max_seq_len: Maximum sequence length for pre-computed mask
            ensure_contiguous: Whether to ensure sliced masks are contiguous
        """
        super().__init__()
        self.encoder = encoder
        self.window_size = window_size
        self.enabled = enabled
        self.is_causal = is_causal
        self.max_seq_len = max_seq_len
        self.ensure_contiguous = ensure_contiguous

        # Pre-compute the largest possible mask to avoid recomputation
        self._cached_mask: Optional[torch.Tensor] = None
        self._cached_device: Optional[torch.device] = None

    def _get_or_create_cached_mask(self, device: torch.device) -> torch.Tensor:
        """Get or create cached mask for maximum sequence length.

        Args:
            device: Device to create mask on

        Returns:
            Pre-computed mask for max_seq_len
        """
        if (
            self._cached_mask is None
            or self._cached_device != device
            or self._cached_mask.device != device
        ):
            self._cached_mask = self._create_full_mask(self.max_seq_len, device)
            # Ensure memory is contiguous for efficient slicing
            self._cached_mask = self._cached_mask.contiguous()
            self._cached_device = device

        return self._cached_mask

    def _create_full_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create full mask for given sequence length.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Boolean mask where True means attend, False means mask out
        """
        # Add debugging for long sequences
        if seq_len > 512:
            print(f"    [DEBUG] Creating mask for long sequence: {seq_len}")

        positions = torch.arange(seq_len, device=device)
        query_pos = positions.unsqueeze(1)  # [seq_len, 1]
        key_pos = positions.unsqueeze(0)  # [1, seq_len]

        if self.is_causal:
            # Causal constraint: can't attend to future positions
            causal_mask = key_pos <= query_pos

            # Sliding window constraint: can only attend to positions within window
            # For causal sliding window, we look back within the window
            window_mask = (key_pos >= query_pos - self.window_size + 1) & (
                key_pos <= query_pos
            )

            # Combine both constraints
            combined_mask = causal_mask & window_mask
            print(f"    [DEBUG] combined_mask: {combined_mask}")
            # quit()
        else:
            # Bidirectional sliding window: attend to positions [i-window_size//2, i+window_size//2]
            # This creates a symmetric window around each position
            half_window = self.window_size // 2
            window_mask = (key_pos >= query_pos - half_window) & (
                key_pos <= query_pos + half_window
            )

            # print(f"    [DEBUG] window_mask: {window_mask}")
            combined_mask = window_mask

        # Convert to PyTorch attention mask format directly (0.0 = attend, -inf = mask out)
        # This avoids the conversion cost in F._canonical_mask
        # attention_mask = torch.where(combined_mask, 0.0, float("-inf"))

        # Add debugging for mask properties
        # if seq_len > 512:
        #     print(
        #         f"    [DEBUG] Mask stats: min={attention_mask.min().item():.6f}, max={attention_mask.max().item():.6f}"
        #     )
        #     print(f"    [DEBUG] Mask unique values: {torch.unique(attention_mask)}")
        #     if torch.isnan(attention_mask).any():
        #         print(f"    ⚠ [DEBUG] WARNING: NaN detected in mask creation!")
        #         nan_count = torch.isnan(attention_mask).sum().item()
        #         print(f"    [DEBUG] NaN count: {nan_count}/{attention_mask.numel()}")

        # Correct mask
        # print(f"    [DEBUG] combined_mask: {~combined_mask}")
        return ~combined_mask  # Invert: True means masked out, False means attend

    def create_sliding_window_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create sliding window attention mask for encoders.

        Efficiently slices from pre-computed mask or creates new one if needed.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Boolean mask where True means attend, False means mask out
        """
        if seq_len <= self.max_seq_len:
            # Use cached mask and create a view (no memory copy)
            cached_mask = self._get_or_create_cached_mask(device)
            mask_slice = cached_mask[:seq_len, :seq_len]
            # print(f"SlidingWindowEncoderWrapper: {mask_slice=}")

            # Make contiguous if requested for GPU performance
            if self.ensure_contiguous and not mask_slice.is_contiguous():
                mask_slice = mask_slice.contiguous()

            # return ~mask_slice  # Invert: True means masked out, False means attend
            return mask_slice  # Invert: True means masked out, False means attend
        else:
            # For sequences larger than max_seq_len, create mask on-the-fly
            return self._create_full_mask(seq_len, device)

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
        # check_nan(src, "forward src")
        # print(f"    [DEBUG] src_mask: {src_mask}")
        # check_nan(src_mask, "forward src_mask")
        # check_nan(src_key_padding_mask, "forward src_key_padding_mask")

        if not self.enabled:
            # Pass through unchanged when sliding window is disabled
            start_time = time.time()
            result = self.encoder(src, src_mask, src_key_padding_mask)  # <<< ERROR
            end_time = time.time() - start_time
            print(
                f"    [DEBUG] Encoder forward pass time: {end_time:.6f}s, {self.enabled=}"
            )
            return result

        # Create sliding window mask
        start_time = time.time()
        seq_len = src.shape[1]
        sliding_mask = self.create_sliding_window_mask(seq_len, src.device)
        # print(sliding_mask)

        # Combine with existing mask if provided (logical AND)
        if src_mask is not None:
            final_mask = sliding_mask & src_mask
        else:
            final_mask = sliding_mask

        # check_nan(final_mask, "final_mask")
        # check_nan(src_mask, "src_mask")
        # check_nan(sliding_mask, "sliding_masks")
        # check_nan(src_key_padding_mask, "src_key_padding_mask")
        # check_nan(src, "src")
        # print(f"SlidingWindowEncoderWrapper: {final_mask=}")
        # print(f"SlidingWindowEncoderWrapper: {src_key_padding_mask=}")
        # quit()

        # ERROR in self.encoder: NaN generated
        result = self.encoder(src, final_mask, src_key_padding_mask)
        end_time = time.time() - start_time
        print(
            f"    [DEBUG] Encoder forward pass time: {end_time:.6f}s, {self.enabled=}"
        )
        # check_nan(result, "forward, result")  # NaN detected
        return result


@beartype
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
        max_seq_len: int = 4096,
        ensure_contiguous: bool = False,  # New parameter
    ) -> None:
        """Initialize the sliding window wrapper.

        Args:
            decoder: The underlying Decoder instance
            window_size: Size of sliding attention window
            enabled: Whether sliding window attention is enabled
            max_seq_len: Maximum sequence length for pre-computed mask
            ensure_contiguous: Whether to ensure sliced masks are contiguous
        """
        super().__init__()
        self.decoder = decoder
        self.window_size = window_size
        self.enabled = enabled
        self.max_seq_len = max_seq_len
        self.ensure_contiguous = ensure_contiguous

        # Pre-compute the largest possible mask to avoid recomputation
        self._cached_mask: Optional[torch.Tensor] = None
        self._cached_device: Optional[torch.device] = None

        print(f"====> SlidingWindowDecoderWrapper, {window_size=}, {enabled=}")

    def _get_or_create_cached_mask(self, device: torch.device) -> torch.Tensor:
        """Get or create cached mask for maximum sequence length.

        Args:
            device: Device to create mask on

        Returns:
            Pre-computed mask for max_seq_len
        """
        if (
            self._cached_mask is None
            or self._cached_device != device
            or self._cached_mask.device != device
        ):
            self._cached_mask = self._create_full_causal_mask(self.max_seq_len, device)
            # Ensure memory is contiguous for efficient slicing
            self._cached_mask = self._cached_mask.contiguous()
            self._cached_device = device

        return self._cached_mask

    def _create_full_causal_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create full causal mask for given sequence length.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Boolean mask where True means attend, False means mask out
        """
        # Add debugging for long sequences
        # if seq_len > 512:
        #     print(f"    [DEBUG] Creating causal mask for long sequence: {seq_len}")

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

        # Convert to PyTorch attention mask format directly (0.0 = attend, -inf = mask out)
        # This avoids the conversion cost in F._canonical_mask
        # attention_mask = torch.where(combined_mask, 0.0, float("-inf"))

        # Add debugging for mask properties
        # if seq_len > 512:
        #     print(
        #         f"    [DEBUG] Causal mask stats: min={attention_mask.min().item():.6f}, max={attention_mask.max().item():.6f}"
        #     )
        #     print(
        #         f"    [DEBUG] Causal mask unique values: {torch.unique(attention_mask)}"
        #     )
        #     if torch.isnan(attention_mask).any():
        #         print(f"    ⚠ [DEBUG] WARNING: NaN detected in causal mask creation!")
        #         nan_count = torch.isnan(attention_mask).sum().item()
        #         print(f"    [DEBUG] NaN count: {nan_count}/{attention_mask.numel()}")

        return ~combined_mask  # Invert: True means masked out, False means attend

    def create_sliding_window_causal_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create causal sliding window attention mask for decoders.

        Efficiently slices from pre-computed mask or creates new one if needed.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Boolean mask where True means attend, False means mask out
        """
        if seq_len <= self.max_seq_len:
            # Use cached mask and create a view (no memory copy)
            cached_mask = self._get_or_create_cached_mask(device)
            mask_slice = cached_mask[:seq_len, :seq_len]

            # Make contiguous if requested for GPU performance
            if self.ensure_contiguous and not mask_slice.is_contiguous():
                mask_slice = mask_slice.contiguous()

            return ~mask_slice  # Invert: True means masked out, False means attend
        else:
            # For sequences larger than max_seq_len, create mask on-the-fly
            return self._create_full_causal_mask(seq_len, device)

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
            start_time = time.time()
            result = self.decoder(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )
            end_time = time.time() - start_time
            print(
                f"    [DEBUG] Decoder forward pass time: {end_time:.6f}s, {self.enabled=}"
            )
            return result

        # Create sliding window causal mask for target
        start_time = time.time()
        seq_len = tgt.shape[1]
        sliding_causal_mask = self.create_sliding_window_causal_mask(
            seq_len, tgt.device
        )

        # Combine with existing target mask if provided (logical AND)
        if tgt_mask is not None:
            final_tgt_mask = sliding_causal_mask & tgt_mask
        else:
            final_tgt_mask = sliding_causal_mask

        result = self.decoder(
            tgt,
            memory,
            final_tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
        )
        end_time = time.time() - start_time
        print(
            f"    [DEBUG] Decoder forward pass time: {end_time:.6f}s, {self.enabled=}"
        )
        return result


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    causal: bool = False,
    device: torch.device = "cpu",
) -> torch.Tensor:
    """Create sliding window masks.

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

    return ~window_mask  # Invert: True means masked out, False means attend


if __name__ == "__main__":
    """Test the sliding window wrapper functionality."""

    # Test data
    batch_size = 2
    seq_len = 1024
    d_model = 512
    window_size = 61

    # Create test tensors
    device = torch.device("cpu")
    src = torch.randn(batch_size, seq_len, d_model, device=device)
    memory = torch.randn(batch_size, seq_len, d_model, device=device)
    tgt = torch.randn(batch_size, seq_len, d_model, device=device)

    print("=== Testing Encoder Wrapper ===")

    # Test encoder wrapper with causal masking
    encoder = nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
    encoder_wrapper = SlidingWindowEncoderWrapper(
        encoder,
        window_size=window_size,
        enabled=True,
        is_causal=True,
        max_seq_len=512,
        ensure_contiguous=True,  # Test contiguous option
    )

    # Test forward pass
    output = encoder_wrapper(src)
    print(f"Encoder output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ Encoder wrapper forward pass test passed")

    # Test mask caching efficiency and memory usage for encoder
    import time

    # First call (creates cache)
    start_time = time.time()
    mask1 = encoder_wrapper.create_sliding_window_mask(64, device)
    first_call_time = time.time() - start_time

    # Second call (uses cache)
    start_time = time.time()
    mask2 = encoder_wrapper.create_sliding_window_mask(64, device)
    second_call_time = time.time() - start_time

    print(f"Encoder first call time: {first_call_time:.6f}s")
    print(f"Encoder second call time: {second_call_time:.6f}s")
    print(f"Encoder speedup: {first_call_time/second_call_time:.2f}x")

    # Test that we're using views, not copies for encoder
    cached_mask = encoder_wrapper._get_or_create_cached_mask(device)
    mask_slice = encoder_wrapper.create_sliding_window_mask(64, device)

    # Check if they share the same underlying memory
    print(f"Encoder cached mask data_ptr: {cached_mask.data_ptr()}")
    print(f"Encoder mask slice data_ptr: {mask_slice.data_ptr()}")
    print(f"Encoder are they the same tensor? {cached_mask is mask_slice}")
    print(
        f"Encoder are they sharing memory? {cached_mask.data_ptr() == mask_slice.data_ptr()}"
    )

    # Test memory continuity for encoder
    print(f"Encoder cached mask is contiguous: {cached_mask.is_contiguous()}")
    print(f"Encoder mask slice is contiguous: {mask_slice.is_contiguous()}")

    # Test different sequence lengths for encoder
    test_lengths = [32, 64, 128, 256, 1024]
    for test_len in test_lengths:
        test_mask = encoder_wrapper.create_sliding_window_mask(test_len, device)
        print(
            f"Encoder mask for seq_len={test_len}: shape={test_mask.shape}, dtype={test_mask.dtype}"
        )
        assert test_mask.shape == (test_len, test_len)
        assert test_mask.dtype == torch.bool

    print("✓ Encoder wrapper all tests passed")

    print("\n=== Testing Decoder Wrapper ===")

    # Test decoder wrapper
    decoder = nn.TransformerDecoderLayer(d_model, nhead=8, batch_first=True)
    decoder_wrapper = SlidingWindowDecoderWrapper(
        decoder,
        window_size=window_size,
        enabled=True,
        max_seq_len=512,
        ensure_contiguous=True,  # Test contiguous option
    )

    # Test forward pass
    output = decoder_wrapper(tgt, memory)
    print(f"Decoder output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model)
    print("✓ Decoder wrapper forward pass test passed")

    # Test mask caching efficiency and memory usage for decoder
    # First call (creates cache)
    start_time = time.time()
    mask1 = decoder_wrapper.create_sliding_window_causal_mask(64, device)
    first_call_time = time.time() - start_time

    # Second call (uses cache)
    start_time = time.time()
    mask2 = decoder_wrapper.create_sliding_window_causal_mask(64, device)
    second_call_time = time.time() - start_time

    print(f"Decoder first call time: {first_call_time:.6f}s")
    print(f"Decoder second call time: {second_call_time:.6f}s")
    print(f"Decoder speedup: {first_call_time/second_call_time:.2f}x")

    # Test that we're using views, not copies for decoder
    cached_mask = decoder_wrapper._get_or_create_cached_mask(device)
    mask_slice = decoder_wrapper.create_sliding_window_causal_mask(64, device)

    # Check if they share the same underlying memory
    print(f"Decoder cached mask data_ptr: {cached_mask.data_ptr()}")
    print(f"Decoder mask slice data_ptr: {mask_slice.data_ptr()}")
    print(f"Decoder are they the same tensor? {cached_mask is mask_slice}")
    print(
        f"Decoder are they sharing memory? {cached_mask.data_ptr() == mask_slice.data_ptr()}"
    )

    # Test memory continuity for decoder
    print(f"Decoder cached mask is contiguous: {cached_mask.is_contiguous()}")
    print(f"Decoder mask slice is contiguous: {mask_slice.is_contiguous()}")

    # Test different sequence lengths for decoder
    for test_len in test_lengths:
        test_mask = decoder_wrapper.create_sliding_window_causal_mask(test_len, device)
        print(
            f"Decoder mask for seq_len={test_len}: shape={test_mask.shape}, dtype={test_mask.dtype}"
        )
        assert test_mask.shape == (test_len, test_len)
        assert test_mask.dtype == torch.bool

    print("✓ Decoder wrapper all tests passed")

    print("\n=== Testing Mask Properties ===")

    # Test that masks have correct values (0.0 for attend, -inf for mask out)
    encoder_mask = encoder_wrapper.create_sliding_window_mask(16, device)
    decoder_mask = decoder_wrapper.create_sliding_window_causal_mask(16, device)

    print(f"Encoder mask unique values: {torch.unique(encoder_mask)}")
    print(f"Decoder mask unique values: {torch.unique(decoder_mask)}")

    # Verify we have 0.0 and -inf values
    # assert torch.allclose(
    # torch.unique(encoder_mask), torch.tensor([float("-inf"), 0.0])
    # )
    # assert torch.allclose(
    # torch.unique(decoder_mask), torch.tensor([float("-inf"), 0.0])
    # )

    # Test that causal constraint is enforced in decoder mask
    # In a causal mask, position i should not attend to positions j > i
    for i in range(16):
        for j in range(16):
            if j > i:  # Future positions
                assert (
                    decoder_mask[i, j] == False
                ), f"Position {i} can attend to future position {j}"

    print("✓ Mask properties test passed")

    print("\n=== Testing Disabled Mode ===")

    # Test that disabled mode works correctly
    disabled_encoder = SlidingWindowEncoderWrapper(
        encoder,
        window_size=window_size,
        enabled=False,  # Disabled
        is_causal=True,
        max_seq_len=512,
    )

    disabled_decoder = SlidingWindowDecoderWrapper(
        decoder,
        window_size=window_size,
        enabled=False,  # Disabled
        max_seq_len=512,
    )

    # These should pass through unchanged
    output_disabled_encoder = disabled_encoder(src)
    output_disabled_decoder = disabled_decoder(tgt, memory)

    assert output_disabled_encoder.shape == (batch_size, seq_len, d_model)
    assert output_disabled_decoder.shape == (batch_size, seq_len, d_model)

    print("✓ Disabled mode test passed")

    print("\n=== Testing Large Sequence Fallback ===")

    # Test that large sequences fall back to on-the-fly creation
    large_seq_len = 8192  # Larger than max_seq_len=512

    # This should not use cached mask
    large_encoder_mask = encoder_wrapper.create_sliding_window_mask(
        large_seq_len, device
    )
    large_decoder_mask = decoder_wrapper.create_sliding_window_causal_mask(
        large_seq_len, device
    )

    assert large_encoder_mask.shape == (large_seq_len, large_seq_len)
    assert large_decoder_mask.shape == (large_seq_len, large_seq_len)
    assert large_encoder_mask.dtype == torch.bool
    assert large_decoder_mask.dtype == torch.bool

    print("✓ Large sequence fallback test passed")

    print("\nAll tests passed! ")
