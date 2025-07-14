import datetime
import gc
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float

from bridge.domain.model.benchmarks.sdpa_full_attention_model import (
    SDPAFullLayer,
    SDPAFullModelNotSubclassed,
)
from bridge.domain.model.benchmarks.sdpa_sliding_window_model import (
    SDPASlidingWindowLayer,
    SDPASlidingWindowModelNotSubclassed,
)


def check_cuda_memory():
    """Check CUDA memory usage and print statistics.

    Returns:
        None

    """
    if torch.cuda.is_available():
        # Current memory allocated by PyTorch
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB

        # Peak memory allocated
        peak = torch.cuda.max_memory_allocated() / 1024**3  # GB

        # Total GPU memory
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

        print(f"Allocated: {allocated:.2f}GB")
        print(f"Peak: {peak:.2f}GB")
        print(f"Total: {total:.2f}GB")
        print(f"Free: {total - allocated:.2f}GB")


class EncoderSDPA(nn.Module):
    """SDPA-based encoder with support for full and sliding window attention.

    This encoder can use different attention mechanisms based on the device
    and attention_type parameter. On CUDA, it supports both full attention
    and sliding window attention using SDPA. On CPU, it falls back to
    standard TransformerEncoderLayer.

    """

    #  d_model, nhead,  num_layers
    def __init__(
        self,
        # arguments in Encoder.py::Encoder.__init__
        d_model: int,
        nhead: int,
        num_layers: int,
        # Additional arguments to run sdpa and classical with sliding window
        device: str = "cpu",
        window_size: int = 512,
        causal: bool = True,  # default for LLMs
        seq_len: int | None = None,
        # look_backward: int = 1,
        # look_forward: int | None = None,
        attention_type: str = "sdpa_sliding_window",  # Fixed default
    ) -> None:
        """Initialize EncoderSDPA with local attention on CUDA, standard attention on CPU.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            device: Device to use ("cpu" or "cuda")
            window_size: Local attention window size (only used for CUDA)
            causal: Whether to use causal attention (only used for CUDA)
            # look_backward: Number of windows to look backward (only used for CUDA)
            # look_forward: Number of windows to look forward (only used for CUDA)
            attention_type: Type of attention to use:
                ("sdpa_full", "sdpa_sliding_window")

        """
        super().__init__()

        self.base_kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "batch_first": True,
            "dim_feedforward": 4 * d_model,
            "device": device,
        }
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.device = device
        self.window_size = window_size
        self.causal = causal
        self.seq_len = seq_len
        self.attention_type = attention_type

        # Use local attention on CUDA, standard attention on CPU
        if device == "cuda":
            self._handle_cuda_device()
        elif device == "cpu":
            # CPU fallback - use standard TransformerEncoderLayer
            self.encoder_layer = nn.TransformerEncoderLayer(**self.base_kwargs)
            print("Using standard TransformerEncoderLayer on CPU")
        else:
            raise ValueError(f"Invalid device: {device}")

        # Only create transformer_encoder for the subclassed versions
        # if attention_type in ["sdpa_full", "sdpa_sliding_window"] or device == "cpu":
        #     self.transformer_encoder = nn.TransformerEncoder(
        #         self.encoder_layer, num_layers=num_layers
        #     )

    def _handle_cuda_device(self) -> None:
        """Refine encoder arguments for CUDA-specific configuration."""
        local_kwargs = {
            **self.base_kwargs,
            "window_size": self.window_size,
            "seq_len": self.seq_len,
            "device": self.device,
            # Not used for SDPA. Might be used for classical attention. (Not sure)
            # "causal": causal,
        }

        # Choose attention implementation based on attention_type
        if self.attention_type == "sdpa_full":
            self.encoder_layer = SDPAFullLayer(**local_kwargs)
            print(f"Using SDPAFullLayer with window_size={self.window_size}")
        elif self.attention_type == "sdpa_sliding_window":
            self.encoder_layer = SDPASlidingWindowLayer(**local_kwargs)
            print(f"Using SDPASlidingWindowLayer with window_size={self.window_size}")
        elif self.attention_type == "sdpa_full_not_subclassed":
            self.model = SDPAFullModelNotSubclassed(
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                window_size=self.window_size,
                batch_first=True,
                dropout=0.0,
            )
            print(
                f"Using SDPAFullModelNotSubclassed with window_size={self.window_size}"
            )
            return  # Early return since we're not using transformer_encoder
        elif self.attention_type == "sdpa_sliding_window_not_subclassed":
            self.model = SDPASlidingWindowModelNotSubclassed(
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                window_size=self.window_size,
                seq_len=self.seq_len,  # Pass the seq_len parameter
                device=self.device,
                batch_first=True,
                dropout=0.0,
            )
            print(
                f"Using SDPASlidingWindowModelNotSubclassed with window_size={self.window_size}"
            )
            return  # Early return since we're not using transformer_encoder
        else:
            # Only support the attention types we explicitly handle
            raise ValueError(
                f"Unsupported attention_type: {self.attention_type}. "
                f"Supported types: sdpa_full, sdpa_sliding_window, "
                f"sdpa_full_not_subclassed, sdpa_sliding_window_not_subclassed"
            )

    def forward(
        self,
        src: Float[torch.Tensor, "batch_size seq_len d_model"],
        src_mask: Float[torch.Tensor, "seq_len seq_len"] | None = None,
        src_key_padding_mask: Float[torch.Tensor, "batch_size seq_len"] | None = None,
    ) -> torch.Tensor:
        """Forward pass through the encoder."""
        # Use the appropriate forward method based on attention_type
        if hasattr(self, "model"):
            # NotSubclassed versions - these have their own complete model
            return self.model(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        if hasattr(self, "encoder_layer"):
            # Subclassed versions - use the transformer_encoder with our custom encoder_layer
            return self.encoder_layer(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        # This should never happen if initialization is correct
        raise RuntimeError(
            f"Encoder not properly initialized. attention_type: {self.attention_type}"
        )


# ----------------------------------------------------------------------
if __name__ == "__main__":
    """Execute EncoderSDPA testing and benchmarking suite.
    """

    # Test that all four attention times work on both cpu and GPU. Do
    # this for d_model=256, nhead=1, batch_size=1, seq_len=128, num_layers=2

    d_model = 128
    seq_len = 128
    nhead = 2
    batch_size = 2
    seq_len = 128
    num_layers = 2
    window_size = 32

    # attention_type = "sdpa_full", "sdpa_sliding_window", "sdpa_full_not_subclassed", "sdpa_sliding_window_not_subclassed"

    # When not_subclassed, we don't use transformer_encoder
    #   which implies using the fast path of Pytorch
    attention_types = [
        "sdpa_full",  # Full attention
        "sdpa_full_not_subclassed",  # SDPA
        "sdpa_sliding_window",  # SDPA with sliding window
        "sdpa_sliding_window_not_subclassed",  # Full attention with sliding window
    ]  # Test both models

    print("Tests on the CPU")
    for attn_type in attention_types:
        print(f"==> Testing {attn_type}")
        encoder = EncoderSDPA(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            device="cpu",
            window_size=window_size,
            causal=True,
            seq_len=seq_len,
            attention_type=attn_type,
        )
        encoder.forward(
            src=torch.randn(batch_size, seq_len, d_model),
            src_mask=None,
            src_key_padding_mask=None,
        )

    print()
    print("Tests on the GPU")
    for attn_type in attention_types:
        print(f"==> Testing {attn_type}")
        encoder = EncoderSDPA(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            device="cuda",
            window_size=window_size,
            causal=True,
            seq_len=seq_len,
            attention_type=attn_type,
        )
        encoder.forward(
            src=torch.randn(batch_size, seq_len, d_model),
            src_mask=None,
            src_key_padding_mask=None,
        )
