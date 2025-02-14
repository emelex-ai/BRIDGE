"""
BridgeEncoding: A high-performance data structure for managing orthographic and phonological encodings.
Uses slots and frozen dataclasses for optimal memory usage and access speed.
"""

from dataclasses import dataclass
from typing import Any, Union, Optional
import torch
import functools


@dataclass(frozen=True, slots=True)
class BridgeEncoding:
    """
    Unified container for orthographic and phonological encodings.

    This class is immutable (frozen) and uses slots for better memory usage
    and faster attribute access. All tensor operations maintain device consistency.

    Attributes:
        orth_enc_ids: Tensor[batch_size, seq_len] - Orthographic encoder input IDs
        orth_enc_mask: Tensor[batch_size, seq_len] - Orthographic encoder padding mask
        orth_dec_ids: Tensor[batch_size, seq_len] - Orthographic decoder input IDs
        orth_dec_mask: Tensor[batch_size, seq_len] - Orthographic decoder padding mask
        phon_enc_ids: list[list[Tensor]] - Phonological encoder feature tensors
        phon_enc_mask: Tensor[batch_size, seq_len] - Phonological encoder padding mask
        phon_dec_ids: list[list[Tensor]] - Phonological decoder feature tensors
        phon_dec_mask: Tensor[batch_size, seq_len] - Phonological decoder padding mask
        phon_targets: Tensor[batch_size, seq_len, num_features] - Target phonological features
        device: torch.device - Device all tensors reside on
    """

    orth_enc_ids: torch.Tensor
    orth_enc_mask: torch.Tensor
    orth_dec_ids: torch.Tensor
    orth_dec_mask: torch.Tensor
    phon_enc_ids: list[list[torch.Tensor]]
    phon_enc_mask: torch.Tensor
    phon_dec_ids: list[list[torch.Tensor]]
    phon_dec_mask: torch.Tensor
    phon_targets: torch.Tensor
    device: torch.device

    def __post_init__(self):
        """Validate tensor properties and ensure consistency."""
        # Use object.__setattr__ since the class is frozen
        object.__setattr__(self, "device", self.orth_enc_ids.device)

        self._validate_tensor_properties()
        self._validate_batch_consistency()
        self._validate_device_consistency()

    def _validate_tensor_properties(self):
        """Validate individual tensor properties."""
        # Validate orthographic tensors
        for name, tensor in [
            ("orth_enc_ids", self.orth_enc_ids),
            ("orth_dec_ids", self.orth_dec_ids),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
            if tensor.dim() != 2:
                raise ValueError(f"{name} must be 2-dimensional (batch × sequence)")
            if tensor.dtype not in [torch.long, torch.int]:
                raise ValueError(f"{name} must have dtype torch.long or torch.int")
            if torch.any(tensor < 0):
                raise ValueError(f"{name} cannot contain negative indices")

        # Validate padding masks
        for name, tensor in [
            ("orth_enc_mask", self.orth_enc_mask),
            ("orth_dec_mask", self.orth_dec_mask),
            ("phon_enc_mask", self.phon_enc_mask),
            ("phon_dec_mask", self.phon_dec_mask),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
            if tensor.dim() != 2:
                raise ValueError(f"{name} must be 2-dimensional")
            if tensor.dtype != torch.bool:
                raise ValueError(f"{name} must have dtype torch.bool")

        # Validate phonological feature tensors
        for name, tensor_list in [
            ("phon_enc_ids", self.phon_enc_ids),
            ("phon_dec_ids", self.phon_dec_ids),
        ]:
            if not isinstance(tensor_list, list):
                raise ValueError(f"{name} must be a list of lists of tensors")
            if not all(isinstance(batch, list) for batch in tensor_list):
                raise ValueError(f"Each batch in {name} must be a list")
            if not all(
                isinstance(t, torch.Tensor) for batch in tensor_list for t in batch
            ):
                raise ValueError(
                    f"All elements in {name} must be torch.Tensor got {type(tensor_list[0][0])}"
                )

        # Validate targets tensor
        if not isinstance(self.phon_targets, torch.Tensor):
            raise ValueError("phon_targets must be a torch.Tensor")
        if self.phon_targets.dim() != 3:
            raise ValueError(
                "phon_targets must be 3-dimensional (batch × sequence × features)"
            )

    def _validate_batch_consistency(self):
        """Ensure all components have consistent batch sizes."""
        batch_size = self.orth_enc_ids.size(0)

        # Check all tensor batch dimensions match
        tensors_to_check = [
            ("orth_enc_mask", self.orth_enc_mask),
            ("orth_dec_ids", self.orth_dec_ids),
            ("orth_dec_mask", self.orth_dec_mask),
            ("phon_enc_mask", self.phon_enc_mask),
            ("phon_dec_mask", self.phon_dec_mask),
            ("phon_targets", self.phon_targets),
        ]

        for name, tensor in tensors_to_check:
            if tensor.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: {name} has size {tensor.size(0)}, "
                    f"expected {batch_size}"
                )

        # Check phonological feature lists
        if len(self.phon_enc_ids) != batch_size:
            raise ValueError(
                f"Batch size mismatch: phon_enc_ids has {len(self.phon_enc_ids)} "
                f"batches, expected {batch_size}"
            )
        if len(self.phon_dec_ids) != batch_size:
            raise ValueError(
                f"Batch size mismatch: phon_dec_ids has {len(self.phon_dec_ids)} "
                f"batches, expected {batch_size}"
            )

    def _validate_device_consistency(self):
        """Ensure all tensors are on the same device."""
        device = self.device

        # Check regular tensors
        tensors_to_check = [
            self.orth_enc_ids,
            self.orth_enc_mask,
            self.orth_dec_ids,
            self.orth_dec_mask,
            self.phon_enc_mask,
            self.phon_dec_mask,
            self.phon_targets,
        ]

        for tensor in tensors_to_check:
            if tensor.device != device:
                raise ValueError(
                    f"Device mismatch: found tensor on {tensor.device}, "
                    f"expected {device}"
                )

        # Check phonological feature tensors
        for batch in self.phon_enc_ids + self.phon_dec_ids:
            for tensor in batch:
                if tensor.device != device:
                    raise ValueError(
                        f"Device mismatch: found phonological feature tensor on "
                        f"{tensor.device}, expected {device}"
                    )

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], device: Optional[torch.device] = None
    ) -> "BridgeEncoding":
        """
        Create a BridgeEncoding instance from a dictionary representation.

        Args:
            data: Dictionary containing orthographic and phonological components
            device: Optional device to place tensors on

        Returns:
            BridgeEncoding instance
        """
        if device is None:
            device = torch.device("cpu")

        # Move tensors to device
        orth_data = data["orthographic"]
        phon_data = data["phonological"]

        return cls(
            orth_enc_ids=orth_data["enc_input_ids"].to(device),
            orth_enc_mask=orth_data["enc_pad_mask"].to(device),
            orth_dec_ids=orth_data["dec_input_ids"].to(device),
            orth_dec_mask=orth_data["dec_pad_mask"].to(device),
            phon_enc_ids=[
                [t.to(device) for t in batch] for batch in phon_data["enc_input_ids"]
            ],
            phon_enc_mask=phon_data["enc_pad_mask"].to(device),
            phon_dec_ids=[
                [t.to(device) for t in batch] for batch in phon_data["dec_input_ids"]
            ],
            phon_dec_mask=phon_data["dec_pad_mask"].to(device),
            phon_targets=phon_data["targets"].to(device),
            device=device,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format for compatibility with existing code.

        Returns:
            Dictionary containing orthographic and phonological components
        """
        return {
            "orthographic": {
                "enc_input_ids": self.orth_enc_ids,
                "enc_pad_mask": self.orth_enc_mask,
                "dec_input_ids": self.orth_dec_ids,
                "dec_pad_mask": self.orth_dec_mask,
            },
            "phonological": {
                "enc_input_ids": self.phon_enc_ids,
                "enc_pad_mask": self.phon_enc_mask,
                "dec_input_ids": self.phon_dec_ids,
                "dec_pad_mask": self.phon_dec_mask,
                "targets": self.phon_targets,
            },
        }

    def to(self, device: torch.device) -> "BridgeEncoding":
        """
        Move all tensors to the specified device.

        Args:
            device: Target device

        Returns:
            New BridgeEncoding instance with all tensors on target device
        """
        return BridgeEncoding(
            orth_enc_ids=self.orth_enc_ids.to(device),
            orth_enc_mask=self.orth_enc_mask.to(device),
            orth_dec_ids=self.orth_dec_ids.to(device),
            orth_dec_mask=self.orth_dec_mask.to(device),
            phon_enc_ids=[[t.to(device) for t in batch] for batch in self.phon_enc_ids],
            phon_enc_mask=self.phon_enc_mask.to(device),
            phon_dec_ids=[[t.to(device) for t in batch] for batch in self.phon_dec_ids],
            phon_dec_mask=self.phon_dec_mask.to(device),
            phon_targets=self.phon_targets.to(device),
            device=device,
        )

    def __getitem__(self, idx: Union[int, slice]) -> dict[str, Any]:
        """
        Get a batch slice of the encoding.

        Args:
            idx: Integer index or slice

        Returns:
            Dictionary containing orthographic and phonological components for the slice
        """
        if isinstance(idx, int):
            return {
                "orthographic": {
                    "enc_input_ids": self.orth_enc_ids[idx : idx + 1],
                    "enc_pad_mask": self.orth_enc_mask[idx : idx + 1],
                    "dec_input_ids": self.orth_dec_ids[idx : idx + 1],
                    "dec_pad_mask": self.orth_dec_mask[idx : idx + 1],
                },
                "phonological": {
                    "enc_input_ids": self.phon_enc_ids[idx : idx + 1],
                    "enc_pad_mask": self.phon_enc_mask[idx : idx + 1],
                    "dec_input_ids": self.phon_dec_ids[idx : idx + 1],
                    "dec_pad_mask": self.phon_dec_mask[idx : idx + 1],
                    "targets": self.phon_targets[idx : idx + 1],
                },
            }
        elif isinstance(idx, slice):
            return {
                "orthographic": {
                    "enc_input_ids": self.orth_enc_ids[idx],
                    "enc_pad_mask": self.orth_enc_mask[idx],
                    "dec_input_ids": self.orth_dec_ids[idx],
                    "dec_pad_mask": self.orth_dec_mask[idx],
                },
                "phonological": {
                    "enc_input_ids": self.phon_enc_ids[idx],
                    "enc_pad_mask": self.phon_enc_mask[idx],
                    "dec_input_ids": self.phon_dec_ids[idx],
                    "dec_pad_mask": self.phon_dec_mask[idx],
                    "targets": self.phon_targets[idx],
                },
            }
        else:
            raise TypeError("Index must be int or slice")

    def __len__(self) -> int:
        """Return the batch size."""
        return self.orth_enc_ids.size(0)
