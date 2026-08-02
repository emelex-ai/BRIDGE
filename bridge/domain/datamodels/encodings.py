"""
BridgeEncoding: A high-performance data structure for managing orthographic and phonological encodings.
Uses slots and frozen dataclasses for optimal memory usage and access speed.
"""

from dataclasses import dataclass, field, replace
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class EncodingComponent:
    """A component of a BridgeEncoding, representing either orthographic or phonological data."""

    enc_input_ids: Any  # Tensor for orth, list of lists of tensors for phon
    enc_pad_mask: torch.Tensor
    dec_input_ids: Any  # Tensor for orth, list of lists of tensors for phon
    dec_pad_mask: torch.Tensor
    targets: torch.Tensor | None = None  # Only used for phonological data

    @property
    def phon_targets(self) -> torch.Tensor:
        """Phonological targets tensor.

        Raises:
            AttributeError: if ``targets`` was not provided (i.e. for orthographic
                components).
        """
        if self.targets is None:
            raise AttributeError("Phonological targets are not available")
        return self.targets

    def to(self, device: torch.device) -> "EncodingComponent":
        """Move every tensor in this component to ``device``.

        Handles both component shapes: orthographic ``input_ids`` are tensors,
        phonological ``input_ids`` are ``list[list[Tensor]]`` of feature indices.
        """

        def move(ids: Any) -> Any:
            if isinstance(ids, torch.Tensor):
                return ids.to(device)
            return [[t.to(device) for t in batch] for batch in ids]

        return EncodingComponent(
            enc_input_ids=move(self.enc_input_ids),
            enc_pad_mask=self.enc_pad_mask.to(device),
            dec_input_ids=move(self.dec_input_ids),
            dec_pad_mask=self.dec_pad_mask.to(device),
            targets=self.targets.to(device) if self.targets is not None else None,
        )


@dataclass(frozen=True, slots=True)
class BridgeEncoding:
    """
    Unified container for orthographic and phonological encodings.

    This class is immutable (frozen) and uses slots for better memory usage
    and faster attribute access. All tensor operations maintain device consistency.

    Both components are always populated — the tokenizer fills in placeholder
    tensors for the modality that wasn't actually provided (see
    ``BridgeTokenizer._create_placeholder_*``). Targets on the phonological
    component may still be ``None`` for inference-only encodings; route through
    ``phonological.phon_targets`` for a non-Optional accessor.

    Attributes:
        orthographic: EncodingComponent containing orthographic encodings
        phonological: EncodingComponent containing phonological encodings
        device: torch.device - Device all tensors reside on. Derived from the
            orthographic tensors; the constructor argument is advisory only.
    """

    orthographic: EncodingComponent
    phonological: EncodingComponent
    device: torch.device = field(default=torch.device("cpu"))

    def __post_init__(self):
        """Validate components, then derive the canonical device from the tensors."""
        self._validate_orthographic_component(self.orthographic)
        self._validate_phonological_component(self.phonological)

        # Use object.__setattr__ since the class is frozen.
        device = self.orthographic.enc_input_ids.device
        object.__setattr__(self, "device", device)

        orth_batch_size = self.orthographic.enc_input_ids.size(0)
        phon_batch_size = len(self.phonological.enc_input_ids)
        if orth_batch_size != phon_batch_size:
            raise ValueError(
                f"Batch size mismatch: orthographic component has {orth_batch_size} samples, "
                f"phonological component has {phon_batch_size} samples"
            )

        for batch in self.phonological.enc_input_ids:
            for tensor in batch:
                if tensor.device != device:
                    raise ValueError(
                        f"Device mismatch: phonological tensor on {tensor.device}, "
                        f"expected {device}"
                    )

    @staticmethod
    def _validate_orthographic_component(component: EncodingComponent):
        """Validate orthographic component tensors."""
        # Validate orthographic tensors
        for name, tensor in [
            ("enc_input_ids", component.enc_input_ids),
            ("dec_input_ids", component.dec_input_ids),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Orthographic {name} must be a torch.Tensor")
            if tensor.dim() != 2:
                raise ValueError(f"Orthographic {name} must be 2-dimensional (batch × sequence)")
            if tensor.dtype not in [torch.long, torch.int]:
                raise ValueError(f"Orthographic {name} must have dtype torch.long or torch.int")
            if torch.any(tensor < 0):
                raise ValueError(f"Orthographic {name} cannot contain negative indices")

        # Validate padding masks
        for name, tensor in [
            ("enc_pad_mask", component.enc_pad_mask),
            ("dec_pad_mask", component.dec_pad_mask),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Orthographic {name} must be a torch.Tensor")
            if tensor.dim() != 2:
                raise ValueError(f"Orthographic {name} must be 2-dimensional")
            if tensor.dtype != torch.bool:
                raise ValueError(f"Orthographic {name} must have dtype torch.bool")

        # Validate batch consistency
        batch_size = component.enc_input_ids.size(0)
        for name, tensor in [
            ("enc_pad_mask", component.enc_pad_mask),
            ("dec_input_ids", component.dec_input_ids),
            ("dec_pad_mask", component.dec_pad_mask),
        ]:
            if tensor.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: orthographic {name} has size {tensor.size(0)}, "
                    f"expected {batch_size}"
                )

    @staticmethod
    def _validate_phonological_component(component: EncodingComponent):
        """Validate phonological component tensors."""
        # Validate phonological feature tensors
        for name, tensor_list in [
            ("enc_input_ids", component.enc_input_ids),
            ("dec_input_ids", component.dec_input_ids),
        ]:
            if not isinstance(tensor_list, list):
                raise ValueError(f"Phonological {name} must be a list of lists of tensors")
            if not all(isinstance(batch, list) for batch in tensor_list):
                raise ValueError(f"Each batch in phonological {name} must be a list")
            if not all(isinstance(t, torch.Tensor) for batch in tensor_list for t in batch):
                raise ValueError(f"All elements in phonological {name} must be torch.Tensor")

        # Validate padding masks
        for name, tensor in [
            ("enc_pad_mask", component.enc_pad_mask),
            ("dec_pad_mask", component.dec_pad_mask),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Phonological {name} must be a torch.Tensor")
            if tensor.dim() != 2:
                raise ValueError(f"Phonological {name} must be 2-dimensional")
            if tensor.dtype != torch.bool:
                raise ValueError(f"Phonological {name} must have dtype torch.bool")

        # Validate targets tensor if present
        if component.targets is not None:
            if not isinstance(component.targets, torch.Tensor):
                raise ValueError("Phonological targets must be a torch.Tensor")
            if component.targets.dim() != 3:
                raise ValueError(
                    "Phonological targets must be 3-dimensional (batch × sequence × features)"
                )

        # Validate batch consistency
        batch_size = len(component.enc_input_ids)
        for name, tensor in [
            ("enc_pad_mask", component.enc_pad_mask),
            ("dec_pad_mask", component.dec_pad_mask),
        ]:
            if tensor.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: phonological {name} has size {tensor.size(0)}, "
                    f"expected {batch_size}"
                )

        if len(component.dec_input_ids) != batch_size:
            raise ValueError(
                f"Batch size mismatch: phonological dec_input_ids has {len(component.dec_input_ids)} batches, "
                f"expected {batch_size}"
            )

        if component.targets is not None and component.targets.size(0) != batch_size:
            raise ValueError(
                f"Batch size mismatch: phonological targets has size {component.targets.size(0)}, "
                f"expected {batch_size}"
            )

    def to(self, device: torch.device) -> "BridgeEncoding":
        """Return a BridgeEncoding with all tensors on ``device``.

        Returns ``self`` unchanged when the encoding is already on ``device`` — safe
        because the class is frozen and nothing in the codebase mutates the component
        tensors or their containing lists. (Even before this short-circuit existed,
        ``Tensor.to(same_device)`` returned the identical tensor, so a same-device
        ``to()`` never produced an isolated copy.)
        """
        if device == self.device:
            return self
        return replace(
            self,
            orthographic=self.orthographic.to(device),
            phonological=self.phonological.to(device),
            device=device,
        )

    def __len__(self) -> int:
        """Return the batch size."""
        return self.orthographic.enc_input_ids.size(0)
