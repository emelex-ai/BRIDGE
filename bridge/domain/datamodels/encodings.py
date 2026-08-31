"""
BridgeEncoding: A high-performance data structure for managing orthographic and phonological encodings.
Uses slots and frozen dataclasses for optimal memory usage and access speed.
"""

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True, slots=True)
class EncodingComponent:
    """A component of a BridgeEncoding, representing either orthographic or phonological data.

    Both modalities use the same shape: ``(batch, sequence)`` integer ids. Orthographic ids
    index the character vocabulary; phonological ids index
    :class:`~bridge.core.phonreps.PhonemeTable` rows (*which phoneme*, not which feature).
    """

    enc_input_ids: torch.Tensor
    enc_pad_mask: torch.Tensor
    dec_input_ids: torch.Tensor
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
        """Move every tensor in this component to ``device``."""
        return EncodingComponent(
            enc_input_ids=self.enc_input_ids.to(device),
            enc_pad_mask=self.enc_pad_mask.to(device),
            dec_input_ids=self.dec_input_ids.to(device),
            dec_pad_mask=self.dec_pad_mask.to(device),
            targets=self.targets.to(device) if self.targets is not None else None,
        )


@dataclass(frozen=True, slots=True)
class BridgeEncoding:
    """
    Unified container for orthographic and phonological encodings.

    This class is immutable (frozen) and uses slots for better memory usage
    and faster attribute access. All tensor operations maintain device consistency.

    Both components are always populated. The tokenizer fills in placeholder
    tensors for the modality that wasn't actually provided (see
    ``BridgeTokenizer._create_placeholder_*``). Targets on the phonological
    component may still be ``None`` for inference-only encodings; route through
    ``phonological.phon_targets`` for a non-Optional accessor.

    Attributes:
        orthographic: EncodingComponent containing orthographic encodings
        phonological: EncodingComponent containing phonological encodings
        device: torch.device - Device all tensors reside on, read off the
            orthographic tensors.
    """

    orthographic: EncodingComponent
    phonological: EncodingComponent

    @property
    def device(self) -> torch.device:
        """The device every tensor in this encoding lives on."""
        return self.orthographic.enc_input_ids.device

    def __post_init__(self):
        """Validate both components agree on batch size and device."""
        self._validate_component(self.orthographic, "Orthographic")
        self._validate_phonological_component(self.phonological)

        device = self.device
        orth_batch_size = self.orthographic.enc_input_ids.size(0)
        phon_batch_size = self.phonological.enc_input_ids.size(0)
        if orth_batch_size != phon_batch_size:
            raise ValueError(
                f"Batch size mismatch: orthographic component has {orth_batch_size} samples, "
                f"phonological component has {phon_batch_size} samples"
            )

        if self.phonological.enc_input_ids.device != device:
            raise ValueError(
                f"Device mismatch: phonological tensor on "
                f"{self.phonological.enc_input_ids.device}, expected {device}"
            )

    @staticmethod
    def _validate_component(component: EncodingComponent, modality: str):
        """Validate one component's tensors.

        Both modalities carry ``(batch, sequence)`` integer ids, so one validator covers
        them; ``modality`` ("Orthographic" / "Phonological") only selects the message
        prefix. ``targets`` is phonological-only and checked by the caller.
        """
        lower = modality.lower()

        for name, tensor in [
            ("enc_input_ids", component.enc_input_ids),
            ("dec_input_ids", component.dec_input_ids),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{modality} {name} must be a torch.Tensor")
            if tensor.dim() != 2:
                raise ValueError(f"{modality} {name} must be 2-dimensional (batch × sequence)")
            if tensor.dtype not in [torch.long, torch.int]:
                raise ValueError(f"{modality} {name} must have dtype torch.long or torch.int")
            if torch.any(tensor < 0):
                raise ValueError(f"{modality} {name} cannot contain negative indices")

        for name, tensor in [
            ("enc_pad_mask", component.enc_pad_mask),
            ("dec_pad_mask", component.dec_pad_mask),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{modality} {name} must be a torch.Tensor")
            if tensor.dim() != 2:
                raise ValueError(f"{modality} {name} must be 2-dimensional")
            if tensor.dtype != torch.bool:
                raise ValueError(f"{modality} {name} must have dtype torch.bool")

        batch_size = component.enc_input_ids.size(0)
        for name, tensor in [
            ("enc_pad_mask", component.enc_pad_mask),
            ("dec_input_ids", component.dec_input_ids),
            ("dec_pad_mask", component.dec_pad_mask),
        ]:
            if tensor.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: {lower} {name} has size {tensor.size(0)}, "
                    f"expected {batch_size}"
                )

        # A mask must cover exactly the ids it masks. Both tokenizers build masks as
        # `ids == pad`, so this always holds in production; it is checkable here only
        # because both modalities now carry the same rectangular shape.
        for ids_name, ids, mask_name, mask in [
            ("enc_input_ids", component.enc_input_ids, "enc_pad_mask", component.enc_pad_mask),
            ("dec_input_ids", component.dec_input_ids, "dec_pad_mask", component.dec_pad_mask),
        ]:
            if ids.shape != mask.shape:
                raise ValueError(
                    f"Shape mismatch: {lower} {ids_name} is {tuple(ids.shape)} but "
                    f"{mask_name} is {tuple(mask.shape)}"
                )

    @staticmethod
    def _validate_phonological_component(component: EncodingComponent):
        """Validate the phonological component, including its loss targets."""
        BridgeEncoding._validate_component(component, "Phonological")

        if component.targets is not None:
            if not isinstance(component.targets, torch.Tensor):
                raise ValueError("Phonological targets must be a torch.Tensor")
            if component.targets.dim() != 3:
                raise ValueError(
                    "Phonological targets must be 3-dimensional (batch × sequence × features)"
                )
            if component.targets.size(0) != component.enc_input_ids.size(0):
                raise ValueError(
                    f"Batch size mismatch: phonological targets has size "
                    f"{component.targets.size(0)}, expected {component.enc_input_ids.size(0)}"
                )
            # One target row per decoder position. A mismatch used to surface only as a
            # shape error inside CrossEntropyLoss, several layers downstream.
            if component.targets.size(1) != component.dec_input_ids.size(1):
                raise ValueError(
                    f"Sequence length mismatch: phonological targets has "
                    f"{component.targets.size(1)} positions, but dec_input_ids has "
                    f"{component.dec_input_ids.size(1)}"
                )

    def to(self, device: torch.device) -> "BridgeEncoding":
        """Return a BridgeEncoding with all tensors on ``device``.

        Returns ``self`` unchanged when the encoding is already on ``device``, which is safe
        because the class is frozen and nothing in the codebase mutates the component
        tensors. (Even before this short-circuit existed, ``Tensor.to(same_device)``
        returned the identical tensor, so a same-device ``to()`` never produced an
        isolated copy.)
        """
        if device == self.device:
            return self
        return replace(
            self,
            orthographic=self.orthographic.to(device),
            phonological=self.phonological.to(device),
        )

    def __len__(self) -> int:
        """Return the batch size."""
        return self.orthographic.enc_input_ids.size(0)
